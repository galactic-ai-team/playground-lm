/**
 * PlaygroundLM chat demo — simple conversation UI over local ONNX Runtime Web.
 * Preserves exact GPT-2 BPE, SHA256 model integrity, and pinned ORT WASM 1.24.3.
 */

import { loadGPT2Tokenizer, GPT2_EOT_ID } from './gpt2_bpe.js';
import { renderLossChart, DEFAULT_STAGE_C_MILESTONES } from './loss_chart.js';
import { CheckpointManager } from './checkpoint_loader.js';
import {
  createMessage,
  formatChatPrompt,
  trimMessagesToBudget,
  findStopSequence,
  normalizeCompletionText,
  setTextContent,
  appendText,
  isPinnedToBottom,
  scrollToBottom,
  handleComposerKeydown,
  CONTEXT_LENGTH,
} from './chat_state.js';

export const DEFAULT_RIDDLE_PROMPTS = [
  'Tell me a short riddle.',
  'What has keys but no locks?',
  'Explain gravity in one sentence.',
];

const COLUMN_LABELS = {
  model: 'Model / Step',
  step: 'Step',
  tokens_seen: 'Tokens Seen',
  val_loss: 'Val Loss (↓)',
  perplexity: 'Perplexity (↓)',
  hellaswag_acc_norm: 'HellaSwag acc_norm (↑)',
  lambada_acc: 'LAMBADA acc (↑)',
};
const PERCENT_COLUMNS = new Set(['hellaswag_acc_norm', 'lambada_acc', 'hellaswag', 'lambada']);

let manifest = {
  model: 'PlaygroundLM',
  checkpoints: [],
  milestones: DEFAULT_STAGE_C_MILESTONES.slice(),
  benchmarks: { columns: ['val_loss', 'perplexity', 'hellaswag_acc_norm', 'lambada_acc'], rows: [] },
  riddle_prompts: DEFAULT_RIDDLE_PROMPTS.slice(),
};
let lossCurveData = null;
let activeCheckpointStep = null;
let onnxSession = null;
let hasWebGPU = false;
let isGenerating = false;
let stopRequested = false;
let tokenizer = null;
let checkpointManager = null;
let chatMessages = [];
let stickToBottom = true;

const runtimeDot = document.getElementById('runtime-dot');
const runtimeLabel = document.getElementById('runtime-label');
const statusText = document.getElementById('status-text');
const statusMeta = document.getElementById('status-meta');
const modelTitleEl = document.getElementById('model-title');
const modelBadgeEl = document.getElementById('model-badge');

const chatThread = document.getElementById('chat-thread');
const welcomeCard = document.getElementById('welcome-card');
const composer = document.getElementById('composer');
const composerInput = document.getElementById('composer-input');
const btnSend = document.getElementById('btn-send');
const btnStop = document.getElementById('btn-stop');
const btnNewChat = document.getElementById('btn-new-chat');
const btnToggleAdvanced = document.getElementById('btn-toggle-advanced');
const btnCloseAdvanced = document.getElementById('btn-close-advanced');
const advancedPanel = document.getElementById('advanced-panel');

const ckptEmptyState = document.getElementById('ckpt-empty-state');
const ckptComingSoonBanner = document.getElementById('ckpt-coming-soon-banner');
const ckptControls = document.getElementById('ckpt-controls');
const ckptSlider = document.getElementById('ckpt-slider');
const ckptStageButtons = document.getElementById('ckpt-stage-buttons');
const metaModel = document.getElementById('meta-model');
const metaStep = document.getElementById('meta-step');
const metaTokens = document.getElementById('meta-tokens');
const metaSession = document.getElementById('meta-session');
const onnxUrlInput = document.getElementById('onnx-url-input');
const btnFetchOnnx = document.getElementById('btn-fetch-onnx');

const tempSlider = document.getElementById('temp-slider');
const tempVal = document.getElementById('temp-val');
const topkSlider = document.getElementById('topk-slider');
const topkVal = document.getElementById('topk-val');
const toppSlider = document.getElementById('topp-slider');
const toppVal = document.getElementById('topp-val');
const reppenSlider = document.getElementById('reppen-slider');
const reppenVal = document.getElementById('reppen-val');
const maxTokensSlider = document.getElementById('max-tokens-slider');
const maxTokensVal = document.getElementById('max-tokens-val');
const tokenizerStatus = document.getElementById('tokenizer-status');
const tokSpeedEl = document.getElementById('tok-speed');
const logitBars = document.getElementById('logit-bars');
const inspectedTokenTag = document.getElementById('inspected-token-tag');
const lossChartContainer = document.getElementById('loss-chart-container');
const benchSourceNote = document.getElementById('bench-source-note');
const benchEmptyState = document.getElementById('bench-empty-state');
const benchTableWrap = document.getElementById('bench-table-wrap');
const benchThead = document.getElementById('bench-thead');
const benchTbody = document.getElementById('bench-tbody');

// Compatibility aliases for leftover helper references
const promptInput = document.getElementById('prompt-input');
const btnGenerate = document.getElementById('btn-generate');
const btnRiddle = document.getElementById('btn-riddle');

export function pickRandomRiddle(pool, lastIndex = -1) {
  const list = Array.isArray(pool) && pool.length > 0 ? pool : DEFAULT_RIDDLE_PROMPTS;
  if (list.length === 1) return { text: list[0], index: 0 };
  let idx = Math.floor(Math.random() * list.length);
  if (idx === lastIndex) idx = (idx + 1) % list.length;
  return { text: list[idx], index: idx };
}

export function sampleLogits(
  logitsSlice,
  {
    temperature = 0.8,
    topK = 40,
    topP = 0.92,
    repetitionPenalty = 1.1,
    recentTokenIds = [],
  } = {}
) {
  const temp = Math.max(temperature, 1e-3);
  const k = Math.max(1, Math.min(topK, logitsSlice.length));
  const repPen = Math.max(1.0, repetitionPenalty);
  const seenSet = repPen > 1.0001 ? new Set(recentTokenIds.slice(-64)) : null;

  const top = [];
  let minAllowed = -Infinity;

  for (let i = 0; i < logitsSlice.length; i++) {
    let val = logitsSlice[i];
    if (seenSet && seenSet.has(i)) {
      val = val > 0 ? val / repPen : val * repPen;
    }
    if (top.length < k || val > minAllowed) {
      top.push({ id: i, logit: val / temp });
      top.sort((a, b) => b.logit - a.logit);
      if (top.length > k) top.pop();
      minAllowed = top[top.length - 1].logit * temp;
    }
  }

  const maxLogit = top[0].logit;
  let sumExp = 0;
  for (const item of top) {
    item.prob = Math.exp(item.logit - maxLogit);
    sumExp += item.prob;
  }
  for (const item of top) {
    item.prob /= sumExp;
  }

  const pCutoff = Math.max(0.05, Math.min(1.0, topP));
  const nucleus = [];
  let cumProb = 0;
  for (const item of top) {
    nucleus.push(item);
    cumProb += item.prob;
    if (cumProb >= pCutoff && nucleus.length >= 1) break;
  }

  let nucleusSum = 0;
  for (const item of nucleus) nucleusSum += item.prob;
  const r = Math.random() * nucleusSum;
  let acc = 0;
  let chosenId = nucleus[0].id;
  for (const item of nucleus) {
    acc += item.prob;
    if (r <= acc) {
      chosenId = item.id;
      break;
    }
  }

  return {
    chosenId,
    topCandidates: top.slice(0, 5),
  };
}

export async function loadBenchmarkTable(fallbackBenchmarks = null) {
  let benchData = null;

  try {
    const res = await fetch('./benchmarks.json', { cache: 'no-store' });
    if (res.ok) {
      benchData = await res.json();
    }
  } catch {
    benchData = null;
  }

  const source = benchData || fallbackBenchmarks || manifest.benchmarks || {};
  const cols =
    Array.isArray(source.columns) && source.columns.length
      ? source.columns
      : ['val_loss', 'perplexity', 'hellaswag_acc_norm', 'lambada_acc'];
  const rows = Array.isArray(source.rows)
    ? source.rows
    : Array.isArray(source.models)
    ? source.models
    : [];
  const columnKeys = cols.map((col) =>
    typeof col === 'object' ? col.key || col.id : col
  );
  const groups = Array.isArray(source.column_groups)
    ? source.column_groups
    : [];
  const validGroups =
    groups.length > 0 &&
    groups.every(
      (group) =>
        typeof group.label === 'string' &&
        Array.isArray(group.columns) &&
        group.columns.length > 0
    ) &&
    groups.flatMap((group) => group.columns).join('|') ===
      columnKeys.join('|');

  if (benchSourceNote) {
    const noteText =
      (benchData && benchData.source_type === 'reported_rounded_summary'
        ? 'Reported rounded results. FineWeb-Edu validation loss/PPL measure next-token prediction; HellaSwag and LAMBADA measure separate downstream abilities. Stage B uses a different validation split from Stage C.'
        : null) ||
      source.note ||
      source.source_note ||
      source.description ||
      (benchData
        ? 'Loaded from demo/benchmarks.json (% accuracy columns normalized).'
        : 'Populated from demo/manifest.json & demo/benchmarks.json.');
    benchSourceNote.textContent = noteText;
  }

  if (rows.length === 0) {
    benchEmptyState.classList.remove('hidden');
    benchTableWrap.classList.add('hidden');
    return;
  }

  benchEmptyState.classList.add('hidden');
  benchTableWrap.classList.remove('hidden');

  benchThead.innerHTML = '';
  benchTbody.innerHTML = '';

  if (validGroups) {
    const groupRow = document.createElement('tr');
    groupRow.className = 'benchmark-group-row';

    const modelHeader = document.createElement('th');
    modelHeader.textContent = 'Model / Checkpoint';
    modelHeader.rowSpan = 2;
    modelHeader.scope = 'col';
    groupRow.appendChild(modelHeader);

    for (const group of groups) {
      const th = document.createElement('th');
      th.scope = 'colgroup';
      th.colSpan = group.columns.length;
      th.textContent = group.label;
      if (group.description) th.title = group.description;
      groupRow.appendChild(th);
    }

    benchThead.appendChild(groupRow);
  }

  const trHead = document.createElement('tr');
  const thName = document.createElement('th');
  thName.textContent = 'Model / Checkpoint';
  if (!validGroups) {
    thName.scope = 'col';
    trHead.appendChild(thName);
  }

  for (const col of cols) {
    const th = document.createElement('th');
    const colKey = typeof col === 'object' ? col.key || col.id : col;
    const colLabel =
      typeof col === 'object' ? col.label || colKey : COLUMN_LABELS[colKey] || colKey;
    th.textContent = colLabel;
    th.scope = 'col';
    trHead.appendChild(th);
  }
  benchThead.appendChild(trHead);
  for (const row of rows) {
    const tr = document.createElement('tr');
    if (row.step !== undefined && row.step !== null) {
      tr.dataset.step = String(row.step);
    }
    if (
      activeCheckpointStep !== null &&
      row.step !== undefined &&
      Number(row.step) === activeCheckpointStep
    ) {
      tr.classList.add('active-stage-row');
    }

    const tdLabel = document.createElement('td');
    const rowName =
      row.model ||
      row.name ||
      row.label ||
      (row.step !== undefined ? `Step ${Number(row.step).toLocaleString()}` : 'Run');
    tdLabel.textContent = String(rowName);
    if (row.validation_split === 'stage_b') {
      tdLabel.title = 'Validation loss/PPL use Stage B val.bin';
    } else if (row.validation_split === 'stage_c') {
      tdLabel.title = 'Validation loss/PPL use Stage C val.bin';
    }
    tr.appendChild(tdLabel);

    for (let cIdx = 0; cIdx < cols.length; cIdx++) {
      const col = cols[cIdx];
      const colKey = typeof col === 'object' ? col.key || col.id : col;
      const td = document.createElement('td');
      const rawVal = Array.isArray(row)
        ? row[cIdx]
        : row.metrics && colKey in row.metrics
        ? row.metrics[colKey]
        : row[colKey];
      td.textContent = formatMetricCell(rawVal, colKey);
      tr.appendChild(td);
    }
    benchTbody.appendChild(tr);
  }
}

function formatMetricCell(val, colKey = '') {
  if (val === null || val === undefined || val === '') return '—';
  if (typeof val === 'number') {
    if (PERCENT_COLUMNS.has(colKey)) {
      const pct = val <= 1.0 && val >= 0 ? val * 100 : val;
      return `${pct.toFixed(1)}%`;
    }
    return Number.isInteger(val) ? val.toLocaleString() : val.toFixed(4).replace(/\.?0+$/, '');
  }
  return String(val);
}

function setStatus(text, { meta = '', ready = false, error = false } = {}) {
  setTextContent(statusText, text);
  setTextContent(statusMeta, meta);
  runtimeDot.classList.toggle('ready', ready && !error);
  runtimeDot.classList.toggle('error', error);
  setTextContent(runtimeLabel, text);
}

function updateComposerEnabled() {
  const ready = Boolean(tokenizer && onnxSession && !isGenerating);
  composerInput.disabled = !tokenizer || isGenerating;
  btnSend.disabled = !ready || !composerInput.value.trim();
  btnStop.disabled = !isGenerating;
  btnGenerate.disabled = !ready;
}

function clearChatMessages() {
  chatMessages = [];
  chatThread.querySelectorAll('.msg').forEach((n) => n.remove());
  welcomeCard?.classList.remove('hidden');
}

function addMessageBubble(role, content, { pending = false, error = false } = {}) {
  welcomeCard?.classList.add('hidden');
  const wrap = document.createElement('div');
  wrap.className = `msg ${role}${pending ? ' pending' : ''}${error ? ' error' : ''}`;
  const roleEl = document.createElement('div');
  roleEl.className = 'msg-role';
  setTextContent(roleEl, role === 'user' ? 'You' : error ? 'Error' : 'PlaygroundLM');
  const bubble = document.createElement('div');
  bubble.className = 'msg-bubble';
  setTextContent(bubble, content);
  wrap.append(roleEl, bubble);
  chatThread.appendChild(wrap);
  if (stickToBottom) scrollToBottom(chatThread);
  return bubble;
}

function renderLogitBars(candidates, stepLabel) {
  setTextContent(inspectedTokenTag, stepLabel);
  logitBars.replaceChildren();
  for (const item of candidates || []) {
    const tokStr = tokenizer ? tokenizer.decode(item.id) : `#${item.id}`;
    const row = document.createElement('div');
    row.className = 'logit-row';
    const tok = document.createElement('span');
    tok.className = 'logit-tok';
    setTextContent(tok, `${JSON.stringify(tokStr)} (${item.id})`);
    const track = document.createElement('div');
    track.className = 'logit-bar-track';
    const fill = document.createElement('div');
    fill.className = 'logit-bar-fill';
    fill.style.width = `${Math.max(2, item.prob * 100)}%`;
    track.appendChild(fill);
    const prob = document.createElement('span');
    prob.className = 'logit-prob';
    setTextContent(prob, `${(item.prob * 100).toFixed(1)}%`);
    row.append(tok, track, prob);
    logitBars.appendChild(row);
  }
}

async function detectWebGPU() {
  if (typeof navigator !== 'undefined' && 'gpu' in navigator) {
    try {
      const adapter = await navigator.gpu.requestAdapter();
      hasWebGPU = Boolean(adapter);
    } catch {
      hasWebGPU = false;
    }
  }
  setStatus('Loading tokenizer…', { meta: 'ORT Web 1.24.3 WASM' });
}

async function initTokenizer() {
  const candidates = [
    ['./vocab.json', './merges.txt'],
    ['./encoder.json', './merges.txt'],
    ['./tokenizer.json', './merges.txt'],
  ];
  for (const [vUrl, mUrl] of candidates) {
    try {
      const candidate = await loadGPT2Tokenizer(vUrl, mUrl);
      if (
        candidate.vocabSize !== 50257 ||
        candidate.eotId !== 50256 ||
        JSON.stringify(candidate.encode('Hello world')) !== '[15496,995]'
      ) {
        throw new Error('Invalid GPT-2 tokenizer assets');
      }
      tokenizer = candidate;
      setTextContent(
        tokenizerStatus,
        `GPT-2 BPE · ${tokenizer.vocabSize.toLocaleString()} tokens · EOT ${tokenizer.eotId}`
      );
      composerInput.disabled = false;
      updateComposerEnabled();
      return;
    } catch {
      // try next
    }
  }
  tokenizer = null;
  setTextContent(tokenizerStatus, 'GPT-2 tokenizer unavailable — chat disabled');
  setStatus('Tokenizer failed to load', { error: true });
  updateComposerEnabled();
}

function handleCheckpointStatus(status) {
  const { state, index, checkpoint, session, message, loadedBytes, totalBytes } = status;
  if (ckptSlider && index !== undefined && index >= 0) ckptSlider.value = String(index);
  document.querySelectorAll('.stage-btn').forEach((btn) => {
    btn.classList.toggle('active', Number(btn.dataset.idx) === index);
  });
  if (checkpoint) {
    activeCheckpointStep = checkpoint.step;
    setTextContent(metaModel, manifest.model || 'PlaygroundLM');
    setTextContent(metaStep, Number(checkpoint.step).toLocaleString());
    setTextContent(metaTokens, Number(checkpoint.tokens_seen || 0).toLocaleString());
    setTextContent(modelBadgeEl, checkpoint.label || `step ${checkpoint.step}`);
  }

  if (state === 'mounted' && session) {
    onnxSession = session;
    setTextContent(metaSession, message || 'Ready');
    setStatus('Model ready', {
      ready: true,
      meta: checkpoint ? `${checkpoint.label || 'step ' + checkpoint.step}` : '',
    });
  } else if (state === 'loading') {
    onnxSession = null;
    const pct = totalBytes ? ` ${Math.round((loadedBytes / totalBytes) * 100)}%` : '';
    setStatus(message || `Downloading model…${pct}`, { meta: '≈171 MB first load' });
    setTextContent(metaSession, message || 'Downloading');
  } else if (state === 'compiling') {
    onnxSession = null;
    setStatus(message || 'Compiling WASM…', { meta: 'one-time per checkpoint' });
    setTextContent(metaSession, message || 'Compiling');
  } else if (state === 'coming_soon') {
    onnxSession = null;
    setStatus('Model URL missing for this checkpoint', { error: true });
    setTextContent(metaSession, 'Unavailable');
  } else if (state === 'error') {
    onnxSession = null;
    setStatus(message || 'Model load failed', { error: true });
    setTextContent(metaSession, message || 'Error');
  } else if (state === 'unmounted') {
    onnxSession = null;
  }
  updateComposerEnabled();
}

function renderManifestCheckpoints() {
  setTextContent(modelTitleEl, manifest.model || 'PlaygroundLM');
  setTextContent(metaModel, manifest.model || 'PlaygroundLM');
  const ckpts = manifest.checkpoints || [];
  checkpointManager.setCheckpoints(ckpts);
  if (!ckpts.length) {
    ckptEmptyState.classList.remove('hidden');
    ckptControls.classList.add('hidden');
    return;
  }
  ckptEmptyState.classList.add('hidden');
  ckptControls.classList.remove('hidden');
  if (ckptComingSoonBanner) {
    const pending = ckpts.some((c) => !c.onnx_url);
    ckptComingSoonBanner.classList.toggle('hidden', !pending);
    if (pending) setTextContent(ckptComingSoonBanner, 'Some checkpoints still lack hosted ONNX URLs.');
  }
  ckptSlider.min = '0';
  ckptSlider.max = String(ckpts.length - 1);
  ckptSlider.value = String(ckpts.length - 1);
  ckptStageButtons.replaceChildren();
  checkpointManager.checkpoints.forEach((c, idx) => {
    const btn = document.createElement('button');
    btn.type = 'button';
    btn.className = 'stage-btn';
    btn.dataset.idx = String(idx);
    btn.innerHTML = '';
    const label = document.createElement('div');
    setTextContent(label, `${c.label || 'Step'} · ${Number(c.step).toLocaleString()}${c.quant ? ' (' + c.quant.toUpperCase() + ')' : ''}`);
    const small = document.createElement('small');
    setTextContent(small, `${Number(c.tokens_seen || 0).toLocaleString()} tok`);
    btn.append(label, small);
    btn.addEventListener('click', () => {
      if (!isGenerating) checkpointManager.selectIndex(idx).catch(() => {});
    });
    ckptStageButtons.appendChild(btn);
  });
  checkpointManager.selectIndex(ckpts.length - 1).catch(() => {});
}

async function loadManifestAndArtifacts() {
  try {
    const res = await fetch('./manifest.json', { cache: 'no-store' });
    if (res.ok) {
      const data = await res.json();
      manifest = {
        ...manifest,
        ...data,
        milestones: data.milestones || DEFAULT_STAGE_C_MILESTONES.slice(),
        riddle_prompts: data.riddle_prompts || DEFAULT_RIDDLE_PROMPTS.slice(),
      };
    }
  } catch {
    // keep defaults
  }
  await loadBenchmarkTable(manifest.benchmarks);
  try {
    const lossRes = await fetch('./loss_curve.json', { cache: 'no-store' });
    if (lossRes.ok) {
      lossCurveData = await lossRes.json();
      renderLossChart(lossChartContainer, lossCurveData, {
        milestones: manifest.milestones || DEFAULT_STAGE_C_MILESTONES,
        activeStep: activeCheckpointStep,
      });
    }
  } catch {
    setTextContent(lossChartContainer, 'Loss curve unavailable.');
  }
  renderManifestCheckpoints();
}

async function runChatGeneration(userText) {
  const ort = checkpointManager?.ortModule;
  if (isGenerating || !onnxSession || !ort || !tokenizer) return;
  const cleaned = String(userText || '').trim();
  if (!cleaned) return;

  isGenerating = true;
  stopRequested = false;
  updateComposerEnabled();

  chatMessages.push(createMessage('user', cleaned));
  addMessageBubble('user', cleaned);
  const assistantBubble = addMessageBubble('assistant', 'Thinking…', { pending: true });

  const maxNewTokens = parseInt(maxTokensSlider.value, 10);
  const packed = trimMessagesToBudget(chatMessages, (t) => tokenizer.encode(t), {
    contextLength: CONTEXT_LENGTH,
    reserveNewTokens: maxNewTokens,
  });
  let inputIds = packed.inputIds.slice();
  if (!inputIds.length) inputIds.push(GPT2_EOT_ID);
  const eotId = tokenizer.eotId ?? GPT2_EOT_ID;
  const stream = tokenizer.createStreamDecoder();
  let rawReply = '';
  const t0 = performance.now();

  try {
    assistantBubble.parentElement.classList.remove('pending');
    setTextContent(assistantBubble, '');
    for (let step = 0; step < maxNewTokens; step++) {
      if (stopRequested) break;
      const contextIds = inputIds.slice(-CONTEXT_LENGTH);
      const inputTensor = new ort.Tensor(
        'int64',
        BigInt64Array.from(contextIds.map((id) => BigInt(id))),
        [1, contextIds.length]
      );
      const results = await onnxSession.run({ input_ids: inputTensor });
      const logitsTensor = results.logits ?? results[Object.keys(results)[0]];
      if (!logitsTensor) {
        inputTensor.dispose?.();
        throw new Error('ONNX output did not contain a logits tensor.');
      }
      const dims = logitsTensor.dims;
      const vocabSize = dims[dims.length - 1];
      const seqLen = dims.length >= 2 ? dims[dims.length - 2] : 1;
      const lastPosOffset = (seqLen - 1) * vocabSize;
      const lastTokenLogits = logitsTensor.data.subarray(lastPosOffset, lastPosOffset + vocabSize);
      const { chosenId, topCandidates } = sampleLogits(lastTokenLogits, {
        temperature: parseFloat(tempSlider.value),
        topK: parseInt(topkSlider.value, 10),
        topP: parseFloat(toppSlider.value),
        repetitionPenalty: parseFloat(reppenSlider.value),
        recentTokenIds: inputIds,
      });
      inputTensor.dispose?.();
      for (const tensor of Object.values(results)) tensor.dispose?.();

      if (chosenId === eotId) {
        renderLogitBars(topCandidates, `stop · eot · step ${step + 1}`);
        break;
      }
      inputIds.push(chosenId);
      const piece = stream.pushToken(chosenId);
      if (piece) {
        const prevDisplay = normalizeCompletionText(rawReply, { trimEnd: false });
        rawReply += piece;
        const stopCheck = findStopSequence(rawReply);
        if (stopCheck.stopped) {
          rawReply = rawReply.slice(0, stopCheck.index);
          setTextContent(assistantBubble, stopCheck.text);
          renderLogitBars(topCandidates, `stop · ${stopCheck.sequence.trim()} · step ${step + 1}`);
          break;
        }
        const nextDisplay = stopCheck.text;
        if (nextDisplay.startsWith(prevDisplay)) {
          appendText(assistantBubble, nextDisplay.slice(prevDisplay.length));
        } else {
          setTextContent(assistantBubble, nextDisplay);
        }
        if (stickToBottom) scrollToBottom(chatThread);
      }
      renderLogitBars(topCandidates, `step ${step + 1}`);
      const elapsedSec = Math.max((performance.now() - t0) / 1000, 0.001);
      setTextContent(tokSpeedEl, `${((step + 1) / elapsedSec).toFixed(1)} tok/s`);
      await new Promise((r) => requestAnimationFrame(r));
    }
    rawReply += stream.flush();
    const reply = normalizeCompletionText(rawReply, { trimEnd: true });
    setTextContent(assistantBubble, reply || (stopRequested ? '(stopped)' : '(empty reply)'));
    chatMessages.push(createMessage('assistant', reply || ''));
  } catch (err) {
    setTextContent(assistantBubble, `Generation failed: ${err.message}`);
    assistantBubble.parentElement.classList.add('error');
  } finally {
    isGenerating = false;
    updateComposerEnabled();
    composerInput.focus();
  }
}

function toggleAdvanced(show) {
  const open = show ?? advancedPanel.classList.contains('hidden');
  advancedPanel.classList.toggle('hidden', !open);
  advancedPanel.hidden = !open;
  btnToggleAdvanced.setAttribute('aria-expanded', open ? 'true' : 'false');
}

function autosizeComposer() {
  composerInput.style.height = 'auto';
  composerInput.style.height = `${Math.min(composerInput.scrollHeight, 144)}px`;
}

chatThread.addEventListener('scroll', () => {
  stickToBottom = isPinnedToBottom(chatThread);
});

composerInput.addEventListener('input', () => {
  autosizeComposer();
  updateComposerEnabled();
});
composerInput.addEventListener('keydown', (event) => {
  handleComposerKeydown(event, {
    isDisabled: () => btnSend.disabled,
    onSend: () => composer.requestSubmit(),
  });
});
composer.addEventListener('submit', (event) => {
  event.preventDefault();
  const text = composerInput.value.trim();
  if (!text || btnSend.disabled) return;
  composerInput.value = '';
  autosizeComposer();
  updateComposerEnabled();
  runChatGeneration(text).catch(() => {});
});
btnStop.addEventListener('click', () => {
  stopRequested = true;
});
btnNewChat.addEventListener('click', () => {
  if (isGenerating) stopRequested = true;
  clearChatMessages();
  composerInput.focus();
});
btnToggleAdvanced.addEventListener('click', () => toggleAdvanced());
btnCloseAdvanced.addEventListener('click', () => toggleAdvanced(false));
document.getElementById('suggestion-row')?.addEventListener('click', (event) => {
  const btn = event.target.closest('[data-prompt]');
  if (!btn) return;
  composerInput.value = btn.getAttribute('data-prompt') || '';
  autosizeComposer();
  updateComposerEnabled();
  composerInput.focus();
});

ckptSlider.addEventListener('input', (e) => {
  const idx = Number(e.target.value);
  if (!isGenerating) checkpointManager?.selectIndex(idx).catch(() => {});
});
tempSlider.addEventListener('input', (e) => setTextContent(tempVal, Number(e.target.value).toFixed(2)));
topkSlider.addEventListener('input', (e) => setTextContent(topkVal, e.target.value));
toppSlider.addEventListener('input', (e) => setTextContent(toppVal, Number(e.target.value).toFixed(2)));
reppenSlider.addEventListener('input', (e) => setTextContent(reppenVal, Number(e.target.value).toFixed(2)));
maxTokensSlider.addEventListener('input', (e) => setTextContent(maxTokensVal, e.target.value));
btnFetchOnnx.addEventListener('click', () => {
  const url = onnxUrlInput.value.trim();
  if (url && !isGenerating) checkpointManager?.mountCustomUrl(url).catch(() => {});
});

await detectWebGPU();
checkpointManager = new CheckpointManager({
  hasWebGPU,
  maxCachedSessions: 1,
  onStatusChange: handleCheckpointStatus,
});
await initTokenizer();
await loadManifestAndArtifacts();
updateComposerEnabled();