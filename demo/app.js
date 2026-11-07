/**
 * PlaygroundLM Demo App — Strict Manifest, Benchmarks.json, GPT-2 BPE & Non-Reloading Checkpoint Slider
 *
 * Preserves:
 * - `loadBenchmarkTable()` reading `./benchmarks.json` (with % accuracy columns and source note,
 *   #62/#70) and falling back to `manifest.benchmarks` when needed.
 * - `./gpt2_bpe.js` (`loadGPT2Tokenizer`, `GPT2_EOT_ID`) for exact 50,257-vocab GPT-2 BPE.
 * - `./checkpoint_loader.js` (`CheckpointManager`, `resolveCheckpointUrl`) supporting relative/local
 *   ONNX paths and a clear "Models coming soon" banner when `onnx_url` is null/empty.
 * - `./loss_chart.js` (`renderLossChart`, `DEFAULT_STAGE_C_MILESTONES`) reading `./loss_curve.json`
 *   and overlaying Stage-C session markers (2035 / 10907 / 19366 / 31101 target).
 * - "New riddle" button (`#btn-riddle`) picking a random riddle from `manifest.riddle_prompts`.
 */

import { loadGPT2Tokenizer, GPT2_EOT_ID } from './gpt2_bpe.js';
import { renderLossChart, DEFAULT_STAGE_C_MILESTONES } from './loss_chart.js';
import { CheckpointManager } from './checkpoint_loader.js';

export const DEFAULT_RIDDLE_PROMPTS = [
  'Riddle: I have keys but no locks. I have space but no room. You can enter, but you cannot go outside. What am I? Answer:',
  'Riddle: I have hands but I cannot clap, and a face that never smiles. What am I? Answer:',
  'Riddle: The more of me you take, the more you leave behind. What am I? Answer:',
  'Riddle: I speak without a mouth and hear without ears. I come alive with the wind. What am I? Answer:',
  'Riddle: I have cities, but no houses. I have mountains, but no trees. I have water, but no fish. What am I? Answer:',
  'Riddle: What gets wetter and wetter the more it dries? Answer:',
  'Riddle: I follow you all day long in the sun, but disappear when the night comes. What am I? Answer:',
  'Riddle: I have a single eye, yet I cannot see a thing. I pull a thread wherever I go. What am I? Answer:',
  'Riddle: I am tall when I am young, and I am short when I am old. What am I? Answer:',
  'Riddle: What has to be broken before you can cook and eat it? Answer:',
  'Riddle: I have a neck with no head, and I hold water inside. What am I? Answer:',
  'Riddle: I have teeth that cannot bite, and I glide through tangled hair. What am I? Answer:',
  'Riddle: I fall from the clouds without ever getting hurt, and vanish when the sun gets warm. What am I? Answer:',
  'Riddle: I am full of holes, yet I can still hold plenty of water. What am I? Answer:',
  'Riddle: I have pages instead of leaves, and a spine with no bones. What am I? Answer:',
  'Riddle: What runs around a whole yard without ever moving an inch? Answer:',
  'Riddle: What belongs to you, yet other people use it far more than you do? Answer:',
  'Riddle: I can fill an entire room without taking up any space at all. What am I? Answer:',
  'Riddle: I have four legs in the morning, two at noon, and three in the evening, or I stand still beneath your dinner plate. What am I? Answer:',
  'Riddle: Forward I am heavy, but backward I am not. What word am I? Answer:',
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

// State
let manifest = {
  model: 'PlaygroundLM',
  checkpoints: [],
  milestones: DEFAULT_STAGE_C_MILESTONES.slice(),
  benchmarks: {
    columns: ['val_loss', 'perplexity', 'hellaswag_acc_norm', 'lambada_acc'],
    rows: [],
  },
  riddle_prompts: DEFAULT_RIDDLE_PROMPTS.slice(),
};
let lossCurveData = null;
let activeCheckpointStep = null;
let lastRiddleIndex = -1;
let onnxSession = null;
let hasWebGPU = false;
let isGenerating = false;
let stopRequested = false;
let activeGenerationPromise = null;
let tokenizer = null;
let checkpointManager = null;

// DOM Elements
const modelTitleEl = document.getElementById('model-title');
const modelBadgeEl = document.getElementById('model-badge');
const runtimeDot = document.getElementById('runtime-dot');
const runtimeLabel = document.getElementById('runtime-label');

const ckptEmptyState = document.getElementById('ckpt-empty-state');
const ckptComingSoonBanner = document.getElementById('ckpt-coming-soon-banner');
const ckptControls = document.getElementById('ckpt-controls');
const ckptSlider = document.getElementById('ckpt-slider');
const ckptStageButtons = document.getElementById('ckpt-stage-buttons');
const ckptUrlTag = document.getElementById('ckpt-url-tag');

const metaModel = document.getElementById('meta-model');
const metaStep = document.getElementById('meta-step');
const metaTokens = document.getElementById('meta-tokens');
const metaSession = document.getElementById('meta-session');

const btnRiddle = document.getElementById('btn-riddle');
const btnToggleCustomOnnx = document.getElementById('btn-toggle-custom-onnx');
const onnxUrlBar = document.getElementById('onnx-url-bar');
const onnxUrlInput = document.getElementById('onnx-url-input');
const btnFetchOnnx = document.getElementById('btn-fetch-onnx');

const promptInput = document.getElementById('prompt-input');
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
const btnGenerate = document.getElementById('btn-generate');
const btnStop = document.getElementById('btn-stop');

const tokenizerStatus = document.getElementById('tokenizer-status');
const tokSpeedEl = document.getElementById('tok-speed');
const outputStream = document.getElementById('output-stream');
const logitBars = document.getElementById('logit-bars');
const inspectedTokenTag = document.getElementById('inspected-token-tag');
const lossChartContainer = document.getElementById('loss-chart-container');

const benchEmptyState = document.getElementById('bench-empty-state');
const benchTableWrap = document.getElementById('bench-table-wrap');
const benchThead = document.getElementById('bench-thead');
const benchTbody = document.getElementById('bench-tbody');
const benchSourceNote = document.getElementById('bench-source-note');

function getActiveMilestones() {
  if (Array.isArray(lossCurveData?.milestones) && lossCurveData.milestones.length > 0) {
    return lossCurveData.milestones;
  }
  if (Array.isArray(manifest?.milestones) && manifest.milestones.length > 0) {
    return manifest.milestones;
  }
  return DEFAULT_STAGE_C_MILESTONES;
}

/**
 * Picks a random riddle from `pool`, avoiding `lastIndex` when `pool.length > 1`.
 */
export function pickRandomRiddle(pool, lastIndex = -1) {
  const list = Array.isArray(pool) && pool.length > 0 ? pool : DEFAULT_RIDDLE_PROMPTS;
  if (list.length === 1) {
    return { text: list[0], index: 0 };
  }
  let idx = Math.floor(Math.random() * list.length);
  if (idx === lastIndex) {
    idx = (idx + 1 + Math.floor(Math.random() * (list.length - 1))) % list.length;
  }
  return { text: list[idx], index: idx };
}

/**
 * Applies repetition penalty, Temperature scaling, Top-K filtering, and Top-P (nucleus)
 * cutoff over a Float32Array slice of vocabulary logits.
 */
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

async function detectWebGPU() {
  if (typeof navigator !== 'undefined' && 'gpu' in navigator) {
    try {
      const adapter = await navigator.gpu.requestAdapter();
      hasWebGPU = Boolean(adapter);
    } catch {
      hasWebGPU = false;
    }
  }
  runtimeDot.classList.add('ready');
  runtimeLabel.textContent = hasWebGPU
    ? 'WebGPU Ready · ONNX Runtime Web'
    : 'WASM Ready · ONNX Runtime Web';
}

async function initTokenizer() {
  const candidates = [
    ['./vocab.json', './merges.txt'],
    ['./encoder.json', './merges.txt'],
    ['./tokenizer.json', './merges.txt'],
  ];

  for (const [vUrl, mUrl] of candidates) {
    try {
      tokenizer = await loadGPT2Tokenizer(vUrl, mUrl);
      tokenizerStatus.textContent = `(GPT-2 BPE · ${tokenizer.vocabSize.toLocaleString()} tokens · EOT ${tokenizer.eotId})`;
      return;
    } catch {
      // Try next candidate
    }
  }

  const encoder = new TextEncoder();
  const decoder = new TextDecoder();
  tokenizer = {
    vocabSize: 256,
    eotId: GPT2_EOT_ID,
    encode(text) {
      const bytes = Array.from(encoder.encode(text));
      return bytes.length ? bytes : [32];
    },
    decode(ids) {
      const list = typeof ids === 'number' ? [ids] : Array.from(ids || []);
      const bytes = list.filter((n) => n >= 0 && n <= 255);
      return decoder.decode(new Uint8Array(bytes));
    },
  };
  tokenizerStatus.textContent = '(vocab.json/merges.txt not deployed yet — UTF-8 byte fallback)';
}

function formatMetricCell(val, colKey = '') {
  if (val === null || val === undefined || val === '') return '—';
  if (typeof val === 'number') {
    if (PERCENT_COLUMNS.has(colKey)) {
      const pct = val <= 1.0 && val >= 0 ? val * 100 : val;
      return `${pct.toFixed(1)}%`; // #62: 1 decimal (source values are rounded)
    }
    return Number.isInteger(val) ? val.toLocaleString() : val.toFixed(4).replace(/\.?0+$/, '');
  }
  return String(val);
}

/**
 * Loads `./benchmarks.json` (GPT-6's benchmark artifact, #62/#70) and renders the comparison
 * table with % accuracy formatting and source notes, falling back to `manifest.benchmarks` if needed.
 */
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

function updateStageButtonsCacheBadges() {
  if (!checkpointManager) return;
  document.querySelectorAll('.stage-btn').forEach((btn) => {
    const idx = Number(btn.dataset.idx);
    const ckpt = checkpointManager.checkpoints[idx];
    const cached = ckpt && checkpointManager.isCached(ckpt.resolvedUrl);
    btn.classList.toggle('cached', Boolean(cached));
  });
}

function handleCheckpointStatus(status) {
  const { state, index, checkpoint, session, message } = status;

  if (checkpoint) {
    activeCheckpointStep = checkpoint.step !== undefined ? Number(checkpoint.step) : null;
    metaStep.textContent =
      activeCheckpointStep !== null ? activeCheckpointStep.toLocaleString() : '—';
    metaTokens.textContent =
      checkpoint.tokens_seen !== undefined
        ? Number(checkpoint.tokens_seen).toLocaleString()
        : '—';
    const fileLabel =
      (checkpoint.resolvedUrl || '').split('/').pop() ||
      (checkpoint.quant ? `${checkpoint.quant} · coming soon` : 'coming soon');
    ckptUrlTag.textContent = fileLabel;
  }

  if (index >= 0) {
    ckptSlider.value = String(index);
    document.querySelectorAll('.stage-btn').forEach((btn) => {
      btn.classList.toggle('active', Number(btn.dataset.idx) === index);
    });
  }

  document.querySelectorAll('#bench-tbody tr').forEach((tr) => {
    tr.classList.toggle(
      'active-stage-row',
      tr.dataset.step !== undefined && Number(tr.dataset.step) === activeCheckpointStep
    );
  });

  if (lossChartContainer && lossCurveData) {
    renderLossChart(lossChartContainer, lossCurveData, {
      activeStep: activeCheckpointStep,
      milestones: getActiveMilestones(),
    });
  }

  metaSession.textContent = message || state;
  updateStageButtonsCacheBadges();

  if (ckptComingSoonBanner) {
    ckptComingSoonBanner.classList.toggle('hidden', state !== 'coming_soon');
  }

  if (state === 'mounted' && session) {
    onnxSession = session;
    runtimeDot.classList.remove('error');
    runtimeDot.classList.add('ready');
    btnGenerate.disabled = isGenerating;
    if (!isGenerating && outputStream.querySelector('.muted')) {
      outputStream.innerHTML = `<span class="muted">Checkpoint <code>${checkpoint.resolvedUrl}</code> ready (${message}). Enter a prompt or click <b>New riddle</b> to generate.</span>`;
    }
  } else if (state === 'coming_soon') {
    onnxSession = null;
    btnGenerate.disabled = true;
    runtimeDot.classList.remove('error');
    runtimeDot.classList.add('ready');
  } else if (state === 'loading' || state === 'compiling') {
    onnxSession = null;
    btnGenerate.disabled = true;
  } else if (state === 'error') {
    onnxSession = null;
    btnGenerate.disabled = true;
    runtimeDot.classList.remove('ready');
    runtimeDot.classList.add('error');
    outputStream.innerHTML = `<span class="muted">Failed to load ONNX checkpoint from <code>${checkpoint?.resolvedUrl || ''}</code>: ${status.error?.message || message}</span>`;
  } else if (state === 'unmounted') {
    onnxSession = null;
    btnGenerate.disabled = true;
  }
}

async function loadManifestAndArtifacts() {
  try {
    const res = await fetch('./manifest.json', { cache: 'no-store' });
    if (res.ok) {
      const parsed = await res.json();
      const loadedRiddles =
        Array.isArray(parsed.riddle_prompts) && parsed.riddle_prompts.length > 0
          ? parsed.riddle_prompts
          : DEFAULT_RIDDLE_PROMPTS.slice();
      manifest = {
        model: parsed.model || 'PlaygroundLM',
        checkpoints: Array.isArray(parsed.checkpoints) ? parsed.checkpoints : [],
        milestones:
          Array.isArray(parsed.milestones) && parsed.milestones.length > 0
            ? parsed.milestones
            : DEFAULT_STAGE_C_MILESTONES.slice(),
        benchmarks: {
          columns: Array.isArray(parsed.benchmarks?.columns)
            ? parsed.benchmarks.columns
            : ['val_loss', 'perplexity', 'hellaswag_acc_norm', 'lambada_acc'],
          rows: Array.isArray(parsed.benchmarks?.rows) ? parsed.benchmarks.rows : [],
        },
        riddle_prompts: loadedRiddles,
      };
    }
  } catch (err) {
    console.warn('Could not load ./manifest.json, using default riddle prompts:', err);
  }

  try {
    const lossRes = await fetch('./loss_curve.json', { cache: 'no-store' });
    if (lossRes.ok) {
      lossCurveData = await lossRes.json();
    }
  } catch {
    lossCurveData = null;
  }

  renderManifestCheckpointsAndRiddles();
  await loadBenchmarkTable(manifest.benchmarks);

  if (lossChartContainer) {
    renderLossChart(lossChartContainer, lossCurveData, {
      activeStep: activeCheckpointStep,
      milestones: getActiveMilestones(),
    });
  }
}

function renderManifestCheckpointsAndRiddles() {
  modelTitleEl.textContent = manifest.model || 'PlaygroundLM';
  metaModel.textContent = manifest.model || 'PlaygroundLM';

  const ckpts = manifest.checkpoints || [];
  checkpointManager.setCheckpoints(ckpts);

  if (ckpts.length === 0) {
    modelBadgeEl.textContent = 'Models Coming Soon';
    ckptEmptyState.classList.remove('hidden');
    if (ckptComingSoonBanner) ckptComingSoonBanner.classList.add('hidden');
    ckptControls.classList.add('hidden');
    metaStep.textContent = 'Stage C s3 running';
    metaTokens.textContent = '—';
  } else {
    const hostedCount = checkpointManager.checkpoints.filter((c) => Boolean(c.resolvedUrl)).length;
    modelBadgeEl.textContent =
      hostedCount > 0
        ? `${hostedCount}/${ckpts.length} ONNX Ready`
        : `${ckpts.length} Steps · ONNX Coming Soon`;
    ckptEmptyState.classList.add('hidden');
    ckptControls.classList.remove('hidden');

    ckptSlider.min = '0';
    ckptSlider.max = String(ckpts.length - 1);
    ckptSlider.value = String(ckpts.length - 1);
    ckptStageButtons.innerHTML = '';

    checkpointManager.checkpoints.forEach((c, idx) => {
      const btn = document.createElement('button');
      btn.type = 'button';
      btn.className = 'stage-btn';
      btn.dataset.idx = String(idx);
      const label = c.label ? `${c.label} · ` : '';
      const quantBadge = c.quant ? ` (${c.quant.toUpperCase()})` : '';
      btn.innerHTML = `${label}Step ${c.step.toLocaleString()}${quantBadge}<br><small>${c.tokens_seen.toLocaleString()} tok</small>`;
      btn.addEventListener('click', () => {
        checkpointManager.selectIndex(idx).catch(() => {});
      });
      ckptStageButtons.appendChild(btn);
    });

    checkpointManager.selectIndex(ckpts.length - 1).catch(() => {});
  }

  const riddles =
    Array.isArray(manifest.riddle_prompts) && manifest.riddle_prompts.length > 0
      ? manifest.riddle_prompts
      : DEFAULT_RIDDLE_PROMPTS;
  btnRiddle.disabled = false;
  btnRiddle.textContent = '🧩 New riddle';
  if (!promptInput.value.trim()) {
    const initial = pickRandomRiddle(riddles, lastRiddleIndex);
    lastRiddleIndex = initial.index;
    promptInput.value = initial.text;
  }
}

function renderLogitBars(candidates, stepLabel) {
  inspectedTokenTag.textContent = stepLabel;
  logitBars.innerHTML = '';

  for (const item of candidates) {
    const tokStr = tokenizer ? tokenizer.decode(item.id) : `#${item.id}`;
    const pct = (item.prob * 100).toFixed(1);
    const row = document.createElement('div');
    row.className = 'logit-row';
    row.innerHTML = `
      <span class="logit-tok" title="Token ID ${item.id}">${JSON.stringify(tokStr)} <small class="muted">(${item.id})</small></span>
      <div class="logit-bar-track">
        <div class="logit-bar-fill" style="width: ${Math.max(2, item.prob * 100)}%"></div>
      </div>
      <span class="logit-prob">${pct}%</span>
    `;
    logitBars.appendChild(row);
  }
}

async function runOnnxGeneration() {
  const ort = checkpointManager?.ortModule;
  if (isGenerating || !onnxSession || !ort) return;

  isGenerating = true;
  stopRequested = false;
  btnGenerate.disabled = true;
  btnStop.disabled = false;

  const promptText = promptInput.value || ' ';
  const temperature = parseFloat(tempSlider.value);
  const topK = parseInt(topkSlider.value, 10);
  const topP = toppSlider ? parseFloat(toppSlider.value) : 0.92;
  const repetitionPenalty = reppenSlider ? parseFloat(reppenSlider.value) : 1.1;
  const maxNewTokens = parseInt(maxTokensSlider.value, 10);

  outputStream.innerHTML = '';
  const promptSpan = document.createElement('span');
  promptSpan.className = 'tok-prompt';
  promptSpan.textContent = promptText;
  outputStream.appendChild(promptSpan);

  const inputIds = tokenizer.encode(promptText);
  const eotId = tokenizer.eotId ?? GPT2_EOT_ID;
  const stepHistory = [];
  const t0 = performance.now();

  try {
    for (let step = 0; step < maxNewTokens; step++) {
      if (stopRequested) break;

      const contextIds = inputIds.slice(-1024);
      const bigIntIds = BigInt64Array.from(contextIds.map((id) => BigInt(id)));
      const inputTensor = new ort.Tensor('int64', bigIntIds, [1, contextIds.length]);

      const results = await onnxSession.run({ input_ids: inputTensor });
      const logitsTensor = results.logits ?? results[Object.keys(results)[0]];
      if (!logitsTensor) {
        throw new Error('ONNX output did not contain a "logits" tensor.');
      }

      const dims = logitsTensor.dims;
      const vocabSize = dims[dims.length - 1];
      const seqLen = dims.length >= 2 ? dims[dims.length - 2] : 1;
      const lastPosOffset = (seqLen - 1) * vocabSize;
      const lastTokenLogits = logitsTensor.data.subarray(
        lastPosOffset,
        lastPosOffset + vocabSize
      );

      const { chosenId, topCandidates } = sampleLogits(lastTokenLogits, {
        temperature,
        topK,
        topP,
        repetitionPenalty,
        recentTokenIds: inputIds,
      });

      if (chosenId === eotId) {
        renderLogitBars(topCandidates, `step #${step + 1} (<|endoftext|> reached)`);
        break;
      }

      inputIds.push(chosenId);

      const decodedPiece = tokenizer.decode(chosenId);
      const histIdx = stepHistory.length;
      stepHistory.push({ step: step + 1, chosenId, decodedPiece, topCandidates });

      const span = document.createElement('span');
      span.className = 'tok-gen';
      span.textContent = decodedPiece;
      span.title = `Token ID: ${chosenId} (click to inspect logits)`;
      span.addEventListener('click', () => {
        document.querySelectorAll('.tok-gen.selected').forEach((el) => el.classList.remove('selected'));
        span.classList.add('selected');
        const h = stepHistory[histIdx];
        renderLogitBars(h.topCandidates, `step #${h.step} (id ${h.chosenId})`);
      });
      outputStream.appendChild(span);

      renderLogitBars(topCandidates, `step #${step + 1} (live)`);

      const elapsedSec = Math.max((performance.now() - t0) / 1000, 0.001);
      tokSpeedEl.textContent = `${((step + 1) / elapsedSec).toFixed(1)} tok/s`;

      await new Promise((r) => requestAnimationFrame(r));
    }
  } catch (err) {
    const errSpan = document.createElement('div');
    errSpan.className = 'empty-banner';
    errSpan.textContent = `ONNX Inference Error: ${err.message}`;
    outputStream.appendChild(errSpan);
  } finally {
    isGenerating = false;
    btnGenerate.disabled = !onnxSession;
    btnStop.disabled = true;
  }
}

async function handleNewRiddleClick() {
  const pool =
    Array.isArray(manifest.riddle_prompts) && manifest.riddle_prompts.length > 0
      ? manifest.riddle_prompts
      : DEFAULT_RIDDLE_PROMPTS;

  const { text, index } = pickRandomRiddle(pool, lastRiddleIndex);
  lastRiddleIndex = index;
  promptInput.value = text;

  if (isGenerating) {
    stopRequested = true;
    if (activeGenerationPromise) {
      try {
        await activeGenerationPromise;
      } catch {
        // Ignore cancellation errors
      }
    }
  }

  if (onnxSession && checkpointManager?.ortModule) {
    activeGenerationPromise = runOnnxGeneration();
    await activeGenerationPromise;
  } else if (tokenizer) {
    const tokenIds = tokenizer.encode(text);
    outputStream.innerHTML = '';
    const promptSpan = document.createElement('span');
    promptSpan.className = 'tok-prompt';
    promptSpan.textContent = text;
    outputStream.appendChild(promptSpan);

    const noteSpan = document.createElement('div');
    noteSpan.className = 'muted';
    noteSpan.style.marginTop = '0.55rem';
    noteSpan.textContent = `[Riddle #${index + 1}/${pool.length} tokenized · ${tokenIds.length} GPT-2 BPE tokens: [${tokenIds.slice(0, 12).join(', ')}${tokenIds.length > 12 ? ', ...' : ''}] — ONNX models coming soon]`;
    outputStream.appendChild(noteSpan);
  }
}

// Event Listeners
ckptSlider.addEventListener('input', (e) => {
  const idx = Number(e.target.value);
  checkpointManager?.selectIndex(idx).catch(() => {});
});

tempSlider.addEventListener('input', (e) => (tempVal.textContent = Number(e.target.value).toFixed(2)));
topkSlider.addEventListener('input', (e) => (topkVal.textContent = e.target.value));
if (toppSlider) {
  toppSlider.addEventListener('input', (e) => (toppVal.textContent = Number(e.target.value).toFixed(2)));
}
if (reppenSlider) {
  reppenSlider.addEventListener('input', (e) => (reppenVal.textContent = Number(e.target.value).toFixed(2)));
}
maxTokensSlider.addEventListener('input', (e) => (maxTokensVal.textContent = e.target.value));

btnRiddle.addEventListener('click', () => {
  handleNewRiddleClick().catch(() => {});
});

btnToggleCustomOnnx.addEventListener('click', () => {
  onnxUrlBar.classList.toggle('hidden');
});

btnFetchOnnx.addEventListener('click', () => {
  const url = onnxUrlInput.value.trim();
  if (url) checkpointManager?.mountCustomUrl(url).catch(() => {});
});

btnGenerate.addEventListener('click', () => {
  activeGenerationPromise = runOnnxGeneration();
});
btnStop.addEventListener('click', () => {
  stopRequested = true;
});

// Boot
await detectWebGPU();
checkpointManager = new CheckpointManager({
  hasWebGPU,
  maxCachedSessions: 3,
  onStatusChange: handleCheckpointStatus,
});
await initTokenizer();
await loadManifestAndArtifacts();