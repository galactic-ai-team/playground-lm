/**
 * PlaygroundLM Demo App — Strict Manifest, Real GPT-2 BPE & ONNX Runtime Web Inference
 *
 * Capabilities:
 * - Uses `./gpt2_bpe.js` (`loadGPT2Tokenizer`, `GPT2_EOT_ID`) for exact 50,257-vocab GPT-2 BPE.
 * - Caches mounted ONNX sessions (`Map<url, InferenceSession>`, LRU max 3) so scrubbing the
 *   checkpoint slider swaps between Stage B / Stage C checkpoints instantaneously.
 * - Supports Temperature, Top-K, Top-P (nucleus), and Repetition Penalty sampling over `logits`.
 * - Renders live training/validation loss curves from `./loss_curve.json` (when present) and
 *   syncs the active checkpoint marker on the chart.
 */

import { loadGPT2Tokenizer, GPT2_EOT_ID } from './gpt2_bpe.js';
import { renderLossChart } from './loss_chart.js';

const COLUMN_LABELS = {
  model: 'Model / Step',
  step: 'Step',
  tokens_seen: 'Tokens Seen',
  val_loss: 'Val Loss (↓)',
  perplexity: 'Perplexity (↓)',
  hellaswag_acc_norm: 'HellaSwag acc_norm (↑)',
  lambada_acc: 'LAMBADA acc (↑)',
};

const MAX_CACHED_SESSIONS = 3;

// State
let manifest = {
  model: 'PlaygroundLM',
  checkpoints: [],
  benchmarks: {
    columns: ['val_loss', 'perplexity', 'hellaswag_acc_norm', 'lambada_acc'],
    rows: [],
  },
  riddle_prompts: [],
};
let lossCurveData = null;
let activeCheckpointStep = null;
let riddleIndex = 0;
let ortModule = null;
let onnxSession = null;
let loadedCheckpointUrl = null;
const sessionCache = new Map(); // url -> InferenceSession (LRU order)
let hasWebGPU = false;
let isGenerating = false;
let stopRequested = false;
let tokenizer = null;

// DOM Elements
const modelTitleEl = document.getElementById('model-title');
const modelBadgeEl = document.getElementById('model-badge');
const runtimeDot = document.getElementById('runtime-dot');
const runtimeLabel = document.getElementById('runtime-label');

const ckptEmptyState = document.getElementById('ckpt-empty-state');
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

// Restored from #62: demo/benchmarks.json is the primary benchmark source,
// with manifest.json benchmark rows as the fallback.
const BENCHMARK_COLUMNS = [
  'val_loss',
  'perplexity',
  'hellaswag_acc_norm',
  'lambada_acc',
];

function validBenchmarks(value) {
  return value
    && Array.isArray(value.columns)
    && BENCHMARK_COLUMNS.every((key) => value.columns.includes(key))
    && Array.isArray(value.rows)
    && value.rows.every((row) =>
      row && typeof row === 'object'
      && typeof row.label === 'string'
      && BENCHMARK_COLUMNS.every((key) =>
        typeof row[key] === 'number' && Number.isFinite(row[key])
      )
    );
}

async function loadBenchmarkTable() {
  try {
    const res = await fetch('./benchmarks.json', { cache: 'no-store' });
    if (!res.ok) throw new Error(`HTTP ${res.status}`);

    const table = await res.json();
    if (!validBenchmarks(table)) {
      throw new Error('Invalid benchmarks.json schema');
    }

    manifest.benchmarks = {
      columns: BENCHMARK_COLUMNS,
      rows: table.rows,
    };

    if (benchSourceNote) {
      benchSourceNote.textContent =
        table.source_type === 'reported_rounded_summary'
          ? 'Reported Stage B results (rounded). Full evaluation reports pending.'
          : 'Full benchmark reports · HellaSwag validation · LAMBADA test';
    }
  } catch (err) {
    console.warn('Using benchmark rows from manifest.json:', err);
    if (benchSourceNote) {
      benchSourceNote.textContent = 'Benchmark data from manifest.json';
    }
  }
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

  // Top-P (nucleus) filtering over the sorted Top-K candidates
  const pCutoff = Math.max(0.05, Math.min(1.0, topP));
  const nucleus = [];
  let cumProb = 0;
  for (const item of top) {
    nucleus.push(item);
    cumProb += item.prob;
    if (cumProb >= pCutoff && nucleus.length >= 1) break;
  }

  // Renormalize nucleus probabilities for sampling
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

/**
 * Loads exact GPT-2 byte-level BPE via `./gpt2_bpe.js` (`vocab.json`+`merges.txt` or `tokenizer.json`).
 */
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
      // Try next candidate path
    }
  }

  // Byte fallback only if vocab/merges files have not been deployed to demo/ yet
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

async function loadManifestAndLossCurve() {
  try {
    const res = await fetch('./manifest.json', { cache: 'no-store' });
    if (res.ok) {
      const parsed = await res.json();
      manifest = {
        model: parsed.model || 'PlaygroundLM',
        checkpoints: Array.isArray(parsed.checkpoints) ? parsed.checkpoints : [],
        benchmarks: {
          columns: Array.isArray(parsed.benchmarks?.columns)
            ? parsed.benchmarks.columns
            : ['val_loss', 'perplexity', 'hellaswag_acc_norm', 'lambada_acc'],
          rows: Array.isArray(parsed.benchmarks?.rows) ? parsed.benchmarks.rows : [],
        },
        riddle_prompts: Array.isArray(parsed.riddle_prompts) ? parsed.riddle_prompts : [],
      };
    }
  } catch (err) {
    console.warn('Could not load ./manifest.json:', err);
  }

  try {
    const lossRes = await fetch('./loss_curve.json', { cache: 'no-store' });
    if (lossRes.ok) {
      lossCurveData = await lossRes.json();
    }
  } catch {
    lossCurveData = null;
  }

  // The standalone benchmark table takes precedence when available.
  // Checkpoint URLs and riddle prompts still come from manifest.json.
  await loadBenchmarkTable();
  renderManifestUI();
  if (lossChartContainer) {
    renderLossChart(lossChartContainer, lossCurveData, {
      activeStep: activeCheckpointStep,
    });
  }
}

function formatNumber(val) {
  if (val === null || val === undefined || val === '') return '—';
  if (typeof val === 'number') {
    return Number.isInteger(val) ? val.toLocaleString() : val.toFixed(4).replace(/\.?0+$/, '');
  }
  return String(val);
}

function renderManifestUI() {
  modelTitleEl.textContent = manifest.model || 'PlaygroundLM';
  metaModel.textContent = manifest.model || 'PlaygroundLM';

  const ckpts = manifest.checkpoints || [];
  if (ckpts.length === 0) {
    modelBadgeEl.textContent = 'Training in Progress (0 Exported Checkpoints)';
    ckptEmptyState.classList.remove('hidden');
    ckptControls.classList.add('hidden');
    metaStep.textContent = 'Stage C running';
    metaTokens.textContent = '—';
  } else {
    modelBadgeEl.textContent = `${ckpts.length} Checkpoint${ckpts.length === 1 ? '' : 's'}`;
    ckptEmptyState.classList.add('hidden');
    ckptControls.classList.remove('hidden');

    ckptSlider.max = String(ckpts.length - 1);
    ckptSlider.value = String(ckpts.length - 1);
    ckptStageButtons.innerHTML = '';

    ckpts.forEach((c, idx) => {
      const btn = document.createElement('button');
      btn.className = 'stage-btn';
      btn.dataset.idx = String(idx);
      const label = c.label ? `${c.label}<br>` : '';
      btn.innerHTML = `${label}Step ${Number(c.step ?? idx).toLocaleString()}<br><small>${Number(c.tokens_seen ?? 0).toLocaleString()} tok</small>`;
      btn.addEventListener('click', () => selectCheckpoint(idx));
      ckptStageButtons.appendChild(btn);
    });

    selectCheckpoint(ckpts.length - 1);
  }

  // Riddle Prompts
  const riddles = manifest.riddle_prompts || [];
  if (riddles.length > 0) {
    btnRiddle.disabled = false;
    btnRiddle.textContent = `🧩 Riddle Prompt (1/${riddles.length})`;
    if (!promptInput.value.trim()) {
      promptInput.value = riddles[0];
    }
  } else {
    btnRiddle.disabled = true;
    btnRiddle.textContent = '🧩 Riddle Prompt (none in manifest)';
  }

  // Benchmarks Table
  const cols = manifest.benchmarks?.columns || [
    'val_loss',
    'perplexity',
    'hellaswag_acc_norm',
    'lambada_acc',
  ];
  const rows = manifest.benchmarks?.rows || [];

  if (rows.length === 0) {
    benchEmptyState.classList.remove('hidden');
    benchTableWrap.classList.add('hidden');
  } else {
    benchEmptyState.classList.add('hidden');
    benchTableWrap.classList.remove('hidden');

    const trHead = document.createElement('tr');
    const thName = document.createElement('th');
    thName.textContent = 'Model / Checkpoint';
    trHead.appendChild(thName);

    for (const col of cols) {
      const th = document.createElement('th');
      th.textContent = COLUMN_LABELS[col] || col;
      trHead.appendChild(th);
    }
    benchThead.innerHTML = '';
    benchThead.appendChild(trHead);

    benchTbody.innerHTML = '';
    for (const row of rows) {
      const tr = document.createElement('tr');
      if (row.step !== undefined) tr.dataset.step = String(row.step);

      const tdLabel = document.createElement('td');
      const rowName =
        row.model ||
        row.name ||
        row.label ||
        (row.step !== undefined ? `Step ${row.step}` : 'Run');
      tdLabel.textContent = String(rowName);
      tr.appendChild(tdLabel);

      for (const col of cols) {
        const td = document.createElement('td');
        const val = Array.isArray(row) ? row[cols.indexOf(col)] : row[col];
        td.textContent =
          typeof val === 'number'
          && (col === 'hellaswag_acc_norm' || col === 'lambada_acc')
            ? `${(val * 100).toFixed(1)}%`
            : formatNumber(val);
        tr.appendChild(td);
      }
      benchTbody.appendChild(tr);
    }
  }
}

async function selectCheckpoint(index) {
  const ckpts = manifest.checkpoints || [];
  const ckpt = ckpts[index];
  if (!ckpt) return;

  ckptSlider.value = String(index);
  document.querySelectorAll('.stage-btn').forEach((btn) => {
    btn.classList.toggle('active', Number(btn.dataset.idx) === index);
  });

  activeCheckpointStep = ckpt.step !== undefined ? Number(ckpt.step) : null;
  metaStep.textContent = activeCheckpointStep !== null ? activeCheckpointStep.toLocaleString() : '—';
  metaTokens.textContent =
    ckpt.tokens_seen !== undefined ? Number(ckpt.tokens_seen).toLocaleString() : '—';
  ckptUrlTag.textContent = ckpt.url || 'no url';

  document.querySelectorAll('#bench-tbody tr').forEach((tr) => {
    tr.classList.toggle(
      'active-stage-row',
      tr.dataset.step !== undefined && Number(tr.dataset.step) === activeCheckpointStep
    );
  });

  if (lossChartContainer && lossCurveData) {
    renderLossChart(lossChartContainer, lossCurveData, {
      activeStep: activeCheckpointStep,
    });
  }

  if (ckpt.url) {
    await mountOnnxSession(ckpt.url);
  } else {
    onnxSession = null;
    loadedCheckpointUrl = null;
    metaSession.textContent = 'No URL in checkpoint';
    btnGenerate.disabled = true;
  }
}

async function mountOnnxSession(url) {
  if (!url) return;

  // Instant swap if already cached in LRU session map
  if (sessionCache.has(url)) {
    const cached = sessionCache.get(url);
    sessionCache.delete(url);
    sessionCache.set(url, cached);
    onnxSession = cached;
    loadedCheckpointUrl = url;
    metaSession.textContent = `Cached (${hasWebGPU ? 'WebGPU' : 'WASM'})`;
    ckptUrlTag.textContent = url.split('/').pop() || url;
    btnGenerate.disabled = false;
    return;
  }

  metaSession.textContent = 'Loading ONNX...';
  btnGenerate.disabled = true;

  try {
    if (!ortModule) {
      ortModule = await import('https://cdn.jsdelivr.net/npm/onnxruntime-web/dist/ort.min.mjs');
    }
    const session = await ortModule.InferenceSession.create(url, {
      executionProviders: hasWebGPU ? ['webgpu', 'wasm'] : ['wasm'],
    });

    if (sessionCache.size >= MAX_CACHED_SESSIONS) {
      const oldestKey = sessionCache.keys().next().value;
      sessionCache.delete(oldestKey);
    }
    sessionCache.set(url, session);

    onnxSession = session;
    loadedCheckpointUrl = url;
    metaSession.textContent = `Mounted (${hasWebGPU ? 'WebGPU' : 'WASM'})`;
    ckptUrlTag.textContent = url.split('/').pop() || url;
    btnGenerate.disabled = false;
    outputStream.innerHTML = `<span class="muted">Checkpoint <code>${url}</code> mounted. Enter a prompt and click Generate.</span>`;
  } catch (err) {
    onnxSession = null;
    loadedCheckpointUrl = null;
    metaSession.textContent = 'Mount failed';
    runtimeDot.classList.remove('ready');
    runtimeDot.classList.add('error');
    outputStream.innerHTML = `<span class="muted">Failed to load ONNX checkpoint from <code>${url}</code>: ${err.message}</span>`;
    btnGenerate.disabled = true;
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
  if (isGenerating || !onnxSession || !ortModule) return;

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
      const inputTensor = new ortModule.Tensor('int64', bigIntIds, [1, contextIds.length]);

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

// Event Listeners
ckptSlider.addEventListener('input', (e) => selectCheckpoint(Number(e.target.value)));

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
  const riddles = manifest.riddle_prompts || [];
  if (!riddles.length) return;
  riddleIndex = (riddleIndex + 1) % riddles.length;
  promptInput.value = riddles[riddleIndex];
  btnRiddle.textContent = `🧩 Riddle Prompt (${riddleIndex + 1}/${riddles.length})`;
});

btnToggleCustomOnnx.addEventListener('click', () => {
  onnxUrlBar.classList.toggle('hidden');
});

btnFetchOnnx.addEventListener('click', () => {
  const url = onnxUrlInput.value.trim();
  if (url) mountOnnxSession(url);
});

btnGenerate.addEventListener('click', runOnnxGeneration);
btnStop.addEventListener('click', () => {
  stopRequested = true;
});

// Boot
await detectWebGPU();
await initTokenizer();
await loadManifestAndLossCurve();
