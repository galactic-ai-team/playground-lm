/**
 * PlaygroundLM Demo App — Strict Manifest & Real ONNX Runtime Web Inference
 *
 * Contract:
 * - Reads ./manifest.json:
 *   {
 *     "model": string,
 *     "checkpoints": [{ "step": number, "tokens_seen": number, "url": string }],
 *     "benchmarks": {
 *       "columns": ["val_loss", "perplexity", "hellaswag_acc_norm", "lambada_acc"],
 *       "rows": [...]
 *     },
 *     "riddle_prompts": string[]
 *   }
 * - Zero hard-coded scores or simulated text continuations.
 * - Real ONNX execution using input tensor `input_ids` (int64 [1, seq_len])
 *   and output tensor `logits` (float32 [1, seq_len, vocab_size]).
 */

const COLUMN_LABELS = {
  model: 'Model / Step',
  step: 'Step',
  val_loss: 'Val Loss (↓)',
  perplexity: 'Perplexity (↓)',
  hellaswag_acc_norm: 'HellaSwag acc_norm (↑)',
  lambada_acc: 'LAMBADA acc (↑)',
};

// State
let manifest = {
  model: 'PlaygroundLM',
  checkpoints: [],
  benchmarks: { columns: ['val_loss', 'perplexity', 'hellaswag_acc_norm', 'lambada_acc'], rows: [] },
  riddle_prompts: [],
};
let riddleIndex = 0;
let ortModule = null;
let onnxSession = null;
let loadedCheckpointUrl = null;
let hasWebGPU = false;
let isGenerating = false;
let stopRequested = false;
let tokenizer = null; // Optional { encode(str): number[], decode(ids): string, idToToken(id): string }

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
const maxTokensSlider = document.getElementById('max-tokens-slider');
const maxTokensVal = document.getElementById('max-tokens-val');
const btnGenerate = document.getElementById('btn-generate');
const btnStop = document.getElementById('btn-stop');

const tokenizerStatus = document.getElementById('tokenizer-status');
const tokSpeedEl = document.getElementById('tok-speed');
const outputStream = document.getElementById('output-stream');
const logitBars = document.getElementById('logit-bars');
const inspectedTokenTag = document.getElementById('inspected-token-tag');

const benchEmptyState = document.getElementById('bench-empty-state');
const benchTableWrap = document.getElementById('bench-table-wrap');
const benchThead = document.getElementById('bench-thead');
const benchTbody = document.getElementById('bench-tbody');
const benchSourceNote = document.getElementById('bench-source-note');

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

    benchSourceNote.textContent =
      table.source_type === 'reported_rounded_summary'
        ? 'Reported Stage B results (rounded). Full evaluation reports pending.'
        : 'Full benchmark reports · HellaSwag validation · LAMBADA test';
  } catch (err) {
    console.warn('Using benchmark rows from manifest.json:', err);
    benchSourceNote.textContent =
      'Benchmark data from manifest.json';
  }
}

/**
 * Top-K + Temperature Softmax over a Float32Array slice of vocab logits
 */
export function sampleTopKLogits(logitsSlice, temperature = 0.8, topK = 40) {
  const temp = Math.max(temperature, 1e-3);
  const k = Math.max(1, Math.min(topK, logitsSlice.length));

  // Maintain a simple sorted top-K buffer to avoid sorting all 50,257 vocab items when K is small
  const top = [];
  let minAllowed = -Infinity;

  for (let i = 0; i < logitsSlice.length; i++) {
    const val = logitsSlice[i];
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

  // Multinomial sample from top-K distribution
  const r = Math.random();
  let acc = 0;
  let chosenId = top[0].id;
  for (const item of top) {
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
 * Try loading optional tokenizer artifacts exported by GPT-6 (./tokenizer.json or ./vocab.json)
 */
async function tryLoadTokenizer() {
  try {
    const res = await fetch('./tokenizer.json', { cache: 'no-store' });
    if (!res.ok) throw new Error('No tokenizer.json');
    const data = await res.json();
    const vocabObj = data.model?.vocab || data.vocab || {};
    const tokenToId = new Map(Object.entries(vocabObj).map(([k, v]) => [k, Number(v)]));
    const idToTokenMap = new Map(Object.entries(vocabObj).map(([k, v]) => [Number(v), k]));

    tokenizer = {
      encode(text) {
        // Greedy longest-prefix match against loaded vocabulary with byte fallback
        const ids = [];
        let i = 0;
        const normalized = text.replace(/ /g, 'Ġ');
        while (i < normalized.length) {
          let matched = false;
          for (let len = Math.min(24, normalized.length - i); len >= 1; len--) {
            const sub = normalized.slice(i, i + len);
            if (tokenToId.has(sub)) {
              ids.push(tokenToId.get(sub));
              i += len;
              matched = true;
              break;
            }
          }
          if (!matched) {
            ids.push(normalized.charCodeAt(i) % Math.max(idToTokenMap.size, 256));
            i += 1;
          }
        }
        return ids.length ? ids : [0];
      },
      decode(id) {
        const raw = idToTokenMap.get(Number(id));
        if (raw === undefined) return `<#${id}>`;
        return raw.replace(/Ġ/g, ' ').replace(/Ċ/g, '\n');
      },
    };
    tokenizerStatus.textContent = `(loaded tokenizer.json · ${idToTokenMap.size.toLocaleString()} tokens)`;
  } catch {
    // Fallback to raw UTF-8 byte encoding until tokenizer.json is exported
    const encoder = new TextEncoder();
    const decoder = new TextDecoder();
    tokenizer = {
      encode(text) {
        const bytes = Array.from(encoder.encode(text));
        return bytes.length ? bytes : [32];
      },
      decode(id) {
        const num = Number(id);
        if (num >= 32 && num <= 126) return decoder.decode(new Uint8Array([num]));
        if (num === 10) return '\n';
        return `[tok:${num}]`;
      },
    };
    tokenizerStatus.textContent = '(tokenizer.json not found yet — using UTF-8 byte ID fallback)';
  }
}

async function loadManifest() {
  try {
    const res = await fetch('./manifest.json', { cache: 'no-store' });
    if (!res.ok) throw new Error(`HTTP ${res.status}`);
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
  } catch (err) {
    console.warn('Could not load ./manifest.json, using empty default contract:', err);
  }

  // The standalone benchmark table takes precedence when available.
  // Checkpoint URLs and riddle prompts still come from manifest.json.
  await loadBenchmarkTable();
  renderManifestUI();
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
    modelBadgeEl.textContent = 'Untrained (0 Checkpoints)';
    ckptEmptyState.classList.remove('hidden');
    ckptControls.classList.add('hidden');
    metaStep.textContent = 'Not trained yet';
    metaTokens.textContent = '0';
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
      btn.innerHTML = `Step ${Number(c.step ?? idx).toLocaleString()}<br><small>${Number(c.tokens_seen ?? 0).toLocaleString()} tok</small>`;
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
  const cols = manifest.benchmarks?.columns || ['val_loss', 'perplexity', 'hellaswag_acc_norm', 'lambada_acc'];
  const rows = manifest.benchmarks?.rows || [];

  if (rows.length === 0) {
    benchEmptyState.classList.remove('hidden');
    benchTableWrap.classList.add('hidden');
  } else {
    benchEmptyState.classList.add('hidden');
    benchTableWrap.classList.remove('hidden');

    // Determine row identifier column (e.g. model, name, label, or step)
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
      const rowName = row.model || row.name || row.label || (row.step !== undefined ? `Step ${row.step}` : 'Run');
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

  metaStep.textContent = ckpt.step !== undefined ? Number(ckpt.step).toLocaleString() : '—';
  metaTokens.textContent = ckpt.tokens_seen !== undefined ? Number(ckpt.tokens_seen).toLocaleString() : '—';
  ckptUrlTag.textContent = ckpt.url || 'no url';

  document.querySelectorAll('#bench-tbody tr').forEach((tr) => {
    tr.classList.toggle('active-stage-row', tr.dataset.step !== undefined && Number(tr.dataset.step) === Number(ckpt.step));
  });

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
  if (onnxSession && loadedCheckpointUrl === url) return;

  metaSession.textContent = 'Loading ONNX...';
  btnGenerate.disabled = true;

  try {
    if (!ortModule) {
      ortModule = await import('https://cdn.jsdelivr.net/npm/onnxruntime-web/dist/ort.min.mjs');
    }
    onnxSession = await ortModule.InferenceSession.create(url, {
      executionProviders: hasWebGPU ? ['webgpu', 'wasm'] : ['wasm'],
    });
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
  const maxNewTokens = parseInt(maxTokensSlider.value, 10);

  outputStream.innerHTML = '';
  const promptSpan = document.createElement('span');
  promptSpan.className = 'tok-prompt';
  promptSpan.textContent = promptText;
  outputStream.appendChild(promptSpan);

  const inputIds = tokenizer.encode(promptText);
  const stepHistory = [];
  const t0 = performance.now();

  try {
    for (let step = 0; step < maxNewTokens; step++) {
      if (stopRequested) break;

      // Context window cap at 1024 tokens
      const contextIds = inputIds.slice(-1024);
      const bigIntIds = BigInt64Array.from(contextIds.map((id) => BigInt(id)));
      const inputTensor = new ortModule.Tensor('int64', bigIntIds, [1, contextIds.length]);

      const feeds = { input_ids: inputTensor };
      const results = await onnxSession.run(feeds);

      const logitsTensor = results.logits ?? results[Object.keys(results)[0]];
      if (!logitsTensor) {
        throw new Error('ONNX output did not contain a "logits" tensor.');
      }

      const dims = logitsTensor.dims; // Expected: [1, seq_len, vocab_size] or [seq_len, vocab_size]
      const vocabSize = dims[dims.length - 1];
      const seqLen = dims.length >= 2 ? dims[dims.length - 2] : 1;
      const lastPosOffset = (seqLen - 1) * vocabSize;
      const lastTokenLogits = logitsTensor.data.subarray(lastPosOffset, lastPosOffset + vocabSize);

      const { chosenId, topCandidates } = sampleTopKLogits(lastTokenLogits, temperature, topK);
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

      // Yield to browser paint loop
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
await tryLoadTokenizer();
await loadManifest();