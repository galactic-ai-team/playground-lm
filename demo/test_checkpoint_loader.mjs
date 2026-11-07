import test from 'node:test';
import assert from 'node:assert/strict';
import { CheckpointManager, resolveCheckpointUrl } from './checkpoint_loader.js';

test('resolveCheckpointUrl normalizes relative paths,onnx_url, and empty values', () => {
  assert.equal(resolveCheckpointUrl(null), '');
  assert.equal(resolveCheckpointUrl({ onnx_url: null }), '');
  assert.equal(resolveCheckpointUrl({ onnx_url: 'models/step_2035_q4.onnx' }), './models/step_2035_q4.onnx');
  assert.equal(resolveCheckpointUrl({ path: './checkpoints/s3_q8.onnx' }), './checkpoints/s3_q8.onnx');
  assert.equal(resolveCheckpointUrl({ url: '/static/model.onnx' }), '/static/model.onnx');
  assert.equal(
    resolveCheckpointUrl({ onnx_url: 'https://example.com/step_19366_q8.onnx' }),
    'https://example.com/step_19366_q8.onnx'
  );
});

test('CheckpointManager emits coming_soon state without fetching when onnx_url is missing', async () => {
  const statusEvents = [];
  let fetchCalled = false;
  const originalFetch = globalThis.fetch;
  globalThis.fetch = async () => {
    fetchCalled = true;
    throw new Error('Should not fetch when onnx_url is null');
  };

  try {
    const mgr = new CheckpointManager({
      hasWebGPU: false,
      onStatusChange(ev) {
        statusEvents.push(ev);
      },
    });

    mgr.setCheckpoints([
      { step: 2035, tokens_seen: 133365760, quant: 'q4', onnx_url: null },
      { step: 19366, tokens_seen: 1269161984, quant: 'q8', onnx_url: '' },
    ]);

    const res = await mgr.selectIndex(1);
    assert.equal(fetchCalled, false);
    assert.equal(res.comingSoon, true);
    assert.equal(res.session, null);
    assert.equal(statusEvents.length, 1);
    assert.equal(statusEvents[0].state, 'coming_soon');
    assert.match(statusEvents[0].message, /Models coming soon/i);
  } finally {
    globalThis.fetch = originalFetch;
  }
});

test('CheckpointManager caches ONNX sessions in LRU order and swaps without re-fetching', async () => {
  const fetchCalls = [];
  const releasedUrls = [];
  const statusEvents = [];

  const originalFetch = globalThis.fetch;
  globalThis.fetch = async (url) => {
    fetchCalls.push(url);
    const bytes = new TextEncoder().encode(`onnx-weights:${url}`);
    return {
      ok: true,
      status: 200,
      headers: new Map([['content-length', String(bytes.byteLength)]]),
      async arrayBuffer() {
        return bytes.buffer;
      },
    };
  };

  try {
    const mgr = new CheckpointManager({
      hasWebGPU: false,
      maxCachedSessions: 2,
      onStatusChange(ev) {
        statusEvents.push(ev);
      },
    });

    mgr.ortModule = {
      InferenceSession: {
        async create(buffer) {
          const tag = new TextDecoder().decode(buffer);
          return {
            tag,
            release() {
              releasedUrls.push(tag);
            },
          };
        },
      },
    };

    mgr.prefetchNeighbors = () => {};

    mgr.setCheckpoints([
      { step: 2035, tokens_seen: 133365760, onnx_url: 'models/ckpt_2035.onnx' },
      { step: 10500, tokens_seen: 688128000, url: './ckpt_10500.onnx' },
      { step: 19366, tokens_seen: 1269161984, path: './ckpt_19366.onnx' },
    ]);

    // 1. Mount index 0 (relative path normalized to ./models/ckpt_2035.onnx)
    const res0 = await mgr.selectIndex(0);
    assert.equal(res0.cached, false);
    assert.equal(res0.session.tag, 'onnx-weights:./models/ckpt_2035.onnx');
    assert.equal(fetchCalls.length, 1);

    // 2. Mount index 1 (ckpt_10500)
    const res1 = await mgr.selectIndex(1);
    assert.equal(res1.cached, false);
    assert.equal(fetchCalls.length, 2);

    // 3. Scrub back to index 0 -> instant LRU cache hit, zero new fetches
    const res0Cached = await mgr.selectIndex(0);
    assert.equal(res0Cached.cached, true);
    assert.equal(fetchCalls.length, 2);

    // 4. Mount index 2 (ckpt_19366) -> evicts least-recently-used index 1 (ckpt_10500)
    const res2 = await mgr.selectIndex(2);
    assert.equal(res2.cached, false);
    assert.equal(mgr.isCached('models/ckpt_2035.onnx'), true);
    assert.equal(mgr.isCached('./ckpt_19366.onnx'), true);
    assert.equal(mgr.isCached('./ckpt_10500.onnx'), false);
    assert.deepEqual(releasedUrls, ['onnx-weights:./ckpt_10500.onnx']);
    assert.ok(statusEvents.some((e) => e.state === 'mounted' && e.cached === true));
  } finally {
    globalThis.fetch = originalFetch;
  }
});