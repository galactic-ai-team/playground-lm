import test from 'node:test';
import assert from 'node:assert/strict';
import { CheckpointManager } from './checkpoint_loader.js';

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

    // Inject mock ORT module so tests run offline in Node
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

    // Disable neighbor prefetch in unit test to assert exact on-demand fetch counts
    mgr.prefetchNeighbors = () => {};

    mgr.setCheckpoints([
      { step: 2035, tokens_seen: 133365760, url: './ckpt_2035.onnx' },
      { step: 10000, tokens_seen: 655360000, url: './ckpt_10000.onnx' },
      { step: 31101, tokens_seen: 2038235136, url: './ckpt_31101.onnx' },
    ]);

    // 1. Mount index 0 (ckpt_2035)
    const res0 = await mgr.selectIndex(0);
    assert.equal(res0.cached, false);
    assert.equal(res0.session.tag, 'onnx-weights:./ckpt_2035.onnx');
    assert.equal(fetchCalls.length, 1);

    // 2. Mount index 1 (ckpt_10000)
    const res1 = await mgr.selectIndex(1);
    assert.equal(res1.cached, false);
    assert.equal(fetchCalls.length, 2);

    // 3. Scrub back to index 0 -> instant LRU cache hit, zero new fetches
    const res0Cached = await mgr.selectIndex(0);
    assert.equal(res0Cached.cached, true);
    assert.equal(fetchCalls.length, 2);

    // 4. Mount index 2 (ckpt_31101) -> evicts least-recently-used index 1 (ckpt_10000)
    const res2 = await mgr.selectIndex(2);
    assert.equal(res2.cached, false);
    assert.equal(mgr.isCached('./ckpt_2035.onnx'), true);
    assert.equal(mgr.isCached('./ckpt_31101.onnx'), true);
    assert.equal(mgr.isCached('./ckpt_10000.onnx'), false);
    assert.deepEqual(releasedUrls, ['onnx-weights:./ckpt_10000.onnx']);
    assert.ok(statusEvents.some((e) => e.state === 'mounted' && e.cached === true));
  } finally {
    globalThis.fetch = originalFetch;
  }
});
