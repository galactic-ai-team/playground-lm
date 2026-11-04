/**
 * PlaygroundLM — Non-Reloading ONNX Checkpoint Slider Manager & LRU Cache
 *
 * Fetches an ONNX model per checkpoint entry in `demo/manifest.json`, caches both
 * raw `ArrayBuffer` weights and instantiated `InferenceSession` objects in an LRU
 * cache, cancels stale in-flight fetches via `AbortController` when the user scrubs
 * the slider rapidly, and prefetches adjacent checkpoints in the background.
 */

export class CheckpointManager {
  /**
   * @param {object} options
   * @param {boolean} [options.hasWebGPU=false]
   * @param {number} [options.maxCachedSessions=3]
   * @param {(status: object) => void} [options.onStatusChange]
   */
  constructor({ hasWebGPU = false, maxCachedSessions = 3, onStatusChange = () => {} } = {}) {
    this.hasWebGPU = Boolean(hasWebGPU);
    this.maxCachedSessions = Math.max(1, maxCachedSessions);
    this.onStatusChange = onStatusChange;

    this.checkpoints = [];
    this.activeIndex = -1;
    this.ortModule = null;

    /** @type {Map<string, { session: any, step: number | null, byteLength: number }>} */
    this.sessionCache = new Map();

    /** @type {Map<string, Promise<ArrayBuffer>>} */
    this.inflightBuffers = new Map();

    /** @type {AbortController | null} */
    this.activeAbortController = null;
    this.mountRequestId = 0;
  }

  /**
   * Sets the list of checkpoint descriptors from `demo/manifest.json`.
   * Each item supports: `{ step, tokens_seen, url, onnx_url, path, label }`.
   *
   * @param {Array<object>} checkpoints
   */
  setCheckpoints(checkpoints) {
    this.checkpoints = (Array.isArray(checkpoints) ? checkpoints : []).map((ckpt, idx) => ({
      ...ckpt,
      index: idx,
      step: ckpt.step !== undefined ? Number(ckpt.step) : idx,
      tokens_seen: ckpt.tokens_seen !== undefined ? Number(ckpt.tokens_seen) : 0,
      resolvedUrl: ckpt.url || ckpt.onnx_url || ckpt.path || '',
    }));
  }

  async ensureOrt() {
    if (!this.ortModule) {
      this.ortModule = await import('https://cdn.jsdelivr.net/npm/onnxruntime-web/dist/ort.min.mjs');
    }
    return this.ortModule;
  }

  /**
   * Returns true if a checkpoint URL is already instantiated in the LRU session cache.
   *
   * @param {string} url
   * @returns {boolean}
   */
  isCached(url) {
    return Boolean(url && this.sessionCache.has(url));
  }

  /**
   * Downloads an ONNX file with streaming byte progress and an optional AbortSignal.
   *
   * @param {string} url
   * @param {AbortSignal} [signal]
   * @param {(loadedBytes: number, totalBytes: number) => void} [onProgress]
   * @returns {Promise<ArrayBuffer>}
   */
  async fetchModelBytes(url, signal, onProgress) {
    const res = await fetch(url, { signal, cache: 'force-cache' });
    if (!res.ok) {
      throw new Error(`HTTP ${res.status} (${res.statusText || 'fetch failed'})`);
    }

    const totalBytes = Number(res.headers.get('content-length')) || 0;
    if (!res.body || typeof res.body.getReader !== 'function') {
      const buf = await res.arrayBuffer();
      if (onProgress) onProgress(buf.byteLength, buf.byteLength);
      return buf;
    }

    const reader = res.body.getReader();
    const chunks = [];
    let loadedBytes = 0;

    while (true) {
      const { done, value } = await reader.read();
      if (done) break;
      if (value) {
        chunks.push(value);
        loadedBytes += value.byteLength;
        if (onProgress) onProgress(loadedBytes, totalBytes);
      }
    }

    const combined = new Uint8Array(loadedBytes);
    let offset = 0;
    for (const chunk of chunks) {
      combined.set(chunk, offset);
      offset += chunk.byteLength;
    }
    return combined.buffer;
  }

  /**
   * Selects and mounts the checkpoint at `index` without reloading the page.
   * If already in the LRU cache, swaps in 0ms.
   *
   * @param {number} index
   * @returns {Promise<{ session: any, checkpoint: object, cached: boolean } | null>}
   */
  async selectIndex(index) {
    if (index < 0 || index >= this.checkpoints.length) return null;

    const ckpt = this.checkpoints[index];
    this.activeIndex = index;
    const url = ckpt.resolvedUrl;

    if (!url) {
      this.onStatusChange({
        state: 'unmounted',
        index,
        checkpoint: ckpt,
        cached: false,
        message: 'No ONNX URL in checkpoint entry',
      });
      return { session: null, checkpoint: ckpt, cached: false };
    }

    // Cancel any previous in-flight checkpoint download if the user scrubbed past it
    if (this.activeAbortController) {
      this.activeAbortController.abort();
      this.activeAbortController = null;
    }

    // Fast path: instant 0ms LRU cache hit
    if (this.sessionCache.has(url)) {
      const entry = this.sessionCache.get(url);
      this.sessionCache.delete(url);
      this.sessionCache.set(url, entry);

      this.onStatusChange({
        state: 'mounted',
        index,
        checkpoint: ckpt,
        cached: true,
        byteLength: entry.byteLength,
        session: entry.session,
        message: `Cached (${this.hasWebGPU ? 'WebGPU' : 'WASM'})`,
      });

      this.prefetchNeighbors(index);
      return { session: entry.session, checkpoint: ckpt, cached: true };
    }

    const reqId = ++this.mountRequestId;
    const abortController = new AbortController();
    this.activeAbortController = abortController;

    this.onStatusChange({
      state: 'loading',
      index,
      checkpoint: ckpt,
      cached: false,
      loadedBytes: 0,
      totalBytes: 0,
      message: `Downloading step ${ckpt.step.toLocaleString()}...`,
    });

    try {
      const ort = await this.ensureOrt();
      const buffer = await this.fetchModelBytes(
        url,
        abortController.signal,
        (loadedBytes, totalBytes) => {
          if (reqId !== this.mountRequestId) return;
          const mbLoaded = (loadedBytes / (1024 * 1024)).toFixed(1);
          const pct = totalBytes > 0 ? ` (${Math.round((loadedBytes / totalBytes) * 100)}%)` : '';
          this.onStatusChange({
            state: 'loading',
            index,
            checkpoint: ckpt,
            cached: false,
            loadedBytes,
            totalBytes,
            message: `Loading ${mbLoaded} MB${pct}...`,
          });
        }
      );

      if (reqId !== this.mountRequestId) return null;

      this.onStatusChange({
        state: 'compiling',
        index,
        checkpoint: ckpt,
        cached: false,
        message: `Compiling ONNX (${this.hasWebGPU ? 'WebGPU' : 'WASM'})...`,
      });

      const session = await ort.InferenceSession.create(buffer, {
        executionProviders: this.hasWebGPU ? ['webgpu', 'wasm'] : ['wasm'],
      });

      if (reqId !== this.mountRequestId) return null;

      this.putCache(url, {
        session,
        step: ckpt.step,
        byteLength: buffer.byteLength,
      });

      this.onStatusChange({
        state: 'mounted',
        index,
        checkpoint: ckpt,
        cached: false,
        byteLength: buffer.byteLength,
        session,
        message: `Mounted (${this.hasWebGPU ? 'WebGPU' : 'WASM'})`,
      });

      this.prefetchNeighbors(index);
      return { session, checkpoint: ckpt, cached: false };
    } catch (err) {
      if (err.name === 'AbortError' || reqId !== this.mountRequestId) {
        return null;
      }
      this.onStatusChange({
        state: 'error',
        index,
        checkpoint: ckpt,
        cached: false,
        error: err,
        message: `Mount failed: ${err.message}`,
      });
      throw err;
    }
  }

  /**
   * Mounts an arbitrary custom ONNX URL (from the manual URL bar) into the same LRU cache.
   *
   * @param {string} url
   */
  async mountCustomUrl(url) {
    const cleanUrl = String(url || '').trim();
    if (!cleanUrl) return null;

    const syntheticCkpt = {
      index: -1,
      step: this.checkpoints[this.activeIndex]?.step ?? 0,
      tokens_seen: this.checkpoints[this.activeIndex]?.tokens_seen ?? 0,
      resolvedUrl: cleanUrl,
      url: cleanUrl,
    };

    if (this.sessionCache.has(cleanUrl)) {
      const entry = this.sessionCache.get(cleanUrl);
      this.sessionCache.delete(cleanUrl);
      this.sessionCache.set(cleanUrl, entry);
      this.onStatusChange({
        state: 'mounted',
        index: -1,
        checkpoint: syntheticCkpt,
        cached: true,
        byteLength: entry.byteLength,
        session: entry.session,
        message: `Cached (${this.hasWebGPU ? 'WebGPU' : 'WASM'})`,
      });
      return entry.session;
    }

    const reqId = ++this.mountRequestId;
    const ort = await this.ensureOrt();
    this.onStatusChange({
      state: 'loading',
      index: -1,
      checkpoint: syntheticCkpt,
      cached: false,
      message: 'Loading custom ONNX...',
    });

    const buffer = await this.fetchModelBytes(cleanUrl);
    if (reqId !== this.mountRequestId) return null;

    const session = await ort.InferenceSession.create(buffer, {
      executionProviders: this.hasWebGPU ? ['webgpu', 'wasm'] : ['wasm'],
    });

    this.putCache(cleanUrl, {
      session,
      step: null,
      byteLength: buffer.byteLength,
    });

    this.onStatusChange({
      state: 'mounted',
      index: -1,
      checkpoint: syntheticCkpt,
      cached: false,
      byteLength: buffer.byteLength,
      session,
      message: `Mounted (${this.hasWebGPU ? 'WebGPU' : 'WASM'})`,
    });

    return session;
  }

  putCache(url, entry) {
    if (this.sessionCache.has(url)) {
      this.sessionCache.delete(url);
    } else if (this.sessionCache.size >= this.maxCachedSessions) {
      const oldestKey = this.sessionCache.keys().next().value;
      const evicted = this.sessionCache.get(oldestKey);
      this.sessionCache.delete(oldestKey);
      if (evicted?.session && typeof evicted.session.release === 'function') {
        try {
          evicted.session.release();
        } catch {
          // Ignore release errors on older ort versions
        }
      }
    }
    this.sessionCache.set(url, entry);
  }

  /**
   * Warm the browser HTTP cache for adjacent checkpoints (`index - 1` and `index + 1`)
   * so sliding to the previous/next step is near-instant.
   *
   * @param {number} centerIndex
   */
  prefetchNeighbors(centerIndex) {
    const neighbors = [centerIndex - 1, centerIndex + 1];
    for (const idx of neighbors) {
      const neighbor = this.checkpoints[idx];
      const url = neighbor?.resolvedUrl;
      if (!url || this.sessionCache.has(url) || this.inflightBuffers.has(url)) continue;

      const promise = fetch(url, { cache: 'force-cache' })
        .then((r) => (r.ok ? r.arrayBuffer() : null))
        .catch(() => null)
        .finally(() => {
          this.inflightBuffers.delete(url);
        });

      this.inflightBuffers.set(url, promise);
    }
  }
}
