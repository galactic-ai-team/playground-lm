import test from 'node:test';
import assert from 'node:assert/strict';
import {
  bytesToUnicode,
  createGPT2Tokenizer,
  loadGPT2Tokenizer,
  GPT2_EOT_ID,
  GPT2_EOT_TOKEN,
} from './gpt2_bpe.js';

/**
 * Builds a compact, self-contained fixture containing the exact OpenAI GPT-2
 * base byte vocabulary, real merge ranks, and exact GPT-2 token IDs for known test phrases:
 * - "Hello world" -> [15496, 995]
 * - "Hello, world!" -> [15496, 11, 995, 0]
 * - " don't" -> [836, 470]
 */
function buildRealGPT2Fixture() {
  const { byteEncoder } = bytesToUnicode();
  const vocab = {};

  // Populate all 256 single-byte base tokens so any arbitrary UTF-8 string round-trips
  let nextFallbackId = 60000;
  for (let b = 0; b < 256; b++) {
    const ch = byteEncoder.get(b);
    vocab[ch] = nextFallbackId++;
  }

  // Exact OpenAI GPT-2 token IDs for single-byte punctuation and letters
  vocab['!'] = 0;
  vocab[','] = 11;
  vocab["'"] = 6;
  vocab['H'] = 39;
  vocab['d'] = 67;
  vocab['e'] = 68;
  vocab['l'] = 75;
  vocab['n'] = 77;
  vocab['o'] = 78;
  vocab['r'] = 81;
  vocab['t'] = 83;
  vocab['w'] = 86;
  vocab['Ġ'] = 220;

  // Intermediate and final merged tokens with exact OpenAI GPT-2 IDs
  vocab['ll'] = 297;
  vocab['He'] = 1544;
  vocab['Hell'] = 19097;
  vocab['Hello'] = 15496;

  vocab['Ġw'] = 266;
  vocab['or'] = 273;
  vocab['Ġwor'] = 9855;
  vocab['ld'] = 300;
  vocab['Ġworld'] = 995;

  vocab['Ġd'] = 288;
  vocab['on'] = 261;
  vocab['Ġdon'] = 836;
  vocab["'t"] = 470;

  vocab[GPT2_EOT_TOKEN] = GPT2_EOT_ID;

  // Ordered merges in true rank order
  const mergesText = [
    '#version: 0.2',
    'Ġ t',
    'Ġ a',
    'h e',
    'i n',
    'r e',
    'o n',
    'Ġ w',
    'o r',
    'Ġ d',
    'l l',
    'l d',
    "' t",
    'H e',
    'Ġd on',
    'Ġw or',
    'He ll',
    'Ġwor ld',
    'Hell o',
  ].join('\n');

  return { vocab, mergesText };
}

test('bytesToUnicode maps all 256 byte values bijectively', () => {
  const { byteEncoder, byteDecoder } = bytesToUnicode();
  assert.equal(byteEncoder.size, 256);
  assert.equal(byteDecoder.size, 256);

  // Space (32) maps to 'Ġ' (U+0120) and newline (10) maps to 'Ċ' (U+010A) in GPT-2
  assert.equal(byteEncoder.get(32), 'Ġ');
  assert.equal(byteEncoder.get(10), 'Ċ');

  for (let b = 0; b < 256; b++) {
    const ch = byteEncoder.get(b);
    assert.equal(typeof ch, 'string');
    assert.equal(ch.length, 1);
    assert.equal(byteDecoder.get(ch), b);
  }
});

test('encodes exact GPT-2 IDs for known phrases ("Hello world" -> [15496, 995])', () => {
  const { vocab, mergesText } = buildRealGPT2Fixture();
  const tokenizer = createGPT2Tokenizer(vocab, mergesText);

  assert.deepEqual(tokenizer.encode('Hello world'), [15496, 995]);
  assert.deepEqual(tokenizer.encode('Hello, world!'), [15496, 11, 995, 0]);
  assert.deepEqual(tokenizer.encode("Hello don't world"), [15496, 836, 470, 995]);
});

test('handles <|endoftext|> special token (50256) cleanly', () => {
  const { vocab, mergesText } = buildRealGPT2Fixture();
  const tokenizer = createGPT2Tokenizer(vocab, mergesText);

  const ids = tokenizer.encode(`Hello world${GPT2_EOT_TOKEN}Hello!`);
  assert.deepEqual(ids, [15496, 995, 50256, 15496, 0]);
  assert.equal(tokenizer.decode(ids), `Hello world${GPT2_EOT_TOKEN}Hello!`);
});

test('round-trips ASCII, multi-line whitespace, contractions, and UTF-8 emojis', () => {
  const { vocab, mergesText } = buildRealGPT2Fixture();
  const tokenizer = createGPT2Tokenizer(vocab, mergesText);

  const samples = [
    '',
    'Hello world',
    'Hello, world!\nLine 2   with   spaces.',
    "We're testing don't, I'll, and they've!",
    '🛸 PlaygroundLM @ Crystal Lake (-31.5, 38.5) 🎹✨ — Café & Naïve',
  ];

  for (const sample of samples) {
    const encoded = tokenizer.encode(sample);
    const decoded = tokenizer.decode(encoded);
    assert.equal(decoded, sample);
  }

  // Also supports decoding a single numeric token ID (used by live logit bars in app.js)
  assert.equal(tokenizer.decode(995), ' world');
});

test('loadGPT2Tokenizer fetches vocabUrl and mergesUrl via global fetch', async () => {
  const { vocab, mergesText } = buildRealGPT2Fixture();
  const originalFetch = globalThis.fetch;

  globalThis.fetch = async (url) => {
    if (String(url).includes('vocab')) {
      return {
        ok: true,
        status: 200,
        async json() {
          return vocab;
        },
      };
    }
    if (String(url).includes('merges')) {
      return {
        ok: true,
        status: 200,
        async text() {
          return mergesText;
        },
      };
    }
    return { ok: false, status: 404 };
  };

  try {
    const tok = await loadGPT2Tokenizer('./vocab.json', './merges.txt');
    assert.deepEqual(tok.encode('Hello world'), [15496, 995]);
    assert.equal(tok.decode([15496, 995]), 'Hello world');
  } finally {
    globalThis.fetch = originalFetch;
  }
});
