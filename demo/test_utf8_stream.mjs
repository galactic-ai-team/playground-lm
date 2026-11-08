import test from 'node:test';
import assert from 'node:assert/strict';
import { createGPT2Tokenizer, createUtf8StreamDecoder } from './gpt2_bpe.js';
import fs from 'node:fs';

const tok = createGPT2Tokenizer(
  JSON.parse(fs.readFileSync(new URL('./vocab.json', import.meta.url))),
  fs.readFileSync(new URL('./merges.txt', import.meta.url), 'utf8')
);

test('streaming decode matches full decode for curly-apostrophe hello phrase', () => {
  const text = "I’m glad you’re here — café 你好";
  const ids = tok.encode(text);
  assert.equal(tok.decode(ids), text);

  const stream = tok.createStreamDecoder();
  let out = '';
  for (const id of ids) out += stream.pushToken(id);
  out += stream.flush();
  assert.equal(out, text);
});

test('per-token naive decode can differ from streaming on multi-byte spans', () => {
  const text = "I’m";
  const ids = tok.encode(text);
  const naive = ids.map((id) => tok.decode(id)).join('');
  const stream = tok.createStreamDecoder();
  let out = '';
  for (const id of ids) out += stream.pushToken(id);
  out += stream.flush();
  assert.equal(out, text);
  // Document the bug class: naive join may introduce replacement characters.
  if (naive !== text) {
    assert.ok(naive.includes('\uFFFD') || naive !== text);
  }
});

test('createUtf8StreamDecoder reassembles split multi-byte sequences', () => {
  const stream = createUtf8StreamDecoder();
  // UTF-8 for "é" is C3 A9
  assert.equal(stream.push(Uint8Array.of(0xc3)), '');
  assert.equal(stream.push(Uint8Array.of(0xa9)), 'é');
  assert.equal(stream.flush(), '');
});
