/**
 * Exact GPT-2 Byte-Level BPE Tokenizer (Dependency-Free ES Module)
 *
 * Implements OpenAI's original `encoder.py` byte-level BPE:
 * - Reversible 256-byte to Unicode character bijection (`bytesToUnicode`)
 * - GPT-2 pre-tokenization regex with Unicode property escapes (\p{L}, \p{N})
 * - Iterative lowest-rank bigram merge loop with per-token BPE cache
 * - Full UTF-8 byte decoding (`decode(ids)` accepts either `number` or `number[]`)
 * - EOT special token `<|endoftext|>` (50256) handling
 */

export const GPT2_EOT_TOKEN = '<|endoftext|>';
export const GPT2_EOT_ID = 50256;

/**
 * OpenAI's GPT-2 pre-tokenization pattern translated to ECMAScript Unicode regex (`gu`).
 */
export const GPT2_SPLIT_REGEX =
  /'s|'t|'re|'ve|'m|'ll|'d| ?\p{L}+| ?\p{N}+| ?[^\s\p{L}\p{N}]+|\s+(?!\S)|\s+/gu;

/**
 * Builds the exact 256-entry reversible mapping between raw UTF-8 bytes (0..255)
 * and printable Unicode strings used by GPT-2's `bytes_to_unicode()`.
 *
 * @returns {{ byteEncoder: Map<number, string>, byteDecoder: Map<string, number> }}
 */
export function bytesToUnicode() {
  const bs = [];
  // Printable ASCII: '!' (33) .. '~' (126)
  for (let b = 33; b <= 126; b++) bs.push(b);
  // Latin-1 supplement printable block 1: '¡' (161) .. '¬' (172)
  for (let b = 161; b <= 172; b++) bs.push(b);
  // Latin-1 supplement printable block 2: '®' (174) .. 'ÿ' (255)
  for (let b = 174; b <= 255; b++) bs.push(b);

  const cs = bs.slice();
  const present = new Set(bs);
  let n = 0;

  for (let b = 0; b < 256; b++) {
    if (!present.has(b)) {
      bs.push(b);
      cs.push(256 + n);
      n += 1;
    }
  }

  const byteEncoder = new Map();
  const byteDecoder = new Map();
  for (let i = 0; i < bs.length; i++) {
    const ch = String.fromCharCode(cs[i]);
    byteEncoder.set(bs[i], ch);
    byteDecoder.set(ch, bs[i]);
  }

  return { byteEncoder, byteDecoder };
}

/**
 * Returns all adjacent symbol pairs in `word` as `"first\0second"` keys.
 *
 * @param {string[]} word
 * @returns {Set<string>}
 */
function getPairs(word) {
  const pairs = new Set();
  for (let i = 0; i < word.length - 1; i++) {
    pairs.add(`${word[i]}\0${word[i + 1]}`);
  }
  return pairs;
}

/**
 * Parses `merges.txt` content (or an array of merge lines) into a rank Map.
 * Automatically skips `#version` headers and empty lines.
 *
 * @param {string | string[]} mergesInput
 * @returns {Map<string, number>}
 */
export function parseMerges(mergesInput) {
  const lines = Array.isArray(mergesInput)
    ? mergesInput
    : String(mergesInput).split(/\r?\n/);

  const bpeRanks = new Map();
  let rank = 0;

  for (const rawLine of lines) {
    const line = rawLine.trim();
    if (!line || line.startsWith('#version')) continue;
    const parts = line.split(/\s+/);
    if (parts.length < 2) continue;
    const key = `${parts[0]}\0${parts[1]}`;
    if (!bpeRanks.has(key)) {
      bpeRanks.set(key, rank);
      rank += 1;
    }
  }

  return bpeRanks;
}

/**
 * Creates a synchronous GPT-2 BPE tokenizer instance from in-memory `vocab` and `merges`.
 *
 * @param {Record<string, number>} vocabObj - Mapping of BPE token string -> integer ID
 * @param {string | string[] | Map<string, number>} mergesInput - Contents of `merges.txt` or parsed rank Map
 */
export function createGPT2Tokenizer(vocabObj, mergesInput) {
  const { byteEncoder, byteDecoder } = bytesToUnicode();
  const encoder = new Map();
  const decoder = new Map();

  for (const [tokenStr, idVal] of Object.entries(vocabObj)) {
    const id = Number(idVal);
    encoder.set(tokenStr, id);
    decoder.set(id, tokenStr);
  }

  if (!encoder.has(GPT2_EOT_TOKEN)) {
    encoder.set(GPT2_EOT_TOKEN, GPT2_EOT_ID);
    decoder.set(GPT2_EOT_ID, GPT2_EOT_TOKEN);
  }

  const bpeRanks =
    mergesInput instanceof Map ? mergesInput : parseMerges(mergesInput);

  const cache = new Map();
  const utf8Encoder = new TextEncoder();
  const utf8Decoder = new TextDecoder('utf-8', { fatal: false });

  /**
   * Runs the iterative GPT-2 BPE merge loop on a single byte-mapped token string.
   *
   * @param {string} token
   * @returns {string} Space-separated BPE subword pieces
   */
  function bpe(token) {
    if (cache.has(token)) {
      return cache.get(token);
    }

    let word = Array.from(token);
    if (word.length <= 1) {
      cache.set(token, token);
      return token;
    }

    let pairs = getPairs(word);

    while (pairs.size > 0) {
      let bestPair = null;
      let minRank = Infinity;

      for (const pairKey of pairs) {
        const rank = bpeRanks.get(pairKey);
        if (rank !== undefined && rank < minRank) {
          minRank = rank;
          bestPair = pairKey;
        }
      }

      if (bestPair === null) {
        break;
      }

      const sepIdx = bestPair.indexOf('\0');
      const first = bestPair.slice(0, sepIdx);
      const second = bestPair.slice(sepIdx + 1);
      const newWord = [];
      let i = 0;

      while (i < word.length) {
        const j = word.indexOf(first, i);
        if (j === -1) {
          for (let k = i; k < word.length; k++) newWord.push(word[k]);
          break;
        }
        for (let k = i; k < j; k++) newWord.push(word[k]);
        i = j;

        if (word[i] === first && i < word.length - 1 && word[i + 1] === second) {
          newWord.push(first + second);
          i += 2;
        } else {
          newWord.push(word[i]);
          i += 1;
        }
      }

      word = newWord;
      if (word.length === 1) {
        break;
      }
      pairs = getPairs(word);
    }

    const joined = word.join(' ');
    cache.set(token, joined);
    return joined;
  }

  /**
   * Encodes a plain string without `<|endoftext|>` splitting.
   *
   * @param {string} segment
   * @returns {number[]}
   */
  function encodeOrdinary(segment) {
    if (!segment) return [];
    const bpeTokens = [];
    const matches = segment.match(GPT2_SPLIT_REGEX);
    if (!matches) return [];

    for (const match of matches) {
      const utf8Bytes = utf8Encoder.encode(match);
      let mapped = '';
      for (let i = 0; i < utf8Bytes.length; i++) {
        mapped += byteEncoder.get(utf8Bytes[i]);
      }

      const pieces = bpe(mapped).split(' ');
      for (let i = 0; i < pieces.length; i++) {
        const id = encoder.get(pieces[i]);
        if (id !== undefined) {
          bpeTokens.push(id);
        }
      }
    }

    return bpeTokens;
  }

  /**
   * Encodes arbitrary UTF-8 text into GPT-2 token IDs, preserving `<|endoftext|>` (50256).
   *
   * @param {string} text
   * @returns {number[]}
   */
  function encode(text) {
    if (!text) return [];
    const str = String(text);
    if (!str.includes(GPT2_EOT_TOKEN)) {
      return encodeOrdinary(str);
    }

    const parts = str.split(GPT2_EOT_TOKEN);
    const ids = [];
    const eotId = encoder.get(GPT2_EOT_TOKEN) ?? GPT2_EOT_ID;

    for (let i = 0; i < parts.length; i++) {
      if (parts[i]) {
        const subIds = encodeOrdinary(parts[i]);
        for (let j = 0; j < subIds.length; j++) ids.push(subIds[j]);
      }
      if (i < parts.length - 1) {
        ids.push(eotId);
      }
    }

    return ids;
  }

  /**
   * Decodes an array of token IDs (or a single token ID number) back into a UTF-8 string.
   *
   * @param {number[] | TypedArray | number} ids
   * @returns {string}
   */
  function decode(ids) {
    const list = typeof ids === 'number' ? [ids] : Array.from(ids || []);
    if (list.length === 0) return '';

    let text = '';
    for (let i = 0; i < list.length; i++) {
      const tok = decoder.get(Number(list[i]));
      if (tok !== undefined) {
        text += tok;
      }
    }

    // Handle <|endoftext|> cleanly while converting byte-level characters back to UTF-8 bytes
    if (text.includes(GPT2_EOT_TOKEN)) {
      return text
        .split(GPT2_EOT_TOKEN)
        .map((chunk) => decodeByteString(chunk, byteDecoder, utf8Decoder))
        .join(GPT2_EOT_TOKEN);
    }

    return decodeByteString(text, byteDecoder, utf8Decoder);
  }

  return {
    vocabSize: encoder.size,
    eotId: encoder.get(GPT2_EOT_TOKEN) ?? GPT2_EOT_ID,
    encode,
    decode,
    bpe,
  };
}

function decodeByteString(chunk, byteDecoder, utf8Decoder) {
  if (!chunk) return '';
  const bytes = new Uint8Array(chunk.length);
  let writeIdx = 0;
  for (let i = 0; i < chunk.length; i++) {
    const b = byteDecoder.get(chunk[i]);
    if (b !== undefined) {
      bytes[writeIdx++] = b;
    }
  }
  return utf8Decoder.decode(bytes.subarray(0, writeIdx));
}

/**
 * Fetches standard GPT-2 `vocab.json` (`encoder.json`) and `merges.txt` (or a unified
 * Hugging Face `tokenizer.json` if `mergesUrl` is omitted or embedded) and returns
 * `{ encode(text) -> number[], decode(ids) -> string }`.
 *
 * @param {string} [vocabUrl='./vocab.json']
 * @param {string} [mergesUrl='./merges.txt']
 */
export async function loadGPT2Tokenizer(
  vocabUrl = './vocab.json',
  mergesUrl = './merges.txt'
) {
  const vocabRes = await fetch(vocabUrl, { cache: 'no-store' });
  if (!vocabRes.ok) {
    throw new Error(`Failed to load GPT-2 vocab from ${vocabUrl}: HTTP ${vocabRes.status}`);
  }
  const vocabData = await vocabRes.json();

  // Support both flat `encoder.json`/`vocab.json` and Hugging Face `tokenizer.json`
  const vocabObj = vocabData.model?.vocab || vocabData.vocab || vocabData;
  let mergesInput = vocabData.model?.merges || null;

  if (!mergesInput) {
    const mergesRes = await fetch(mergesUrl, { cache: 'no-store' });
    if (!mergesRes.ok) {
      throw new Error(`Failed to load GPT-2 merges from ${mergesUrl}: HTTP ${mergesRes.status}`);
    }
    mergesInput = await mergesRes.text();
  }

  return createGPT2Tokenizer(vocabObj, mergesInput);
}
