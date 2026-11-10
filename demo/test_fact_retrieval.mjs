// Stage G: BM25 fact retrieval + Facts/Question prompt injection (tests by Lumi).
import test from 'node:test';
import assert from 'node:assert/strict';
import { readFileSync } from 'node:fs';
import { buildIndex, retrieve, tokenize } from './fact_retrieval.js';
import {
  buildFactsQuestion, createMessage, formatChatPrompt, trimMessagesToBudget, FACTS_TOKEN_BUDGET,
} from './chat_state.js';

const SHEET = readFileSync(new URL('../data/facts/fact_sheet.jsonl', import.meta.url), 'utf8')
  .split('\n').filter(Boolean).map((l) => JSON.parse(l));
const TINY = [
  { id: 'a', text: 'Paris is the capital of France.', keywords: ['paris', 'capital', 'france'], aliases: ['Paris'] },
  { id: 'b', text: 'Berlin is the capital of Germany.', keywords: ['berlin', 'capital', 'germany'], aliases: ['Berlin'] },
  { id: 'c', text: 'Bees make honey.', keywords: ['bees', 'honey'], aliases: ['honey'] },
  { id: 'd', text: 'Mars is called the red planet.', keywords: ['mars', 'red', 'planet'], aliases: ['Mars'] },
  { id: 'e', text: 'Water boils at 100 degrees Celsius at sea level.', keywords: ['water', 'boils'], aliases: ['100'] },
];
const fakeEncode = (s) => s.split(/\s+/).filter(Boolean); // 1 token per word, deterministic

test('tokenize lowercases, splits on non-alphanumerics, drops stopwords', () => {
  assert.deepEqual(tokenize('What is the Capital of FRANCE?'), ['what', 'capital', 'of', 'france']);
  assert.deepEqual(tokenize(''), []);
  assert.deepEqual(tokenize(null), []);
});

test('retrieve ranks the matching fact first and is deterministic', () => {
  const idx = buildIndex(TINY);
  const r1 = retrieve(idx, 'What is the capital of France?', { k: 3 });
  assert.equal(r1[0].id, 'a');
  assert.ok(r1.length <= 3);
  assert.deepEqual(retrieve(idx, 'What is the capital of France?', { k: 3 }), r1);
  for (let i = 1; i < r1.length; i++) assert.ok(r1[i - 1].score >= r1[i].score);
});

test('retrieve never returns non-matching facts (zero score), even with minScore 0', () => {
  const idx = buildIndex(TINY);
  assert.deepEqual(retrieve(idx, 'zzzz qqqq', { k: 3, minScore: 0 }), []);
  assert.deepEqual(retrieve(idx, 'the and of', { k: 3 }).filter((r) => r.score <= 0), []);
  assert.deepEqual(retrieve(buildIndex([]), 'capital'), []);
});

test('minScore threshold filters weak matches; ties break by id', () => {
  const idx = buildIndex(TINY);
  const strong = retrieve(idx, 'capital', { k: 5 });
  assert.deepEqual(strong.map((r) => r.id), ['a', 'b']); // equal scores -> id order
  assert.deepEqual(retrieve(idx, 'capital', { k: 5, minScore: 1e9 }), []);
});

test('buildIndex gives deterministic ids when a fact has none', () => {
  const idx = buildIndex([{ text: 'Bees make honey.' }, { text: 'Cats purr.' }]);
  assert.deepEqual(idx.documents.map((d) => d.id), ['fact-000000', 'fact-000001']);
});

test('real fact sheet: capital questions retrieve the right fact', () => {
  const idx = buildIndex(SHEET);
  for (const [q, alias] of [['What is the capital of France?', 'Paris'],
    ['capital of Japan', 'Tokyo'], ['What is the chemical symbol for gold?', 'Au']]) {
    const top = retrieve(idx, q, { k: 3 });
    assert.ok(top.length > 0, q);
    assert.ok(top.some((r) => r.text.includes(alias)), `${q} -> ${top.map((r) => r.text)}`);
  }
});

test('Facts/Question line matches data/build_stage_g_grounded.py facts_prompt byte-for-byte', () => {
  const { content, used } = buildFactsQuestion('  What is the\ncapital of France? ',
    ['Paris is the capital of France.', 'Bees  make honey.', { text: 'Mars is called the red planet.' }]);
  assert.equal(used, 3);
  // Same literal is asserted in data/test_stage_g_demo_parity.py against facts_prompt().
  assert.equal(content, 'Facts: [1] Paris is the capital of France. [2] Bees make honey. '
    + '[3] Mars is called the red planet. Question: What is the capital of France?');
});

test('facts budget drops lowest-ranked facts first and never mutates input', () => {
  const facts = TINY.slice(0, 3);
  const before = JSON.stringify(facts);
  const { content, used } = buildFactsQuestion('Q?', facts, { encode: fakeEncode, budget: 15 });
  assert.equal(used, 2);
  assert.ok(content.includes('[2] Berlin') && !content.includes('[3]'));
  assert.equal(JSON.stringify(facts), before);
  const none = buildFactsQuestion('Q?', facts, { encode: fakeEncode, budget: 1 });
  assert.deepEqual(none, { content: 'Q?', used: 0 });
  assert.equal(FACTS_TOKEN_BUDGET, 160);
});

test('without facts the prompt is byte-identical to v0.5 Chat', () => {
  const msgs = [createMessage('user', 'Hi'), createMessage('assistant', ' Hello!'), createMessage('user', 'Capital of France?')];
  const base = formatChatPrompt(msgs);
  assert.equal(formatChatPrompt(msgs, {}), base);
  assert.equal(formatChatPrompt(msgs, { facts: [] }), base);
  assert.equal(base, 'User: Hi\nAssistant: Hello!\nUser: Capital of France?\nAssistant:');
});

test('facts go only into the newest user turn; older turns unchanged', () => {
  const msgs = [createMessage('user', 'Hi'), createMessage('assistant', 'Hello!'), createMessage('user', 'Capital of France?')];
  const p = formatChatPrompt(msgs, { facts: [TINY[0]] });
  assert.equal(p, 'User: Hi\nAssistant: Hello!\nUser: Facts: [1] Paris is the capital of France. '
    + 'Question: Capital of France?\nAssistant:');
  assert.equal(msgs[2].content, 'Capital of France?'); // visible transcript untouched
});

test('trimMessagesToBudget threads facts through and still fits the context', () => {
  const msgs = [createMessage('user', 'Capital of France?')];
  const packed = trimMessagesToBudget(msgs, fakeEncode, { contextLength: 1024, facts: TINY.slice(0, 3) });
  assert.ok(packed.prompt.startsWith('User: Facts: [1] Paris'));
  assert.ok(packed.inputIds.length <= 1024);
  const plain = trimMessagesToBudget(msgs, fakeEncode, { contextLength: 1024 });
  assert.equal(plain.prompt, 'User: Capital of France?\nAssistant:');
});
