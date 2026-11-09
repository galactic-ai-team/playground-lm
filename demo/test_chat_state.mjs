import test from 'node:test';
import assert from 'node:assert/strict';
import { bytesToUnicode, createGPT2Tokenizer, GPT2_EOT_ID } from './gpt2_bpe.js';
import {
  CONTEXT_LENGTH,
  createMessage,
  formatChatPrompt,
  trimMessagesToBudget,
  findStopSequence,
  normalizeCompletionText,
  setTextContent,
  appendText,
  handleComposerKeydown,
} from './chat_state.js';

function buildStageEBpeFixture() {
  const { byteEncoder } = bytesToUnicode();
  const vocab = {};
  let nextId = 60000;
  for (let b = 0; b < 256; b++) {
    vocab[byteEncoder.get(b)] = nextId++;
  }

  vocab[':'] = 25;
  vocab['\n'] = 198;
  vocab['Ċ'] = 198;
  vocab['Ġ'] = 220;
  vocab['U'] = 52;
  vocab['s'] = 82;
  vocab['e'] = 68;
  vocab['r'] = 81;
  vocab['A'] = 32;
  vocab['i'] = 72;
  vocab['t'] = 83;
  vocab['a'] = 64;
  vocab['n'] = 77;
  vocab['H'] = 39;
  vocab['l'] = 75;
  vocab['o'] = 78;

  vocab['Us'] = 1001;
  vocab['er'] = 1002;
  vocab['User'] = 12982;
  vocab['As'] = 1003;
  vocab['sis'] = 1004;
  vocab['tant'] = 1005;
  vocab['Assis'] = 1006;
  vocab['Assistant'] = 48902;
  vocab['ĠH'] = 1007;
  vocab['ell'] = 1008;
  vocab['ĠHell'] = 1009;
  vocab['ĠHello'] = 18435;
  vocab['<|endoftext|>'] = GPT2_EOT_ID;

  const merges = [
    '#version: 0.2',
    'U s',
    'e r',
    'Us er',
    'A s',
    's i',
    'si s',
    't a',
    'ta n',
    'tan t',
    'As sis',
    'Assis tant',
    'Ġ H',
    'e l',
    'el l',
    'ĠH ell',
    'ĠHell o',
  ].join('\n');

  return createGPT2Tokenizer(vocab, merges);
}

test('formatChatPrompt matches data/build_stage_e_sft.py prompt/completion boundary', () => {
  const tokenizer = buildStageEBpeFixture();

  // Matches tokenize_rows() in data/build_stage_e_sft.py:
  // prompt = f"User: {row['user']}\nAssistant:"
  // completion = f" {row['assistant']}"
  const prompt = formatChatPrompt([createMessage('user', 'Hello')]);
  assert.equal(prompt, 'User: Hello\nAssistant:');
  assert.equal(prompt.endsWith('Assistant:'), true);
  assert.equal(prompt.endsWith('Assistant: '), false);

  const pIds = tokenizer.encode(prompt);
  assert.deepEqual(pIds.slice(-2), [48902, 25]); // ["Assistant", ":"]

  const cIds = [...tokenizer.encode(' Hello'), GPT2_EOT_ID];
  assert.deepEqual(cIds, [18435, 50256]); // ["ĠHello", "<|endoftext|>"]
});

test('formatChatPrompt normalizes assistant replies with or without a leading completion space', () => {
  // Whether assistant reply was stored raw (" 4") or stripped ("4"), multi-turn prompt has single space
  const messagesWithLeadingSpace = [
    createMessage('user', 'What is 2 + 2?'),
    createMessage('assistant', ' 4'),
    createMessage('user', 'What is 3 + 3?'),
  ];
  const messagesTrimmed = [
    createMessage('user', 'What is 2 + 2?'),
    createMessage('assistant', '4'),
    createMessage('user', 'What is 3 + 3?'),
  ];

  const expected = 'User: What is 2 + 2?\nAssistant: 4\nUser: What is 3 + 3?\nAssistant:';
  assert.equal(formatChatPrompt(messagesWithLeadingSpace), expected);
  assert.equal(formatChatPrompt(messagesTrimmed), expected);
});

test('normalizeCompletionText and findStopSequence handle leading space and stop markers', () => {
  assert.equal(normalizeCompletionText(' Hello from PlaygroundLM'), 'Hello from PlaygroundLM');
  assert.equal(normalizeCompletionText('Hello from PlaygroundLM'), 'Hello from PlaygroundLM');
  assert.equal(
    normalizeCompletionText(' 42\nUser: What is next?'),
    '42'
  );
  assert.equal(
    normalizeCompletionText(' OK\nAssistant: Extra turn'),
    'OK'
  );
  assert.equal(
    normalizeCompletionText(' Yes<|endoftext|>trailing'),
    'Yes'
  );

  const stopUser = findStopSequence(' Short answer.\nUser: Another question');
  assert.equal(stopUser.stopped, true);
  assert.equal(stopUser.sequence, '\nUser:');
  assert.equal(stopUser.text, 'Short answer.');

  const stopAssistant = findStopSequence(' Short answer.\nAssistant: Hallucinated');
  assert.equal(stopAssistant.stopped, true);
  assert.equal(stopAssistant.sequence, '\nAssistant:');
  assert.equal(stopAssistant.text, 'Short answer.');
});

test('trimMessagesToBudget drops oldest turns, enforces budget, and preserves User:/Assistant: boundaries on oversized turns', () => {
  const tokenizer = buildStageEBpeFixture();

  const multiTurn = [
    createMessage('user', 'Old question '.repeat(8)),
    createMessage('assistant', 'Old answer '.repeat(8)),
    createMessage('user', 'Hello'),
  ];

  const packed = trimMessagesToBudget(multiTurn, (t) => tokenizer.encode(t), {
    contextLength: 48,
    reserveNewTokens: 16,
  });

  assert.equal(packed.truncated, true);
  assert.equal(packed.messages.length, 1);
  assert.equal(packed.prompt, 'User: Hello\nAssistant:');
  assert.deepEqual(packed.inputIds.slice(-2), [48902, 25]);

  // Single oversized turn still preserves leading "User: " and trailing "\nAssistant:"
  const oversized = [createMessage('user', 'Hello '.repeat(60))];
  const packedOversized = trimMessagesToBudget(oversized, (t) => tokenizer.encode(t), {
    contextLength: 40,
    reserveNewTokens: 12,
  });
  const budget = 40 - 12 - 4; // 24
  assert.equal(packedOversized.truncated, true);
  assert.equal(packedOversized.inputIds.length, budget);
  assert.deepEqual(packedOversized.inputIds.slice(0, 3), [12982, 25, 220]); // "User: "
  assert.deepEqual(packedOversized.inputIds.slice(-3), [198, 48902, 25]); // "\nAssistant:"
});

test('Safe DOM helpers and handleComposerKeydown preserve existing behavior', () => {
  const el = { textContent: '' };
  setTextContent(el, '<b>Safe</b>');
  assert.equal(el.textContent, '<b>Safe</b>');
  appendText(el, ' text');
  assert.equal(el.textContent, '<b>Safe</b> text');
  assert.equal(CONTEXT_LENGTH, 1024);

  let sent = 0;
  const handled = handleComposerKeydown(
    { key: 'Enter', shiftKey: false, isComposing: false, preventDefault() {} },
    { isDisabled: () => false, onSend: () => { sent += 1; } }
  );
  assert.equal(handled, true);
  assert.equal(sent, 1);
});