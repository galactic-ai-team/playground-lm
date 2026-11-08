import test from 'node:test';
import assert from 'node:assert/strict';
import {
  createMessage,
  formatChatPrompt,
  trimMessagesToBudget,
  setTextContent,
  appendText,
  isPinnedToBottom,
  handleComposerKeydown,
} from './chat_state.js';
import { createGPT2Tokenizer } from './gpt2_bpe.js';
import fs from 'node:fs';

const tok = createGPT2Tokenizer(
  JSON.parse(fs.readFileSync(new URL('./vocab.json', import.meta.url))),
  fs.readFileSync(new URL('./merges.txt', import.meta.url), 'utf8')
);

test('formatChatPrompt builds honest User/Assistant transcript ending with Assistant cue', () => {
  const prompt = formatChatPrompt([
    createMessage('user', 'hello'),
    createMessage('assistant', 'hi there'),
    createMessage('user', 'how are you'),
  ]);
  assert.equal(
    prompt,
    'User: hello\nAssistant: hi there\nUser: how are you\nAssistant:'
  );
});

test('trimMessagesToBudget drops oldest turns before exceeding context', () => {
  const msgs = [];
  for (let i = 0; i < 40; i++) {
    msgs.push(createMessage('user', `question number ${i} with some padding words`));
    msgs.push(createMessage('assistant', `answer number ${i} with additional padding words`));
  }
  const result = trimMessagesToBudget(msgs, (t) => tok.encode(t), {
    contextLength: 128,
    reserveNewTokens: 32,
  });
  assert.ok(result.truncated);
  assert.ok(result.messages.length < msgs.length);
  assert.ok(result.inputIds.length <= 128 - 32 - 4);
  const latestUser = [...msgs].reverse().find((m) => m.role === 'user');
  assert.ok(result.messages.some((m) => m.role === 'user' && m.content === latestUser.content));
});

test('setTextContent and appendText never interpret HTML from model text', () => {
  const el = { textContent: '' };
  setTextContent(el, '<img src=x onerror=alert(1)>');
  assert.equal(el.textContent, '<img src=x onerror=alert(1)>');
  appendText(el, '<script>evil()</script>');
  assert.equal(el.textContent, '<img src=x onerror=alert(1)><script>evil()</script>');
});

test('isPinnedToBottom respects threshold', () => {
  assert.equal(isPinnedToBottom({ scrollHeight: 500, scrollTop: 430, clientHeight: 60 }, 20), true);
  assert.equal(isPinnedToBottom({ scrollHeight: 500, scrollTop: 100, clientHeight: 60 }, 20), false);
});

test('Enter sends and Shift+Enter does not', () => {
  let sent = 0;
  const sendEvent = { key: 'Enter', shiftKey: false, preventDefault() {}, isComposing: false };
  assert.equal(handleComposerKeydown(sendEvent, { onSend: () => { sent += 1; } }), true);
  assert.equal(sent, 1);
  const newlineEvent = { key: 'Enter', shiftKey: true, preventDefault() {}, isComposing: false };
  assert.equal(handleComposerKeydown(newlineEvent, { onSend: () => { sent += 1; } }), false);
  assert.equal(sent, 1);
});
