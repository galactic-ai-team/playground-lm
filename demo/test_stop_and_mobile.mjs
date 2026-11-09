import test from 'node:test';
import assert from 'node:assert/strict';
import fs from 'node:fs';
import path from 'node:path';
import { fileURLToPath } from 'node:url';
import {
  requestStopGeneration,
  handleStopHotkey,
} from './chat_state.js';

const __dirname = path.dirname(fileURLToPath(import.meta.url));

test('requestStopGeneration sets stopRequested and reports Stopping', () => {
  const state = { stopRequested: false };
  let status = null;
  let disabled = false;
  const result = requestStopGeneration(state, {
    setStatus: (text, opts) => { status = { text, opts }; },
    disableStopButton: () => { disabled = true; },
  });
  assert.equal(state.stopRequested, true);
  assert.equal(result.stopped, true);
  assert.equal(result.message, 'Stopping generation…');
  assert.equal(status.text, 'Stopping generation…');
  assert.equal(status.opts.ready, true);
  assert.equal(disabled, true);
});

test('requestStopGeneration is idempotent and safe on bad state', () => {
  assert.equal(requestStopGeneration(null).stopped, false);
  const state = { stopRequested: true };
  const again = requestStopGeneration(state);
  assert.equal(again.stopped, true);
  assert.equal(state.stopRequested, true);
});

test('handleStopHotkey Escape aborts only while generating', () => {
  let calls = 0;
  const onStop = () => { calls += 1; };
  assert.equal(handleStopHotkey({ key: 'Escape' }, { isGenerating: false, onStop }), false);
  assert.equal(calls, 0);
  assert.equal(handleStopHotkey({ key: 'Escape' }, { isGenerating: true, onStop }), true);
  assert.equal(calls, 1);
  assert.equal(handleStopHotkey({ key: 'Enter' }, { isGenerating: true, onStop }), false);
  assert.equal(calls, 1);
});

test('style.css enforces ≥44px tap targets and ≤640px mobile layout', () => {
  const css = fs.readFileSync(path.join(__dirname, 'style.css'), 'utf8');
  assert.match(css, /min-height:\s*44px/);
  assert.match(css, /min-width:\s*44px/);
  assert.match(css, /@media\s*\(max-width:\s*640px\)/);
  // Send/Stop share a two-column grid on narrow screens
  const mobile = css.split('@media (max-width: 640px)')[1] || '';
  assert.match(mobile, /\.btn-send,\s*\.btn-stop/);
  assert.match(mobile, /grid-template-columns:\s*1fr 1fr/);
});

test('app.js wires Stop click and Escape to cooperative abort', () => {
  const app = fs.readFileSync(path.join(__dirname, 'app.js'), 'utf8');
  assert.match(app, /requestStopGeneration/);
  assert.match(app, /handleStopHotkey/);
  assert.match(app, /function triggerStop/);
  assert.match(app, /btnStop\.addEventListener\('click'/);
  assert.match(app, /document\.addEventListener\('keydown'/);
  const chat = fs.readFileSync(path.join(__dirname, 'chat_state.js'), 'utf8');
  assert.match(chat, /Stopping generation…/);
});
