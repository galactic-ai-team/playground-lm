/**
 * PlaygroundLM chat transcript helpers.
 * Honest bounded multi-turn formatting for a small pretrained/finetuned LM —
 * not an instruction-tuned chatbot.
 */

export const CONTEXT_LENGTH = 1024;
export const DEFAULT_REPLY_RESERVE = 64;

/**
 * @typedef {{role: 'user' | 'assistant' | 'system', content: string, id?: string}} ChatMessage
 */

export function createMessage(role, content) {
  return {
    id: `${role}-${Date.now()}-${Math.random().toString(36).slice(2, 8)}`,
    role,
    content: String(content ?? ''),
  };
}

/**
 * Format messages into a plain prompt the model continues from.
 * Ends with an Assistant: cue so generation continues as the reply.
 */
export function formatChatPrompt(messages) {
  const lines = [];
  for (const msg of messages || []) {
    if (!msg || !msg.content) continue;
    if (msg.role === 'user') lines.push(`User: ${msg.content}`);
    else if (msg.role === 'assistant') lines.push(`Assistant: ${msg.content}`);
    else if (msg.role === 'system') lines.push(msg.content);
  }
  lines.push('Assistant:');
  return lines.join('\n');
}

/**
 * Keep only the newest messages that fit inside the context budget after encoding.
 * Always keeps the latest user message when possible.
 */
export function trimMessagesToBudget(messages, encode, {
  contextLength = CONTEXT_LENGTH,
  reserveNewTokens = DEFAULT_REPLY_RESERVE,
} = {}) {
  const budget = Math.max(8, contextLength - Math.max(1, reserveNewTokens) - 4);
  const list = Array.isArray(messages) ? messages.slice() : [];
  while (list.length > 0) {
    // Prefer transcripts that start on a user turn.
    while (list.length > 1 && list[0].role !== 'user') list.shift();
    const prompt = formatChatPrompt(list);
    const ids = encode(prompt);
    if (ids.length <= budget) {
      return { messages: list, prompt, inputIds: ids, truncated: list.length !== messages.length };
    }
    if (list.length === 1) {
      // Single oversized turn: keep the prompt and rely on generation-time right-slice.
      return { messages: list, prompt, inputIds: ids, truncated: true };
    }
    // Drop oldest turn pair when possible.
    list.shift();
  }
  const prompt = formatChatPrompt([]);
  return { messages: [], prompt, inputIds: encode(prompt), truncated: true };
}

/**
 * Safe DOM text rendering — never assign model output to innerHTML.
 */
export function setTextContent(el, text) {
  if (!el) return;
  el.textContent = String(text ?? '');
}

export function appendText(el, text) {
  if (!el) return '';
  const piece = String(text ?? '');
  el.textContent = `${el.textContent || ''}${piece}`;
  return piece;
}

/**
 * Whether the transcript scroller is already near the bottom.
 */
export function isPinnedToBottom(scroller, thresholdPx = 72) {
  if (!scroller) return true;
  return scroller.scrollHeight - scroller.scrollTop - scroller.clientHeight <= thresholdPx;
}

export function scrollToBottom(scroller) {
  if (!scroller) return;
  scroller.scrollTop = scroller.scrollHeight;
}

/**
 * Keyboard helper for composer: Enter sends, Shift+Enter inserts newline.
 */
export function handleComposerKeydown(event, { onSend, isDisabled } = {}) {
  if (!event || event.key !== 'Enter' || event.shiftKey) return false;
  if (event.isComposing || event.keyCode === 229) return false;
  event.preventDefault();
  if (isDisabled && isDisabled()) return true;
  if (typeof onSend === 'function') onSend();
  return true;
}
