/**
 * PlaygroundLM chat transcript helpers.
 * Honest bounded multi-turn formatting for a small pretrained/finetuned LM —
 * not an instruction-tuned chatbot.
 */

export const CONTEXT_LENGTH = 1024;
export const DEFAULT_REPLY_RESERVE = 64;
export const STOP_SEQUENCES = Object.freeze([
  '<|endoftext|>',
  '\nUser:',
  '\nAssistant:',
]);

const ASSISTANT_CUE = 'Assistant:';

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
 * Strips the single leading space that GPT-2 BPE emits after "Assistant:" (since Stage E
 * completions are tokenized as `" " + assistant`), plus any trailing stop marker.
 */
export function normalizeCompletionText(rawText, { trimEnd = true } = {}) {
  let text = String(rawText ?? '').replace(/\r\n?/g, '\n');
  const stop = findStopSequence(text);
  if (stop.stopped) {
    text = text.slice(0, stop.index);
  }
  if (text.startsWith(' ')) {
    text = text.slice(1);
  }
  return trimEnd ? text.replace(/\s+$/, '') : text;
}

/**
 * Detects whether accumulated completion text contains a turn-boundary stop sequence
 * (`<|endoftext|>`, `\nUser:`, or `\nAssistant:`).
 */
export function findStopSequence(text, stopSequences = STOP_SEQUENCES) {
  const str = String(text ?? '');
  let earliestIndex = -1;
  let matched = null;
  for (const seq of stopSequences) {
    if (!seq) continue;
    const idx = str.indexOf(seq);
    if (idx !== -1 && (earliestIndex === -1 || idx < earliestIndex)) {
      earliestIndex = idx;
      matched = seq;
    }
  }
  if (earliestIndex !== -1) {
    let clean = str.slice(0, earliestIndex).replace(/\r\n?/g, '\n');
    if (clean.startsWith(' ')) clean = clean.slice(1);
    clean = clean.replace(/\s+$/, '');
    return {
      stopped: true,
      index: earliestIndex,
      sequence: matched,
      text: clean,
    };
  }
  let clean = str.replace(/\r\n?/g, '\n');
  if (clean.startsWith(' ')) clean = clean.slice(1);
  return {
    stopped: false,
    index: -1,
    sequence: null,
    text: clean,
  };
}

/**
 * Format messages into a plain prompt the model continues from.
 * Ends with an exact "Assistant:" cue (no trailing space) so GPT-2 BPE predicts
 * the leading-space completion token ("Ġ...") trained in data/build_stage_e_sft.py.
 */
export function formatChatPrompt(messages) {
  const lines = [];
  for (const msg of messages || []) {
    if (!msg || msg.content === undefined || msg.content === null) continue;
    const raw = String(msg.content).replace(/\r\n?/g, '\n');
    if (!raw.trim()) continue;
    if (msg.role === 'user') {
      lines.push(`User: ${raw.trim()}`);
    } else if (msg.role === 'assistant') {
      const cleanAssistant = normalizeCompletionText(raw, { trimEnd: true });
      if (cleanAssistant) lines.push(`Assistant: ${cleanAssistant}`);
    } else if (msg.role === 'system') {
      lines.push(raw.trim());
    }
  }
  lines.push(ASSISTANT_CUE);
  return lines.join('\n');
}

/**
 * Keep only the newest messages that fit inside the context budget after encoding.
 * Always keeps the latest user message and preserves the `User: ... \nAssistant:` boundary.
 */
export function trimMessagesToBudget(messages, encode, {
  contextLength = CONTEXT_LENGTH,
  reserveNewTokens = DEFAULT_REPLY_RESERVE,
  maxTurns = null,
} = {}) {
  const budget = Math.max(8, contextLength - Math.max(1, reserveNewTokens) - 4);
  let list = (Array.isArray(messages) ? messages : []).filter(
    (m) => m && typeof m.content === 'string' && m.content.trim().length > 0
  );
  if (typeof maxTurns === 'number' && maxTurns > 0) {
    let userCount = 0;
    let cutIndex = 0;
    for (let i = list.length - 1; i >= 0; i--) {
      if (list[i].role === 'user') {
        userCount += 1;
        if (userCount === maxTurns) {
          cutIndex = i;
          break;
        }
      }
    }
    if (cutIndex > 0) list = list.slice(cutIndex);
  }
  const initialLength = Array.isArray(messages) ? messages.length : 0;
  while (list.length > 0) {
    // Prefer transcripts that start on a user turn.
    while (list.length > 1 && list[0].role !== 'user') list.shift();
    const prompt = formatChatPrompt(list);
    const ids = encode(prompt);
    if (ids.length <= budget) {
      return { messages: list, prompt, inputIds: ids, truncated: list.length !== initialLength };
    }
    if (list.length === 1) {
      // Single oversized turn: preserve leading `User: ` and trailing `\nAssistant:` tokens.
      const prefixIds = encode('User: ');
      const suffixIds = encode(`\n${ASSISTANT_CUE}`);
      let trimmedIds = ids;
      if (prefixIds.length + suffixIds.length < budget && ids.length > budget) {
        const bodyBudget = budget - prefixIds.length - suffixIds.length;
        const bodyIds = ids.slice(
          ids.length - suffixIds.length - bodyBudget,
          ids.length - suffixIds.length
        );
        trimmedIds = [...prefixIds, ...bodyIds, ...suffixIds];
      } else if (ids.length > budget) {
        trimmedIds = ids.slice(-budget);
      }
      return { messages: list, prompt, inputIds: trimmedIds, truncated: true };
    }
    // Drop oldest turn pair when possible.
    list.shift();
    while (list.length > 1 && list[0].role !== 'user') list.shift();
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