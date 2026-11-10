/**
 * BM25 Fact Retrieval for PlaygroundLM Stage G.
 * Zero external dependencies.
 */

const STOPWORDS = new Set([
  "a", "an", "the", "and", "or", "but", "is", "are", "was", "were",
  "in", "on", "at", "to", "for", "with", "by", "about", "against",
  "between", "into", "through", "during", "before", "after", "above",
  "below", "from", "up", "down", "out", "off", "over", "under", "again",
  "further", "then", "once", "here", "there", "when", "where", "why",
  "how", "all", "any", "both", "each", "few", "more", "most", "other",
  "some", "such", "no", "nor", "not", "only", "own", "same", "so",
  "than", "too", "very", "s", "t", "can", "will", "just", "don", "should", "now"
]);

export function tokenize(text) {
  if (!text) return [];
  return String(text)
    .toLowerCase()
    .split(/[^a-z0-9]+/)
    .filter(t => t.length > 0 && !STOPWORDS.has(t));
}

export function buildIndex(facts) {
  const documents = (Array.isArray(facts) ? facts : []).map(f => {
    const rawText = f.text || "";
    const kw = Array.isArray(f.keywords) ? f.keywords.join(" ") : "";
    const al = Array.isArray(f.aliases) ? f.aliases.join(" ") : "";
    const combined = `${rawText} ${kw} ${al}`;
    const tokens = tokenize(combined);
    return {
      id: f.id || String(Math.random()),
      text: rawText,
      tokens,
      length: tokens.length
    };
  });

  const N = documents.length;
  let totalLength = 0;
  const docFreqs = new Map();

  for (const doc of documents) {
    totalLength += doc.length;
    const uniqueTokens = new Set(doc.tokens);
    for (const t of uniqueTokens) {
      docFreqs.set(t, (docFreqs.get(t) || 0) + 1);
    }
  }

  const avgdl = N > 0 ? totalLength / N : 0;
  return { documents, N, avgdl, docFreqs };
}

export function retrieve(index, query, { k = 3, minScore = 0.0 } = {}) {
  if (!index || !index.documents || index.documents.length === 0) return [];
  const queryTokens = tokenize(query);
  if (queryTokens.length === 0) return [];

  const k1 = 1.2;
  const b = 0.75;
  const N = index.N;
  const avgdl = index.avgdl;

  const scores = index.documents.map(doc => {
    let score = 0;
    const tf = new Map();
    for (const t of doc.tokens) {
      tf.set(t, (tf.get(t) || 0) + 1);
    }

    for (const qt of queryTokens) {
      const freq = tf.get(qt) || 0;
      if (freq === 0) continue;

      const df = index.docFreqs.get(qt) || 0;
      const idf = Math.log(1 + (N - df + 0.5) / (df + 0.5));
      const numerator = freq * (k1 + 1);
      const denominator = freq + k1 * (1 - b + b * (doc.length / (avgdl || 1)));
      score += idf * (numerator / denominator);
    }

    return { id: doc.id, text: doc.text, score };
  });

  scores.sort((a, b) => {
    if (Math.abs(b.score - a.score) > 1e-6) {
      return b.score - a.score;
    }
    return a.id.localeCompare(b.id);
  });

  return scores
    .filter(s => s.score >= minScore)
    .slice(0, k)
    .map(s => ({ id: s.id, text: s.text, score: s.score }));
}