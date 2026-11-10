"""Python twin of demo/fact_retrieval.js (BM25, k1=1.2, b=0.75).

Used by the Stage G pipeline scorer so offline evaluation retrieves exactly
the facts the demo would inject. Parity with the JS module is tested in
tests/test_fact_retrieval_parity.py (Lumi glue).
"""
from __future__ import annotations

import functools
import hashlib
import json
import math
import re
from pathlib import Path

STOPWORDS = frozenset("""
a an the and or but is are was were in on at to for with by about against
between into through during before after above below from up down out off over
under again further then once here there when where why how all any both each
few more most other some such no nor not only own same so than too very s t
can will just don should now
""".split())
_SPLIT = re.compile(r"[^a-z0-9]+")


def tokenize(text) -> list[str]:
    if not text:
        return []
    return [t for t in _SPLIT.split(str(text).lower()) if t and t not in STOPWORDS]


def build_index(facts: list[dict]) -> dict:
    docs = []
    for i, f in enumerate(facts):
        raw = f.get("text") or ""
        kw = " ".join(f["keywords"]) if isinstance(f.get("keywords"), list) else ""
        al = " ".join(f["aliases"]) if isinstance(f.get("aliases"), list) else ""
        toks = tokenize(f"{raw} {kw} {al}")
        docs.append({"id": str(f["id"]) if f.get("id") else f"fact-{i:06d}",
                     "text": raw, "tokens": toks, "length": len(toks)})
    df: dict[str, int] = {}
    for d in docs:
        for t in set(d["tokens"]):
            df[t] = df.get(t, 0) + 1
    n = len(docs)
    avgdl = sum(d["length"] for d in docs) / n if n else 0
    return {"documents": docs, "N": n, "avgdl": avgdl, "docFreqs": df}


def _cmp(a, b) -> int:
    if abs(b["score"] - a["score"]) > 1e-6:
        return -1 if b["score"] < a["score"] else 1
    return (a["id"] > b["id"]) - (a["id"] < b["id"])


def retrieve(index: dict, query: str, k: int = 3, min_score: float = 0.0) -> list[dict]:
    if not index or not index["documents"]:
        return []
    q = tokenize(query)
    if not q:
        return []
    k1, b, n, avgdl = 1.2, 0.75, index["N"], index["avgdl"]
    out = []
    for d in index["documents"]:
        tf: dict[str, int] = {}
        for t in d["tokens"]:
            tf[t] = tf.get(t, 0) + 1
        s = 0.0
        for qt in q:
            f = tf.get(qt, 0)
            if not f:
                continue
            dfq = index["docFreqs"].get(qt, 0)
            idf = math.log(1 + (n - dfq + 0.5) / (dfq + 0.5))
            s += idf * (f * (k1 + 1)) / (f + k1 * (1 - b + b * (d["length"] / (avgdl or 1))))
        out.append({"id": d["id"], "text": d["text"], "score": s})
    out.sort(key=functools.cmp_to_key(_cmp))
    return [r for r in out if r["score"] > 0 and r["score"] >= min_score][:k]


def load_sheet(path: str | Path) -> tuple[list[dict], str]:
    raw = Path(path).read_bytes()
    facts = [json.loads(l) for l in raw.decode("utf-8").splitlines() if l.strip()]
    return facts, hashlib.sha256(raw).hexdigest()


def make_retriever(path: str | Path, k: int = 3):
    """Return (callable question -> [fact text], sheet sha256)."""
    facts, sha = load_sheet(path)
    idx = build_index(facts)
    return (lambda q: [r["text"] for r in retrieve(idx, q, k=k)]), sha
