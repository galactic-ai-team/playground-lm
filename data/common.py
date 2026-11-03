"""Shared dataset utilities for PlaygroundLM (Nova's lane).

One record format everywhere — JSONL, one JSON object per line:
  {"text": str,
   "source": "fineweb-edu" | "playground-lobby" | "playground-wall",
   "author": str | None,          # canonical agent id for Playground text
   "id": str}                     # first 16 hex of sha256(normalized text)

Splits are hash-stable: split_for(text) depends only on the text itself,
so (a) identical text always lands in the same split (no train/val
leakage, even across dedupe misses), and (b) re-running a prep script
with more data never moves an existing document between splits.
"""

from __future__ import annotations

import hashlib
import json
import re
import unicodedata
from pathlib import Path
from typing import Iterable, Iterator

# GPT-2 BPE averages ~4 characters per token on English text. This is an
# ESTIMATE for budgeting/stats only; exact counts come from GPT-6's
# tokenizer pass before training starts.
CHARS_PER_TOKEN = 4.0

_CTRL = re.compile(r"[\x00-\x08\x0b\x0c\x0e-\x1f\x7f]")
_SPACES = re.compile(r"[ \t]+")
_BLANKS = re.compile(r"\n{3,}")


def approx_tokens(text: str) -> int:
    return max(1, round(len(text) / CHARS_PER_TOKEN))


def normalize_for_hash(text: str) -> str:
    """Canonical form used for ids, dedupe keys and split hashing."""
    return " ".join(text.lower().split())


def stable_id(text: str) -> str:
    return hashlib.sha256(normalize_for_hash(text).encode("utf-8")).hexdigest()[:16]


def clean_text(text: str, *, min_chars: int = 1, max_chars: int = 200_000) -> str | None:
    """Light, lossless-ish cleaning. Returns None if the doc is rejected.

    NFC-normalizes, strips control characters (keeping \\n and \\t),
    collapses runs of spaces and 3+ blank lines, trims. Does NOT
    lowercase or alter wording — the model trains on what this returns.
    """
    if not isinstance(text, str):
        return None
    text = unicodedata.normalize("NFC", text)
    text = _CTRL.sub("", text)
    text = _SPACES.sub(" ", text)
    text = _BLANKS.sub("\n\n", text)
    text = text.strip()
    if len(text) < min_chars or len(text) > max_chars:
        return None
    return text


def split_for(text: str, val_pct: int) -> str:
    """'val' for val_pct percent of texts, 'train' otherwise. Stable."""
    if not 0 <= val_pct <= 100:
        raise ValueError("val_pct must be in [0, 100]")
    bucket = int(hashlib.sha256(normalize_for_hash(text).encode("utf-8")).hexdigest()[:8], 16) % 100
    return "val" if bucket < val_pct else "train"


class Deduper:
    """Exact dedupe + optional near-dedupe via word-shingle overlap.

    Near-dedupe keeps a set of 5-word shingles per kept document and
    rejects a new doc if >= threshold of its shingles were already seen.
    It is O(corpus shingles) in memory — use it for the (small)
    Playground corpus, not for FineWeb (already deduped upstream).
    """

    def __init__(self, near: bool = False, shingle_words: int = 5, threshold: float = 0.8):
        self._seen: set[str] = set()
        self._shingles: set[int] = set()
        self.near = near
        self.shingle_words = shingle_words
        self.threshold = threshold
        self.exact_dups = 0
        self.near_dups = 0

    def _shingle_hashes(self, text: str) -> set[int]:
        words = normalize_for_hash(text).split()
        n = self.shingle_words
        if len(words) < n:
            return {hash(" ".join(words))} if words else set()
        return {hash(" ".join(words[i : i + n])) for i in range(len(words) - n + 1)}

    def is_new(self, text: str) -> bool:
        """True (and record it) if this text should be kept."""
        key = stable_id(text)
        if key in self._seen:
            self.exact_dups += 1
            return False
        if self.near:
            sh = self._shingle_hashes(text)
            if sh:
                overlap = len(sh & self._shingles) / len(sh)
                if overlap >= self.threshold:
                    self.near_dups += 1
                    return False
                self._shingles |= sh
        self._seen.add(key)
        return True


def make_record(text: str, source: str, author: str | None = None) -> dict:
    return {"text": text, "source": source, "author": author, "id": stable_id(text)}


def write_jsonl(path: Path, records: Iterable[dict]) -> int:
    path.parent.mkdir(parents=True, exist_ok=True)
    count = 0
    with path.open("w", encoding="utf-8") as f:
        for rec in records:
            f.write(json.dumps(rec, ensure_ascii=False) + "\n")
            count += 1
    return count


def read_jsonl(path: Path) -> Iterator[dict]:
    with path.open(encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                yield json.loads(line)
