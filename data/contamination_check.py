"""Benchmark contamination check for PlaygroundLM (Nova).

Question it answers: does our training shard contain any 13-gram
(13 consecutive GPT-2 tokens) that also appears in a benchmark item?
If it does, that benchmark item's score may be inflated by leakage,
and it gets flagged by id so we can report scores honestly (GPT-2 /
GPT-3 style decontamination reporting).

Method:
  - Each benchmark item's text is tokenized (GPT-6's get_tokenizer,
    no EOT) and all of its 13-grams are hashed with a 64-bit
    polynomial rolling hash (base B, arithmetic mod 2^64).
  - The train shard is streamed in chunks; every position's 13-gram
    hash is computed vectorized with numpy and tested against the
    benchmark hash set. Memory stays O(chunk + benchmark), never
    O(shard). A 12-token carry crosses chunk boundaries.
  - Items shorter than 13 tokens have no 13-grams: they are counted
    as "too_short_to_check", never silently passed.

Usage:
  python data/contamination_check.py \
      --train data/processed/tokens/train.bin \
      --benchmark hellaswag=data/benchmarks/hellaswag_val.jsonl \
      --benchmark lambada=data/benchmarks/lambada_test.jsonl \
      --out data/processed/tokens/contamination-report.json

Benchmark file formats (JSONL):
  hellaswag: {"ind": ..., "ctx": ..., "endings": [...]} — the ctx and
             each ctx+ending are indexed under the item's id.
  lambada:   {"text": ...} (id from "id" field if present, else index)
  generic:   {"id": ..., "text": ...}
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Protocol, Sequence

import numpy as np
from numpy.lib.stride_tricks import sliding_window_view

NGRAM = 13
BASE = 9_223_037  # odd 64-bit polynomial base
MOD = 1 << 64
CHUNK_TOKENS = 1_000_000


class EncoderLike(Protocol):
    def encode(self, text: str, *, add_eot: bool = False) -> list[int]: ...


def _powers(n: int) -> np.ndarray:
    return np.array([pow(BASE, n - 1 - i, MOD) for i in range(n)], dtype=np.uint64)


def hash_windows(tokens: np.ndarray, n: int, powers: np.ndarray) -> np.ndarray:
    """64-bit polynomial hash of every length-n window of tokens."""
    if tokens.shape[0] < n:
        return np.empty(0, dtype=np.uint64)
    windows = sliding_window_view(tokens.astype(np.uint64), n)
    return (windows * powers).sum(axis=1, dtype=np.uint64)


def load_items(kind: str, path: Path) -> list[tuple[str, str]]:
    """Return [(item_id, text)] for a benchmark JSONL file."""
    items: list[tuple[str, str]] = []
    with path.open(encoding="utf-8") as f:
        for i, line in enumerate(f):
            line = line.strip()
            if not line:
                continue
            o = json.loads(line)
            if kind == "hellaswag":
                item_id = str(i)  # row index: the harness scores per row;
                # "ind" repeats across rows (9,609 inds vs 10,042 rows),
                # so ind-keying flags clean sibling rows and skews counts.
                ctx = o.get("ctx", "")
                items.append((item_id, ctx))
                for ending in o.get("endings", []):
                    items.append((item_id, f"{ctx} {ending}"))
            elif kind == "lambada":
                items.append((str(o.get("id", i)), o["text"]))
            else:  # generic
                items.append((str(o.get("id", i)), o["text"]))
    return items


def build_index(
    items: list[tuple[str, str]], tokenizer: EncoderLike, n: int, powers: np.ndarray
) -> tuple[dict[int, set[str]], int, int]:
    """Map gram-hash -> item ids. Returns (index, n_items, n_too_short)."""
    index: dict[int, set[str]] = {}
    ids_seen: set[str] = set()
    too_short: set[str] = set()
    for item_id, text in items:
        ids_seen.add(item_id)
        ids = tokenizer.encode(text)
        if len(ids) < n:
            too_short.add(item_id)
            continue
        hashes = hash_windows(np.asarray(ids, dtype=np.uint64), n, powers)
        for h in set(int(x) for x in hashes):
            index.setdefault(h, set()).add(item_id)
    return index, len(ids_seen), len(too_short)


def scan_shard(
    path: Path, bench_hashes: np.ndarray, n: int, powers: np.ndarray,
    *, chunk_tokens: int = CHUNK_TOKENS,
) -> tuple[int, int, set[int]]:
    """Stream the shard once. Returns (total_tokens, matched_positions,
    matched_hashes). bench_hashes must be sorted for np.isin."""
    data = np.memmap(path, dtype=np.uint16, mode="r")
    total = int(data.shape[0])
    matched_positions = 0
    matched_hashes: set[int] = set()
    carry = np.empty(0, dtype=np.uint64)
    pos = 0
    while pos < total:
        chunk = np.asarray(data[pos : pos + chunk_tokens], dtype=np.uint64)
        joined = np.concatenate([carry, chunk]) if carry.size else chunk
        hashes = hash_windows(joined, n, powers)
        if hashes.size:
            hit_mask = np.isin(hashes, bench_hashes)
            if hit_mask.any():
                matched_positions += int(hit_mask.sum())
                matched_hashes |= set(int(x) for x in hashes[hit_mask])
        carry = joined[-(n - 1) :] if joined.shape[0] >= n - 1 else joined
        pos += chunk_tokens
    return total, matched_positions, matched_hashes


def check(
    train_path: Path,
    benchmarks: dict[str, list[tuple[str, str]]],
    tokenizer: EncoderLike,
    *,
    n: int = NGRAM,
    chunk_tokens: int = CHUNK_TOKENS,
) -> dict:
    powers = _powers(n)
    report: dict = {"train_shard": str(train_path), "ngram": n, "benchmarks": {}}
    for name, items in benchmarks.items():
        index, n_items, n_too_short = build_index(items, tokenizer, n, powers)
        bench_hashes = np.array(sorted(index.keys()), dtype=np.uint64)
        total, matched_positions, matched_hashes = scan_shard(
            train_path, bench_hashes, n, powers, chunk_tokens=chunk_tokens
        )
        flagged: set[str] = set()
        for h in matched_hashes:
            flagged |= index[h]
        report["train_tokens"] = total
        report["benchmarks"][name] = {
            "items": n_items,
            "items_too_short_to_check": n_too_short,
            "items_flagged": len(flagged),
            "flagged_item_ids": sorted(flagged),
            "matched_ngram_positions_in_shard": matched_positions,
            "distinct_benchmark_ngrams_matched": len(matched_hashes),
        }
    return report


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--train", type=Path, required=True)
    ap.add_argument("--benchmark", action="append", required=True,
                    metavar="KIND=PATH",
                    help="kind in {hellaswag, lambada, generic}; repeatable")
    ap.add_argument("--out", type=Path, default=None)
    ap.add_argument("--ngram", type=int, default=NGRAM)
    args = ap.parse_args()

    repo_root = Path(__file__).resolve().parent.parent
    sys.path.insert(0, str(repo_root / "src"))
    from playground_lm.tokenizer import get_tokenizer

    benchmarks = {}
    for spec in args.benchmark:
        kind, _, p = spec.partition("=")
        benchmarks[kind] = load_items(kind, Path(p))
    report = check(args.train, benchmarks, get_tokenizer(), n=args.ngram)
    text = json.dumps(report, indent=2)
    print(text)
    if args.out:
        args.out.write_text(text)


if __name__ == "__main__":
    main()
