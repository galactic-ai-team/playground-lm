"""Build the Stage D Playground fine-tune corpus in one step (Nova).

Pipeline:
  1. Fetch public Playground room messages and Memory Wall notes with
     fetch_playground.collect() (public APIs only; no relay link).
  2. Clean and dedupe with the shared helpers in common.py, canonicalize
     authors with prepare_playground.py's alias/display tables, and
     prefix each document "Name: " exactly as prepare_playground.py does.
  3. Split with common.split_for(), the same hash-stable split used by
     the rest of the dataset lane: identical text can never straddle
     train and held-out eval, and a later rebuild keeps the same eval
     documents held out.

Outputs (common.py JSONL record schema, ready for tokenize_shards.py):
  <out>/playground-train.jsonl   Stage D fine-tune train documents
  <out>/playground-val.jsonl     held-out eval documents (do not train)
  <out>/playground-finetune-stats.json

The split is at document level, as in prepare_playground.py. The
held-out file is for Playground perplexity / voice checks and for
measuring generalization in the verbatim-copy check; the copy check
itself is run against train documents by the eval lane.

Optional D.1 rebalancing (both off by default, both recorded in the
stats sidecar):
  --min-val-per-author N  deterministically promote N held-out docs
      per author (smallest record ids from that author's train docs)
      so every voice is measurable in val; at least one train doc per
      author is always kept.
  --min-per-author N --max-repeat R  oversample minority authors in
      TRAIN ONLY by repeating their deduped train docs (round-robin)
      until each author has N train docs or a doc would exceed R total
      copies. The public corpus is finite, so this reweights existing
      text — it creates none. Val is never repeated. Repetition trades
      against the verbatim-copy check, so caps stay small.

Usage:
  python data/build_finetune_corpus.py --out data/finetune
  python data/build_finetune_corpus.py --out data/finetune_d1 \
      --min-val-per-author 1 --min-per-author 20 --max-repeat 4
"""

from __future__ import annotations

import argparse
import json
from collections import Counter
from datetime import datetime, timezone
from pathlib import Path
from typing import Callable, Iterable

from common import Deduper, approx_tokens, clean_text, make_record, split_for, write_jsonl
from fetch_playground import DEFAULT_BASE, collect, fetch_json
from prepare_playground import DISPLAY, canonical_author


def transform_records(
    raw_records: Iterable[dict], *, val_pct: int = 10
) -> tuple[list[dict], list[dict], dict]:
    """Clean, dedupe, author-label and split fetched Playground records.

    Returns (train, val, stats). Dedupe happens before author prefixing,
    matching prepare_playground.py: the same message text kept once, no
    matter which room or author label it arrived with.
    """
    deduper = Deduper(near=True)
    train: list[dict] = []
    val: list[dict] = []
    per_author: Counter = Counter()
    per_source: Counter = Counter()
    rejected = 0

    for raw in raw_records:
        text = clean_text(raw.get("text") or "", min_chars=2)
        if text is None or not deduper.is_new(text):
            rejected += 1
            continue
        author = canonical_author(raw.get("author"))
        kind = str(raw.get("kind") or "lobby").lower()
        source = "playground-wall" if "wall" in kind else "playground-lobby"
        doc = f"{DISPLAY[author]}: {text}"
        rec = make_record(doc, source=source, author=author)
        (val if split_for(doc, val_pct) == "val" else train).append(rec)
        per_author[author] += 1
        per_source[source] += 1

    stats = {
        "docs_train": len(train),
        "docs_val": len(val),
        "docs_rejected_or_dup": rejected,
        "exact_dups": deduper.exact_dups,
        "near_dups": deduper.near_dups,
        "per_author_docs": dict(sorted(per_author.items())),
        "per_source_docs": dict(sorted(per_source.items())),
        "approx_tokens_train": sum(approx_tokens(r["text"]) for r in train),
        "approx_tokens_val": sum(approx_tokens(r["text"]) for r in val),
        "val_pct": val_pct,
    }
    return train, val, stats


def ensure_min_val_per_author(
    train: list[dict], val: list[dict], *, min_val_per_author: int = 0
) -> tuple[list[dict], list[dict], dict]:
    """Promote train docs into val so every author is held out at least
    min_val_per_author times. Deterministic: an author's smallest
    record ids move first, and an author always keeps >= 1 train doc.
    Returns (train, val, {author: [moved ids]})."""
    if min_val_per_author <= 0:
        return train, val, {}
    val_counts = Counter(r["author"] for r in val)
    by_author: dict[str, list[dict]] = {}
    for rec in train:
        by_author.setdefault(rec["author"], []).append(rec)
    moved: dict[str, list[str]] = {}
    moved_ids: set[str] = set()
    for author in sorted(by_author):
        need = min_val_per_author - val_counts.get(author, 0)
        if need <= 0:
            continue
        docs = sorted(by_author[author], key=lambda r: r["id"])
        for rec in docs[: max(0, min(need, len(docs) - 1))]:
            moved.setdefault(author, []).append(rec["id"])
            moved_ids.add(rec["id"])
            val.append(rec)
    if not moved_ids:
        return train, val, moved
    train = [r for r in train if r["id"] not in moved_ids]
    return train, val, moved


def apply_min_per_author(
    train: list[dict], *, min_per_author: int = 0, max_repeat: int = 4
) -> tuple[list[dict], dict]:
    """Oversample minority authors in train by repeating their docs.

    Repeats cycle round-robin through an author's deduped docs; no doc
    exceeds max_repeat total copies. Returns (train + repeats appended,
    info) with per-author before/after counts; authors that hit the
    repeat cap below the target are reported as capped, not silently
    padded further.
    """
    info: dict = {
        "min_per_author": min_per_author,
        "max_repeat": max_repeat,
        "per_author": {},
    }
    if min_per_author <= 0:
        return train, info
    by_author: dict[str, list[dict]] = {}
    for rec in train:
        by_author.setdefault(rec["author"], []).append(rec)
    copies = Counter(r["id"] for r in train)
    counts = Counter(r["author"] for r in train)
    repeats: list[dict] = []
    for author in sorted(by_author):
        docs = by_author[author]
        start = counts[author]
        i = 0
        stalled = 0
        while counts[author] < min_per_author and stalled < len(docs):
            rec = docs[i % len(docs)]
            i += 1
            if copies[rec["id"]] >= max_repeat:
                stalled += 1
                continue
            stalled = 0
            copies[rec["id"]] += 1
            counts[author] += 1
            repeats.append(rec)
        info["per_author"][author] = {
            "unique_docs": start,
            "final_docs": counts[author],
            "repeats_added": counts[author] - start,
            "capped": counts[author] < min_per_author,
        }
    return train + repeats, info


def build_finetune_corpus(
    fetch: Callable[[str], dict],
    base: str = DEFAULT_BASE,
    *,
    rooms: list[str] | None = None,
    include_simulated: bool = False,
    val_pct: int = 10,
    pause_s: float = 0.2,
    snapshot_utc: str | None = None,
    min_per_author: int = 0,
    max_repeat: int = 4,
    min_val_per_author: int = 0,
) -> tuple[list[dict], list[dict], dict]:
    """Fetch from the public Playground APIs and build the split corpus."""
    lobby, wall = collect(
        fetch,
        base,
        rooms=rooms,
        include_simulated=include_simulated,
        pause_s=pause_s,
    )
    raw = sorted(lobby + wall, key=lambda r: r.get("ts") or "")
    train, val, stats = transform_records(raw, val_pct=val_pct)
    stats.update(
        {
            "source": "playground-finetune",
            "base_url": base.rstrip("/"),
            "snapshot_utc": snapshot_utc
            or datetime.now(timezone.utc).isoformat(timespec="seconds"),
            "fetched_lobby_docs": len(lobby),
            "fetched_wall_docs": len(wall),
            "include_simulated": include_simulated,
        }
    )
    if min_val_per_author > 0 or min_per_author > 0:
        train, val, moved = ensure_min_val_per_author(
            train, val, min_val_per_author=min_val_per_author
        )
        if moved:
            stats["val_promotions"] = moved
        train, rebalance = apply_min_per_author(
            train, min_per_author=min_per_author, max_repeat=max_repeat
        )
        if min_per_author > 0:
            stats["rebalance"] = rebalance
        # Recompute split stats so they describe the files as written.
        stats["docs_train"] = len(train)
        stats["docs_val"] = len(val)
        stats["approx_tokens_train"] = sum(
            approx_tokens(r["text"]) for r in train
        )
        stats["approx_tokens_val"] = sum(approx_tokens(r["text"]) for r in val)
        per_author_train: dict[str, dict] = {}
        for rec in train:
            entry = per_author_train.setdefault(
                rec["author"], {"docs": 0, "approx_tokens": 0}
            )
            entry["docs"] += 1
            entry["approx_tokens"] += approx_tokens(rec["text"])
        stats["per_author_train"] = dict(sorted(per_author_train.items()))
    return train, val, stats


def write_finetune_corpus(
    out_dir: Path, train: list[dict], val: list[dict], stats: dict
) -> dict:
    """Write train/eval JSONL plus the stats sidecar. Returns the stats."""
    n_train = write_jsonl(out_dir / "playground-train.jsonl", train)
    n_val = write_jsonl(out_dir / "playground-val.jsonl", val)
    written = dict(stats)
    written["docs_train"] = n_train
    written["docs_val"] = n_val
    out_dir.mkdir(parents=True, exist_ok=True)
    (out_dir / "playground-finetune-stats.json").write_text(
        json.dumps(written, indent=2)
    )
    return written


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--base", default=DEFAULT_BASE)
    ap.add_argument("--out", type=Path, default=Path("data/finetune"))
    ap.add_argument("--val-pct", type=int, default=10)
    ap.add_argument("--rooms", default=None,
                    help="comma-separated room ids (default: all with messages)")
    ap.add_argument("--include-simulated", action="store_true")
    ap.add_argument("--min-per-author", type=int, default=0,
                    help="oversample train so each author has >= N docs (0=off)")
    ap.add_argument("--max-repeat", type=int, default=4,
                    help="max total copies of any doc when oversampling")
    ap.add_argument("--min-val-per-author", type=int, default=0,
                    help="promote docs so each author has >= N val docs (0=off)")
    args = ap.parse_args()

    rooms = args.rooms.split(",") if args.rooms else None
    train, val, stats = build_finetune_corpus(
        fetch_json,
        args.base,
        rooms=rooms,
        include_simulated=args.include_simulated,
        val_pct=args.val_pct,
        min_per_author=args.min_per_author,
        max_repeat=args.max_repeat,
        min_val_per_author=args.min_val_per_author,
    )
    written = write_finetune_corpus(args.out, train, val, stats)
    print(json.dumps(written, indent=2))


if __name__ == "__main__":
    main()
