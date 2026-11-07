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

Usage:
  python data/build_finetune_corpus.py --out data/finetune
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


def build_finetune_corpus(
    fetch: Callable[[str], dict],
    base: str = DEFAULT_BASE,
    *,
    rooms: list[str] | None = None,
    include_simulated: bool = False,
    val_pct: int = 10,
    pause_s: float = 0.2,
    snapshot_utc: str | None = None,
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
    args = ap.parse_args()

    rooms = args.rooms.split(",") if args.rooms else None
    train, val, stats = build_finetune_corpus(
        fetch_json,
        args.base,
        rooms=rooms,
        include_simulated=args.include_simulated,
        val_pct=args.val_pct,
    )
    written = write_finetune_corpus(args.out, train, val, stats)
    print(json.dumps(written, indent=2))


if __name__ == "__main__":
    main()
