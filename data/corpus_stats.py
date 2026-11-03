"""Corpus statistics for PlaygroundLM (Nova).

Reads a processed-data directory (the prepare scripts' outputs) and
reports, per source and per author:
  - document counts and approximate token counts (4 chars/token; see
    common.CHARS_PER_TOKEN — replaced by exact BPE counts later)
  - train/val split sizes
  - dedupe rate, from the prepare runs' sidecar stats
    (fineweb-stats.json / playground-stats.json: kept vs rejected)

Writes corpus-report.json and prints a Markdown summary suitable for
pasting into a PR or the training log.

Usage: python data/corpus_stats.py data/processed
"""

from __future__ import annotations

import json
import sys
from collections import Counter, defaultdict
from pathlib import Path

from common import approx_tokens, read_jsonl


def split_of(filename: str) -> str:
    if "val" in filename:
        return "val"
    if "train" in filename:
        return "train"
    return "other"


def main() -> None:
    root = Path(sys.argv[1] if len(sys.argv) > 1 else "data/processed")
    per_source: dict[str, Counter] = defaultdict(Counter)
    per_author: dict[str, Counter] = defaultdict(Counter)
    per_split: dict[str, Counter] = defaultdict(Counter)

    for path in sorted(root.glob("*.jsonl")):
        split = split_of(path.name)
        for rec in read_jsonl(path):
            tokens = approx_tokens(rec["text"])
            for bucket, key in (
                (per_source, rec["source"]),
                (per_split, split),
            ):
                bucket[key]["docs"] += 1
                bucket[key]["approx_tokens"] += tokens
                bucket[key]["chars"] += len(rec["text"])
            if rec.get("author"):
                per_author[rec["author"]]["docs"] += 1
                per_author[rec["author"]]["approx_tokens"] += tokens

    # Dedupe rates from the prepare runs' sidecar stats, when present.
    dedupe: dict[str, dict] = {}
    for sidecar, label in (("fineweb-stats.json", "fineweb-edu"),
                           ("playground-stats.json", "playground")):
        p = root / sidecar
        if not p.exists():
            continue
        s = json.loads(p.read_text())
        if label == "fineweb-edu":
            kept = s["docs_kept_train"] + s["docs_kept_val"]
            dropped = s["docs_rejected_or_dup"]
        else:
            kept = s["docs_train"] + s["docs_val"]
            dropped = s["docs_rejected_or_dup"]
        total = kept + dropped
        dedupe[label] = {
            "kept": kept, "dropped": dropped,
            "dedupe_rate_pct": round(100 * dropped / total, 2) if total else 0.0,
        }

    report = {
        "per_source": {k: dict(v) for k, v in sorted(per_source.items())},
        "per_author": {k: dict(v) for k, v in sorted(per_author.items())},
        "per_split": {k: dict(v) for k, v in sorted(per_split.items())},
        "dedupe": dedupe,
        "chars_per_token_assumption": 4.0,
    }
    (root / "corpus-report.json").write_text(json.dumps(report, indent=2))

    def table(title: str, rows: dict) -> None:
        print(f"\n### {title}\n")
        print("| key | docs | approx tokens |")
        print("|---|---:|---:|")
        for k, v in rows.items():
            print(f"| {k} | {v['docs']:,} | {v['approx_tokens']:,} |")

    print("# Corpus report (approximate tokens, 4 chars/token)")
    table("By source", report["per_source"])
    table("By author (Playground)", report["per_author"])
    table("By split", report["per_split"])
    if dedupe:
        print("\n### Dedupe\n")
        print("| corpus | kept | dropped | rate |")
        print("|---|---:|---:|---:|")
        for k, v in dedupe.items():
            print(f"| {k} | {v['kept']:,} | {v['dropped']:,} | {v['dedupe_rate_pct']}% |")


if __name__ == "__main__":
    main()
