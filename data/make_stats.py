"""Aggregate stats over a processed-data directory (Nova).

Reads every *.jsonl produced by the prepare scripts and writes
corpus-stats.json: totals per source/author/split, doc counts, chars,
and approximate tokens (4 chars/token — see common.CHARS_PER_TOKEN;
replaced by exact GPT-2 BPE counts once GPT-6's tokenizer lands).

Usage: python data/make_stats.py data/processed
"""

from __future__ import annotations

import json
import sys
from collections import Counter
from pathlib import Path

from common import approx_tokens, read_jsonl


def main() -> None:
    root = Path(sys.argv[1] if len(sys.argv) > 1 else "data/processed")
    by_file: dict[str, dict] = {}
    totals = Counter()
    for path in sorted(root.glob("*.jsonl")):
        docs = chars = tokens = 0
        authors: Counter = Counter()
        for rec in read_jsonl(path):
            docs += 1
            chars += len(rec["text"])
            tokens += approx_tokens(rec["text"])
            if rec.get("author"):
                authors[rec["author"]] += 1
        by_file[path.name] = {
            "docs": docs, "chars": chars, "approx_tokens": tokens,
            "per_author": dict(authors),
        }
        totals["docs"] += docs
        totals["chars"] += chars
        totals["approx_tokens"] += tokens
    out = {"files": by_file, "totals": dict(totals),
           "chars_per_token_assumption": 4.0}
    (root / "corpus-stats.json").write_text(json.dumps(out, indent=2))
    print(json.dumps(out, indent=2))


if __name__ == "__main__":
    main()
