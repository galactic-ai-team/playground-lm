#!/usr/bin/env python3
"""Split the already-built Stage F train set into facts and chat streams.

Does not add or drop examples; it only packs the existing G5-PASS train.jsonl
so Claude's three-way mixer can draw fact/arith, chat, and FineWeb rows
at exact shares. TriviaQA/SQuAD stay skipped. Val is not rewritten.
"""
from __future__ import annotations

import argparse
import json
import sys
from collections import Counter
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "data"))
sys.path.insert(0, str(ROOT / "src"))

from build_stage_e_sft import write_split  # noqa: E402
from playground_lm.tokenizer import get_tokenizer  # noqa: E402

FACT_SOURCES = frozenset({"openai/gsm8k", "generated-arithmetic"})
CHAT_SOURCES = frozenset({"databricks/databricks-dolly-15k", "generated-identity"})


def split_rows(train: list[dict]) -> tuple[list[dict], list[dict]]:
    facts = [r for r in train if r.get("source") in FACT_SOURCES]
    chat = [r for r in train if r.get("source") in CHAT_SOURCES]
    other = [r.get("source") for r in train if r.get("source") not in FACT_SOURCES | CHAT_SOURCES]
    if other:
        raise ValueError(f"unclassified sources: {Counter(other)}")
    if len(facts) + len(chat) != len(train):
        raise ValueError("split does not cover the train set")
    return facts, chat


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--sft-dir", type=Path, required=True, help="dir with train.jsonl")
    args = ap.parse_args()
    train_path = args.sft_dir / "train.jsonl"
    train = [json.loads(l) for l in train_path.read_text().splitlines() if l.strip()]
    facts, chat = split_rows(train)
    tok = get_tokenizer()
    facts_meta = write_split(args.sft_dir / "facts", "train", facts, tok)
    chat_meta = write_split(args.sft_dir / "chat", "train", chat, tok)
    manifest = {
        "schema_version": 1,
        "kind": "stage_f_three_way_streams",
        "note": "Split of the existing Stage F train set; no new examples. G5 still applies to the combined corpus.",
        "fact_sources": sorted(FACT_SOURCES),
        "chat_sources": sorted(CHAT_SOURCES),
        "facts": facts_meta,
        "chat": chat_meta,
        "parent_train_examples": len(train),
    }
    out = args.sft_dir / "streams_manifest.json"
    out.write_text(json.dumps(manifest, indent=2) + "\n")
    print(json.dumps(manifest, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
