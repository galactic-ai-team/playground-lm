"""Tests for data/contamination_check.py (Nova). Offline, synthetic.

Stub tokenizer: one token per character (ord value), so a "13-gram"
is literally 13 characters — planted passages are easy to construct.
Run: python -m pytest data/
"""

import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).parent))

from contamination_check import check


class StubTokenizer:
    def encode(self, text, *, add_eot=False):
        ids = [ord(c) for c in text]
        if add_eot:
            ids.append(50256)
        return ids


def _shard(tmp: Path, texts: list[str]) -> Path:
    ids: list[int] = []
    for t in texts:
        ids += [ord(c) for c in t] + [50256]
    p = tmp / "train.bin"
    np.asarray(ids, dtype=np.uint16).tofile(p)
    return p


def test_flags_planted_item_only(tmp_path):
    planted = "the quick brown fox jumps"      # 25 tokens >= 13
    clean = "something entirely different here"
    shard = _shard(tmp_path, ["filler text one", planted, "filler text two"])
    report = check(
        shard,
        {"generic": [("item-a", planted), ("item-b", clean), ("tiny", "short")]},
        StubTokenizer(),
        chunk_tokens=16,  # force the planted passage across chunk edges
    )
    b = report["benchmarks"]["generic"]
    assert b["items"] == 3
    assert b["items_too_short_to_check"] == 1       # "short" has no 13-gram
    assert b["flagged_item_ids"] == ["item-a"]
    assert b["matched_ngram_positions_in_shard"] > 0


def test_clean_shard_flags_nothing(tmp_path):
    shard = _shard(tmp_path, ["completely unrelated training text"])
    report = check(
        shard,
        {"generic": [("item-a", "the quick brown fox jumps")]},
        StubTokenizer(),
    )
    b = report["benchmarks"]["generic"]
    assert b["items_flagged"] == 0
    assert b["matched_ngram_positions_in_shard"] == 0
    assert report["train_tokens"] > 0


def test_hellaswag_loader_indexes_ctx_and_endings(tmp_path):
    import json

    from contamination_check import load_items

    p = tmp_path / "hs.jsonl"
    p.write_text(json.dumps({
        "ind": 7, "ctx": "A man is sitting",
        "endings": ["and eats.", "and sleeps."],
    }) + "\n")
    items = load_items("hellaswag", p)
    assert [i for i, _ in items] == ["7", "7", "7"]
    assert items[1][1] == "A man is sitting and eats."
