"""Tests for data/common.py (Nova). Run: python -m pytest data/"""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent))

from common import Deduper, approx_tokens, clean_text, split_for, stable_id


def test_clean_text_normalizes_and_trims():
    assert clean_text("  hello \t world  \n\n\n\nbye ") == "hello world\n\nbye"


def test_clean_text_rejects_short_and_control_chars():
    assert clean_text("ab", min_chars=3) is None
    assert clean_text("a\x00b") == "ab"


def test_split_is_stable_and_leak_free():
    text = "Nova placed a neon block by Crystal Lake."
    assert split_for(text, 10) == split_for(text, 10)
    # Case/whitespace variants are the same document after normalization,
    # so they can never straddle the split.
    assert split_for(text, 10) == split_for("  nova placed a neon block by crystal lake. ", 10)


def test_split_roughly_matches_pct():
    vals = sum(split_for(f"document number {i}", 10) == "val" for i in range(2000))
    assert 100 < vals < 300  # ~10%, deterministic hash buckets


def test_deduper_exact_and_near():
    d = Deduper(near=True)
    assert d.is_new("the quick brown fox jumps over the lazy dog again today")
    assert not d.is_new("the quick brown fox jumps over the lazy dog again today")
    assert d.exact_dups == 1
    assert not d.is_new("the quick brown fox jumps over the lazy dog again tomorrow")
    assert d.near_dups == 1


def test_stable_id_and_tokens():
    assert stable_id("Hello") == stable_id("  hello ")
    assert approx_tokens("x" * 40) == 10
