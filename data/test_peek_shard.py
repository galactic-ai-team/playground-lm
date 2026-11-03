"""Tests for data/peek_shard.py (Nova). Run: python -m pytest data/"""

import sys
from pathlib import Path

import numpy as np
import pytest

sys.path.insert(0, str(Path(__file__).parent))

from peek_shard import peek


class StubDecoder:
    """Decodes ids to a readable, checkable string."""

    def decode(self, token_ids):
        return " ".join(str(int(t)) for t in token_ids)


def _bin(tmp: Path, ids: list[int]) -> Path:
    p = tmp / "train.bin"
    np.asarray(ids, dtype=np.uint16).tofile(p)
    return p


def test_peek_reports_count_and_max(tmp_path):
    p = _bin(tmp_path, [10, 50256, 42, 7])
    result = peek(p, StubDecoder(), windows=1, length=2, seed=0)
    assert result["tokens"] == 4
    assert result["max_id"] == 50256


def test_peek_windows_are_real_slices_and_deterministic(tmp_path):
    ids = list(range(100))
    p = _bin(tmp_path, ids)
    a = peek(p, StubDecoder(), windows=3, length=10, seed=42)
    b = peek(p, StubDecoder(), windows=3, length=10, seed=42)
    assert a == b  # same seed, same windows
    for w in a["windows"]:
        expected = " ".join(str(t) for t in ids[w["start"] : w["start"] + 10])
        assert w["text"] == expected


def test_peek_small_shard_single_window(tmp_path):
    p = _bin(tmp_path, [5, 6, 7])
    result = peek(p, StubDecoder(), windows=3, length=256, seed=1)
    assert len(result["windows"]) == 1
    assert result["windows"][0]["start"] == 0
    assert result["windows"][0]["text"] == "5 6 7"


def test_peek_empty_shard_raises(tmp_path):
    p = _bin(tmp_path, [])
    with pytest.raises(ValueError):
        peek(p, StubDecoder())
