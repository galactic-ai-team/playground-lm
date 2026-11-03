# src/playground_lm/train/test_data.py
# Tests for the dataloader. Uses a synthetic token file: a known
# arithmetic-like sequence so we can assert x/y alignment exactly.

import os
import tempfile
from pathlib import Path

import numpy as np
import pytest
import torch

from playground_lm.train.data import build_dataloader


def _write_tokens(path: Path, token_ids: list[int]) -> None:
    """Write a list of ints as a uint16 binary file."""
    arr = np.array(token_ids, dtype=np.uint16)
    arr.tofile(path)


def test_raises_on_missing_file():
    with pytest.raises(FileNotFoundError):
        build_dataloader("/nonexistent/file.bin", context_len=4, batch_size=2)


def test_raises_on_odd_byte_length():
    with tempfile.TemporaryDirectory() as tmp:
        p = Path(tmp) / "bad.bin"
        p.write_bytes(b"\x01\x02\x03")  # 3 bytes — not a uint16 array
        with pytest.raises(ValueError, match="multiple of 2"):
            build_dataloader(p, context_len=4, batch_size=2)


def test_raises_on_too_few_tokens():
    with tempfile.TemporaryDirectory() as tmp:
        p = Path(tmp) / "short.bin"
        _write_tokens(p, [1, 2, 3])  # 3 tokens, context_len=4 needs 5
        with pytest.raises(ValueError, match="need at least"):
            build_dataloader(p, context_len=4, batch_size=2)


def test_raises_on_bad_context_len():
    with tempfile.TemporaryDirectory() as tmp:
        p = Path(tmp) / "tokens.bin"
        _write_tokens(p, list(range(100)))
        with pytest.raises(ValueError, match="context_len must be positive"):
            build_dataloader(p, context_len=0, batch_size=2)


def test_x_y_alignment():
    """y is x shifted left by one token — this is the fundamental invariant."""
    with tempfile.TemporaryDirectory() as tmp:
        p = Path(tmp) / "seq.bin"
        _write_tokens(p, list(range(20)))
        loader = build_dataloader(p, context_len=4, batch_size=2, shuffle=False)

        x, y = next(loader)
        assert x.dtype == torch.int64
        assert y.dtype == torch.int64
        assert x.shape == (2, 4)
        assert y.shape == (2, 4)

        # y should equal x shifted left by 1, for every position.
        # For sequential (shuffle=False) starting at 0:
        #   x[0] = [0,1,2,3], y[0] = [1,2,3,4]
        #   x[1] = [4,5,6,7], y[1] = [5,6,7,8]
        assert x[0].tolist() == [0, 1, 2, 3]
        assert y[0].tolist() == [1, 2, 3, 4]
        assert x[1].tolist() == [4, 5, 6, 7]
        assert y[1].tolist() == [5, 6, 7, 8]


def test_shuffle_produces_different_order():
    """Same seed -> same order; different seeds -> (probabilistically) different."""
    with tempfile.TemporaryDirectory() as tmp:
        p = Path(tmp) / "seq.bin"
        _write_tokens(p, list(range(1000)))

        a = next(build_dataloader(p, 8, 4, shuffle=True, seed=1))
        b = next(build_dataloader(p, 8, 4, shuffle=True, seed=1))
        c = next(build_dataloader(p, 8, 4, shuffle=True, seed=2))

        assert torch.equal(a[0], b[0]), "same seed should reproduce same batch"
        assert torch.equal(a[1], b[1])
        # Different seeds should give different batches (probabilistic; chance
        # of collision is negligible at 1000 tokens, batch 4, context 8).
        assert not torch.equal(a[0], c[0]), "different seeds should differ"


def test_shuffle_false_is_deterministic():
    """No shuffle = sequential, same order every time."""
    with tempfile.TemporaryDirectory() as tmp:
        p = Path(tmp) / "seq.bin"
        _write_tokens(p, list(range(50)))

        a = next(build_dataloader(p, 4, 2, shuffle=False))
        b = next(build_dataloader(p, 4, 2, shuffle=False))
        assert torch.equal(a[0], b[0])
        assert torch.equal(a[1], b[1])


def test_infinite_iterator():
    """Should be able to pull many batches without exhausting (wraps)."""
    with tempfile.TemporaryDirectory() as tmp:
        p = Path(tmp) / "seq.bin"
        _write_tokens(p, list(range(20)))  # 20 tokens, context 4, batch 2
        loader = build_dataloader(p, 4, 2, shuffle=False)

        # Should not raise or StopIteration, even after many more samples than
        # the file could produce in one pass.
        for _ in range(20):
            x, y = next(loader)
            assert x.shape == (2, 4)


def test_eval_loader_no_shuffle_uses_all_tokens():
    """With shuffle=False and drop_last=False, sequential iteration should
    visit every token as a start position over enough batches."""
    with tempfile.TemporaryDirectory() as tmp:
        p = Path(tmp) / "seq.bin"
        _write_tokens(p, list(range(16)))  # 16 tokens, context 4 -> 12 starts
        loader = build_dataloader(p, 4, 4, shuffle=False, drop_last=False)

        seen_starts = set()
        # 12 valid starts, batch 4 -> 3 batches to cover them.
        for _ in range(3):
            x, _ = next(loader)
            for row in x:
                seen_starts.add(int(row[0].item()))
        # Starts should be {0,1,2,...,11} — every valid start position.
        assert seen_starts == set(range(12))
