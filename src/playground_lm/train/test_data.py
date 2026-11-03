# src/playground_lm/train/test_data.py
# Tests for the dataloader.
#
# Two distinct stride semantics under test:
#   - stride=1 (train): every token position is a start; adjacent windows overlap.
#   - stride=context_len (eval): non-overlapping; each token in exactly one batch.
#
# Valid starts are in [0, num_tokens - context_len - 1] inclusive, because each
# sample needs context_len+1 tokens (context_len inputs + 1 target offset).
#
# Also tests resume behavior: seed_offset changes the shuffle order so a
# resumed session does not replay the same windows.
#
# Test data note: token IDs written as uint16 must be < 65536 (GT-2 vocab is
# 50257). Use `i % 50257` for large synthetic corpora.

import tempfile
from pathlib import Path

import numpy as np
import pytest
import torch

from playground_lm.train.data import build_dataloader


def _write_tokens(path: Path, token_ids: list[int]) -> None:
    arr = np.array(token_ids, dtype=np.uint16)
    arr.tofile(path)


def test_raises_on_missing_file():
    with pytest.raises(FileNotFoundError):
        build_dataloader("/nonexistent/file.bin", context_len=4, batch_size=2)


def test_raises_on_odd_byte_length():
    with tempfile.TemporaryDirectory() as tmp:
        p = Path(tmp) / "bad.bin"
        p.write_bytes(b"\x01\x02\x03")
        with pytest.raises(ValueError, match="multiple of 2"):
            build_dataloader(p, context_len=4, batch_size=2)


def test_raises_on_too_few_tokens():
    with tempfile.TemporaryDirectory() as tmp:
        p = Path(tmp) / "short.bin"
        _write_tokens(p, [1, 2, 3])
        with pytest.raises(ValueError, match="need at least"):
            build_dataloader(p, context_len=4, batch_size=2)


def test_raises_on_bad_context_len():
    with tempfile.TemporaryDirectory() as tmp:
        p = Path(tmp) / "tokens.bin"
        _write_tokens(p, list(range(100)))
        with pytest.raises(ValueError, match="context_len must be positive"):
            build_dataloader(p, context_len=0, batch_size=2)


def test_raises_on_bad_stride():
    with tempfile.TemporaryDirectory() as tmp:
        p = Path(tmp) / "tokens.bin"
        _write_tokens(p, list(range(100)))
        with pytest.raises(ValueError, match="stride must be positive"):
            build_dataloader(p, context_len=4, batch_size=2, stride=0)


def test_x_y_alignment_stride_1():
    """Train mode (stride=1): starts advance by 1, windows overlap."""
    with tempfile.TemporaryDirectory() as tmp:
        p = Path(tmp) / "seq.bin"
        _write_tokens(p, list(range(20)))
        loader = build_dataloader(
            p, context_len=4, batch_size=2, shuffle=False, stride=1,
        )

        x, y = next(loader)
        assert x.dtype == torch.int64
        assert y.dtype == torch.int64
        assert x.shape == (2, 4)
        assert y.shape == (2, 4)

        assert x[0].tolist() == [0, 1, 2, 3]
        assert y[0].tolist() == [1, 2, 3, 4]
        assert x[1].tolist() == [1, 2, 3, 4]
        assert y[1].tolist() == [2, 3, 4, 5]


def test_x_y_alignment_stride_context_len():
    """Eval mode (stride=context_len, the default): non-overlapping windows."""
    with tempfile.TemporaryDirectory() as tmp:
        p = Path(tmp) / "seq.bin"
        _write_tokens(p, list(range(20)))
        loader = build_dataloader(p, context_len=4, batch_size=2, shuffle=False)

        x, y = next(loader)
        assert x[0].tolist() == [0, 1, 2, 3]
        assert y[0].tolist() == [1, 2, 3, 4]
        assert x[1].tolist() == [4, 5, 6, 7]
        assert y[1].tolist() == [5, 6, 7, 8]


def test_shuffle_same_seed_reproduces():
    """Same seed + same seed_offset → same batches."""
    with tempfile.TemporaryDirectory() as tmp:
        p = Path(tmp) / "seq.bin"
        _write_tokens(p, list(range(1000)))

        a = next(build_dataloader(p, 8, 4, shuffle=True, stride=1, seed=1))
        b = next(build_dataloader(p, 8, 4, shuffle=True, stride=1, seed=1))
        assert torch.equal(a[0], b[0])
        assert torch.equal(a[1], b[1])


def test_shuffle_different_seeds_differ():
    """Different seeds → different batches (probabilistically)."""
    with tempfile.TemporaryDirectory() as tmp:
        p = Path(tmp) / "seq.bin"
        _write_tokens(p, list(range(1000)))

        a = next(build_dataloader(p, 8, 4, shuffle=True, stride=1, seed=1))
        c = next(build_dataloader(p, 8, 4, shuffle=True, stride=1, seed=2))
        assert not torch.equal(a[0], c[0])


def test_shuffle_seed_offset_changes_order():
    """seed_offset changes the RNG stream — used for resume."""
    with tempfile.TemporaryDirectory() as tmp:
        p = Path(tmp) / "seq.bin"
        _write_tokens(p, list(range(1000)))

        fresh = next(build_dataloader(p, 8, 4, shuffle=True, stride=1, seed=42, seed_offset=0))
        resumed = next(build_dataloader(p, 8, 4, shuffle=True, stride=1, seed=42, seed_offset=500))
        assert not torch.equal(fresh[0], resumed[0]), \
            "resumed session should sample different windows"


def test_shuffle_seed_offset_is_deterministic():
    """Same seed_offset (e.g. same resume point) reproduces."""
    with tempfile.TemporaryDirectory() as tmp:
        p = Path(tmp) / "seq.bin"
        _write_tokens(p, list(range(1000)))

        a = next(build_dataloader(p, 8, 4, shuffle=True, stride=1, seed=42, seed_offset=500))
        b = next(build_dataloader(p, 8, 4, shuffle=True, stride=1, seed=42, seed_offset=500))
        assert torch.equal(a[0], b[0])


def test_shuffle_does_not_allocate_start_array():
    """Shuffle mode exercises the no-precompute path on a moderately sized file.

    Token IDs must stay within uint16 range (GPT-2 vocab 50257), so use i % 50257.
    """
    with tempfile.TemporaryDirectory() as tmp:
        p = Path(tmp) / "seq.bin"
        _write_tokens(p, [i % 50257 for i in range(100_000)])
        loader = build_dataloader(p, context_len=4, batch_size=4, shuffle=True, stride=1)
        x, y = next(loader)
        assert x.shape == (4, 4)
        assert y.shape == (4, 4)


def test_sequential_deterministic():
    with tempfile.TemporaryDirectory() as tmp:
        p = Path(tmp) / "seq.bin"
        _write_tokens(p, list(range(50)))

        a = next(build_dataloader(p, 4, 2, shuffle=False))
        b = next(build_dataloader(p, 4, 2, shuffle=False))
        assert torch.equal(a[0], b[0])
        assert torch.equal(a[1], b[1])


def test_infinite_iterator():
    with tempfile.TemporaryDirectory() as tmp:
        p = Path(tmp) / "seq.bin"
        _write_tokens(p, list(range(20)))
        loader = build_dataloader(p, 4, 2, shuffle=False, stride=1)

        for _ in range(20):
            x, _ = next(loader)
            assert x.shape == (2, 4)


def test_eval_loader_respects_token_boundary():
    """With 16 tokens and context_len=4, valid starts are 0, 4, 8 — not 12."""
    with tempfile.TemporaryDirectory() as tmp:
        p = Path(tmp) / "seq.bin"
        _write_tokens(p, list(range(16)))
        loader = build_dataloader(
            p, context_len=4, batch_size=4, shuffle=False, drop_last=False,
        )

        x, _ = next(loader)
        starts = x[:, 0].tolist()
        assert starts == [0, 4, 8]


def test_eval_loader_uses_last_valid_start_when_available():
    """With 17 tokens and context_len=4, start 12 IS valid."""
    with tempfile.TemporaryDirectory() as tmp:
        p = Path(tmp) / "seq.bin"
        _write_tokens(p, list(range(17)))
        loader = build_dataloader(
            p, context_len=4, batch_size=4, shuffle=False, drop_last=False,
        )

        x, y = next(loader)
        starts = x[:, 0].tolist()
        assert starts == [0, 4, 8, 12]
        assert y[3].tolist() == [13, 14, 15, 16]


def test_train_loader_with_stride_1_covers_every_position():
    """With stride=1, sequential reads should visit every valid start."""
    with tempfile.TemporaryDirectory() as tmp:
        p = Path(tmp) / "seq.bin"
        _write_tokens(p, list(range(16)))
        loader = build_dataloader(
            p, context_len=4, batch_size=4, shuffle=False, stride=1,
            drop_last=False,
        )

        seen_starts = set()
        for _ in range(3):
            x, _ = next(loader)
            for row in x:
                seen_starts.add(int(row[0].item()))
        assert seen_starts == set(range(12))