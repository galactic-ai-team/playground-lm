# src/playground_lm/train/test_data.py
# Tests for the dataloader.
#
# Two distinct semantics under test:
#   - stride=1 (train): every token position is a start; adjacent windows overlap.
#   - stride=context_len (eval): non-overlapping; each token in exactly one batch.

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
    """Train mode (stride=1): starts advance by 1, windows overlap.
    y is x shifted left by 1 token — the fundamental invariant."""
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

        # stride=1 sequential: x[0] = [0,1,2,3], x[1] = [1,2,3,4].
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
        # stride=4 sequential: x[0] = [0,1,2,3], x[1] = [4,5,6,7].
        assert x[0].tolist() == [0, 1, 2, 3]
        assert y[0].tolist() == [1, 2, 3, 4]
        assert x[1].tolist() == [4, 5, 6, 7]
        assert y[1].tolist() == [5, 6, 7, 8]


def test_shuffle_produces_different_order():
    with tempfile.TemporaryDirectory() as tmp:
        p = Path(tmp) / "seq.bin"
        _write_tokens(p, list(range(1000)))

        a = next(build_dataloader(p, 8, 4, shuffle=True, stride=1, seed=1))
        b = next(build_dataloader(p, 8, 4, shuffle=True, stride=1, seed=1))
        c = next(build_dataloader(p, 8, 4, shuffle=True, stride=1, seed=2))

        assert torch.equal(a[0], b[0])
        assert torch.equal(a[1], b[1])
        assert not torch.equal(a[0], c[0])


def test_shuffle_false_is_deterministic():
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
            x, y = next(loader)
            assert x.shape == (2, 4)


def test_eval_loader_no_shuffle_uses_all_tokens():
    """With stride=context_len, sequential iteration visits every token
    exactly once over enough batches — no overlap, no gaps."""
    with tempfile.TemporaryDirectory() as tmp:
        p = Path(tmp) / "seq.bin"
        _write_tokens(p, list(range(16)))  # 16 tokens, ctx 4 -> starts 0,4,8,12
        loader = build_dataloader(
            p, context_len=4, batch_size=4, shuffle=False, drop_last=False,
        )

        x, _ = next(loader)
        # Should have exactly the 4 non-overlapping starts.
        starts = x[:, 0].tolist()
        assert starts == [0, 4, 8, 12]


def test_train_loader_with_stride_1_covers_every_position():
    """With stride=1, sequential reads should visit every valid start."""
    with tempfile.TemporaryDirectory() as tmp:
        p = Path(tmp) / "seq.bin"
        _write_tokens(p, list(range(16)))  # 16 tokens, ctx 4 -> starts 0..11
        loader = build_dataloader(
            p, context_len=4, batch_size=4, shuffle=False, stride=1,
            drop_last=False,
        )

        seen_starts = set()
        for _ in range(3):  # 12 valid starts / batch 4 = 3 batches
            x, _ = next(loader)
            for row in x:
                seen_starts.add(int(row[0].item()))
        assert seen_starts == set(range(12))
