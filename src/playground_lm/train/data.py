# src/playground_lm/train/data.py
# Training/validation dataloader for PlaygroundLM.
#
# Consumes a flat uint16 binary file of token IDs (Nova's pipeline produces
# this: tokenize the corpus with get_tokenizer(), concatenate, write to
# train.bin / val.bin as uint16 — GPT-2 BPE vocab is 50257, which fits in
# uint16 up to 65535).
#
# Returns a batch iterator of (x, y) where:
#   x: LongTensor [batch_size, context_len]
#   y: LongTensor [batch_size, context_len]  (x shifted left by 1 token)
#
# Memory: the file is memory-mapped, so we never load the corpus into RAM.
# The loader does NOT precompute an array of start positions — at 2.5B tokens
# that would be ~20 GB. Starts are generated per batch on the fly:
#   - shuffle=True:  rng.integers(0, max_start, size=batch_size) — random windows
#   - shuffle=False: np.arange(pos, pos+batch_size, stride) — sequential windows
#
# Stride semantics:
#   - stride=1 (train): every token position is a valid start. Adjacent
#     windows overlap by context_len-1 tokens. Use with shuffle=True.
#   - stride=context_len (eval): non-overlapping windows. Each token
#     contributes to exactly one eval batch, so val_loss is an unbiased
#     estimate over the held-out file. Use with shuffle=False.
#   The caller picks the stride; default is stride=context_len (eval-safe).
#
# Resume semantics: pass `seed_offset=start_step` when resuming a run. Without
# it, a resumed session replays the same sampled windows as the previous
# session's first N batches (same seed, same RNG stream). The offset shifts
# the RNG seed per session, keeping sampling deterministic but non-repeating
# across resumes.
#
# Several files (Claude, for Stage C's 8 data-prep outputs): `path` may also
# be a list of .bin files. Each file keeps its own memmap; a window always
# lies inside one file (never spans the join between two). With shuffle=True
# every valid window across all files is equally likely, i.e. each file is
# picked in proportion to its number of valid start positions
# (num_tokens - context_len), which is its token count to within one window.
# Implementation: one rng.integers draw over the total number of windows,
# then a searchsorted into the per-file cumulative counts, so a single file
# (or a one-element list) gives exactly the same batches as before. Pass the
# files in a fixed order (e.g. sorted): the order is part of the sampling,
# so resumes are reproducible only if it doesn't change. Sequential mode
# walks the files in order. DDP sharding is unchanged (per-rank seed_offset).

import os
from pathlib import Path
from typing import Iterator, Sequence

import numpy as np
import torch


def build_dataloader(
    path: str | os.PathLike | Sequence[str | os.PathLike],
    context_len: int,
    batch_size: int,
    shuffle: bool = True,
    stride: int | None = None,
    seed: int = 42,
    seed_offset: int = 0,
    drop_last: bool = True,
) -> Iterator[tuple[torch.Tensor, torch.Tensor]]:
    """
    Build an infinite iterator of (x, y) batches from a flat uint16 token file.

    Args:
        path:        Path to a .bin file of uint16 token IDs, or a list of
                     such paths (windows are sampled across all of them,
                     never across a file boundary; see the header).
        context_len: Sequence length per sample (e.g. 1024 for GPT-2 small).
        batch_size:  Batch size per yielded batch (micro-batch, not effective).
        shuffle:     Random start positions per batch (True for train).
        stride:      Step between consecutive start positions in non-shuffled
                     mode. None means context_len (non-overlapping, eval-safe).
                     Pass 1 for training when shuffle=True (stride irrelevant
                     on the shuffle path, kept for symmetry).
        seed:        Base RNG seed for shuffle.
        seed_offset: Added to `seed` before RNG init. Pass `start_step` on
                     resume to avoid replaying the same window order.
        drop_last:   Drop final incomplete batch (True for train to keep
                     gradient accumulation exact, False for eval to use all
                     tokens).

    Yields:
        (x, y) batch tuples, both LongTensor [batch_size, context_len].
    """
    if isinstance(path, (str, os.PathLike)):
        paths = [Path(path)]
    else:
        paths = [Path(p) for p in path]
        if not paths:
            raise ValueError("path list is empty")
    for p in paths:
        if not p.exists():
            raise FileNotFoundError(f"token file not found: {p}")
    if len(paths) > 1:
        return _build_multi_file_loader(
            paths, context_len, batch_size, shuffle, stride, seed, seed_offset, drop_last
        )
    path = paths[0]
    if context_len <= 0:
        raise ValueError(f"context_len must be positive, got {context_len}")
    if batch_size <= 0:
        raise ValueError(f"batch_size must be positive, got {batch_size}")

    effective_stride = stride if stride is not None else context_len
    if effective_stride <= 0:
        raise ValueError(f"stride must be positive, got {effective_stride}")

    file_size = path.stat().st_size
    if file_size % 2 != 0:
        raise ValueError(
            f"token file size {file_size} is not a multiple of 2 — "
            f"expected uint16 array of token IDs"
        )

    num_tokens = file_size // 2
    if num_tokens < context_len + 1:
        raise ValueError(
            f"token file has {num_tokens} tokens, need at least "
            f"{context_len + 1} for one sample of context_len {context_len}"
        )

    # np.memmap gives zero-copy reads from disk.
    tokens = np.memmap(path, dtype=np.uint16, mode="r")

    # Valid start positions: a sample needs context_len+1 tokens.
    #   start from s: x = tokens[s : s+context_len]
    #                 y = tokens[s+1 : s+context_len+1]
    # So s ranges over [0, num_tokens - context_len - 1].
    max_start = num_tokens - context_len

    # Shuffle path: RNG seeded with (seed + seed_offset). Non-shuffle path
    # ignores RNG entirely.
    rng = np.random.default_rng(seed + seed_offset) if shuffle else None

    def generator() -> Iterator[tuple[torch.Tensor, torch.Tensor]]:
        pos = 0  # for sequential mode

        while True:
            if rng is not None:
                # Random starts with replacement. At 2.5B possible starts,
                # a collision within a batch of 12 is negligible; repeats are
                # harmless for training. O(batch_size) memory per call.
                starts = rng.integers(0, max_start, size=batch_size)
                yield _batch_from_starts(tokens, starts, context_len)
                continue

            # Sequential mode: walk through positions in stride steps.
            if pos >= max_start:
                pos = 0
            remaining_starts = (max_start - pos + effective_stride - 1) // effective_stride
            if remaining_starts <= 0:
                pos = 0
                remaining_starts = (max_start + effective_stride - 1) // effective_stride

            if remaining_starts < batch_size:
                if drop_last:
                    # Wrap and top up from the beginning.
                    tail = pos + np.arange(remaining_starts) * effective_stride
                    needed = batch_size - remaining_starts
                    head = np.arange(needed) * effective_stride
                    starts = np.concatenate([tail, head]).astype(np.int64)
                    pos = int(needed * effective_stride)
                else:
                    starts = (pos + np.arange(remaining_starts) * effective_stride).astype(np.int64)
                    pos = max_start  # force wrap next call
                    if len(starts) > 0:
                        yield _batch_from_starts(tokens, starts, context_len)
                    return
            else:
                starts = (pos + np.arange(batch_size) * effective_stride).astype(np.int64)
                pos += batch_size * effective_stride

            yield _batch_from_starts(tokens, starts, context_len)

    return generator()


def _batch_from_starts(
    tokens: np.ndarray,
    starts: np.ndarray,
    context_len: int,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Given start positions, build (x, y) batch tensors."""
    batch_size = len(starts)
    x = np.empty((batch_size, context_len), dtype=np.int64)
    y = np.empty((batch_size, context_len), dtype=np.int64)
    for i, s in enumerate(starts):
        s = int(s)
        x[i] = tokens[s : s + context_len].astype(np.int64)
        y[i] = tokens[s + 1 : s + context_len + 1].astype(np.int64)
    return torch.from_numpy(x), torch.from_numpy(y)


def _open_tokens(path: Path, context_len: int) -> np.ndarray:
    """Memory-map one uint16 token file, with the same checks as the single-file path."""
    file_size = path.stat().st_size
    if file_size % 2 != 0:
        raise ValueError(
            f"token file {path} size {file_size} is not a multiple of 2 — "
            f"expected uint16 array of token IDs"
        )
    num_tokens = file_size // 2
    if num_tokens < context_len + 1:
        raise ValueError(
            f"token file {path} has {num_tokens} tokens, need at least "
            f"{context_len + 1} for one sample of context_len {context_len}"
        )
    return np.memmap(path, dtype=np.uint16, mode="r")


def _build_multi_file_loader(
    paths: list[Path],
    context_len: int,
    batch_size: int,
    shuffle: bool,
    stride: int | None,
    seed: int,
    seed_offset: int,
    drop_last: bool,
) -> Iterator[tuple[torch.Tensor, torch.Tensor]]:
    """Several token files: same windows, batches and semantics as one file,
    with every window inside a single file."""
    if context_len <= 0:
        raise ValueError(f"context_len must be positive, got {context_len}")
    if batch_size <= 0:
        raise ValueError(f"batch_size must be positive, got {batch_size}")
    effective_stride = stride if stride is not None else context_len
    if effective_stride <= 0:
        raise ValueError(f"stride must be positive, got {effective_stride}")

    files = [_open_tokens(p, context_len) for p in paths]
    # Valid starts per file: [0, num_tokens - context_len), as in one file.
    starts_per_file = np.array([len(t) - context_len for t in files], dtype=np.int64)
    ends = np.cumsum(starts_per_file)  # global window index where each file stops
    begins = ends - starts_per_file
    total = int(ends[-1])

    if shuffle:
        rng = np.random.default_rng(seed + seed_offset)

        def shuffled() -> Iterator[tuple[torch.Tensor, torch.Tensor]]:
            while True:
                g = rng.integers(0, total, size=batch_size)  # uniform over all windows
                which = np.searchsorted(ends, g, side="right")
                yield _batch_from_windows(files, which, g - begins[which], context_len)

        return shuffled()

    def one_pass() -> Iterator[tuple[int, int]]:
        for i, n in enumerate(starts_per_file):
            for s in range(0, int(n), effective_stride):
                yield i, s

    def sequential() -> Iterator[tuple[torch.Tensor, torch.Tensor]]:
        pending: list[tuple[int, int]] = []
        while True:
            for window in one_pass():
                pending.append(window)
                if len(pending) == batch_size:
                    yield _windows_to_batch(files, pending, context_len)
                    pending = []
            if not drop_last:
                if pending:  # final partial batch, then stop (as with one file)
                    yield _windows_to_batch(files, pending, context_len)
                return
            # drop_last: carry the partial batch over and top it up from the start.

    return sequential()


def _windows_to_batch(files, windows, context_len):
    which = np.array([w[0] for w in windows], dtype=np.int64)
    starts = np.array([w[1] for w in windows], dtype=np.int64)
    return _batch_from_windows(files, which, starts, context_len)


def _batch_from_windows(
    files: list[np.ndarray],
    which: np.ndarray,
    starts: np.ndarray,
    context_len: int,
) -> tuple[torch.Tensor, torch.Tensor]:
    """(x, y) from (file index, start) pairs; each window is read from one file."""
    batch_size = len(starts)
    x = np.empty((batch_size, context_len), dtype=np.int64)
    y = np.empty((batch_size, context_len), dtype=np.int64)
    for i, (f, s) in enumerate(zip(which, starts)):
        tokens = files[int(f)]
        s = int(s)
        x[i] = tokens[s : s + context_len].astype(np.int64)
        y[i] = tokens[s + 1 : s + context_len + 1].astype(np.int64)
    return torch.from_numpy(x), torch.from_numpy(y)


# --- Without-replacement sampling (Lumi, for Claude's Stage C coverage note) ---
#
# Random windows WITH replacement cover only ~1 - 1/e ≈ 63% of the distinct
# windows when the number of samples equals the number of windows (Stage C:
# ~2.79M non-overlapping 1024-token windows in 2.857B tokens, ~29k steps x 96
# sequences). build_permuted_dataloader instead walks a fixed pseudo-random
# permutation of the non-overlapping windows, so one pass touches every window
# exactly once.
#
# The permutation is affine, window(k) = (a * k + b) mod n with gcd(a, n) = 1,
# a near n * 0.618 (golden ratio, so consecutive samples land far apart) and b
# drawn from `seed`. It needs no O(n) memory and is a pure function of the
# global sample index k, so resume and DDP are exact: the caller passes the
# index of its first sample and how far k advances per yielded batch.
# Notebook: first_sample = (start_step - sample_start_step) * accum * world *
# batch + rank * batch, sample_stride = world * batch. Later passes reuse the
# same order (k wraps mod n).

_GOLDEN = 0.6180339887498949


def _affine_multiplier(n: int) -> int:
    import math

    a = max(1, int(n * _GOLDEN))
    while math.gcd(a, n) != 1:
        a += 1
    return a % n if n > 1 else 0


def build_permuted_dataloader(
    path: str | os.PathLike,
    context_len: int,
    batch_size: int,
    first_sample: int = 0,
    sample_stride: int | None = None,
    seed: int = 42,
) -> Iterator[tuple[torch.Tensor, torch.Tensor]]:
    """Infinite (x, y) batches over non-overlapping windows of one uint16
    token file, sampled without replacement (see the comment above).

    Batch j (0-based) holds global samples first_sample + j * sample_stride
    + i for i in range(batch_size). sample_stride defaults to batch_size
    (one process); under DDP use world_size * batch_size with each rank's
    first_sample offset by rank * batch_size.
    """
    path = Path(path)
    if context_len <= 0:
        raise ValueError(f"context_len must be positive, got {context_len}")
    if batch_size <= 0:
        raise ValueError(f"batch_size must be positive, got {batch_size}")
    stride = batch_size if sample_stride is None else sample_stride
    if stride < batch_size:
        raise ValueError(f"sample_stride {stride} must be >= batch_size {batch_size}")
    if first_sample < 0:
        raise ValueError(f"first_sample must be >= 0, got {first_sample}")
    if not path.exists():
        raise FileNotFoundError(f"token file not found: {path}")
    tokens = _open_tokens(path, context_len)
    n = (len(tokens) - 1) // context_len  # windows need context_len + 1 tokens
    a = _affine_multiplier(n)
    b = int(np.random.default_rng(seed).integers(0, n))

    def generator() -> Iterator[tuple[torch.Tensor, torch.Tensor]]:
        k0 = first_sample
        while True:
            k = np.arange(k0, k0 + batch_size, dtype=np.int64) % n
            windows = (a * k + b) % n  # a, k < n < 2**31: no int64 overflow
            yield _batch_from_starts(tokens, windows * context_len, context_len)
            k0 += stride

    return generator()
