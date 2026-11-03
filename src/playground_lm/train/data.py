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

import os
from pathlib import Path
from typing import Iterator

import numpy as np
import torch


def build_dataloader(
    path: str | os.PathLike,
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
        path:        Path to .bin file of uint16 token IDs.
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
    path = Path(path)
    if not path.exists():
        raise FileNotFoundError(f"token file not found: {path}")
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