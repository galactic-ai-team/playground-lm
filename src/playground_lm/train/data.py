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
# Memory-mapped, so we can train on a corpus larger than RAM.
#
# Stride semantics:
#   - stride=1 (train): every token position is a valid start. Adjacent
#     windows overlap by context_len-1 tokens. This is how LM pretraining
#     works — every token becomes the target of *some* prediction. Use with
#     shuffle=True.
#   - stride=context_len (eval): non-overlapping windows. Each token
#     contributes to exactly one eval batch, so val_loss is an unbiased
#     estimate over the held-out file. Use with shuffle=False.
#
#   The caller picks the stride; the loader doesn't guess. Default is
#   stride=context_len (safe eval default); training code should pass
#   stride=1 explicitly.

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
    drop_last: bool = True,
) -> Iterator[tuple[torch.Tensor, torch.Tensor]]:
    """
    Build an infinite iterator of (x, y) batches from a flat uint16 token file.

    Args:
        path:        Path to .bin file of uint16 token IDs.
        context_len: Sequence length per sample (e.g. 1024 for GPT-2 small).
        batch_size:  Batch size per yielded batch.
        shuffle:     Randomize start positions per batch (True for train).
        stride:      Step between consecutive start positions in non-shuffled
                     mode. None means context_len (non-overlapping, eval-safe).
                     Pass 1 for training when shuffle=True.
        seed:        RNG seed for shuffle. Irrelevant when shuffle=False.
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
    # List of valid starts spaced by effective_stride.
    all_starts = np.arange(0, max_start, effective_stride, dtype=np.int64)

    rng = np.random.default_rng(seed) if shuffle else None

    def generator() -> Iterator[tuple[torch.Tensor, torch.Tensor]]:
        pos = 0  # for sequential mode

        while True:
            if rng is not None:
                # Random starts with replacement (train mode).
                starts = rng.choice(all_starts, size=batch_size, replace=True)
                yield _batch_from_starts(tokens, starts, context_len)
                continue

            # Sequential mode: walk through all_starts in order.
            remaining = len(all_starts) - pos
            if remaining <= 0:
                # Wrapped around: restart from 0 for the next pass.
                pos = 0
                remaining = len(all_starts)

            if remaining < batch_size:
                if drop_last:
                    # Not enough to fill the batch — wrap and top up.
                    tail = all_starts[pos:]
                    head = all_starts[: batch_size - len(tail)]
                    starts = np.concatenate([tail, head])
                    pos = batch_size - len(tail)
                else:
                    # Yield the partial batch, then stop.
                    starts = all_starts[pos:]
                    pos = len(all_starts)
                    if len(starts) > 0:
                        yield _batch_from_starts(tokens, starts, context_len)
                    return
            else:
                starts = all_starts[pos : pos + batch_size]
                pos += batch_size

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
