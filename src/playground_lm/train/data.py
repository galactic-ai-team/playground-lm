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
# Memory-mapped, so we can train on a corpus larger than RAM. On Kaggle with
# 13GB RAM and a 1-2GB FineWeb sample, this matters less than on a laptop,
# but it's the right pattern regardless.
#
# Determinism: seedable shuffle via a passed-in generator. Eval loader uses
# shuffle=False and sequential reads, so val loss is comparable across runs
# and across checkpoints.

import os
from pathlib import Path
from typing import Iterator, Optional

import numpy as np
import torch


def build_dataloader(
    path: str | os.PathLike,
    context_len: int,
    batch_size: int,
    shuffle: bool = True,
    seed: int = 42,
    drop_last: bool = True,
) -> Iterator[tuple[torch.Tensor, torch.Tensor]]:
    """
    Build an infinite iterator of (x, y) batches from a flat uint16 token file.

    Args:
        path:        Path to .bin file of uint16 token IDs.
        context_len: Sequence length per sample (e.g. 1024 for GPT-2 small).
        batch_size:  Batch size per yielded batch.
        shuffle:     Randomize batch order (True for train, False for val).
        seed:        RNG seed for shuffle. Irrelevant when shuffle=False.
        drop_last:   Drop final incomplete batch (True for train to keep
                     gradient accumulation exact, False for eval to use all
                     tokens).

    Yields:
        (x, y) batch tuples, both LongTensor [batch_size, context_len].

    Raises:
        FileNotFoundError: if the token file doesn't exist.
        ValueError: if the file has fewer than context_len+1 tokens, or its
                    byte length is not divisible by 2 (uint16), or batch_size
                    or context_len are non-positive.
    """
    path = Path(path)
    if not path.exists():
        raise FileNotFoundError(f"token file not found: {path}")
    if context_len <= 0:
        raise ValueError(f"context_len must be positive, got {context_len}")
    if batch_size <= 0:
        raise ValueError(f"batch_size must be positive, got {batch_size}")

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

    rng = np.random.default_rng(seed) if shuffle else None

    def generator() -> Iterator[tuple[torch.Tensor, torch.Tensor]]:
        while True:
            # Number of valid start positions: need context_len+1 tokens per
            # sample (context_len inputs + 1 target offset).
            num_samples = num_tokens - context_len  # start positions 0..num_tokens-context_len-1
            if num_samples <= 0:
                return
            if rng is not None:
                starts = rng.integers(0, num_samples, size=batch_size)
            else:
                # Sequential: cycle through the file in order.
                if not hasattr(generator, "_pos"):
                    generator._pos = 0  # type: ignore[attr-defined]
                starts = np.arange(
                    generator._pos,  # type: ignore[attr-defined]
                    min(generator._pos + batch_size, num_samples),  # type: ignore[attr-defined]
                )
                generator._pos = (generator._pos + batch_size) % num_samples  # type: ignore[attr-defined]
                if len(starts) < batch_size:
                    # Not enough samples left to fill a batch on this pass.
                    if drop_last:
                        # Wrap and take from the beginning.
                        remaining = batch_size - len(starts)
                        starts = np.concatenate([starts, np.arange(remaining)])
                    else:
                        yield _batch_from_starts(tokens, starts, context_len)
                        return

            if len(starts) < batch_size and drop_last:
                continue
            if len(starts) == 0:
                return

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
