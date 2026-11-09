"""Masked SFT dataloader for completion-only fine-tuning.

Consumes paired files:
  tokens.bin  — uint16 GPT-2 token ids (packed examples)
  masks.bin   — uint8 flags aligned with tokens (1 = supervised / completion)

Batches are contiguous windows. For next-token prediction, y[t] receives
tokens[t+1] when masks[t+1] == 1, otherwise IGNORE_INDEX (-100).
"""

from __future__ import annotations

import os
from pathlib import Path
from typing import Iterator

import numpy as np
import torch

from playground_lm.model import IGNORE_INDEX


def build_sft_dataloader(
    tokens_path: str | os.PathLike,
    masks_path: str | os.PathLike,
    context_len: int,
    batch_size: int,
    shuffle: bool = True,
    seed: int = 42,
    seed_offset: int = 0,
    drop_last: bool = True,
) -> Iterator[tuple[torch.Tensor, torch.Tensor]]:
    tokens_path = Path(tokens_path)
    masks_path = Path(masks_path)
    if not tokens_path.is_file():
        raise FileNotFoundError(tokens_path)
    if not masks_path.is_file():
        raise FileNotFoundError(masks_path)
    if context_len <= 0 or batch_size <= 0:
        raise ValueError("context_len and batch_size must be positive")

    n_tokens = tokens_path.stat().st_size // 2
    n_masks = masks_path.stat().st_size
    if tokens_path.stat().st_size % 2:
        raise ValueError(f"{tokens_path}: size not multiple of 2")
    if n_masks != n_tokens:
        raise ValueError(f"mask/token length mismatch: {n_masks} vs {n_tokens}")
    if n_tokens < context_len + 1:
        raise ValueError(f"need >{context_len} tokens, got {n_tokens}")

    tokens = np.memmap(tokens_path, dtype=np.uint16, mode="r")
    masks = np.memmap(masks_path, dtype=np.uint8, mode="r")
    max_start = n_tokens - (context_len + 1)
    rng = np.random.default_rng(seed + seed_offset)
    cursor = 0

    while True:
        if shuffle:
            starts = rng.integers(0, max_start + 1, size=batch_size)
        else:
            if cursor > max_start:
                if drop_last and cursor + batch_size - 1 > max_start:
                    cursor = 0
                elif cursor > max_start:
                    cursor = 0
            starts = np.arange(cursor, cursor + batch_size)
            if starts[-1] > max_start:
                if drop_last:
                    cursor = 0
                    continue
                starts = starts[starts <= max_start]
            cursor = int(starts[-1]) + context_len

        xs = np.empty((len(starts), context_len), dtype=np.int64)
        ys = np.empty((len(starts), context_len), dtype=np.int64)
        for i, start in enumerate(starts):
            window = tokens[start : start + context_len + 1].astype(np.int64)
            mask_w = masks[start : start + context_len + 1]
            xs[i] = window[:-1]
            targets = window[1:].copy()
            # Supervise only when the predicted token is marked completion.
            targets[mask_w[1:] == 0] = IGNORE_INDEX
            ys[i] = targets
        yield torch.from_numpy(xs), torch.from_numpy(ys)
