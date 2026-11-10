"""Pretraining replay for SFT (Stage E round 3, Claude).

Mixes rows from the Stage C pretraining data (plain next-token windows, every
target supervised) into completion-only SFT batches, so fine-tuning keeps
practising the original distribution and FineWeb-Edu val loss stays within
the regression gate.

`replay_ratio` is the share of ROWS (1024-token windows) that come from
replay. Rows are assigned with a deterministic error-diffusion schedule on
each rank's row counter: after k rows, exactly floor(k * ratio) were replay.
So every optimizer step on a rank (micro_batch x grad_accum rows) gets
exactly ratio x rows replay rows whenever that product is an integer (it is
for 0.25 and 0.5 at 6 x 2 rows per rank), and individual micro-batches get
the nearest whole numbers (1 or 2 of 6 at 0.25).

Token weighting: the loss averages over supervised tokens. Replay rows
supervise every token, SFT rows only their completions, so replay's share
of the loss is larger than its share of rows (about 36% of tokens at a 0.25
row ratio and 63% at 0.5 with round 1's ~61% completion density). That is
the knob the sweep turns; it is stated here so nobody reads 0.25 as "a
quarter of the loss".

Nothing here changes SFT masking: SFT rows come from build_sft_dataloader
unchanged (G3/G4 still apply to them), replay rows from build_dataloader.
"""

from __future__ import annotations

import math
from typing import Iterator

import numpy as np
import torch


def replay_count(rows_seen: int, ratio: float) -> int:
    """Replay rows among the first `rows_seen` rows."""
    return math.floor(rows_seen * ratio + 1e-9)


def check_exact(ratio: float, rows_per_step: int) -> int:
    """Replay rows per optimizer step on one rank; raises unless whole."""
    exact = ratio * rows_per_step
    if abs(exact - round(exact)) > 1e-9:
        raise ValueError(f"replay_ratio {ratio} x {rows_per_step} rows per step is not a whole number of rows")
    return int(round(exact))


def _cat(parts):
    if hasattr(torch, "cat") and isinstance(parts[0], torch.Tensor) and not isinstance(parts[0], np.ndarray):
        return torch.cat(parts, dim=0)
    return np.concatenate(parts, axis=0)


class ReplayMixer:
    """Iterator of (x, y) micro-batches mixing SFT rows and replay rows.

    sft_rows / replay_rows: iterators yielding (x, y) with ONE row each
    (build the underlying loaders with batch_size=1).
    """

    def __init__(self, sft_rows: Iterator, replay_rows: Iterator, ratio: float, batch_size: int,
                 rows_per_step: int | None = None):
        if not 0.0 <= ratio < 1.0:
            raise ValueError(f"replay_ratio must be in [0, 1), got {ratio}")
        if batch_size <= 0:
            raise ValueError("batch_size must be positive")
        if rows_per_step is not None:
            check_exact(ratio, rows_per_step)
        self.sft_rows, self.replay_rows = sft_rows, replay_rows
        self.ratio, self.batch_size = ratio, batch_size
        self.rows = 0
        self.replay_rows_served = 0

    def __iter__(self):
        return self

    def __next__(self):
        xs, ys = [], []
        for _ in range(self.batch_size):
            replay = replay_count(self.rows + 1, self.ratio) > replay_count(self.rows, self.ratio)
            x, y = next(self.replay_rows if replay else self.sft_rows)
            xs.append(x)
            ys.append(y)
            self.rows += 1
            self.replay_rows_served += int(replay)
        return _cat(xs), _cat(ys)


def build_replay_mixed_loader(train_bin, train_masks, replay_bin, *, ratio: float, context_len: int,
                              batch_size: int, seed_offset: int = 0, rows_per_step: int | None = None):
    """SFT train loader with Stage C replay mixed in (ratio 0 -> plain SFT loader)."""
    from playground_lm.train.data import build_dataloader
    from playground_lm.train.sft_data import build_sft_dataloader

    if ratio == 0:
        return build_sft_dataloader(train_bin, train_masks, context_len=context_len, batch_size=batch_size,
                                    shuffle=True, seed_offset=seed_offset)
    sft = build_sft_dataloader(train_bin, train_masks, context_len=context_len, batch_size=1,
                               shuffle=True, seed_offset=seed_offset)
    replay = build_dataloader(replay_bin, context_len=context_len, batch_size=1, shuffle=True, stride=1,
                              seed=4242, seed_offset=seed_offset)
    return ReplayMixer(sft, replay, ratio, batch_size, rows_per_step)


# ---- Stage F: three sources (fact/arith SFT, chat SFT, FineWeb replay) ---------------

def step_weights(shares: list[float], rows_per_step: int) -> list[int]:
    """Whole rows per optimizer step for each source; raises unless exact."""
    if abs(sum(shares) - 1.0) > 1e-9 or any(s < 0 for s in shares):
        raise ValueError(f"shares must be non-negative and sum to 1, got {shares}")
    weights = [check_exact(s, rows_per_step) for s in shares]
    if sum(weights) != rows_per_step:
        raise ValueError("shares do not split the step into whole rows")
    return weights


class MultiSourceMixer:
    """Micro-batches drawing each row from one of several one-row iterators.

    Rows follow a smooth weighted round-robin over each rank's optimizer step,
    so every step (rows_per_step rows) gets exactly share x rows_per_step rows
    from each source, spread evenly across its micro-batches. Each session
    starts on a step boundary, so resumes keep the same pattern.
    """

    def __init__(self, sources: list[Iterator], shares: list[float], batch_size: int, rows_per_step: int):
        if len(sources) != len(shares) or not sources:
            raise ValueError("need one share per source")
        if batch_size <= 0 or rows_per_step % batch_size:
            raise ValueError("rows_per_step must be a positive multiple of batch_size")
        self.sources = sources
        self.weights = step_weights(shares, rows_per_step)
        self.batch_size, self.total = batch_size, rows_per_step
        self.current = [0] * len(sources)
        self.served = [0] * len(sources)

    def __iter__(self):
        return self

    def _pick(self) -> int:
        for i, w in enumerate(self.weights):
            self.current[i] += w
        best = max(range(len(self.current)), key=lambda i: (self.current[i], -i))
        self.current[best] -= self.total
        return best

    def __next__(self):
        xs, ys = [], []
        for _ in range(self.batch_size):
            i = self._pick()
            x, y = next(self.sources[i])
            xs.append(x)
            ys.append(y)
            self.served[i] += 1
        return _cat(xs), _cat(ys)


def build_stage_f_mixed_loader(facts_bin, facts_masks, chat_bin, chat_masks, replay_bin, *, replay_ratio: float,
                               chat_ratio: float, context_len: int, batch_size: int, rows_per_step: int,
                               seed_offset: int = 0):
    """Fact/arith SFT + chat SFT (both completion-only masks) + FineWeb replay (full loss)."""
    from playground_lm.train.data import build_dataloader
    from playground_lm.train.sft_data import build_sft_dataloader

    facts = build_sft_dataloader(facts_bin, facts_masks, context_len=context_len, batch_size=1,
                                 shuffle=True, seed=42, seed_offset=seed_offset)
    chat = build_sft_dataloader(chat_bin, chat_masks, context_len=context_len, batch_size=1,
                                shuffle=True, seed=4343, seed_offset=seed_offset)
    replay = build_dataloader(replay_bin, context_len=context_len, batch_size=1, shuffle=True, stride=1,
                              seed=4242, seed_offset=seed_offset)
    shares = [1.0 - replay_ratio - chat_ratio, chat_ratio, replay_ratio]
    return MultiSourceMixer([facts, chat, replay], shares, batch_size, rows_per_step)
