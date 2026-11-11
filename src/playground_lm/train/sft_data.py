"""Masked SFT dataloader for completion-only fine-tuning (Stage E).

Consumes paired files written by data/build_stage_e_sft.py:
  tokens.bin  — uint16 GPT-2 token ids, examples packed back to back
  masks.bin   — uint8 flags aligned with tokens: 1 = completion (assistant
                text incl. its closing <|endoftext|>), 0 = prompt

Each example is one prompt run (mask 0) followed by one completion run
(mask 1) that ends with EOT:

    User: ...\\nAssistant:| ...answer...<|endoftext|>
    0 0 0 ... 0 0 0      | 1 1 1 ... 1 1

Targets (next-token prediction, shift by one):
    x[t] = tokens[s + t]            t = 0 .. context_len-1
    y[t] = tokens[s + t + 1]        if that token is a completion token whose
                                    example's prompt starts inside the window
         = IGNORE_INDEX (-100)      otherwise (prompt, padding, or an example
                                    whose prompt was cut off by the window)

Audit (Claude): before any batch is produced, both files are checked once:
mask values are 0/1, token ids are < vocab_size, every completion run ends
with EOT and follows a non-empty prompt, no completion is EOT-only (empty),
no EOT sits inside a prompt, and the file doesn't end in a dangling prompt.
Malformed boundaries raise SFTDataError (strict=True, the default) or are
counted and logged (strict=False). Only counts are logged, never text.

Truncation: a completion token is only supervised when its prompt starts at
or after the window start, so the model never learns to answer an
instruction it can't see. Training windows start at example starts (so
nothing is lost at the front); tokens more than context_len past their
example's start can never be supervised and are reported as

`unsupervisable_tokens`. Validation windows are laid out greedily from
example starts so one pass scores every example exactly once.

Padding: windows are cut from the packed stream, so there is no pad token;
anything in a window that is not a visible completion token (prompts, the
tail of a previous example, the head of the next one) is IGNORE_INDEX.

Shapes: every batch is (batch_size, context_len) int64 for x and y, except
that a final partial validation batch (drop_last=False) has fewer rows.
"""

from __future__ import annotations

import logging
import os
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Callable, Iterator

import numpy as np
import torch

from playground_lm.model import IGNORE_INDEX

logger = logging.getLogger(__name__)

EOT_ID = 50256
GPT2_VOCAB_SIZE = 50257


class SFTDataError(ValueError):
    """The token/mask pair is malformed (raised before any batch is built)."""



@dataclass(frozen=True)
class SFTAudit:
    """Counts only; safe to log (contains no token ids or text)."""

    tokens: int
    supervised_tokens: int
    examples: int
    oversized_examples: int          # prompt + completion longer than context_len + 1
    unsupervisable_tokens: int       # completion tokens beyond context_len from their example start
    empty_completions: int           # completion run is just EOT
    completions_missing_eot: int     # completion run doesn't end with EOT
    completions_missing_prompt: int  # completion run with no prompt right before it
    eot_in_prompt: int               # EOT token with mask 0
    dangling_prompt_tokens: int      # prompt tokens after the last completion
    max_token_id: int

    @property
    def malformed(self) -> int:
        return (self.empty_completions + self.completions_missing_eot
                + self.completions_missing_prompt + self.eot_in_prompt
                + int(self.dangling_prompt_tokens > 0))

    @property
    def supervisable_fraction(self) -> float:
        """Share of completion tokens that some window can supervise."""
        if self.supervised_tokens == 0:
            return 0.0
        return 1.0 - self.unsupervisable_tokens / self.supervised_tokens

    def summary(self) -> str:

        return (
            f"sft audit: {self.examples:,} examples, {self.tokens:,} tokens, "
            f"{self.supervised_tokens:,} completion ({self.supervised_tokens / max(1, self.tokens):.1%}), "
            f"supervisable {self.supervisable_fraction:.1%}, oversized {self.oversized_examples}, "
            f"malformed {self.malformed} (empty {self.empty_completions}, no-EOT "
            f"{self.completions_missing_eot}, no-prompt {self.completions_missing_prompt}, "
            f"EOT-in-prompt {self.eot_in_prompt}, dangling-prompt-tokens {self.dangling_prompt_tokens})"
        )

    def to_dict(self) -> dict:
        return {**asdict(self), "malformed": self.malformed,
                "supervisable_fraction": self.supervisable_fraction}


@dataclass(frozen=True)
class _Layout:
    starts: np.ndarray  # example start: first prompt token (or first completion token if no prompt)
    ends: np.ndarray    # example end: last completion token, inclusive


def _runs(flags: np.ndarray, value: int) -> tuple[np.ndarray, np.ndarray]:
    """(starts, ends) of maximal runs where flags == value; ends inclusive."""
    hit = (flags == value).astype(np.int8)
    edges = np.diff(np.concatenate(([0], hit, [0])))
    return np.flatnonzero(edges == 1), np.flatnonzero(edges == -1) - 1


def audit_sft_arrays(
    tokens: np.ndarray,
    masks: np.ndarray,
    context_len: int,
    eot_id: int = EOT_ID,

    vocab_size: int = GPT2_VOCAB_SIZE,
    strict: bool = True,
) -> tuple[SFTAudit, _Layout]:
    """Validate a token/mask pair and lay out its examples.

    Hard errors (always raised): length mismatch, mask values other than
    0/1, token ids >= vocab_size, no completion tokens at all, fewer than
    context_len + 1 tokens. Boundary problems (empty completion, completion
    without EOT or without a prompt, EOT inside a prompt, trailing prompt
    with no completion) raise when strict, otherwise they're counted.
    """
    tokens = np.asarray(tokens)
    masks = np.asarray(masks)
    if tokens.ndim != 1 or masks.ndim != 1:
        raise SFTDataError("tokens and masks must be 1-D")
    if tokens.shape[0] != masks.shape[0]:
        raise SFTDataError(f"mask/token length mismatch: {masks.shape[0]} vs {tokens.shape[0]}")
    if context_len <= 0:
        raise SFTDataError("context_len must be positive")
    n = int(tokens.shape[0])
    if n < context_len + 1:
        raise SFTDataError(f"need more than {context_len} tokens, got {n}")
    if int(masks.max()) > 1:
        raise SFTDataError(f"mask values must be 0 or 1; {int(np.count_nonzero(masks > 1))} are not")
    max_token = int(tokens.max())
    if max_token >= vocab_size:
        raise SFTDataError(f"{int(np.count_nonzero(tokens >= vocab_size))} token ids >= vocab_size {vocab_size}")

    c_starts, c_ends = _runs(masks, 1)
    if c_starts.size == 0:
        raise SFTDataError("no completion tokens (mask has no 1s)")
    p_starts, p_ends = _runs(masks, 0)


    is_eot = tokens == eot_id
    missing_eot = int(np.count_nonzero(~is_eot[c_ends]))
    empty = int(np.count_nonzero((c_starts == c_ends) & is_eot[c_ends]))
    # An EOT inside a completion run (not at its end) means two completions
    # with no prompt between them; a completion at position 0 has no prompt.
    merged = int(np.count_nonzero(is_eot & (masks == 1)) - np.count_nonzero(is_eot[c_ends]))
    missing_prompt = merged + int(c_starts[0] == 0)
    eot_in_prompt = int(np.count_nonzero(is_eot & (masks == 0)))
    dangling = int(n - 1 - c_ends[-1])

    # Runs alternate, so every completion run not at position 0 has a prompt
    # run ending right before it; the example starts where that prompt starts.
    ex_starts = c_starts.copy()
    has_prompt = c_starts > 0
    ex_starts[has_prompt] = p_starts[np.searchsorted(p_ends, c_starts[has_prompt] - 1)]
    ex_ends = c_ends
    oversized = int(np.count_nonzero(ex_ends - ex_starts + 1 > context_len + 1))
    # A target at position p can only be supervised from a window starting at
    # its example start s, i.e. if p <= s + context_len.
    reach = ex_starts + context_len
    unsupervisable = int(np.sum(np.clip(ex_ends - np.maximum(reach, c_starts - 1), 0, None)))

    audit = SFTAudit(
        tokens=n,
        supervised_tokens=int(np.count_nonzero(masks)),
        examples=int(c_starts.size),
        oversized_examples=oversized,
        unsupervisable_tokens=unsupervisable,
        empty_completions=empty,
        completions_missing_eot=missing_eot,
        completions_missing_prompt=missing_prompt,

        eot_in_prompt=eot_in_prompt,
        dangling_prompt_tokens=dangling,
        max_token_id=max_token,
    )
    if strict and audit.malformed:
        raise SFTDataError("malformed example boundaries; " + audit.summary())
    return audit, _Layout(starts=ex_starts.astype(np.int64), ends=ex_ends.astype(np.int64))


def _open_pair(tokens_path: Path, masks_path: Path) -> tuple[np.ndarray, np.ndarray]:
    if not tokens_path.is_file():
        raise FileNotFoundError(tokens_path)
    if not masks_path.is_file():
        raise FileNotFoundError(masks_path)
    if tokens_path.stat().st_size % 2:
        raise SFTDataError(f"{tokens_path.name}: size is not a multiple of 2 (uint16 tokens)")
    tokens = np.memmap(tokens_path, dtype=np.uint16, mode="r")
    masks = np.memmap(masks_path, dtype=np.uint8, mode="r")
    return tokens, masks


def audit_sft_files(
    tokens_path: str | os.PathLike,
    masks_path: str | os.PathLike,
    context_len: int,
    **kwargs,
) -> SFTAudit:
    """audit_sft_arrays on a pair of files (for pre-launch checks and logging)."""
    tokens, masks = _open_pair(Path(tokens_path), Path(masks_path))
    return audit_sft_arrays(tokens, masks, context_len, **kwargs)[0]



def window_targets(
    tokens: np.ndarray,
    masks: np.ndarray,
    example_starts: np.ndarray,
    start: int,
    context_len: int,
    examples: tuple[int, int] | None = None,
) -> tuple[np.ndarray, np.ndarray]:
    """(x, y) for one window starting at `start`, with completion-only targets.

    A completion token is supervised only if its example is in `examples`
    (a [lo, hi) range of example indices). The default is every example that
    starts at or after the window start, i.e. whose prompt is fully visible;
    validation passes a narrower range so no example is scored twice.
    """
    window = np.asarray(tokens[start : start + context_len + 1], dtype=np.int64)
    if window.shape[0] != context_len + 1:
        raise SFTDataError(f"window at {start} runs past the end of the data")
    if examples is None:
        examples = (int(np.searchsorted(example_starts, start, side="left")), example_starts.size)
    lo, hi = examples
    positions = np.arange(start + 1, start + context_len + 1)
    owner = np.searchsorted(example_starts, positions, side="right") - 1
    allowed = (owner >= lo) & (owner < hi)
    supervised = (np.asarray(masks[start + 1 : start + context_len + 1]) == 1) & allowed
    return window[:-1], np.where(supervised, window[1:], IGNORE_INDEX)


def _eval_windows(layout: _Layout, context_len: int, max_start: int) -> list[tuple[int, int, int]]:
    """Deterministic validation windows as (start, lo, hi): each window starts
    at the first example not yet scored and scores the examples [lo, hi) that
    fit entirely inside it, so one pass scores every example exactly once.

    An example longer than a window gets one window from its own start (its
    remaining tokens are counted as unsupervisable in the audit). Near the end
    of the data a window may start earlier than its first example; the
    examples before it are already scored and stay masked."""
    windows: list[tuple[int, int, int]] = []
    i, n = 0, layout.starts.size
    while i < n:
        s = min(int(layout.starts[i]), max_start)
        last = s + context_len  # last predicted position in this window
        j = i
        while j < n and layout.ends[j] <= last:
            j += 1
        hi = max(j, i + 1)
        windows.append((s, i, hi))
        i = hi
    return windows


def build_sft_dataloader(
    tokens_path: str | os.PathLike,
    masks_path: str | os.PathLike,
    context_len: int,
    batch_size: int,
    shuffle: bool = True,
    seed: int = 42,
    seed_offset: int = 0,
    drop_last: bool = True,
    *,
    eot_id: int = EOT_ID,
    vocab_size: int = GPT2_VOCAB_SIZE,
    strict: bool = True,
    log: Callable[[str], None] | None = None,
    answer_weight: float = 1.0,
    answer_mode: str | None = None,
    answer_marker: tuple[int, ...] | None = None,
) -> Iterator[tuple[torch.Tensor, torch.Tensor]]:
    """Infinite iterator of (x, y) int64 batches with completion-only targets.

    shuffle=True (train): each row is a window starting at a random example
    start (seeded by seed + seed_offset; pass start_step on resume, plus a
    per-rank offset under DDP). shuffle=False (val): example-aligned windows
    in order, repeating; drop_last=False yields a final short batch.
    The pair is audited before the first batch: malformed data raises
    SFTDataError here, not mid-training. The audit summary (counts only)
    goes to `log` (default: this module's logger at INFO).

    Optional answer-token loss weight (Stage G, default off): with
    answer_weight != 1.0 and an answer_mode, batches are (x, y, w) where w is
    a float32 weight per target (see answer_weights). With the default 1.0
    the loader yields exactly the (x, y) batches it always has.
    """
    if context_len <= 0 or batch_size <= 0:
        raise ValueError("context_len and batch_size must be positive")
    tokens, masks = _open_pair(Path(tokens_path), Path(masks_path))
    audit, layout = audit_sft_arrays(tokens, masks, context_len, eot_id, vocab_size, strict)
    (log or logger.info)(f"{Path(tokens_path).name}: {audit.summary()}")
    max_start = int(tokens.shape[0]) - (context_len + 1)
    batches = _batches(tokens, masks, layout, context_len, batch_size, shuffle,
                       seed + seed_offset, drop_last, max_start)
    if answer_weight == 1.0 or answer_mode is None:
        return batches
    if answer_mode not in ANSWER_MODES:
        raise ValueError(f"answer_mode must be one of {ANSWER_MODES}, got {answer_mode!r}")
    if answer_mode == "after_marker" and not answer_marker:
        raise ValueError("answer_mode 'after_marker' needs answer_marker token ids")
    if not (answer_weight > 0 and np.isfinite(answer_weight)):
        raise ValueError("answer_weight must be positive")

    def weighted():
        for x, y in batches:
            w = answer_weights(np.asarray(x), np.asarray(y), answer_mode, answer_weight, answer_marker, eot_id)
            yield x, y, torch.from_numpy(w)

    return weighted()


ANSWER_MODES = ("after_marker", "completion")


def answer_weights(x: np.ndarray, y: np.ndarray, mode: str, weight: float,
                   marker: tuple[int, ...] | None = None, eot_id: int = EOT_ID) -> np.ndarray:
    """Per-target loss weights (float32, same shape as y): `weight` on answer
    targets, 1.0 elsewhere (ignored targets don't count either way).

    mode "completion": every supervised target is an answer (grounded-facts
    rows, whose whole completion is the short answer or the not-sure line).
    mode "after_marker": targets after the marker token sequence (e.g. the
    tokens of " Answer:") up to and including the next EOT, i.e. the final
    answer of a scratchpad row, not its worked steps.
    """
    w = np.ones(y.shape, dtype=np.float32)
    supervised = y != IGNORE_INDEX
    if mode == "completion":
        w[supervised] = weight
        return w
    k = len(marker)
    for r in range(x.shape[0]):
        # Window tokens: x plus the final target (x[t+1] == y[t] where supervised).
        tok = np.append(x[r], y[r, -1] if supervised[r, -1] else -1)
        hits = [i + k - 1 for i in range(tok.size - k + 1) if tuple(tok[i : i + k]) == tuple(marker)]
        for e in hits:
            eots = np.flatnonzero(tok[e + 1 :] == eot_id)
            q = e + 1 + int(eots[0]) if eots.size else tok.size - 1
            t = np.arange(e, q)  # targets predicting tokens e+1 .. q
            t = t[t < y.shape[1]]
            w[r, t[supervised[r, t]]] = weight
    return w


def _batches(tokens, masks, layout, context_len, batch_size, shuffle, seed, drop_last, max_start):
    def build(specs) -> tuple[torch.Tensor, torch.Tensor]:
        xs = np.empty((len(specs), context_len), dtype=np.int64)
        ys = np.empty((len(specs), context_len), dtype=np.int64)
        for row, (s, examples) in enumerate(specs):
            xs[row], ys[row] = window_targets(tokens, masks, layout.starts, int(s), context_len, examples)
        return torch.from_numpy(xs), torch.from_numpy(ys)

    if shuffle:
        candidates = np.unique(np.minimum(layout.starts, max_start))
        rng = np.random.default_rng(seed)

        while True:
            yield build([(s, None) for s in rng.choice(candidates, size=batch_size, replace=True)])

    windows = [(s, (lo, hi)) for s, lo, hi in _eval_windows(layout, context_len, max_start)]
    pos = 0
    while True:
        if pos + batch_size <= len(windows):
            chunk = windows[pos : pos + batch_size]
            pos += batch_size
        elif drop_last:
            need = batch_size - (len(windows) - pos)
            chunk = windows[pos:] + [windows[k % len(windows)] for k in range(need)]
            pos = need % len(windows)
        else:
            chunk, pos = windows[pos:], 0
            if not chunk:
                continue
        yield build(chunk)