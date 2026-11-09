"""Completion-only masking: shift, boundaries, truncation, padding, coverage, invalid inputs (Claude).

Synthetic examples only (no corpus or held-out eval text). Token ids encode
their role so targets can be checked by value: prompts use 100+, completions
500+, and EOT is 999 in a 1000-token vocabulary.
"""

import numpy as np
import pytest

from playground_lm.model import IGNORE_INDEX
from playground_lm.train.sft_data import (
    SFTDataError,
    _eval_windows,
    audit_sft_arrays,
    audit_sft_files,
    build_sft_dataloader,
    window_targets,
)

EOT, VOCAB = 999, 1000
KW = dict(eot_id=EOT, vocab_size=VOCAB)


def pack(*examples):
    """examples: (prompt_len, completion_len_without_eot) -> (tokens, masks, starts)."""
    toks, masks, starts, pos = [], [], [], 0
    for i, (p, c) in enumerate(examples):
        starts.append(pos)
        toks += [100 + (10 * i + j) % 300 for j in range(p)]
        toks += [500 + (10 * i + j) % 400 for j in range(c)] + [EOT]
        masks += [0] * p + [1] * (c + 1)

        pos += p + c + 1
    return np.array(toks, dtype=np.uint16), np.array(masks, dtype=np.uint8), starts


def write(tmp_path, tokens, masks, name="train"):
    tokens.tofile(tmp_path / f"{name}.bin")
    masks.tofile(tmp_path / f"{name}.masks.bin")
    return tmp_path / f"{name}.bin", tmp_path / f"{name}.masks.bin"


def as_np(t):
    return np.asarray(t)


# ---- shift and boundaries ------------------------------------------------------

def test_shift_prompt_masked_completion_and_eot_supervised():
    tokens, masks, starts = pack((3, 2), (2, 1))  # [p p p c c E][p p c E]
    _, layout = audit_sft_arrays(tokens, masks, context_len=9, **KW)
    x, y = window_targets(tokens, masks, layout.starts, 0, 9)
    assert np.array_equal(x, tokens[0:9])  # inputs are the window itself
    # y[t] predicts tokens[t+1]: supervised exactly where tokens[t+1] is a completion token.
    expected = np.where(masks[1:10] == 1, tokens[1:10].astype(np.int64), IGNORE_INDEX)
    assert np.array_equal(y, expected)
    assert y[1] == IGNORE_INDEX            # predicting the last prompt token: masked
    assert y[2] == tokens[3]               # first completion token, predicted from the prompt
    assert y[4] == EOT                     # EOT is supervised (the model learns to stop)
    assert y[5] == IGNORE_INDEX            # next example's first prompt token: masked
    assert set(y[y != IGNORE_INDEX]).issubset(set(range(500, 1000)))


def test_targets_are_only_vocab_ids_or_ignore():

    tokens, masks, _ = pack((4, 6), (5, 3), (2, 8))
    _, layout = audit_sft_arrays(tokens, masks, context_len=12, **KW)
    for s in range(0, tokens.size - 12):
        _, y = window_targets(tokens, masks, layout.starts, s, 12)
        assert np.all((y == IGNORE_INDEX) | ((y >= 0) & (y < VOCAB)))


# ---- truncation and padding ----------------------------------------------------

def test_window_starting_mid_example_does_not_supervise_its_completion():
    tokens, masks, starts = pack((3, 4), (3, 2))
    _, layout = audit_sft_arrays(tokens, masks, context_len=8, **KW)
    mid = starts[0] + 2  # prompt of example 0 cut off by the window start
    _, y = window_targets(tokens, masks, layout.starts, mid, 8)
    owner0 = np.arange(mid + 1, mid + 9) < starts[1]
    assert np.all(y[owner0] == IGNORE_INDEX)  # example 0's answer without its question: masked
    second = (np.arange(mid + 1, mid + 9) >= starts[1]) & (masks[mid + 1 : mid + 9] == 1)
    assert np.all(y[second] != IGNORE_INDEX)  # example 1 starts inside the window: supervised


def test_oversized_example_counts_and_end_truncation():
    tokens, masks, starts = pack((3, 10), (2, 2))  # example 0 is 14 tokens, context 8
    audit, layout = audit_sft_arrays(tokens, masks, context_len=8, **KW)
    assert audit.oversized_examples == 1
    # From its start, targets reach position 8: completion positions 3..8 are supervisable,
    # 9..13 (incl. EOT at 13) never are.
    assert audit.unsupervisable_tokens == 5
    assert audit.supervisable_fraction == pytest.approx(1 - 5 / (11 + 3))
    _, y = window_targets(tokens, masks, layout.starts, 0, 8)
    assert np.array_equal(y != IGNORE_INDEX, masks[1:9] == 1)



def test_prompt_longer_than_context_leaves_completion_unsupervisable():
    tokens, masks, _ = pack((10, 2), (2, 30))
    audit, _ = audit_sft_arrays(tokens, masks, context_len=8, **KW)
    assert audit.unsupervisable_tokens >= 3  # all of example 0's completion (2 + EOT)


def test_tail_of_previous_example_is_padding_like_ignore():
    tokens, masks, starts = pack((2, 3), (2, 2), (2, 1))  # starts 0, 6, 11; 15 tokens
    _, layout = audit_sft_arrays(tokens, masks, context_len=6, **KW)
    s = 7  # inside example 1, so the window opens on its tail
    _, y = window_targets(tokens, masks, layout.starts, s, 6)
    inside_prev = np.arange(s + 1, s + 7) < starts[2]
    assert np.all(y[inside_prev] == IGNORE_INDEX)
    assert np.all(y[~inside_prev & (masks[s + 1 : s + 7] == 1)] != IGNORE_INDEX)


# ---- loaders: shapes, coverage, determinism ---------------------------------------

def test_train_batches_shapes_and_example_aligned_starts(tmp_path):
    tokens, masks, starts = pack(*[(3 + i % 4, 2 + i % 5) for i in range(30)])
    t, m = write(tmp_path, tokens, masks)
    loader = build_sft_dataloader(t, m, context_len=16, batch_size=4, seed=3, **KW)
    first_prompt_tokens = {int(tokens[s]) for s in starts} | {int(tokens[min(s, tokens.size - 17)]) for s in starts}
    for _ in range(25):
        x, y = (as_np(v) for v in next(loader))
        assert x.shape == y.shape == (4, 16) and x.dtype == y.dtype == np.int64
        assert all(int(row[0]) in first_prompt_tokens for row in x)
        assert (y != IGNORE_INDEX).sum() > 0  # every batch supervises something


def test_seed_offset_is_reproducible_and_changes_the_stream(tmp_path):
    tokens, masks, _ = pack(*[(3, 3)] * 40)

    t, m = write(tmp_path, tokens, masks)
    def first(offset):
        return as_np(next(build_sft_dataloader(t, m, 8, 4, seed_offset=offset, **KW))[0])
    assert np.array_equal(first(0), first(0))
    assert not np.array_equal(first(0), first(7))


@pytest.mark.parametrize("context_len", [6, 12, 31])
def test_validation_pass_scores_every_completion_token_once(tmp_path, context_len):
    tokens, masks, _ = pack(*[(2 + i % 3, 1 + i % 4) for i in range(25)])
    t, m = write(tmp_path, tokens, masks, "val")
    audit, layout = audit_sft_arrays(tokens, masks, context_len, **KW)
    n_windows = len(_eval_windows(layout, context_len, tokens.size - context_len - 1))
    loader = build_sft_dataloader(t, m, context_len, 3, shuffle=False, drop_last=False, **KW)
    supervised, rows = 0, 0
    while rows < n_windows:
        x, y = (as_np(v) for v in next(loader))
        rows += x.shape[0]
        supervised += int((y != IGNORE_INDEX).sum())
    assert rows == n_windows  # the last batch is short, not padded with repeats
    # One pass: every supervisable completion token exactly once (no double counting).
    assert supervised == audit.supervised_tokens - audit.unsupervisable_tokens


def test_validation_drop_last_keeps_full_batches(tmp_path):
    tokens, masks, _ = pack(*[(3, 3)] * 10)
    t, m = write(tmp_path, tokens, masks, "val")
    loader = build_sft_dataloader(t, m, 8, 4, shuffle=False, drop_last=True, **KW)
    for _ in range(10):
        assert as_np(next(loader)[0]).shape == (4, 8)



def test_audit_log_has_counts_only(tmp_path):
    tokens, masks, _ = pack((3, 2), (2, 2), (4, 1))
    tokens[tokens == 101] = 31337 % VOCAB
    t, m = write(tmp_path, tokens, masks)
    lines = []
    build_sft_dataloader(t, m, 8, 2, log=lines.append, **KW)
    assert len(lines) == 1 and "3 examples" in lines[0] and "malformed 0" in lines[0]
    assert "337" not in lines[0]  # no token ids in the log


# ---- invalid inputs and malformed boundaries ----------------------------------------

def test_hard_errors():
    tokens, masks, _ = pack((3, 3), (3, 3))
    with pytest.raises(SFTDataError, match="length mismatch"):
        audit_sft_arrays(tokens, masks[:-1], 4, **KW)
    bad = masks.copy(); bad[2] = 2
    with pytest.raises(SFTDataError, match="0 or 1"):
        audit_sft_arrays(tokens, bad, 4, **KW)
    big = tokens.copy(); big[0] = VOCAB
    with pytest.raises(SFTDataError, match="vocab_size"):
        audit_sft_arrays(big, masks, 4, **KW)
    with pytest.raises(SFTDataError, match="no completion"):
        audit_sft_arrays(tokens, np.zeros_like(masks), 4, **KW)
    with pytest.raises(SFTDataError, match="need more than"):
        audit_sft_arrays(tokens, masks, tokens.size, **KW)
    with pytest.raises(SFTDataError, match="positive"):
        audit_sft_arrays(tokens, masks, 0, **KW)


def test_file_errors(tmp_path):
    tokens, masks, _ = pack((3, 3), (3, 3))

    t, m = write(tmp_path, tokens, masks)
    with pytest.raises(FileNotFoundError):
        build_sft_dataloader(tmp_path / "missing.bin", m, 4, 2, **KW)
    (tmp_path / "odd.bin").write_bytes(b"\x01\x02\x03")
    with pytest.raises(SFTDataError, match="multiple of 2"):
        build_sft_dataloader(tmp_path / "odd.bin", m, 4, 2, **KW)
    with pytest.raises(ValueError):
        build_sft_dataloader(t, m, 4, 0, **KW)


@pytest.mark.parametrize("case, field", [
    ("empty", "empty_completions"),
    ("no_eot", "completions_missing_eot"),
    ("merged", "completions_missing_prompt"),
    ("leading_completion", "completions_missing_prompt"),
    ("eot_in_prompt", "eot_in_prompt"),
    ("dangling", "dangling_prompt_tokens"),
])
def test_malformed_boundaries_raise_when_strict_and_count_when_not(case, field):
    tokens, masks, starts = pack((3, 3), (3, 0), (3, 3))  # example 1 has an empty completion (just EOT)
    if case != "empty":
        tokens, masks, starts = pack((3, 3), (3, 3), (3, 3))
    if case == "no_eot":
        tokens[starts[1] - 1] = 777                # example 0's completion ends without EOT
    elif case == "merged":
        masks[starts[1] : starts[1] + 3] = 1       # example 1's prompt marked as completion
    elif case == "leading_completion":
        masks[: starts[1]] = 1                     # file starts with a completion
    elif case == "eot_in_prompt":
        tokens[starts[2]] = EOT                    # EOT inside a prompt
    elif case == "dangling":
        tokens = np.concatenate([tokens, np.array([100, 101], dtype=np.uint16)])

        masks = np.concatenate([masks, np.zeros(2, dtype=np.uint8)])
    with pytest.raises(SFTDataError, match="malformed"):
        audit_sft_arrays(tokens, masks, 6, **KW)
    audit, _ = audit_sft_arrays(tokens, masks, 6, strict=False, **KW)
    assert getattr(audit, field) > 0 and audit.malformed > 0


def test_well_formed_corpus_has_no_malformed_counts():
    tokens, masks, _ = pack(*[(2 + i % 5, 1 + i % 6) for i in range(1, 40)])
    audit, layout = audit_sft_arrays(tokens, masks, 16, **KW)
    assert audit.malformed == 0 and audit.examples == 39
    assert audit.supervised_tokens == int(masks.sum())
    assert layout.starts.size == layout.ends.size == 39