"""Offline regression tests for both benchmark scoring paths."""

import math

import pytest
import torch

import evals.benchmarks as benchmarks

from evals.benchmarks import (
    _completion_ids,
    _completion_nll,
    _completion_stats,
    _hellaswag_context,
    _hellaswag_preprocess,
    _lambada_parts,
    score_hellaswag,
    score_lambada,
)


class CharTokenizer:
    def encode(self, text):
        return [ord(char) for char in text]


class BoundaryMergeTokenizer:
    """Merge 'c ' across the context/ending boundary."""

    def encode(self, text):
        result = []
        i = 0
        while i < len(text):
            if text[i:i + 2] == "c ":
                result.append(250)
                i += 2
            else:
                result.append(ord(text[i]))
                i += 1
        return result


class UniformLM(torch.nn.Module):
    def forward(self, x):
        return (
            torch.zeros(
                (*x.shape, 256),
                device=x.device,
            ),
            None,
        )


class EchoNextLM(torch.nn.Module):
    """Greedy: space after 'answer', then repeated z."""

    def __init__(self):
        super().__init__()
        self.calls = 0

    def forward(self, x):
        self.calls += 1

        logits = torch.full(
            (*x.shape, 256),
            -10.0,
            device=x.device,
        )
        logits[..., ord("z")] = 10.0

        # Correct prediction for the first target token,
        # which follows the 'r' in 'answer'.
        logits[
            x == ord("r"), ord(" ")
        ] = 20.0

        return logits, None


def test_completion_positions_and_token_weighting():
    tok = CharTokenizer()

    ids, positions = _completion_ids(
        tok, "hello", " xy"
    )

    assert positions == [5, 6, 7]

    sums, lengths = _completion_nll(
        UniformLM(),
        [ids],
        [positions],
        context_len=32,
        device="cpu",
    )

    assert lengths == [3]
    assert sums[0] == pytest.approx(
        3 * math.log(256)
    )


def test_bpe_boundary_merge_does_not_raise():
    tok = BoundaryMergeTokenizer()

    context_ids = tok.encode("abc")
    joined_ids = tok.encode("abc x")

    # Independent context tokenization differs from
    # the combined sequence across the boundary.
    assert joined_ids[:len(context_ids)] != context_ids

    ids, positions = _completion_ids(
        tok,
        "abc",
        " x",
    )

    assert ids == joined_ids
    assert positions == [len(context_ids)]

    nll, lengths = _completion_nll(
        UniformLM(),
        [ids],
        [positions],
        context_len=32,
        device="cpu",
    )

    assert lengths == [1]
    assert math.isfinite(nll[0])


def test_trailing_context_whitespace_moves_to_suffix():
    tok = CharTokenizer()

    ids, positions = _completion_ids(
        tok,
        "Hello  ",
        "world",
    )

    assert ids == tok.encode("Hello  world")
    assert positions == list(
        range(len(tok.encode("Hello")), len(ids))
    )


def test_hellaswag_preprocessing_matches_reference():
    assert _hellaswag_preprocess(
        "  Do this [title] [artifact] now  "
    ) == "Do this.   now".replace("  ", " ")

    row = {
        "activity_label": "Cooking",
        "ctx_a": "Stir",
        "ctx_b": "the soup",
        "endings": ["serve", "wait", "leave", "taste"],
        "label": "0",
    }

    assert _hellaswag_context(row) == (
        "Cooking: Stir The soup"
    )


def test_hellaswag_normalized_accuracy_and_limit():
    rows = [
        {
            "activity_label": "Task",
            "ctx_a": "This",
            "ctx_b": "is",
            "endings": ["x", "y", "z", "w"],
            "label": "0",
        },
        {
            "activity_label": "Task",
            "ctx_a": "That",
            "ctx_b": "was",
            "endings": ["x", "y", "z", "w"],
            "label": "1",
        },
    ]

    # Equal NLL deterministically selects option zero.
    result = score_hellaswag(
        UniformLM(),
        rows,
        CharTokenizer(),
        context_len=40,
        limit=2,
    )

    assert result["examples"] == 2
    assert result["correct"] == 1
    assert result["accuracy"] == 0.5
    assert result["complete"] is False


def test_hellaswag_boundary_merges_do_not_skip_rows():
    row = {
        "activity_label": "Activity",
        "ctx_a": "abc",
        "ctx_b": "go",
        "endings": ["x", "y", "z", "w"],
        "label": "0",
    }

    # Uniform logits make the answer deterministic,
    # but all four endings must still be evaluated.
    result = score_hellaswag(
        UniformLM(),
        [row] * 6,
        BoundaryMergeTokenizer(),
        context_len=64,
    )

    assert result["examples"] == 6
    assert result["correct"] == 6
    assert result["complete"] is False


def test_hellaswag_full_split_accounting_no_skip(monkeypatch):
    row = {
        "activity_label": "Task",
        "ctx_a": "abc",
        "ctx_b": "go",
        "endings": ["x", "y", "z", "w"],
        "label": "0",
    }
    calls = []

    def fake_completion_nll(
        model,
        sequences,
        positions,
        *,
        context_len,
        device,
    ):
        assert len(sequences) == 4
        assert len(positions) == 4
        assert all(len(p) > 0 for p in positions)
        calls.append(1)
        return [1.0, 2.0, 3.0, 4.0], [1, 1, 1, 1]

    monkeypatch.setattr(
        benchmarks,
        "_completion_nll",
        fake_completion_nll,
    )

    # Fast synthetic full-split test. No 40k model forwards.
    rows = (row for _ in range(10042))

    result = score_hellaswag(
        UniformLM(),
        rows,
        BoundaryMergeTokenizer(),
        context_len=64,
    )

    assert result["examples"] == 10042
    assert result["correct"] == 10042
    assert result["complete"] is True
    assert len(calls) == 10042


def test_lambada_reference_word_split():
    assert _lambada_parts(
        "Hello world"
    ) == ("Hello", " world")

    # Literal-space splitting is the reference task's rule.
    assert _lambada_parts(
        "Hello   world"
    ) == ("Hello  ", " world")

    with pytest.raises(ValueError):
        _lambada_parts("single")


def test_lambada_full_word_greedy_accuracy():
    model = EchoNextLM()
    rows = [
        {"text": "the answer zz"},
        {"text": "the answer xy"},
    ]

    result = score_lambada(
        model,
        rows,
        CharTokenizer(),
        context_len=32,
    )

    assert result["examples"] == 2
    assert result["correct"] == 1
    assert result["last_word_accuracy"] == 0.5
    assert result["target_tokens"] == 6

    # Short reference sequences are scored using a
    # single teacher-forced forward pass per example.
    assert model.calls == 2


def test_lambada_word_vs_token_perplexity():
    rows = [
        {"text": "the answer zz"},
        {"text": "the answer xy"},
    ]

    result = score_lambada(
        UniformLM(),
        rows,
        CharTokenizer(),
        context_len=32,
    )

    # Both words tokenize as three character tokens:
    # leading space plus two characters.
    assert result["target_tokens"] == 6
    assert result["target_token_nll"] == pytest.approx(
        math.log(256)
    )
    assert result["target_token_perplexity"] == pytest.approx(
        256
    )

    # Per-word NLL is 3 times the per-token NLL.
    assert math.log(
        result["target_word_perplexity"]
    ) == pytest.approx(
        3 * math.log(256)
    )


def test_lambada_full_split_not_skipped(monkeypatch):
    def fake_completion_stats(
        model,
        sequences,
        positions,
        *,
        context_len,
        device,
    ):
        assert len(sequences) == 1
        assert len(positions[0]) > 0
        return [2.0], [2], [True]

    monkeypatch.setattr(
        benchmarks,
        "_completion_stats",
        fake_completion_stats,
    )

    rows = (
        {"text": "context word"}
        for _ in range(5153)
    )

    result = score_lambada(
        UniformLM(),
        rows,
        CharTokenizer(),
    )

    assert result["examples"] == 5153
    assert result["correct"] == 5153
    assert result["complete"] is True


def test_context_overflow_uses_rolling_window():
    ids, positions = _completion_ids(
        CharTokenizer(),
        "abcd",
        " efg",
    )

    sums, counts, greedy_flags = _completion_stats(
        UniformLM(),
        [ids],
        [positions],
        context_len=2,
        device="cpu",
    )

    assert counts == [4]
    assert sums[0] == pytest.approx(
        4 * math.log(256)
    )
    assert greedy_flags == [False]
