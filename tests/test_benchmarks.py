"""Benchmark tests: no downloads, GPU, or pretrained weights."""

import math

import pytest
import torch

from evals.benchmarks import (
    _completion_nll,
    _completion_ids,
    _lambada_parts,
    score_hellaswag,
    score_lambada,
)


class CharTokenizer:
    def encode(self, text):
        return [ord(c) for c in text]


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
    """Predict space after 'answer', then z characters."""

    def forward(self, x):
        logits = torch.full(
            (*x.shape, 256),
            -10.0,
            device=x.device,
        )

        next_ids = torch.where(
            x[:, -1] == ord("r"),
            ord(" "),
            ord("z"),
        )

        logits[:, -1, :] = -10.0

        for row, token_id in enumerate(
            next_ids.tolist()
        ):
            logits[
                row, -1, token_id
            ] = 10.0

        return logits, None


def test_causal_completion_mask_and_normalization():
    tok = CharTokenizer()

    ids, positions = _completion_ids(
        tok, "hello", " xy"
    )

    assert positions == [5, 6, 7]

    sums, counts = _completion_nll(
        UniformLM(),
        [ids],
        [positions],
        context_len=32,
        device="cpu",
    )

    assert counts == [3]
    assert (
        sums[0] / 3
        == pytest.approx(math.log(256))
    )


def test_hellaswag_normalized_accuracy_and_limit():
    rows = [
        {
            "ctx_a": "This",
            "ctx_b": "is",
            "endings": ["x", "y", "z", "w"],
            "label": "0",
        },
        {
            "ctx_a": "That",
            "ctx_b": "was",
            "endings": ["x", "y", "z", "w"],
            "label": "1",
        },
    ]

    # Equal NLL creates a deterministic tie to choice 0.
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


def test_lambada_multi_token_word_accuracy_and_perplexity():
    rows = [
        {"text": "the answer zz"},
        {"text": "the answer xy"},
    ]

    result = score_lambada(
        EchoNextLM(),
        rows,
        CharTokenizer(),
        context_len=32,
    )

    assert result["examples"] == 2
    assert result["correct"] == 1
    assert result["last_word_accuracy"] == 0.5

    # Character-level tokenizer includes leading spaces:
    # " zz" and " xy" each contain 3 target tokens.
    assert result["target_tokens"] == 6
    assert (
        result["target_word_perplexity"]
        is not None
    )


def test_lambada_preserves_leading_space_and_rejects_missing_context():
    assert _lambada_parts(
        "Hello   world"
    ) == (
        "Hello",
        "   world",
    )

    with pytest.raises(ValueError):
        _lambada_parts("single")


def test_context_overflow_uses_rolling_window():
    ids, positions = _completion_ids(
        CharTokenizer(),
        "abcd",
        " efg",
    )

    nll, counts = _completion_nll(
        UniformLM(),
        [ids],
        [positions],
        context_len=2,
        device="cpu",
    )

    assert counts == [4]
    assert nll[0] == pytest.approx(
        4 * math.log(256)
    )
