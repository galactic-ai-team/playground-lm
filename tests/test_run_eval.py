"""Offline tests for the model-agnostic evaluator."""

import math

import numpy as np
import pytest

torch = pytest.importorskip("torch")

from evals.run_eval import (
    copy_matches,
    generate_tokens,
    get_logits,
    validation_metrics,
)


class UniformModel(torch.nn.Module):
    """Returns equal logits for every vocabulary token."""

    def __init__(self, vocab_size=64):
        super().__init__()
        self.vocab_size = vocab_size

    def forward(self, x, y=None):
        b, t = x.shape
        logits = torch.zeros(
            b, t, self.vocab_size, device=x.device
        )
        return logits, None


class HFStyleOutput:
    def __init__(self, logits):
        self.logits = logits


class HFStyleModel(UniformModel):
    def forward(self, x):
        logits, _ = super().forward(x)
        return HFStyleOutput(logits)


class ConstantNextToken(torch.nn.Module):
    """Always predicts token 7."""

    def forward(self, x):
        b, t = x.shape
        logits = torch.full((b, t, 16), -100.0)
        logits[..., 7] = 100.0
        return logits, None


def write_tokens(path, ids):
    np.asarray(ids, dtype="<u2").tofile(path)


@pytest.mark.parametrize(
    "model",
    [UniformModel(64), HFStyleModel(64)],
)
def test_validation_perplexity_and_full_coverage(
    tmp_path, model
):
    val = tmp_path / "val.bin"

    # 12 input tokens -> 11 next-token predictions.
    # Context=4 => 4 + 4 + 3 predictions.
    write_tokens(val, list(range(12)))

    metrics = validation_metrics(
        model,
        val,
        context_len=4,
        batch_size=2,
        device="cpu",
    )

    assert metrics["tokens_scored"] == 11
    assert metrics["available_targets"] == 11
    assert metrics["coverage"] == 1.0
    assert metrics["complete"] is True
    assert metrics["loss"] == pytest.approx(math.log(64))
    assert metrics["perplexity"] == pytest.approx(64)


def test_partial_validation_is_explicit(tmp_path):
    val = tmp_path / "val.bin"
    write_tokens(val, list(range(30)))

    metrics = validation_metrics(
        UniformModel(64),
        val,
        context_len=4,
        batch_size=2,
        device="cpu",
        max_batches=1,
    )

    assert metrics["tokens_scored"] == 8
    assert metrics["available_targets"] == 29
    assert metrics["coverage"] == pytest.approx(8 / 29)
    assert metrics["complete"] is False


def test_output_normalization():
    x = torch.tensor([[1, 2, 3]])

    assert get_logits(
        UniformModel(), x
    ).shape == (1, 3, 64)

    assert get_logits(
        HFStyleModel(), x
    ).shape == (1, 3, 64)


def test_greedy_generation_is_continuation_only():
    ids = generate_tokens(
        ConstantNextToken(),
        [1, 2, 3],
        max_new_tokens=5,
        context_len=4,
        device="cpu",
        eot_id=15,
    )

    assert ids == [7] * 5


def test_exact_copy_50_tokens(tmp_path):
    train = tmp_path / "train.bin"

    # Distinct enough that the test covers an exact middle match.
    source = list(range(1, 200))
    write_tokens(train, source)

    generation = [900] + source[40:95] + [901]

    matches = copy_matches(train, [generation])

    assert matches[0]["flagged"] is True
    assert matches[0]["threshold_tokens"] == 50
    assert matches[0]["match_start_in_generation"] == 1
    assert matches[0]["match_start_in_training"] == 40


def test_49_tokens_not_flagged(tmp_path):
    train = tmp_path / "train.bin"
    source = list(range(1, 200))
    write_tokens(train, source)

    matches = copy_matches(
        train,
        [[900] + source[40:89] + [901]],
    )

    assert matches[0]["flagged"] is False


def test_prompt_not_included_in_copy_check(tmp_path):
    train = tmp_path / "train.bin"
    source = list(range(1, 100))
    write_tokens(train, source)

    # The evaluator receives continuation tokens only.
    # A long matching prompt therefore cannot trigger the flag.
    continuation = [500] * 10

    matches = copy_matches(train, [continuation])

    assert matches[0]["flagged"] is False


def test_multiple_generations_checked(tmp_path):
    train = tmp_path / "train.bin"
    source = list(range(200))
    write_tokens(train, source)

    matches = copy_matches(
        train,
        [
            source[20:75],
            [999] * 60,
            source[80:130],
        ],
    )

    assert [m["flagged"] for m in matches] == [
        True, False, True
    ]


def test_unaligned_training_file_rejected(tmp_path):
    train = tmp_path / "invalid.bin"
    train.write_bytes(b"abc")

    with pytest.raises(ValueError, match="uint16-aligned"):
        copy_matches(train, [[1] * 50])
