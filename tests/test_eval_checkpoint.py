"""Offline tests for scripts/eval_checkpoint.py.

Uses a tiny PlaygroundLM and genuine safetensors loading.
No Hugging Face downloads, pretrained weights, or GPU required.
"""

from __future__ import annotations

import json
import math

import numpy as np
import pytest

torch = pytest.importorskip("torch")
safetensors = pytest.importorskip("safetensors.torch")

import scripts.eval_checkpoint as evaluator
from playground_lm.model import PlaygroundLM
from evals.publish_checkpoint import write_json_atomic


CONFIG = {
    "vocab_size": 128,
    "n_layers": 1,
    "n_heads": 2,
    "d_model": 16,
    "context_len": 16,
    "dropout": 0.0,
}


class AsciiTokenizer:
    def encode(self, text):
        return [ord(char) for char in text]


def make_checkpoint(tmp_path):
    """Use Dee's actual split weights/run-manifest layout."""
    torch.manual_seed(42)

    run_dir = tmp_path / "run"
    ckpt_dir = run_dir / "checkpoints"
    ckpt_dir.mkdir(parents=True)

    model = PlaygroundLM(**CONFIG).eval()
    weights = ckpt_dir / "step_00000002.safetensors"

    safetensors.save_file(
        {
            name: tensor.detach().cpu().contiguous()
            for name, tensor in model.state_dict().items()
        },
        str(weights),
    )

    # This is the same nested configuration structure
    # load_model discovers in a real training run.
    (run_dir / "manifest.json").write_text(
        json.dumps({
            "run_id": "tiny",
            "config": {
                "model": CONFIG,
                "train": {
                    "batch_size": 2,
                    "lr": 0.001,
                },
            },
            "checkpoints": [],
        })
    )

    val_bin = tmp_path / "val.bin"
    np.asarray(
        [i % 100 for i in range(129)],
        dtype="<u2",
    ).tofile(val_bin)

    return weights, val_bin


def synthetic_rows(name):
    if name == "hellaswag":
        return [
            {
                "activity_label": "Task",
                "ctx_a": "The person",
                "ctx_b": "is ready",
                "endings": [
                    "to go",
                    "to stay",
                    "to run",
                    "to sleep",
                ],
                "label": "0",
            },
            {
                "activity_label": "Task",
                "ctx_a": "The person",
                "ctx_b": "is tired",
                "endings": [
                    "to rest",
                    "to work",
                    "to play",
                    "to walk",
                ],
                "label": "1",
            },
        ]

    if name == "lambada":
        return [
            {"text": "The last word is blue"},
            {"text": "The last word is red"},
        ]

    raise ValueError(name)


def test_real_split_checkpoint_smoke(
    tmp_path, monkeypatch
):
    weights, val_bin = make_checkpoint(tmp_path)

    monkeypatch.setattr(
        evaluator,
        "get_tokenizer",
        lambda: AsciiTokenizer(),
    )

    calls = []

    def fake_rows(name, *, offline=False):
        calls.append(name)
        return synthetic_rows(name)

    monkeypatch.setattr(
        evaluator, "load_rows", fake_rows
    )

    report = evaluator.evaluate_checkpoint(
        checkpoint=weights,
        val_bin=val_bin,
        checkpoint_id="tiny_step_2",
        label="Tiny step 2",
        device="cpu",
        batch_size=2,
        limit=2,
        max_val_batches=1,
    )

    assert calls == ["hellaswag", "lambada"]

    # Proves the split checkpoint was loaded and actually
    # used in evaluation.
    assert report["context_len"] == 16
    assert report["backend"] == "playground"

    assert report["validation"]["tokens_scored"] > 0
    assert math.isfinite(
        report["validation"]["loss"]
    )

    hs = report["benchmarks"]["hellaswag"]
    la = report["benchmarks"]["lambada"]

    assert hs["examples"] == 2
    assert la["examples"] == 2
    assert hs["complete"] is False
    assert la["complete"] is False

    # Limited benchmark scores cannot enter production rows.
    assert report["publishable"] is False
    assert report["row"] is None
    assert report["preview_row"]["id"] == "tiny_step_2"

    output = tmp_path / "tiny_results.json"
    write_json_atomic(output, report)

    assert output.is_file()
    saved = json.loads(output.read_text())

    assert saved["publishable"] is False
    assert saved["row"] is None
    assert saved["benchmarks"]["hellaswag"]["examples"] == 2


def test_complete_report_matches_demo_row_contract():
    loss = 3.0
    validation = {
        "loss": loss,
        "perplexity": math.exp(loss),
        "tokens_scored": 128,
        "available_targets": 128,
        "complete": True,
    }
    hellaswag = {
        "accuracy": 0.25,
        "examples": 10042,
        "complete": True,
    }
    lambada = {
        "last_word_accuracy": 0.10,
        "examples": 5153,
        "complete": True,
    }

    report = evaluator.build_report(
        checkpoint="stage_b.safetensors",
        checkpoint_id="stage_b",
        label="Stage B",
        context_len=1024,
        validation=validation,
        hellaswag=hellaswag,
        lambada=lambada,
    )

    assert report["publishable"] is True
    assert report["status"] == "complete"
    assert report["preview_row"] is None

    row = report["row"]
    assert row["id"] == "stage_b"
    assert row["val_loss"] == pytest.approx(3.0)
    assert row["perplexity"] == pytest.approx(math.exp(3.0))
    assert row["hellaswag_acc_norm"] == 0.25
    assert row["lambada_acc"] == 0.10


def test_limit_never_publishes_complete_looking_metrics():
    validation = {
        "loss": 2.0,
        "perplexity": math.exp(2.0),
        "complete": True,
        "tokens_scored": 100,
        "available_targets": 100,
    }
    hellaswag = {
        "accuracy": 0.3,
        "examples": 10042,
        "complete": True,
    }
    lambada = {
        "last_word_accuracy": 0.2,
        "examples": 5153,
        "complete": True,
    }

    report = evaluator.build_report(
        checkpoint="tiny.safetensors",
        checkpoint_id="smoke",
        label="Smoke only",
        context_len=16,
        validation=validation,
        hellaswag=hellaswag,
        lambada=lambada,
        limit=20,
    )

    assert report["publishable"] is False
    assert report["row"] is None
    assert report["preview_row"]["id"] == "smoke"


def test_invalid_context_rejected(tmp_path):
    weights, val_bin = make_checkpoint(tmp_path)

    with pytest.raises(
        ValueError, match="exceeds model context"
    ):
        evaluator.evaluate_checkpoint(
            checkpoint=weights,
            val_bin=val_bin,
            checkpoint_id="bad_context",
            label="Bad context",
            context_len=32,
            limit=1,
        )
