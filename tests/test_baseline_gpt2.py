"""Offline tests for the GPT-2 baseline runner."""

import json
import math
import sys

import numpy as np
import pytest

torch = pytest.importorskip("torch")

from evals import baseline_gpt2 as baseline
from evals.run_eval import validation_metrics


class TinyMockGPT2(torch.nn.Module):
    """Uniform 64-token causal LM for validation scoring."""

    def forward(self, x):
        batch, length = x.shape
        logits = torch.zeros(
            batch, length, 64,
            device=x.device,
        )
        return type("Output", (), {"logits": logits})()


def write_val(path):
    np.asarray(
        list(range(13)),
        dtype="<u2",
    ).tofile(path)


def patch_baseline(monkeypatch, *, complete=True):
    model = TinyMockGPT2()
    tokenizer = object()
    calls = []

    def fake_load_model(backend, model_ref, *, device):
        assert backend == "hf"
        assert model_ref == "gpt2"
        assert device == "cpu"
        calls.append("model")
        return model

    def fake_tokenizer():
        calls.append("tokenizer")
        return tokenizer

    def fake_load_rows(name, *, offline=False):
        calls.append(name)
        return [name]

    def fake_hellaswag(
        supplied_model,
        rows,
        supplied_tokenizer,
        **kwargs,
    ):
        assert supplied_model is model
        assert supplied_tokenizer is tokenizer
        assert rows == ["hellaswag"]
        assert kwargs["limit"] is None
        return {
            "complete": complete,
            "examples": 10042 if complete else 20,
            "accuracy": 0.31,
        }

    def fake_lambada(
        supplied_model,
        rows,
        supplied_tokenizer,
        **kwargs,
    ):
        assert supplied_model is model
        assert supplied_tokenizer is tokenizer
        assert rows == ["lambada"]
        assert kwargs["limit"] is None
        return {
            "complete": True,
            "examples": 5153,
            "last_word_accuracy": 0.18,
            "target_word_perplexity": 60.0,
        }

    monkeypatch.setattr(
        baseline,
        "load_model",
        fake_load_model,
    )
    monkeypatch.setattr(
        baseline,
        "get_tokenizer",
        fake_tokenizer,
    )
    monkeypatch.setattr(
        baseline,
        "load_rows",
        fake_load_rows,
    )
    monkeypatch.setattr(
        baseline,
        "score_hellaswag",
        fake_hellaswag,
    )
    monkeypatch.setattr(
        baseline,
        "score_lambada",
        fake_lambada,
    )

    return calls


def test_baseline_same_validation_path(
    tmp_path, monkeypatch
):
    val_path = tmp_path / "val.bin"
    write_val(val_path)

    calls = patch_baseline(monkeypatch)

    # The actual evals.run_eval.validation_metrics function
    # remains in use, with a mocked model.
    assert baseline.validation_metrics is validation_metrics

    result = baseline.run_baseline(
        val_bin=val_path,
        context_len=4,
        batch_size=2,
        device="cpu",
    )

    assert calls == [
        "model",
        "tokenizer",
        "hellaswag",
        "lambada",
    ]

    assert result["model"] == "gpt2"
    assert result["backend"] == "hf"
    assert result["checkpoint_id"] == "gpt2_124m"

    validation = result["validation"]
    assert validation["complete"]
    assert validation["tokens_scored"] == 12
    assert validation["loss"] == pytest.approx(
        math.log(64)
    )
    assert validation["perplexity"] == pytest.approx(
        64
    )

    row = result["row"]
    assert row["id"] == "gpt2_124m"
    assert row["val_loss"] == pytest.approx(
        math.log(64)
    )
    assert row["perplexity"] == pytest.approx(64)
    assert row["hellaswag_acc_norm"] == 0.31
    assert row["lambada_acc"] == 0.18


def test_partial_benchmark_cannot_publish(
    tmp_path, monkeypatch
):
    val_path = tmp_path / "val.bin"
    write_val(val_path)

    patch_baseline(monkeypatch, complete=False)

    with pytest.raises(
        ValueError,
        match="incomplete hellaswag",
    ):
        baseline.run_baseline(
            val_bin=val_path,
            context_len=4,
            batch_size=2,
        )


def test_baseline_cli_writes_report_and_manifest(
    tmp_path, monkeypatch
):
    val_path = tmp_path / "val.bin"
    write_val(val_path)
    patch_baseline(monkeypatch)

    output = tmp_path / "results" / "gpt2.json"
    manifest_path = tmp_path / "manifest.json"

    original_manifest = {
        "model": "PlaygroundLM",
        "checkpoints": [
            {"id": "step_1000"}
        ],
        "benchmarks": {
            "columns": [
                "val_loss",
                "perplexity",
                "hellaswag_acc_norm",
                "lambada_acc",
            ],
            "rows": [],
        },
        "riddle_prompts": [
            "What opens without a key?"
        ],
    }
    manifest_path.write_text(
        json.dumps(original_manifest)
    )

    monkeypatch.setattr(
        sys,
        "argv",
        [
            "baseline_gpt2",
            "--val-bin", str(val_path),
            "--context-len", "4",
            "--batch-size", "2",
            "--output", str(output),
            "--manifest", str(manifest_path),
            "--publish",
        ],
    )

    baseline.main()

    report = json.loads(output.read_text())
    manifest = json.loads(
        manifest_path.read_text()
    )

    assert report["row"]["id"] == "gpt2_124m"
    assert manifest["benchmarks"]["rows"] == [
        report["row"]
    ]
    assert manifest["checkpoints"] == [
        {"id": "step_1000"}
    ]
    assert manifest["riddle_prompts"] == [
        "What opens without a key?"
    ]

    # Publishing twice replaces the existing baseline row.
    baseline.main()
    manifest = json.loads(
        manifest_path.read_text()
    )
    assert len(
        manifest["benchmarks"]["rows"]
    ) == 1


def test_invalid_context_rejected(
    tmp_path, monkeypatch
):
    val_path = tmp_path / "val.bin"
    write_val(val_path)
    patch_baseline(monkeypatch)

    with pytest.raises(
        ValueError, match="context_len"
    ):
        baseline.run_baseline(
            val_bin=val_path,
            context_len=1025,
        )
