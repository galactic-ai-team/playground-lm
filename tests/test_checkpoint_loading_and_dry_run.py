"""Split checkpoint loading and publication dry-run regression tests.

CPU-only, no network, GPT-2 download, or official benchmark data.
"""

import json
import math
import sys

import pytest

torch = pytest.importorskip("torch")
safetensors = pytest.importorskip("safetensors.torch")

from playground_lm.model import PlaygroundLM
from evals.run_eval import load_model
from evals import publish_checkpoint as publisher


CONFIG = {
    "vocab_size": 32,
    "n_layers": 1,
    "n_heads": 2,
    "d_model": 16,
    "context_len": 16,
    "dropout": 0.0,
}


def make_weights(tmp_path):
    model = PlaygroundLM(**CONFIG).eval()
    run_dir = tmp_path / "run"
    ckpt_dir = run_dir / "checkpoints"
    ckpt_dir.mkdir(parents=True)

    weights = ckpt_dir / "step_00000003.safetensors"
    safetensors.save_file(
        {
            k: v.detach().cpu().contiguous()
            for k, v in model.state_dict().items()
        },
        str(weights),
    )
    return model, weights, run_dir


def assert_same_parameters(original, loaded):
    assert not loaded.training
    for a, b in zip(
        original.parameters(),
        loaded.parameters(),
    ):
        torch.testing.assert_close(a, b)


def test_split_checkpoint_run_manifest(tmp_path):
    original, weights, run_dir = make_weights(tmp_path)

    (run_dir / "manifest.json").write_text(
        json.dumps({
            "run_id": "smoke",
            "config": {
                "model": CONFIG,
                "train": {"batch_size": 2},
            },
            "latest_step": 3,
        })
    )

    loaded = load_model(
        "playground",
        str(weights),
        device="cpu",
        factory="playground_lm.model:PlaygroundLM",
    )

    assert_same_parameters(original, loaded)


def test_split_checkpoint_explicit_json(tmp_path):
    original, weights, run_dir = make_weights(tmp_path)

    config_file = run_dir / "model-config.json"
    config_file.write_text(
        json.dumps({
            "model": CONFIG,
            "train": {"lr": 0.001},
        })
    )

    loaded = load_model(
        "playground",
        str(weights),
        device="cpu",
        factory="playground_lm.model:PlaygroundLM",
        config_path=str(config_file),
    )
    assert_same_parameters(original, loaded)


def test_split_checkpoint_paired_state_fallback(tmp_path):
    original, weights, _ = make_weights(tmp_path)

    state_file = weights.with_name(
        weights.stem + "_state.pt"
    )
    torch.save(
        {
            "step": 3,
            "config": {
                "model": CONFIG,
                "train": {"batch_size": 2},
            },
        },
        state_file,
    )

    loaded = load_model(
        "playground",
        str(weights),
        device="cpu",
        factory="playground_lm.model:PlaygroundLM",
    )

    assert_same_parameters(original, loaded)


def test_split_checkpoint_missing_config_rejected(tmp_path):
    _, weights, _ = make_weights(tmp_path)

    with pytest.raises(
        FileNotFoundError,
        match="No model config",
    ):
        load_model(
            "playground",
            str(weights),
            device="cpu",
            factory="playground_lm.model:PlaygroundLM",
        )


def test_combined_checkpoint_still_supported(tmp_path):
    original = PlaygroundLM(**CONFIG).eval()
    checkpoint = tmp_path / "combined.pt"

    torch.save({
        "model": original.state_dict(),
        "config": {
            "model": CONFIG,
            "train": {"batch_size": 2},
        },
    }, checkpoint)

    loaded = load_model(
        "playground",
        str(checkpoint),
        device="cpu",
        factory="playground_lm.model:PlaygroundLM",
    )

    assert_same_parameters(original, loaded)


def complete_metrics():
    return (
        {
            "complete": True,
            "loss": 3.0,
            "perplexity": math.exp(3.0),
            "tokens_scored": 100,
            "available_targets": 100,
        },
        {
            "complete": True,
            "examples": 10042,
            "accuracy": 0.25,
        },
        {
            "complete": True,
            "examples": 5153,
            "last_word_accuracy": 0.10,
        },
    )


def test_dry_run_no_writes(
    tmp_path, monkeypatch, capsys
):
    manifest = tmp_path / "manifest.json"
    report = tmp_path / "results" / "report.json"

    original = {
        "model": "PlaygroundLM",
        "checkpoints": [],
        "benchmarks": {
            "columns": list(publisher.COLUMNS),
            "rows": [],
        },
        "riddle_prompts": ["demo riddle"],
    }
    manifest.write_text(json.dumps(original))
    before = manifest.read_bytes()

    calls = []

    def fake_evaluate(**kwargs):
        calls.append(kwargs)
        return complete_metrics()

    monkeypatch.setattr(
        publisher, "evaluate", fake_evaluate
    )

    monkeypatch.setattr(
        sys,
        "argv",
        [
            "publish_checkpoint",
            "--backend", "playground",
            "--model", "step_00000003.safetensors",
            "--factory", "playground_lm.model:PlaygroundLM",
            "--id", "step_00000003",
            "--val-bin", "data/val.bin",
            "--manifest", str(manifest),
            "--report", str(report),
            "--dry-run",
        ],
    )

    publisher.main()
    output = json.loads(capsys.readouterr().out)

    assert len(calls) == 1
    assert output["dry_run"] is True
    assert output["would_publish"]["id"] == "step_00000003"
    assert len(
        output["resulting_manifest"]["benchmarks"]["rows"]
    ) == 1

    assert manifest.read_bytes() == before
    assert not report.exists()
    assert not report.parent.exists()


def test_dry_run_rejects_incomplete_scores(
    tmp_path, monkeypatch
):
    manifest = tmp_path / "manifest.json"
    manifest.write_text(json.dumps({
        "benchmarks": {
            "columns": list(publisher.COLUMNS),
            "rows": [],
        }
    }))
    before = manifest.read_bytes()

    def incomplete(**kwargs):
        reports = list(complete_metrics())
        reports[1]["complete"] = False
        return tuple(reports)

    monkeypatch.setattr(
        publisher, "evaluate", incomplete
    )
    monkeypatch.setattr(
        sys,
        "argv",
        [
            "publish_checkpoint",
            "--backend", "hf",
            "--model", "gpt2",
            "--id", "gpt2_124m",
            "--val-bin", "data/val.bin",
            "--manifest", str(manifest),
            "--dry-run",
        ],
    )

    with pytest.raises(
        ValueError,
        match="incomplete hellaswag",
    ):
        publisher.main()

    assert manifest.read_bytes() == before
