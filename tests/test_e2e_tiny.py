
"""CPU-only end-to-end smoke test for PlaygroundLM.

Covers:
    synthetic uint16 dataset
      -> real training loop
      -> Dee's CheckpointManager
      -> split safetensors training checkpoint
      -> direct run_eval.load_model()
      -> safetensors inference export
      -> validation loss and perplexity
      -> held-out prompt generation
      -> optional ONNX export
      -> benchmark manifest dry-run preview

No temporary combined PyTorch checkpoint is created.

No GPU, network, GPT-2 weights, or benchmark downloads required.

HellaSwag and LAMBADA values are MOCKED solely to test the
publication schema. They are never published as real scores.

Run:
    pytest -q tests/test_e2e_tiny.py
"""

from __future__ import annotations

import copy
import importlib.util
import json
import math
from pathlib import Path

import numpy as np
import pytest

torch = pytest.importorskip("torch")
safetensors = pytest.importorskip("safetensors.torch")

from playground_lm.model import PlaygroundLM
from playground_lm.train.data import build_dataloader
from playground_lm.train.loop import train
from playground_lm.train.checkpoint import CheckpointManager
from playground_lm.export_checkpoint import export_checkpoint

from evals.run_eval import (
    evaluate_prompts,
    load_model,
    validation_metrics,
)

from evals.publish_checkpoint import (
    make_row,
    merge_manifest,
    write_json_atomic,
)


MODEL_CONFIG = {
    "vocab_size": 128,
    "n_layers": 2,
    "n_heads": 4,
    "d_model": 64,
    "context_len": 64,
    "dropout": 0.0,
}

TRAIN_CONFIG = {
    "batch_size": 2,
    "grad_accum_steps": 1,
    "context_len": 64,
    "lr": 1e-3,
    "warmup_steps": 1,
    "grad_clip": 1.0,
}


class TinyTokenizer:
    """Deterministic ASCII tokenizer for offline smoke tests.

    Deliberately not GPT-2 BPE. Tests the tokenizer interface
    without requiring downloads. All IDs fit the tiny model.
    """

    eot_token_id = 127

    def encode(self, text: str) -> list[int]:
        if any(ord(char) >= 127 for char in text):
            raise ValueError(
                "TinyTokenizer supports ASCII only"
            )
        return [ord(char) for char in text]

    def decode(self, ids: list[int]) -> str:
        return "".join(chr(token) for token in ids)


def write_synthetic_data(
    directory: Path,
) -> tuple[Path, Path]:
    """Create deterministic training and validation token files."""
    directory.mkdir(
        parents=True,
        exist_ok=True,
    )

    rng = np.random.default_rng(2026)

    train_tokens = rng.integers(
        low=1,
        high=100,
        size=1025,
        dtype=np.uint16,
    )

    val_tokens = rng.integers(
        low=1,
        high=100,
        size=257,
        dtype=np.uint16,
    )

    train_path = directory / "train.bin"
    val_path = directory / "val.bin"

    train_tokens.astype("<u2").tofile(train_path)
    val_tokens.astype("<u2").tofile(val_path)

    assert train_path.stat().st_size == 2050
    assert val_path.stat().st_size == 514

    return train_path, val_path


def make_mock_benchmark_reports():
    """Synthetic benchmark fixtures for schema testing ONLY.

    These values are fabricated, not measured benchmark scores.
    The official counts are present because make_row() requires
    complete reports.
    """
    hellaswag = {
        "dataset": "MOCK_HELLASWAG",
        "split": "validation",
        "metric": "accuracy_normalized_completion_nll",
        "complete": True,
        "expected_full_examples": 10042,
        "examples": 10042,
        "accuracy": 0.25,
        "correct": 2510,
    }

    lambada = {
        "dataset": "MOCK_LAMBADA",
        "split": "test",
        "metric": "exact_last_word_token_accuracy",
        "complete": True,
        "expected_full_examples": 5153,
        "examples": 5153,
        "last_word_accuracy": 0.10,
        "correct": 515,
    }

    return hellaswag, lambada


def test_tiny_checkpoint_end_to_end(tmp_path: Path):
    """Train, save, reload, evaluate, export and preview."""
    torch.manual_seed(2026)
    torch.set_num_threads(1)

    # ------------------------------------------------------------
    # 1. Generate synthetic training and validation data.
    # ------------------------------------------------------------
    data_dir = tmp_path / "data"

    train_bin, val_bin = write_synthetic_data(
        data_dir
    )

    assert train_bin.is_file()
    assert val_bin.is_file()

    # ------------------------------------------------------------
    # 2. Construct the actual PlaygroundLM architecture.
    # ------------------------------------------------------------
    model = PlaygroundLM(**MODEL_CONFIG).cpu()

    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=TRAIN_CONFIG["lr"],
        weight_decay=0.0,
    )

    train_loader = build_dataloader(
        train_bin,
        context_len=64,
        batch_size=2,
        shuffle=True,
        stride=1,
        seed=2026,
    )

    val_loader = build_dataloader(
        val_bin,
        context_len=64,
        batch_size=2,
        shuffle=False,
        drop_last=False,
    )

    # ------------------------------------------------------------
    # 3. Train with the real training loop for three CPU updates.
    # ------------------------------------------------------------
    run_dir = tmp_path / "run"

    manager = CheckpointManager(
        run_dir=run_dir,
        run_id="e2e-tiny",
        config={
            "model": MODEL_CONFIG,
            "train": TRAIN_CONFIG,
        },
        interval_s=3600,
        milestones=(3,),
        rolling_keep=2,
    )

    manager.bootstrap()

    train(
        model=model,
        optimizer=optimizer,
        train_loader=train_loader,
        val_loader=val_loader,
        checkpoint_manager=manager,
        config=TRAIN_CONFIG,
        max_steps=3,
        device="cpu",
    )

    metrics_path = run_dir / "metrics.jsonl"
    training_manifest_path = run_dir / "manifest.json"

    assert metrics_path.is_file()
    assert training_manifest_path.is_file()

    metrics = [
        json.loads(line)
        for line in metrics_path.read_text().splitlines()
    ]

    assert len(metrics) == 3
    assert [row["step"] for row in metrics] == [
        1, 2, 3
    ]
    assert all(
        math.isfinite(row["loss"])
        for row in metrics
    )

    training_manifest = json.loads(
        training_manifest_path.read_text()
    )

    assert training_manifest["latest_step"] == 3
    assert training_manifest["config"]["model"] == MODEL_CONFIG

    entries = manager.list_checkpoints()
    assert entries

    last_checkpoint = entries[-1]
    assert last_checkpoint.step == 3

    weights_path = run_dir / last_checkpoint.path
    state_path = (
        run_dir
        / "checkpoints"
        / "step_00000003_state.pt"
    )

    assert weights_path.is_file()
    assert weights_path.suffix == ".safetensors"
    assert state_path.is_file()

    # ------------------------------------------------------------
    # 4. Load Dee's split safetensors checkpoint DIRECTLY.
    #
    # The loader discovers model kwargs from:
    #     run/manifest.json -> config["model"]
    #
    # No conversion or temporary combined .pt file is needed.
    # ------------------------------------------------------------
    loaded_model = load_model(
        "playground",
        str(weights_path),
        device="cpu",
        factory="playground_lm.model:PlaygroundLM",
    )

    assert isinstance(loaded_model, PlaygroundLM)
    assert loaded_model.context_len == 64
    assert loaded_model.vocab_size == 128
    assert not loaded_model.training

    # Confirm that the checkpoint preserved trained parameters.
    for original, restored in zip(
        model.parameters(),
        loaded_model.parameters(),
    ):
        torch.testing.assert_close(
            original.detach(),
            restored.detach(),
        )

    # Check numerical predictions as well as parameter equality.
    probe_ids = torch.tensor(
        [[1, 2, 3, 4]],
        dtype=torch.long,
    )

    with torch.inference_mode():
        model.eval()
        original_logits, _ = model(probe_ids)
        restored_logits, _ = loaded_model(probe_ids)

    torch.testing.assert_close(
        original_logits,
        restored_logits,
    )

    # ------------------------------------------------------------
    # 5. Export the same original safetensors checkpoint.
    #
    # export_checkpoint() now understands Dee's split format,
    # using the same config discovery as run_eval.load_model().
    # ------------------------------------------------------------
    export_dir = tmp_path / "weights_export"

    export_manifest = export_checkpoint(
        weights_path,
        export_dir,
    )

    exported_weights = (
        export_dir / "model.safetensors"
    )
    exported_manifest = (
        export_dir / "manifest.json"
    )

    assert exported_weights.is_file()
    assert exported_manifest.is_file()
    assert export_manifest["tensor_count"] > 0

    assert (
        export_manifest["model_config"]
        == MODEL_CONFIG
    )

    exported_state = safetensors.load_file(
        str(exported_weights)
    )

    assert "token_embedding.weight" in exported_state

    original_state = safetensors.load_file(
        str(weights_path)
    )

    assert set(exported_state) == set(original_state)

    for name, tensor in original_state.items():
        torch.testing.assert_close(
            tensor,
            exported_state[name],
        )

    # ------------------------------------------------------------
    # 6. Run the real validation loss/perplexity evaluator.
    # ------------------------------------------------------------
    validation = validation_metrics(
        loaded_model,
        val_bin,
        context_len=64,
        batch_size=2,
        device="cpu",
    )

    assert validation["complete"] is True
    assert validation["tokens_scored"] == 256
    assert validation["available_targets"] == 256

    assert math.isfinite(validation["loss"])
    assert math.isfinite(validation["perplexity"])
    assert validation["perplexity"] >= 1.0

    assert validation["perplexity"] == pytest.approx(
        math.exp(validation["loss"]),
        rel=1e-5,
    )

    # ------------------------------------------------------------
    # 7. Run held-out generation with an offline tokenizer.
    # ------------------------------------------------------------
    prompts_path = tmp_path / "prompts.json"

    prompts_path.write_text(
        json.dumps(
            {
                "version": 1,
                "held_out_probes": [
                    {
                        "id": "tiny_prompt_01",
                        "category": "smoke",
                        "prompt": "Hello ",
                        "evaluation": "coherent_continuation",
                    },
                    {
                        "id": "tiny_prompt_02",
                        "category": "smoke",
                        "prompt": "A tiny model says ",
                        "evaluation": "coherent_continuation",
                    },
                ],
                "demo_riddles": [],
            }
        ),
        encoding="utf-8",
    )

    generations, token_lists = evaluate_prompts(
        loaded_model,
        prompts_path,
        tokenizer=TinyTokenizer(),
        context_len=64,
        max_new_tokens=3,
        device="cpu",
        temperature=0.0,
        top_k=0,
        seed=2026,
    )

    assert len(generations) == 2
    assert len(token_lists) == 2

    assert all(
        isinstance(item["generation"], str)
        for item in generations
    )

    assert all(
        len(ids) <= 3
        for ids in token_lists
    )

    evaluation_path = tmp_path / "eval_results.json"

    write_json_atomic(
        evaluation_path,
        {
            "schema_version": 1,
            "model": "e2e-tiny",
            "backend": "playground",
            "validation": validation,
            "generations": generations,
        },
    )

    assert evaluation_path.is_file()

    evaluation_report = json.loads(
        evaluation_path.read_text()
    )

    assert evaluation_report["validation"]["complete"]

    # ------------------------------------------------------------
    # 8. Export ONNX if dependencies are installed.
    #
    # Pass the ORIGINAL safetensors checkpoint for provenance.
    # ------------------------------------------------------------
    onnx_available = (
        importlib.util.find_spec("onnx") is not None
        and importlib.util.find_spec("onnxruntime") is not None
    )

    if onnx_available:
        from evals.export_onnx import export_bundle

        onnx_dir = tmp_path / "onnx_export"

        onnx_manifest = export_bundle(
            loaded_model,
            onnx_dir,
            checkpoint=weights_path,
            context_len=64,
            vocab_size=128,
            sample_length=4,
            quantize=False,
        )

        assert (onnx_dir / "model.onnx").is_file()
        assert (onnx_dir / "manifest.json").is_file()

        assert (
            onnx_manifest["artifacts"]["fp32"]
            ["validation"]["passed"]
        )
        assert (
            onnx_manifest["tokenizer"]["vocab_size"]
            == 128
        )
        assert (
            "checkpoint_sha256" in onnx_manifest
        )

    # ------------------------------------------------------------
    # 9. Preview benchmark publication in memory.
    #
    # These are MOCKED benchmark fixtures.
    # No official HellaSwag/LAMBADA run happens here.
    # No production manifest is modified.
    # ------------------------------------------------------------
    hellaswag, lambada = make_mock_benchmark_reports()

    preview_row = make_row(
        "e2e-tiny",
        "Tiny CPU integration smoke test",
        validation,
        hellaswag,
        lambada,
    )

    demo_manifest = {
        "model": "PlaygroundLM",
        "checkpoints": [],
        "benchmarks": {
            "columns": [
                "val_loss",
                "perplexity",
                "hellaswag_acc_norm",
                "lambada_acc",
            ],
            "rows": [],
        },
        "riddle_prompts": [],
    }

    original = copy.deepcopy(demo_manifest)

    preview_manifest = merge_manifest(
        demo_manifest,
        preview_row,
    )

    assert demo_manifest == original
    assert len(
        preview_manifest["benchmarks"]["rows"]
    ) == 1

    assert (
        preview_manifest["benchmarks"]["rows"][0]["id"]
        == "e2e-tiny"
    )
    assert preview_row["val_loss"] == pytest.approx(
        validation["loss"]
    )

    # A local test artifact, explicitly marked non-publishable.
    preview_path = (
        tmp_path / "publication_dry_run.json"
    )

    write_json_atomic(
        preview_path,
        {
            "dry_run": True,
            "mocked_benchmarks": True,
            "publishable": False,
            "manifest_preview": preview_manifest,
        },
    )

    assert preview_path.is_file()

    preview = json.loads(
        preview_path.read_text()
    )

    assert preview["dry_run"] is True
    assert preview["mocked_benchmarks"] is True
    assert preview["publishable"] is False

    # ------------------------------------------------------------
    # 10. Confirm every expected pipeline artifact exists.
    #
    # Notice there is NO combined_path / tiny_eval.pt.
    # ------------------------------------------------------------
    expected_files = (
        train_bin,
        val_bin,
        metrics_path,
        training_manifest_path,
        weights_path,
        state_path,
        exported_weights,
        exported_manifest,
        evaluation_path,
        preview_path,
    )

    assert all(
        path.is_file()
        for path in expected_files
    )

    # No conversion artifact should exist.
    assert not (tmp_path / "tiny_eval.pt").exists()
