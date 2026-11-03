#!/usr/bin/env python3
"""Export a trained causal LM to ONNX, with optional INT8 weights.

Uses the existing evals.run_eval.load_model() contract.

Example:
    python -m evals.export_onnx \
      --checkpoint checkpoints/step_10000.pt \
      --factory playground_lm.model:PlaygroundLM \
      --output-dir exports/step_10000 \
      --quantize

Outputs:
    model.onnx
    model.int8.onnx             (when --quantize)
    manifest.json

FP32 numerical agreement and ONNX Runtime execution are verified
before publication. Quantized inference is checked for finite outputs,
but must receive additional browser and quality verification.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
from pathlib import Path
from typing import Any

import numpy as np
import torch

from evals.run_eval import get_logits, load_model
from playground_lm.tokenizer import VOCAB_SIZE, EOT_ID

OPSET = 17


class LogitsOnly(torch.nn.Module):
    """Remove the training-only loss output from the exported graph."""

    def __init__(self, model: torch.nn.Module):
        super().__init__()
        self.model = model

    def forward(self, input_ids: torch.Tensor) -> torch.Tensor:
        output = self.model(input_ids)

        if isinstance(output, torch.Tensor):
            return output
        if isinstance(output, (tuple, list)):
            return output[0]
        if hasattr(output, "logits"):
            return output.logits
        raise TypeError("Unsupported model output")


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(
            lambda: stream.read(1024 * 1024), b""
        ):
            digest.update(chunk)
    return digest.hexdigest()


def artifact_info(path: Path) -> dict[str, Any]:
    return {
        "file": path.name,
        "bytes": path.stat().st_size,
        "sha256": sha256_file(path),
    }


def export_fp32(
    model: torch.nn.Module,
    path: Path,
    *,
    sample_length: int = 8,
    opset: int = OPSET,
) -> None:
    """Export batch/sequence-dynamic logits-only graph."""
    if sample_length < 2:
        raise ValueError("sample_length must be >= 2")

    wrapper = LogitsOnly(model.cpu().eval())
    example = torch.randint(
        low=0,
        high=min(100, VOCAB_SIZE),
        size=(1, sample_length),
        dtype=torch.long,
    )

    with torch.inference_mode():
        torch.onnx.export(
            wrapper,
            (example,),
            str(path),
            export_params=True,
            opset_version=opset,
            input_names=["input_ids"],
            output_names=["logits"],
            dynamic_axes={
                "input_ids": {
                    0: "batch",
                    1: "sequence",
                },
                "logits": {
                    0: "batch",
                    1: "sequence",
                },
            },
            do_constant_folding=True,
            dynamo=False,
        )


def check_onnx(
    path: Path,
    model: torch.nn.Module | None = None,
    *,
    lengths: tuple[int, ...] = (1, 8),
    vocab_size: int = VOCAB_SIZE,
    atol: float = 1e-3,
    rtol: float = 1e-3,
) -> dict[str, Any]:
    """Check graph and inference on multiple sequence lengths.

    When model is supplied, compare FP32 ONNX logits with PyTorch.
    """
    import onnx
    import onnxruntime as ort

    graph = onnx.load(str(path), load_external_data=True)
    onnx.checker.check_model(graph)

    session = ort.InferenceSession(
        str(path),
        providers=["CPUExecutionProvider"],
    )

    inputs = session.get_inputs()
    outputs = session.get_outputs()

    if (
        len(inputs) != 1
        or inputs[0].name != "input_ids"
        or len(outputs) != 1
        or outputs[0].name != "logits"
    ):
        raise ValueError("Unexpected ONNX input/output contract")

    max_abs_error = 0.0
    lengths_checked = []

    for length in lengths:
        if length <= 0:
            raise ValueError("Sequence lengths must be positive")

        # Deterministic, in-vocabulary input IDs.
        ids = (
            np.arange(length, dtype=np.int64) % 100
        ).reshape(1, length)

        actual = session.run(
            ["logits"],
            {"input_ids": ids},
        )[0]

        if actual.shape != (1, length, vocab_size):
            raise ValueError(
                f"Incorrect logits shape: {actual.shape}"
            )

        if not np.isfinite(actual).all():
            raise ValueError("ONNX produced nonfinite logits")

        if model is not None:
            with torch.inference_mode():
                expected = get_logits(
                    model.cpu().eval(),
                    torch.from_numpy(ids),
                ).detach().float().numpy()

            error = float(
                np.max(np.abs(actual - expected))
            )
            max_abs_error = max(max_abs_error, error)

            np.testing.assert_allclose(
                actual,
                expected,
                atol=atol,
                rtol=rtol,
                err_msg=f"ONNX/PyTorch mismatch at T={length}",
            )

        lengths_checked.append(length)

    return {
        "provider": "CPUExecutionProvider",
        "sequence_lengths": lengths_checked,
        "pytorch_compared": model is not None,
        "max_abs_error": (
            max_abs_error if model is not None else None
        ),
        "passed": True,
    }


def quantize_model(
    source: Path,
    destination: Path,
) -> None:
    """Apply dynamic, weight-oriented INT8 quantization."""
    from onnxruntime.quantization import (
        QuantType,
        quantize_dynamic,
    )

    quantize_dynamic(
        model_input=str(source),
        model_output=str(destination),
        weight_type=QuantType.QInt8,
        per_channel=False,
        op_types_to_quantize=["MatMul", "Gemm"],
    )


def export_bundle(
    model: torch.nn.Module,
    output_dir: str | Path,
    *,
    checkpoint: str | Path | None = None,
    context_len: int = 1024,
    vocab_size: int = VOCAB_SIZE,
    sample_length: int = 8,
    quantize: bool = True,
) -> dict[str, Any]:
    """Export, validate, then publish the bundle manifest."""
    if context_len <= 0:
        raise ValueError("context_len must be positive")
    if not 2 <= sample_length <= context_len:
        raise ValueError("Invalid sample_length")
    if vocab_size <= 0:
        raise ValueError("vocab_size must be positive")

    destination = Path(output_dir)
    destination.mkdir(parents=True, exist_ok=True)

    fp32_path = destination / "model.onnx"
    int8_path = destination / "model.int8.onnx"
    manifest_path = destination / "manifest.json"

    # Avoid leaving a stale manifest when re-exporting fails.
    manifest_path.unlink(missing_ok=True)

    export_fp32(
        model,
        fp32_path,
        sample_length=sample_length,
    )

    fp32_check = check_onnx(
        fp32_path,
        model,
        lengths=tuple(dict.fromkeys(
            (1, sample_length)
        )),
        vocab_size=vocab_size,
    )

    artifacts = {
        "fp32": {
            **artifact_info(fp32_path),
            "validation": fp32_check,
        }
    }

    if quantize:
        quantize_model(fp32_path, int8_path)

        int8_check = check_onnx(
            int8_path,
            lengths=tuple(dict.fromkeys(
                (1, sample_length)
            )),
            vocab_size=vocab_size,
        )

        artifacts["int8"] = {
            **artifact_info(int8_path),
            "validation": int8_check,
            "quantization": "dynamic_qint8",
            "browser_verified": False,
            "quality_verified": False,
        }

    manifest: dict[str, Any] = {
        "format_version": 1,
        "architecture": "PlaygroundLM",
        "artifact_type": "onnx_inference",
        "tokenizer": {
            "name": "gpt2",
            "vocab_size": vocab_size,
            "eot_token_id": EOT_ID,
        },
        "runtime": {
            "format": "onnx",
            "opset": OPSET,
            "input_name": "input_ids",
            "input_dtype": "int64",
            "output_name": "logits",
            "output_dtype": "float32",
            "context_length": context_len,
            "batch_dynamic": True,
            "sequence_dynamic": True,
            "uses_kv_cache": False,
        },
        "artifacts": artifacts,
    }

    if checkpoint is not None:
        manifest["checkpoint_sha256"] = sha256_file(
            Path(checkpoint)
        )

    manifest_path.write_text(
        json.dumps(
            manifest,
            indent=2,
            sort_keys=True,
            allow_nan=False,
        ) + "\n",
        encoding="utf-8",
    )
    return manifest


def main() -> None:
    parser = argparse.ArgumentParser(
        description=__doc__,
    )
    parser.add_argument(
        "--checkpoint",
        required=True,
    )
    parser.add_argument(
        "--factory",
        default="playground_lm.model:PlaygroundLM",
    )
    parser.add_argument("--config")
    parser.add_argument(
        "--output-dir",
        required=True,
    )
    parser.add_argument(
        "--context-len",
        type=int,
        default=1024,
    )
    parser.add_argument(
        "--sample-length",
        type=int,
        default=8,
    )
    parser.add_argument(
        "--quantize",
        action="store_true",
        help="Also emit dynamically quantized INT8 ONNX",
    )
    args = parser.parse_args()

    model = load_model(
        "playground",
        args.checkpoint,
        device="cpu",
        factory=args.factory,
        config_path=args.config,
    )

    manifest = export_bundle(
        model,
        args.output_dir,
        checkpoint=args.checkpoint,
        context_len=args.context_len,
        sample_length=args.sample_length,
        quantize=args.quantize,
    )

    print(
        json.dumps(
            {
                "output_dir": args.output_dir,
                "artifacts": manifest["artifacts"],
            },
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
