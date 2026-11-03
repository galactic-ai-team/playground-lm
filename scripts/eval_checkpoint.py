#!/usr/bin/env python3
"""Evaluate a split-safetensors PlaygroundLM checkpoint.

Uses the existing run_eval and benchmarks implementations, without
introducing any alternative scoring conventions.

Full runs emit a validated `row` compatible with demo/manifest.json.
Partial runs emit `preview_row`, but `row` is null and publishable is
false. This script never modifies the demo manifest.

Usage:
    python -m scripts.eval_checkpoint \
      --checkpoint runs/plm-b/checkpoints/step_00002035.safetensors \
      --val-bin data/val.bin \
      --id stage_b_2035 \
      --label "Stage B — 200M tokens" \
      --device cuda \
      --output evals/results/stage_b_2035.json

Quick smoke:
    python -m scripts.eval_checkpoint \
      --checkpoint runs/plm-b/checkpoints/step_00002035.safetensors \
      --val-bin data/val.bin \
      --id stage_b_smoke \
      --limit 20 \
      --max-val-batches 2 \
      --device cuda \
      --output evals/results/stage_b_smoke.json
"""

from __future__ import annotations

import argparse
import json
import math
from pathlib import Path
from typing import Any, Mapping

from evals.benchmarks import (
    load_rows,
    score_hellaswag,
    score_lambada,
)
from evals.publish_checkpoint import (
    make_row,
    write_json_atomic,
)
from evals.run_eval import (
    load_model,
    validation_metrics,
)
from playground_lm.tokenizer import get_tokenizer

FACTORY = "playground_lm.model:PlaygroundLM"


def _number(value: Any, name: str) -> float:
    if (
        isinstance(value, bool)
        or not isinstance(value, (int, float))
    ):
        raise ValueError(f"{name} must be numeric")

    value = float(value)
    if not math.isfinite(value):
        raise ValueError(f"{name} must be finite")

    return value


def preview_metrics_row(
    checkpoint_id: str,
    label: str,
    validation: Mapping[str, Any],
    hellaswag: Mapping[str, Any],
    lambada: Mapping[str, Any],
) -> dict[str, Any]:
    """Construct a nonpublishable metrics preview.

    A preview is never added automatically to the demo manifest.
    The numeric columns have exactly the same definitions as the
    production benchmark table.
    """
    if not checkpoint_id.strip() or not label.strip():
        raise ValueError("Checkpoint ID and label are required")

    loss = _number(validation["loss"], "val_loss")
    perplexity = _number(
        validation["perplexity"], "perplexity"
    )
    hs = _number(
        hellaswag["accuracy"], "hellaswag_acc_norm"
    )
    la = _number(
        lambada["last_word_accuracy"], "lambada_acc"
    )

    if loss < 0 or perplexity < 1:
        raise ValueError("Invalid loss or perplexity")
    if not math.isclose(
        math.log(perplexity),
        loss,
        rel_tol=1e-5,
        abs_tol=1e-5,
    ):
        raise ValueError("Perplexity does not match loss")
    if not (0 <= hs <= 1 and 0 <= la <= 1):
        raise ValueError("Benchmark accuracy outside [0, 1]")

    return {
        "id": checkpoint_id,
        "label": label,
        "val_loss": loss,
        "perplexity": perplexity,
        "hellaswag_acc_norm": hs,
        "lambada_acc": la,
    }


def build_report(
    *,
    checkpoint: str | Path,
    checkpoint_id: str,
    label: str,
    context_len: int,
    validation: Mapping[str, Any],
    hellaswag: Mapping[str, Any],
    lambada: Mapping[str, Any],
    max_val_batches: int | None = None,
    limit: int | None = None,
) -> dict[str, Any]:
    """Validate metrics and distinguish full from limited runs."""
    preview = preview_metrics_row(
        checkpoint_id, label, validation, hellaswag, lambada
    )

    full = all(
        item.get("complete") is True
        for item in (validation, hellaswag, lambada)
    )

    if full and limit is None and max_val_batches is None:
        # Existing production gate also verifies official counts
        # and complete validation-token coverage.
        row = make_row(
            checkpoint_id,
            label,
            validation,
            hellaswag,
            lambada,
        )
    else:
        row = None

    return {
        "schema_version": 1,
        "model": str(checkpoint),
        "backend": "playground",
        "checkpoint_id": checkpoint_id,
        "context_len": context_len,
        "tokenizer": "gpt2",
        "status": "complete" if row is not None else "partial",
        "publishable": row is not None,
        "limits": {
            "benchmark_examples": limit,
            "validation_batches": max_val_batches,
        },
        "row": row,
        "preview_row": None if row is not None else preview,
        "validation": dict(validation),
        "benchmarks": {
            "hellaswag": dict(hellaswag),
            "lambada": dict(lambada),
        },
    }


def evaluate_checkpoint(
    *,
    checkpoint: str | Path,
    val_bin: str | Path,
    checkpoint_id: str,
    label: str,
    device: str = "cpu",
    config: str | Path | None = None,
    context_len: int | None = None,
    batch_size: int = 4,
    limit: int | None = None,
    max_val_batches: int | None = None,
    offline: bool = False,
) -> dict[str, Any]:
    """Load once; run all three existing evaluation paths."""
    if limit is not None and limit <= 0:
        raise ValueError("--limit must be positive")
    if max_val_batches is not None and max_val_batches <= 0:
        raise ValueError("--max-val-batches must be positive")
    if batch_size <= 0:
        raise ValueError("--batch-size must be positive")

    model = load_model(
        "playground",
        str(checkpoint),
        factory=FACTORY,
        config_path=str(config) if config is not None else None,
        device=device,
    )

    model_context = getattr(model, "context_len", None)
    if context_len is None:
        if not isinstance(model_context, int):
            raise ValueError(
                "Model has no context_len; supply --context-len"
            )
        context_len = model_context

    if context_len <= 0:
        raise ValueError("--context-len must be positive")
    if (
        model_context is not None
        and context_len > model_context
    ):
        raise ValueError(
            "Requested context length exceeds model context"
        )

    tokenizer = get_tokenizer()

    validation = validation_metrics(
        model,
        val_bin,
        context_len=context_len,
        batch_size=batch_size,
        device=device,
        max_batches=max_val_batches,
    )

    hellaswag = score_hellaswag(
        model,
        load_rows("hellaswag", offline=offline),
        tokenizer,
        context_len=context_len,
        device=device,
        limit=limit,
    )

    lambada = score_lambada(
        model,
        load_rows("lambada", offline=offline),
        tokenizer,
        context_len=context_len,
        device=device,
        limit=limit,
    )

    return build_report(
        checkpoint=checkpoint,
        checkpoint_id=checkpoint_id,
        label=label,
        context_len=context_len,
        validation=validation,
        hellaswag=hellaswag,
        lambada=lambada,
        max_val_batches=max_val_batches,
        limit=limit,
    )


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--checkpoint",
        type=Path,
        required=True,
        help="Dee's step_XXXXXXXX.safetensors checkpoint",
    )
    parser.add_argument(
        "--config",
        type=Path,
        help="Optional model config or run manifest JSON",
    )
    parser.add_argument(
        "--val-bin", type=Path, required=True
    )
    parser.add_argument("--id", required=True)
    parser.add_argument("--label")
    parser.add_argument(
        "--device", default="cpu"
    )
    parser.add_argument(
        "--context-len", type=int
    )
    parser.add_argument(
        "--batch-size", type=int, default=4
    )
    parser.add_argument(
        "--limit",
        type=int,
        help="Max HellaSwag and LAMBADA examples each",
    )
    parser.add_argument(
        "--max-val-batches",
        type=int,
        help="Limit validation batches for smoke testing",
    )
    parser.add_argument(
        "--offline", action="store_true"
    )
    parser.add_argument(
        "--output", type=Path, required=True
    )
    args = parser.parse_args()

    report = evaluate_checkpoint(
        checkpoint=args.checkpoint,
        config=args.config,
        val_bin=args.val_bin,
        checkpoint_id=args.id,
        label=args.label or args.id,
        device=args.device,
        context_len=args.context_len,
        batch_size=args.batch_size,
        limit=args.limit,
        max_val_batches=args.max_val_batches,
        offline=args.offline,
    )

    write_json_atomic(args.output, report)

    print(json.dumps({
        "output": str(args.output),
        "publishable": report["publishable"],
        "row": (
            report["row"]
            if report["publishable"]
            else report["preview_row"]
        ),
        "hellaswag_examples": report["benchmarks"][
            "hellaswag"
        ]["examples"],
        "lambada_examples": report["benchmarks"][
            "lambada"
        ]["examples"],
        "validation_coverage": report["validation"][
            "coverage"
        ],
    }, indent=2))


if __name__ == "__main__":
    main()
