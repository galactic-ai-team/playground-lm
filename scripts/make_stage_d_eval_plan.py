#!/usr/bin/env python3
"""Generate a deterministic Stage D fine-tuning evaluation plan.

Reads the actual uint16 token count from train.bin, or accepts an
explicit count. Never invents training metrics or benchmark scores.

Usage:
    python -m scripts.make_stage_d_eval_plan \
      --train-bin data/stage_d/train.bin \
      --target-train-tokens 1000000

    python -m scripts.make_stage_d_eval_plan \
      --train-tokens 18000 \
      --target-train-tokens 1000000 \
      --benchmark-evals 10 \
      --early-stop-patience 3

Optional:
    --output evals/results/stage_d_eval_plan.json
"""

from __future__ import annotations

import argparse
import json
import math
from pathlib import Path
from typing import Any


def count_uint16_tokens(path: str | Path) -> int:
    """Count tokens without loading the training corpus into RAM."""
    path = Path(path)
    size = path.stat().st_size

    if size % 2:
        raise ValueError(
            f"Training file is not uint16-aligned: {path}"
        )
    if size == 0:
        raise ValueError("Training file is empty")

    return size // 2


def positive_int(value: int, name: str) -> int:
    if isinstance(value, bool) or not isinstance(value, int):
        raise ValueError(f"{name} must be an integer")
    if value <= 0:
        raise ValueError(f"{name} must be positive")
    return value


def make_plan(
    *,
    train_tokens: int,
    target_train_tokens: int,
    benchmark_evals: int = 10,
    validation_every_epochs: int = 1,
    early_stop_patience: int = 3,
    min_delta: float = 0.0,
) -> dict[str, Any]:
    """Create an advisory schedule based on real token counts.

    Epoch estimates assume one full pass over train.bin per epoch.
    Actual optimizer-token accounting depends on data-loader
    batching, context windows, and dropped incomplete batches.
    """
    positive_int(train_tokens, "train_tokens")
    positive_int(
        target_train_tokens, "target_train_tokens"
    )
    positive_int(benchmark_evals, "benchmark_evals")
    positive_int(
        validation_every_epochs,
        "validation_every_epochs",
    )
    positive_int(
        early_stop_patience, "early_stop_patience"
    )

    if (
        isinstance(min_delta, bool)
        or not isinstance(min_delta, (int, float))
        or not math.isfinite(min_delta)
        or min_delta < 0
    ):
        raise ValueError(
            "min_delta must be finite and nonnegative"
        )

    epochs = math.ceil(
        target_train_tokens / train_tokens
    )

    estimated_total_tokens = epochs * train_tokens
    overshoot = (
        estimated_total_tokens - target_train_tokens
    )

    # Spread expensive full benchmarks across the run.
    benchmark_every = max(
        1, math.ceil(epochs / benchmark_evals)
    )

    def schedule(interval: int) -> list[int]:
        checkpoints = list(
            range(interval, epochs + 1, interval)
        )
        if not checkpoints or checkpoints[-1] != epochs:
            checkpoints.append(epochs)
        return checkpoints

    validation_epochs = schedule(
        validation_every_epochs
    )
    benchmark_epochs = schedule(benchmark_every)

    # One schedule for the full downstream evaluation and
    # exact-copy audit makes checkpoint comparisons consistent.
    return {
        "schema_version": 1,
        "stage": "D",
        "inputs": {
            "train_tokens_per_epoch": train_tokens,
            "target_train_tokens": target_train_tokens,
        },
        "training": {
            "suggested_epochs": epochs,
            "estimated_tokens_processed": estimated_total_tokens,
            "estimated_target_overshoot_tokens": overshoot,
            "note": (
                "Epochs are estimated from corpus tokens. "
                "Use the training loop's tokens_seen counter "
                "for actual consumption."
            ),
        },
        "evaluation": {
            "validation": {
                "every_epochs": validation_every_epochs,
                "epochs": validation_epochs,
                "metric": "held_out_validation_loss",
            },
            "hellaswag": {
                "every_epochs": benchmark_every,
                "epochs": benchmark_epochs,
                "examples_required": 10042,
                "metric": "accuracy_normalized_completion_nll",
            },
            "lambada": {
                "every_epochs": benchmark_every,
                "epochs": benchmark_epochs,
                "examples_required": 5153,
                "metric": "exact_last_word_token_accuracy",
            },
            "verbatim_copy": {
                "every_epochs": benchmark_every,
                "epochs": benchmark_epochs,
                "threshold_tokens": 50,
                "require_full_training_corpus": True,
                "note": (
                    "Check generated continuations against "
                    "every Stage D training shard. Also consider "
                    "checking against pretraining data when available. "
                    "Flag any exact 50-token overlap. "
                    "Generations shorter than 50 tokens "
                    "are inconclusive."
                ),
            },
            "recommended_checks": [
                "Evaluate validation loss after each scheduled epoch.",
                "Run full HellaSwag and LAMBADA at benchmark epochs.",
                "Generate held-out prompts with fixed seeds.",
                "Run the 50-token exact-copy audit on those generations.",
                "Save the best validation checkpoint independently.",
                "Retain the last checkpoint for debugging.",
                "Never publish partial benchmark results as full scores.",
            ],
        },
        "early_stop": {
            "enabled": True,
            "metric": "held_out_validation_loss",
            "mode": "min",
            "rule": "consecutive_validation_increases",
            "patience_evaluations": early_stop_patience,
            "min_delta": float(min_delta),
            "description": (
                "Stop if validation loss increases by more than "
                "min_delta compared with the immediately preceding "
                "validation evaluation for patience_evaluations "
                "consecutive checks. Reset the counter when loss "
                "stabilizes or decreases. Preserve the best checkpoint."
            ),
        },
    }


def should_early_stop(
    validation_losses: list[float],
    *,
    patience: int = 3,
    min_delta: float = 0.0,
) -> bool:
    """Apply the exact consecutive-rise rule described in the plan."""
    positive_int(patience, "patience")

    if (
        isinstance(min_delta, bool)
        or not isinstance(min_delta, (int, float))
        or not math.isfinite(min_delta)
        or min_delta < 0
    ):
        raise ValueError("Invalid min_delta")

    if any(
        isinstance(loss, bool)
        or not isinstance(loss, (int, float))
        or not math.isfinite(loss)
        for loss in validation_losses
    ):
        raise ValueError("Validation losses must be finite numbers")

    consecutive_rises = 0

    for previous, current in zip(
        validation_losses,
        validation_losses[1:],
    ):
        if current > previous + min_delta:
            consecutive_rises += 1
        else:
            consecutive_rises = 0

    return consecutive_rises >= patience


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)

    source = parser.add_mutually_exclusive_group(required=True)
    source.add_argument(
        "--train-bin",
        type=Path,
        help="Stage D training tokens stored as uint16",
    )
    source.add_argument(
        "--train-tokens",
        type=int,
        help="Actual token count when train.bin is unavailable",
    )

    parser.add_argument(
        "--target-train-tokens",
        required=True,
        type=int,
    )
    parser.add_argument(
        "--benchmark-evals",
        type=int,
        default=10,
        help="Approximate number of full benchmark evaluations",
    )
    parser.add_argument(
        "--validation-every-epochs",
        type=int,
        default=1,
    )
    parser.add_argument(
        "--early-stop-patience",
        type=int,
        default=3,
    )
    parser.add_argument(
        "--min-delta",
        type=float,
        default=0.0,
    )
    parser.add_argument(
        "--output",
        type=Path,
        help="Optional JSON output file",
    )

    args = parser.parse_args()

    try:
        train_tokens = (
            count_uint16_tokens(args.train_bin)
            if args.train_bin is not None
            else args.train_tokens
        )

        plan = make_plan(
            train_tokens=train_tokens,
            target_train_tokens=args.target_train_tokens,
            benchmark_evals=args.benchmark_evals,
            validation_every_epochs=args.validation_every_epochs,
            early_stop_patience=args.early_stop_patience,
            min_delta=args.min_delta,
        )
    except (ValueError, OSError) as exc:
        parser.error(str(exc))

    serialized = json.dumps(
        plan,
        indent=2,
        ensure_ascii=False,
        allow_nan=False,
    ) + "\n"

    if args.output is not None:
        args.output.parent.mkdir(
            parents=True,
            exist_ok=True,
        )
        args.output.write_text(
            serialized,
            encoding="utf-8",
        )

    print(serialized, end="")


if __name__ == "__main__":
    main()
