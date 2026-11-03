#!/usr/bin/env python3
"""Run the original GPT-2 124M through PlaygroundLM benchmarks.

Uses exactly the same scoring functions and validation data as
PlaygroundLM. Produces a detailed JSON report and optionally upserts
a demo/manifest.json benchmark row.

Example:
    python -m evals.baseline_gpt2 \
      --val-bin data/val.bin \
      --device cuda \
      --output evals/results/gpt2_124m.json \
      --publish

A full run is required. No evaluation limits are exposed.
The HF model is loaded once and reused for all three metrics.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

from evals.benchmarks import (
    load_rows,
    score_hellaswag,
    score_lambada,
)
from evals.publish_checkpoint import (
    make_row,
    publish_row,
    write_json_atomic,
)
from evals.run_eval import load_model, validation_metrics
from playground_lm.tokenizer import get_tokenizer

BASELINE_MODEL = "gpt2"
BASELINE_ID = "gpt2_124m"
BASELINE_LABEL = "OpenAI GPT-2 124M"


def run_baseline(
    *,
    val_bin: str | Path,
    device: str = "cpu",
    context_len: int = 1024,
    batch_size: int = 4,
    offline: bool = False,
) -> dict[str, Any]:
    """Evaluate GPT-2 and return a full, publishable result report.

    Raises if validation or either benchmark is incomplete.
    Does not modify any files.
    """
    if context_len <= 0 or context_len > 1024:
        raise ValueError("context_len must be in [1, 1024]")
    if batch_size <= 0:
        raise ValueError("batch_size must be positive")

    model = load_model(
        "hf",
        BASELINE_MODEL,
        device=device,
    )
    tokenizer = get_tokenizer()

    # Same token-weighted validation scorer as PlaygroundLM.
    validation = validation_metrics(
        model,
        val_bin,
        context_len=context_len,
        batch_size=batch_size,
        device=device,
    )

    # Same benchmark datasets, scoring functions and conventions.
    hellaswag = score_hellaswag(
        model,
        load_rows("hellaswag", offline=offline),
        tokenizer,
        context_len=context_len,
        device=device,
        limit=None,
    )

    lambada = score_lambada(
        model,
        load_rows("lambada", offline=offline),
        tokenizer,
        context_len=context_len,
        device=device,
        limit=None,
    )

    # Reuse the publishing contract. Reject missing, partial,
    # out-of-range and inconsistent metrics before returning.
    row = make_row(
        BASELINE_ID,
        BASELINE_LABEL,
        validation,
        hellaswag,
        lambada,
    )

    return {
        "schema_version": 1,
        "model": BASELINE_MODEL,
        "backend": "hf",
        "checkpoint_id": BASELINE_ID,
        "context_len": context_len,
        "tokenizer": "gpt2",
        "validation": validation,
        "benchmarks": {
            "hellaswag": hellaswag,
            "lambada": lambada,
        },
        "row": row,
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)

    parser.add_argument(
        "--val-bin",
        type=Path,
        required=True,
        help="The SAME held-out val.bin used for PlaygroundLM",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=Path("evals/results/gpt2_124m.json"),
    )
    parser.add_argument(
        "--device",
        default="cpu",
        help="cpu or cuda",
    )
    parser.add_argument(
        "--context-len",
        type=int,
        default=1024,
    )
    parser.add_argument(
        "--batch-size",
        type=int,
        default=4,
    )
    parser.add_argument(
        "--offline",
        action="store_true",
        help="Use locally cached benchmark datasets",
    )
    parser.add_argument(
        "--publish",
        action="store_true",
        help="Upsert baseline row into demo manifest",
    )
    parser.add_argument(
        "--manifest",
        type=Path,
        default=Path("demo/manifest.json"),
    )

    args = parser.parse_args()

    report = run_baseline(
        val_bin=args.val_bin,
        device=args.device,
        context_len=args.context_len,
        batch_size=args.batch_size,
        offline=args.offline,
    )

    # Only full, validated results reach this point.
    # Check output paths before publication.
    if args.publish:
        if (
            args.output.resolve()
            == args.manifest.resolve()
        ):
            parser.error(
                "--output and --manifest must be different files"
            )

        publish_row(
            args.manifest,
            report["row"],
        )

    write_json_atomic(args.output, report)

    print(
        json.dumps(
            {
                "model": BASELINE_LABEL,
                "validation_loss": report["row"]["val_loss"],
                "perplexity": report["row"]["perplexity"],
                "hellaswag_acc_norm": report["row"][
                    "hellaswag_acc_norm"
                ],
                "lambada_acc": report["row"]["lambada_acc"],
                "result_file": str(args.output),
                "manifest_updated": args.publish,
            },
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
