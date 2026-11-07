#!/usr/bin/env python3
"""Upsert one measured Stage D checkpoint into demo/benchmarks.json.

Requires a complete scripts/eval_checkpoint.py-style report with
full HellaSwag/LAMBADA and validation-token coverage.

Example:
    python -m scripts.add_stage_d_benchmark \
      --eval evals/results/stage_d_31104.json \
      --step 31104 \
      --benchmarks demo/benchmarks.json

Does not fabricate scores or modify existing benchmark rows.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any, Mapping

from scripts.make_benchmark_table import (
    COLUMNS,
    extract_row,
    write_atomic,
)

VALIDATION_NOTE = (
    "Validation loss and perplexity use the held-out Playground "
    "Stage D finetuning validation split, NOT FineWeb-Edu. "
    "Do not compare these values directly with Stage B/C or GPT-2 "
    "FineWeb-Edu validation results."
)


def add_stage_d_row(
    table: Mapping[str, Any],
    report: Mapping[str, Any],
    step: int,
) -> dict[str, Any]:
    """Idempotently add or replace one Stage D benchmark row."""
    if (
        isinstance(step, bool)
        or not isinstance(step, int)
        or step <= 0
    ):
        raise ValueError("step must be a positive integer")

    if list(table.get("columns", [])) != list(COLUMNS):
        raise ValueError("Unexpected benchmark columns")

    rows = table.get("rows")
    if not isinstance(rows, list) or any(
        not isinstance(r, dict) for r in rows
    ):
        raise ValueError(
            "benchmarks.rows must be a list of objects"
        )

    expected_id = f"step_{step:08d}"

    if any(
        r.get("step") == step and r.get("id") != expected_id
        for r in rows
    ):
        raise ValueError(
            "Step already belongs to another benchmark row"
        )

    report_id = report.get("checkpoint_id")
    if report_id is not None and report_id not in (
        expected_id,
        f"stage_d_{step}",
        f"step_{step}",
    ):
        raise ValueError(
            "Checkpoint report ID does not match --step"
        )

    # Reuse the existing strict evaluator contract.
    # This rejects partial datasets, incomplete validation,
    # nonfinite values, and inconsistent loss/perplexity.
    row = extract_row(
        report,
        checkpoint_id=expected_id,
        label=f"PlaygroundLM Stage D finetune — Step {step:,}",
        step=step,
    )

    row["validation_split"] = "stage_d_playground_finetune"
    row["validation_note"] = VALIDATION_NOTE

    updated_rows = [
        dict(r)
        for r in rows
        if r.get("id") != expected_id
    ]
    updated_rows.append(row)

    # Ascending checkpoint steps; GPT-2 baseline last.
    updated_rows.sort(
        key=lambda r: (
            r.get("step") is None,
            (
                r.get("step")
                if isinstance(r.get("step"), int)
                else 0
            ),
            str(r.get("id", "")),
        )
    )

    updated = dict(table)
    updated["rows"] = updated_rows

    # Once Stage D is included, the first column group can
    # no longer accurately be called FineWeb-Edu validation.
    groups = updated.get("column_groups")

    if isinstance(groups, list):
        updated_groups = [
            dict(group) for group in groups
        ]

        for group in updated_groups:
            if group.get("columns") == [
                "val_loss",
                "perplexity",
            ]:
                group["label"] = "Validation (split-specific)"
                group["description"] = (
                    "Stage B/C use their respective FineWeb-Edu "
                    "validation splits; Stage D uses held-out "
                    "Playground finetuning data. Compare loss/PPL "
                    "only within a shared validation split."
                )

        updated["column_groups"] = updated_groups

    existing_note = str(
        updated.get("notes", "")
    ).strip()

    if VALIDATION_NOTE not in existing_note:
        updated["notes"] = (
            existing_note + " " + VALIDATION_NOTE
        ).strip()

    return updated


def main() -> None:
    parser = argparse.ArgumentParser(
        description=__doc__
    )
    parser.add_argument(
        "--eval",
        type=Path,
        required=True,
    )
    parser.add_argument(
        "--step",
        type=int,
        required=True,
    )
    parser.add_argument(
        "--benchmarks",
        type=Path,
        default=Path("demo/benchmarks.json"),
    )
    parser.add_argument(
        "--output",
        type=Path,
        help="Defaults to --benchmarks",
    )

    args = parser.parse_args()

    try:
        report = json.loads(
            args.eval.read_text(encoding="utf-8")
        )
        table = json.loads(
            args.benchmarks.read_text(encoding="utf-8")
        )

        if not isinstance(report, dict) or not isinstance(
            table, dict
        ):
            raise ValueError(
                "Both inputs must be JSON objects"
            )

        result = add_stage_d_row(
            table,
            report,
            args.step,
        )

        output = args.output or args.benchmarks

        write_atomic(
            output,
            json.dumps(
                result,
                indent=2,
                ensure_ascii=False,
                allow_nan=False,
            ) + "\n",
        )

    except (
        OSError,
        ValueError,
        KeyError,
        TypeError,
    ) as exc:
        parser.error(str(exc))

    print(
        f"Wrote Stage D step {args.step} to {output}"
    )


if __name__ == "__main__":
    main()
