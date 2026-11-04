
#!/usr/bin/env python3
"""Build deterministic JSON and Markdown benchmark comparisons.

Usage:
    python -m scripts.make_benchmark_table \
      --checkpoint 2035=evals/results/stage_b_2035.json \
      --baseline evals/results/gpt2_124m.json \
      --json-output demo/benchmarks.json \
      --markdown-output demo/BENCHMARKS.md

--checkpoint STEP=FILE is repeatable.

Only complete full-dataset reports are accepted. Neither output
modifies demo/manifest.json. The demo currently reads benchmarks
from that manifest, so frontend integration is separate.
"""

from __future__ import annotations

import argparse
import json
import math
import os
import re
import tempfile
from pathlib import Path
from typing import Any, Mapping

COLUMNS = (
    "val_loss",
    "perplexity",
    "hellaswag_acc_norm",
    "lambada_acc",
)
HELLASWAG_COUNT = 10042
LAMBADA_COUNT = 5153


def read_json(path: str | Path) -> dict[str, Any]:
    document = json.loads(
        Path(path).read_text(encoding="utf-8")
    )
    if not isinstance(document, dict):
        raise ValueError(f"Expected JSON object: {path}")
    return document


def finite(value: Any, field: str) -> float:
    if isinstance(value, bool) or not isinstance(
        value, (int, float)
    ):
        raise ValueError(f"{field} must be numeric")
    result = float(value)
    if not math.isfinite(result):
        raise ValueError(f"{field} must be finite")
    return result


def extract_row(
    report: Mapping[str, Any],
    *,
    checkpoint_id: str,
    label: str,
    step: int | None = None,
) -> dict[str, Any]:
    """Extract one full, validated result from an evaluation report.

    Recompute columns from the raw metrics rather than trusting a
    preformatted row. This also supports combined evaluation reports
    that have no `row` field.
    """
    if report.get("publishable") is False:
        raise ValueError("Partial report cannot enter benchmark table")
    if report.get("status") == "partial":
        raise ValueError("Partial report cannot enter benchmark table")

    validation = report.get("validation")
    benchmarks = report.get("benchmarks")

    if not isinstance(validation, Mapping):
        raise ValueError("Missing validation metrics")
    if not isinstance(benchmarks, Mapping):
        raise ValueError("Missing benchmark metrics")

    hellaswag = benchmarks.get("hellaswag")
    lambada = benchmarks.get("lambada")

    if not isinstance(hellaswag, Mapping):
        raise ValueError("Missing HellaSwag metrics")
    if not isinstance(lambada, Mapping):
        raise ValueError("Missing LAMBADA metrics")

    if validation.get("complete") is not True:
        raise ValueError("Incomplete validation")
    if hellaswag.get("complete") is not True:
        raise ValueError("Incomplete HellaSwag")
    if lambada.get("complete") is not True:
        raise ValueError("Incomplete LAMBADA")

    if (
        hellaswag.get("examples") != HELLASWAG_COUNT
        or lambada.get("examples") != LAMBADA_COUNT
    ):
        raise ValueError("Incorrect full benchmark example counts")

    scored = validation.get("tokens_scored")
    available = validation.get("available_targets")
    if (
        isinstance(scored, bool)
        or not isinstance(scored, int)
        or scored <= 0
        or scored != available
    ):
        raise ValueError("Incomplete validation token coverage")

    values = {
        "val_loss": finite(validation.get("loss"), "val_loss"),
        "perplexity": finite(
            validation.get("perplexity"), "perplexity"
        ),
        "hellaswag_acc_norm": finite(
            hellaswag.get("accuracy"), "hellaswag_acc_norm"
        ),
        "lambada_acc": finite(
            lambada.get("last_word_accuracy"), "lambada_acc"
        ),
    }

    if values["val_loss"] < 0 or values["perplexity"] < 1:
        raise ValueError("Invalid validation loss/perplexity")

    if not math.isclose(
        math.log(values["perplexity"]),
        values["val_loss"],
        rel_tol=1e-5,
        abs_tol=1e-5,
    ):
        raise ValueError("Perplexity and loss disagree")

    for key in ("hellaswag_acc_norm", "lambada_acc"):
        if not 0 <= values[key] <= 1:
            raise ValueError(f"{key} must be between zero and one")

    row = {
        "id": checkpoint_id,
        "label": label,
        **values,
    }

    if step is not None:
        row["step"] = step

    # Fail on mismatched duplicate values rather than publishing
    # conflicting row and raw-metrics representations.
    existing = report.get("row")
    if existing is not None:
        if not isinstance(existing, Mapping):
            raise ValueError("report.row must be an object or null")
        for key in COLUMNS:
            if key in existing:
                other = finite(existing[key], f"row.{key}")
                if not math.isclose(
                    row[key], other, rel_tol=1e-8, abs_tol=1e-8
                ):
                    raise ValueError(
                        f"Report row disagrees with raw metric: {key}"
                    )

    return row


def parse_checkpoint_pairs(
    values: list[str],
) -> list[tuple[int, Path]]:
    pairs = []
    seen = set()

    for item in values:
        if "=" not in item:
            raise ValueError(
                "--checkpoint requires STEP=FILE"
            )

        raw_step, raw_file = item.split("=", 1)
        if not re.fullmatch(r"[1-9][0-9]*", raw_step):
            raise ValueError(f"Invalid step: {raw_step}")
        if not raw_file:
            raise ValueError("Checkpoint report path is empty")

        step = int(raw_step)
        if step in seen:
            raise ValueError(f"Duplicate checkpoint step: {step}")
        seen.add(step)
        pairs.append((step, Path(raw_file)))

    return sorted(pairs, key=lambda item: item[0])


def build_table(
    checkpoints: list[tuple[int, Mapping[str, Any]]],
    baseline: Mapping[str, Any],
) -> dict[str, Any]:
    """Create a deterministic benchmark table document."""
    rows = []
    seen = set()

    for step, report in sorted(
        checkpoints, key=lambda item: item[0]
    ):
        if (
            isinstance(step, bool)
            or not isinstance(step, int)
            or step <= 0
        ):
            raise ValueError("Checkpoint step must be positive")
        if step in seen:
            raise ValueError(f"Duplicate checkpoint step: {step}")
        seen.add(step)

        rows.append(
            extract_row(
                report,
                checkpoint_id=f"step_{step:08d}",
                label=f"PlaygroundLM Step {step:,}",
                step=step,
            )
        )

    rows.append(
        extract_row(
            baseline,
            checkpoint_id="gpt2_124m",
            label="OpenAI GPT-2 124M",
        )
    )

    return {
        "columns": list(COLUMNS),
        "rows": rows,
    }


def markdown_table(table: Mapping[str, Any]) -> str:
    """Render a stable, README-friendly comparison table."""
    headings = (
        "| Model / checkpoint | Val loss ↓ | "
        "Perplexity ↓ | HellaSwag ↑ | LAMBADA ↑ |"
    )
    separator = "|---|---:|---:|---:|---:|"
    lines = [
        "## PlaygroundLM vs. OpenAI GPT-2 124M",
        "",
        headings,
        separator,
    ]

    for row in table["rows"]:
        label = str(row["label"]).replace("|", r"\|")
        lines.append(
            f"| {label} "
            f"| {row['val_loss']:.3f} "
            f"| {row['perplexity']:.2f} "
            f"| {100 * row['hellaswag_acc_norm']:.2f}% "
            f"| {100 * row['lambada_acc']:.2f}% |"
        )

    lines.extend([
        "",
        "Lower validation loss/perplexity is better; "
        "higher benchmark accuracy is better.",
        "",
        "HellaSwag is normalized completion accuracy on "
        "10,042 validation examples. LAMBADA is greedy "
        "full-final-word accuracy on 5,153 test examples.",
        "",
        "All rows require complete evaluations. "
        "Validation loss is comparable only when the same "
        "held-out token split and evaluation settings are used.",
        "",
    ])
    return "\n".join(lines)


def write_atomic(path: Path, content: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temp_path = None

    try:
        with tempfile.NamedTemporaryFile(
            mode="w",
            encoding="utf-8",
            dir=path.parent,
            prefix=f".{path.name}.",
            suffix=".tmp",
            delete=False,
        ) as stream:
            temp_path = Path(stream.name)
            stream.write(content)
            stream.flush()
            os.fsync(stream.fileno())

        os.replace(temp_path, path)
    finally:
        if temp_path is not None:
            temp_path.unlink(missing_ok=True)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--checkpoint",
        action="append",
        default=[],
        metavar="STEP=FILE",
    )
    parser.add_argument(
        "--baseline",
        type=Path,
        required=True,
    )
    parser.add_argument(
        "--json-output",
        type=Path,
        default=Path("demo/benchmarks.json"),
    )
    parser.add_argument(
        "--markdown-output",
        type=Path,
        default=Path("demo/BENCHMARKS.md"),
    )
    args = parser.parse_args()

    if (
        args.json_output.resolve()
        == args.markdown_output.resolve()
    ):
        parser.error("Output paths must be different")

    try:
        pairs = parse_checkpoint_pairs(args.checkpoint)
        checkpoint_reports = [
            (step, read_json(path))
            for step, path in pairs
        ]
        baseline = read_json(args.baseline)

        table = build_table(checkpoint_reports, baseline)
        markdown = markdown_table(table)

        serialized = json.dumps(
            table,
            indent=2,
            ensure_ascii=False,
            allow_nan=False,
        ) + "\n"

        write_atomic(args.json_output, serialized)
        write_atomic(args.markdown_output, markdown)

    except (ValueError, KeyError, OSError) as exc:
        parser.error(str(exc))

    print(json.dumps({
        "json": str(args.json_output),
        "markdown": str(args.markdown_output),
        "checkpoint_rows": len(table["rows"]) - 1,
        "baseline_rows": 1,
    }, indent=2))


if __name__ == "__main__":
    main()
