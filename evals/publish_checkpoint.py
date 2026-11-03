
#!/usr/bin/env python3
"""Evaluate a checkpoint and publish its benchmark row.

Runs the existing HellaSwag, LAMBADA and validation evaluators
with ONE loaded model. Only complete results can enter the demo
manifest. Other manifest sections remain unchanged.

Examples:
  python -m evals.publish_checkpoint \
    --backend playground \
    --model checkpoints/step_10000.pt \
    --factory playground_lm.model:PlaygroundLM \
    --id step_10000 \
    --val-bin data/val.bin \
    --device cuda

  python -m evals.publish_checkpoint \
    --backend hf \
    --model openai-community/gpt2 \
    --id gpt2_124m \
    --val-bin data/val.bin \
    --device cuda
"""

from __future__ import annotations

import argparse
import json
import math
import os
import tempfile
from pathlib import Path
from typing import Any, Mapping

from evals.benchmarks import (
    load_rows,
    score_hellaswag,
    score_lambada,
)
from evals.run_eval import load_model, validation_metrics
from playground_lm.tokenizer import get_tokenizer

COLUMNS = (
    "val_loss",
    "perplexity",
    "hellaswag_acc_norm",
    "lambada_acc",
)
DEFAULT_MANIFEST = Path("demo/manifest.json")


def _finite_number(value: Any, name: str) -> float:
    if isinstance(value, bool) or not isinstance(
        value, (int, float)
    ):
        raise ValueError(f"{name} must be numeric")
    result = float(value)
    if not math.isfinite(result):
        raise ValueError(f"{name} must be finite")
    return result


def make_row(
    checkpoint_id: str,
    label: str,
    validation: Mapping[str, Any],
    hellaswag: Mapping[str, Any],
    lambada: Mapping[str, Any],
) -> dict[str, Any]:
    """Reject incomplete or invalid results before publication."""
    if not checkpoint_id.strip():
        raise ValueError("Checkpoint ID cannot be empty")
    if not label.strip():
        raise ValueError("Checkpoint label cannot be empty")

    for name, report, expected in (
        ("validation", validation, None),
        ("hellaswag", hellaswag, 10042),
        ("lambada", lambada, 5153),
    ):
        if report.get("complete") is not True:
            raise ValueError(
                f"Cannot publish incomplete {name} results"
            )
        if expected is not None:
            if report.get("examples") != expected:
                raise ValueError(
                    f"{name} expected {expected} examples"
                )

    if (
        validation.get("tokens_scored")
        != validation.get("available_targets")
        or not isinstance(
            validation.get("tokens_scored"), int
        )
        or validation["tokens_scored"] <= 0
    ):
        raise ValueError(
            "Validation did not score all targets"
        )

    values = {
        "val_loss": _finite_number(
            validation["loss"], "val_loss"
        ),
        "perplexity": _finite_number(
            validation["perplexity"], "perplexity"
        ),
        "hellaswag_acc_norm": _finite_number(
            hellaswag["accuracy"], "hellaswag_acc_norm"
        ),
        "lambada_acc": _finite_number(
            lambada["last_word_accuracy"], "lambada_acc"
        ),
    }

    if values["val_loss"] < 0 or values["perplexity"] < 1:
        raise ValueError("Invalid loss or perplexity")

    for name in ("hellaswag_acc_norm", "lambada_acc"):
        if not 0 <= values[name] <= 1:
            raise ValueError(f"{name} must be in [0, 1]")

    if not math.isclose(
        math.log(values["perplexity"]),
        values["val_loss"],
        rel_tol=1e-5,
        abs_tol=1e-5,
    ):
        raise ValueError(
            "Perplexity does not agree with validation loss"
        )

    return {
        "id": checkpoint_id,
        "label": label,
        **values,
    }


def merge_manifest(
    manifest: Mapping[str, Any],
    row: Mapping[str, Any],
) -> dict[str, Any]:
    """Upsert by stable checkpoint ID; preserve other fields."""
    updated = dict(manifest)
    benchmarks = dict(
        updated.get("benchmarks", {})
    )

    existing_columns = benchmarks.get("columns")
    if existing_columns is not None:
        if list(existing_columns) != list(COLUMNS):
            raise ValueError(
                "Manifest benchmark columns differ "
                "from evaluator contract"
            )

    rows = benchmarks.get("rows", [])
    if not isinstance(rows, list):
        raise ValueError("benchmarks.rows must be an array")

    if not isinstance(row.get("id"), str):
        raise ValueError("Row must contain a string ID")

    merged_rows = []
    found = False
    seen_ids: set[str] = set()

    for old in rows:
        if not isinstance(old, dict):
            raise ValueError(
                "Existing benchmark rows must be objects"
            )
        old_id = old.get("id")
        if not isinstance(old_id, str):
            raise ValueError(
                "Existing row has no string ID"
            )
        if old_id in seen_ids:
            raise ValueError(
                f"Duplicate existing ID: {old_id}"
            )
        seen_ids.add(old_id)

        if old_id == row["id"]:
            merged_rows.append(dict(row))
            found = True
        else:
            merged_rows.append(old)

    if not found:
        merged_rows.append(dict(row))

    benchmarks["columns"] = list(COLUMNS)
    benchmarks["rows"] = merged_rows
    updated["benchmarks"] = benchmarks
    return updated


def write_json_atomic(
    path: Path,
    document: Mapping[str, Any],
) -> None:
    """Write complete JSON via replace, avoiding partial manifests."""
    path.parent.mkdir(
        parents=True, exist_ok=True
    )
    text = json.dumps(
        document,
        indent=2,
        ensure_ascii=False,
        allow_nan=False,
    ) + "\n"

    temp_path: Path | None = None
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
            stream.write(text)
            stream.flush()
            os.fsync(stream.fileno())

        os.replace(temp_path, path)
    finally:
        if temp_path is not None:
            temp_path.unlink(missing_ok=True)


def publish_row(
    manifest_path: str | Path,
    row: Mapping[str, Any],
) -> dict[str, Any]:
    """Validate, merge, and atomically publish a result row."""
    path = Path(manifest_path)
    if not path.exists():
        raise FileNotFoundError(
            f"Demo manifest not found: {path}"
        )

    manifest = json.loads(
        path.read_text(encoding="utf-8")
    )
    updated = merge_manifest(manifest, row)
    write_json_atomic(path, updated)
    return updated


def evaluate(
    *,
    backend: str,
    model_ref: str,
    factory: str | None,
    config: str | None,
    val_bin: str,
    device: str,
    context_len: int,
    batch_size: int,
    offline: bool,
) -> tuple[dict[str, Any], dict[str, Any], dict[str, Any]]:
    """Run all three existing scorers on one model."""
    model = load_model(
        backend,
        model_ref,
        factory=factory,
        config_path=config,
        device=device,
    )
    tokenizer = get_tokenizer()

    validation = validation_metrics(
        model,
        val_bin,
        context_len=context_len,
        batch_size=batch_size,
        device=device,
    )

    hellaswag = score_hellaswag(
        model,
        load_rows("hellaswag", offline=offline),
        tokenizer,
        context_len=context_len,
        device=device,
    )

    lambada = score_lambada(
        model,
        load_rows("lambada", offline=offline),
        tokenizer,
        context_len=context_len,
        device=device,
    )

    return validation, hellaswag, lambada


def main() -> None:
    parser = argparse.ArgumentParser(
        description=__doc__
    )
    parser.add_argument(
        "--backend",
        choices=("playground", "hf"),
        required=True,
    )
    parser.add_argument(
        "--model", required=True
    )
    parser.add_argument(
        "--factory",
        help="Model factory module:callable (PlaygroundLM)",
    )
    parser.add_argument(
        "--config",
        help="Optional JSON architecture config",
    )
    parser.add_argument(
        "--id",
        required=True,
        help="Stable row ID, e.g. step_10000 or gpt2_124m",
    )
    parser.add_argument(
        "--label",
        help="Display label; defaults to --id",
    )
    parser.add_argument(
        "--val-bin", required=True
    )
    parser.add_argument(
        "--device", default="cpu"
    )
    parser.add_argument(
        "--context-len", type=int, default=1024
    )
    parser.add_argument(
        "--batch-size", type=int, default=4
    )
    parser.add_argument(
        "--offline", action="store_true"
    )
    parser.add_argument(
        "--manifest",
        type=Path,
        default=DEFAULT_MANIFEST,
    )
    parser.add_argument(
        "--report",
        type=Path,
        help="Optional detailed metrics JSON output",
    )

    args = parser.parse_args()

    if args.context_len <= 0:
        parser.error(
            "--context-len must be positive"
        )
    if args.batch_size <= 0:
        parser.error(
            "--batch-size must be positive"
        )
    if args.backend == "playground" and not args.factory:
        parser.error(
            "--factory is required for PlaygroundLM"
        )

    validation, hellaswag, lambada = evaluate(
        backend=args.backend,
        model_ref=args.model,
        factory=args.factory,
        config=args.config,
        val_bin=args.val_bin,
        device=args.device,
        context_len=args.context_len,
        batch_size=args.batch_size,
        offline=args.offline,
    )

    row = make_row(
        args.id,
        args.label or args.id,
        validation,
        hellaswag,
        lambada,
    )

    # No incomplete report can get past make_row().
    published = publish_row(
        args.manifest, row
    )

    if args.report is not None:
        write_json_atomic(
            args.report,
            {
                "schema_version": 1,
                "model": args.model,
                "backend": args.backend,
                "checkpoint_id": args.id,
                "context_len": args.context_len,
                "validation": validation,
                "benchmarks": {
                    "hellaswag": hellaswag,
                    "lambada": lambada,
                },
            },
        )

    print(
        json.dumps(
            {
                "published_id": row["id"],
                "row": row,
                "manifest": str(args.manifest),
                "total_rows": len(
                    published["benchmarks"]["rows"]
                ),
            },
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
