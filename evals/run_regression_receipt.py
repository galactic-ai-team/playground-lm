#!/usr/bin/env python3
"""Public Stage E Round-2 general-ability regression receipt.

Reuse the repository's existing FineWeb-Edu validation, HellaSwag and LAMBADA
scorers. No private prompts or answer keys are opened by this script.

Example (baseline; repeat computation for deterministic self-check):
    PYTHONPATH=src:. python -m evals.run_regression_receipt \
      --checkpoint /path/to/step_00031101.safetensors \
      --val-bin /path/to/fineweb_stage_c_val.bin \
      --config /path/to/model-config.json --offline --repeat-check \
      --out receipts/c31101.json

Use IDENTICAL --val-bin, --context-len, --batch-size, --benchmark-limit,
--max-val-batches, and --seed when evaluating candidates. Omit both limits
for official full-split results. Partial runs are NEVER publishable.

The --repeat-check option re-evaluates the same weights/datasets, aborting
without writing if any metric differs beyond --tolerance.
"""
from __future__ import annotations

import argparse
from datetime import datetime, timezone
import hashlib
import json
import math
from pathlib import Path
from typing import Any, Mapping

SCHEMA_VERSION = 1
DATASETS = {
    "validation": "FineWeb-Edu (local held-out val.bin)",
    "hellaswag": "Rowan/hellaswag:validation",
    "lambada": "EleutherAI/lambada_openai:default:test",
}
METRIC_NAMES = ("val_loss", "perplexity", "hellaswag_acc_norm", "lambada_acc")


def sha256_path(path: str | Path) -> str:
    path = Path(path)
    h = hashlib.sha256()
    with path.open("rb") as file:
        for chunk in iter(lambda: file.read(1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()


def _number(v: Any, name: str) -> float:
    if isinstance(v, bool) or not isinstance(v, (int, float)) or not math.isfinite(v):
        raise ValueError(f"{name} must be a finite number")
    return float(v)


def validate_metrics(validation: Mapping[str, Any], hs: Mapping[str, Any], la: Mapping[str, Any], *,
                     full_requested: bool) -> dict[str, Any]:
    """Refuse inconsistent metrics or incomplete purported full evaluations."""
    loss = _number(validation.get("loss"), "val loss")
    ppl = _number(validation.get("perplexity"), "perplexity")
    hs_acc = _number(hs.get("accuracy"), "HellaSwag accuracy")
    la_acc = _number(la.get("last_word_accuracy"), "LAMBADA accuracy")
    if loss < 0 or ppl < 1 or not math.isclose(math.log(ppl), loss, rel_tol=1e-5, abs_tol=1e-5):
        raise ValueError("Validation loss/perplexity inconsistent")
    if not (0 <= hs_acc <= 1 and 0 <= la_acc <= 1):
        raise ValueError("Benchmark accuracy outside [0,1]")
    for label, part in (("validation", validation), ("hellaswag", hs), ("lambada", la)):
        if not isinstance(part, Mapping):
            raise ValueError(f"{label} result must be an object")
    scored = validation.get("tokens_scored")
    targets = validation.get("available_targets")
    if (isinstance(scored, bool) or not isinstance(scored, int) or scored <= 0
            or isinstance(targets, bool) or not isinstance(targets, int) or targets < scored):
        raise ValueError("Invalid validation token coverage")
    for label, part in (("hellaswag", hs), ("lambada", la)):
        if (isinstance(part.get("examples"), bool) or not isinstance(part.get("examples"), int)
                or part["examples"] < 1):
            raise ValueError(f"Invalid {label} example count")
    complete = (validation.get("complete") is True and scored == targets
                and hs.get("complete") is True and hs.get("examples") == 10042
                and la.get("complete") is True and la.get("examples") == 5153)
    if full_requested and not complete:
        raise ValueError("Full evaluation requested but scorer reports incomplete coverage")
    return {"val_loss": loss, "perplexity": ppl, "hellaswag_acc_norm": hs_acc,
            "lambada_acc": la_acc, "complete": complete}


def metric_delta(a: Mapping[str, Any], b: Mapping[str, Any], *, tolerance: float) -> dict[str, float]:
    """Check absolute numeric repeat stability, fail on count or coverage drift."""
    if not 0 <= tolerance < float("inf"):
        raise ValueError("tolerance must be finite and nonnegative")
    for key in ("validation", "hellaswag", "lambada"):
        x, y = a[key], b[key]
        count_keys = ("tokens_scored", "available_targets") if key == "validation" else ("examples", "correct")
        for count_key in count_keys:
            if x.get(count_key) != y.get(count_key):
                raise ValueError(f"Nondeterministic {key}.{count_key}")
    deltas = {k: abs(_number(a["metrics"][k], k) - _number(b["metrics"][k], k))
              for k in METRIC_NAMES}
    if any(d > tolerance for d in deltas.values()):
        raise ValueError(f"Repeat-check failed, max delta {max(deltas.values()):.9g} > {tolerance}")
    return deltas


def make_receipt(*, checkpoint: Path, val_bin: Path, config: Path | None, context_len: int,
                 batch_size: int, benchmark_limit: int | None, max_val_batches: int | None,
                 seed: int, device: str, offline: bool, validation: Mapping[str, Any],
                 hellaswag: Mapping[str, Any], lambada: Mapping[str, Any],
                 now: str | None = None, repeat_deltas: Mapping[str, float] | None = None) -> dict[str, Any]:
    full_requested = benchmark_limit is None and max_val_batches is None
    metrics = validate_metrics(validation, hellaswag, lambada, full_requested=full_requested)
    return {
        "schema_version": SCHEMA_VERSION,
        "receipt_type": "regression_receipt",
        "suite_type": "public_general_ability_only",
        "timestamp_utc": now or datetime.now(timezone.utc).isoformat(timespec="seconds"),
        "checkpoint": {"path": str(checkpoint), "sha256": sha256_path(checkpoint),
                       "config_sha256": sha256_path(config) if config else None},
        "datasets": {"validation": {"name": DATASETS["validation"], "sha256": sha256_path(val_bin),
                                    "path": str(val_bin)},
                     "hellaswag": DATASETS["hellaswag"], "lambada": DATASETS["lambada"]},
        "settings": {"context_len": context_len, "batch_size": batch_size, "benchmark_limit": benchmark_limit,
                     "max_val_batches": max_val_batches, "seed": seed, "device": device,
                     "offline": offline, "scoring": "existing evals.run_eval + evals.benchmarks"},
        "metrics": {k: metrics[k] for k in METRIC_NAMES},
        "coverage": {"validation": {k: validation.get(k) for k in ("tokens_scored", "available_targets", "complete")},
                     "hellaswag": {k: hellaswag.get(k) for k in ("examples", "correct", "complete")},
                     "lambada": {k: lambada.get(k) for k in ("examples", "correct", "complete")}},
        "complete": metrics["complete"],
        "publishable": full_requested and metrics["complete"],
        "repeat_check": {"performed": repeat_deltas is not None,
                         "absolute_metric_deltas": dict(repeat_deltas) if repeat_deltas is not None else None},
        "limitations": "Public aggregate regression metrics only; subset results are not comparable to full benchmark scores. "
                       "Do not infer private held-out performance or promote a model from this receipt alone."
    }


def evaluate_once(model: Any, tokenizer: Any, val_bin: Path, *, context_len: int, batch_size: int,
                  device: str, offline: bool, benchmark_limit: int | None,
                  max_val_batches: int | None) -> dict[str, Any]:
    from evals.run_eval import validation_metrics
    from evals.benchmarks import load_rows, score_hellaswag, score_lambada
    validation = validation_metrics(model, val_bin, context_len=context_len, batch_size=batch_size,
                                    device=device, max_batches=max_val_batches)
    hs = score_hellaswag(model, load_rows("hellaswag", offline=offline), tokenizer,
                         context_len=context_len, device=device, limit=benchmark_limit)
    la = score_lambada(model, load_rows("lambada", offline=offline), tokenizer,
                       context_len=context_len, device=device, limit=benchmark_limit)
    return {"validation": validation, "hellaswag": hs, "lambada": la,
            "metrics": {"val_loss": validation["loss"], "perplexity": validation["perplexity"],
                        "hellaswag_acc_norm": hs["accuracy"], "lambada_acc": la["last_word_accuracy"]}}


def run(*, checkpoint: Path, val_bin: Path, out: Path, config: Path | None = None,
        context_len: int = 1024, batch_size: int = 4, benchmark_limit: int | None = None,
        max_val_batches: int | None = None, seed: int = 42, device: str = "cpu", offline: bool = False,
        repeat_check: bool = False, tolerance: float = 1e-5) -> dict[str, Any]:
    if context_len <= 0 or batch_size <= 0 or (benchmark_limit is not None and benchmark_limit <= 0) or (
            max_val_batches is not None and max_val_batches <= 0):
        raise ValueError("Invalid evaluation dimensions or limits")
    if not math.isfinite(tolerance) or tolerance < 0:
        raise ValueError("Invalid repeat tolerance")
    if not checkpoint.is_file() or not val_bin.is_file():
        raise FileNotFoundError("Checkpoint and held-out validation bin must exist")
    if config is not None and not config.is_file():
        raise FileNotFoundError(config)
    # Dependencies are only loaded when actually running models (not when inspecting receipts).
    import random
    import numpy as np
    import torch
    from evals.run_eval import load_model
    from playground_lm.tokenizer import get_tokenizer
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    model = load_model("playground", str(checkpoint), factory="playground_lm.model:PlaygroundLM",
                       config_path=str(config) if config else None, device=device)
    tokenizer = get_tokenizer()
    first = evaluate_once(model, tokenizer, val_bin, context_len=context_len, batch_size=batch_size,
                          device=device, offline=offline, benchmark_limit=benchmark_limit,
                          max_val_batches=max_val_batches)
    deltas = None
    if repeat_check:
        second = evaluate_once(model, tokenizer, val_bin, context_len=context_len, batch_size=batch_size,
                               device=device, offline=offline, benchmark_limit=benchmark_limit,
                               max_val_batches=max_val_batches)
        deltas = metric_delta(first, second, tolerance=tolerance)
    receipt = make_receipt(checkpoint=checkpoint, val_bin=val_bin, config=config, context_len=context_len,
                           batch_size=batch_size, benchmark_limit=benchmark_limit, max_val_batches=max_val_batches,
                           seed=seed, device=device, offline=offline, validation=first["validation"],
                           hellaswag=first["hellaswag"], lambada=first["lambada"], repeat_deltas=deltas)
    out.parent.mkdir(parents=True, exist_ok=True)
    from evals.publish_checkpoint import write_json_atomic
    write_json_atomic(out, receipt)
    return receipt


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--checkpoint", type=Path, required=True)
    p.add_argument("--val-bin", type=Path, required=True)
    p.add_argument("--config", type=Path)
    p.add_argument("--out", type=Path, required=True)
    p.add_argument("--context-len", type=int, default=1024)
    p.add_argument("--batch-size", type=int, default=4)
    p.add_argument("--benchmark-limit", type=int, help="Subset size for BOTH HellaSwag and LAMBADA; nonpublishable")
    p.add_argument("--max-val-batches", type=int, help="Validation smoke-test subset; nonpublishable")
    p.add_argument("--device", default="cpu")
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--offline", action="store_true", help="Datasets must already be in HF local cache")
    p.add_argument("--repeat-check", action="store_true")
    p.add_argument("--tolerance", type=float, default=1e-5)
    args = p.parse_args()
    try:
        receipt = run(checkpoint=args.checkpoint, val_bin=args.val_bin, out=args.out, config=args.config,
                      context_len=args.context_len, batch_size=args.batch_size,
                      benchmark_limit=args.benchmark_limit, max_val_batches=args.max_val_batches,
                      seed=args.seed, device=args.device, offline=args.offline,
                      repeat_check=args.repeat_check, tolerance=args.tolerance)
    except (ValueError, OSError) as e:
        p.error(str(e))
    print(json.dumps({"out": str(args.out), "complete": receipt["complete"], "publishable": receipt["publishable"],
                      "checkpoint_sha256": receipt["checkpoint"]["sha256"],
                      "metrics": receipt["metrics"], "repeat_check": receipt["repeat_check"]}, indent=2))


if __name__ == "__main__":
    main()
