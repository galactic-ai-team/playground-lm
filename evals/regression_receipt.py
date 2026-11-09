"""Regression receipt: FineWeb-Edu val loss, HellaSwag, LAMBADA for C31101 vs candidates (Claude).

Scores the Stage C 31101 seed and one or more Stage E candidates on the same
data with the existing harness (run_eval.validation_metrics and
benchmarks.score_hellaswag / score_lambada), then writes a receipt with
metrics and counts only (no text from any dataset) plus a per-candidate
verdict from evals/regression_gate.py. evals/stage_e_review_gates.py (G9)
requires this receipt.

Subsets: --val-max-batches, --hellaswag-limit and --lambada-limit take the
first N batches/examples in the datasets' fixed order, identically for every
model; the limits are recorded and a full run leaves them unset. On a free
T4, the documented subset is 200 val batches, 2,000 HellaSwag and 2,000
LAMBADA examples per model; the full sets are better when time allows.

Usage:
  python evals/regression_receipt.py --baseline baseline_31101/step_00031101.safetensors \\
      --candidate r2-ep1-lr3e-5=<weights> --candidate r2-ep3-lr1e-5=<weights> \\
      --fineweb-val /kaggle/input/plm-stage-c-data/val.bin \\
      --val-max-batches 200 --hellaswag-limit 2000 --lambada-limit 2000 --device cuda --out regression.json
"""

from __future__ import annotations

import argparse
import hashlib
import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Callable

from evals.regression_gate import DEFAULT_TOLERANCES, validate_receipt, verdict


def sha256_file(path: str | Path) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for block in iter(lambda: f.read(1 << 20), b""):
            h.update(block)
    return h.hexdigest()


@dataclass
class Scorers:
    """The three scorers; real ones by default, fakes in tests."""
    fineweb: Callable[[Any], dict]
    hellaswag: Callable[[Any], dict]
    lambada: Callable[[Any], dict]


def real_scorers(fineweb_val: Path, *, context_len: int, batch_size: int, device: str,
                 val_max_batches: int | None, hellaswag_limit: int | None,
                 lambada_limit: int | None, offline: bool) -> Scorers:
    from evals.benchmarks import load_rows, score_hellaswag, score_lambada
    from evals.run_eval import validation_metrics
    from playground_lm.tokenizer import get_tokenizer

    tok = get_tokenizer()
    hs_rows = list(load_rows("hellaswag", offline=offline))
    lb_rows = list(load_rows("lambada", offline=offline))
    return Scorers(
        fineweb=lambda m: validation_metrics(m, fineweb_val, context_len=context_len, batch_size=batch_size,
                                             device=device, max_batches=val_max_batches),
        hellaswag=lambda m: score_hellaswag(m, hs_rows, tok, context_len=context_len, device=device,
                                            limit=hellaswag_limit),
        lambada=lambda m: score_lambada(m, lb_rows, tok, context_len=context_len, device=device,
                                        limit=lambada_limit),
    )


KEEP = {
    "fineweb": ("loss", "perplexity", "tokens_scored", "available_targets", "complete"),
    "hellaswag": ("accuracy", "correct", "examples", "complete"),
    "lambada": ("last_word_accuracy", "correct", "examples", "target_token_nll", "complete"),
}


def build_receipt(models: dict[str, tuple[Path, Callable[[], Any]]], scorers: Scorers, eval_info: dict,
                  baseline: str = "C", tolerances: dict | None = None) -> dict:
    """models: name -> (weights path, loader returning the model). Loads one model at a time."""
    out_models = {}
    for name, (weights, load) in models.items():
        model = load()
        out_models[name] = {
            "weights": Path(weights).name,
            "weights_sha256": sha256_file(weights),
            "fineweb": {k: v for k, v in scorers.fineweb(model).items() if k in KEEP["fineweb"]},
            "hellaswag": {k: v for k, v in scorers.hellaswag(model).items() if k in KEEP["hellaswag"]},
            "lambada": {k: v for k, v in scorers.lambada(model).items() if k in KEEP["lambada"]},
        }
        del model
    receipt = {
        "schema_version": 1,
        "kind": "stage_e_regression",
        "baseline": baseline,
        "eval": eval_info,
        "tolerances": {**DEFAULT_TOLERANCES, **(tolerances or {})},
        "models": out_models,
        "claim_scope": "general-ability regression check only; not a promotion decision",
    }
    problems = validate_receipt(receipt)
    receipt["problems"] = problems
    receipt["verdicts"] = {} if problems else {
        name: verdict(receipt, name) for name in out_models if name != baseline}
    return receipt


def main(argv=None) -> None:
    p = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    p.add_argument("--baseline", required=True, help="Stage C 31101 weights")
    p.add_argument("--candidate", action="append", required=True, help="NAME=weights path")
    p.add_argument("--fineweb-val", required=True, help="Stage C FineWeb-Edu val.bin")
    p.add_argument("--factory", default="playground_lm.model:PlaygroundLM")
    p.add_argument("--context-len", type=int, default=1024)
    p.add_argument("--batch-size", type=int, default=8)
    p.add_argument("--val-max-batches", type=int)
    p.add_argument("--hellaswag-limit", type=int)
    p.add_argument("--lambada-limit", type=int)
    p.add_argument("--device", default="cpu")
    p.add_argument("--offline", action="store_true")
    p.add_argument("--out", required=True)
    args = p.parse_args(argv)

    from evals.run_eval import load_model

    def loader(path):
        return lambda: load_model("playground", str(path), device=args.device, factory=args.factory)

    models = {"C": (Path(args.baseline), loader(args.baseline))}
    for spec in args.candidate:
        name, _, path = spec.partition("=")
        if not name or not path or name == "C":
            p.error(f"--candidate must be NAME=path with NAME != C, got {spec!r}")
        models[name] = (Path(path), loader(path))
    scorers = real_scorers(Path(args.fineweb_val), context_len=args.context_len, batch_size=args.batch_size,
                           device=args.device, val_max_batches=args.val_max_batches,
                           hellaswag_limit=args.hellaswag_limit, lambada_limit=args.lambada_limit,
                           offline=args.offline)
    eval_info = {"fineweb_val_sha256": sha256_file(args.fineweb_val), "val_max_batches": args.val_max_batches,
                 "hellaswag_limit": args.hellaswag_limit, "lambada_limit": args.lambada_limit,
                 "context_len": args.context_len}
    receipt = build_receipt(models, scorers, eval_info)
    Path(args.out).write_text(json.dumps(receipt, indent=2) + "\n")
    print(json.dumps({"problems": receipt["problems"],
                      "verdicts": {k: v["status"] for k, v in receipt["verdicts"].items()}}, indent=2))


if __name__ == "__main__":
    main()
