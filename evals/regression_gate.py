"""General-ability regression gate: Stage C 31101 vs Stage E candidates (Claude).

Torch-free. Reads a receipt written by evals/regression_receipt.py and decides,
per candidate, whether fine-tuning cost general ability:

  FineWeb-Edu val loss  (same tokens for every model, so deterministic)
      FAIL if it rises by more than `fineweb_val_loss_increase`.
  HellaSwag accuracy, LAMBADA last-word accuracy (finite samples)
      PASS if the drop is within tolerance;
      WARN if the drop exceeds tolerance but is within sampling noise
           (|drop| <= 1.96 x the standard error of the difference);
      FAIL if the drop exceeds both.
  Overall verdict = worst metric. Only PASS lets G9 consider promotion.

A receipt is valid only if every model was scored on the same FineWeb-Edu val
file, the same HellaSwag/LAMBADA example counts and limits, and the baseline
is the Stage C 31101 seed by sha256.
"""

from __future__ import annotations

import math

STAGE_C_SEED_SHA256 = "4604d3a1b105cc70601352fbd83d6d61b34435b3cf09151b0562592a7b076275"

DEFAULT_TOLERANCES = {
    "fineweb_val_loss_increase": 0.02,  # nats/token
    "hellaswag_drop": 0.01,             # absolute accuracy
    "lambada_drop": 0.015,              # absolute accuracy
}

ORDER = {"PASS": 0, "WARN": 1, "FAIL": 2}


def _se_diff(p1: float, n1: int, p2: float, n2: int) -> float:
    return math.sqrt(p1 * (1 - p1) / n1 + p2 * (1 - p2) / n2)


def validate_receipt(receipt: dict) -> list[str]:
    """Problems that make the receipt unusable (empty list = valid)."""
    problems = []
    if receipt.get("kind") != "stage_e_regression" or receipt.get("schema_version") != 1:
        problems.append("not a stage_e_regression v1 receipt")
        return problems
    models = receipt.get("models") or {}
    base = receipt.get("baseline")
    if base not in models:
        return problems + [f"baseline {base!r} missing from models"]
    if models[base].get("weights_sha256") != STAGE_C_SEED_SHA256:
        problems.append("baseline is not the Stage C 31101 seed (sha256 mismatch)")
    if len(models) < 2:
        problems.append("no candidate models")
    for key in ("fineweb_val_sha256", "val_max_batches", "hellaswag_limit", "lambada_limit"):
        if key not in (receipt.get("eval") or {}):
            problems.append(f"eval.{key} missing")
    for name, m in models.items():
        for metric, fields in (("fineweb", ("loss", "tokens_scored")),
                               ("hellaswag", ("accuracy", "examples")),
                               ("lambada", ("last_word_accuracy", "examples"))):
            if not all(f in (m.get(metric) or {}) for f in fields):
                problems.append(f"{name}: {metric} metrics incomplete")
    for metric, field in (("fineweb", "tokens_scored"), ("hellaswag", "examples"), ("lambada", "examples")):
        counts = {(m.get(metric) or {}).get(field) for m in models.values()}
        if len(counts) > 1:
            problems.append(f"models scored on different {metric} sets ({field} differs)")
    return problems


def verdict(receipt: dict, candidate: str, tolerances: dict | None = None) -> dict:
    tol = {**DEFAULT_TOLERANCES, **(receipt.get("tolerances") or {}), **(tolerances or {})}
    base = receipt["models"][receipt["baseline"]]
    cand = receipt["models"][candidate]
    out: dict = {"candidate": candidate, "tolerances": tol, "metrics": {}}

    rise = cand["fineweb"]["loss"] - base["fineweb"]["loss"]
    out["metrics"]["fineweb_val_loss"] = {
        "baseline": base["fineweb"]["loss"], "candidate": cand["fineweb"]["loss"], "delta": rise,
        "status": "FAIL" if rise > tol["fineweb_val_loss_increase"] else "PASS"}

    for name, field, key in (("hellaswag", "accuracy", "hellaswag_drop"),
                             ("lambada", "last_word_accuracy", "lambada_drop")):
        pb, pc = base[name][field], cand[name][field]
        nb, nc = base[name]["examples"], cand[name]["examples"]
        drop = pb - pc
        noise = 1.96 * _se_diff(pb, nb, pc, nc)
        status = "PASS" if drop <= tol[key] else ("WARN" if drop <= noise else "FAIL")
        out["metrics"][name] = {"baseline": pb, "candidate": pc, "delta": -drop,
                                "noise_95": noise, "examples": nc, "status": status}

    out["status"] = max((m["status"] for m in out["metrics"].values()), key=ORDER.get)
    return out
