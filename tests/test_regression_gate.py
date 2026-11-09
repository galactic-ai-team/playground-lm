"""General-ability regression receipt + gate, and G9 wiring (Claude). Fake scorers; no weights, no datasets."""

import copy
import json

import pytest

from evals import stage_e_review_gates as g
from evals.regression_gate import STAGE_C_SEED_SHA256, validate_receipt, verdict
from evals.regression_receipt import Scorers, build_receipt


def model_metrics(val_loss, hs, lb, n_hs=2000, n_lb=2000, tokens=1_638_400, sha=STAGE_C_SEED_SHA256):
    return {"weights": "w.safetensors", "weights_sha256": sha,
            "fineweb": {"loss": val_loss, "tokens_scored": tokens},
            "hellaswag": {"accuracy": hs, "examples": n_hs},
            "lambada": {"last_word_accuracy": lb, "examples": n_lb}}


def receipt(**cands):
    models = {"C": model_metrics(3.174, 0.300, 0.230)}
    for name, (v, h, l) in cands.items():
        models[name] = model_metrics(v, h, l, sha="f" * 64)
    return {"schema_version": 1, "kind": "stage_e_regression", "baseline": "C",
            "eval": {"fineweb_val_sha256": "a" * 64, "val_max_batches": 200,
                     "hellaswag_limit": 2000, "lambada_limit": 2000},
            "models": models}


# ---- verdicts -----------------------------------------------------------------------

def test_small_changes_pass():
    v = verdict(receipt(E=(3.181, 0.300, 0.245)), "E")  # Stage D-sized drift
    assert v["status"] == "PASS"
    assert v["metrics"]["fineweb_val_loss"]["delta"] == pytest.approx(0.007)


def test_val_loss_rise_beyond_tolerance_fails():
    v = verdict(receipt(E=(3.20, 0.300, 0.230)), "E")
    assert v["metrics"]["fineweb_val_loss"]["status"] == "FAIL" and v["status"] == "FAIL"


def test_accuracy_drop_beyond_tolerance_but_within_noise_warns():
    # 1.5 points on 2,000 examples: over the 1-point tolerance, under ~2.8 points of noise.
    v = verdict(receipt(E=(3.174, 0.285, 0.230)), "E")
    assert v["metrics"]["hellaswag"]["status"] == "WARN" and v["status"] == "WARN"
    assert v["metrics"]["hellaswag"]["noise_95"] == pytest.approx(0.0284, abs=5e-4)


def test_accuracy_drop_beyond_noise_fails():
    v = verdict(receipt(E=(3.174, 0.300, 0.180)), "E")  # LAMBADA -5 points
    assert v["metrics"]["lambada"]["status"] == "FAIL" and v["status"] == "FAIL"


# ---- receipt validation -------------------------------------------------------------

def test_valid_receipt_has_no_problems():
    assert validate_receipt(receipt(E=(3.18, 0.30, 0.24))) == []


@pytest.mark.parametrize("break_it, message", [
    (lambda r: r["models"]["C"].update(weights_sha256="0" * 64), "Stage C 31101 seed"),
    (lambda r: r["models"]["E"]["hellaswag"].update(examples=1999), "different hellaswag"),
    (lambda r: r["models"]["E"]["fineweb"].update(tokens_scored=1), "different fineweb"),
    (lambda r: r["eval"].pop("lambada_limit"), "eval.lambada_limit"),
    (lambda r: r["models"]["E"].pop("lambada"), "lambada metrics incomplete"),
    (lambda r: r.update(kind="other"), "not a stage_e_regression"),
])
def test_invalid_receipts_are_rejected(break_it, message):
    r = receipt(E=(3.18, 0.30, 0.24))
    break_it(r)
    assert any(message in p for p in validate_receipt(r))


# ---- receipt builder (fake scorers) ----------------------------------------------------

def test_build_receipt_keeps_metrics_only_and_judges_each_candidate(tmp_path):
    weights = {}
    for name, content in (("C", b"seed"), ("r2a", b"cand-a"), ("r2b", b"cand-b")):
        weights[name] = tmp_path / f"{name}.safetensors"
        weights[name].write_bytes(content)
    fake = {"C": (3.174, 0.300, 0.230), "r2a": (3.180, 0.301, 0.240), "r2b": (3.230, 0.300, 0.230)}
    scorers = Scorers(
        fineweb=lambda m: {"loss": fake[m][0], "tokens_scored": 1000, "available_targets": 9999,
                           "complete": False, "text": "should not be kept"},
        hellaswag=lambda m: {"accuracy": fake[m][1], "correct": 1, "examples": 500, "complete": False},
        lambada=lambda m: {"last_word_accuracy": fake[m][2], "correct": 1, "examples": 500,
                           "target_token_nll": 2.9, "complete": False},
    )
    models = {name: (path, (lambda n=name: n)) for name, path in weights.items()}
    eval_info = {"fineweb_val_sha256": "a" * 64, "val_max_batches": 10, "hellaswag_limit": 500, "lambada_limit": 500}
    r = build_receipt(models, scorers, eval_info)
    # The fake seed file isn't the real Stage C seed, so the receipt is (correctly) invalid:
    assert any("Stage C 31101 seed" in p for p in r["problems"]) and r["verdicts"] == {}
    assert "text" not in r["models"]["C"]["fineweb"]
    r["models"]["C"]["weights_sha256"] = STAGE_C_SEED_SHA256
    assert validate_receipt(r) == []
    assert verdict(r, "r2a")["status"] == "PASS" and verdict(r, "r2b")["status"] == "FAIL"
    json.dumps(r)  # serialisable


# ---- G9 -----------------------------------------------------------------------------

BLIND = {"suite_sha256_frozen_before_scoring": "b" * 64}


def test_g9_blocks_without_a_regression_receipt_or_approval():
    gate = g.gate_promotion(BLIND, None, "Stephon")
    assert gate.status == g.BLOCK and any("regression receipt" in m for m in gate.detail["missing"])
    gate = g.gate_promotion(BLIND, receipt(E=(3.18, 0.30, 0.24)), None)
    assert gate.status == g.BLOCK and gate.detail["missing"] == ["Stephon's approval"]


def test_g9_blocks_on_failed_or_invalid_regression_and_on_leakage():
    assert g.gate_promotion(BLIND, receipt(E=(3.30, 0.30, 0.24)), "Stephon").status == g.BLOCK
    bad = receipt(E=(3.18, 0.30, 0.24))
    bad["models"]["C"]["weights_sha256"] = "0" * 64
    assert g.gate_promotion(BLIND, bad, "Stephon").status == g.BLOCK
    gate = g.gate_promotion(BLIND, receipt(E=(3.18, 0.30, 0.24)), "Stephon", leakage_status=g.FAIL)
    assert gate.status == g.BLOCK and any("G5 leakage" in m for m in gate.detail["missing"])
    assert g.gate_promotion(BLIND, receipt(E=(3.18, 0.30, 0.24)), "Stephon", candidate="r2x").status == g.BLOCK


def test_g9_passes_only_with_everything_in_place():
    gate = g.gate_promotion(BLIND, receipt(E=(3.18, 0.30, 0.24)), "Stephon", leakage_status=g.PASS)
    assert gate.status == g.PASS and gate.detail["regression"]["status"] == "PASS"


def test_g6_reads_a_sweep_run_config(tmp_path):
    from playground_lm.train.sweep_config import expected_lr, make_run

    run = make_run(640_047, 3, 1e-5)
    (tmp_path / "run.json").write_text(json.dumps(run.to_dict()))
    rows = [json.dumps({"step": s, "lr": expected_lr(run, s)}) for s in range(31102, run.max_steps + 1)]
    (tmp_path / "metrics.jsonl").write_text("\n".join(rows) + "\n")
    out = tmp_path / "review.json"
    code = g.main(["--metrics", str(tmp_path / "metrics.jsonl"), "--sweep-run", str(tmp_path / "run.json"),
                   "--out", str(out)])
    statuses = {x["gate"]: x["status"] for x in json.loads(out.read_text())["gates"]}
    assert code == 0 and statuses["G6 lr-schedule"] == g.PASS and statuses["G9 promotion"] == g.BLOCK
