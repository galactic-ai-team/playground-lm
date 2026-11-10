"""Stage F (facts + simple math): sweep, three-way mix, receipt adapter, G10 (Claude;
G10 bar reconciled to Stephon's p<0.05 AND net >= +5).

Offline: synthetic data and receipts only; no weights, no private items.
"""

import json

import numpy as np
import pytest

from evals import stage_e_review_gates as g
from evals.regression_gate import (
    STAGE_C_SEED_SHA256,
    combine_checkpoint_receipts,
    min_detectable_drop,
    validate_receipt,
    verdict,
)
from playground_lm.train.replay_mix import MultiSourceMixer, build_stage_f_mixed_loader, step_weights
from playground_lm.train.sweep_config import SEEDS, build_sweep_stage_f, expected_lr, run_env, verify_seed

FACT_TOKENS = 300_000  # example; Nova's real count goes in at launch
E_SHA = "ce1fa2098236656d149437638601dfafa7744ab3f69ff52739be0c95199f818b"


# ---- sweep ------------------------------------------------------------------------

def test_stage_f_grid_seeds_shares_and_ids():
    runs = build_sweep_stage_f(FACT_TOKENS, include_control=True)
    assert [(r.seed, r.replay_ratio, round(r.chat_ratio, 4)) for r in runs] == [
        ("E31140", 0.5, 0.1667), ("E31140", 0.25, 0.25), ("C31101", 0.5, 0.1667)]
    assert [r.run_id for r in runs] == ["stage-f-e31140-rr50-ch17-ep1-lr1e-5",
                                        "stage-f-e31140-rr25-ch25-ep1-lr1e-5",
                                        "stage-f-c31101-rr50-ch17-ep1-lr1e-5"]
    assert len(build_sweep_stage_f(FACT_TOKENS)) == 2  # control only on request
    assert [r.schedule_start for r in runs] == [31140, 31140, 31101]  # restart at the seed step
    # One epoch over the facts: 8 fact rows/update for rr50, 12 for rr25 (of 24).
    assert [r.updates for r in runs] == [37, 25, 37]  # ceil(300000/8192), ceil(300000/12288)


def test_stage_f_run_env():
    env = run_env(build_sweep_stage_f(FACT_TOKENS)[0])
    assert env["PLM_SEED_STEP"] == "31140" and env["PLM_SEED_SHA256"] == E_SHA
    assert env["PLM_REPLAY_RATIO"] == "0.5" and float(env["PLM_CHAT_RATIO"]) == pytest.approx(1 / 6)
    assert env["PLM_LR"] == "1e-05" and env["PLM_MAX_STEPS"] == str(31140 + 37)
    assert env["PLM_SCHEDULE_START"] == "31140" and env["PLM_RESET_OPTIMIZER"] == "1"
    assert env["PLM_RUN_ID"] == "stage-f-e31140-rr50-ch17-ep1-lr1e-5"


def test_stage_f_expected_lr_from_the_seed_step():
    run = build_sweep_stage_f(FACT_TOKENS)[0]
    assert run.warmup == 9  # a quarter of 37 updates
    assert expected_lr(run, 31141) == pytest.approx(1e-5 / 9)
    assert expected_lr(run, 31140 + 9) == pytest.approx(1e-5)
    assert expected_lr(run, run.max_steps) == pytest.approx(1e-6)
    with pytest.raises(ValueError):
        expected_lr(run, 31140)


def test_seeds_and_verify(tmp_path):
    assert SEEDS["E31140"] == (31140, E_SHA) and SEEDS["C31101"][1] == STAGE_C_SEED_SHA256
    f = tmp_path / "w.safetensors"
    f.write_bytes(b"x")
    with pytest.raises(ValueError, match="E31140 seed"):
        verify_seed(f, "E31140")
    with pytest.raises(ValueError, match="Stage C 31101 seed"):
        verify_seed(f)


# ---- three-way mix ------------------------------------------------------------------

def rows(tag):
    while True:
        yield np.full((1, 4), tag), np.full((1, 4), tag)


@pytest.mark.parametrize("replay, chat, want", [(0.5, 1 / 6, [4, 2, 6]), (0.25, 0.25, [6, 3, 3])])
def test_each_step_gets_exact_rows_per_source(replay, chat, want):
    mixer = MultiSourceMixer([rows(0), rows(1), rows(2)], [1 - replay - chat, chat, replay],
                             batch_size=6, rows_per_step=12)
    for _ in range(10):
        a, b = next(mixer), next(mixer)
        tags = np.concatenate([a[0][:, 0], b[0][:, 0]])
        assert [int((tags == k).sum()) for k in range(3)] == want
    assert mixer.served == [w * 10 for w in want]


def test_step_weights_validation():
    assert step_weights([1 / 3, 1 / 6, 1 / 2], 12) == [4, 2, 6]
    with pytest.raises(ValueError):
        step_weights([0.5, 0.3, 0.2], 12)  # 3.6 rows
    with pytest.raises(ValueError):
        step_weights([0.5, 0.5, 0.5], 12)  # doesn't sum to 1
    with pytest.raises(ValueError):
        MultiSourceMixer([rows(0)], [1.0], batch_size=5, rows_per_step=12)


def _sft(tmp_path, name, base, n=30, eot=50256):
    toks, masks = [], []
    for i in range(n):
        p, c = 3 + i % 3, 2 + i % 4
        toks += [base + j for j in range(p)] + [base + 50 + j for j in range(c)] + [eot]
        masks += [0] * p + [1] * (c + 1)
    np.array(toks, dtype=np.uint16).tofile(tmp_path / f"{name}.bin")
    np.array(masks, dtype=np.uint8).tofile(tmp_path / f"{name}.masks.bin")
    return tmp_path / f"{name}.bin", tmp_path / f"{name}.masks.bin"


def test_real_loaders_masks_per_source(tmp_path):
    from playground_lm.model import IGNORE_INDEX

    fb, fm = _sft(tmp_path, "facts", 100)
    cb, cm = _sft(tmp_path, "chat", 1000)
    np.arange(5000, 7000, dtype=np.uint16).tofile(tmp_path / "replay.bin")
    loader = build_stage_f_mixed_loader(fb, fm, cb, cm, tmp_path / "replay.bin", replay_ratio=0.5,
                                        chat_ratio=1 / 6, context_len=12, batch_size=6, rows_per_step=12)
    xs, ys = zip(*(next(loader) for _ in range(2)))
    x, y = np.concatenate([np.asarray(v) for v in xs]), np.concatenate([np.asarray(v) for v in ys])
    src = np.where(x[:, 0] >= 5000, 2, np.where(x[:, 0] >= 1000, 1, 0))
    assert [int((src == k).sum()) for k in range(3)] == [4, 2, 6]
    assert np.all(y[src == 2] != IGNORE_INDEX)                       # replay: full loss
    assert np.all((y[src < 2] == IGNORE_INDEX).sum(axis=1) > 0)       # SFT: prompts masked


# ---- regression receipts ------------------------------------------------------------------

def ckpt_receipt(sha, val_loss, hs, lb, **settings):
    s = {"context_len": 1024, "batch_size": 4, "benchmark_limit": 200, "max_val_batches": 20, "seed": 20261108}
    s.update(settings)
    return {"receipt_type": "regression_receipt", "checkpoint": {"path": f"/x/{sha[:6]}.safetensors", "sha256": sha},
            "datasets": {"validation": {"sha256": "1fe7" + "0" * 60}}, "settings": s,
            "metrics": {"val_loss": val_loss, "hellaswag_acc_norm": hs, "lambada_acc": lb},
            "coverage": {"validation": {"tokens_scored": 81920}, "hellaswag": {"examples": 200},
                         "lambada": {"examples": 200}}}


def test_combine_per_checkpoint_receipts_and_judge():
    c = ckpt_receipt(STAGE_C_SEED_SHA256, 3.052, 0.390, 0.265)
    f_ok = ckpt_receipt("a" * 64, 3.070, 0.390, 0.260)
    f_bad = ckpt_receipt("b" * 64, 3.075, 0.390, 0.260)  # +0.023 vs C
    combined = combine_checkpoint_receipts(c, {"F1": f_ok, "F2": f_bad})
    assert validate_receipt(combined) == []
    assert verdict(combined, "F1")["status"] == "PASS"
    assert verdict(combined, "F2")["status"] == "FAIL"  # FineWeb rise > 0.02 vs C
    assert combined["eval"]["hellaswag_limit"] == 200 and combined["eval"]["val_max_batches"] == 20


def test_combine_refuses_different_subsets():
    c = ckpt_receipt(STAGE_C_SEED_SHA256, 3.052, 0.39, 0.265)
    with pytest.raises(ValueError, match="benchmark_limit"):
        combine_checkpoint_receipts(c, {"F1": ckpt_receipt("a" * 64, 3.06, 0.39, 0.26, benchmark_limit=2000)})
    other_val = ckpt_receipt("a" * 64, 3.06, 0.39, 0.26)
    other_val["datasets"]["validation"]["sha256"] = "f" * 64
    with pytest.raises(ValueError, match="val.bin"):
        combine_checkpoint_receipts(c, {"F1": other_val})


def test_round3_subset_cannot_fail_small_accuracy_drops():
    # Why the doc recommends a larger same-set rerun for finalists: at 200 examples
    # the noise rule only FAILs drops of about 9-10 points.
    assert min_detectable_drop(0.39, 200) == pytest.approx(0.0956, abs=1e-3)
    assert min_detectable_drop(0.39, 2000) == pytest.approx(0.0302, abs=1e-3)


# ---- G10 ----------------------------------------------------------------------------------

def private(cand, inc, suite="s" * 64):
    return {"suite_sha256_frozen_before_scoring": suite,
            "results": {"F1": {"rows": [{"id": f"q{i}", "pass": bool(v)} for i, v in enumerate(cand)]},
                        "E31140": {"rows": [{"id": f"q{i}", "pass": bool(v)} for i, v in enumerate(inc)]}}}


def test_g10_needs_a_clear_win_over_e31140():
    base = [1] + [0] * 59                                   # incumbent 1/60
    one_more = g.gate_private_win(private([1, 1] + [0] * 58, base), "F1")
    assert one_more.status == g.BLOCK and one_more.detail["mcnemar_exact_p"] == 1.0  # a 1/60 bump is not enough
    assert one_more.detail["net"] == 1
    six_clean = g.gate_private_win(private([1] * 7 + [0] * 53, base), "F1")
    assert six_clean.status == g.PASS and six_clean.detail["gained"] == 6
    assert six_clean.detail["net"] == 6
    assert six_clean.detail["min_clean_gains_for_p"] == 6
    # Official bar is p < 0.05 AND net >= +5. Five clean gains is net +5 but p = 0.0625.
    five_clean = g.gate_private_win(private([1] * 6 + [0] * 54, base), "F1")
    assert five_clean.status == g.BLOCK and five_clean.detail["net"] == 5
    assert five_clean.detail["mcnemar_exact_p"] == 0.0625
    mixed = g.gate_private_win(private([1] * 9 + [0] * 51, [1] + [0] * 8 + [1] * 3 + [0] * 48), "F1")
    assert mixed.status == g.BLOCK  # 8 gained, 3 lost: net +5 but p = 0.227
    assert mixed.detail["net"] == 5


def test_g10_blocks_without_a_fresh_frozen_suite():
    assert g.gate_private_win(None, "F1").status == g.BLOCK
    unfrozen = private([1] * 10, [0] * 10)
    unfrozen["suite_sha256_frozen_before_scoring"] = None
    assert g.gate_private_win(unfrozen, "F1").status == g.BLOCK
    reused = g.gate_private_win(private([1] * 10, [0] * 10, suite="r3"), "F1", used_suites=["r3"])
    assert reused.status == g.BLOCK and "FRESH" in reused.detail["missing"][0]
    missing_inc = private([1] * 10, [0] * 10)
    del missing_inc["results"]["E31140"]
    assert g.gate_private_win(missing_inc, "F1").status == g.BLOCK


def test_cli_promotion_needs_both_g9_and_g10(tmp_path):
    c = ckpt_receipt(STAGE_C_SEED_SHA256, 3.052, 0.39, 0.265)
    f = ckpt_receipt("a" * 64, 3.060, 0.39, 0.265)
    for name, r in (("c", c), ("f", f)):
        (tmp_path / f"{name}.json").write_text(json.dumps(r))
    (tmp_path / "blind.json").write_text(json.dumps({"suite_sha256_frozen_before_scoring": "s" * 64}))
    (tmp_path / "private.json").write_text(json.dumps(private([1] * 7 + [0] * 53, [1] + [0] * 59)))
    args = ["--candidate", "F1", "--regression-checkpoint-receipt", f"C={tmp_path / 'c.json'}",
            "--regression-checkpoint-receipt", f"F1={tmp_path / 'f.json'}",
            "--blind-receipt", str(tmp_path / "blind.json"), "--out", str(tmp_path / "r.json")]
    g.main(args + ["--approved-by", "Stephon"])
    assert json.loads((tmp_path / "r.json").read_text())["promotion"] == g.BLOCK  # no private-suite receipt
    g.main(args + ["--approved-by", "Stephon", "--private-receipt", str(tmp_path / "private.json")])
    report = json.loads((tmp_path / "r.json").read_text())
    assert {x["gate"]: x["status"] for x in report["gates"]}["G9 promotion"] == g.PASS
    assert report["promotion"] == g.PASS
    g.main(args + ["--private-receipt", str(tmp_path / "private.json")])  # no approval
    assert json.loads((tmp_path / "r.json").read_text())["promotion"] == g.BLOCK
