"""Stage E round-2 sweep config (Claude): epochs -> max_steps, schedule-local LR, launcher env."""

import hashlib
import math

import pytest

from playground_lm.train.stage_config import finetune_max_steps
from playground_lm.train.sweep_config import (
    LIVE_STAGE_D_STEP,
    SEED_SHA256,
    SEED_STEP,
    build_sweep,
    default_warmup,
    expected_lr,
    make_run,
    run_env,
    verify_seed,
)

ROUND1_TOKENS = 640_047  # round-1 train tokens (packet)


def test_epochs_to_max_steps_match_round1():
    # 640,047 / (24 * 1024) = 26.04 updates per epoch.
    assert finetune_max_steps(ROUND1_TOKENS, 24, 1024, 1.0) == 31101 + 27  # round 1 ran 27 updates
    assert finetune_max_steps(ROUND1_TOKENS, 24, 1024, 3.0) == 31101 + 79
    assert finetune_max_steps(ROUND1_TOKENS, 24, 1024, 2.0) == 31101 + 53


def test_default_grid_is_four_distinct_runs_from_31101():
    runs = build_sweep(ROUND1_TOKENS)
    assert [(r.epochs, r.peak_lr) for r in runs] == [(1, 3e-5), (1, 1e-5), (3, 3e-5), (3, 1e-5)]
    assert [r.updates for r in runs] == [27, 27, 79, 79]
    assert sum(r.updates for r in runs) == 212
    assert len({r.run_id for r in runs}) == 4
    assert all(r.schedule_start == SEED_STEP == 31101 for r in runs)
    assert all(r.max_steps != LIVE_STAGE_D_STEP for r in runs)
    assert runs[0].run_id == "stage-e-r2-ep1-lr3e-5"


def test_warmup_rule_matches_round1_and_always_fits():
    assert default_warmup(27) == 6 and default_warmup(79) == 19 and default_warmup(400) == 20
    assert default_warmup(2) == 1
    with pytest.raises(ValueError):
        make_run(ROUND1_TOKENS, 1, 3e-5, warmup=27)  # warmup must end before the run does
    with pytest.raises(ValueError):
        make_run(ROUND1_TOKENS, 1, 0.0)


def test_expected_lr_reproduces_round1_logged_values():
    run = make_run(ROUND1_TOKENS, 1, 3e-5)
    assert expected_lr(run, 31102) == pytest.approx(5e-6)   # packet: first LR
    assert expected_lr(run, 31107) == pytest.approx(3e-5)   # end of warmup: peak
    assert expected_lr(run, 31128) == pytest.approx(3e-6)   # packet: last LR = 10% of peak
    lrs = [expected_lr(run, s) for s in range(31107, 31129)]
    assert all(a >= b for a, b in zip(lrs, lrs[1:]))       # monotone cosine decay
    with pytest.raises(ValueError):
        expected_lr(run, 31101)  # the seed step itself is not an update


def test_expected_lr_matches_the_training_loop_formula():
    loop = pytest.importorskip("playground_lm.train.loop")
    if not hasattr(loop, "_learning_rate"):
        pytest.skip("needs the real training loop")
    run = make_run(ROUND1_TOKENS, 3, 1e-5)
    for step in range(31102, run.max_steps + 1):
        want = loop._learning_rate(step - 31101, run.peak_lr, run.warmup, run.updates)
        assert expected_lr(run, step) == pytest.approx(want, rel=1e-12)


def test_run_env_for_the_launcher():
    env = run_env(make_run(ROUND1_TOKENS, 3, 1e-5))
    assert env == {
        "PLM_STAGE": "E", "PLM_RUN_ID": "stage-e-r2-ep3-lr1e-5", "PLM_MAX_STEPS": str(31101 + 79),
        "PLM_WARMUP": "19", "PLM_LR": "1e-05", "PLM_SCHEDULE_START": "31101",
        "PLM_EFFECTIVE_BATCH": "24", "PLM_EVAL_INTERVAL": "5", "PLM_RESET_OPTIMIZER": "1",
    }


def test_verify_seed(tmp_path):
    other = tmp_path / "d31104.safetensors"
    other.write_bytes(b"not the seed")
    with pytest.raises(ValueError, match="not the Stage C 31101 seed"):
        verify_seed(other)
    assert len(SEED_SHA256) == 64 and SEED_SHA256 != hashlib.sha256(b"not the seed").hexdigest()
