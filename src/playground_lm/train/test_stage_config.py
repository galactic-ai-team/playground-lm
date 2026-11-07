# src/playground_lm/train/test_stage_config.py
# Tests for stage preset resolution and env overrides.

import pytest

from playground_lm.train.stage_config import (
    DEFAULT_MAX_STEPS,
    DEFAULT_WARMUP_STEPS,
    STAGE_PRESETS,
    finetune_max_steps,
    resolve_stage_plan,
    warmup_warning,
)


def test_default_no_stage():
    """No stage → notebook's historical defaults."""
    plan = resolve_stage_plan()
    assert plan.max_steps == DEFAULT_MAX_STEPS
    assert plan.warmup_steps == DEFAULT_WARMUP_STEPS
    assert plan.milestones is None
    assert plan.stage is None


def test_stage_a():
    plan = resolve_stage_plan("A")
    assert plan.max_steps == 204
    assert plan.warmup_steps == 50
    assert plan.milestones == (50, 100, 150, 204)
    assert plan.stage == "A"


def test_stage_b():
    plan = resolve_stage_plan("B")
    assert plan.max_steps == 2035
    assert plan.warmup_steps == 200
    assert plan.milestones == (500, 1000, 1500, 2035)
    assert plan.stage == "B"


def test_stage_c():
    plan = resolve_stage_plan("C")
    assert plan.max_steps == 31101
    assert plan.warmup_steps == 1000
    assert plan.milestones == (2035, 5000, 10000, 15000, 20000, 25000, 31101)
    assert plan.stage == "C"


def test_stage_case_insensitive():
    assert resolve_stage_plan("a").max_steps == 204
    assert resolve_stage_plan("b").max_steps == 2035
    assert resolve_stage_plan("C").max_steps == 31101


def test_invalid_stage_raises():
    with pytest.raises(ValueError, match="unknown stage"):
        resolve_stage_plan("E")
    with pytest.raises(ValueError, match="unknown stage"):
        resolve_stage_plan("xyz")


def test_max_steps_override():
    plan = resolve_stage_plan("A", max_steps_override=10)
    assert plan.max_steps == 10
    assert plan.warmup_steps == 50  # warmup from stage unchanged
    assert plan.milestones == (50, 100, 150, 204)  # milestones unchanged


def test_warmup_override():
    plan = resolve_stage_plan("A", warmup_override=5)
    assert plan.max_steps == 204
    assert plan.warmup_steps == 5
    assert plan.milestones == (50, 100, 150, 204)


def test_both_overrides():
    plan = resolve_stage_plan("B", max_steps_override=100, warmup_override=10)
    assert plan.max_steps == 100
    assert plan.warmup_steps == 10
    assert plan.milestones == (500, 1000, 1500, 2035)


def test_overrides_apply_without_stage():
    plan = resolve_stage_plan(max_steps_override=500, warmup_override=25)
    assert plan.max_steps == 500
    assert plan.warmup_steps == 25
    assert plan.milestones is None
    assert plan.stage is None


def _span(preset):
    """Updates in a stage's own schedule (from its schedule start to max_steps)."""
    return preset["max_steps"] - preset.get("schedule_start", 0)


def test_every_stage_warmup_below_max_steps():
    """Sanity: no preset ships with warmup >= its scheduled steps (would never reach peak LR)."""
    for name, preset in STAGE_PRESETS.items():
        assert preset["warmup_steps"] < _span(preset), (
            f"stage {name}: warmup {preset['warmup_steps']} >= "
            f"scheduled steps {_span(preset)}"
        )


def test_every_stage_warmup_reaches_peak():
    """Sanity: warmup should complete before end of run for every stage.

    This is what the whole fix is about — before, Stage A (204 steps) had
    warmup_steps=2000 and never left warmup. Now warmup is well inside."""
    for name, preset in STAGE_PRESETS.items():
        assert preset["warmup_steps"] <= _span(preset) * 0.5, (
            f"stage {name}: warmup {preset['warmup_steps']} is more than "
            f"50% of its {_span(preset)} scheduled steps"
        )


def test_warmup_warning_none_when_below_max():
    plan = resolve_stage_plan("A")
    assert warmup_warning(plan) is None


def test_warmup_warning_when_at_or_above_max():
    plan = resolve_stage_plan("A", warmup_override=204)
    w = warmup_warning(plan)
    assert w is not None
    assert "never reach peak" in w

    plan = resolve_stage_plan("A", warmup_override=1000)
    w = warmup_warning(plan)
    assert w is not None


def test_warmup_warning_uses_overrides():
    """Warning respects both overrides together."""
    plan = resolve_stage_plan(max_steps_override=100, warmup_override=150)
    w = warmup_warning(plan)
    assert w is not None
    assert "150" in w
    assert "100" in w


# --- Stage D (Claude) ---------------------------------------------------------

def test_stage_d_continues_stage_c_with_fine_tune_settings():
    plan = resolve_stage_plan("D")
    assert plan.stage == "D"
    assert plan.schedule_start == 31101  # Stage C's final step
    assert plan.lr == pytest.approx(6e-5)  # = Stage C's final LR, so no jump
    assert plan.effective_batch == 24
    assert plan.eval_interval == 25
    assert plan.max_steps > plan.schedule_start
    assert all(plan.schedule_start < m <= plan.max_steps for m in plan.milestones)
    assert warmup_warning(plan) is None


def test_earlier_stages_leave_the_new_settings_to_the_notebook():
    for name in ("A", "B", "C"):
        plan = resolve_stage_plan(name)
        assert (plan.lr, plan.schedule_start, plan.effective_batch, plan.eval_interval) == (None,) * 4
    assert resolve_stage_plan().lr is None


def test_stage_d_overrides():
    plan = resolve_stage_plan("D", max_steps_override=31300, lr_override=1e-4,
                              schedule_start_override=31000, effective_batch_override=48,
                              eval_interval_override=10)
    assert (plan.max_steps, plan.lr, plan.schedule_start, plan.effective_batch, plan.eval_interval) == (
        31300, 1e-4, 31000, 48, 10)


def test_warmup_warning_counts_from_the_schedule_start():
    # 30 scheduled steps (31101 -> 31131): warmup 20 fits, warmup 30 doesn't.
    assert warmup_warning(resolve_stage_plan("D", max_steps_override=31131)) is None
    assert warmup_warning(resolve_stage_plan("D", max_steps_override=31131, warmup_override=30)) is not None


def test_finetune_max_steps():
    # 2,457,600 tokens = 2,400 windows of 1024; 24 per update -> 100 updates per epoch.
    assert finetune_max_steps(2_457_600) == 31101 + 300  # 3 epochs by default
    assert finetune_max_steps(2_457_600, epochs=1) == 31201
    assert finetune_max_steps(2_457_600, effective_batch=48, epochs=2, start_step=0) == 100
    assert finetune_max_steps(2_000, epochs=0.01) == 31102  # at least one update
    with pytest.raises(ValueError):
        finetune_max_steps(500)
    with pytest.raises(ValueError):
        finetune_max_steps(10_000, effective_batch=0)
