# src/playground_lm/train/test_stage_config.py
# Tests for stage preset resolution and env overrides.

import pytest

from playground_lm.train.stage_config import (
    DEFAULT_MAX_STEPS,
    DEFAULT_WARMUP_STEPS,
    STAGE_PRESETS,
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
    assert plan.max_steps == 10172
    assert plan.warmup_steps == 1000
    assert plan.milestones == (2000, 5000, 8000, 10172)
    assert plan.stage == "C"


def test_stage_case_insensitive():
    assert resolve_stage_plan("a").max_steps == 204
    assert resolve_stage_plan("b").max_steps == 2035
    assert resolve_stage_plan("C").max_steps == 10172


def test_invalid_stage_raises():
    with pytest.raises(ValueError, match="unknown stage"):
        resolve_stage_plan("D")
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


def test_every_stage_warmup_below_max_steps():
    """Sanity: no preset ships with warmup >= max_steps (would never reach peak LR)."""
    for name, preset in STAGE_PRESETS.items():
        assert preset["warmup_steps"] < preset["max_steps"], (
            f"stage {name}: warmup {preset['warmup_steps']} >= "
            f"max_steps {preset['max_steps']}"
        )


def test_every_stage_warmup_reaches_peak():
    """Sanity: warmup should complete before end of run for every stage.

    This is what the whole fix is about — before, Stage A (204 steps) had
    warmup_steps=2000 and never left warmup. Now warmup is well inside."""
    for name, preset in STAGE_PRESETS.items():
        assert preset["warmup_steps"] <= preset["max_steps"] * 0.5, (
            f"stage {name}: warmup {preset['warmup_steps']} is more than "
            f"50% of max_steps {preset['max_steps']}"
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