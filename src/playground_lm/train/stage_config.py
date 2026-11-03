# src/playground_lm/train/stage_config.py
# Stage presets and env-var overrides for PlaygroundLM training.
#
# Extracted from the notebook so the config-resolution logic (stage → max_steps
# / warmup_steps / milestones, env overrides, sanity warnings) is unit-testable
# without executing a notebook.
#
# The notebook imports `resolve_stage_plan()` and uses the returned values as
# if it had computed them inline.

import os
from dataclasses import dataclass


# Tokens per step at the default effective batch size: 12 micro × 8 accum ×
# 1024 context. Step counts below are derived from the token budgets in
# data/PRETRAIN_PLAN.md divided by this value.
TOKENS_PER_STEP = 98_304


@dataclass(frozen=True)
class StagePlan:
    max_steps: int
    warmup_steps: int
    milestones: tuple[int, ...] | None   # None → CheckpointManager defaults
    stage: str | None                    # None → no stage preset


# Stage presets. Warmup is tuned per stage so short stages (A) reach peak LR
# instead of spending the entire run in warmup.
STAGE_PRESETS: dict[str, dict] = {
    "A": {"max_steps": 204,   "warmup_steps": 50,   "milestones": (50, 100, 150, 204)},
    "B": {"max_steps": 2035,  "warmup_steps": 200,  "milestones": (500, 1000, 1500, 2035)},
    "C": {"max_steps": 10172, "warmup_steps": 1000, "milestones": (2000, 5000, 8000, 10172)},
}

DEFAULT_MAX_STEPS = 200_000
DEFAULT_WARMUP_STEPS = 2000


def resolve_stage_plan(
    stage: str | None = None,
    max_steps_override: int | None = None,
    warmup_override: int | None = None,
) -> StagePlan:
    """
    Resolve (max_steps, warmup_steps, milestones) from a stage name and env
    overrides.

    Args:
        stage: 'A', 'B', 'C', or None. Case-insensitive. Invalid values raise
               ValueError with the valid set listed.
        max_steps_override: If not None, overrides the stage's max_steps.
        warmup_override: If not None, overrides the stage's warmup_steps.

    Returns:
        StagePlan with resolved values. milestones is None when no stage
        preset is set (the caller should use CheckpointManager's defaults).

    Raises:
        ValueError if stage is not None and not in STAGE_PRESETS.
    """
    stage_upper = (stage or "").upper() or None
    if stage_upper is not None and stage_upper not in STAGE_PRESETS:
        raise ValueError(
            f"unknown stage {stage!r}; expected one of "
            f"{sorted(STAGE_PRESETS)} or None"
        )

    if stage_upper is not None:
        preset = STAGE_PRESETS[stage_upper]
        max_steps = preset["max_steps"]
        warmup_steps = preset["warmup_steps"]
        milestones = preset["milestones"]
    else:
        max_steps = DEFAULT_MAX_STEPS
        warmup_steps = DEFAULT_WARMUP_STEPS
        milestones = None

    if max_steps_override is not None:
        max_steps = max_steps_override
    if warmup_override is not None:
        warmup_steps = warmup_override

    return StagePlan(
        max_steps=max_steps,
        warmup_steps=warmup_steps,
        milestones=milestones,
        stage=stage_upper,
    )


def warmup_warning(plan: StagePlan) -> str | None:
    """Return a warning string if warmup >= max_steps, else None."""
    if plan.warmup_steps >= plan.max_steps:
        return (
            f"warmup_steps ({plan.warmup_steps}) >= max_steps ({plan.max_steps}); "
            f"LR will never reach peak. Consider PLM_WARMUP or a larger stage."
        )
    return None