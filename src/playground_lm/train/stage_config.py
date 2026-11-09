# src/playground_lm/train/stage_config.py
# Stage presets and env-var overrides for PlaygroundLM training.
#
# Extracted from the notebook so the config-resolution logic (stage → max_steps
# / warmup_steps / milestones, env overrides, sanity warnings) is unit-testable
# without executing a notebook.
#
# The notebook imports `resolve_stage_plan()` and uses the returned values as
# if it had computed them inline.

import math
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
    # Optional per-stage training settings (Claude, for Stage D). None means
    # "use the notebook's default" (6e-4 peak LR, schedule from step 0,
    # 96-sequence global batch, eval every 500 updates).
    lr: float | None = None
    schedule_start: int | None = None
    effective_batch: int | None = None
    eval_interval: int | None = None


# Stage presets. Warmup is tuned per stage so short stages (A) reach peak LR
# instead of spending the entire run in warmup.
STAGE_PRESETS: dict[str, dict] = {
    "A": {"max_steps": 204,   "warmup_steps": 50,   "milestones": (50, 100, 150, 204)},
    "B": {"max_steps": 2035,  "warmup_steps": 200,  "milestones": (500, 1000, 1500, 2035)},
    # Stage C continues Stage B (step 2035) for one pass over its 2,857,393,718
    # train tokens: 2,790,423 non-overlapping 1024-token windows / 96 per step
    # = 29,066 steps -> final step 31,101 (Claude's review: 10172 was only ~1.0B
    # tokens). Run with PLM_SCHEDULE_START=2035 so the LR re-warms.
    "C": {"max_steps": 31101, "warmup_steps": 1000,
          "milestones": (2035, 5000, 10000, 15000, 20000, 25000, 31101)},
    # Stage D (Claude): fine-tune on the Playground corpus (Nova's
    # data/build_finetune_corpus.py -> tokenized train.bin / val.bin),
    # continuing from Stage C's final checkpoint at step 31101. Same loop,
    # same notebook; only these settings change:
    #   * peak LR 6e-5 = Stage C's final LR (10% of 6e-4), so there is no jump
    #     when it starts, then a short warmup and cosine down to 6e-6;
    #   * the schedule restarts at 31101 (PLM_SCHEDULE_START is implied);
    #   * a 24-sequence global batch: the corpus is small, so smaller batches
    #     give more, gentler updates (6 per GPU x 2 GPUs x 2 accumulation);
    #   * eval every 25 updates, so a run of a few hundred updates gets a val
    #     curve (the default 500 would never fire).
    # max_steps here is a placeholder for ~400 updates; the launcher should
    # set PLM_MAX_STEPS from the corpus size with finetune_max_steps().
    "D": {"max_steps": 31501, "warmup_steps": 20,
          "milestones": (31201, 31301, 31401, 31501),
          "lr": 6e-5, "schedule_start": 31101, "effective_batch": 24, "eval_interval": 25},
    # Stage E: instruction/SFT continuation from Stage C final (31101) with
    # completion-only loss. Lower LR, smaller batch, frequent eval. max_steps
    # is a placeholder; launcher sets PLM_MAX_STEPS from corpus size.
    "E": {"max_steps": 32001, "warmup_steps": 50,
          "milestones": (31201, 31401, 31601, 31801, 32001),
          "lr": 3e-5, "schedule_start": 31101, "effective_batch": 24, "eval_interval": 25},
}

STAGE_C_FINAL_STEP = 31101

DEFAULT_MAX_STEPS = 200_000
DEFAULT_WARMUP_STEPS = 2000


def resolve_stage_plan(
    stage: str | None = None,
    max_steps_override: int | None = None,
    warmup_override: int | None = None,
    lr_override: float | None = None,
    schedule_start_override: int | None = None,
    effective_batch_override: int | None = None,
    eval_interval_override: int | None = None,
) -> StagePlan:
    """
    Resolve (max_steps, warmup_steps, milestones) from a stage name and env
    overrides.

    Args:
        stage: 'A', 'B', 'C', 'D', or None. Case-insensitive. Invalid values
               raise ValueError with the valid set listed.
        max_steps_override: If not None, overrides the stage's max_steps.
        warmup_override: If not None, overrides the stage's warmup_steps.
        lr_override, schedule_start_override, effective_batch_override,
        eval_interval_override: If not None, override the stage's value
               (PLM_LR, PLM_SCHEDULE_START, PLM_EFFECTIVE_BATCH,
               PLM_EVAL_INTERVAL in the notebook).

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

    preset = STAGE_PRESETS.get(stage_upper or "", {})

    def pick(override, key):
        return override if override is not None else preset.get(key)

    return StagePlan(
        max_steps=max_steps,
        warmup_steps=warmup_steps,
        milestones=milestones,
        stage=stage_upper,
        lr=pick(lr_override, "lr"),
        schedule_start=pick(schedule_start_override, "schedule_start"),
        effective_batch=pick(effective_batch_override, "effective_batch"),
        eval_interval=pick(eval_interval_override, "eval_interval"),
    )


def finetune_max_steps(
    train_tokens: int,
    effective_batch: int = 24,
    context_len: int = 1024,
    epochs: float = 3.0,
    start_step: int = STAGE_C_FINAL_STEP,
) -> int:
    """max_steps for a fine-tune that sees the train set about `epochs` times.

    One update covers effective_batch windows of context_len tokens, so one
    epoch is train_tokens / (effective_batch * context_len) updates. The
    result is absolute (start_step + updates), like every other max_steps.
    Small fine-tune corpora overfit quickly: start around 2-3 epochs and
    watch the Playground val loss (eval every 25 updates in Stage D).
    """
    if train_tokens <= context_len:
        raise ValueError(f"train set too small: {train_tokens} tokens")
    if effective_batch <= 0 or context_len <= 0 or epochs <= 0:
        raise ValueError("effective_batch, context_len and epochs must be positive")
    updates = math.ceil(epochs * train_tokens / (effective_batch * context_len))
    return start_step + max(1, updates)


def warmup_warning(plan: StagePlan) -> str | None:
    """Return a warning string if warmup doesn't fit in the stage's schedule.

    The schedule runs from schedule_start (0 unless the stage continues an
    earlier one) to max_steps, so that's the span warmup must fit in."""
    span = plan.max_steps - (plan.schedule_start or 0)
    if plan.warmup_steps >= span:
        return (
            f"warmup_steps ({plan.warmup_steps}) >= the stage's {span} scheduled steps "
            f"(max_steps {plan.max_steps} - schedule start {plan.schedule_start or 0}); "
            f"LR will never reach peak. Consider PLM_WARMUP or a larger stage."
        )
    return None
