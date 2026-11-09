"""Stage E round-2 sweep: epochs x peak LR from the Stage C 31101 seed (Claude).

Pure configuration (no torch). Dee's launcher reads `run_env(run)` for each
run; the review gates read `expected_lr(run, step)` to check the logged LR
(G6). Every run:
  * starts from Stage C step 31101 (SEED_SHA256 checked by verify_seed),
    never from the live Stage D 31104 checkpoint;
  * resets AdamW moments (PLM_RESET_OPTIMIZER=1) and restarts the schedule
    at 31101: linear warmup, then cosine to 10% of the peak at max_steps;
  * trains with completion-only masks (the SFT loader; nothing here changes
    masking);
  * gets its own run_id, so outputs never overwrite another run or the
    round-1 candidate.

Default grid: epochs {1, 3} x peak LR {3e-5, 1e-5} = 4 runs. With round 1's
corpus (640,047 train tokens, 24 x 1024 tokens per update) that is
27 + 27 + 79 + 79 = 212 updates in total.
"""

from __future__ import annotations

import hashlib
import math
import os
from dataclasses import asdict, dataclass
from pathlib import Path

from playground_lm.train.stage_config import STAGE_C_FINAL_STEP, finetune_max_steps

SEED_STEP = STAGE_C_FINAL_STEP  # 31101
SEED_SHA256 = "4604d3a1b105cc70601352fbd83d6d61b34435b3cf09151b0562592a7b076275"
LIVE_STAGE_D_STEP = 31104  # never a seed, never overwritten

DEFAULT_EPOCHS = (1.0, 3.0)
DEFAULT_PEAK_LRS = (3e-5, 1e-5)


@dataclass(frozen=True)
class SweepRun:
    run_id: str
    epochs: float
    peak_lr: float
    warmup: int
    train_tokens: int
    max_steps: int
    effective_batch: int = 24
    context_len: int = 1024
    schedule_start: int = SEED_STEP
    eval_interval: int = 5

    @property
    def updates(self) -> int:
        return self.max_steps - self.schedule_start

    def to_dict(self) -> dict:
        return {**asdict(self), "updates": self.updates}


def default_warmup(updates: int) -> int:
    """Short warmup that always fits: a quarter of the run, between 1 and 20
    updates (the rule round 1 ran with: 27 updates -> 6)."""
    return max(1, min(20, updates // 4))


def _lr_tag(lr: float) -> str:
    return f"{lr:.0e}".replace("e-0", "e-")


def make_run(train_tokens: int, epochs: float, peak_lr: float, *, prefix: str = "stage-e-r2",
             effective_batch: int = 24, context_len: int = 1024, warmup: int | None = None,
             eval_interval: int = 5) -> SweepRun:
    if peak_lr <= 0 or not math.isfinite(peak_lr):
        raise ValueError("peak_lr must be positive")
    max_steps = finetune_max_steps(train_tokens, effective_batch, context_len, epochs, SEED_STEP)
    updates = max_steps - SEED_STEP
    warm = default_warmup(updates) if warmup is None else warmup
    if not 1 <= warm < updates:
        raise ValueError(f"warmup {warm} must be in [1, {updates})")
    epochs_tag = f"{epochs:g}".replace(".", "p")
    return SweepRun(
        run_id=f"{prefix}-ep{epochs_tag}-lr{_lr_tag(peak_lr)}",
        epochs=float(epochs), peak_lr=float(peak_lr), warmup=warm,
        train_tokens=int(train_tokens), max_steps=max_steps,
        effective_batch=effective_batch, context_len=context_len, eval_interval=eval_interval,
    )


def build_sweep(train_tokens: int, epochs=DEFAULT_EPOCHS, peak_lrs=DEFAULT_PEAK_LRS, **kw) -> list[SweepRun]:
    runs = [make_run(train_tokens, e, lr, **kw) for e in epochs for lr in peak_lrs]
    ids = [r.run_id for r in runs]
    if len(set(ids)) != len(ids):
        raise ValueError("duplicate run ids in sweep")
    return runs


def expected_lr(run: SweepRun | dict, step: int) -> float:
    """LR the loop sets for update `step` (same formula as train/loop.py,
    on steps counted from the schedule start)."""
    r = run if isinstance(run, dict) else run.to_dict()
    local, total, warmup, peak = step - r["schedule_start"], r["max_steps"] - r["schedule_start"], r["warmup"], r["peak_lr"]
    if not 1 <= local <= total:
        raise ValueError(f"step {step} outside the run ({r['schedule_start'] + 1}..{r['max_steps']})")
    if local <= warmup:
        return peak * local / warmup
    progress = (local - warmup) / (total - warmup)
    return peak * (0.1 + 0.45 * (1.0 + math.cos(math.pi * progress)))


def run_env(run: SweepRun) -> dict[str, str]:
    """Environment for the training notebook (Dee's launcher passes it through)."""
    return {
        "PLM_STAGE": "E",
        "PLM_RUN_ID": run.run_id,
        "PLM_MAX_STEPS": str(run.max_steps),
        "PLM_WARMUP": str(run.warmup),
        "PLM_LR": repr(run.peak_lr),
        "PLM_SCHEDULE_START": str(run.schedule_start),
        "PLM_EFFECTIVE_BATCH": str(run.effective_batch),
        "PLM_EVAL_INTERVAL": str(run.eval_interval),
        "PLM_RESET_OPTIMIZER": "1",
    }


def verify_seed(weights_path: str | os.PathLike) -> str:
    """Raise unless the file is the Stage C 31101 seed (by sha256)."""
    h = hashlib.sha256()
    with open(weights_path, "rb") as f:
        for block in iter(lambda: f.read(1 << 20), b""):
            h.update(block)
    digest = h.hexdigest()
    if digest != SEED_SHA256:
        raise ValueError(f"{Path(weights_path).name} is not the Stage C {SEED_STEP} seed (sha256 {digest[:12]}...)")
    return digest
