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

Round 3 (build_sweep_round3): every round-2 cell raised FineWeb-Edu val loss
by more than the gate's 0.02, so round 3 holds a gentle LR (1e-5, one SFT
epoch) and varies the share of rows replayed from Stage C pretraining data
(replay_mix.py): replay_ratio {0.25, 0.5}, plus an optional 0.5 x 3e-5 cell.
"epochs" always counts passes over the SFT data: with replay, fewer rows per
update are SFT rows, so the same epoch takes more updates.

Stage F (build_sweep_stage_f): facts + simple math. Seeds are named
(SEEDS): E31140 (live v0.5 Chat) for the main cells, C31101 as a control.
Each update's rows are split three ways: new fact/arith SFT, existing
leak-free chat SFT (chat_ratio), and Stage C FineWeb replay (replay_ratio).
"epochs" then counts passes over the fact/arith data; chat and replay rows
are sampled at their shares. The schedule restarts at the seed's step.
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

# Named seeds: (step, weights sha256). E31140 is live v0.5 Chat (Stage E round 3,
# rr50 @ 1e-5); C31101 is the Stage C base every regression gate compares against.
SEEDS = {
    "C31101": (SEED_STEP, SEED_SHA256),
    "E31140": (31140, "ce1fa2098236656d149437638601dfafa7744ab3f69ff52739be0c95199f818b"),
}

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
    replay_ratio: float = 0.0  # share of rows from Stage C pretraining replay (round 3)
    chat_ratio: float = 0.0    # share of rows from existing chat SFT (Stage F)
    seed: str = "C31101"       # key into SEEDS

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


def _whole_rows(share: float, effective_batch: int, what: str) -> int:
    rows = effective_batch * share
    if abs(rows - round(rows)) > 1e-9:
        raise ValueError(f"{what} {share} x batch {effective_batch} is not a whole number of rows")
    return int(round(rows))


def make_run(train_tokens: int, epochs: float, peak_lr: float, *, prefix: str = "stage-e-r2",
             effective_batch: int = 24, context_len: int = 1024, warmup: int | None = None,
             eval_interval: int = 5, replay_ratio: float = 0.0, chat_ratio: float = 0.0,
             seed: str = "C31101") -> SweepRun:
    if peak_lr <= 0 or not math.isfinite(peak_lr):
        raise ValueError("peak_lr must be positive")
    if seed not in SEEDS:
        raise ValueError(f"unknown seed {seed!r}; expected one of {sorted(SEEDS)}")
    if replay_ratio < 0 or chat_ratio < 0 or replay_ratio + chat_ratio >= 1.0:
        raise ValueError("replay_ratio and chat_ratio must be >= 0 and leave room for the trained SFT rows")
    replay_rows = _whole_rows(replay_ratio, effective_batch, "replay_ratio")
    chat_rows = _whole_rows(chat_ratio, effective_batch, "chat_ratio")
    sft_rows = effective_batch - replay_rows - chat_rows  # these rows set the epoch length
    seed_step = SEEDS[seed][0]
    max_steps = finetune_max_steps(train_tokens, sft_rows, context_len, epochs, seed_step)
    updates = max_steps - seed_step
    warm = default_warmup(updates) if warmup is None else warmup
    if not 1 <= warm < updates:
        raise ValueError(f"warmup {warm} must be in [1, {updates})")
    epochs_tag = f"{epochs:g}".replace(".", "p")
    replay_tag = f"-rr{round(replay_ratio * 100):02d}" if replay_ratio else ""
    chat_tag = f"-ch{round(chat_ratio * 100):02d}" if chat_ratio else ""
    return SweepRun(
        run_id=f"{prefix}{replay_tag}{chat_tag}-ep{epochs_tag}-lr{_lr_tag(peak_lr)}",
        epochs=float(epochs), peak_lr=float(peak_lr), warmup=warm,
        train_tokens=int(train_tokens), max_steps=max_steps,
        effective_batch=effective_batch, context_len=context_len, eval_interval=eval_interval,
        replay_ratio=float(replay_ratio), chat_ratio=float(chat_ratio), seed=seed,
        schedule_start=seed_step,
    )


def build_sweep(train_tokens: int, epochs=DEFAULT_EPOCHS, peak_lrs=DEFAULT_PEAK_LRS, **kw) -> list[SweepRun]:
    runs = [make_run(train_tokens, e, lr, **kw) for e in epochs for lr in peak_lrs]
    ids = [r.run_id for r in runs]
    if len(set(ids)) != len(ids):
        raise ValueError("duplicate run ids in sweep")
    return runs


ROUND3_REPLAY_RATIOS = (0.25, 0.5)
ROUND3_PEAK_LR = 1e-5
ROUND3_OPTIONAL_CELL = (0.5, 3e-5)  # (replay_ratio, peak_lr), only if quota allows


def build_sweep_round3(train_tokens: int, replay_ratios=ROUND3_REPLAY_RATIOS, peak_lr: float = ROUND3_PEAK_LR,
                       epochs: float = 1.0, include_optional: bool = False, **kw) -> list[SweepRun]:
    """Round 3: replay_ratio x a gentle LR, one SFT epoch, from Stage C 31101."""
    cells = [(r, peak_lr) for r in replay_ratios]
    if include_optional and ROUND3_OPTIONAL_CELL not in cells:
        cells.append(ROUND3_OPTIONAL_CELL)
    runs = [make_run(train_tokens, epochs, lr, prefix="stage-e-r3", replay_ratio=r, **kw) for r, lr in cells]
    if len({r.run_id for r in runs}) != len(runs):
        raise ValueError("duplicate run ids in sweep")
    return runs


# Stage F cells: (seed, replay_ratio, chat_ratio). Rows per 24-row update are
# facts/chat/replay = 8/4/12 for the rr50 cells and 12/6/6 for rr25; per rank
# (12 rows a step) that is 4/2/6 and 6/3/3, all whole numbers.
STAGE_F_PEAK_LR = 1e-5
STAGE_F_CELLS = (
    ("E31140", 0.50, 1 / 6),  # primary: most replay, given E31140's ~0.013 FineWeb headroom
    ("E31140", 0.25, 0.25),   # second: more fact rows, less replay
)
STAGE_F_CONTROL = ("C31101", 0.50, 1 / 6)  # optional control from the Stage C base


def build_sweep_stage_f(fact_tokens: int, *, include_control: bool = False, peak_lr: float = STAGE_F_PEAK_LR,
                        epochs: float = 1.0, **kw) -> list[SweepRun]:
    """Stage F: facts + simple math. `fact_tokens` is the new fact/arith SFT train set's token count."""
    cells = list(STAGE_F_CELLS) + ([STAGE_F_CONTROL] if include_control else [])
    runs = [make_run(fact_tokens, epochs, peak_lr, prefix=f"stage-f-{seed.lower()}", seed=seed,
                     replay_ratio=rr, chat_ratio=ch, **kw) for seed, rr, ch in cells]
    if len({r.run_id for r in runs}) != len(runs):
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
        # Which checkpoint to start from; the launcher must verify the sha256.
        "PLM_SEED_STEP": str(SEEDS[run.seed][0]),
        "PLM_SEED_SHA256": SEEDS[run.seed][1],
        # Round 3: share of rows from Stage C replay. Dee mounts the Stage C
        # train.bin and sets PLM_REPLAY_BIN; with ratio 0 the key is omitted.
        **({"PLM_REPLAY_RATIO": repr(run.replay_ratio)} if run.replay_ratio else {}),
        # Stage F: share of rows from the existing chat SFT (PLM_CHAT_DIR holds
        # its train.bin + train.masks.bin); omitted when 0.
        **({"PLM_CHAT_RATIO": repr(run.chat_ratio)} if run.chat_ratio else {}),
    }


def verify_seed(weights_path: str | os.PathLike, seed: str = "C31101") -> str:
    """Raise unless the file is the named seed (by sha256)."""
    step, want = SEEDS[seed]
    h = hashlib.sha256()
    with open(weights_path, "rb") as f:
        for block in iter(lambda: f.read(1 << 20), b""):
            h.update(block)
    digest = h.hexdigest()
    if digest != want:
        label = "Stage C 31101" if seed == "C31101" else seed
        raise ValueError(f"{Path(weights_path).name} is not the {label} seed (sha256 {digest[:12]}...)")
    return digest
