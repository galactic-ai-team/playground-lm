# notebooks/playground_lm_train.py
# Kaggle/Colab training entry point for PlaygroundLM.
#
# This file is the single source of truth for the training notebook. The
# committed .ipynb is generated from it by scripts/make_notebook.py (Jupytext-
# style `# %%` markers); CI runs `make_notebook.py --check` to fail if the two
# drift apart.
#
# ---
#
# Run this on Kaggle (or Colab) as:
#   1. Attach the PlaygroundLM data as a Kaggle dataset.
#   2. Set the env vars in the "Config" cell (or via Kaggle secrets).
#   3. Run all cells.
#
# The training loop lives in train/loop.py (Codex's lane). This notebook
# imports it and drives it with checkpointing (my lane) and dataset loading
# (Nova's + my dataloader).

# %% [markdown]
# # PlaygroundLM training
#
# From-scratch GPT-2-small-class model (124M params, 12 layers, 12 heads,
# 768 dims, 1024 context) trained on FineWeb-Edu + Playground text.
#
# Free-tools only: Kaggle P100 or T4, checkpointed to a local run dir,
# uploaded to a GitHub release via `gh`.
#
# See `README.md` for the full plan and eval setup.

# %% [markdown]
# ## Config
#
# Set these before running the next cell:
#
# | Env var | Meaning | Default |
# |---|---|---|
# | `PLM_STAGE` | Training stage preset: `A`, `B`, `C` — sets `max_steps` and milestones | unset |
# | `PLM_MAX_STEPS` | Override the stage's `max_steps` | stage default, else 200_000 |
# | `PLM_BATCH_SIZE` | Micro-batch size; `grad_accum_steps` is adjusted to keep the effective batch at 96 | `12` (P100); use `6` on T4 |
# | `PLM_RUN_DIR` | Local checkpoint dir (survives within a session) | `/kaggle/working/runs/plm-001` |
# | `PLM_DATA_DIR` | Directory with `train.bin`, `val.bin`, `tokenizer/` | `/kaggle/input/plm-data` |
# | `PLM_REPO` | GitHub repo for release upload | `galactic-ai-team/playground-lm` |
# | `PLM_RELEASE_TAG` | Release tag for checkpoints | `v0.1-checkpoints` |
# | `PLM_RUN_ID` | Manifest run id | `plm-001` |
# | `PLM_COMPILE` | Set to `1` to enable `torch.compile` | unset (off) |
#
# **Stage presets** (from `data/PRETRAIN_PLAN.md` token budgets, converted to
# step counts at 98,304 tokens/step = 12 micro-batch × 8 accum × 1024 context):
#
# | Stage | Token budget | Approx steps | Milestones |
# |---|---|---|---|
# | A | 20M | 204 | 50, 100, 150, 204 |
# | B | 200M | 2,035 | 500, 1000, 1500, 2035 |
# | C | 1B | 10,172 | 2000, 5000, 8000, 10172 |
#
# **GPU memory note:** batch 12 fits on a P100 (16 GB) with bf16, but NOT on
# a T4 (16 GB, no bf16). For T4, set `PLM_BATCH_SIZE=6` — grad accum
# auto-adjusts to 16 to keep the effective batch at 96. `train_loop` handles
# fp16 + GradScaler on non-bf16 hardware; the notebook only passes the batch
# size and the resumed scaler state.
#
# `GH_TOKEN` must be set as a Kaggle secret with `repo` scope for the upload
# step to work. Without it, training still runs; only checkpoint upload is
# skipped (you'll see a warning).

# %%
import os
import sys
import time
from pathlib import Path

import torch

RUN_DIR = Path(os.environ.get("PLM_RUN_DIR", "/kaggle/working/runs/plm-001"))
DATA_DIR = Path(os.environ.get("PLM_DATA_DIR", "/kaggle/input/plm-data"))
REPO = os.environ.get("PLM_REPO", "galactic-ai-team/playground-lm")
RELEASE_TAG = os.environ.get("PLM_RELEASE_TAG", "v0.1-checkpoints")
RUN_ID = os.environ.get("PLM_RUN_ID", "plm-001")
UPLOAD_EVERY_N_SAVES = int(os.environ.get("PLM_UPLOAD_EVERY", "4"))
COMPILE = os.environ.get("PLM_COMPILE", "0") == "1"

# --- Model architecture (passed to PlaygroundLM(**MODEL_CONFIG)) ---------
MODEL_CONFIG = {
    "vocab_size": 50257,
    "n_layers": 12,
    "n_heads": 12,
    "d_model": 768,
    "context_len": 1024,
}

# --- Batch size + gradient accumulation --------------------------------
TARGET_EFFECTIVE_BATCH = 96
DEFAULT_MICRO_BATCH = 12
micro_batch = int(os.environ.get("PLM_BATCH_SIZE", str(DEFAULT_MICRO_BATCH)))
if micro_batch <= 0:
    raise ValueError(f"PLM_BATCH_SIZE must be positive, got {micro_batch}")
if TARGET_EFFECTIVE_BATCH % micro_batch != 0:
    raise ValueError(
        f"PLM_BATCH_SIZE={micro_batch} does not divide "
        f"TARGET_EFFECTIVE_BATCH={TARGET_EFFECTIVE_BATCH}; "
        f"grad_accum_steps would not be an integer. "
        f"Pick a divisor of {TARGET_EFFECTIVE_BATCH}: "
        f"{[d for d in range(1, TARGET_EFFECTIVE_BATCH + 1) if TARGET_EFFECTIVE_BATCH % d == 0]}"
    )
grad_accum = TARGET_EFFECTIVE_BATCH // micro_batch

# --- Training hyperparameters ------------------------------------------
TRAIN_CONFIG = {
    "batch_size": micro_batch,
    "grad_accum_steps": grad_accum,
    # Required by train_loop (tokens_per_step); keep in sync with the model.
    "context_len": MODEL_CONFIG["context_len"],
    "lr": 6e-4,
    "warmup_steps": 2000,
    "weight_decay": 0.1,
    "grad_clip": 1.0,
}
# Optional: stop cleanly (after a save) before Kaggle's 12-hour session cap.
if os.environ.get("PLM_MAX_SESSION_S"):
    TRAIN_CONFIG["max_session_s"] = float(os.environ["PLM_MAX_SESSION_S"])

# --- Stage presets -----------------------------------------------------
STAGE_PRESETS = {
    "A": {"max_steps": 204, "milestones": (50, 100, 150, 204)},
    "B": {"max_steps": 2035, "milestones": (500, 1000, 1500, 2035)},
    "C": {"max_steps": 10172, "milestones": (2000, 5000, 8000, 10172)},
}

stage = os.environ.get("PLM_STAGE", "").upper() or None
if stage is not None and stage not in STAGE_PRESETS:
    raise ValueError(
        f"PLM_STAGE={stage!r} is not a known stage "
        f"(expected one of {sorted(STAGE_PRESETS)} or unset)"
    )

if stage is not None:
    preset = STAGE_PRESETS[stage]
    MAX_STEPS = preset["max_steps"]
    SAVE_MILESTONES = preset["milestones"]
else:
    MAX_STEPS = 200_000
    SAVE_MILESTONES = None

max_steps_env = os.environ.get("PLM_MAX_STEPS")
if max_steps_env:
    MAX_STEPS = int(max_steps_env)
    print(f"[train] PLM_MAX_STEPS override → max_steps = {MAX_STEPS}")

if micro_batch != DEFAULT_MICRO_BATCH:
    print(
        f"[train] PLM_BATCH_SIZE override → batch_size = {micro_batch}, "
        f"grad_accum_steps = {grad_accum} (effective batch {micro_batch * grad_accum})"
    )

MANIFEST_CONFIG = {
    "model": MODEL_CONFIG,
    "train": TRAIN_CONFIG,
    "stage": stage,
    "max_steps": MAX_STEPS,
    "tokens_per_step": TARGET_EFFECTIVE_BATCH * MODEL_CONFIG["context_len"],
}

# Make the src/ package importable when running from the repo root.
sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))

# %% [markdown]
# ## Imports from other lanes
#
# - `playground_lm.model.PlaygroundLM` (Codex / Claude): `forward(x, y=None) -> (logits, loss)`.
# - `playground_lm.tokenizer.get_tokenizer()` (GPT-6): GPT-2 BPE via tiktoken.
# - `playground_lm.train.data.build_dataloader` (mine): memmap'd uint16 token file.
# - `playground_lm.train.checkpoint.CheckpointManager` (mine): save / resume / prune.
# - `playground_lm.train.loop` (Codex): the training loop driver.

# %%
from playground_lm.model import PlaygroundLM
from playground_lm.tokenizer import get_tokenizer
from playground_lm.train.checkpoint import CheckpointManager
from playground_lm.train.data import build_dataloader
from playground_lm.train.loop import train as train_loop
from playground_lm.train.upload_release import upload as upload_release

# %% [markdown]
# ## Device check
#
# Kaggle's free GPU is usually a P100 (16GB) or T4 (16GB). If neither is
# available, training still runs on CPU — extremely slowly.
#
# `torch.compile` is opt-in (`PLM_COMPILE=1`). It can produce a meaningful
# speedup but sometimes fails on older torch versions or Kaggle's specific
# CUDA toolchain, so the default is off — a fresh session shouldn't crash on
# compile.

# %%
device = "cuda" if torch.cuda.is_available() else "cpu"
if device == "cpu":
    print("[train] WARNING: no GPU. This will be extremely slow.")
else:
    print(f"[train] device: {torch.cuda.get_device_name(0)}")
print(f"[train] stage: {stage or '(none)'}  max_steps: {MAX_STEPS}")
print(f"[train] micro-batch: {micro_batch}  grad_accum: {grad_accum}  "
      f"effective: {micro_batch * grad_accum}")
print(f"[train] torch.compile: {'ON' if COMPILE else 'off (set PLM_COMPILE=1 to enable)'}")

# %% [markdown]
# ## Model + optimizer
#
# `PlaygroundLM` is initialized from random init — no pretrained weights.
# Only `MODEL_CONFIG` goes to the constructor; `TRAIN_CONFIG` stays in this
# notebook and goes to `train_loop`.

# %%
model = PlaygroundLM(**MODEL_CONFIG).to(device)
if COMPILE and device == "cuda":
    model = torch.compile(model)
optimizer = torch.optim.AdamW(
    model.parameters(),
    lr=TRAIN_CONFIG["lr"],
    weight_decay=TRAIN_CONFIG["weight_decay"],
    betas=(0.9, 0.95),
)

# %% [markdown]
# ## Checkpoint manager + resume
#
# Runs BEFORE the data loaders are built, so that `start_step` can be passed
# as `seed_offset` on the train loader — otherwise a resumed session replays
# the exact same sampled windows as the previous session's first N batches.
#
# `CheckpointManager` handles:
# - Resume from the latest checkpoint if one exists in `RUN_DIR`.
# - Save every ~30 min (via `interval_s`) and at the stage's milestone steps.
# - Keep the last N rolling checkpoints + all milestones.
# - Emit a `manifest.json` the release script and demo frontend read.
#
# Resume also pulls `wall_time_s` (cumulative session time) and
# `extra.grad_scaler` (GradScaler state from Claude's fp16 path) so a T4
# session doesn't lose the scaler between resumes.

# %%
ckpt_kwargs = {
    "run_dir": RUN_DIR,
    "run_id": RUN_ID,
    "config": MANIFEST_CONFIG,
}
if SAVE_MILESTONES is not None:
    ckpt_kwargs["milestones"] = SAVE_MILESTONES
ckpt = CheckpointManager(**ckpt_kwargs)

start_step = 0
start_tokens_seen = 0
start_wall_time_s = 0.0
start_scaler_state = None
resumed = ckpt.resume(model, optimizer)
if resumed is not None:
    start_step = resumed["step"]
    start_tokens_seen = resumed["tokens_seen"]
    start_wall_time_s = resumed.get("wall_time_s", 0.0)
    start_scaler_state = resumed.get("extra", {}).get("grad_scaler")
    print(
        f"[train] resumed from step {start_step} "
        f"({start_tokens_seen:,} tokens, {start_wall_time_s:.1f}s wall)"
    )
    if start_scaler_state is not None:
        print("[train] grad_scaler state restored")

# %% [markdown]
# ## Data
#
# Nova's data lane publishes three files under `PLM_DATA_DIR`:
#
# - `train.bin` — flat uint16 token IDs, all of FineWeb-Edu sample + Playground text
# - `val.bin`   — held-out split, same format
# - `tokenizer/` — cached GPT-2 BPE assets
#
# `build_dataloader` memory-maps the file, so the corpus can be larger than RAM.
# Train uses `stride=1` (every position is a sample) and `seed_offset=start_step`
# so resumed sessions sample different windows. Val uses the default
# `stride=context_len` (non-overlapping, unbiased loss estimate).

# %%
train_loader = build_dataloader(
    DATA_DIR / "train.bin",
    context_len=MODEL_CONFIG["context_len"],
    batch_size=TRAIN_CONFIG["batch_size"],
    shuffle=True,
    stride=1,
    seed_offset=start_step,   # <-- resume non-replay
)
val_loader = build_dataloader(
    DATA_DIR / "val.bin",
    context_len=MODEL_CONFIG["context_len"],
    batch_size=TRAIN_CONFIG["batch_size"],
    shuffle=False,   # stride defaults to context_len (eval mode)
)

# %% [markdown]
# ## Train
#
# Everything below is driven by `train_loop`. The loop:
# - Runs forward+backward with `grad_accum_steps` accumulation.
# - Applies the LR schedule (linear warmup, cosine decay).
# - Every N steps, checks `ckpt.should_save(step)` and saves if due.
# - Every save, uploads to the GitHub release (every `UPLOAD_EVERY_N_SAVES` saves).
# - Every 500 steps, computes val loss on 20 val batches.
#
# **T4 users:** `TRAIN_CONFIG["batch_size"]` is the micro-batch.
# `train_loop` reads `grad_accum_steps` and applies fp16 autocast +
# GradScaler on hardware without bf16 (T4), restoring from
# `start_scaler_state` on resume.

# %%
def upload_hook(step: int, save_count: int) -> None:
    """Called by train_loop after each save. Uploads every N saves."""
    if save_count % UPLOAD_EVERY_N_SAVES != 0:
        return
    try:
        upload_release(
            run_dir=RUN_DIR,
            repo=REPO,
            tag=RELEASE_TAG,
            gh_token=os.environ["GH_TOKEN"],
        )
    except KeyError:
        print("[train] WARN: GH_TOKEN not set; skipping checkpoint upload")
    except Exception as e:
        # Upload failure should not kill training.
        print(f"[train] WARN: upload failed: {e}")


try:
    train_loop(
        model=model,
        optimizer=optimizer,
        train_loader=train_loader,
        val_loader=val_loader,
        checkpoint_manager=ckpt,
        config=TRAIN_CONFIG,
        start_step=start_step,
        start_tokens_seen=start_tokens_seen,
        start_wall_time_s=start_wall_time_s,
        start_scaler_state=start_scaler_state,
        max_steps=MAX_STEPS,
        device=device,
        on_save=upload_hook,
    )
except KeyboardInterrupt:
    print("[train] interrupted by user; last checkpoint saved")
    raise

# %% [markdown]
# ## Final upload
#
# Make sure the last checkpoint is on the release even if the loop exited
# between save points.

# %%
try:
    upload_release(
        run_dir=RUN_DIR,
        repo=REPO,
        tag=RELEASE_TAG,
        gh_token=os.environ["GH_TOKEN"],
    )
except KeyError:
    print("[train] GH_TOKEN not set; final upload skipped")
except Exception as e:
    print(f"[train] final upload failed: {e}")