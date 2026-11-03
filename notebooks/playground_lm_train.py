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
# Multi-GPU: launch with
#   torchrun --standalone --nproc_per_node=2 notebooks/playground_lm_train.py
# Each rank runs the training loop; only rank 0 saves checkpoints and uploads.
# Single-GPU and CPU runs use rank 0 only and behave exactly as before.
#
# The training loop lives in train/loop.py (Codex/Claude's lane). This notebook
# imports it and drives it with checkpointing (my lane) and dataset loading
# (Nova's + my dataloader).

# %% [markdown]
# # PlaygroundLM training
#
# From-scratch GPT-2-small-class model (124M params, 12 layers, 12 heads,
# 768 dims, 1024 context) trained on FineWeb-Edu + Playground text.
#
# Free-tools only: Kaggle P100/T4, checkpointed to a local run dir, uploaded
# to a GitHub release via `gh`.

#
# See `README.md` for the full plan and eval setup.

# %% [markdown]
# ## Config
#
# Set these before running the next cell:
#
# | Env var | Meaning | Default |
# |---|---|---|
# | `PLM_STAGE` | Training stage preset: `A`, `B`, `C` — sets `max_steps`, `warmup_steps`, milestones | unset |
# | `PLM_MAX_STEPS` | Override the stage's `max_steps` | stage default, else 200_000 |
# | `PLM_WARMUP` | Override the stage's `warmup_steps` | stage default, else 2000 |
# | `PLM_BATCH_SIZE` | Micro-batch size per GPU; `grad_accum_steps` is adjusted to keep the global effective batch at 96 sequences | `12` (P100); use `6` on T4 |
# | `PLM_MAX_SESSION_S` | Wall-clock cap in seconds; loop stops gracefully before Kaggle's 12h GPU session kill | unset |
# | `PLM_RUN_DIR` | Local checkpoint dir (survives within a session) | `/kaggle/working/runs/plm-001` |
# | `PLM_DATA_DIR` | Directory with `train.bin`, `val.bin`, `tokenizer/` | `/kaggle/input/plm-data` |
# | `PLM_REPO` | GitHub repo for release upload | `galactic-ai-team/playground-lm` |
# | `PLM_RELEASE_TAG` | Release tag for checkpoints | `v0.1-checkpoints` |
# | `PLM_RUN_ID` | Manifest run id | `plm-001` |
# | `PLM_COMPILE` | Set to `1` to enable `torch.compile` | unset (off) |
# | `PLM_DIST_TIMEOUT_MIN` | Multi-GPU only: how long a GPU waits for the other (e.g. during a rank-0 checkpoint upload) before giving up | `60` |
#
# **Stage presets** — see `src/playground_lm/train/stage_config.py` for the
# source of truth. Resolved at runtime by `resolve_stage_plan()`.
#
# **GPU memory note:** batch 12 fits on a P100 (16 GB) with bf16, but NOT on
# a T4 (16 GB, no bf16). For T4, set `PLM_BATCH_SIZE=6` — grad accum
# auto-adjusts to keep the global effective batch at 96 sequences: 16 on one
# T4, 8 per GPU on two. `train_loop` handles fp16 + GradScaler on non-bf16
# hardware.
#

# **Kaggle session limits:** GPU sessions are killed at 12 hours; TPU sessions
# at 9 hours. Set `PLM_MAX_SESSION_S` below the limit so a final checkpoint
# lands before the session dies.
#
# `GH_TOKEN` must be set as a Kaggle secret with `repo` scope for the upload
# step to work. Without it, training still runs; only checkpoint upload is
# skipped (you'll see a warning).

# %%
import os
import sys
from pathlib import Path

import torch

RUN_DIR = Path(os.environ.get("PLM_RUN_DIR", "/kaggle/working/runs/plm-001"))
DATA_DIR = Path(os.environ.get("PLM_DATA_DIR", "/kaggle/input/plm-data"))
REPO = os.environ.get("PLM_REPO", "galactic-ai-team/playground-lm")
RELEASE_TAG = os.environ.get("PLM_RELEASE_TAG", "v0.1-checkpoints")
RUN_ID = os.environ.get("PLM_RUN_ID", "plm-001")
UPLOAD_EVERY_N_SAVES = int(os.environ.get("PLM_UPLOAD_EVERY", "4"))
COMPILE = os.environ.get("PLM_COMPILE", "0") == "1"

# Make the src/ package importable when running from the repo root.
# (Must come before any playground_lm import.)
sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))

# --- Distributed setup (Claude's distributed.py) -----------------------
# setup() reads torchrun's RANK/LOCAL_RANK/WORLD_SIZE, calls
# init_process_group, and sets the CUDA device. No-op without torchrun.
from playground_lm.train.distributed import (
    setup,

    device_for,
    wrap,
    barrier,
    cleanup,
)

info = setup()
IS_MAIN = info.is_main
RANK = info.rank
WORLD_SIZE = info.world_size

def _log(msg: str) -> None:
    """Print only from rank 0 so multi-GPU logs aren't duplicated."""
    if IS_MAIN:
        print(msg)

_log(f"[train] rank {RANK}/{WORLD_SIZE}")

# --- Model architecture (passed to PlaygroundLM(**MODEL_CONFIG)) ---------
MODEL_CONFIG = {
    "vocab_size": 50257,
    "n_layers": 12,
    "n_heads": 12,
    "d_model": 768,
    "context_len": 1024,
}

# --- Batch size + gradient accumulation --------------------------------
# TARGET_EFFECTIVE_BATCH is the GLOBAL effective batch: sequences per
# optimizer step summed over all GPUs. Keeping it fixed (Claude's review)
# means the LR schedule, the stage presets (which count steps) and
# tokens_per_step stay the same whether a session runs on one T4 or two, so

# a run can resume on either. Two GPUs halve the per-GPU accumulation instead.
TARGET_EFFECTIVE_BATCH = 96
DEFAULT_MICRO_BATCH = 12
micro_batch = int(os.environ.get("PLM_BATCH_SIZE", str(DEFAULT_MICRO_BATCH)))
if micro_batch <= 0:
    raise ValueError(f"PLM_BATCH_SIZE must be positive, got {micro_batch}")
_per_micro_step = micro_batch * WORLD_SIZE  # sequences per micro-step, all GPUs
if TARGET_EFFECTIVE_BATCH % _per_micro_step != 0:
    raise ValueError(
        f"PLM_BATCH_SIZE={micro_batch} x {WORLD_SIZE} GPU(s) does not divide "
        f"TARGET_EFFECTIVE_BATCH={TARGET_EFFECTIVE_BATCH}; "
        f"grad_accum_steps would not be an integer. "
        f"Valid PLM_BATCH_SIZE values here: "
        f"{[d for d in range(1, TARGET_EFFECTIVE_BATCH + 1) if TARGET_EFFECTIVE_BATCH % (d * WORLD_SIZE) == 0]}"
    )
grad_accum = TARGET_EFFECTIVE_BATCH // _per_micro_step

# --- Stage presets + env overrides (logic in stage_config.py) ----------
from playground_lm.train.stage_config import resolve_stage_plan, warmup_warning

_plan = resolve_stage_plan(
    stage=os.environ.get("PLM_STAGE"),
    max_steps_override=(
        int(os.environ["PLM_MAX_STEPS"])
        if os.environ.get("PLM_MAX_STEPS") else None
    ),
    warmup_override=(
        int(os.environ["PLM_WARMUP"])
        if os.environ.get("PLM_WARMUP") else None
    ),
)
MAX_STEPS = _plan.max_steps

WARMUP_STEPS = _plan.warmup_steps
SAVE_MILESTONES = _plan.milestones
stage = _plan.stage

_warn = warmup_warning(_plan)
if _warn:
    _log(f"[train] WARN: {_warn}")

if os.environ.get("PLM_MAX_STEPS"):
    _log(f"[train] PLM_MAX_STEPS override → max_steps = {MAX_STEPS}")
if os.environ.get("PLM_WARMUP"):
    _log(f"[train] PLM_WARMUP override → warmup_steps = {WARMUP_STEPS}")
if micro_batch != DEFAULT_MICRO_BATCH:
    _log(
        f"[train] PLM_BATCH_SIZE override → batch_size = {micro_batch}, "
        f"grad_accum_steps = {grad_accum} (global effective "
        f"{micro_batch * grad_accum * WORLD_SIZE})"
    )

# Optional wall-clock cap. Kaggle GPU sessions are killed at 12 hours; set
# this lower (e.g. 11 * 3600) so train_loop saves a final checkpoint and
# exits cleanly instead of being SIGKILLed mid-step.
max_session_env = os.environ.get("PLM_MAX_SESSION_S")
MAX_SESSION_S = float(max_session_env) if max_session_env else None
if MAX_SESSION_S is not None:
    _log(f"[train] PLM_MAX_SESSION_S = {MAX_SESSION_S:.0f}s ({MAX_SESSION_S/3600:.2f}h)")

# --- Training hyperparameters ------------------------------------------
# max_session_s goes into TRAIN_CONFIG — loop.py reads config.get("max_session_s").
TRAIN_CONFIG = {
    "batch_size": micro_batch,
    "grad_accum_steps": grad_accum,

    "context_len": MODEL_CONFIG["context_len"],
    "lr": 6e-4,
    "warmup_steps": WARMUP_STEPS,
    "weight_decay": 0.1,
    "grad_clip": 1.0,
}
if MAX_SESSION_S is not None:
    TRAIN_CONFIG["max_session_s"] = MAX_SESSION_S

MANIFEST_CONFIG = {
    "model": MODEL_CONFIG,
    "train": TRAIN_CONFIG,
    "stage": stage,
    "max_steps": MAX_STEPS,
    "warmup_steps": WARMUP_STEPS,
    "max_session_s": MAX_SESSION_S,
    "tokens_per_step": TARGET_EFFECTIVE_BATCH * MODEL_CONFIG["context_len"],
    "world_size": WORLD_SIZE,
}

# %% [markdown]
# ## Imports from other lanes
#
# - `playground_lm.model.PlaygroundLM` (Codex / Claude): `forward(x, y=None) -> (logits, loss)`.
# - `playground_lm.tokenizer.get_tokenizer()` (GPT-6): GPT-2 BPE via tiktoken.
# - `playground_lm.train.data.build_dataloader` (mine): memmap'd uint16 token file.
# - `playground_lm.train.checkpoint.CheckpointManager` (mine): save / resume / prune.
# - `playground_lm.train.loop` (Codex/Claude): the training loop driver.
# - `playground_lm.train.distributed` (Claude): setup / wrap / cleanup / barrier.

# %%
from playground_lm.model import PlaygroundLM

from playground_lm.tokenizer import get_tokenizer
from playground_lm.train.checkpoint import CheckpointManager
from playground_lm.train.data import build_dataloader
from playground_lm.train.loop import train as train_loop
from playground_lm.train.upload_release import upload as upload_release

# %% [markdown]
# ## Device
#
# `device_for(info)` returns `cuda:LOCAL_RANK` under torchrun, `cuda` or
# `cpu` otherwise. `setup()` has already bound the process to the right GPU.

# %%
device = device_for(info)
_log(f"[train] device: {device}")
if str(device).startswith("cpu"):
    _log("[train] WARNING: no GPU. This will be extremely slow.")
else:
    _log(f"[train] GPU: {torch.cuda.get_device_name(torch.cuda.current_device())}")
_log(f"[train] stage: {stage or '(none)'}  max_steps: {MAX_STEPS}  warmup_steps: {WARMUP_STEPS}")
_log(f"[train] micro-batch: {micro_batch}  grad_accum: {grad_accum}  "
     f"per-rank effective: {micro_batch * grad_accum}  "
     f"global effective: {micro_batch * grad_accum * WORLD_SIZE}")
_log(f"[train] torch.compile: {'ON' if COMPILE else 'off (set PLM_COMPILE=1 to enable)'}")

# %% [markdown]
# ## Model + optimizer (before DDP wrap)
#
# `PlaygroundLM` is initialized from random init — no pretrained weights.
# Only `MODEL_CONFIG` goes to the constructor; `TRAIN_CONFIG` goes to
# `train_loop`.
#

# **Order matters:** build the raw model, resume weights from the checkpoint
# into it, THEN call `wrap(model, info)`, THEN `torch.compile` (if enabled).
# `wrap()` broadcasts rank 0's weights to the other ranks, so resume has to
# happen on rank 0 first. Compiling last matters too: a compiled module's
# state_dict keys gain an `_orig_mod.` prefix, so resuming a checkpoint
# (saved from the plain model) into an already-compiled model fails with
# missing keys; and PyTorch recommends compile(DDP(model)) for DDP.

# %%
model = PlaygroundLM(**MODEL_CONFIG).to(device)
optimizer = torch.optim.AdamW(
    model.parameters(),
    lr=TRAIN_CONFIG["lr"],
    weight_decay=TRAIN_CONFIG["weight_decay"],
    betas=(0.9, 0.95),
)

# %% [markdown]
# ## Checkpoint manager + resume
#
# **Every rank** builds a `CheckpointManager` on the same `RUN_DIR`. Under
# torchrun all ranks share the filesystem (same machine), so each rank's
# `resume()` reads the same checkpoint independently — no broadcast needed.
# `train_loop` already skips saves and `on_save` on non-main ranks.
#
# Runs BEFORE the data loaders are built, so `start_step` can be passed as
# `seed_offset` on the train loader.

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
    _log(
        f"[train] resumed from step {start_step} "
        f"({start_tokens_seen:,} tokens, {start_wall_time_s:.1f}s wall)"
    )
    if start_scaler_state is not None:
        _log("[train] grad_scaler state restored")

# %% [markdown]
# ## DDP wrap
#
# `wrap(model, info)` broadcasts rank 0's weights to all ranks and returns
# the DDP-wrapped model (or the plain model if `WORLD_SIZE == 1`). Runs AFTER
# resume so the broadcast includes the resumed state on rank 0.

# %%
model = wrap(model, info)

if COMPILE and str(device).startswith("cuda"):
    model = torch.compile(model)  # after resume and wrap (see above)

# %% [markdown]
# ## Data
#
# Nova's data lane publishes three files under `PLM_DATA_DIR`:
#
# - `train.bin` — flat uint16 token IDs
# - `val.bin`   — held-out split, same format
# - `tokenizer/` — cached GPT-2 BPE assets
#
# `seed_offset=start_step + RANK * 1_000_000` gives each rank a distinct
# window stream while keeping determinism. On resume, all ranks shift by the
# same `start_step`, preserving the no-replay guarantee.

# %%
train_loader = build_dataloader(
    DATA_DIR / "train.bin",
    context_len=MODEL_CONFIG["context_len"],
    batch_size=TRAIN_CONFIG["batch_size"],
    shuffle=True,
    stride=1,
    seed_offset=start_step + RANK * 1_000_000,
)
val_loader = build_dataloader(
    DATA_DIR / "val.bin",
    context_len=MODEL_CONFIG["context_len"],
    batch_size=TRAIN_CONFIG["batch_size"],
    shuffle=False,
)


# %% [markdown]
# ## Train
#
# All ranks call `train_loop`. Rank 0 handles checkpoint saves and release
# uploads; `train_loop` internally skips those on non-main ranks.

# %%
def upload_hook(step: int, save_count: int) -> None:
    """Called by train_loop after each save on rank 0. Uploads every N saves."""
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
        _log("[train] WARN: GH_TOKEN not set; skipping checkpoint upload")
    except Exception as e:
        _log(f"[train] WARN: upload failed: {e}")


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
    if IS_MAIN:
        _log("[train] interrupted by user; last checkpoint saved")
    raise

# %% [markdown]
# ## Teardown, then final upload (rank 0 only)
#
# All ranks meet at the barrier and leave the process group FIRST; only then
# does rank 0 upload. The other order (upload, then barrier) left rank 1
# waiting at the barrier for the whole upload, and a multi-GB upload can
# outlast the process-group timeout, which would crash the job while rank 0
# is still uploading. Once rank 1 has exited cleanly, torchrun simply waits
# for rank 0 to finish.

# %%
barrier(info)
cleanup()

if IS_MAIN:
    try:
        upload_release(
            run_dir=RUN_DIR,
            repo=REPO,

            tag=RELEASE_TAG,
            gh_token=os.environ["GH_TOKEN"],
        )
    except KeyError:
        _log("[train] GH_TOKEN not set; final upload skipped")
    except Exception as e:
        _log(f"[train] final upload failed: {e}")
