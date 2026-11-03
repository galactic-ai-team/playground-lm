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
# Each rank runs the training loop; only rank 0 prints, saves checkpoints and
# uploads to the release. Single-GPU and CPU runs use rank 0 only and behave
# exactly as before.
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
# | `PLM_BATCH_SIZE` | Micro-batch size; `grad_accum_steps` is adjusted to keep the effective batch at 96 | `12` (P100); use `6` on T4 |
# | `PLM_MAX_SESSION_S` | Wall-clock cap in seconds; loop stops gracefully before Kaggle's 12h GPU session kill | unset |
# | `PLM_RUN_DIR` | Local checkpoint dir (survives within a session) | `/kaggle/working/runs/plm-001` |
# | `PLM_DATA_DIR` | Directory with `train.bin`, `val.bin`, `tokenizer/` | `/kaggle/input/plm-data` |
# | `PLM_REPO` | GitHub repo for release upload | `galactic-ai-team/playground-lm` |
# | `PLM_RELEASE_TAG` | Release tag for checkpoints | `v0.1-checkpoints` |
# | `PLM_RUN_ID` | Manifest run id | `plm-001` |
# | `PLM_COMPILE` | Set to `1` to enable `torch.compile` | unset (off) |
#
# **Stage presets** — see `src/playground_lm/train/stage_config.py` for the
# source of truth. Resolved at runtime by `resolve_stage_plan()`.
#
# **GPU memory note:** batch 12 fits on a P100 (16 GB) with bf16, but NOT on
# a T4 (16 GB, no bf16). For T4, set `PLM_BATCH_SIZE=6` — grad accum
# auto-adjusts to 16 to keep the effective batch at 96. `train_loop` handles
# fp16 + GradScaler on non-bf16 hardware; the notebook only passes the batch
# size and the resumed scaler state.
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

# --- Distributed launch detection ---------------------------------------
# torchrun sets RANK and WORLD_SIZE in the environment. Single-process runs
# (plain `python notebook.py` or Jupyter) have neither.
RANK = int(os.environ.get("RANK", "0"))
WORLD_SIZE = int(os.environ.get("WORLD_SIZE", "1"))
LOCAL_RANK = int(os.environ.get("LOCAL_RANK", "0"))
IS_MAIN = RANK == 0

RUN_DIR = Path(os.environ.get("PLM_RUN_DIR", "/kaggle/working/runs/plm-001"))
DATA_DIR = Path(os.environ.get("PLM_DATA_DIR", "/kaggle/input/plm-data"))
REPO = os.environ.get("PLM_REPO", "galactic-ai-team/playground-lm")
RELEASE_TAG = os.environ.get("PLM_RELEASE_TAG", "v0.1-checkpoints")
RUN_ID = os.environ.get("PLM_RUN_ID", "plm-001")
UPLOAD_EVERY_N_SAVES = int(os.environ.get("PLM_UPLOAD_EVERY", "4"))
COMPILE = os.environ.get("PLM_COMPILE", "0") == "1"

def _log(msg: str) -> None:
    """Print only from rank 0 so multi-GPU logs aren't duplicated."""
    if IS_MAIN:
        print(msg)

# --- Model architecture (passed to PlaygroundLM(**MODEL_CONFIG)) ---------
MODEL_CONFIG = {
    "vocab_size": 50257,
    "n_layers": 12,
    "n_heads": 12,
    "d_model": 768,
    "context_len": 1024,
}

# --- Batch size + gradient accumulation --------------------------------
# TARGET_EFFECTIVE_BATCH is the per-rank effective batch. Under torchrun,
# the true global batch is TARGET_EFFECTIVE_BATCH * WORLD_SIZE. Set
# PLM_BATCH_SIZE so the per-rank micro-batch fits your GPU.
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

# Make the src/ package importable when running from the repo root.
# (Must come before the stage_config import below.)
sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))

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
        f"grad_accum_steps = {grad_accum} (effective per-rank batch {micro_batch * grad_accum})"
    )

# Optional wall-clock cap. Kaggle GPU sessions are killed at 12 hours; set
# this lower (e.g. 11 * 3600) so train_loop saves a final checkpoint and
# exits cleanly instead of being SIGKILLed mid-step.
max_session_env = os.environ.get("PLM_MAX_SESSION_S")
MAX_SESSION_S = float(max_session_env) if max_session_env else None
if MAX_SESSION_S is not None:
    _log(f"[train] PLM_MAX_SESSION_S = {MAX_SESSION_S:.0f}s ({MAX_SESSION_S/3600:.2f}h)")

# --- Training hyperparameters ------------------------------------------
# max_session_s goes into TRAIN_CONFIG (not a separate kwarg) — loop.py
# reads it via config.get("max_session_s").
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
# - `playground_lm.train.distributed` (Claude): multi-GPU helpers.

# %%
from playground_lm.model import PlaygroundLM
from playground_lm.tokenizer import get_tokenizer
from playground_lm.train.checkpoint import CheckpointManager
from playground_lm.train.data import build_dataloader
from playground_lm.train.loop import train as train_loop
from playground_lm.train.upload_release import upload as upload_release

# %% [markdown]
# ## Device + distributed init
#
# Under `torchrun`, each rank binds to `cuda:LOCAL_RANK`. Single-GPU runs
# (`RANK`/`WORLD_SIZE` unset) fall through to `cuda:0` or `cpu` unchanged.
#
# The loop is responsible for calling `init_process_group` (Claude's
# `distributed.py`); the notebook only sets up the device.

# %%
if torch.cuda.is_available():
    device = f"cuda:{LOCAL_RANK}" if WORLD_SIZE > 1 else "cuda"
    torch.cuda.set_device(LOCAL_RANK if WORLD_SIZE > 1 else 0)
else:
    device = "cpu"

_log(f"[train] rank {RANK}/{WORLD_SIZE}  device: {device}")
if device == "cpu":
    _log("[train] WARNING: no GPU. This will be extremely slow.")
else:
    _log(f"[train] GPU: {torch.cuda.get_device_name(LOCAL_RANK if WORLD_SIZE > 1 else 0)}")
_log(f"[train] stage: {stage or '(none)'}  max_steps: {MAX_STEPS}  warmup_steps: {WARMUP_STEPS}")
_log(f"[train] micro-batch: {micro_batch}  grad_accum: {grad_accum}  "
     f"per-rank effective: {micro_batch * grad_accum}  "
     f"global effective: {micro_batch * grad_accum * WORLD_SIZE}")
_log(f"[train] torch.compile: {'ON' if COMPILE else 'off (set PLM_COMPILE=1 to enable)'}")

# %% [markdown]
# ## Model + optimizer
#
# `PlaygroundLM` is initialized from random init — no pretrained weights.
# Only `MODEL_CONFIG` goes to the constructor; `TRAIN_CONFIG` stays in this
# notebook and goes to `train_loop`.
#
# Under torchrun, each rank builds the same model in its own CUDA context.
# `distributed.py` (called by `train_loop`) wraps the model in DDP.

# %%
model = PlaygroundLM(**MODEL_CONFIG).to(device)
if COMPILE and device.startswith("cuda"):
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
# Runs BEFORE the data loaders are built, so `start_step` can be passed as
# `seed_offset` on the train loader.
#
# Under multi-GPU, only rank 0 constructs the CheckpointManager (it writes
# files). Rank 0 resumes, then broadcasts `start_step` / `start_tokens_seen` /
# `start_wall_time_s` / scaler state to the other ranks via the distributed
# process group. **The notebook assumes `distributed.py` exposes a broadcast
# helper; if not, `torch.distributed.broadcast_object_list` is the fallback.**

# %%
# Single-rank resume state (world_size 1) — or placeholder before broadcast.
start_step = 0
start_tokens_seen = 0
start_wall_time_s = 0.0
start_scaler_state = None
ckpt = None

if IS_MAIN:
    ckpt_kwargs = {
        "run_dir": RUN_DIR,
        "run_id": RUN_ID,
        "config": MANIFEST_CONFIG,
    }
    if SAVE_MILESTONES is not None:
        ckpt_kwargs["milestones"] = SAVE_MILESTONES
    ckpt = CheckpointManager(**ckpt_kwargs)

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

# Broadcast resume state from rank 0 to all ranks (no-op for world_size 1).
if WORLD_SIZE > 1 and torch.distributed.is_initialized():
    _state = [start_step, start_tokens_seen, start_wall_time_s, start_scaler_state]
    torch.distributed.broadcast_object_list(_state, src=0)
    start_step, start_tokens_seen, start_wall_time_s, start_scaler_state = _state
    _log(f"[train] rank {RANK} received resume state: step={start_step}")

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
# Everything below is driven by `train_loop`. Under multi-GPU, all ranks call
# `train_loop`; rank 0 handles checkpoint saves and release uploads. The
# `on_save` hook only fires on rank 0 (checkpoint_manager is None on other
# ranks, so the loop should not call save there — see Claude's note in
# distributed.py).

# %%
def upload_hook(step: int, save_count: int) -> None:
    """Called by train_loop after each save. Uploads every N saves.
    Only ever called on rank 0 (ckpt is None on other ranks)."""
    if not IS_MAIN:
        return
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
        checkpoint_manager=ckpt,   # None on non-main ranks
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
# ## Final upload
#
# Rank 0 only. Non-main ranks have no ckpt and skip this entirely.

# %%
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