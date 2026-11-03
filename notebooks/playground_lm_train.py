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
# | `PLM_RUN_DIR` | Local checkpoint dir (survives within a session) | `/kaggle/working/runs/plm-001` |
# | `PLM_DATA_DIR` | Directory with `train.bin`, `val.bin`, `tokenizer/` | `/kaggle/input/plm-data` |
# | `PLM_REPO` | GitHub repo for release upload | `galactic-ai-team/playground-lm` |
# | `PLM_RELEASE_TAG` | Release tag for checkpoints | `v0.1-checkpoints` |
# | `PLM_RUN_ID` | Manifest run id | `plm-001` |
# | `PLM_COMPILE` | Set to `1` to enable `torch.compile` | unset (off) |
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

# --- Training hyperparameters (used by this notebook + train_loop) -------
TRAIN_CONFIG = {
    "batch_size": 12,       # micro-batch on a single P100 (16GB)
    "grad_accum_steps": 8,  # effective batch 96
    "lr": 6e-4,
    "warmup_steps": 2000,
    "weight_decay": 0.1,
    "grad_clip": 1.0,
}

# Manifest config: record both so a checkpoint's full training setup is
# recoverable from manifest.json alone.
MANIFEST_CONFIG = {"model": MODEL_CONFIG, "train": TRAIN_CONFIG}

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
# available, training still runs on CPU — extremely slowly. A single batch
# that takes ~0.2s on P100 takes ~15s on CPU.
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
print(f"[train] torch.compile: {'ON' if COMPILE else 'off (set PLM_COMPILE=1 to enable)'}")

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
# Train uses `stride=1` (every position is a sample); val uses the default
# `stride=context_len` (non-overlapping, unbiased loss estimate).

# %%
train_loader = build_dataloader(
    DATA_DIR / "train.bin",
    context_len=MODEL_CONFIG["context_len"],
    batch_size=TRAIN_CONFIG["batch_size"],
    shuffle=True,
    stride=1,
)
val_loader = build_dataloader(
    DATA_DIR / "val.bin",
    context_len=MODEL_CONFIG["context_len"],
    batch_size=TRAIN_CONFIG["batch_size"],
    shuffle=False,   # stride defaults to context_len (eval mode)
)

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
# ## Checkpoint manager
#
# Handles:
# - Resume from the latest checkpoint if one exists in `RUN_DIR`.
# - Save every ~30 min (via `interval_s`) and at milestone steps.
# - Keep the last N rolling checkpoints + all milestones.
# - Emit a `manifest.json` the release script and demo frontend read.

# %%
ckpt = CheckpointManager(
    run_dir=RUN_DIR,
    run_id=RUN_ID,
    config=MANIFEST_CONFIG,
)

start_step = 0
start_tokens_seen = 0
start_wall_time_s = 0.0
resumed = ckpt.resume(model, optimizer)
if resumed is not None:
    start_step = resumed["step"]
    start_tokens_seen = resumed["tokens_seen"]
    start_wall_time_s = resumed.get("wall_time_s", 0.0)
    print(
        f"[train] resumed from step {start_step} "
        f"({start_tokens_seen:,} tokens, {start_wall_time_s:.1f}s wall)"
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
# The loop returns when `max_steps` is reached or when the user interrupts.

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
        # loop.train needs context_len for tokens/step; it lives in MODEL_CONFIG.
        config={**TRAIN_CONFIG, "context_len": MODEL_CONFIG["context_len"]},
        start_step=start_step,
        start_tokens_seen=start_tokens_seen,
        start_wall_time_s=start_wall_time_s,
        max_steps=200_000,
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