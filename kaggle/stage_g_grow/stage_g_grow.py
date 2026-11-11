#!/usr/bin/env python3
"""Stage G Track 2: continue pretraining the grown 355M (24 x 1024) on free Kaggle T4 x2.

Lumi glue around kaggle/stage_c_train/stage_c_train.py (same FineWeb-Edu data
from the 8 data-prep kernels, same permute sampler, same 11 h session pattern).
Resume order each session:
  1. previous version of THIS kernel (self-reference kernel_source, runs/ tree), else
  2. the grown seed dataset galacticaiteam/plm-g355-seed (weights only).
No GitHub upload (checkpoints exceed release asset limits): everything stays in
/kaggle/working/runs, which Kaggle keeps as kernel output for the next session.
Free only; the per-session cap is PLM_MAX_SESSION_S.
"""
from __future__ import annotations

import importlib.util
import os
import shutil
import subprocess
import sys
from pathlib import Path

SELF_SLUG = os.environ.get("PLM_GROW_SLUG", "plm-stage-g-grow355")
SEED_SLUG = "plm-g355-seed"
SEED_STEP = 31101
GIT_REPO = "https://github.com/galactic-ai-team/playground-lm.git"
BRANCH = os.environ.get("PLM_GIT_BRANCH", "feat/stage-g-scratchpad")
COMMIT = os.environ.get("PLM_GIT_COMMIT", "").strip()
WORKING = Path("/kaggle/working")
INPUT = Path("/kaggle/input")
REPO_DIR = WORKING / "playground-lm"
RUNS_DIR = WORKING / "runs"

GROW_ENV = {
    "PLM_STAGE": "C",
    "PLM_N_LAYERS": "24", "PLM_N_HEADS": "16", "PLM_D_MODEL": "1024",
    "PLM_RUN_ID": "stage-g-grow355",
    "PLM_MAX_STEPS": os.environ.get("PLM_MAX_STEPS", "46360"),
    "PLM_SCHEDULE_START": str(SEED_STEP),
    "PLM_LR": os.environ.get("PLM_LR", "0.0003"),
    "PLM_WARMUP": os.environ.get("PLM_WARMUP", "500"),
    "PLM_BATCH_SIZE": "3", "PLM_EFFECTIVE_BATCH": "96",
    "PLM_MAX_SESSION_S": os.environ.get("PLM_MAX_SESSION_S", "37800"),
    "PLM_CKPT_INTERVAL_S": os.environ.get("PLM_CKPT_INTERVAL_S", "1800"),
    "PLM_WEIGHTS_ONLY_RESUME": "1",
    "PLM_UPLOAD_EVERY": "1000000",
    "PLM_RELEASE_TAG": "stage-g-grow355-local-only",
}


def log(msg: str) -> None:
    print(f"[grow355] {msg}", flush=True)


def clone() -> None:
    if not REPO_DIR.exists():
        subprocess.run(["git", "clone", "--branch", BRANCH, GIT_REPO, str(REPO_DIR)], check=True)
    if COMMIT:
        subprocess.run(["git", "-C", str(REPO_DIR), "fetch", "origin", COMMIT], check=False)
        subprocess.run(["git", "-C", str(REPO_DIR), "checkout", "--force", COMMIT], check=True)
    log("repo at " + subprocess.run(["git", "-C", str(REPO_DIR), "rev-parse", "HEAD"],
                                    capture_output=True, text=True).stdout.strip())


def find_seed_runs() -> Path | None:
    for root in (INPUT / "datasets" / "galacticaiteam" / SEED_SLUG, INPUT / SEED_SLUG):
        if root.exists():
            for m in root.rglob("manifest.json"):
                if (m.parent / "checkpoints" / f"step_{SEED_STEP:08d}.safetensors").is_file():
                    return m.parent
    return None


def install_seed() -> bool:
    src = find_seed_runs()
    if src is None:
        log(f"FATAL: grown seed not found under {SEED_SLUG}")
        return False
    if RUNS_DIR.exists():
        shutil.rmtree(RUNS_DIR)
    shutil.copytree(src, RUNS_DIR)
    log(f"seeded runs/ from {src}")
    return True


def load_stage_c():
    path = REPO_DIR / "kaggle" / "stage_c_train" / "stage_c_train.py"
    spec = importlib.util.spec_from_file_location("stage_c_train", path)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


def main() -> int:
    clone()
    sc = load_stage_c()
    sc.STAGE_C_TRAIN_SLUG = SELF_SLUG  # self-reference = previous session of this kernel
    train_path, _ = sc.concat_session_bins()
    digest = sc.verify_train_bin(train_path)
    if not sc.copy_previous_train_runs():
        if not install_seed():
            return 1
    sc.write_train_fingerprint(digest)
    env = os.environ.copy()
    env.update(sc.ENV_OVERRIDES)
    env.update(GROW_ENV)
    env.pop("GH_TOKEN", None)
    for k in sorted(GROW_ENV):
        log(f"{k}={env[k]}")
    nproc = sc._count_gpus()
    if os.environ.get("PLM_DRY_RUN") == "1":
        env["CUDA_VISIBLE_DEVICES"] = ""
        env["PLM_MAX_STEPS"] = str(SEED_STEP + 1)
        env["PLM_EFFECTIVE_BATCH"] = "3"  # one micro-batch on CPU
        nproc = 1
        log("DRY RUN: 1 CPU update")
    cmd = [sys.executable, "-m", "torch.distributed.run", "--standalone",
           f"--nproc_per_node={nproc}", str(REPO_DIR / "notebooks" / "playground_lm_train.py")]
    log(f"launching torchrun on {nproc} proc(s)")
    rc = subprocess.run(cmd, env=env, cwd=str(REPO_DIR)).returncode
    log(f"torchrun exited {rc}")
    for p in (sc.CONCAT_DIR / "train.bin", sc.CONCAT_DIR / "val.bin"):  # /tmp scratch only
        p.unlink(missing_ok=True)
    return rc


if __name__ == "__main__":
    sys.exit(main())
