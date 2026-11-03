# stage_c_train.py
# Kaggle GPU script kernel: Stage C PlaygroundLM training.
#
# Assumes kernel_sources contains:
#   - galacticaiteam/plm-stage-c-data-prep-s0 through -s7
#     (each producing stage-c/session-<N>/final/train.bin; s0 also val.bin)
#   - galacticaiteam/plm-stage-b-launch
#     (its /runs/checkpoints/ has step_00002035.safetensors + _state.pt,
#      for full optimizer/RNG continuity into Stage C)
#   - galacticaiteam/plm-stage-c-train (self-reference, resume metadata only)
#
# Flow:
#   1. Concatenate the 8 session train.bin files into /kaggle/working/train.bin.
#   2. Copy session 0's val.bin to /kaggle/working/val.bin.
#   3. Copy Stage B's step_00002035 checkpoint (weights + optimizer state) into
#      /kaggle/working/runs/ so CheckpointManager.resume() picks it up with
#      full continuity — no GH_TOKEN, no synthetic state.
#   4. Clone main, launch notebooks/playground_lm_train.py via torchrun with
#      PLM_STAGE=C and PLM_MAX_STEPS (default 29000, one pass over 2.857B).
#
# Free-tools only: Kaggle 2x T4, no paid services.

import json
import os
import shutil
import subprocess
import sys
import time
from pathlib import Path

# --- Paths ----------------------------------------------------------------

WORKING = Path("/kaggle/working")
INPUT = Path("/kaggle/input")
REPO_DIR = WORKING / "playground-lm"
RUNS_DIR = WORKING / "runs"
CONCAT_DIR = WORKING

GIT_REPO = "https://github.com/galactic-ai-team/playground-lm.git"
GIT_BRANCH = "main"

# Stage B checkpoint mount (kernel_sources).
STAGE_B_INPUT = INPUT / "plm-stage-b-launch"
STAGE_B_RUNS = STAGE_B_INPUT / "runs"
STAGE_B_RESUME_STEP = 2035

# Previous version of THIS kernel (self-reference, resume metadata only).
PREV_TRAIN_RUNS = INPUT / "plm-stage-c-train" / "runs"

# --- Config ---------------------------------------------------------------

N_SESSIONS = 8

# 2.857B tokens / 98,304 tokens per step ≈ 29,065 steps for one full pass.
# Override via PLM_STAGE_C_MAX_STEPS env at push time (e.g. 500 for smoke).
STAGE_C_MAX_STEPS = int(os.environ.get("PLM_STAGE_C_MAX_STEPS", "29000"))

ENV_OVERRIDES = {
    "PLM_STAGE": "C",
    "PLM_MAX_STEPS": str(STAGE_C_MAX_STEPS),
    "PLM_BATCH_SIZE": "6",              # T4-friendly; grad_accum → 16
    "PLM_MAX_SESSION_S": str(11 * 3600),  # under Kaggle's 12h GPU kill
    "PLM_DATA_DIR": str(CONCAT_DIR),      # holds train.bin / val.bin
    "PLM_RUN_DIR": str(RUNS_DIR),
    "PLM_RUN_ID": "stage-c",
    "PLM_REPO": "galactic-ai-team/playground-lm",
    "PLM_RELEASE_TAG": "stage-c-checkpoints",
    "PLM_UPLOAD_EVERY": "1",
}

# --- Helpers --------------------------------------------------------------

def log(msg: str) -> None:
    print(f"[stage-c-train] {msg}", flush=True)


def load_gh_token() -> str | None:
    """Optional GH_TOKEN for uploading Stage C checkpoints to the release.
    Training itself doesn't need it — Stage B resume reads from kernel_sources."""
    try:
        from kaggle_secrets import UserSecretsClient
        token = UserSecretsClient().get_secret("GH_TOKEN")
        if token:
            log("GH_TOKEN loaded from Kaggle Secrets")
            return token
    except Exception as e:
        log(f"Kaggle Secrets unavailable or secret not set ({e})")
    token = os.environ.get("GH_TOKEN")
    if token:
        log("GH_TOKEN loaded from env")
    return token


def concat_session_bins() -> tuple[Path, Path]:
    """Concatenate all N sessions' train.bin into one file. Copy session 0's
    val.bin. Returns (train_path, val_path)."""
    out_train = CONCAT_DIR / "train.bin"
    out_val = CONCAT_DIR / "val.bin"

    if out_train.exists():
        out_train.unlink()
    if out_val.exists():
        out_val.unlink()

    total_tokens = 0
    with open(out_train, "wb") as f_out:
        for n in range(N_SESSIONS):
            src = (
                INPUT
                / f"plm-stage-c-data-prep-s{n}"
                / "stage-c"
                / f"session-{n}"
                / "final"
                / "train.bin"
            )
            if not src.exists():
                raise FileNotFoundError(
                    f"session {n} train.bin not found at {src}"
                )
            n_tok = src.stat().st_size // 2
            log(f"concat s{n}: {n_tok:,} tokens from {src}")
            with open(src, "rb") as f_in:
                while True:
                    chunk = f_in.read(8 * 1024 * 1024)
                    if not chunk:
                        break
                    f_out.write(chunk)
            total_tokens += n_tok

    log(f"train.bin: {total_tokens:,} tokens at {out_train}")

    val_src = (
        INPUT
        / "plm-stage-c-data-prep-s0"
        / "stage-c"
        / "session-0"
        / "final"
        / "val.bin"
    )
    if not val_src.exists():
        raise FileNotFoundError(f"session 0 val.bin not found at {val_src}")
    shutil.copy(val_src, out_val)
    log(f"val.bin: {out_val.stat().st_size // 2:,} tokens copied from s0")

    return out_train, out_val


def copy_previous_train_runs() -> bool:
    """Copy previous Stage C session's runs/ into RUNS_DIR. Enables torchrun
    resume across Kaggle GPU sessions."""
    if not PREV_TRAIN_RUNS.exists():
        log(f"no previous Stage C runs at {PREV_TRAIN_RUNS}")
        return False
    if not (PREV_TRAIN_RUNS / "manifest.json").exists():
        log("previous Stage C runs dir exists but no manifest.json")
        return False

    log(f"copying previous Stage C runs from {PREV_TRAIN_RUNS} → {RUNS_DIR}")
    if RUNS_DIR.exists():
        shutil.rmtree(RUNS_DIR)
    RUNS_DIR.parent.mkdir(parents=True, exist_ok=True)
    shutil.copytree(PREV_TRAIN_RUNS, RUNS_DIR)
    n_files = sum(1 for _ in RUNS_DIR.rglob("*") if _.is_file())
    log(f"copied {n_files} files from previous Stage C session")
    return True


def stage_b_checkpoint_paths() -> tuple[Path, Path]:
    """Return (weights_path, state_path) for the Stage B final checkpoint
    from the kernel_sources mount."""
    weights = STAGE_B_RUNS / "checkpoints" / f"step_{STAGE_B_RESUME_STEP:08d}.safetensors"
    state = STAGE_B_RUNS / "checkpoints" / f"step_{STAGE_B_RESUME_STEP:08d}_state.pt"
    return weights, state


def copy_stage_b_checkpoint() -> bool:
    """Copy Stage B's final checkpoint (weights + optimizer state) into
    RUNS_DIR so CheckpointManager.resume() picks it up with full continuity.
    Reads from the plm-stage-b-launch kernel mount — no GH_TOKEN needed.
    """
    if (RUNS_DIR / "manifest.json").exists():
        log("Stage C resume state already present; skipping Stage B copy")
        return True

    weights, state = stage_b_checkpoint_paths()
    if not weights.exists():
        log(f"FATAL: Stage B weights not found at {weights}")
        log(f"  available under {STAGE_B_RUNS / 'checkpoints'}:")
        ckpt_dir = STAGE_B_RUNS / "checkpoints"
        if ckpt_dir.exists():
            for p in sorted(ckpt_dir.iterdir()):
                log(f"    {p.name}")
        return False
    if not state.exists():
        log(f"FATAL: Stage B optimizer state not found at {state}")
        return False

    ckpt_dir = RUNS_DIR / "checkpoints"
    ckpt_dir.mkdir(parents=True, exist_ok=True)
    shutil.copy(weights, ckpt_dir / weights.name)
    shutil.copy(state, ckpt_dir / state.name)
    log(f"copied {weights.name} ({weights.stat().st_size // 1024 // 1024} MB)")
    log(f"copied {state.name} ({state.stat().st_size // 1024 // 1024} MB)")

    # Build a CheckpointManager-shaped manifest pointing at the copied files.
    # If Stage B had a manifest in its output, use its token/wall_time values;
    # otherwise fall back to known figures (200M tokens, milestone).
    tokens_seen = 200_000_000
    wall_time_s = 0.0
    stage_b_manifest = STAGE_B_RUNS / "manifest.json"
    if stage_b_manifest.exists():
        try:
            with open(stage_b_manifest) as f:
                bm = json.load(f)
            for entry in bm.get("checkpoints", []):
                if entry.get("step") == STAGE_B_RESUME_STEP:
                    tokens_seen = entry.get("tokens_seen", tokens_seen)
                    wall_time_s = entry.get("wall_time_s", wall_time_s)
                    break
        except Exception as e:
            log(f"WARN: could not read Stage B manifest ({e}); using defaults")

    synthetic = {
        "run_id": "stage-b-final",
        "config": {},
        "checkpoints": [{
            "step": STAGE_B_RESUME_STEP,
            "path": f"checkpoints/{weights.name}",
            "tokens_seen": tokens_seen,
            "wall_time_s": wall_time_s,
            "created_at": time.time(),
            "is_milestone": True,
        }],
        "latest_step": STAGE_B_RESUME_STEP,
        "latest_tokens_seen": tokens_seen,
        "latest_wall_time_s": wall_time_s,
    }
    with open(RUNS_DIR / "manifest.json", "w") as f:
        json.dump(synthetic, f, indent=2)

    log(
        f"Stage B checkpoint staged at {RUNS_DIR}: step {STAGE_B_RESUME_STEP}, "
        f"{tokens_seen:,} tokens, full optimizer/RNG continuity"
    )
    return True


def clone_repo() -> None:
    if REPO_DIR.exists():
        log(f"{REPO_DIR} exists; pulling")
        subprocess.run(["git", "-C", str(REPO_DIR), "pull", "--ff-only"], check=True)
        return
    log(f"cloning {GIT_REPO} ({GIT_BRANCH})")
    subprocess.run(
        ["git", "clone", "--depth", "1", "--branch", GIT_BRANCH, GIT_REPO, str(REPO_DIR)],
        check=True,
    )


def _count_gpus() -> int:
    try:
        import torch
        return max(1, torch.cuda.device_count())
    except Exception:
        return 1


# --- Main -----------------------------------------------------------------

def main() -> int:
    log(f"starting Stage C training launch (max_steps={STAGE_C_MAX_STEPS})")

    # 1. Concatenate the 8 sessions.
    concat_session_bins()

    # 2. Resume: prefer a previous Stage C session, else Stage B.
    resumed_c = copy_previous_train_runs()
    if not resumed_c:
        if not copy_stage_b_checkpoint():
            log("FATAL: could not prepare resume state from Stage B")
            return 1

    # 3. GH_TOKEN for uploading Stage C checkpoints (optional).
    token = load_gh_token()
    if token:
        ENV_OVERRIDES["GH_TOKEN"] = token
    else:
        log("no GH_TOKEN — Stage C checkpoints stay in /kaggle/working/runs and "
            "are carried forward via the plm-stage-c-train self-reference mount")

    # 4. Clone repo.
    clone_repo()

    # 5. Launch torchrun.
    env = os.environ.copy()
    env.update(ENV_OVERRIDES)

    notebook = REPO_DIR / "notebooks" / "playground_lm_train.py"
    if not notebook.exists():
        log(f"FATAL: {notebook} not found")
        return 1

    nproc = _count_gpus()
    log(f"launching torchrun --nproc_per_node={nproc}")
    cmd = [
        sys.executable, "-m", "torch.distributed.run",
        "--standalone",
        f"--nproc_per_node={nproc}",
        str(notebook),
    ]
    result = subprocess.run(cmd, env=env, cwd=str(REPO_DIR))
    log(f"torchrun exited with code {result.returncode}")
    return result.returncode


if __name__ == "__main__":
    sys.exit(main())
