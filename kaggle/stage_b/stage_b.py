# stage_b.py
# Kaggle script kernel: Stage B PlaygroundLM training.
#
# Launches notebooks/playground_lm_train.py via torchrun with PLM_STAGE=B,
# on all available GPUs. Resumes across sessions by:
#   1. Mounting the previous version of this same kernel via kernel_sources
#      (gives us its /kaggle/working output, which contains runs/).
#   2. Copying that runs/ into PLM_RUN_DIR before torchrun starts.
#   3. The notebook's CheckpointManager.resume() then picks up the latest
#      checkpoint from the copied directory.
#
# Free-tools only: Kaggle T4 x2 or P100, no paid services.
#
# Assumes kernel_sources contains:
#   - galacticaiteam/plm-stageb-data-prep  (train.bin, val.bin, stats under /kaggle/input)
#   - galacticaiteam/plm-stageb-launch     (previous session's output, if resuming)
#
# First run: remove the second kernel_source — there's nothing to resume from.
# Subsequent runs: keep it; it's the previous version's output.

import os
import shutil
import subprocess
import sys
from pathlib import Path

# --- Paths ----------------------------------------------------------------

WORKING = Path("/kaggle/working")
INPUT = Path("/kaggle/input")
RUNS_DIR = WORKING / "runs"
REPO_DIR = WORKING / "playground-lm"

# Data prep kernel mounts under /kaggle/input/<slug>/. Its output has
# plm-data/ with train.bin, val.bin, tokenizer/, stats.
DATA_DIR = INPUT / "plm-stageb-data-prep" / "plm-data"

# Previous session's output, if resuming. Same slug as this kernel.
PREVIOUS_RUNS = INPUT / "plm-stageb-launch" / "runs"

# --- Config ---------------------------------------------------------------

GIT_REPO = "https://github.com/galactic-ai-team/playground-lm.git"
GIT_BRANCH = "main"

# Stage B settings per Lumi's request.
ENV_OVERRIDES = {
    "PLM_STAGE": "B",
    "PLM_BATCH_SIZE": "6",
    "PLM_MAX_SESSION_S": str(11 * 3600),   # 11h, under Kaggle's 12h GPU kill
    "PLM_DATA_DIR": str(DATA_DIR),
    "PLM_RUN_DIR": str(RUNS_DIR),
    "PLM_RUN_ID": "stage-b",
    "PLM_REPO": "galactic-ai-team/playground-lm",
    "PLM_RELEASE_TAG": "stage-b-checkpoints",
    "PLM_UPLOAD_EVERY": "1",               # upload every save on Stage B
}

# --- Helpers --------------------------------------------------------------

def log(msg: str) -> None:
    print(f"[stage-b] {msg}", flush=True)


def load_gh_token() -> str | None:
    """Read GH_TOKEN from Kaggle Secrets. Falls back to env var for local runs."""
    try:
        from kaggle_secrets import UserSecretsClient
        token = UserSecretsClient().get_secret("GH_TOKEN")
        if token:
            log("GH_TOKEN loaded from Kaggle Secrets")
            return token
    except Exception as e:
        log(f"Kaggle Secrets unavailable ({e}); checking env var")
    token = os.environ.get("GH_TOKEN")
    if token:
        log("GH_TOKEN loaded from env")
    return token


def copy_previous_runs() -> bool:
    """Copy previous session's runs/ into RUNS_DIR. Returns True if resumed."""
    if not PREVIOUS_RUNS.exists():
        log(f"no previous runs at {PREVIOUS_RUNS}; starting fresh")
        return False

    # Sanity: manifest must exist for resume to mean anything.
    if not (PREVIOUS_RUNS / "manifest.json").exists():
        log(f"previous runs dir exists but has no manifest.json; starting fresh")
        return False

    log(f"copying previous runs from {PREVIOUS_RUNS} → {RUNS_DIR}")
    RUNS_DIR.parent.mkdir(parents=True, exist_ok=True)
    if RUNS_DIR.exists():
        shutil.rmtree(RUNS_DIR)
    shutil.copytree(PREVIOUS_RUNS, RUNS_DIR)
    log(f"copied {sum(1 for _ in RUNS_DIR.rglob('*') if _.is_file())} files")
    return True


def clone_repo() -> None:
    """Clone playground-lm main into REPO_DIR."""
    if REPO_DIR.exists():
        log(f"{REPO_DIR} exists; pulling latest")
        subprocess.run(["git", "-C", str(REPO_DIR), "pull", "--ff-only"], check=True)
        return
    log(f"cloning {GIT_REPO} ({GIT_BRANCH})")
    subprocess.run(
        ["git", "clone", "--depth", "1", "--branch", GIT_BRANCH, GIT_REPO, str(REPO_DIR)],
        check=True,
    )


def find_data_dir() -> Path:
    """Resolve DATA_DIR, handling dataset_sources vs kernel_sources layouts."""
    if DATA_DIR.exists() and (DATA_DIR / "train.bin").exists():
        return DATA_DIR

    # kernel_sources mounts directly under /kaggle/input/<slug>/, but some
    # versions nest an extra output/ or working/ layer. Walk to find train.bin.
    for candidate in INPUT.rglob("train.bin"):
        if "plm-stageb-data-prep" in str(candidate):
            resolved = candidate.parent
            log(f"resolved data dir by walking: {resolved}")
            return resolved

    raise FileNotFoundError(
        f"could not find train.bin under {INPUT} for plm-stageb-data-prep"
    )


# --- Main -----------------------------------------------------------------

def main() -> int:
    log("starting Stage B launch")

    # 1. Resolve data.
    data_dir = find_data_dir()
    ENV_OVERRIDES["PLM_DATA_DIR"] = str(data_dir)
    log(f"data dir: {data_dir}")

    # 2. Resume: copy previous session's runs.
    copy_previous_runs()

    # 3. Clone repo.
    clone_repo()

    # 4. GH_TOKEN for checkpoint upload.
    token = load_gh_token()
    if token:
        ENV_OVERRIDES["GH_TOKEN"] = token
    else:
        log("WARN: no GH_TOKEN — checkpoints will not be uploaded to the release")

    # 5. Build env for torchrun subprocess.
    env = os.environ.copy()
    env.update(ENV_OVERRIDES)

    # 6. Launch torchrun on all GPUs.
    notebook = REPO_DIR / "notebooks" / "playground_lm_train.py"
    if not notebook.exists():
        log(f"FATAL: {notebook} not found after clone")
        return 1

    nproc = _count_gpus()
    log(f"launching torchrun with nproc_per_node={nproc}")
    cmd = [
        sys.executable, "-m", "torch.distributed.run",
        "--standalone",
        f"--nproc_per_node={nproc}",
        str(notebook),
    ]

    result = subprocess.run(cmd, env=env, cwd=str(REPO_DIR))
    log(f"torchrun exited with code {result.returncode}")
    return result.returncode


def _count_gpus() -> int:
    """Detect available GPUs. Falls back to 1 if detection fails."""
    try:
        import torch
        n = torch.cuda.device_count()
        return max(1, n)
    except Exception:
        return 1


if __name__ == "__main__":
    sys.exit(main())
