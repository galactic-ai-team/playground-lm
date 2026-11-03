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
# Two metadata files drive two phases:
#   kernel-metadata.json         — first run, no self-reference (kernels can't
#                                  self-reference before their first version)
#   kernel-metadata-resume.json  — subsequent runs, includes self-reference so
#                                  the previous session's output mounts as a source
#
# Both use id galacticaiteam/plm-stage-b-launch. The title "PLM Stage B Launch"
# slugifies to that id.
#
# If GH_TOKEN isn't set as a Kaggle secret, checkpoints stay in the kernel's
# /kaggle/working output (which is what the next session's kernel_sources mount
# reads). The release upload is a convenience, not a requirement for resume.

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

# Data prep kernel mounts under /kaggle/input/plm-stageb-data-prep/. Its
# output has plm-data/ with train.bin, val.bin, tokenizer/, stats.
DATA_DIR = INPUT / "plm-stageb-data-prep" / "plm-data"

# Previous session's output, if resuming. Matches the self-reference slug in
# kernel-metadata-resume.json. Missing on the first run; copy_previous_runs()
# handles that gracefully.
PREVIOUS_RUNS = INPUT / "plm-stage-b-launch" / "runs"

# --- Config ---------------------------------------------------------------

GIT_REPO = "https://github.com/galactic-ai-team/playground-lm.git"
GIT_BRANCH = "main"

# Stage B settings.
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
        log(f"Kaggle Secrets unavailable or secret not set ({e})")
    token = os.environ.get("GH_TOKEN")
    if token:
        log("GH_TOKEN loaded from env")
        return token
    return None


def copy_previous_runs() -> bool:
    """Copy previous session's runs/ into RUNS_DIR. Returns True if resumed.

    On the first run there's no previous session, so PREVIOUS_RUNS doesn't
    exist and this returns False. Every subsequent run finds the prior
    session's output mounted at PREVIOUS_RUNS and copies it forward so the
    notebook's CheckpointManager.resume() picks up the latest checkpoint.
    """
    if not PREVIOUS_RUNS.exists():
        log(f"no previous runs at {PREVIOUS_RUNS}; starting fresh")
        return False

    # Sanity: manifest must exist for resume to mean anything.
    if not (PREVIOUS_RUNS / "manifest.json").exists():
        log("previous runs dir exists but has no manifest.json; starting fresh")
        return False

    log(f"copying previous runs from {PREVIOUS_RUNS} → {RUNS_DIR}")
    RUNS_DIR.parent.mkdir(parents=True, exist_ok=True)
    if RUNS_DIR.exists():
        shutil.rmtree(RUNS_DIR)
    shutil.copytree(PREVIOUS_RUNS, RUNS_DIR)
    n_files = sum(1 for _ in RUNS_DIR.rglob("*") if _.is_file())
    log(f"copied {n_files} files from previous session")
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


def _count_gpus() -> int:
    """Detect available GPUs. Falls back to 1 if detection fails."""
    try:
        import torch
        n = torch.cuda.device_count()
        return max(1, n)
    except Exception:
        return 1


# --- Main -----------------------------------------------------------------

def main() -> int:
    log("starting Stage B launch")

    # 1. Resolve data.
    data_dir = find_data_dir()
    ENV_OVERRIDES["PLM_DATA_DIR"] = str(data_dir)
    log(f"data dir: {data_dir}")

    # 2. Resume: copy previous session's runs if available.
    copy_previous_runs()

    # 3. Clone repo.
    clone_repo()

    # 4. GH_TOKEN for checkpoint upload (optional).
    token = load_gh_token()
    if token:
        ENV_OVERRIDES["GH_TOKEN"] = token
    else:
        log("no GH_TOKEN — checkpoints will stay in /kaggle/working/runs "
            "and be picked up by the next session's kernel_sources mount")

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


if __name__ == "__main__":
    sys.exit(main())
