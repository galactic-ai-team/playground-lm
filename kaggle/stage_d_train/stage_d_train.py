# kaggle/stage_d_train/stage_d_train.py
# Stage D: fine-tune PlaygroundLM on the Playground corpus (Claude, sketch).
#
# Same notebook and loop as Stages B/C; everything Stage-D-specific comes
# from the "D" preset in src/playground_lm/train/stage_config.py (peak LR
# 6e-5, schedule restarting at 31101, 24-sequence batch, eval every 25).
#
# Flow:
#   1. Stage the fine-tune data under /tmp/plm_stage_d: train.bin from
#      PLM_FINETUNE_BIN, val.bin from PLM_FINETUNE_VAL_BIN (default: the
#      val.bin next to it). Both are flat uint16 GPT-2 token files.
#   2. Resume state, in order of preference:
#        a. a previous Stage D session's runs/ (self-reference mount, if
#           plm-stage-d-train is added to kernel_sources after the first run);
#        b. Stage C's final checkpoint (step 31101: weights + _state.pt) from
#           the plm-stage-c-train mount, with a manifest so
#           CheckpointManager.resume() continues from it.
#   3. Clone main, set max_steps from the corpus size (finetune_max_steps,
#      ~3 epochs by default), launch the notebook under torchrun.
#
# Kaggle script kernels don't take custom env vars at push time, so the
# PLM_FINETUNE_* defaults below are where the real mount path goes once
# Nova's shard exists.

import json
import os
import shutil
import subprocess
import sys
import time
from pathlib import Path


WORKING = Path("/kaggle/working")
INPUT = Path("/kaggle/input")
REPO_DIR = WORKING / "playground-lm"
RUNS_DIR = WORKING / "runs"
DATA_DIR = Path("/tmp/plm_stage_d")
GIT_REPO = "https://github.com/galactic-ai-team/playground-lm.git"

STAGE_C_SLUG = "plm-stage-c-train"
STAGE_D_SLUG = "plm-stage-d-train"
STAGE_C_FINAL_STEP = 31101

FINETUNE_BIN = Path(os.environ.get(
    "PLM_FINETUNE_BIN", "/kaggle/input/plm-stage-d-data/finetune/train.bin"))  # TODO: Nova's mount
FINETUNE_VAL_BIN = Path(os.environ.get("PLM_FINETUNE_VAL_BIN", str(FINETUNE_BIN.with_name("val.bin"))))
FINETUNE_EPOCHS = float(os.environ.get("PLM_FINETUNE_EPOCHS", "3"))


def log(msg: str) -> None:
    print(f"[stage-d] {msg}", flush=True)


def find_mount(slug: str) -> Path | None:
    """Root of a kernel_sources mount (Kaggle uses either layout)."""
    for root in (INPUT / "notebooks" / "galacticaiteam" / slug, INPUT / slug):
        if root.is_dir():
            return root
    return None


def token_count(path: Path, context_len: int = 1024) -> int:
    size = path.stat().st_size
    if size % 2:

        raise ValueError(f"{path}: {size} bytes is not a uint16 token file")
    if size // 2 <= context_len:
        raise ValueError(f"{path}: only {size // 2} tokens")
    return size // 2


def stage_data() -> int:
    """Copy train/val into DATA_DIR (the notebook reads DATA_DIR/{train,val}.bin)."""
    for src in (FINETUNE_BIN, FINETUNE_VAL_BIN):
        if not src.is_file():
            raise FileNotFoundError(f"fine-tune data not found: {src} (set PLM_FINETUNE_BIN / _VAL_BIN)")
    DATA_DIR.mkdir(parents=True, exist_ok=True)
    shutil.copy(FINETUNE_BIN, DATA_DIR / "train.bin")
    shutil.copy(FINETUNE_VAL_BIN, DATA_DIR / "val.bin")
    n_train = token_count(DATA_DIR / "train.bin")
    log(f"train.bin {n_train:,} tokens; val.bin {token_count(DATA_DIR / 'val.bin'):,} tokens")
    return n_train


def resume_previous_stage_d() -> bool:
    root = find_mount(STAGE_D_SLUG)
    prev = root / "runs" if root else None
    if not prev or not (prev / "manifest.json").exists():
        return False
    if RUNS_DIR.exists():
        shutil.rmtree(RUNS_DIR)
    shutil.copytree(prev, RUNS_DIR)
    log(f"resuming previous Stage D session from {prev}")
    return True


def seed_from_stage_c() -> None:

    """Copy Stage C's step-31101 checkpoint (weights, optimizer state and
    model-config sidecar) into RUNS_DIR with a manifest pointing at it."""
    root = find_mount(STAGE_C_SLUG)
    if root is None:
        raise FileNotFoundError(f"kernel source {STAGE_C_SLUG} is not mounted")
    stem = f"step_{STAGE_C_FINAL_STEP:08d}"
    src = root / "runs" / "checkpoints"
    needed = [src / f"{stem}.safetensors", src / f"{stem}_state.pt"]
    missing = [p for p in needed if not p.is_file()]
    if missing:
        raise FileNotFoundError(f"Stage C final checkpoint incomplete, missing {missing}")
    dst = RUNS_DIR / "checkpoints"
    dst.mkdir(parents=True, exist_ok=True)
    for p in needed + [src / f"{stem}_model_config.json"]:
        if p.is_file():
            shutil.copy(p, dst / p.name)

    # Carry Stage C's token count, wall time and config (the model section is
    # what evals read) into a CheckpointManager-shaped manifest.
    c_manifest = json.loads((root / "runs" / "manifest.json").read_text())
    entry = next((e for e in c_manifest["checkpoints"] if e["step"] == STAGE_C_FINAL_STEP), None)
    if entry is None:
        raise ValueError(f"Stage C manifest has no step {STAGE_C_FINAL_STEP}; has Stage C finished?")
    manifest = {
        "run_id": "stage-c-final",
        "config": c_manifest.get("config", {}),
        "checkpoints": [dict(entry, path=f"checkpoints/{stem}.safetensors",
                             created_at=time.time(), is_milestone=True)],
        "latest_step": STAGE_C_FINAL_STEP,
        "latest_tokens_seen": entry["tokens_seen"],
        "latest_wall_time_s": entry["wall_time_s"],
    }

    (RUNS_DIR / "manifest.json").write_text(json.dumps(manifest, indent=2))
    log(f"seeded from Stage C step {STAGE_C_FINAL_STEP} ({entry['tokens_seen']:,} tokens)")


def gh_token() -> str | None:
    try:
        from kaggle_secrets import UserSecretsClient
        return UserSecretsClient().get_secret("GH_TOKEN") or None
    except Exception:
        return os.environ.get("GH_TOKEN")


def main() -> int:
    n_train = stage_data()
    if not resume_previous_stage_d():
        seed_from_stage_c()

    if REPO_DIR.exists():
        subprocess.run(["git", "-C", str(REPO_DIR), "pull", "--ff-only"], check=True)
    else:
        subprocess.run(["git", "clone", "--depth", "1", GIT_REPO, str(REPO_DIR)], check=True)
    sys.path.insert(0, str(REPO_DIR / "src"))
    from playground_lm.train.stage_config import finetune_max_steps

    max_steps = finetune_max_steps(n_train, epochs=FINETUNE_EPOCHS)
    span = max_steps - STAGE_C_FINAL_STEP
    log(f"max_steps {max_steps} ({span} updates, ~{FINETUNE_EPOCHS:g} epochs)")

    env = dict(os.environ)
    env.update({
        "PLM_STAGE": "D",                      # LR 6e-5, schedule from 31101, batch 24, eval 25
        "PLM_MAX_STEPS": str(max_steps),

        "PLM_WARMUP": str(max(1, min(20, span // 4))),  # tiny corpus: keep warmup inside the run
        "PLM_BATCH_SIZE": "6",                 # T4: 24 = 6 x 2 GPUs x 2 accumulation
        "PLM_DATA_DIR": str(DATA_DIR),
        "PLM_RUN_DIR": str(RUNS_DIR),
        "PLM_RUN_ID": "stage-d",
        "PLM_REPO": "galactic-ai-team/playground-lm",
        "PLM_RELEASE_TAG": "stage-d-checkpoints",
        "PLM_UPLOAD_EVERY": "1",
        "PLM_MAX_SESSION_S": str(11 * 3600),
    })
    token = gh_token()
    if token:
        env["GH_TOKEN"] = token

    try:
        import torch
        nproc = max(1, torch.cuda.device_count())
    except Exception:
        nproc = 1
    cmd = [sys.executable, "-m", "torch.distributed.run", "--standalone",
           f"--nproc_per_node={nproc}", str(REPO_DIR / "notebooks" / "playground_lm_train.py")]
    log(f"launching torchrun on {nproc} GPU(s)")
    return subprocess.run(cmd, env=env, cwd=str(REPO_DIR)).returncode


if __name__ == "__main__":
    sys.exit(main())
