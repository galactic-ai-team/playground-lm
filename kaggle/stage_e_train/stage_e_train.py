# kaggle/stage_e_train/stage_e_train.py
# Stage E: instruction/SFT continuation from Stage C step 31101.
#
# Contract (from data/stage_e/SCHEMA.md):
#   PLM_DATA_DIR holds {train,val}.bin (uint16 tokens) and
#   {train,val}.masks.bin (uint8, 1=supervised). Lengths must match.
#
# Resume safety: never delete an existing RUNS_DIR before a verified copy.
# Copy to RUNS_DIR.tmp, verify manifest, then os.replace the staging dir
# into place. If any step fails, RUNS_DIR is untouched.
#
# Stage E preset comes from stage_config.STAGE_PRESETS["E"] (lr 3e-5,
# schedule_start 31101, effective_batch 24, eval_interval 25).
#
# Free Kaggle only. PLM_DRY_RUN=1 runs a single CPU step with no GPU quota.
# No real GPU training in this file — it is launch-ready but not launched.

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
DATA_DIR = Path("/tmp/plm_stage_e")
GIT_REPO = "https://github.com/galactic-ai-team/playground-lm.git"
GIT_BRANCH = "dee/stage-e-kaggle-launcher"

STAGE_C_SLUG = "plm-stage-c-train"
STAGE_E_SLUG = "plm-stage-e-train"
STAGE_C_FINAL_STEP = 31101

FINETUNE_EPOCHS = float(os.environ.get("PLM_FINETUNE_EPOCHS", "2"))
DRY_RUN = os.environ.get("PLM_DRY_RUN", "0") == "1"
MIN_DISK_GB = float(os.environ.get("PLM_MIN_DISK_GB", "8.0"))
QUOTA_GUARD = os.environ.get("PLM_QUOTA_GUARD", "0") == "1"
QUOTA_LIMIT_H = float(os.environ.get("PLM_QUOTA_LIMIT_H", "30"))
EST_SESSION_H = float(os.environ.get("PLM_EST_SESSION_H", "1.0"))


def log(msg: str) -> None:
    print(f"[stage-e] {msg}", flush=True)



def _archive_existing_old(old: Path) -> None:
    """Preserve a prior .old tree as a durable timestamped backup (Lumi glue).

    Dee's revised launcher rotated runs -> runs.old then deleted .old. Deleting
    non-identical checkpoint trees risks data loss; keep archives instead.
    """
    if not old.exists():
        return
    stamp = time.strftime("%Y%m%dT%H%M%SZ", time.gmtime())
    archived = old.parent / f"{old.name}.bak.{stamp}"
    n = 0
    while archived.exists():
        n += 1
        archived = old.parent / f"{old.name}.bak.{stamp}.{n}"
    os.replace(old, archived)
    log(f"preserved previous backup as {archived}")


def find_mount(slug: str) -> Path | None:
    for root in (INPUT / "notebooks" / "galacticaiteam" / slug, INPUT / slug):
        if root.is_dir():
            return root
    return None


def _resolve_pair(dir_hint: Path) -> tuple[Path, Path, Path, Path]:
    """Return (train.bin, train.masks.bin, val.bin, val.masks.bin) under dir_hint.

    Raises FileNotFoundError if any are missing.
    """
    train_bin = dir_hint / "train.bin"
    train_mask = dir_hint / "train.masks.bin"
    val_bin = dir_hint / "val.bin"
    val_mask = dir_hint / "val.masks.bin"
    missing = [p for p in (train_bin, train_mask, val_bin, val_mask) if not p.is_file()]
    if missing:
        raise FileNotFoundError(f"missing Stage E data files under {dir_hint}: {missing}")
    return train_bin, train_mask, val_bin, val_mask


def resolve_finetune_files() -> tuple[Path, Path, Path, Path]:
    """Locate the paired Stage E artifacts from env or the data mount.

    Accepted layouts, in priority order:
      1. PLM_FINETUNE_BIN / PLM_FINETUNE_VAL_BIN env (both must be set).
      2. /kaggle/input/plm-stage-e-data[/finetune]/
      3. /kaggle/input/datasets/galacticaiteam/plm-stage-e-data[/finetune]/
      4. rglob under any plm-stage-e-data mount that contains train.bin
    """
    env_train = os.environ.get("PLM_FINETUNE_BIN")
    env_val = os.environ.get("PLM_FINETUNE_VAL_BIN")
    if env_train and env_val:
        return _resolve_pair(Path(env_train).parent)

    candidates = [
        INPUT / "plm-stage-e-data" / "finetune",
        INPUT / "datasets" / "galacticaiteam" / "plm-stage-e-data" / "finetune",
        INPUT / "plm-stage-e-data",
        INPUT / "datasets" / "galacticaiteam" / "plm-stage-e-data",
    ]
    for c in candidates:
        if (c / "train.bin").is_file():
            return _resolve_pair(c)

    for root in INPUT.rglob("plm-stage-e-data"):
        if not root.is_dir():
            continue
        for hit in root.rglob("train.bin"):
            try:
                return _resolve_pair(hit.parent)
            except FileNotFoundError:
                continue
    raise FileNotFoundError(
        "Stage E paired data not found (need train.bin, train.masks.bin, "
        "val.bin, val.masks.bin)"
    )


def token_count(path: Path, context_len: int = 1024) -> int:
    size = path.stat().st_size
    if size % 2:
        raise ValueError(f"{path}: {size} bytes is not a uint16 token file")
    n = size // 2
    if n <= context_len:
        raise ValueError(f"{path}: only {n} tokens")
    return n


def stage_data() -> int:
    """Copy the paired train/val bins+masks into DATA_DIR and verify masks
    align with tokens byte-for-byte. Returns train token count."""
    train_bin, train_mask, val_bin, val_mask = resolve_finetune_files()
    DATA_DIR.mkdir(parents=True, exist_ok=True)
    for src in (train_bin, train_mask, val_bin, val_mask):
        shutil.copy(src, DATA_DIR / src.name)
        log(f"staged {src.name} ({src.stat().st_size} bytes)")

    # Mask/token alignment: train.masks.bin must have exactly one uint8 per
    # uint16 token. Same for val.
    train_tok = token_count(DATA_DIR / "train.bin")
    val_tok = token_count(DATA_DIR / "val.bin")
    train_mask_bytes = (DATA_DIR / "train.masks.bin").stat().st_size
    val_mask_bytes = (DATA_DIR / "val.masks.bin").stat().st_size
    if train_mask_bytes != train_tok:
        raise ValueError(
            f"train mask/token mismatch: {train_mask_bytes} masks vs {train_tok} tokens"
        )
    if val_mask_bytes != val_tok:
        raise ValueError(
            f"val mask/token mismatch: {val_mask_bytes} masks vs {val_tok} tokens"
        )
    log(f"train {train_tok:,} tokens / masks aligned; val {val_tok:,} tokens / masks aligned")
    return train_tok


def preflight_disk() -> bool:
    free = shutil.disk_usage(WORKING).free / (1024 ** 3)
    log(f"disk free: {free:.1f} GB (need >= {MIN_DISK_GB:.1f})")
    if free < MIN_DISK_GB:
        log("FATAL: insufficient disk")
        return False
    return True


def preflight_quota() -> bool:
    if not QUOTA_GUARD:
        log("quota guard off (set PLM_QUOTA_GUARD=1 to enforce)")
        return True
    log(f"quota guard: {EST_SESSION_H:.1f} h vs limit {QUOTA_LIMIT_H:.1f} h")
    if EST_SESSION_H > QUOTA_LIMIT_H:
        log("FATAL: estimated session exceeds quota limit")
        return False
    return True


def _atomic_stage_copy(src: Path, dst: Path) -> None:
    """Copy src -> dst atomically. If dst exists, it is preserved on failure.

    Procedure:
      1. Copy src into dst.parent / (dst.name + ".tmp"), removing any stale tmp.
      2. Verify tmp has a manifest.json (caller's contract).
      3. os.replace(dst -> dst.parent / (dst.name + ".old")) if dst exists,
         then os.replace(tmp -> dst), then remove .old.
    """
    tmp = dst.parent / (dst.name + ".tmp")
    old = dst.parent / (dst.name + ".old")
    if tmp.exists():
        shutil.rmtree(tmp)
    shutil.copytree(src, tmp)
    if not (tmp / "manifest.json").is_file():
        shutil.rmtree(tmp)
        raise RuntimeError(f"staged copy at {tmp} has no manifest.json; existing {dst} preserved")
    if dst.exists():
        if old.exists():
            _archive_existing_old(old)
        os.replace(dst, old)
        os.replace(tmp, dst)
        # Keep dst.old as durable backup (Lumi glue; do not rmtree).
    else:
        os.replace(tmp, dst)
    log(f"atomically installed {src} -> {dst}")


def resume_previous_stage_e() -> bool:
    root = find_mount(STAGE_E_SLUG)
    if root is None:
        return False
    prev = root / "runs"
    if not prev.is_dir() or not (prev / "manifest.json").exists():
        log("no previous Stage E runs with manifest; will seed from Stage C")
        return False
    _atomic_stage_copy(prev, RUNS_DIR)
    manifest = json.loads((RUNS_DIR / "manifest.json").read_text())
    log(f"resumed at step {manifest.get('latest_step')}")
    return True


def seed_from_stage_c() -> None:
    """Copy Stage C step 31101 (weights + state + model_config) into a
    freshly-staged RUNS_DIR.tmp, write a manifest, and atomically install."""
    root = find_mount(STAGE_C_SLUG)
    if root is None:
        raise FileNotFoundError(f"kernel source {STAGE_C_SLUG} is not mounted")
    src = root / "runs" / "checkpoints"
    stem = f"step_{STAGE_C_FINAL_STEP:08d}"
    needed = [src / f"{stem}.safetensors", src / f"{stem}_state.pt"]
    missing = [p for p in needed if not p.is_file()]
    if missing:
        raise FileNotFoundError(f"Stage C final checkpoint incomplete, missing {missing}")

    tmp = RUNS_DIR.parent / (RUNS_DIR.name + ".tmp")
    old = RUNS_DIR.parent / (RUNS_DIR.name + ".old")
    if tmp.exists():
        shutil.rmtree(tmp)
    (tmp / "checkpoints").mkdir(parents=True)

    for p in needed:
        shutil.copy(p, tmp / "checkpoints" / p.name)
    model_cfg = src / f"{stem}_model_config.json"
    if model_cfg.is_file():
        shutil.copy(model_cfg, tmp / "checkpoints" / model_cfg.name)

    c_manifest = json.loads((root / "runs" / "manifest.json").read_text())
    entry = next(
        (e for e in c_manifest["checkpoints"] if e["step"] == STAGE_C_FINAL_STEP),
        None,
    )
    if entry is None:
        shutil.rmtree(tmp)
        raise ValueError(
            f"Stage C manifest has no step {STAGE_C_FINAL_STEP}; has Stage C finished?"
        )
    manifest = {
        "run_id": "stage-c-final",
        "config": c_manifest.get("config", {}),
        "checkpoints": [dict(entry, path=f"checkpoints/{stem}.safetensors",
                             created_at=time.time(), is_milestone=True)],
        "latest_step": STAGE_C_FINAL_STEP,
        "latest_tokens_seen": entry["tokens_seen"],
        "latest_wall_time_s": entry["wall_time_s"],
    }
    (tmp / "manifest.json").write_text(json.dumps(manifest, indent=2))

    if RUNS_DIR.exists():
        if old.exists():
            _archive_existing_old(old)
        os.replace(RUNS_DIR, old)
        os.replace(tmp, RUNS_DIR)
        # Keep RUNS_DIR.old as durable backup (Lumi glue; do not rmtree).
    else:
        os.replace(tmp, RUNS_DIR)
    log(f"seeded from Stage C step {STAGE_C_FINAL_STEP} ({entry['tokens_seen']:,} tokens)")


def gh_token() -> str | None:
    try:
        from kaggle_secrets import UserSecretsClient
        return UserSecretsClient().get_secret("GH_TOKEN") or None
    except Exception:
        return os.environ.get("GH_TOKEN")


def clone_repo() -> None:
    if REPO_DIR.exists():
        subprocess.run(["git", "-C", str(REPO_DIR), "pull", "--ff-only"], check=True)
    else:
        subprocess.run(
            ["git", "clone", "--depth", "1", "--branch", GIT_BRANCH, GIT_REPO, str(REPO_DIR)],
            check=True,
        )


def main() -> int:
    if not preflight_disk():
        return 1
    if not preflight_quota():
        return 1

    n_train = stage_data()

    # Resume, safely: atomic stage copy, no rmtree on existing runs.
    if not resume_previous_stage_e():
        seed_from_stage_c()

    clone_repo()
    sys.path.insert(0, str(REPO_DIR / "src"))
    from playground_lm.train.stage_config import finetune_max_steps

    max_steps = finetune_max_steps(n_train, epochs=FINETUNE_EPOCHS)
    span = max_steps - STAGE_C_FINAL_STEP
    log(f"max_steps {max_steps} ({span} updates, ~{FINETUNE_EPOCHS:g} epochs)")
    if n_train < 24 * 1024:
        log(f"WARN: train set ({n_train:,} tokens) is smaller than one 24-sequence batch")

    env = dict(os.environ)
    env.update({
        "PLM_STAGE": "E",                          # fixed from "D"
        "PLM_MAX_STEPS": str(max_steps),
        "PLM_WARMUP": str(max(1, min(20, span // 4))),
        "PLM_EVAL_INTERVAL": "1" if span <= 50 else "25",
        "PLM_BATCH_SIZE": "6",
        "PLM_DATA_DIR": str(DATA_DIR),
        "PLM_RUN_DIR": str(RUNS_DIR),
        "PLM_RUN_ID": "stage-e",
        "PLM_REPO": "galactic-ai-team/playground-lm",
        "PLM_RELEASE_TAG": "stage-e-checkpoints",
        "PLM_UPLOAD_EVERY": "1",
        "PLM_MAX_SESSION_S": str(11 * 3600),
    })
    token = gh_token()
    if token:
        env["GH_TOKEN"] = token

    if DRY_RUN:
        env["CUDA_VISIBLE_DEVICES"] = ""
        env["PLM_MAX_STEPS"] = str(STAGE_C_FINAL_STEP + 1)   # one update
        log("DRY RUN: 1 CPU update, no GPU quota consumed")
        nproc = 1
    else:
        try:
            import torch
            nproc = max(1, torch.cuda.device_count())
        except Exception:
            nproc = 1

    cmd = [
        sys.executable, "-m", "torch.distributed.run", "--standalone",
        f"--nproc_per_node={nproc}",
        str(REPO_DIR / "notebooks" / "playground_lm_train.py"),
    ]
    log(f"launching torchrun on {nproc} proc(s)")
    return subprocess.run(cmd, env=env, cwd=str(REPO_DIR)).returncode


if __name__ == "__main__":
    sys.exit(main())