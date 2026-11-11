# kaggle/stage_g_train/stage_g_train.py
# Stage G Track 1 launcher: scratchpad math + grounded facts + chat SFT + FineWeb replay,
# seeded from E31140 (v0.5 Chat). Design: Dee (Stage F launcher adaptation, 20 GB disk
# milestone spacing, no secrets, skip upload without tokens). Completed by Lumi (glue):
# reuses the tested Stage F launcher from the cloned repo instead of copying it, adds the
# grounded-facts stream, and refuses to train until the Stage G private suite is frozen.
#
# Kaggle dataset `plm-stage-g-data` (flat names; Kaggle --dir-mode skip drops subdirs):
#   facts_train.bin / facts_train.masks.bin       <- scratchpad arithmetic stream
#   grounded_train.bin / grounded_train.masks.bin <- grounded-facts stream
#   chat_train.bin / chat_train.masks.bin         <- leak-free Stage E chat SFT
#   val.bin / val.masks.bin                       <- frozen SFT val
# Env from build_sweep_stage_g(...).run_env(): PLM_GROUNDED_RATIO, PLM_CHAT_RATIO,
# PLM_REPLAY_RATIO, PLM_EFFECTIVE_BATCH, PLM_BATCH_SIZE, PLM_LR, PLM_MAX_STEPS, PLM_RUN_ID,
# optional PLM_ANSWER_LOSS_WEIGHT. Required: PLM_STAGE_G_SUITE_SHA256 (the frozen ~90-item
# suite's freeze.json SHA) unless PLM_DRY_RUN=1.
#
# Free Kaggle T4 x2 only. No paid compute. No secrets in code.

from __future__ import annotations

import importlib.util
import os
import re
import subprocess
import sys
from pathlib import Path

WORKING = Path(os.environ.get("PLM_WORKING", "/kaggle/working"))
REPO_DIR = WORKING / "playground-lm"
GROUNDED_DIR = Path(os.environ.get("PLM_GROUNDED_STAGE_DIR", "/tmp/plm_stage_g_grounded"))
GIT_REPO = "https://github.com/galactic-ai-team/playground-lm.git"
DEFAULTS = {
    "PLM_GIT_BRANCH": "feat/stage-g-scratchpad",
    "PLM_RUN_ID": "stage-g-sft",
    "PLM_RELEASE_TAG": "stage-g-checkpoints-candidate",
    "PLM_DATA_SLUG": "plm-stage-g-data",
    # ~1.5 GB per 124M milestone (weights + optimizer state): 50-step grid keeps < 15 GB.
    "PLM_CKPT_INTERVAL_S": "1800",
}
SHA_RE = re.compile(r"^[0-9a-f]{64}$")


def log(msg: str) -> None:
    print(f"[stage-g] {msg}", flush=True)


def require_frozen_suite(env=os.environ) -> None:
    """Track 1 rule: no training until GPT-6's final Stage G suite is frozen."""
    if env.get("PLM_DRY_RUN") == "1":
        return
    sha = (env.get("PLM_STAGE_G_SUITE_SHA256") or "").strip().lower()
    if not SHA_RE.match(sha):
        raise SystemExit("PLM_STAGE_G_SUITE_SHA256 (frozen Stage G suite) is required before training")


def apply_defaults(env=os.environ) -> None:
    for k, v in DEFAULTS.items():
        env.setdefault(k, v)
    if float(env.get("PLM_GROUNDED_RATIO") or 0) <= 0:
        raise SystemExit("PLM_GROUNDED_RATIO must be > 0 for Stage G (use the Stage F launcher otherwise)")
    if float(env.get("PLM_CHAT_RATIO") or 0) <= 0:
        raise SystemExit("PLM_CHAT_RATIO must be > 0 (the Stage F three-way staging needs it)")


def load_stage_f(repo: Path):
    path = repo / "kaggle" / "stage_f_train" / "stage_f_train.py"
    spec = importlib.util.spec_from_file_location("stage_f_train", path)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


def stage_grounded(root: Path, stage_pair, dest: Path = GROUNDED_DIR, env=os.environ) -> Path:
    if (root / "grounded" / "train.bin").is_file():
        gb, gm = root / "grounded" / "train.bin", root / "grounded" / "train.masks.bin"
    else:
        gb, gm = root / "grounded_train.bin", root / "grounded_train.masks.bin"
    if not gb.is_file() or not gm.is_file():
        raise SystemExit(f"grounded-facts stream missing under {root}")
    stage_pair(gb, gm, dest, "train")
    env["PLM_GROUNDED_DIR"] = str(dest)
    return dest


def clone_repo(branch: str) -> None:
    if REPO_DIR.exists():
        subprocess.run(["git", "-C", str(REPO_DIR), "fetch", "--depth", "1", "origin", branch], check=True)
        subprocess.run(["git", "-C", str(REPO_DIR), "checkout", "--force", f"origin/{branch}"], check=True)
    else:
        subprocess.run(["git", "clone", "--depth", "1", "--branch", branch, GIT_REPO, str(REPO_DIR)], check=True)


def configure(stage_f, env=os.environ) -> None:
    """Point the Stage F launcher at Stage G data, branch, and release tag."""
    stage_f.STAGE_F_DATA_SLUG = env["PLM_DATA_SLUG"]
    stage_f.GIT_BRANCH = env["PLM_GIT_BRANCH"]


def main() -> int:
    apply_defaults()
    require_frozen_suite()
    branch = os.environ["PLM_GIT_BRANCH"]
    clone_repo(branch)
    stage_f = load_stage_f(REPO_DIR)
    configure(stage_f)
    root = stage_f.find_stage_f_root()
    stage_grounded(root, stage_f._stage_pair)
    log(f"grounded -> {os.environ['PLM_GROUNDED_DIR']}; scratch/chat/val via Stage F staging from {root}")
    return stage_f.main()


if __name__ == "__main__":
    sys.exit(main())
