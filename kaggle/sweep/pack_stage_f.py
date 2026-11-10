#!/usr/bin/env python3
"""Package a Stage F sweep cell: <slug>.py + kernel-metadata.json + cell.json."""
from __future__ import annotations

import argparse
import json
import re
import sys
from pathlib import Path

SLUG_RE = re.compile(r"^[a-z0-9][a-z0-9-]{1,60}$")
SEED_STEP = 31140
DEFAULT_REPLAY_BIN = "/kaggle/input/datasets/galacticaiteam/plm-stageb-data/train.bin"
DATASETS = [
    "galacticaiteam/plm-stage-f-data",
    "galacticaiteam/plm-stageb-data",
    "galacticaiteam/plm-e31140-seed",
    "galacticaiteam/plm-stagec-val",
]


def validate_slug(slug: str) -> None:
    if not SLUG_RE.match(slug):
        raise ValueError(f"invalid slug {slug!r}")


def package_cell(
    template: Path,
    out_root: Path,
    slug: str,
    run_id: str,
    epochs: float,
    lr: float,
    max_steps: int,
    replay_ratio: float,
    git_commit: str,
    seed_step: int = SEED_STEP,
    force: bool = False,
) -> Path:
    validate_slug(slug)
    cell_dir = out_root / slug
    if cell_dir.exists() and not force:
        raise FileExistsError(f"cell directory already exists: {cell_dir}")
    if not template.is_file():
        raise FileNotFoundError(f"template not found: {template}")
    cell_dir.mkdir(parents=True, exist_ok=force)

    updates = max_steps - seed_step
    warmup = max(1, min(20, updates // 4))
    cell = {
        "slug": slug,
        "run_id": run_id,
        "epochs": epochs,
        "lr": lr,
        "max_steps": max_steps,
        "replay_ratio": replay_ratio,
        "git_commit": git_commit,
        "seed_step": seed_step,
        "warmup": warmup,
    }
    block = (
        "# --- sweep cell overrides (Stage F) ---\n"
        "import os as _os_sweep\n"
        f'_os_sweep.environ["PLM_REPLAY_RATIO"] = "{replay_ratio}"\n'
        f'_os_sweep.environ["PLM_REPLAY_BIN"] = "{DEFAULT_REPLAY_BIN}"\n'
        f'_os_sweep.environ["PLM_LR"] = "{lr}"\n'
        f'_os_sweep.environ["PLM_FINETUNE_EPOCHS"] = "{epochs}"\n'
        f'_os_sweep.environ["PLM_MAX_STEPS"] = "{max_steps}"\n'
        f'_os_sweep.environ["PLM_WARMUP"] = "{warmup}"\n'
        f'_os_sweep.environ["PLM_SCHEDULE_START"] = "{seed_step}"\n'
        f'_os_sweep.environ["PLM_SEED_STEP"] = "{seed_step}"\n'
        f'_os_sweep.environ["PLM_RUN_ID"] = "{run_id}"\n'
        f'_os_sweep.environ["PLM_GIT_COMMIT"] = "{git_commit}"\n'
        f'_os_sweep.environ["PLM_GIT_BRANCH"] = "feat/stage-f-facts-arith"\n'
        '_os_sweep.environ["PLM_RESET_OPTIMIZER"] = "1"\n'
        '_os_sweep.environ["PLM_WEIGHTS_ONLY_RESUME"] = "1"\n'
        '_os_sweep.environ["PLM_QUOTA_GUARD"] = "1"\n'
        "# --- end sweep cell overrides ---\n\n"
    )
    (cell_dir / f"{slug}.py").write_text(block + template.read_text())
    metadata = {
        "id": f"galacticaiteam/{slug}",
        "title": slug.replace("-", " ").title(),
        "code_file": f"{slug}.py",
        "language": "python",
        "kernel_type": "script",
        "is_private": True,
        "enable_gpu": True,
        "enable_internet": True,
        "dataset_sources": list(DATASETS),
        "kernel_sources": [],
        "competition_sources": [],
        "model_sources": [],
        "machine_shape": "NvidiaTeslaT4",
    }
    (cell_dir / "kernel-metadata.json").write_text(json.dumps(metadata, indent=2) + "\n")
    (cell_dir / "cell.json").write_text(json.dumps(cell, indent=2) + "\n")
    on_disk = json.loads((cell_dir / "kernel-metadata.json").read_text())
    if on_disk["code_file"] != f"{slug}.py":
        raise RuntimeError("code_file invariant violated")
    return cell_dir


def main(argv=None) -> int:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--template", type=Path, required=True)
    p.add_argument("--out-root", type=Path, required=True)
    p.add_argument("--slug", required=True)
    p.add_argument("--run-id", required=True)
    p.add_argument("--epochs", type=float, required=True)
    p.add_argument("--lr", type=float, required=True)
    p.add_argument("--max-steps", type=int, required=True)
    p.add_argument("--replay-ratio", type=float, required=True)
    p.add_argument("--git-commit", required=True)
    p.add_argument("--seed-step", type=int, default=SEED_STEP)
    p.add_argument("--force", action="store_true")
    args = p.parse_args(argv)
    if not 0.0 <= args.replay_ratio < 1.0:
        print("replay_ratio must be in [0,1)", file=sys.stderr)
        return 1
    try:
        cell = package_cell(
            args.template, args.out_root, args.slug, args.run_id,
            args.epochs, args.lr, args.max_steps, args.replay_ratio,
            args.git_commit, args.seed_step, args.force,
        )
    except (ValueError, FileExistsError, FileNotFoundError, RuntimeError) as e:
        print(f"[pack_stage_f] ERROR: {e}", file=sys.stderr)
        return 1
    print(f"[pack_stage_f] wrote {cell}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
