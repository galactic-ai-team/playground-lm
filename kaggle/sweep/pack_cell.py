#!/usr/bin/env python3
# kaggle/sweep/pack_cell.py (Round 3)
# Emit one sweep cell's kernel package: <slug>.py + kernel-metadata.json + cell.json.
#
# Round-3 deltas vs Round 2:
#   - Injects PLM_REPLAY_RATIO into the emitted launcher.
#   - Adds Stage C replay shard mounts (verify exact slugs before first push).
#   - cell.json records replay_ratio so LAUNCH.md / run_sweep can report it.
#   - Never deletes anything; --force overwrites files in the cell only.
#
# code_file == "<slug>.py" is enforced three times: at build time, after
# copy, and after writing metadata (round-1 lesson).

import argparse
import json
import re
import shutil
import sys
from pathlib import Path


SLUG_RE = re.compile(r"^[a-z0-9][a-z0-9-]{1,60}$")

# Verified Kaggle dataset for FineWeb-Edu-style pretraining replay (train.bin).
# Stage B corpus is the same FineWeb-Edu family Stage C continued from.
REPLAY_DATASET_SLUGS = [
    "galacticaiteam/plm-stageb-data",
]
# Mounted path for Claude online mixer (Kaggle input layout).
DEFAULT_REPLAY_BIN = "/kaggle/input/plm-stageb-data/train.bin"


def validate_slug(slug: str) -> None:
    if not SLUG_RE.match(slug):
        raise ValueError(
            f"invalid slug {slug!r}: must match {SLUG_RE.pattern}"
        )


def build_metadata(slug: str) -> dict:
    return {
        "id": f"galacticaiteam/{slug}",
        "title": slug.replace("-", " ").title(),
        "code_file": f"{slug}.py",
        "language": "python",
        "kernel_type": "script",
        "is_private": True,
        "enable_gpu": True,
        "enable_internet": True,
        "dataset_sources": list(dict.fromkeys([
            "galacticaiteam/plm-stage-e-data",
            *REPLAY_DATASET_SLUGS,
            "galacticaiteam/plm-stagec-val",  # FineWeb val for optional in-kernel checks
        ])),
        "kernel_sources": ["galacticaiteam/plm-stage-c-train"],
        "competition_sources": [],
        "model_sources": [],
        "machine_shape": "NvidiaTeslaT4",
    }


def build_cell(
    slug: str,
    run_id: str,
    epochs: float,
    lr: float,
    max_steps: int,
    replay_ratio: float,
    git_commit: str,
) -> dict:
    return {
        "slug": slug,
        "run_id": run_id,
        "epochs": epochs,
        "lr": lr,
        "max_steps": max_steps,
        "replay_ratio": replay_ratio,
        "git_commit": git_commit,
    }


def _inject_env_block(launcher_src: str, cell: dict) -> str:
    """Insert Round-3 env overrides before the torchrun env block.

    Bakes PLM_REPLAY_RATIO / PLM_REPLAY_BIN / PLM_LR / PLM_MAX_STEPS /
    PLM_WARMUP / epochs / run id / git pin so cell.json cannot drift from
    the launched script (Claude note: Dee must wire MAX_STEPS + WARMUP).
    """
    warmup = max(1, min(20, (int(cell["max_steps"]) - 31101) // 4))
    replay_bin = cell.get("replay_bin") or DEFAULT_REPLAY_BIN
    assignments = (
        f'env["PLM_REPLAY_RATIO"] = "{cell["replay_ratio"]}"\n'
        f'env["PLM_REPLAY_BIN"] = "{replay_bin}"\n'
        f'env["PLM_LR"] = "{cell["lr"]}"\n'
        f'env["PLM_FINETUNE_EPOCHS"] = "{cell["epochs"]}"\n'
        f'env["PLM_MAX_STEPS"] = "{cell["max_steps"]}"\n'
        f'env["PLM_WARMUP"] = "{warmup}"\n'
        f'env["PLM_RUN_ID"] = "{cell["run_id"]}"\n'
        f'env["PLM_GIT_COMMIT"] = "{cell["git_commit"]}"\n'
        'env["PLM_RESET_OPTIMIZER"] = "1"\n'
        'env["PLM_QUOTA_GUARD"] = "1"\n'
    )
    marker = "env = dict(os.environ)"
    if marker in launcher_src:
        return launcher_src.replace(marker, marker + "\n" + assignments, 1)
    marker2 = "env.update({"
    if marker2 in launcher_src:
        inject = (
            "    env.update({\n"
            f'        "PLM_REPLAY_RATIO": "{cell["replay_ratio"]}",\n'
            f'        "PLM_REPLAY_BIN": "{replay_bin}",\n'
            f'        "PLM_LR": "{cell["lr"]}",\n'
            f'        "PLM_FINETUNE_EPOCHS": "{cell["epochs"]}",\n'
            f'        "PLM_MAX_STEPS": "{cell["max_steps"]}",\n'
            f'        "PLM_WARMUP": "{warmup}",\n'
            f'        "PLM_RUN_ID": "{cell["run_id"]}",\n'
            f'        "PLM_GIT_COMMIT": "{cell["git_commit"]}",\n'
        )
        return launcher_src.replace(marker2, inject + marker2, 1)
    raise RuntimeError(
        "launcher template does not contain 'env = dict(os.environ)' "
        "or 'env.update({'; cannot inject sweep env safely"
    )



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
    force: bool = False,
) -> Path:
    validate_slug(slug)
    cell_dir = out_root / slug
    if cell_dir.exists() and not force:
        raise FileExistsError(
            f"cell directory already exists: {cell_dir} "
            f"(refusing to overwrite; use --force only if certain)"
        )
    if not template.is_file():
        raise FileNotFoundError(f"template not found: {template}")

    cell_dir.mkdir(parents=True, exist_ok=force)

    # Read, inject, write — never copy-then-modify so the injection is atomic.
    src = template.read_text()
    cell = build_cell(slug, run_id, epochs, lr, max_steps, replay_ratio, git_commit)
    injected = _inject_env_block(src, cell)

    target_script = cell_dir / f"{slug}.py"
    target_script.write_text(injected)

    metadata = build_metadata(slug)
    assert metadata["code_file"] == target_script.name, (
        f"code_file {metadata['code_file']} != {target_script.name}"
    )
    (cell_dir / "kernel-metadata.json").write_text(json.dumps(metadata, indent=2))
    (cell_dir / "cell.json").write_text(json.dumps(cell, indent=2))

    # Post-write invariant.
    on_disk_meta = json.loads((cell_dir / "kernel-metadata.json").read_text())
    if on_disk_meta["code_file"] != target_script.name:
        raise RuntimeError(
            f"post-write invariant violated: {on_disk_meta['code_file']} vs {target_script.name}"
        )

    # Verify injection took.
    written = target_script.read_text()
    for key in ("PLM_REPLAY_RATIO", "PLM_REPLAY_BIN", "PLM_LR", "PLM_FINETUNE_EPOCHS",
                "PLM_MAX_STEPS", "PLM_WARMUP", "PLM_RUN_ID", "PLM_GIT_COMMIT"):
        if key not in written:
            raise RuntimeError(f"injection missing {key} in {target_script}")

    return cell_dir


def main(argv: list[str] | None = None) -> int:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--template", type=Path, required=True)
    p.add_argument("--out-root", type=Path, required=True)
    p.add_argument("--slug", required=True)
    p.add_argument("--run-id", required=True)
    p.add_argument("--epochs", type=float, required=True)
    p.add_argument("--lr", type=float, required=True)
    p.add_argument("--max-steps", type=int, required=True)
    p.add_argument("--replay-ratio", type=float, required=True,
                   help="Fraction of batch drawn from Stage C replay (0..1)")
    p.add_argument("--git-commit", required=True,
                   help="Commit SHA of the launcher+model code being packaged")
    p.add_argument("--force", action="store_true")
    args = p.parse_args(argv)

    if not 0.0 <= args.replay_ratio <= 1.0:
        print(f"[pack_cell] ERROR: --replay-ratio must be in [0,1], got {args.replay_ratio}",
              file=sys.stderr)
        return 1

    try:
        cell = package_cell(
            args.template, args.out_root, args.slug, args.run_id,
            args.epochs, args.lr, args.max_steps, args.replay_ratio,
            args.git_commit, args.force,
        )
    except (ValueError, FileExistsError, FileNotFoundError, RuntimeError) as e:
        print(f"[pack_cell] ERROR: {e}", file=sys.stderr)
        return 1
    print(f"[pack_cell] wrote {cell}")
    return 0


if __name__ == "__main__":
    sys.exit(main())