#!/usr/bin/env python3
# kaggle/sweep/pack_cell.py
# Emit one sweep cell's kernel package: a folder containing
#   <slug>.py             — the launcher, renamed to the slug (round-1 lesson)
#   kernel-metadata.json  — code_file == "<slug>.py"
#   cell.json             — sweep config (epochs, lr, max_steps, run_id)
#
# Refuses to write into a directory that already exists unless --force, and
# never deletes anything.
#
# Usage:
#   python kaggle/sweep/pack_cell.py \
#       --template kaggle/stage_e_train/stage_e_train.py \
#       --out-root /workspace/plm_stageE/sweep_round2 \
#       --slug plm-stage-e-train-sweep-lr3e5-ep2 \
#       --run-id sweep-lr3e5-ep2 \
#       --epochs 2 --lr 3e-5 --max-steps 31228

import argparse
import json
import re
import shutil
import sys
from pathlib import Path


SLUG_RE = re.compile(r"^[a-z0-9][a-z0-9-]{1,60}$")


def validate_slug(slug: str) -> None:
    if not SLUG_RE.match(slug):
        raise ValueError(
            f"invalid slug {slug!r}: must match {SLUG_RE.pattern} "
            f"(lowercase alnum + hyphens, Kaggle-compatible)"
        )


def build_metadata(slug: str) -> dict:
    """Return the kernel-metadata.json for this cell.

    code_file is the slug + '.py', always. That's the invariant the round-1
    failure taught us: Kaggle script kernels require the code_file basename
    to match the kernel id suffix, or the kernel goes to ERROR on launch.
    """
    return {
        "id": f"galacticaiteam/{slug}",
        "title": slug.replace("-", " ").title(),
        "code_file": f"{slug}.py",
        "language": "python",
        "kernel_type": "script",
        "is_private": True,
        "enable_gpu": True,
        "enable_internet": True,
        "dataset_sources": ["galacticaiteam/plm-stage-e-data"],
        "kernel_sources": ["galacticaiteam/plm-stage-c-train"],
        "competition_sources": [],
        "model_sources": [],
        "machine_shape": "NvidiaTeslaT4",
    }


def build_cell(slug: str, run_id: str, epochs: float, lr: float, max_steps: int) -> dict:
    return {
        "slug": slug,
        "run_id": run_id,
        "epochs": epochs,
        "lr": lr,
        "max_steps": max_steps,
    }


def package_cell(
    template: Path,
    out_root: Path,
    slug: str,
    run_id: str,
    epochs: float,
    lr: float,
    max_steps: int,
    force: bool = False,
) -> Path:
    """Create out_root/<slug>/ with the three files. Returns the cell dir."""
    validate_slug(slug)
    cell_dir = out_root / slug
    if cell_dir.exists() and not force:
        raise FileExistsError(
            f"cell directory already exists: {cell_dir}\n"
            f"refusing to overwrite (round-1 lesson: never delete prior runs). "
            f"Use --force only if you are certain this cell is unused."
        )
    if not template.is_file():
        raise FileNotFoundError(f"template not found: {template}")

    cell_dir.mkdir(parents=True, exist_ok=force)

    # Copy the template launcher, renamed to "<slug>.py".
    target_script = cell_dir / f"{slug}.py"
    shutil.copy(template, target_script)

    # Verify: after copy, the basename matches code_file.
    metadata = build_metadata(slug)
    assert metadata["code_file"] == target_script.name, (
        f"code_file {metadata['code_file']} != on-disk {target_script.name}"
    )
    (cell_dir / "kernel-metadata.json").write_text(json.dumps(metadata, indent=2))

    cell = build_cell(slug, run_id, epochs, lr, max_steps)
    (cell_dir / "cell.json").write_text(json.dumps(cell, indent=2))

    # Verify code_file == basename(script) one more time, as a hard check.
    on_disk_meta = json.loads((cell_dir / "kernel-metadata.json").read_text())
    if on_disk_meta["code_file"] != target_script.name:
        raise RuntimeError(
            f"post-write invariant violated: code_file={on_disk_meta['code_file']} "
            f"vs script={target_script.name}"
        )

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
    p.add_argument("--force", action="store_true")
    args = p.parse_args(argv)

    try:
        cell = package_cell(
            args.template, args.out_root, args.slug, args.run_id,
            args.epochs, args.lr, args.max_steps, args.force,
        )
    except (ValueError, FileExistsError, FileNotFoundError) as e:
        print(f"[pack_cell] ERROR: {e}", file=sys.stderr)
        return 1
    print(f"[pack_cell] wrote {cell}")
    return 0


if __name__ == "__main__":
    sys.exit(main())