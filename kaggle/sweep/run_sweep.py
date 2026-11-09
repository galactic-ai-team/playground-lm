#!/usr/bin/env python3
# kaggle/sweep/run_sweep.py
# Multi-run free-GPU sweep launcher.
#
# DRY-RUN ONLY by default. With --execute, it would run `kaggle kernels push`
# for each cell — but the guard below refuses unless the caller explicitly
# sets --execute AND --i-have-read-launch-md. This is intentional friction:
# the sweep must not start until Stephon/Lumi says push-to-run.
#
# Durable outputs: each run_id writes to /workspace/plm_stageE/runs/<run_id>/.
# Existing run directories are preserved — a COMPLETE run is skipped, an
# incomplete run is left for the operator to inspect. The launcher never
# rm -rf's anything.

import argparse
import json
import subprocess
import sys
from pathlib import Path


COMPLETE_MARKER = "COMPLETE.json"


def log(msg: str) -> None:
    print(f"[sweep] {msg}", flush=True)


def is_complete(run_dir: Path) -> bool:
    marker = run_dir / COMPLETE_MARKER
    if not marker.is_file():
        return False
    try:
        m = json.loads(marker.read_text())
    except json.JSONDecodeError:
        return False
    return m.get("status") == "complete"


def run_dir_for(out_root: Path, run_id: str) -> Path:
    return out_root / run_id


def push_cell(cell_dir: Path) -> int:
    """Push a single kernel. Returns the Kaggle CLI exit code."""
    cmd = ["kaggle", "kernels", "push", "-p", str(cell_dir)]
    log(f"$ {' '.join(cmd)}")
    return subprocess.run(cmd).returncode


def build_sweep_grid(
    base_slug: str,
    lr_values: list[float],
    epochs_values: list[float],
    max_steps: int,
) -> list[dict]:
    """Cartesian product of LR × epochs, one cell per (lr, epochs)."""
    cells = []
    for lr in lr_values:
        for ep in epochs_values:
            slug = f"{base_slug}-lr{_fmt_sci(lr)}-ep{_fmt_num(ep)}"
            run_id = f"{base_slug}-lr{_fmt_sci(lr)}-ep{_fmt_num(ep)}"
            cells.append({
                "slug": slug,
                "run_id": run_id,
                "lr": lr,
                "epochs": ep,
                "max_steps": max_steps,
            })
    return cells


def _fmt_sci(x: float) -> str:
    # 3e-5 -> "3e5"; 1e-5 -> "1e5" (normalize exponent width)
    s = f"{x:.0e}"  # e.g. 3e-05
    return s.replace(".", "").replace("-0", "").replace("-", "")


def _fmt_num(x: float) -> str:
    return f"{x:g}"


def main(argv: list[str] | None = None) -> int:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--pack-root", type=Path, required=True,
                   help="root containing per-cell folders written by pack_cell.py")
    p.add_argument("--out-root", type=Path, required=True,
                   help="root where per-run_id output dirs live (durable)")
    p.add_argument("--execute", action="store_true",
                   help="actually push kernels (refused without --i-have-read-launch-md)")
    p.add_argument("--i-have-read-launch-md", action="store_true",
                   help="required alongside --execute; explicit consent to launch")
    args = p.parse_args(argv)

    if args.execute and not args.i_have_read_launch_md:
        log("REFUSED: --execute requires --i-have-read-launch-md")
        log("        see kaggle/sweep/LAUNCH.md for the runbook")
        return 2

    cells = sorted([d for d in args.pack_root.iterdir() if d.is_dir()])
    if not cells:
        log(f"no cell directories under {args.pack_root}")
        return 1

    results = []
    for cell_dir in cells:
        cell = json.loads((cell_dir / "cell.json").read_text())
        rd = run_dir_for(args.out_root, cell["run_id"])
        status = "skipped-complete" if is_complete(rd) else "pending"

        if status == "skipped-complete":
            log(f"{cell['slug']}: already COMPLETE, skipping")
            results.append({"slug": cell["slug"], "status": status})
            continue

        if not args.execute:
            log(f"{cell['slug']}: DRY-RUN pending (would push {cell_dir})")
            results.append({"slug": cell["slug"], "status": "dry-run"})
            continue

        log(f"{cell['slug']}: pushing")
        rc = push_cell(cell_dir)
        results.append({"slug": cell["slug"], "status": "pushed" if rc == 0 else f"push-exit-{rc}"})

    summary = {
        "cells": len(cells),
        "dry_run": not args.execute,
        "results": results,
    }
    print(json.dumps(summary, indent=2))
    return 0


if __name__ == "__main__":
    sys.exit(main())