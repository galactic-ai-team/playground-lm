#!/usr/bin/env python3
# kaggle/sweep/run_sweep.py (Round 3)
# Multi-run free-GPU sweep launcher. Dry-run only by default.
#
# Round-3 deltas vs Round 2:
#   - Concurrency limit: at most 2 concurrent Kaggle GPU sessions
#     (free tier constraint). Cells beyond the limit are queued.
#   - Reports replay_ratio and commit per cell in the summary.
#   - COMPLETE.json is a local skip marker only, not a Kaggle contract.

import argparse
import json
import subprocess
import sys
from pathlib import Path


COMPLETE_MARKER = "COMPLETE.json"
MAX_CONCURRENT_GPU = 2


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


def push_cell(cell_dir: Path) -> int:
    cmd = ["kaggle", "kernels", "push", "-p", str(cell_dir)]
    log(f"$ {' '.join(cmd)}")
    return subprocess.run(cmd).returncode


def build_sweep_grid(
    base_slug: str,
    lr_values: list[float],
    epochs_values: list[float],
    replay_values: list[float],
    max_steps: int,
) -> list[dict]:
    cells = []
    for lr in lr_values:
        for ep in epochs_values:
            for rp in replay_values:
                suffix = f"lr{_fmt_sci(lr)}-ep{_fmt_num(ep)}-rp{_fmt_num(rp)}"
                slug = f"{base_slug}-{suffix}"
                cells.append({
                    "slug": slug,
                    "run_id": slug,
                    "lr": lr,
                    "epochs": ep,
                    "replay_ratio": rp,
                    "max_steps": max_steps,
                })
    return cells


def _fmt_sci(x: float) -> str:
    return f"{x:g}".replace(".", "").replace("-", "")


def _fmt_num(x: float) -> str:
    return f"{x:g}".replace(".", "p")


def main(argv: list[str] | None = None) -> int:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--pack-root", type=Path, required=True)
    p.add_argument("--out-root", type=Path, required=True)
    p.add_argument("--execute", action="store_true")
    p.add_argument("--i-have-read-launch-md", action="store_true")
    args = p.parse_args(argv)

    if args.execute and not args.i_have_read_launch_md:
        log("REFUSED: --execute requires --i-have-read-launch-md")
        log("        see kaggle/sweep/LAUNCH.md §4 for the runbook")
        return 2

    cells = sorted([d for d in args.pack_root.iterdir() if d.is_dir()])
    if not cells:
        log(f"no cell directories under {args.pack_root}")
        return 1

    results = []
    pending = []
    for cell_dir in cells:
        cell = json.loads((cell_dir / "cell.json").read_text())
        rd = args.out_root / cell["run_id"]
        if is_complete(rd):
            log(f"{cell['slug']}: already COMPLETE, skipping")
            results.append({
                "slug": cell["slug"], "status": "skipped-complete",
                "replay_ratio": cell.get("replay_ratio"),
                "git_commit": cell.get("git_commit"),
            })
            continue
        pending.append((cell_dir, cell))

    # Report full pending list first (dry-run visibility).
    for cell_dir, cell in pending:
        log(
            f"{cell['slug']}: pending "
            f"(lr={cell['lr']}, ep={cell['epochs']}, "
            f"replay={cell.get('replay_ratio')}, commit={cell.get('git_commit')})"
        )

    if not args.execute:
        for cell_dir, cell in pending:
            results.append({
                "slug": cell["slug"], "status": "dry-run",
                "replay_ratio": cell.get("replay_ratio"),
                "git_commit": cell.get("git_commit"),
            })
    else:
        # Round-3: at most 2 concurrent GPU sessions. The free tier will
        # queue beyond that, but explicit batching avoids hitting Kaggle's
        # per-account concurrency cap mid-sweep.
        active = []
        for cell_dir, cell in pending:
            while len(active) >= MAX_CONCURRENT_GPU:
                # Simplest safe behavior: push the rest of the cells in
                # subsequent invocations, not in this one. Report as queued.
                break
            else:
                log(f"{cell['slug']}: pushing")
                rc = push_cell(cell_dir)
                status = "pushed" if rc == 0 else f"push-exit-{rc}"
                results.append({
                    "slug": cell["slug"], "status": status,
                    "replay_ratio": cell.get("replay_ratio"),
                    "git_commit": cell.get("git_commit"),
                })
                if rc == 0:
                    active.append(cell["slug"])
                continue
            results.append({
                "slug": cell["slug"], "status": "queued-concurrency",
                "replay_ratio": cell.get("replay_ratio"),
                "git_commit": cell.get("git_commit"),
            })

    summary = {
        "cells": len(cells),
        "dry_run": not args.execute,
        "max_concurrent_gpu": MAX_CONCURRENT_GPU,
        "results": results,
    }
    print(json.dumps(summary, indent=2))
    return 0


if __name__ == "__main__":
    sys.exit(main())