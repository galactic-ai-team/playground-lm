#!/usr/bin/env python3
"""Package Stage G Track 1 cells (G1/G2/G3) from build_sweep_stage_g.

Each cell = <slug>.py (env-override block + kaggle/stage_g_train/stage_g_train.py)
+ kernel-metadata.json + cell.json. Env comes from sweep_config.run_env so the
launcher and the plan cannot drift. Only the public frozen-suite *hash* is
embedded (PLM_STAGE_G_SUITE_SHA256); no private items.
"""
from __future__ import annotations

import argparse
import json
import re
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "src"))
from playground_lm.train.sweep_config import build_sweep_stage_g, run_env  # noqa: E402

SHA_RE = re.compile(r"^[0-9a-f]{64}$")
REPLAY_BIN = "/kaggle/input/datasets/galacticaiteam/plm-stageb-data/train.bin"
DATASETS = [
    "galacticaiteam/plm-stage-g-data",
    "galacticaiteam/plm-stageb-data",
    "galacticaiteam/plm-e31140-seed",
    "galacticaiteam/plm-stagec-val",
]


def cell_env(run, *, suite_sha: str, git_commit: str, milestones: str,
             eval_interval: int = 25, est_session_h: float = 1.5) -> dict[str, str]:
    if not SHA_RE.match(suite_sha):
        raise ValueError("suite sha must be 64 hex")
    if not re.match(r"^[0-9a-f]{40}$", git_commit):
        raise ValueError("git commit must be a full 40-hex sha")
    env = run_env(run)
    env.update({
        "PLM_REPLAY_BIN": REPLAY_BIN,
        "PLM_CHAT_DIR": "/tmp/plm_stage_f_chat",
        "PLM_FINETUNE_EPOCHS": str(run.epochs),
        "PLM_EVAL_INTERVAL": str(eval_interval),
        "PLM_MILESTONES": milestones,
        "PLM_GIT_COMMIT": git_commit,
        "PLM_GIT_BRANCH": "feat/stage-g-scratchpad",
        "PLM_WEIGHTS_ONLY_RESUME": "1",
        "PLM_QUOTA_GUARD": "1",
        "PLM_EST_SESSION_H": str(est_session_h),
        "PLM_STAGE_G_SUITE_SHA256": suite_sha,
    })
    return env


def package(run, slug: str, out_root: Path, template: Path, **kw) -> Path:
    cell_dir = out_root / slug
    if cell_dir.exists():
        raise FileExistsError(cell_dir)
    cell_dir.mkdir(parents=True)
    env = cell_env(run, **kw)
    block = "# --- Stage G cell overrides ---\nimport os as _os_cell\n" + "".join(
        f"_os_cell.environ[{k!r}] = {v!r}\n" for k, v in sorted(env.items())
    ) + "# --- end overrides ---\n\n"
    (cell_dir / f"{slug}.py").write_text(block + template.read_text())
    meta = {
        "id": f"galacticaiteam/{slug}", "title": slug.replace("-", " ").title(),
        "code_file": f"{slug}.py", "language": "python", "kernel_type": "script",
        "is_private": True, "enable_gpu": True, "enable_internet": True,
        "dataset_sources": DATASETS, "kernel_sources": [], "competition_sources": [],
        "model_sources": [], "machine_shape": "NvidiaTeslaT4",
    }
    (cell_dir / "kernel-metadata.json").write_text(json.dumps(meta, indent=2) + "\n")
    (cell_dir / "cell.json").write_text(json.dumps({"slug": slug, "run_id": run.run_id, "env": env}, indent=2) + "\n")
    return cell_dir


def main(argv=None) -> int:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--out-root", type=Path, required=True)
    p.add_argument("--suite-sha", required=True)
    p.add_argument("--git-commit", required=True)
    p.add_argument("--tokens", required=True, help="scratch,grounded,chat token counts")
    p.add_argument("--which", default="G1,G2", help="subset of G1,G2,G3")
    a = p.parse_args(argv)
    s, g, c = (int(x) for x in a.tokens.split(","))
    runs = build_sweep_stage_g({"scratch": s, "grounded": g, "chat": c}, include_optional=True)
    names = {"G1": runs[0], "G2": runs[1], "G3": runs[2]}
    template = ROOT / "kaggle" / "stage_g_train" / "stage_g_train.py"
    for name in a.which.split(","):
        run = names[name]
        last = run.max_steps
        ms = f"{run.schedule_start + (last - run.schedule_start) // 3},{run.schedule_start + 2 * (last - run.schedule_start) // 3},{last}"
        slug = f"plm-stage-g-{name.lower()}"
        d = package(run, slug, a.out_root, template, suite_sha=a.suite_sha,
                    git_commit=a.git_commit, milestones=ms)
        print(f"[pack_stage_g] {name} -> {d} ({run.run_id}, milestones {ms})")
    return 0


if __name__ == "__main__":
    sys.exit(main())
