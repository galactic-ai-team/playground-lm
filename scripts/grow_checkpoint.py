#!/usr/bin/env python3
"""Grow a PlaygroundLM safetensors checkpoint (Stage G Track 2) - Lumi glue.

Depth first (G_stack, 12 -> 24), then width (768/12 heads -> 1024/16 heads),
using src/playground_lm/grow.py. Writes <out>/runs/checkpoints/step_NNNNNNNN.safetensors,
its _model_config.json and a runs/manifest.json so the trainer's CheckpointManager
resumes it weights-only (PLM_WEIGHTS_ONLY_RESUME=1). Refuses to overwrite.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import sys
import time
from pathlib import Path

import torch
from safetensors.torch import load_file, save_file

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
from playground_lm.grow import grow_depth, grow_width  # noqa: E402
from playground_lm.model import PlaygroundLM  # noqa: E402


def sha256(p: Path) -> str:
    h = hashlib.sha256()
    with open(p, "rb") as f:
        for c in iter(lambda: f.read(1 << 20), b""):
            h.update(c)
    return h.hexdigest()


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--checkpoint", type=Path, required=True)
    ap.add_argument("--config", type=Path, required=True)
    ap.add_argument("--out", type=Path, required=True)
    ap.add_argument("--depth-factor", type=int, default=2)
    ap.add_argument("--depth-mode", default="stack")
    ap.add_argument("--d-model", type=int, default=1024)
    ap.add_argument("--n-heads", type=int, default=16)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--run-id", default="stage-g-grow355")
    a = ap.parse_args(argv)
    if a.out.exists():
        raise SystemExit(f"refuse to overwrite {a.out}")
    cfg = json.loads(a.config.read_text())
    m = cfg["config"]["model"]
    sd = load_file(str(a.checkpoint))
    sd = grow_depth(sd, a.depth_factor, mode=a.depth_mode)
    if a.d_model != m["d_model"]:
        sd = grow_width(sd, a.d_model, n_heads_old=m["n_heads"], n_heads_new=a.n_heads, seed=a.seed)
    new_model = {**m, "n_layers": m["n_layers"] * a.depth_factor, "d_model": a.d_model, "n_heads": a.n_heads}
    model = PlaygroundLM(**new_model)
    model.load_state_dict(sd, strict=True)
    n_params = sum(p.numel() for p in model.parameters())
    step = cfg["step"]
    ck = a.out / "runs" / "checkpoints"
    ck.mkdir(parents=True)
    w = ck / f"step_{step:08d}.safetensors"
    save_file({k: v.contiguous() for k, v in model.state_dict().items()}, str(w))
    new_cfg = {**cfg, "run_id": a.run_id, "config": {**cfg["config"], "model": new_model},
               "grown_from": {"sha256": sha256(a.checkpoint), "step": step, "model": m,
                              "depth_factor": a.depth_factor, "depth_mode": a.depth_mode,
                              "width_seed": a.seed}}
    (ck / f"step_{step:08d}_model_config.json").write_text(json.dumps(new_cfg, indent=2))
    manifest = {"run_id": a.run_id, "config": new_cfg["config"],
                "checkpoints": [{"step": step, "path": f"checkpoints/step_{step:08d}.safetensors",
                                 "tokens_seen": cfg["tokens_seen"], "wall_time_s": 0.0,
                                 "created_at": time.time(), "is_milestone": True}],
                "latest_step": step, "latest_tokens_seen": cfg["tokens_seen"], "latest_wall_time_s": 0.0}
    (a.out / "runs" / "manifest.json").write_text(json.dumps(manifest, indent=2))
    print(json.dumps({"params": n_params, "model": new_model, "weights": str(w), "sha256": sha256(w)}))
    return 0


if __name__ == "__main__":
    sys.exit(main())
