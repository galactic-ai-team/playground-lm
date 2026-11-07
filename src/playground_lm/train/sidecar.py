# src/playground_lm/train/sidecar.py
# Self-describing checkpoints (Claude).
#
# CheckpointManager writes, beside every step_XXXXXXXX.safetensors, a small
# step_XXXXXXXX_model_config.json, and puts the same JSON in the safetensors
# header under "playground_lm". Evals and the ONNX export can then rebuild the
# model from the weights alone, without the ~1 GB _state.pt (optimizer state),
# which only resume needs. evals/run_eval.py looks for exactly this file name
# first, then the header.
#
# Torch-free on purpose: upload_release.py and scripts/backfill_model_configs.py
# use these helpers without importing torch.

from __future__ import annotations

import json
import os
from pathlib import Path
from typing import Optional

SIDECAR_SUFFIX = "_model_config.json"
SIDECAR_FORMAT = 1
SAFETENSORS_METADATA_KEY = "playground_lm"


def sidecar_path(weights_path: str | os.PathLike) -> Path:
    """step_XXXXXXXX.safetensors -> step_XXXXXXXX_model_config.json (same directory)."""
    p = Path(weights_path)
    return p.with_name(p.stem + SIDECAR_SUFFIX)


def model_config_document(
    run_id: str, config: dict, step: int, tokens_seen: int, wall_time_s: float
) -> dict:
    """What the sidecar and the safetensors header carry. "config" is the run
    config as in manifest.json; evals read the architecture from config["model"]."""
    return {
        "format": SIDECAR_FORMAT,
        "run_id": run_id,
        "step": step,
        "tokens_seen": tokens_seen,
        "wall_time_s": wall_time_s,
        "config": config,
    }


def write_model_config_sidecar(weights_path: str | os.PathLike, document: dict) -> Path:
    """Write the sidecar JSON atomically next to weights_path; returns its path."""
    path = sidecar_path(weights_path)
    tmp = path.with_name(path.name + ".tmp")
    with open(tmp, "w") as f:
        json.dump(document, f, indent=2)
    os.replace(tmp, path)
    return path


def read_model_config_document(weights_path: str | os.PathLike) -> Optional[dict]:
    """The model-config document for a weights file: the sidecar JSON if
    present, else the .safetensors header; None if neither has one."""
    path = Path(weights_path)
    side = sidecar_path(path)
    if side.is_file():
        with open(side) as f:
            return json.load(f)
    if path.suffix == ".safetensors" and path.is_file():
        try:
            from safetensors import safe_open
        except ImportError:
            return None
        with safe_open(str(path), framework="np") as f:  # header only; no torch needed
            meta = f.metadata() or {}
        if SAFETENSORS_METADATA_KEY in meta:
            return json.loads(meta[SAFETENSORS_METADATA_KEY])
    return None