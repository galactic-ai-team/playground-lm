# train/checkpoint.py
# Resumable checkpoint management for PlaygroundLM.
#
# Responsibilities:
#   - Save checkpoints to a local directory on a fixed step interval (~30 min).
#   - Rotate: keep the N most recent rolling checkpoints + all "milestone"
#     checkpoints (fixed step counts) + the latest.
#   - Resume: detect the newest checkpoint and return its state.
#   - Emit a manifest.json that the demo frontend and the release script both
#     consume.
#
# Design notes:
#   - Checkpoints are stored as two files: model weights (safetensors) and
#     optimizer/training state (pickle, or torch.save). Separating them lets
#     the export pipeline package weights for inference without pulling
#     optimizer state.
#   - Manifest is written atomically (write to .tmp, then os.replace) so a
#     killed process doesn't leave a half-written manifest.
#   - All paths are relative to a single run directory so a Colab/Kaggle
#     session can be resumed just by re-mounting the same directory.

import json
import os
import shutil
import time
from dataclasses import dataclass, asdict, field
from pathlib import Path
from typing import Optional

import torch


# --- Config ---------------------------------------------------------------

# Rolling checkpoints: keep this many recent ones. Older rolling checkpoints
# are deleted. Milestones are always kept regardless of this limit.
ROLLING_KEEP = 3

# Milestone steps: these checkpoints are never deleted. Pick something that
# maps to useful "scrub points" in the demo slider. 1024 steps at ~500 tok/s
# effective throughput on a P100 is roughly 15 min, so these represent
# progressively more-trained models without keeping everything.
DEFAULT_MILESTONES = (1000, 5000, 10000, 25000, 50000, 100000, 200000)

# Checkpoint interval in seconds. 30 min = 1800 s. Override via env var for
# testing.
DEFAULT_INTERVAL_S = int(os.environ.get("PLM_CKPT_INTERVAL_S", "1800"))


# --- Manifest -------------------------------------------------------------

@dataclass
class CheckpointEntry:
    step: int
    path: str                       # relative to run dir
    tokens_seen: int
    wall_time_s: float
    created_at: float               # unix timestamp
    is_milestone: bool


@dataclass
class Manifest:
    run_id: str
    config: dict
    checkpoints: list[CheckpointEntry] = field(default_factory=list)
    latest_step: int = -1
    latest_tokens_seen: int = 0
    latest_wall_time_s: float = 0.0


# --- Manager --------------------------------------------------------------

class CheckpointManager:
    def __init__(
        self,
        run_dir: str | Path,
        run_id: str,
        config: dict,
        interval_s: int = DEFAULT_INTERVAL_S,
        milestones: tuple[int, ...] = DEFAULT_MILESTONES,
        rolling_keep: int = ROLLING_KEEP,
    ):
        self.run_dir = Path(run_dir)
        self.run_dir.mkdir(parents=True, exist_ok=True)
        self.ckpt_dir = self.run_dir / "checkpoints"
        self.ckpt_dir.mkdir(exist_ok=True)
        self.manifest_path = self.run_dir / "manifest.json"

        self.run_id = run_id
        self.config = config
        self.interval_s = interval_s
        self.milestones = set(milestones)
        self.rolling_keep = rolling_keep

        self._last_save_time = 0.0
        self._manifest = self._load_or_init_manifest()

    # --- public API -------------------------------------------------------

    def should_save(self, step: int) -> bool:
        """True if a checkpoint is due by time interval or by milestone step."""
        if step in self.milestones:
            return True
        if self._last_save_time == 0.0:
            return True  # always save once early
        return (time.time() - self._last_save_time) >= self.interval_s

    def save(
        self,
        model: torch.nn.Module,
        optimizer: torch.optim.Optimizer,
        step: int,
        tokens_seen: int,
        wall_time_s: float,
    ) -> CheckpointEntry:
        """Save a checkpoint. Returns the manifest entry. Rotates old rolling
        checkpoints after saving."""
        stem = f"step_{step:08d}"
        weights_path = self.ckpt_dir / f"{stem}.safetensors"
        state_path = self.ckpt_dir / f"{stem}_state.pt"

        # Weights: safetensors for inference/export.
        try:
            from safetensors.torch import save_file
            # Convert bf16/fp16 state dict to fp16 for portability? No:
            # safetensors handles whatever dtype the tensors are. Save as-is
            # so the export pipeline can decide what to quantize.
            state_dict = {k: v.detach().cpu() for k, v in model.state_dict().items()}
            save_file(state_dict, str(weights_path))
        except ImportError:
            # Fallback if safetensors unavailable in this env.
            torch.save(model.state_dict(), str(weights_path).replace(".safetensors", ".pt"))
            weights_path = Path(str(weights_path).replace(".safetensors", ".pt"))

        # Training state: optimizer, step, rng — everything needed to resume.
        torch.save(
            {
                "step": step,
                "tokens_seen": tokens_seen,
                "wall_time_s": wall_time_s,
                "optimizer": optimizer.state_dict(),
                "torch_rng_state": torch.get_rng_state(),
                "cuda_rng_state": torch.cuda.get_rng_state_all() if torch.cuda.is_available() else None,
                "config": self.config,
            },
            str(state_path),
        )

        entry = CheckpointEntry(
            step=step,
            path=str(weights_path.relative_to(self.run_dir)),
            tokens_seen=tokens_seen,
            wall_time_s=wall_time_s,
            created_at=time.time(),
            is_milestone=step in self.milestones,
        )

        self._manifest.checkpoints.append(entry)
        self._manifest.latest_step = step
        self._manifest.latest_tokens_seen = tokens_seen
        self._manifest.latest_wall_time_s = wall_time_s
        self._manifest.checkpoints.sort(key=lambda e: e.step)

        self._rotate()
        self._write_manifest()
        self._last_save_time = time.time()
        return entry

    def resume(
        self, model: torch.nn.Module, optimizer: torch.optim.Optimizer
    ) -> Optional[dict]:
        """Load the latest checkpoint into model + optimizer. Returns the state
        dict (so caller can read step/tokens_seen/rng), or None if no
        checkpoint exists."""
        if not self._manifest.checkpoints:
            return None
        latest = self._manifest.checkpoints[-1]
        weights_path = self.run_dir / latest.path
        # Find matching state file: same stem, "_state.pt" suffix.
        stem = weights_path.stem.replace(".safetensors", "")
        if stem.endswith("_state"):
            stem = stem[: -len("_state")]
        state_path = self.ckpt_dir / f"{stem}_state.pt"

        if not weights_path.exists() or not state_path.exists():
            raise FileNotFoundError(
                f"Manifest references missing files: {weights_path}, {state_path}"
            )

        # Load weights.
        if str(weights_path).endswith(".safetensors"):
            from safetensors.torch import load_file
            state_dict = load_file(str(weights_path))
        else:
            state_dict = torch.load(str(weights_path), map_location="cpu")
        model.load_state_dict(state_dict)

        # Load training state.
        state = torch.load(str(state_path), map_location="cpu")
        optimizer.load_state_dict(state["optimizer"])
        torch.set_rng_state(state["torch_rng_state"])
        if state.get("cuda_rng_state") is not None and torch.cuda.is_available():
            torch.cuda.set_rng_state_all(state["cuda_rng_state"])
        return state

    def list_checkpoints(self) -> list[CheckpointEntry]:
        """Sorted by step ascending."""
        return list(self._manifest.checkpoints)

    # --- internals --------------------------------------------------------

    def _rotate(self) -> None:
        """Delete rolling checkpoints beyond the rolling_keep limit, keeping
        milestones and the latest checkpoint always."""
        rolling = [
            e for e in self._manifest.checkpoints if not e.is_milestone
        ]
        if len(rolling) <= self.rolling_keep:
            return
        to_delete = rolling[: -self.rolling_keep]
        delete_steps = {e.step for e in to_delete}
        self._manifest.checkpoints = [
            e for e in self._manifest.checkpoints if e.step not in delete_steps
        ]
        for e in to_delete:
            weights_path = self.run_dir / e.path
            stem = weights_path.stem.replace(".safetensors", "")
            if stem.endswith("_state"):
                stem = stem[: -len("_state")]
            state_path = self.ckpt_dir / f"{stem}_state.pt"
            for p in (weights_path, state_path):
                try:
                    p.unlink()
                except FileNotFoundError:
                    pass

    def _write_manifest(self) -> None:
        tmp = self.manifest_path.with_suffix(".json.tmp")
        with open(tmp, "w") as f:
            json.dump(
                {
                    "run_id": self._manifest.run_id,
                    "config": self._manifest.config,
                    "checkpoints": [asdict(e) for e in self._manifest.checkpoints],
                    "latest_step": self._manifest.latest_step,
                    "latest_tokens_seen": self._manifest.latest_tokens_seen,
                    "latest_wall_time_s": self._manifest.latest_wall_time_s,
                },
                f,
                indent=2,
            )
        os.replace(tmp, self.manifest_path)

    def _load_or_init_manifest(self) -> Manifest:
        if self.manifest_path.exists():
            with open(self.manifest_path) as f:
                data = json.load(f)
            manifest = Manifest(
                run_id=data["run_id"],
                config=data["config"],
                checkpoints=[CheckpointEntry(**e) for e in data["checkpoints"]],
                latest_step=data["latest_step"],
                latest_tokens_seen=data["latest_tokens_seen"],
                latest_wall_time_s=data["latest_wall_time_s"],
            )
            if manifest.checkpoints:
                self._last_save_time = manifest.checkpoints[-1].created_at
            return manifest
        return Manifest(run_id=self.run_id, config=self.config)
