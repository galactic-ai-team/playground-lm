# src/playground_lm/train/checkpoint.py
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
#     optimizer/training state (torch.save). Separating them lets the export
#     pipeline package weights for inference without pulling optimizer state.
#   - Manifest is written atomically (write to .tmp, then os.replace) so a
#     killed process doesn't leave a half-written manifest.
#   - All paths are relative to a single run directory so a Colab/Kaggle
#     session can be resumed just by re-mounting the same directory.
#   - extra_state (Claude): a small dict of extra training state saved in the
#     _state.pt file and returned by resume() under "extra". The training
#     loop uses it for the FP16 GradScaler, so a resumed T4 session continues
#     with the loss scale it had reached instead of restarting at 65536 and
#     skipping updates on overflow until it settles. Checkpoints written
#     before this existed resume with an empty "extra".
#   - Self-describing weights (Claude): every save also writes
#     step_XXXXXXXX_model_config.json beside the weights (the first place
#     evals/run_eval.py looks) and puts the same JSON in the .safetensors
#     header under "playground_lm". Evals and exports can then rebuild the
#     model from the weights file alone, without the ~1 GB _state.pt, which
#     only resume needs. The JSON holds this manager's config (which
#     contains the model architecture under "model"), run_id, step,
#     tokens_seen and wall_time_s.

import json
import os
import time
from dataclasses import dataclass, asdict, field
from pathlib import Path
from typing import Optional

import torch

# Sidecar helpers live in sidecar.py (torch-free) so upload and backfill
# tools can use them without importing torch. read_model_config_document is
# re-exported here for callers that already import this module.
from .sidecar import (
    SAFETENSORS_METADATA_KEY,
    model_config_document,
    read_model_config_document,  # noqa: F401
    sidecar_path,
    write_model_config_sidecar,
)


ROLLING_KEEP = 3

DEFAULT_MILESTONES = (1000, 5000, 10000, 25000, 50000, 100000, 200000)

DEFAULT_INTERVAL_S = int(os.environ.get("PLM_CKPT_INTERVAL_S", "1800"))


@dataclass
class CheckpointEntry:
    step: int
    path: str
    tokens_seen: int
    wall_time_s: float
    created_at: float
    is_milestone: bool


@dataclass
class Manifest:
    run_id: str
    config: dict
    checkpoints: list[CheckpointEntry] = field(default_factory=list)
    latest_step: int = -1
    latest_tokens_seen: int = 0
    latest_wall_time_s: float = 0.0


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

    def bootstrap(self) -> None:
        """Mark 'now' as the last-save time without writing a checkpoint.
        Useful for tests and for callers that want to defer the first save
        to a later interval."""
        self._last_save_time = time.time()

    def should_save(self, step: int) -> bool:
        """True if a checkpoint is due by time interval or by milestone step."""
        if step in self.milestones:
            return True
        if self._last_save_time == 0.0:
            return True
        return (time.time() - self._last_save_time) >= self.interval_s

    def save(
        self,
        model: torch.nn.Module,
        optimizer: torch.optim.Optimizer,
        step: int,
        tokens_seen: int,
        wall_time_s: float,
        extra_state: Optional[dict] = None,
    ) -> CheckpointEntry:
        stem = f"step_{step:08d}"
        weights_path = self.ckpt_dir / f"{stem}.safetensors"
        state_path = self.ckpt_dir / f"{stem}_state.pt"
        document = model_config_document(self.run_id, self.config, step, tokens_seen, wall_time_s)

        try:
            from safetensors.torch import save_file
            state_dict = {k: v.detach().cpu() for k, v in model.state_dict().items()}
            save_file(
                state_dict,
                str(weights_path),
                metadata={SAFETENSORS_METADATA_KEY: json.dumps(document)},
            )
        except ImportError:
            torch.save(model.state_dict(), str(weights_path).replace(".safetensors", ".pt"))
            weights_path = Path(str(weights_path).replace(".safetensors", ".pt"))
        write_model_config_sidecar(weights_path, document)

        torch.save(
            {
                "step": step,
                "tokens_seen": tokens_seen,
                "wall_time_s": wall_time_s,
                "optimizer": optimizer.state_dict(),
                "torch_rng_state": torch.get_rng_state(),
                "cuda_rng_state": torch.cuda.get_rng_state_all() if torch.cuda.is_available() else None,
                "config": self.config,
                "extra": dict(extra_state or {}),
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
        if not self._manifest.checkpoints:
            return None
        latest = self._manifest.checkpoints[-1]
        weights_path = self.run_dir / latest.path
        stem = weights_path.stem.replace(".safetensors", "")
        if stem.endswith("_state"):
            stem = stem[: -len("_state")]
        state_path = self.ckpt_dir / f"{stem}_state.pt"

        if not weights_path.exists() or not state_path.exists():
            raise FileNotFoundError(
                f"Manifest references missing files: {weights_path}, {state_path}"
            )

        if str(weights_path).endswith(".safetensors"):
            from safetensors.torch import load_file
            state_dict = load_file(str(weights_path))
        else:
            state_dict = torch.load(str(weights_path), map_location="cpu")
        model.load_state_dict(state_dict)

        state = torch.load(str(state_path), map_location="cpu")
        optimizer.load_state_dict(state["optimizer"])
        torch.set_rng_state(state["torch_rng_state"])
        if state.get("cuda_rng_state") is not None and torch.cuda.is_available():
            torch.cuda.set_rng_state_all(state["cuda_rng_state"])
        state.setdefault("extra", {})  # checkpoints from before extra_state existed
        return state

    def list_checkpoints(self) -> list[CheckpointEntry]:
        return list(self._manifest.checkpoints)

    # --- internals --------------------------------------------------------

    def _rotate(self) -> None:
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
            for p in (weights_path, state_path, sidecar_path(weights_path)):
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