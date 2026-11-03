"""Multi-GPU helpers for PlaygroundLM (Claude).

Data-parallel training with torch DistributedDataParallel (DDP), one process
per GPU, launched by torchrun. On Kaggle's 2 x T4 that's:

    !torchrun --standalone --nproc_per_node=2 notebooks/playground_lm_train.py

Without torchrun (WORLD_SIZE unset or 1) every helper here is a no-op, so the
single-GPU and CPU paths are exactly what they were.

Why DDP and not nn.DataParallel: DataParallel runs both GPUs from one Python
process, re-copies the model to the second GPU on every forward and gathers
outputs on GPU 0. For our model that means shipping the full 1.2 GB logits
tensor between GPUs every micro-batch. DDP keeps one model per process and
only all-reduces gradients, once per optimizer step (see no_sync in the loop).

What each process does differently (the rest of the script is identical):
  * device: cuda:LOCAL_RANK;
  * data: its own train-loader seed (seed_offset + rank), so the GPUs see
    different batches;
  * only rank 0 writes metrics, saves checkpoints and uploads releases.
The training loop handles the last two; the notebook passes the seed.

Timeout: while rank 0 saves a checkpoint (and, every few saves, uploads it
as a GitHub release), the other GPU waits at the next gradient all-reduce.
NCCL's default wait limit is 10 minutes, which a large upload can exceed,
so setup() uses 60 minutes (PLM_DIST_TIMEOUT_MIN to change it).
"""

from __future__ import annotations

import os

from dataclasses import dataclass
from datetime import timedelta

import torch
import torch.distributed as dist
from torch.nn.parallel import DistributedDataParallel


@dataclass(frozen=True)
class DistInfo:
    rank: int = 0
    local_rank: int = 0
    world_size: int = 1

    @property
    def enabled(self) -> bool:
        return self.world_size > 1

    @property
    def is_main(self) -> bool:
        return self.rank == 0


def setup(backend: str | None = None, timeout_min: float | None = None) -> DistInfo:
    """Join the process group torchrun set up; single process if there is none.

    Reads RANK, LOCAL_RANK and WORLD_SIZE (set by torchrun). Uses NCCL on GPUs
    and gloo on CPU (tests). timeout_min (default: PLM_DIST_TIMEOUT_MIN or 60)
    is how long a rank waits for the others in any collective.
    """
    world = int(os.environ.get("WORLD_SIZE", "1"))
    if world <= 1:

        return DistInfo()
    rank = int(os.environ["RANK"])
    local_rank = int(os.environ.get("LOCAL_RANK", rank))
    if backend is None:
        backend = "nccl" if torch.cuda.is_available() else "gloo"
    if backend == "nccl":
        torch.cuda.set_device(local_rank)
    if timeout_min is None:
        timeout_min = float(os.environ.get("PLM_DIST_TIMEOUT_MIN", "60"))
    if not dist.is_initialized():
        dist.init_process_group(backend=backend, timeout=timedelta(minutes=timeout_min))
    return DistInfo(rank=rank, local_rank=local_rank, world_size=world)


def device_for(info: DistInfo) -> torch.device:
    if torch.cuda.is_available():
        return torch.device("cuda", info.local_rank)
    return torch.device("cpu")


def wrap(model: torch.nn.Module, info: DistInfo) -> torch.nn.Module:
    """DDP-wrap a model already on its device; unchanged when not distributed.

    Wrap after loading a checkpoint and before torch.compile (if used).
    DDP broadcasts rank 0's weights to the other ranks at construction, so
    all GPUs start from identical weights.
    """
    if not info.enabled:
        return model
    ids = [info.local_rank] if torch.cuda.is_available() else None
    return DistributedDataParallel(model, device_ids=ids)



def unwrap(model: torch.nn.Module) -> torch.nn.Module:
    """The plain model inside DDP and/or torch.compile wrappers (any order)."""
    while True:
        if isinstance(model, DistributedDataParallel):
            model = model.module
        elif hasattr(model, "_orig_mod"):
            model = model._orig_mod
        else:
            return model


def barrier(info: DistInfo) -> None:
    if info.enabled:
        dist.barrier()


def cleanup() -> None:
    if dist.is_available() and dist.is_initialized():
        dist.destroy_process_group()
