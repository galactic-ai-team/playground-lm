"""Multi-GPU (DDP) training loop, checked on CPU with 2 gloo processes (Claude).

Two processes with one micro-batch schedule each must end with exactly the
weights a single process gets when it accumulates both processes' batches.
"""

import json
import multiprocessing
import os
import sys
from pathlib import Path

import pytest

torch = pytest.importorskip("torch")
import torch.distributed as dist  # noqa: E402

pytestmark = pytest.mark.skipif(
    not dist.is_available() or not dist.is_gloo_available() or sys.platform == "win32",
    reason="needs torch.distributed with the gloo backend",
)

WORLD = 2
ACCUM = 2  # per process; exercises DDP's no_sync on the first micro-batch
STEPS = 4
CONFIG = dict(batch_size=2, context_len=3, lr=0.05, warmup_steps=1, grad_clip=100.0)


class Net(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.lin = torch.nn.Linear(3, 1)

    def forward(self, x, y):
        out = self.lin(x.float())
        return out, ((out - y[:, :1].float()) ** 2).mean()


def batch(step, rank, micro):
    g = torch.Generator().manual_seed(1000 * step + 10 * rank + micro)
    return torch.randint(0, 5, (2, 3), generator=g), torch.randint(0, 5, (2, 3), generator=g)


class ListLoader:
    def __init__(self, items):
        self.items = iter(items)

    def __next__(self):
        return next(self.items)


class RecordingManager:
    def __init__(self, run_dir):
        self.run_dir = Path(run_dir)
        self.saved = []

    def should_save(self, step):
        return step == 2

    def save(self, model, optimizer, step, tokens_seen, wall_time_s, **kw):
        assert not isinstance(model, torch.nn.parallel.DistributedDataParallel)
        self.saved.append((step, tokens_seen))


def _worker(rank, tmpdir, session_s):
    from playground_lm.train.distributed import wrap, DistInfo
    from playground_lm.train.loop import train

    dist.init_process_group("gloo", init_method=f"file://{tmpdir}/pg", rank=rank, world_size=WORLD)
    try:
        torch.manual_seed(0)
        model = Net()
        ddp = wrap(model, DistInfo(rank=rank, local_rank=rank, world_size=WORLD))
        loader = ListLoader([batch(s, rank, m) for s in range(STEPS * 10) for m in range(ACCUM)])
        manager = RecordingManager(Path(tmpdir) / f"run{rank}")
        config = dict(CONFIG, grad_accum_steps=ACCUM)
        if session_s is not None:
            config["max_session_s"] = session_s
        train(model=ddp, optimizer=torch.optim.SGD(model.parameters(), lr=0.05),
              train_loader=loader, val_loader=ListLoader([]),
              checkpoint_manager=manager, config=config, device="cpu", max_steps=STEPS)
        torch.save({"state": model.state_dict(), "saved": manager.saved}, Path(tmpdir) / f"rank{rank}.pt")
    finally:
        dist.destroy_process_group()


def run_two_processes(tmpdir, session_s=None):
    ctx = multiprocessing.get_context("spawn")
    procs = [ctx.Process(target=_worker, args=(r, str(tmpdir), session_s)) for r in range(WORLD)]
    for p in procs:
        p.start()
    for p in procs:
        p.join(timeout=180)
    for p in procs:
        if p.is_alive():
            p.kill()
            pytest.fail("a DDP worker hung (collectives out of step?)")
        assert p.exitcode == 0
    return [torch.load(Path(tmpdir) / f"rank{r}.pt") for r in range(WORLD)]


def single_process_reference(tmp_path):
    from playground_lm.train.loop import train

    torch.manual_seed(0)
    model = Net()
    # One process accumulating both ranks' micro-batches for each step.
    items = [batch(s, r, m) for s in range(STEPS * 10) for r in range(WORLD) for m in range(ACCUM)]
    manager = RecordingManager(tmp_path / "single")
    train(model=model, optimizer=torch.optim.SGD(model.parameters(), lr=0.05),
          train_loader=ListLoader(items), val_loader=ListLoader([]),
          checkpoint_manager=manager, config=dict(CONFIG, grad_accum_steps=ACCUM * WORLD),
          device="cpu", max_steps=STEPS)
    return model, manager


def test_two_processes_match_one_process_accumulating_both(tmp_path):
    results = run_two_processes(tmp_path)
    reference, ref_manager = single_process_reference(tmp_path)

    for result in results:  # both ranks hold identical weights...
        for k, v in reference.state_dict().items():
            torch.testing.assert_close(result["state"][k], v, rtol=0, atol=1e-6)  # ...equal to one process

    # Only rank 0 saves (step 2 by schedule, then the final step), with global token counts.
    tokens_per_step = CONFIG["batch_size"] * ACCUM * CONFIG["context_len"] * WORLD
    assert results[0]["saved"] == [(2, 2 * tokens_per_step), (STEPS, STEPS * tokens_per_step)]
    assert results[1]["saved"] == []
    assert ref_manager.saved == results[0]["saved"]  # same counters as the single process

    # Only rank 0 writes metrics; logged loss is the mean over both GPUs.
    assert not (tmp_path / "run1" / "metrics.jsonl").exists()
    ddp_rows = [json.loads(line) for line in (tmp_path / "run0" / "metrics.jsonl").read_text().splitlines()]
    ref_rows = [json.loads(line) for line in (tmp_path / "single" / "metrics.jsonl").read_text().splitlines()]
    assert [r["step"] for r in ddp_rows] == list(range(1, STEPS + 1))
    assert [r["tokens_seen"] for r in ddp_rows] == [r["tokens_seen"] for r in ref_rows]
    for a, b in zip(ddp_rows, ref_rows):
        assert a["loss"] == pytest.approx(b["loss"], abs=1e-6)


def test_session_budget_stops_every_rank_on_the_same_update(tmp_path):
    results = run_two_processes(tmp_path, session_s=0)  # out of time after the first update
    tokens_per_step = CONFIG["batch_size"] * ACCUM * CONFIG["context_len"] * WORLD
    assert results[0]["saved"] == [(1, tokens_per_step)]
    assert results[1]["saved"] == []


def test_helpers_are_no_ops_without_torchrun(monkeypatch):
    from playground_lm.train import distributed as d

    monkeypatch.delenv("WORLD_SIZE", raising=False)
    info = d.setup()
    assert not info.enabled and info.is_main
    model = Net()
    assert d.wrap(model, info) is model
    assert d.unwrap(model) is model
    d.barrier(info)  # no process group: nothing to wait for