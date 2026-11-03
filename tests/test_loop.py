import json

import pytest

torch = pytest.importorskip("torch")

from playground_lm.train.loop import train


class TinyModel(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.weight = torch.nn.Parameter(torch.tensor(0.5))
        self.bias = torch.nn.Parameter(torch.tensor(0.0))
        self.calls = []

    def forward(self, x, y):
        assert x.dtype == y.dtype == torch.int64
        assert x.device.type == y.device.type == "cpu"
        self.calls.append((self.training, torch.is_grad_enabled()))
        logits = self.weight * x.float() + self.bias
        assert logits.dtype == torch.float32
        return logits, ((logits - y.float()) ** 2).mean()


class Batches:
    def __init__(self, interrupt_at=None):
        self.count = 0
        self.interrupt_at = interrupt_at

    def __next__(self):
        if self.count == self.interrupt_at:
            raise KeyboardInterrupt
        self.count += 1
        return torch.ones(2, 3, dtype=torch.int64), torch.zeros(2, 3, dtype=torch.int64)


class FakeCheckpointManager:
    def __init__(self, run_dir, due=()):
        self.run_dir, self.due = run_dir, due
        self.saved, self.checked = [], []

    def should_save(self, step):
        self.checked.append(step)
        return step in self.due

    def save(self, model, optimizer, step, tokens_seen, wall_time_s):
        self.saved.append((model, optimizer, step, tokens_seen, wall_time_s))


class RecordingAdamW(torch.optim.AdamW):
    def __init__(self, model, batches):
        super().__init__([{"params": [model.weight]}, {"params": [model.bias]}],
                         lr=1.0, weight_decay=0.0)
        self.batches, self.updates = batches, []
        self.gradients = []

    def step(self, closure=None):
        self.updates.append((self.batches.count,
                             [group["lr"] for group in self.param_groups]))
        self.gradients.append([p.grad.item() for group in self.param_groups
                               for p in group["params"]])
        return super().step(closure)


@pytest.fixture
def setup_run(tmp_path):
    model, batches = TinyModel(), Batches()
    optimizer = RecordingAdamW(model, batches)
    ckpt = FakeCheckpointManager(tmp_path)
    config = dict(batch_size=2, context_len=3, grad_accum_steps=3,
                  lr=0.01, warmup_steps=2, grad_clip=100.0)
    return dict(model=model, optimizer=optimizer, train_loader=batches,
                val_loader=Batches(), checkpoint_manager=ckpt, config=config,
                device="cpu", max_steps=6)


def read_metrics(run):
    path = run["checkpoint_manager"].run_dir / "metrics.jsonl"
    return [json.loads(line) for line in path.read_text().splitlines()]


def test_warmup_cosine_accumulation_and_metrics(setup_run):
    run = setup_run
    train(**run)
    updates = run["optimizer"].updates
    assert [count for count, _ in updates] == [3, 6, 9, 12, 15, 18]
    lrs = [groups[0] for _, groups in updates]
    assert lrs[:2] == pytest.approx([0.005, 0.01])
    assert lrs[2:] == sorted(lrs[2:], reverse=True)
    assert lrs[3] == pytest.approx(0.0055)
    assert lrs[-1] == pytest.approx(0.001)
    assert all(groups[0] == groups[1] for _, groups in updates)
    # Each micro-batch has gradient 1 for both parameters; averaging preserves it.
    assert run["optimizer"].gradients[0] == pytest.approx([1.0, 1.0])
    rows = read_metrics(run)
    assert [row["step"] for row in rows] == list(range(1, 7))
    assert rows[0]["loss"] == pytest.approx(0.25)
    assert all(row["tokens_per_sec"] > 0 for row in rows)
    assert [row["tokens_seen"] for row in rows] == [18, 36, 54, 72, 90, 108]
    assert all(row["wall_time_s"] >= 0 for row in rows)
    assert run["val_loader"].count == 0


def test_clipping(setup_run):
    run = setup_run
    run["config"]["grad_clip"] = 0.1
    train(**{**run, "max_steps": 1})
    gradient = torch.tensor(run["optimizer"].gradients[0])
    assert gradient.norm().item() == pytest.approx(0.1, abs=1e-6)


@pytest.mark.parametrize("due", [(2,), (2, 6)])
def test_saves_and_callbacks_without_duplicate_final(setup_run, due):
    run, callbacks = setup_run, []
    ckpt = run["checkpoint_manager"]
    ckpt.due = due

    def on_save(step, count):
        assert ckpt.saved[-1][2] == step
        callbacks.append((step, count))

    train(**run, on_save=on_save)
    assert callbacks == [(2, 1), (6, 2)]
    assert ckpt.checked == list(range(1, 7))
    assert [saved[3] for saved in ckpt.saved] == [36, 108]
    assert all(saved[4] >= 0 for saved in ckpt.saved)


def test_resume_and_append(setup_run):
    run = setup_run
    path = run["checkpoint_manager"].run_dir / "metrics.jsonl"
    path.write_text('{"step": 3}\n')
    train(**run, start_step=3)
    assert [row["step"] for row in read_metrics(run)] == [3, 4, 5, 6]
    assert run["optimizer"].updates[0][1] == pytest.approx([0.0055] * 2)
    assert run["train_loader"].count == 9
    assert run["checkpoint_manager"].saved[-1][3] == 54


def test_resume_counters_in_saves_and_metrics(setup_run, monkeypatch):
    run = setup_run
    ckpt = run["checkpoint_manager"]
    ckpt.due = (11,)
    clock = iter(float(value) for value in range(20))
    monkeypatch.setattr("playground_lm.train.loop.time.perf_counter",
                        lambda: next(clock))

    train(**{**run, "start_step": 10, "max_steps": 12},
          start_tokens_seen=12345, start_wall_time_s=100.0)

    assert [saved[2] for saved in ckpt.saved] == [11, 12]
    assert [saved[3] for saved in ckpt.saved] == [12363, 12381]
    assert [saved[4] for saved in ckpt.saved] == [104.0, 108.0]
    rows = read_metrics(run)
    assert [row["step"] for row in rows] == [11, 12]
    assert [row["tokens_seen"] for row in rows] == [12363, 12381]
    assert [row["wall_time_s"] for row in rows] == [103.0, 107.0]
    assert [row["tokens_per_sec"] for row in rows] == [18.0, 18.0]


@pytest.mark.parametrize("interrupt_at, completed", [(1, 0), (4, 1)])
def test_resume_interrupt_preserves_counters(setup_run, interrupt_at, completed):
    run = setup_run
    run["train_loader"].interrupt_at = interrupt_at

    with pytest.raises(KeyboardInterrupt):
        train(**{**run, "start_step": 10, "max_steps": 12},
              start_tokens_seen=12345, start_wall_time_s=100.0)

    saved = run["checkpoint_manager"].saved[-1]
    assert saved[2] == 10 + completed
    assert saved[3] == 12345 + completed * 18
    assert saved[4] >= 100.0
    rows = read_metrics(run)
    assert len(rows) == completed
    if completed:
        assert rows[-1]["tokens_seen"] == saved[3]
        assert 100.0 <= rows[-1]["wall_time_s"] <= saved[4]


def test_resume_already_complete_preserves_counters(setup_run):
    run = setup_run
    train(**{**run, "start_step": 10, "max_steps": 10},
          start_tokens_seen=12345, start_wall_time_s=100.0)

    assert run["train_loader"].count == 0
    assert read_metrics(run) == []
    saved = run["checkpoint_manager"].saved[-1]
    assert saved[2] == 10
    assert saved[3] == 12345
    assert saved[4] >= 100.0


def test_validation_averages_20_batches_and_restores_train(setup_run):
    run = setup_run
    train(**{**run, "start_step": 499, "max_steps": 500})
    model = run["model"]
    assert model.calls == [(True, True)] * 3 + [(False, False)] * 20
    assert model.training
    assert run["val_loader"].count == 20
    expected = (model.weight + model.bias).detach().square().item()
    assert read_metrics(run)[0]["val_loss"] == pytest.approx(expected)


@pytest.mark.parametrize("interrupt_at, completed", [(1, 0), (4, 1)])
def test_interrupt_saves_completed_update_and_reraises(setup_run, interrupt_at, completed):
    run, callbacks = setup_run, []
    run["train_loader"].interrupt_at = interrupt_at
    with pytest.raises(KeyboardInterrupt):
        train(**run, on_save=lambda *args: callbacks.append(args))
    assert callbacks == [(completed, 1)]
    assert run["checkpoint_manager"].saved[-1][3] == completed * 18
    assert len(run["optimizer"].updates) == completed
    assert all(p.grad is None for p in run["model"].parameters())


def test_validation_interrupt_restores_mode_and_saves(setup_run):
    run = setup_run
    run["val_loader"].interrupt_at = 2
    with pytest.raises(KeyboardInterrupt):
        train(**{**run, "start_step": 499, "max_steps": 500})
    assert run["model"].training
    assert run["checkpoint_manager"].saved[-1][2] == 500


def test_compiled_wrapper_is_unwrapped_for_save(setup_run):
    class Wrapper(torch.nn.Module):
        def __init__(self, original):
            super().__init__()
            self._orig_mod = original

        def forward(self, x, y):
            return self._orig_mod(x, y)

    run = setup_run
    original = run["model"]
    train(**{**run, "model": Wrapper(original), "max_steps": 1})
    assert run["checkpoint_manager"].saved[-1][0] is original


@pytest.mark.parametrize("warmup", [0, 6, 10])
def test_schedule_boundaries(setup_run, warmup):
    run = setup_run
    run["config"]["warmup_steps"] = warmup
    train(**run)
    last_lr = run["optimizer"].updates[-1][1][0]
    expected = 0.001 if warmup == 0 else 0.01 * 6 / warmup
    assert last_lr == pytest.approx(expected)


def test_already_complete_still_saves(setup_run):
    run = setup_run
    train(**run, start_step=6)
    assert run["train_loader"].count == 0
    assert run["checkpoint_manager"].saved[-1][2] == 6