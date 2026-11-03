"""Health metrics in metrics.jsonl: grad norm, fp16 loss scale, overflow retries, peak GPU memory (Claude)."""

import json

import pytest

torch = pytest.importorskip("torch")

from playground_lm.train import loop as loop_mod  # noqa: E402
from playground_lm.train.loop import train  # noqa: E402

CONFIG = dict(batch_size=2, context_len=3, grad_accum_steps=1, lr=0.01, warmup_steps=0, grad_clip=100.0)


class TinyModel(torch.nn.Module):
    """loss = w**2 for the constant batches below, so d(loss)/dw = 2w."""

    def __init__(self):
        super().__init__()
        self.weight = torch.nn.Parameter(torch.tensor(0.5))

    def forward(self, x, y):
        logits = self.weight * x.float()
        return logits, ((logits - y.float()) ** 2).mean()


class Batches:
    def __init__(self):
        self.count = 0

    def __next__(self):
        self.count += 1
        return torch.ones(2, 3, dtype=torch.int64), torch.zeros(2, 3, dtype=torch.int64)


class Manager:
    def __init__(self, run_dir):
        self.run_dir = run_dir

    def should_save(self, step):
        return False

    def save(self, *args, **kwargs):
        pass


def run(tmp_path, max_steps, **config):
    tmp_path.mkdir(parents=True, exist_ok=True)
    model = TinyModel()
    train(model=model, optimizer=torch.optim.SGD(model.parameters(), lr=0.01),
          train_loader=Batches(), val_loader=Batches(), checkpoint_manager=Manager(tmp_path),
          config=dict(CONFIG, **config), device="cpu", max_steps=max_steps)
    return [json.loads(line) for line in (tmp_path / "metrics.jsonl").read_text().splitlines()]


def test_grad_norm_is_logged_before_clipping(tmp_path):
    rows = run(tmp_path, max_steps=2, grad_clip=0.1)
    assert rows[0]["grad_norm"] == pytest.approx(1.0)  # 2w at w = 0.5, not the clipped 0.1
    assert all("grad_norm" in r for r in rows)


def test_cpu_rows_have_no_gpu_only_fields(tmp_path):
    rows = run(tmp_path, max_steps=2)
    for r in rows:
        assert "loss_scale" not in r and "overflow_retries" not in r and "peak_mem_gib" not in r


class ScriptedScaler:
    """Fake GradScaler: overflows on the attempts listed, halving the scale each time."""

    def __init__(self, overflow_attempts=(1, 2)):
        self.scale_value = 65536.0
        self.attempt = 0
        self.overflow_attempts = set(overflow_attempts)

    def scale(self, loss):
        return loss

    def unscale_(self, optimizer):
        self.attempt += 1

    def step(self, optimizer):
        if self.attempt not in self.overflow_attempts:
            optimizer.step()

    def update(self):
        if self.attempt in self.overflow_attempts:
            self.scale_value /= 2

    def get_scale(self):
        return self.scale_value

    def state_dict(self):
        return {"scale": self.scale_value}


def test_fp16_rows_log_loss_scale_and_overflow_retries(tmp_path, monkeypatch):
    monkeypatch.setattr(loop_mod, "_pick_amp_dtype", lambda device, requested: torch.float16)
    monkeypatch.setattr(loop_mod, "_make_grad_scaler", ScriptedScaler)
    real_autocast = torch.autocast
    monkeypatch.setattr(loop_mod.torch, "autocast", lambda **kw: real_autocast("cpu", enabled=False))
    rows = run(tmp_path, max_steps=2)
    assert rows[0]["overflow_retries"] == 2  # attempts 1 and 2 overflowed, attempt 3 applied
    assert rows[0]["loss_scale"] == 16384.0  # halved twice
    assert "overflow_retries" not in rows[1]
    assert rows[1]["loss_scale"] == 16384.0


def test_peak_memory_field_and_lr_schedule_untouched(tmp_path, monkeypatch):
    plain = run(tmp_path / "plain", max_steps=3)
    monkeypatch.setattr(loop_mod, "_peak_mem_gib", lambda device: 1.5)
    rows = run(tmp_path / "gpu", max_steps=3)
    assert [r["peak_mem_gib"] for r in rows] == [1.5, 1.5, 1.5]
    # Logging memory must not disturb anything else (e.g. the LR schedule).
    assert [r["lr"] for r in rows] == [r["lr"] for r in plain]


def test_peak_mem_helper(monkeypatch):
    assert loop_mod._peak_mem_gib(torch.device("cpu")) is None
    monkeypatch.setattr(loop_mod.torch.cuda, "max_memory_allocated", lambda device=None: 3 * 2**30)
    assert loop_mod._peak_mem_gib(torch.device("cuda")) == 3.0
