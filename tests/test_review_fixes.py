"""Tests for Claude's review fixes to model.py and train/loop.py."""

import json

import pytest

torch = pytest.importorskip("torch")
from torch.nn import functional as F  # noqa: E402

import playground_lm.model as model_mod  # noqa: E402
from playground_lm.model import PlaygroundLM, chunked_cross_entropy  # noqa: E402
from playground_lm.train import loop as loop_mod  # noqa: E402
from playground_lm.train.loop import train  # noqa: E402

TINY = dict(n_layers=2, n_heads=2, d_model=32, context_len=16, vocab_size=100)


# ---- chunked loss ---------------------------------------------------------------

@pytest.mark.parametrize("chunk", [1, 7, 32, 4096])
def test_chunked_loss_matches_plain_loss_and_gradients(chunk):
    torch.manual_seed(0)
    logits = torch.randn(3, 11, 100, requires_grad=True)
    targets = torch.randint(100, (3, 11))
    plain = F.cross_entropy(logits.reshape(-1, 100), targets.reshape(-1))
    (g_plain,) = torch.autograd.grad(plain, logits)
    chunked = chunked_cross_entropy(logits, targets, chunk)
    (g_chunked,) = torch.autograd.grad(chunked, logits)
    torch.testing.assert_close(chunked, plain)
    torch.testing.assert_close(g_chunked, g_plain)


def test_model_loss_is_unchanged_by_chunking(monkeypatch):
    torch.manual_seed(0)
    model = PlaygroundLM(**TINY)
    x = torch.randint(100, (2, 16))
    y = torch.randint(100, (2, 16))
    logits, whole = model(x, y)
    whole.backward()
    grads = [p.grad.clone() for p in model.parameters()]
    model.zero_grad(set_to_none=True)
    monkeypatch.setattr(model_mod, "LOSS_CHUNK_TOKENS", 5)  # 32 rows -> 7 chunks
    _, chunked = model(x, y)
    chunked.backward()
    torch.testing.assert_close(chunked, whole)
    for p, g in zip(model.parameters(), grads):
        torch.testing.assert_close(p.grad, g)
    torch.testing.assert_close(whole, F.cross_entropy(logits.reshape(-1, 100), y.flatten()))


def test_chunked_loss_without_grad_and_under_autocast():
    torch.manual_seed(0)
    model = PlaygroundLM(**TINY)
    x = torch.randint(100, (2, 8))
    y = torch.randint(100, (2, 8))
    with torch.no_grad():
        _, loss = model(x, y)
    assert torch.isfinite(loss)
    with torch.autocast("cpu", dtype=torch.bfloat16):
        logits, loss = model(x, y)
    assert logits.dtype == torch.bfloat16 and loss.dtype == torch.float32
    loss.backward()
    assert all(p.grad is not None and torch.isfinite(p.grad).all() for p in model.parameters())


# ---- mixed precision choice --------------------------------------------------------

@pytest.mark.parametrize("capability, expected", [((6, 0), torch.float16), ((7, 5), torch.float16),
                                                  ((8, 0), torch.bfloat16), ((8, 9), torch.bfloat16)])
def test_amp_dtype_follows_compute_capability(monkeypatch, capability, expected):
    # T4 = 7.5 and P100 = 6.0 report emulated bf16 as "supported"; they must get fp16.
    monkeypatch.setattr(torch.cuda, "get_device_capability", lambda device=None: capability)
    monkeypatch.setattr(torch.cuda, "is_bf16_supported", lambda *a, **k: True)
    assert loop_mod._pick_amp_dtype(torch.device("cuda"), "auto") == expected


def test_amp_dtype_overrides_and_cpu():
    assert loop_mod._pick_amp_dtype(torch.device("cpu"), "auto") is None
    assert loop_mod._pick_amp_dtype(torch.device("cuda"), "fp16") == torch.float16
    assert loop_mod._pick_amp_dtype(torch.device("cuda"), "bf16") == torch.bfloat16
    assert loop_mod._pick_amp_dtype(torch.device("cuda"), "fp32") is None
    with pytest.raises(ValueError):
        loop_mod._pick_amp_dtype(torch.device("cuda"), "fp8")


# ---- loop behaviour ---------------------------------------------------------------

class TinyModel(torch.nn.Module):
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


class FakeCheckpointManager:
    def __init__(self, run_dir):
        self.run_dir, self.saved = run_dir, []

    def should_save(self, step):
        return False

    def save(self, model, optimizer, step, tokens_seen, wall_time_s):
        self.saved.append(step)


def make_run(tmp_path, **config):
    model = TinyModel()
    base = dict(batch_size=2, context_len=3, grad_accum_steps=1, lr=0.01, warmup_steps=0, grad_clip=1.0)
    base.update(config)
    return dict(model=model, optimizer=torch.optim.SGD(model.parameters(), lr=0.01),
                train_loader=Batches(), val_loader=Batches(),
                checkpoint_manager=FakeCheckpointManager(tmp_path), config=base, device="cpu")


def test_validation_reuses_the_same_batches(tmp_path):
    run = make_run(tmp_path, eval_interval=2, eval_batches=3)
    train(**run, max_steps=6)
    assert run["val_loader"].count == 3  # read once, reused at steps 4 and 6
    rows = [json.loads(line) for line in (tmp_path / "metrics.jsonl").read_text().splitlines()]
    assert [r["step"] for r in rows if "val_loss" in r] == [2, 4, 6]
    assert rows[0]["amp_dtype"] == "fp32"


def test_session_budget_saves_and_stops(tmp_path):
    run = make_run(tmp_path, max_session_s=0)
    train(**run, max_steps=100)
    assert run["checkpoint_manager"].saved == [1]


def test_persistent_fp16_overflow_stops_instead_of_looping(tmp_path, monkeypatch):
    class AlwaysOverflowScaler:
        def __init__(self):
            self.scale_value = 65536.0

        def scale(self, loss):
            return loss

        def unscale_(self, optimizer):
            pass

        def step(self, optimizer):
            pass  # an overflowed step is skipped

        def update(self):
            self.scale_value /= 2

        def get_scale(self):
            return self.scale_value

    monkeypatch.setattr(loop_mod, "_pick_amp_dtype", lambda device, requested: torch.float16)
    monkeypatch.setattr(loop_mod, "_make_grad_scaler", AlwaysOverflowScaler)
    real_autocast = torch.autocast
    monkeypatch.setattr(loop_mod.torch, "autocast", lambda **kw: real_autocast("cpu", enabled=False))
    run = make_run(tmp_path, max_overflow_retries=5)
    with pytest.raises(FloatingPointError):
        train(**run, max_steps=3)
    assert run["train_loader"].count == 6  # 1 try + 5 retries, then stop
