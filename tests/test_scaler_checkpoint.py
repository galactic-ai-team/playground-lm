"""GradScaler state survives a checkpoint (Claude)."""

import json

import pytest

torch = pytest.importorskip("torch")
pytest.importorskip("safetensors")

from playground_lm.train import loop as loop_mod  # noqa: E402
from playground_lm.train.checkpoint import CheckpointManager  # noqa: E402
from playground_lm.train.loop import train  # noqa: E402

SCALER_STATE = {"scale": 2048.0, "growth_factor": 2.0, "backoff_factor": 0.5,
                "growth_interval": 2000, "_growth_tracker": 137}


def make_manager(tmp_path):
    return CheckpointManager(tmp_path, run_id="test", config={"lr": 0.01}, milestones=())


def tiny_model_and_opt():
    model = torch.nn.Linear(4, 4)
    return model, torch.optim.AdamW(model.parameters(), lr=1e-3)


# ---- CheckpointManager ------------------------------------------------------------

def test_extra_state_round_trips(tmp_path):
    ckpt = make_manager(tmp_path)
    model, opt = tiny_model_and_opt()
    ckpt.save(model, opt, step=7, tokens_seen=700, wall_time_s=1.5,
              extra_state={"grad_scaler": dict(SCALER_STATE)})
    resumed = CheckpointManager(tmp_path, run_id="test", config={"lr": 0.01}).resume(*tiny_model_and_opt())
    assert resumed["step"] == 7
    assert resumed["extra"] == {"grad_scaler": SCALER_STATE}


def test_save_without_extra_state_still_works(tmp_path):
    ckpt = make_manager(tmp_path)
    model, opt = tiny_model_and_opt()
    ckpt.save(model, opt, step=1, tokens_seen=10, wall_time_s=0.1)
    assert ckpt.resume(*tiny_model_and_opt())["extra"] == {}


def test_old_checkpoints_without_extra_resume_with_empty_extra(tmp_path):
    ckpt = make_manager(tmp_path)
    model, opt = tiny_model_and_opt()
    ckpt.save(model, opt, step=3, tokens_seen=30, wall_time_s=0.3)
    state_path = tmp_path / "checkpoints" / "step_00000003_state.pt"
    state = torch.load(state_path)
    del state["extra"]  # what a checkpoint from before this change looks like
    torch.save(state, state_path)
    assert ckpt.resume(*tiny_model_and_opt())["extra"] == {}


# ---- training loop ----------------------------------------------------------------

class FakeScaler:
    """Stands in for torch.amp.GradScaler on CPU (a real one is disabled without CUDA)."""

    instances = []

    def __init__(self):
        self.state = {"scale": 65536.0, "_growth_tracker": 0}
        self.loaded = None
        FakeScaler.instances.append(self)

    def scale(self, loss):
        return loss

    def unscale_(self, optimizer):
        pass

    def step(self, optimizer):
        optimizer.step()

    def update(self):
        self.state["_growth_tracker"] += 1

    def get_scale(self):
        return self.state["scale"]

    def state_dict(self):
        return dict(self.state)

    def load_state_dict(self, state):
        self.loaded = dict(state)
        self.state = dict(state)


class TinyModel(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.weight = torch.nn.Parameter(torch.tensor(0.5))

    def forward(self, x, y):
        logits = self.weight * x.float()
        return logits, ((logits - y.float()) ** 2).mean()


class Batches:
    def __next__(self):
        return torch.ones(2, 3, dtype=torch.int64), torch.zeros(2, 3, dtype=torch.int64)


CONFIG = dict(batch_size=2, context_len=3, grad_accum_steps=1, lr=0.01, warmup_steps=0, grad_clip=1.0)


@pytest.fixture
def fake_fp16(monkeypatch):
    FakeScaler.instances.clear()
    monkeypatch.setattr(loop_mod, "_pick_amp_dtype", lambda device, requested: torch.float16)
    monkeypatch.setattr(loop_mod, "_make_grad_scaler", FakeScaler)
    real_autocast = torch.autocast
    monkeypatch.setattr(loop_mod.torch, "autocast", lambda **kw: real_autocast("cpu", enabled=False))


def run_train(tmp_path, max_steps, **kw):
    model = TinyModel()
    train(model=model, optimizer=torch.optim.SGD(model.parameters(), lr=0.01),
          train_loader=Batches(), val_loader=Batches(),
          checkpoint_manager=make_manager(tmp_path), config=dict(CONFIG),
          device="cpu", max_steps=max_steps, **kw)


def test_loop_saves_scaler_state_and_restores_it_on_resume(tmp_path, fake_fp16):
    run_train(tmp_path, max_steps=3)
    resumed = make_manager(tmp_path).resume(TinyModel(), torch.optim.SGD(TinyModel().parameters(), lr=0.01))
    saved = resumed["extra"]["grad_scaler"]
    assert saved == {"scale": 65536.0, "_growth_tracker": 3}

    run_train(tmp_path, max_steps=5, start_step=3, start_scaler_state=saved)
    second = FakeScaler.instances[-1]
    assert second.loaded == saved  # the new session's scaler picked up where the old one stopped
    final = make_manager(tmp_path).resume(TinyModel(), torch.optim.SGD(TinyModel().parameters(), lr=0.01))
    assert final["step"] == 5
    assert final["extra"]["grad_scaler"]["_growth_tracker"] == 5


def test_scaler_state_is_ignored_without_fp16(tmp_path):
    # e.g. a T4 checkpoint resumed on a bf16 GPU, or on CPU: no scaler, no error.
    run_train(tmp_path, max_steps=2, start_scaler_state=dict(SCALER_STATE))
    resumed = make_manager(tmp_path).resume(TinyModel(), torch.optim.SGD(TinyModel().parameters(), lr=0.01))
    assert resumed["extra"] == {}
    rows = [json.loads(line) for line in (tmp_path / "metrics.jsonl").read_text().splitlines()]
    assert [r["step"] for r in rows] == [1, 2]