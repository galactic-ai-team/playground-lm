"""Per-stage LR schedule: Stage C continuing from Stage B's checkpoint warms up again (Claude)."""

import json
import math

import pytest

torch = pytest.importorskip("torch")

from playground_lm.train.loop import _learning_rate, train  # noqa: E402

PEAK = 0.01
CONFIG = dict(batch_size=2, context_len=3, grad_accum_steps=1, lr=PEAK, warmup_steps=2, grad_clip=100.0)


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


class Manager:
    def __init__(self, run_dir):
        self.run_dir = run_dir

    def should_save(self, step):
        return False

    def save(self, *args, **kwargs):
        pass


def lrs(tmp_path, start_step, max_steps, **config):
    tmp_path.mkdir(parents=True, exist_ok=True)
    model = TinyModel()
    train(model=model, optimizer=torch.optim.SGD(model.parameters(), lr=PEAK),
          train_loader=Batches(), val_loader=Batches(), checkpoint_manager=Manager(tmp_path),
          config=dict(CONFIG, **config), device="cpu", start_step=start_step, max_steps=max_steps)
    rows = [json.loads(line) for line in (tmp_path / "metrics.jsonl").read_text().splitlines()]
    return {r["step"]: r["lr"] for r in rows}


def test_default_schedule_is_unchanged(tmp_path):
    got = lrs(tmp_path, start_step=0, max_steps=6)
    assert got == {s: pytest.approx(_learning_rate(s, PEAK, 2, 6)) for s in range(1, 7)}


def test_new_stage_warms_up_from_its_own_start(tmp_path):
    # "Stage B" ended at step 3; "Stage C" runs steps 4..7 with warmup 2.
    got = lrs(tmp_path, start_step=3, max_steps=7, schedule_start_step=3)
    assert got[4] == pytest.approx(PEAK * 1 / 2)  # warming up again, not ~PEAK
    assert got[5] == pytest.approx(PEAK)
    assert got[6] == pytest.approx(PEAK * (0.1 + 0.45 * (1 + math.cos(math.pi * 0.5))))
    assert got[7] == pytest.approx(PEAK * 0.1)  # decays to 10% at the stage's max_steps


def test_without_it_a_resumed_stage_jumps_straight_to_peak(tmp_path):
    # The problem this fixes: past the warmup step count, step 4 gets ~peak LR.
    got = lrs(tmp_path, start_step=3, max_steps=7)
    assert got[4] == pytest.approx(_learning_rate(4, PEAK, 2, 7))
    assert got[4] > 0.5 * PEAK


def test_later_sessions_of_the_stage_continue_its_schedule(tmp_path):
    whole = lrs(tmp_path / "whole", start_step=3, max_steps=7, schedule_start_step=3)
    second_half = lrs(tmp_path / "resumed", start_step=5, max_steps=7, schedule_start_step=3)
    assert second_half == {s: whole[s] for s in (6, 7)}


@pytest.mark.parametrize("bad", [-1, 4, 2.5, "3"])
def test_rejects_bad_schedule_start(tmp_path, bad):
    with pytest.raises(ValueError):
        lrs(tmp_path, start_step=3, max_steps=7, schedule_start_step=bad)


def test_rejects_schedule_start_at_max_steps(tmp_path):
    with pytest.raises(ValueError):
        lrs(tmp_path, start_step=7, max_steps=7, schedule_start_step=7)
