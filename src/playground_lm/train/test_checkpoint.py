# src/playground_lm/train/test_checkpoint.py
# Tests for CheckpointManager: save/load round-trip, resume restores
# step/optimizer/RNG, keep-last-N pruning (milestones never pruned), manifest
# atomicity.
#
# Uses a tiny real nn.Module + AdamW so torch.save/load and state_dict behavior
# are the actual production code, not a mock.

import json
import os
import tempfile
import time
from pathlib import Path

import pytest
import torch

from playground_lm.train.checkpoint import (
    CheckpointManager,
    DEFAULT_MILESTONES,
)


def _make_model_and_opt(d_model: int = 16):
    model = torch.nn.Linear(d_model, d_model)
    opt = torch.optim.AdamW(model.parameters(), lr=1e-3)
    return model, opt


def _train_step(model, opt, seed: int = 0):
    """Do one tiny optimizer step so optimizer state is non-trivial."""
    torch.manual_seed(seed)
    x = torch.randn(4, model.in_features)
    y = torch.randn(4, model.out_features)
    loss = ((model(x) - y) ** 2).mean()
    opt.zero_grad()
    loss.backward()
    opt.step()


CONFIG = {"d_model": 16, "context_len": 8, "vocab_size": 100}


@pytest.fixture
def run_dir():
    with tempfile.TemporaryDirectory() as tmp:
        yield Path(tmp)


def test_save_load_roundtrip(run_dir):
    """Saved weights load back into a fresh model with identical outputs."""
    model, opt = _make_model_and_opt()
    _train_step(model, opt)

    ckpt = CheckpointManager(run_dir, run_id="test", config=CONFIG)
    ckpt.save(model, opt, step=1, tokens_seen=32, wall_time_s=0.5)

    # Fresh model at random init should differ from saved.
    model2, opt2 = _make_model_and_opt()
    x = torch.randn(2, 16)
    out_before = model2(x)
    out_orig = model(x)
    assert not torch.allclose(out_before, out_orig), "pre-load weights should differ"

    # Now load into model2.
    resumed = ckpt.resume(model2, opt2)
    assert resumed is not None
    out_after = model2(x)
    assert torch.allclose(out_after, out_orig), "loaded weights should match saved"


def test_resume_restores_step_and_tokens(run_dir):
    """resume() returns the state dict with step and tokens_seen."""
    model, opt = _make_model_and_opt()
    _train_step(model, opt)

    ckpt = CheckpointManager(run_dir, run_id="test", config=CONFIG)
    ckpt.save(model, opt, step=42, tokens_seen=1_000_000, wall_time_s=12.3)

    model2, opt2 = _make_model_and_opt()
    state = ckpt.resume(model2, opt2)
    assert state is not None
    assert state["step"] == 42
    assert state["tokens_seen"] == 1_000_000
    assert state["wall_time_s"] == 12.3


def test_resume_restores_optimizer_state(run_dir):
    """Optimizer state_dict round-trips — step counter and moments preserved."""
    model, opt = _make_model_and_opt()
    # Run 3 steps so Adam moments are non-zero.
    for i in range(3):
        _train_step(model, opt, seed=i)

    ckpt = CheckpointManager(run_dir, run_id="test", config=CONFIG)
    ckpt.save(model, opt, step=3, tokens_seen=100, wall_time_s=1.0)

    model2, opt2 = _make_model_and_opt()
    ckpt.resume(model2, opt2)

    # Compare optimizer state dicts (deep check: step counters should match).
    orig_state = opt.state_dict()
    loaded_state = opt2.state_dict()
    assert orig_state["param_groups"] == loaded_state["param_groups"]
    # param_groups and state keys should match; individual tensors should be
    # equal for at least the 'step' scalars.
    for pid in orig_state["state"]:
        assert pid in loaded_state["state"]
        for k in orig_state["state"][pid]:
            a = orig_state["state"][pid][k]
            b = loaded_state["state"][pid][k]
            if isinstance(a, torch.Tensor):
                assert torch.equal(a, b), f"optimizer state {pid}.{k} mismatch"


def test_rng_state_roundtrip(run_dir):
    """Save when rng is in a known state; resume; next draw matches."""
    model, opt = _make_model_and_opt()

    # Save with a known rng state.
    torch.manual_seed(777)
    ckpt = CheckpointManager(run_dir, run_id="test", config=CONFIG)
    ckpt.save(model, opt, step=1, tokens_seen=32, wall_time_s=0.1)
    # Draw what the "next" values would be from this state.
    expected = torch.randn(5)

    # Fresh session: fill rng with garbage, then resume.
    torch.manual_seed(42)
    _ = torch.randn(100)  # advance rng
    model2, opt2 = _make_model_and_opt()
    ckpt.resume(model2, opt2)
    got = torch.randn(5)
    assert torch.equal(expected, got), "rng state was not restored on resume"


def test_keep_last_n_pruning(run_dir):
    """Rolling checkpoints beyond rolling_keep are pruned; milestones kept."""
    # Custom milestones so we control which saves get preserved.
    ckpt = CheckpointManager(
        run_dir, run_id="test", config=CONFIG,
        interval_s=0,                 # disable interval; we call save() directly
        milestones=(3,),              # step 3 is a milestone
        rolling_keep=2,
    )
    model, opt = _make_model_and_opt()

    for step in [1, 2, 3, 4, 5, 6]:
        ckpt.save(model, opt, step=step, tokens_seen=step * 100, wall_time_s=step * 0.1)

    steps = [e.step for e in ckpt.list_checkpoints()]
    # After 6 saves with rolling_keep=2 and milestone at 3:
    #   - 3 is a milestone → kept
    #   - last 2 rolling = 5, 6 → kept
    #   - 1, 2, 4 pruned
    assert 3 in steps
    assert 5 in steps
    assert 6 in steps
    assert 1 not in steps
    assert 2 not in steps
    assert 4 not in steps
    assert sorted(steps) == [3, 5, 6]


def test_milestone_never_pruned(run_dir):
    """Explicitly: a milestone checkpoint survives aggressive pruning."""
    ckpt = CheckpointManager(
        run_dir, run_id="test", config=CONFIG,
        interval_s=0,
        milestones=(2,),
        rolling_keep=1,
    )
    model, opt = _make_model_and_opt()
    for step in range(1, 11):
        ckpt.save(model, opt, step=step, tokens_seen=step * 100, wall_time_s=step * 0.1)

    steps = [e.step for e in ckpt.list_checkpoints()]
    assert 2 in steps, "milestone step 2 should never be pruned"
    # Latest is always kept.
    assert 10 in steps
    # rolling_keep=1 means at most 1 non-milestone rolling checkpoint besides
    # the latest (which is the most recent rolling).
    rolling = [s for s in steps if s != 2]
    # Either 9 alone, or 9+10 depending on whether 10 counts as rolling (it does).
    # With rolling_keep=1, only 10 should remain as rolling.
    assert rolling == [10], f"expected [10], got {rolling}"


def test_manifest_atomic_write(run_dir):
    """Manifest is written atomically — no partial JSON on disk."""
    ckpt = CheckpointManager(run_dir, run_id="test", config=CONFIG)
    model, opt = _make_model_and_opt()
    ckpt.save(model, opt, step=1, tokens_seen=100, wall_time_s=0.1)

    # Manifest should be valid JSON and reference the checkpoint.
    manifest_path = run_dir / "manifest.json"
    assert manifest_path.exists()
    with open(manifest_path) as f:
        data = json.load(f)
    assert data["run_id"] == "test"
    assert data["latest_step"] == 1
    assert len(data["checkpoints"]) == 1
    assert data["checkpoints"][0]["step"] == 1

    # No .tmp file left behind.
    assert not (run_dir / "manifest.json.tmp").exists()


def test_resume_on_empty_dir_returns_none(run_dir):
    """Fresh run dir → resume returns None, model untouched."""
    ckpt = CheckpointManager(run_dir, run_id="test", config=CONFIG)
    model, opt = _make_model_and_opt()
    before = {k: v.clone() for k, v in model.state_dict().items()}

    state = ckpt.resume(model, opt)
    assert state is None
    for k, v in model.state_dict().items():
        assert torch.equal(v, before[k]), f"model should not change on empty resume"


def test_should_save_by_interval(run_dir, monkeypatch):
    """should_save fires on interval, respects milestone steps."""
    ckpt = CheckpointManager(
        run_dir, run_id="test", config=CONFIG,
        interval_s=60,
        milestones=(10,),
        rolling_keep=2,
    )

    # Fake time so we control elapsed.
    fake_time = [1000.0]
    monkeypatch.setattr(time, "time", lambda: fake_time[0])

    # First call always saves (initial save bootstrap).
    assert ckpt.should_save(step=0) is True
    # Simulate a save so _last_save_time advances.
    ckpt._last_save_time = fake_time[0]

    # 30s later: below interval, not a milestone → False.
    fake_time[0] = 1030.0
    assert ckpt.should_save(step=1) is False

    # 60s later: at interval → True.
    fake_time[0] = 1060.0
    assert ckpt.should_save(step=2) is True

    # Milestone step always saves regardless of time.
    fake_time[0] = 1060.5  # still below next interval
    assert ckpt.should_save(step=10) is True


def test_resume_after_prune_uses_newest(run_dir):
    """After pruning, resume uses the latest retained checkpoint."""
    ckpt = CheckpointManager(
        run_dir, run_id="test", config=CONFIG,
        interval_s=0,
        milestones=(),
        rolling_keep=2,
    )
    model, opt = _make_model_and_opt()

    for step in [1, 2, 3, 4, 5]:
        _train_step(model, opt, seed=step)
        ckpt.save(model, opt, step=step, tokens_seen=step * 100, wall_time_s=step * 0.1)

    # Only 4, 5 should remain.
    steps = sorted(e.step for e in ckpt.list_checkpoints())
    assert steps == [4, 5]

    # Resume should land on 5.
    model2, opt2 = _make_model_and_opt()
    state = ckpt.resume(model2, opt2)
    assert state is not None
    assert state["step"] == 5
