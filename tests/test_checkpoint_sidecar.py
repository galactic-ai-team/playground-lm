"""CheckpointManager writes self-describing weights; evals no longer need _state.pt (Claude)."""

import json

import pytest

torch = pytest.importorskip("torch")
pytest.importorskip("safetensors")

from evals.run_eval import split_checkpoint_config  # noqa: E402
from playground_lm.model import PlaygroundLM  # noqa: E402
from playground_lm.train.checkpoint import CheckpointManager  # noqa: E402
from playground_lm.train.sidecar import read_model_config_document, sidecar_path  # noqa: E402

MODEL = dict(vocab_size=64, n_layers=1, n_heads=2, d_model=8, context_len=16)
CONFIG = {"model": MODEL, "train": {"lr": 6e-4}, "stage": "C"}


def save_one(tmp_path, step=7, **kw):
    ckpt = CheckpointManager(tmp_path, run_id="plm-test", config=CONFIG, milestones=(), **kw)
    model = PlaygroundLM(**MODEL)
    opt = torch.optim.AdamW(model.parameters(), lr=1e-3)
    ckpt.save(model, opt, step=step, tokens_seen=step * 100, wall_time_s=1.5)
    return ckpt, tmp_path / "checkpoints" / f"step_{step:08d}.safetensors"


def test_save_writes_sidecar_and_header(tmp_path):
    _, weights = save_one(tmp_path)
    side = json.loads(sidecar_path(weights).read_text())
    assert side == {"format": 1, "run_id": "plm-test", "step": 7, "tokens_seen": 700,
                    "wall_time_s": 1.5, "config": CONFIG}
    from safetensors import safe_open

    with safe_open(str(weights), framework="pt") as f:
        assert json.loads(f.metadata()["playground_lm"]) == side


def test_eval_config_resolves_without_state_pt(tmp_path):
    _, weights = save_one(tmp_path)
    weights.with_name(weights.stem + "_state.pt").unlink()
    (tmp_path / "manifest.json").unlink()
    assert split_checkpoint_config(weights) == MODEL  # from the sidecar


def test_eval_config_resolves_from_the_weights_file_alone(tmp_path):
    _, weights = save_one(tmp_path)
    for extra in (weights.with_name(weights.stem + "_state.pt"), sidecar_path(weights),
                  tmp_path / "manifest.json"):
        extra.unlink()
    assert split_checkpoint_config(weights) == MODEL  # from the safetensors header
    assert read_model_config_document(weights)["step"] == 7
    # And the model actually loads with it.
    from safetensors.torch import load_file

    model = PlaygroundLM(**split_checkpoint_config(weights))
    model.load_state_dict(load_file(str(weights)))


def test_rotation_removes_sidecars_with_their_checkpoints(tmp_path):
    ckpt = CheckpointManager(tmp_path, run_id="plm-test", config=CONFIG, milestones=(), rolling_keep=2)
    model = PlaygroundLM(**MODEL)
    opt = torch.optim.AdamW(model.parameters(), lr=1e-3)
    for step in (1, 2, 3):
        ckpt.save(model, opt, step=step, tokens_seen=step, wall_time_s=0.0)
    names = sorted(p.name for p in (tmp_path / "checkpoints").iterdir())
    assert "step_00000001_model_config.json" not in names
    assert {"step_00000002_model_config.json", "step_00000003_model_config.json"} <= set(names)


def test_resume_is_unaffected(tmp_path):
    _, _ = save_one(tmp_path)
    model = PlaygroundLM(**MODEL)
    state = CheckpointManager(tmp_path, run_id="plm-test", config=CONFIG).resume(
        model, torch.optim.AdamW(model.parameters(), lr=1e-3))
    assert state["step"] == 7 and state["extra"] == {}