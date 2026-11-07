"""Model-config sidecars without torch: helpers, release upload, backfill (Claude)."""

import json
import sys
from pathlib import Path

import pytest

from playground_lm.train import upload_release as ur
from playground_lm.train.sidecar import (
    model_config_document,
    read_model_config_document,
    sidecar_path,
    write_model_config_sidecar,
)

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "scripts"))
import backfill_model_configs  # noqa: E402

MODEL = {"vocab_size": 50257, "n_layers": 12, "n_heads": 12, "d_model": 768, "context_len": 1024}
CONFIG = {"model": MODEL, "train": {"lr": 6e-4}}


def test_sidecar_name_matches_what_run_eval_looks_for(tmp_path):
    weights = tmp_path / "step_00019366.safetensors"
    assert sidecar_path(weights) == tmp_path / "step_00019366_model_config.json"
    assert sidecar_path(weights) == weights.with_name(weights.stem + "_model_config.json")


def test_write_and_read_round_trip(tmp_path):
    weights = tmp_path / "step_00000010.safetensors"
    doc = model_config_document("plm-001", CONFIG, 10, 983_040, 12.5)
    write_model_config_sidecar(weights, doc)
    assert read_model_config_document(weights) == doc
    assert doc["config"]["model"] == MODEL and doc["format"] == 1
    assert not list(tmp_path.glob("*.tmp"))  # atomic write leaves no temp file


def test_read_returns_none_when_there_is_nothing(tmp_path):
    assert read_model_config_document(tmp_path / "step_00000001.safetensors") is None


# ---- release upload ---------------------------------------------------------------

def _run_dir(tmp_path, steps, with_sidecars):
    ckpt = tmp_path / "checkpoints"
    ckpt.mkdir(parents=True)
    entries = []
    for step in steps:
        name = f"step_{step:08d}.safetensors"
        (ckpt / name).write_bytes(b"weights")
        if with_sidecars:
            write_model_config_sidecar(ckpt / name, model_config_document("t", CONFIG, step, step, 0.0))
        entries.append({"step": step, "path": f"checkpoints/{name}", "tokens_seen": step,
                        "wall_time_s": 0.0, "created_at": 0.0, "is_milestone": False})
    (tmp_path / "manifest.json").write_text(json.dumps({
        "run_id": "t", "config": CONFIG, "checkpoints": entries,
        "latest_step": max(steps), "latest_tokens_seen": 0, "latest_wall_time_s": 0.0}))
    return tmp_path


@pytest.fixture
def fake_gh(monkeypatch):
    uploaded = []
    state = {"present": set()}
    monkeypatch.setenv("GH_TOKEN", "test-token")  # upload() requires one; gh itself is faked below
    monkeypatch.setattr(ur, "_ensure_release", lambda repo, tag: False)
    monkeypatch.setattr(ur, "_existing_assets", lambda repo, tag: set(state["present"]))
    monkeypatch.setattr(ur, "_upload_one", lambda repo, tag, path: uploaded.append(Path(path).name))
    return uploaded, state


def test_upload_sends_sidecars_with_new_weights(tmp_path, fake_gh):
    uploaded, _ = fake_gh
    run = _run_dir(tmp_path, [5, 10], with_sidecars=True)
    ur.upload(run, "o/r", "tag")
    assert set(uploaded) == {"manifest.json",
                             "step_00000005.safetensors", "step_00000005_model_config.json",
                             "step_00000010.safetensors", "step_00000010_model_config.json"}


def test_upload_backfills_sidecars_for_weights_already_released(tmp_path, fake_gh):
    uploaded, state = fake_gh
    run = _run_dir(tmp_path, [5], with_sidecars=True)
    state["present"] = {"step_00000005.safetensors"}
    result = ur.upload(run, "o/r", "tag")
    assert "step_00000005_model_config.json" in uploaded
    assert "step_00000005.safetensors" not in uploaded
    assert "step_00000005.safetensors" in result.skipped


def test_old_runs_without_sidecars_upload_exactly_as_before(tmp_path, fake_gh):
    uploaded, _ = fake_gh
    run = _run_dir(tmp_path, [5], with_sidecars=False)
    result = ur.upload(run, "o/r", "tag")
    assert uploaded == ["manifest.json", "step_00000005.safetensors"]
    assert result.uploaded == ["step_00000005.safetensors"]


# ---- backfill ---------------------------------------------------------------------

def test_backfill_writes_one_sidecar_per_checkpoint(tmp_path):
    run = _run_dir(tmp_path / "run", [10907, 19366], with_sidecars=False)
    out = tmp_path / "sidecars"
    backfill_model_configs.main(["--manifest", str(run / "manifest.json"), "--out", str(out)])
    for step in (10907, 19366):
        doc = json.loads((out / f"step_{step:08d}_model_config.json").read_text())
        assert doc["step"] == step and doc["config"]["model"] == MODEL


def test_backfill_can_pick_steps_and_refuses_a_manifest_without_a_model(tmp_path):
    run = _run_dir(tmp_path / "run", [10907, 19366], with_sidecars=False)
    out = tmp_path / "sidecars"
    backfill_model_configs.main(["--manifest", str(run / "manifest.json"), "--out", str(out), "--steps", "19366"])
    assert [p.name for p in out.iterdir()] == ["step_00019366_model_config.json"]
    bare = tmp_path / "bare.json"
    bare.write_text(json.dumps({"run_id": "x", "config": {}, "checkpoints": []}))
    with pytest.raises(ValueError):
        backfill_model_configs.backfill(bare, out)