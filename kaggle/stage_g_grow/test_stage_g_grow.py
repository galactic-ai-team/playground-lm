import importlib.util
import json
from pathlib import Path

HERE = Path(__file__).resolve().parent


def load():
    spec = importlib.util.spec_from_file_location("stage_g_grow", HERE / "stage_g_grow.py")
    m = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(m)
    return m


def test_grow_env_shape_and_budget():
    m = load()
    e = m.GROW_ENV
    assert (e["PLM_N_LAYERS"], e["PLM_N_HEADS"], e["PLM_D_MODEL"]) == ("24", "16", "1024")
    assert int(e["PLM_MAX_SESSION_S"]) <= 11 * 3600 and e["PLM_WEIGHTS_ONLY_RESUME"] == "1"
    assert int(e["PLM_UPLOAD_EVERY"]) > 0 and int(e["PLM_EFFECTIVE_BATCH"]) % int(e["PLM_BATCH_SIZE"]) == 0


def test_find_seed_runs(tmp_path, monkeypatch):
    m = load()
    monkeypatch.setattr(m, "INPUT", tmp_path)
    assert m.find_seed_runs() is None
    runs = tmp_path / "datasets" / "galacticaiteam" / m.SEED_SLUG / "runs"
    (runs / "checkpoints").mkdir(parents=True)
    (runs / "manifest.json").write_text(json.dumps({}))
    (runs / "checkpoints" / "step_00031101.safetensors").write_bytes(b"x")
    assert m.find_seed_runs() == runs


def test_kernel_metadata_self_references_for_resume():
    meta = json.loads((HERE / "kernel-metadata.json").read_text())
    assert meta["id"].endswith("/plm-stage-g-grow355")
    assert "galacticaiteam/plm-stage-g-grow355" in meta["kernel_sources"]
    assert meta["code_file"] == "stage_g_grow.py"
    assert "galacticaiteam/plm-g355-seed" in meta["dataset_sources"]
