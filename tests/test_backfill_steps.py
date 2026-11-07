"""backfill_model_configs --steps fails loudly on steps the manifest doesn't list (Claude)."""

import json
import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "scripts"))
import backfill_model_configs  # noqa: E402

CONFIG = {"model": {"vocab_size": 50257, "n_layers": 12, "n_heads": 12, "d_model": 768, "context_len": 1024}}


def _manifest(tmp_path, steps):
    path = tmp_path / "manifest.json"
    path.write_text(json.dumps({
        "run_id": "stage-c", "config": CONFIG,
        "checkpoints": [{"step": s, "path": f"checkpoints/step_{s:08d}.safetensors",
                         "tokens_seen": s * 98_304, "wall_time_s": 0.0} for s in steps]}))
    return path


def test_requested_step_missing_from_manifest_is_an_error(tmp_path):
    manifest = _manifest(tmp_path, [10907, 19366])
    with pytest.raises(ValueError, match=r"\[12345\]"):
        backfill_model_configs.backfill(manifest, tmp_path / "out", steps={19366, 12345})
    assert not (tmp_path / "out").exists()  # nothing half-written


def test_listed_steps_still_work(tmp_path):
    manifest = _manifest(tmp_path, [10907, 19366])
    written = backfill_model_configs.backfill(manifest, tmp_path / "out", steps={10907})
    assert [p.name for p in written] == ["step_00010907_model_config.json"]
    assert json.loads(written[0].read_text())["tokens_seen"] == 10907 * 98_304
