"""Offline tests for staging -> demo ONNX manifest references."""

from __future__ import annotations

import hashlib
import json

import pytest

from scripts import publish_onnx_manifest as publisher


def setup_staging(tmp_path):
    staging = tmp_path / "hf_staging"
    staging.mkdir()
    (staging / "step_02035_q4.onnx").write_bytes(b"q4 model")
    (staging / "step_19366_q8.onnx").write_bytes(b"q8 model")
    manifest = tmp_path / "manifest.json"
    original = {
        "model": "PlaygroundLM-124M",
        "checkpoints": [],
        "benchmarks": {"columns": ["val_loss", "perplexity"], "rows": [{"label": "GPT-2", "val_loss": 3.279}]},
        "riddle_prompts": ["Keep this riddle"],
        "extra": {"preserved": True},
    }
    manifest.write_text(json.dumps(original), encoding="utf-8")
    return staging, manifest, original


def entries():
    return [
        {"step": 19366, "tokens_seen": 1900000000, "filename": "step_19366_q8.onnx", "quant": "q8"},
        {"step": 2035, "tokens_seen": 200000000, "filename": "step_02035_q4.onnx", "quant": "q4"},
    ]


def test_rows_sorted_urls_and_real_hashes(tmp_path):
    staging, _, _ = setup_staging(tmp_path)
    rows = publisher.build_rows(staging, entries(), repo="galactic-ai-team/plm-onnx", revision="abc123")
    assert [r["step"] for r in rows] == [2035, 19366]
    assert rows[0]["quant"] == "q4" and rows[1]["quant"] == "q8"
    assert rows[1]["url"] == "https://huggingface.co/galactic-ai-team/plm-onnx/resolve/abc123/step_19366_q8.onnx"
    assert rows[0]["onnx_url"] == rows[0]["url"]
    assert rows[0]["sha256"] == hashlib.sha256(b"q4 model").hexdigest()


def test_preserves_benchmarks_riddles_and_metadata(tmp_path):
    staging, _, original = setup_staging(tmp_path)
    rows = publisher.build_rows(staging, entries(), repo="galactic-ai-team/plm-onnx")
    result = publisher.update_manifest(original, rows)
    assert result["checkpoints"] == rows
    assert result["benchmarks"] == original["benchmarks"]
    assert result["riddle_prompts"] == original["riddle_prompts"]
    assert result["extra"] == original["extra"]
    assert original["checkpoints"] == []


def test_cli_dry_run_is_read_only(tmp_path, capsys):
    staging, manifest, _ = setup_staging(tmp_path)
    index = tmp_path / "index.json"
    index.write_text(json.dumps({"checkpoints": entries()}), encoding="utf-8")
    before = manifest.read_bytes()
    publisher.main(["--staging-dir", str(staging), "--index", str(index),
                    "--repo", "team/models", "--output", str(manifest), "--dry-run"])
    output = json.loads(capsys.readouterr().out)
    assert len(output["checkpoints"]) == 2
    assert manifest.read_bytes() == before


def test_cli_output_is_deterministic(tmp_path):
    staging, manifest, _ = setup_staging(tmp_path)
    argv = ["--staging-dir", str(staging), "--repo", "team/models", "--output", str(manifest),
            "--checkpoint", "2035", "200000000", "step_02035_q4.onnx", "q4",
            "--checkpoint", "19366", "1900000000", "step_19366_q8.onnx", "q8"]
    publisher.main(argv)
    first = manifest.read_bytes()
    publisher.main(argv)
    assert manifest.read_bytes() == first


@pytest.mark.parametrize("bad", [
    {"step": 1, "tokens_seen": 1, "filename": "../outside.onnx", "quant": "q4"},
    {"step": 1, "tokens_seen": 1, "filename": "/tmp/outside.onnx", "quant": "q4"},
    {"step": 1, "tokens_seen": 1, "filename": "step_02035_q4.onnx", "quant": "q3"},
    {"step": 1, "tokens_seen": -1, "filename": "step_02035_q4.onnx", "quant": "q4"},
    {"step": 1, "tokens_seen": 1, "filename": "missing.onnx", "quant": "q4"},
])
def test_invalid_metadata_rejected(tmp_path, bad):
    staging, _, _ = setup_staging(tmp_path)
    with pytest.raises((ValueError, FileNotFoundError)):
        publisher.build_rows(staging, [bad], repo="team/models")


def test_duplicate_step_rejected(tmp_path):
    staging, _, _ = setup_staging(tmp_path)
    with pytest.raises(ValueError, match="Duplicate"):
        publisher.build_rows(staging, entries() + [entries()[0]], repo="team/models")


def test_index_parsing(tmp_path):
    index = tmp_path / "index.json"
    index.write_text(json.dumps({"checkpoints": entries()}), encoding="utf-8")
    result = publisher.load_index(index, [])
    assert [e["step"] for e in result] == [2035, 19366]


def test_url_template_and_repo_validated(tmp_path):
    staging, _, _ = setup_staging(tmp_path)
    with pytest.raises(ValueError):
        publisher.build_rows(staging, entries(), repo="https://evil.test")
    with pytest.raises(ValueError):
        publisher.build_rows(staging, entries(), repo="team/repo", url_template="http://host/{repo}/{filename}")
    rows = publisher.build_rows(staging, entries(), repo="team/repo",
        url_template="https://huggingface.co/{repo}/resolve/main/{filename}")
    assert "/resolve/main/" in rows[0]["url"]
