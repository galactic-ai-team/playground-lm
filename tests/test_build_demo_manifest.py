"""Deterministic, offline tests for the demo manifest builder."""

from __future__ import annotations

import hashlib
import json
import math
import sys

import pytest

from scripts import build_demo_manifest as builder


def full_report():
    return {
        "validation": {
            "loss": 2.0,
            "perplexity": math.exp(2.0),
            "complete": True,
            "tokens_scored": 100,
            "available_targets": 100,
        },
        "benchmarks": {
            "hellaswag": {
                "accuracy": 0.3,
                "examples": 10042,
                "complete": True,
            },
            "lambada": {
                "last_word_accuracy": 0.12,
                "examples": 5153,
                "complete": True,
            },
        },
    }


def checkpoint(step, tokens):
    return {"step": step, "tokens_seen": tokens,
            "url": f"https://assets.example/step_{step}/model.int8.onnx"}


def test_sorted_checkpoints_rows_baseline_and_riddles():
    manifest = builder.build_manifest(
        [checkpoint(200, 2000), checkpoint(50, 500)],
        {200: full_report(), 50: full_report()},
        full_report(),
        ["What has keys but no locks?"],
    )
    assert [c["step"] for c in manifest["checkpoints"]] == [50, 200]
    rows = manifest["benchmarks"]["rows"]
    assert [r["step"] for r in rows[:2]] == [50, 200]
    assert [r["label"] for r in rows] == [
        "PlaygroundLM Step 50", "PlaygroundLM Step 200",
        "OpenAI GPT-2 124M",
    ]
    assert rows[0]["hellaswag_acc_norm"] == 0.3
    assert rows[0]["lambada_acc"] == 0.12
    assert "step" not in rows[-1]
    assert manifest["benchmarks"]["columns"] == list(builder.COLUMNS)
    assert manifest["riddle_prompts"] == ["What has keys but no locks?"]


def test_missing_reports_leave_checkpoint_without_fabricated_row():
    manifest = builder.build_manifest([checkpoint(30, 900)], {})
    assert len(manifest["checkpoints"]) == 1
    assert manifest["benchmarks"]["rows"] == []


def test_incomplete_benchmark_and_partial_report_rejected(tmp_path):
    report = full_report()
    report["benchmarks"]["hellaswag"]["complete"] = False
    with pytest.raises(ValueError, match="incomplete"):
        builder.build_manifest([checkpoint(10, 100)], {10: report})

    path = tmp_path / "partial.json"
    path.write_text(json.dumps({"publishable": False, **full_report()}))
    with pytest.raises(ValueError, match="Incomplete"):
        builder.read_report_pairs([f"10={path}"])


def test_split_reports_combine_by_step_and_conflicts_rejected(tmp_path):
    full = full_report()
    val = tmp_path / "validation.json"
    bench = tmp_path / "benchmark.json"
    val.write_text(json.dumps({"validation": full["validation"]}))
    bench.write_text(json.dumps({"benchmarks": full["benchmarks"]}))
    merged = builder.read_report_pairs([f"12={bench}", f"12={val}"])
    assert merged[12] == full
    manifest = builder.build_manifest([checkpoint(12, 100)], merged)
    assert manifest["benchmarks"]["rows"][0]["step"] == 12

    bad = tmp_path / "bad.json"
    bad.write_text(json.dumps({"validation": {**full["validation"], "loss": 5}}))
    with pytest.raises(ValueError, match="Conflicting"):
        builder.read_report_pairs([f"12={val}", f"12={bad}"])


def test_duplicate_steps_and_bad_urls_rejected():
    with pytest.raises(ValueError, match="Duplicate"):
        builder.build_manifest([checkpoint(10, 1), checkpoint(10, 2)], {})
    with pytest.raises(ValueError, match="missing checkpoint"):
        builder.build_manifest([checkpoint(10, 1)], {20: full_report()})
    with pytest.raises(ValueError, match="INT8"):
        builder.build_manifest(
            [{"step": 1, "tokens_seen": 10, "url": "model.onnx"}], {}
        )


def test_cli_writes_exact_assets_and_deterministic_manifest(tmp_path, monkeypatch):
    output = tmp_path / "demo" / "manifest.json"
    output.parent.mkdir()
    output.write_text(json.dumps({"riddle_prompts": ["Riddle one"]}))
    checkpoints = tmp_path / "checkpoints.json"
    checkpoints.write_text(json.dumps([checkpoint(200, 2000), checkpoint(50, 500)]))
    report_path = tmp_path / "report.json"
    report_path.write_text(json.dumps(full_report()))
    baseline_path = tmp_path / "gpt2.json"
    baseline_path.write_text(json.dumps(full_report()))

    encoder = b'{"<|endoftext|>": 50256}'
    merges = b'#version: 0.2\na b\n'
    monkeypatch.setattr(builder, "load_gpt2_assets", lambda: (encoder, merges))
    monkeypatch.setattr(sys, "argv", [
        "build_demo_manifest", "--checkpoints", str(checkpoints),
        "--report", f"50={report_path}", "--report", f"200={report_path}",
        "--baseline", str(baseline_path), "--output", str(output),
    ])

    builder.main()
    first = output.read_bytes()
    assert (output.parent / "encoder.json").read_bytes() == encoder
    assert (output.parent / "merges.txt").read_bytes() == merges
    manifest = json.loads(first)
    assert [r["step"] for r in manifest["benchmarks"]["rows"][:-1]] == [50, 200]
    assert manifest["riddle_prompts"] == ["Riddle one"]
    assert manifest["benchmarks"]["rows"][-1]["label"] == "OpenAI GPT-2 124M"

    builder.main()
    assert output.read_bytes() == first


def test_tokenizer_reads_exact_verified_source_assets(monkeypatch):
    checked = []
    fake_encoder = {f"token_{i}": i for i in range(builder.VOCAB_SIZE - 1)}
    fake_encoder["<|endoftext|>"] = builder.EOT_ID
    encoder = json.dumps(fake_encoder).encode()
    merges = b"#version: 0.2\na b\n"
    source = {
        "encoder.json": encoder,
        "vocab.bpe": merges,
    }
    original_assets = builder.ASSETS
    assets = {
        url: hashlib.sha256(source[url.rsplit("/", 1)[-1]]).hexdigest()
        for url in original_assets
    }
    monkeypatch.setattr(builder, "ASSETS", assets)
    monkeypatch.setattr(builder, "get_tokenizer", lambda: checked.append("initialized"))
    import tiktoken.load
    def fake_cached(url, expected_hash):
        checked.append((url, expected_hash))
        return source[url.rsplit("/", 1)[-1]]
    monkeypatch.setattr(tiktoken.load, "read_file_cached", fake_cached)
    result = builder.load_gpt2_assets()
    assert result == (encoder, merges)
    assert checked[0] == "initialized"
    assert len(checked) == 3
