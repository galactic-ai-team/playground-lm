"""Unit tests for scripts/plot_loss.py."""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from scripts.plot_loss import (
    compute_summary,
    downsample_points,
    generate_loss_artifacts,
    load_and_merge_metrics,
    main,
)


def _write_jsonl(path: Path, lines: list[str]) -> Path:
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")
    return path


def test_dedupe_on_repeated_steps_keeps_last_and_sorts(tmp_path: Path) -> None:
    file1 = _write_jsonl(
        tmp_path / "session1.jsonl",
        [
            json.dumps({"step": 0, "loss": 10.8, "tokens_seen": 0, "amp_dtype": "float16"}),
            json.dumps({"step": 100, "loss": 6.5, "tokens_seen": 52428800}),
            json.dumps({"step": 200, "loss": 5.2, "tokens_seen": 104857600, "val_loss": 5.3}),
        ],
    )
    # Session 2 resumes from step 100 with updated loss and adds step 300
    file2 = _write_jsonl(
        tmp_path / "session2.jsonl",
        [
            json.dumps({"step": 300, "loss": 4.1, "tokens_seen": 157286400}),
            json.dumps({"step": 100, "loss": 6.1, "tokens_seen": 52428800}),
            json.dumps({"step": 200, "loss": 4.9, "tokens_seen": 104857600, "val_loss": 5.0}),
        ],
    )

    rows, warnings = load_and_merge_metrics([file1, file2])
    assert warnings == 0
    assert [r["step"] for r in rows] == [0, 100, 200, 300]
    # Step 100 and 200 must reflect the later occurrence in session2
    assert rows[1]["loss"] == pytest.approx(6.1)
    assert rows[2]["loss"] == pytest.approx(4.9)
    assert rows[2]["val_loss"] == pytest.approx(5.0)


def test_malformed_and_blank_lines_skipped_with_warning_count(
    tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    log_file = _write_jsonl(
        tmp_path / "dirty.jsonl",
        [
            "",
            "   ",
            "{not valid json",
            json.dumps(["a", "list", "not", "dict"]),
            json.dumps({"step": "bad_step", "loss": 4.0}),
            json.dumps({"step": 10, "loss": "bad_loss"}),
            json.dumps({"step": 20, "loss": 4.25, "tokens_seen": 1000}),
        ],
    )

    rows, warnings = load_and_merge_metrics([log_file])
    assert warnings == 4
    assert len(rows) == 1
    assert rows[0]["step"] == 20
    assert rows[0]["loss"] == pytest.approx(4.25)

    captured = capsys.readouterr()
    assert "WARNING: skipped 4" in captured.err


def test_downsampling_always_keeps_val_rows_and_last_row() -> None:
    val_steps = {10, 25, 50, 75, 90}
    rows = []
    for step in range(1, 101):
        row = {
            "step": step,
            "tokens_seen": step * 1000,
            "loss": 10.0 / (step + 1),
        }
        if step in val_steps:
            row["val_loss"] = 9.5 / (step + 1)
        rows.append(row)

    downsampled = downsample_points(rows, max_points=12)
    assert len(downsampled) <= 12

    kept_steps = {pt["step"] for pt in downsampled}
    # Every val_loss step and the final step (100) must be preserved
    assert val_steps.issubset(kept_steps)
    assert 100 in kept_steps
    # Steps must remain strictly ascending
    assert [pt["step"] for pt in downsampled] == sorted(kept_steps)


def test_summary_values_correct() -> None:
    rows = [
        {"step": 0, "tokens_seen": 0, "loss": 10.9},
        {"step": 100, "tokens_seen": 50_000_000, "loss": 5.4, "val_loss": 5.5},
        {"step": 200, "tokens_seen": 100_000_000, "loss": 4.2, "val_loss": 4.15},
        {"step": 300, "tokens_seen": 150_000_000, "loss": 3.9, "val_loss": 4.22},
        {"step": 350, "tokens_seen": 175_000_000, "loss": 3.75},
    ]

    summary = compute_summary(rows)
    assert summary == {
        "final_step": 350,
        "final_train_loss": pytest.approx(3.75),
        "best_val_loss": pytest.approx(4.15),
        "best_val_step": 200,
        "tokens_seen": 175_000_000,
    }


def test_cli_writes_png_and_json_with_gpt2_baseline(tmp_path: Path) -> None:
    metrics_path = _write_jsonl(
        tmp_path / "metrics.jsonl",
        [
            json.dumps(
                {
                    "step": i * 10,
                    "loss": 8.0 - i * 0.15,
                    "tokens_seen": i * 5_000_000,
                    **(
                        {"val_loss": 7.9 - i * 0.14}
                        if i % 5 == 0 and i > 0
                        else {}
                    ),
                }
            )
            for i in range(25)
        ],
    )
    out_dir = tmp_path / "reports"

    rc = main(
        [
            str(metrics_path),
            "--out-dir",
            str(out_dir),
            "--smooth",
            "0.85",
            "--max-points",
            "10",
            "--gpt2-baseline",
            "3.28",
        ]
    )
    assert rc == 0

    png_path = out_dir / "loss.png"
    json_path = out_dir / "loss_curve.json"

    assert png_path.is_file()
    assert png_path.stat().st_size > 1000
    # Verify PNG magic header bytes
    assert png_path.read_bytes()[:8] == b"\x89PNG\r\n\x1a\n"

    assert json_path.is_file()
    payload = json.loads(json_path.read_text(encoding="utf-8"))
    assert "points" in payload and "summary" in payload
    assert len(payload["points"]) == 10
    assert payload["points"][-1]["step"] == 240
    assert payload["summary"]["final_step"] == 240
    assert payload["summary"]["best_val_step"] == 200
