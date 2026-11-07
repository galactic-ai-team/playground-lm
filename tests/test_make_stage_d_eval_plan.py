"""CPU-only tests for Stage D evaluation planning.

No downloads, model weights, or GPU required.
"""

from __future__ import annotations

import json
import sys

import pytest

from scripts.make_stage_d_eval_plan import (
    count_uint16_tokens,
    make_plan,
    main,
    should_early_stop,
)


def test_counts_actual_uint16_training_tokens(tmp_path):
    path = tmp_path / "train.bin"
    path.write_bytes(b"\x01\x00" * 18000)

    assert count_uint16_tokens(path) == 18000


def test_rejects_misaligned_training_file(tmp_path):
    path = tmp_path / "train.bin"
    path.write_bytes(b"\x01\x00\x02")

    with pytest.raises(ValueError, match="uint16-aligned"):
        count_uint16_tokens(path)


def test_rejects_empty_training_file(tmp_path):
    path = tmp_path / "train.bin"
    path.write_bytes(b"")

    with pytest.raises(ValueError, match="empty"):
        count_uint16_tokens(path)


def test_epoch_estimate_uses_ceiling():
    plan = make_plan(
        train_tokens=18000,
        target_train_tokens=1000000,
    )

    training = plan["training"]

    assert training["suggested_epochs"] == 56
    assert training["estimated_tokens_processed"] == 1008000
    assert training["estimated_target_overshoot_tokens"] == 8000


def test_benchmark_schedule_includes_final_epoch():
    plan = make_plan(
        train_tokens=18000,
        target_train_tokens=1000000,
        benchmark_evals=10,
    )

    evaluation = plan["evaluation"]

    assert evaluation["hellaswag"]["every_epochs"] == 6
    assert evaluation["hellaswag"]["epochs"] == [
        6, 12, 18, 24, 30, 36, 42, 48, 54, 56
    ]
    assert evaluation["lambada"]["epochs"] == (
        evaluation["hellaswag"]["epochs"]
    )
    assert evaluation["verbatim_copy"]["epochs"] == (
        evaluation["hellaswag"]["epochs"]
    )


def test_validation_runs_every_epoch_by_default():
    plan = make_plan(
        train_tokens=1000,
        target_train_tokens=4000,
    )

    assert plan["evaluation"]["validation"]["epochs"] == [
        1, 2, 3, 4
    ]


def test_short_run_schedules_every_epoch():
    plan = make_plan(
        train_tokens=1000,
        target_train_tokens=2000,
        benchmark_evals=10,
    )

    assert plan["training"]["suggested_epochs"] == 2
    assert plan["evaluation"]["hellaswag"]["epochs"] == [
        1, 2
    ]


def test_consecutive_validation_rises_trigger_stop():
    assert should_early_stop(
        [1.0, 1.1, 1.2, 1.3],
        patience=3,
    )


def test_nonconsecutive_rises_do_not_trigger_stop():
    assert not should_early_stop(
        [1.0, 1.2, 1.1, 1.2],
        patience=3,
    )


def test_plateau_resets_early_stop_counter():
    assert not should_early_stop(
        [1.0, 1.1, 1.1, 1.2, 1.3],
        patience=3,
    )


def test_min_delta_ignores_small_rises():
    assert not should_early_stop(
        [1.0, 1.001, 1.002, 1.003],
        patience=3,
        min_delta=0.01,
    )


@pytest.mark.parametrize(
    "kwargs",
    [
        {"train_tokens": 0, "target_train_tokens": 100},
        {"train_tokens": 10, "target_train_tokens": 0},
        {
            "train_tokens": 10,
            "target_train_tokens": 100,
            "benchmark_evals": 0,
        },
        {
            "train_tokens": 10,
            "target_train_tokens": 100,
            "early_stop_patience": 0,
        },
        {
            "train_tokens": 10,
            "target_train_tokens": 100,
            "min_delta": float("nan"),
        },
    ],
)
def test_invalid_plan_inputs_rejected(kwargs):
    with pytest.raises(ValueError):
        make_plan(**kwargs)


def test_cli_reads_training_file_and_writes_json(
    tmp_path, monkeypatch, capsys
):
    train_bin = tmp_path / "train.bin"
    train_bin.write_bytes(b"\x01\x00" * 125)

    output = tmp_path / "stage_d_plan.json"

    monkeypatch.setattr(sys, "argv", [
        "make_stage_d_eval_plan",
        "--train-bin", str(train_bin),
        "--target-train-tokens", "1000",
        "--output", str(output),
    ])

    main()

    saved = json.loads(output.read_text(encoding="utf-8"))
    printed = json.loads(capsys.readouterr().out)

    assert saved == printed
    assert saved["inputs"]["train_tokens_per_epoch"] == 125
    assert saved["training"]["suggested_epochs"] == 8
    assert saved["evaluation"]["hellaswag"]["epochs"][-1] == 8


def test_deterministic_plan_output():
    first = make_plan(
        train_tokens=17600,
        target_train_tokens=1000000,
    )
    second = make_plan(
        train_tokens=17600,
        target_train_tokens=1000000,
    )

    assert json.dumps(first, sort_keys=True) == json.dumps(
        second, sort_keys=True
    )
