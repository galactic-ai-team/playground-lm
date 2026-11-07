"""Offline tests for Stage D benchmark updates."""

import copy
import math

import pytest

from scripts.add_stage_d_benchmark import (
    VALIDATION_NOTE,
    add_stage_d_row,
)


def full_report():
    # Synthetic fixture values only, not real Stage D scores.
    return {
        "checkpoint_id": "stage_d_31104",
        "publishable": True,
        "validation": {
            "complete": True,
            "loss": 2.0,
            "perplexity": math.exp(2.0),
            "tokens_scored": 100,
            "available_targets": 100,
        },
        "benchmarks": {
            "hellaswag": {
                "complete": True,
                "examples": 10042,
                "accuracy": 0.25,
            },
            "lambada": {
                "complete": True,
                "examples": 5153,
                "last_word_accuracy": 0.1,
            },
        },
    }


def base_table():
    return {
        "columns": [
            "val_loss",
            "perplexity",
            "hellaswag_acc_norm",
            "lambada_acc",
        ],
        "column_groups": [
            {
                "label": "FineWeb-Edu validation",
                "columns": [
                    "val_loss",
                    "perplexity",
                ],
            },
            {
                "label": "Downstream benchmarks",
                "columns": [
                    "hellaswag_acc_norm",
                    "lambada_acc",
                ],
            },
        ],
        "rows": [
            {
                "id": "step_00031101",
                "step": 31101,
                "label": "Stage C",
                "val_loss": 3.174,
            },
            {
                "id": "gpt2_124m",
                "label": "OpenAI GPT-2 124M",
                "val_loss": 3.279,
            },
        ],
        "notes": "Existing Stage C note.",
    }


def test_insert_preserves_prior_data_and_marks_split():
    original = base_table()
    frozen = copy.deepcopy(original)

    result = add_stage_d_row(
        original,
        full_report(),
        31104,
    )

    assert original == frozen

    assert [
        row["id"] for row in result["rows"]
    ] == [
        "step_00031101",
        "step_00031104",
        "gpt2_124m",
    ]

    assert result["rows"][0] == frozen["rows"][0]
    assert result["rows"][-1] == frozen["rows"][-1]

    stage_d = result["rows"][1]

    assert stage_d["label"] == (
        "PlaygroundLM Stage D finetune — Step 31,104"
    )
    assert stage_d["validation_split"] == (
        "stage_d_playground_finetune"
    )
    assert stage_d["val_loss"] == 2.0
    assert "NOT FineWeb-Edu" in stage_d[
        "validation_note"
    ]

    assert result["column_groups"][0]["label"] == (
        "Validation (split-specific)"
    )
    assert VALIDATION_NOTE in result["notes"]


def test_idempotent_upsert():
    once = add_stage_d_row(
        base_table(),
        full_report(),
        31104,
    )

    twice = add_stage_d_row(
        once,
        full_report(),
        31104,
    )

    assert once == twice
    assert sum(
        row["id"] == "step_00031104"
        for row in twice["rows"]
    ) == 1


@pytest.mark.parametrize(
    "change",
    [
        lambda r: r.update(publishable=False),
        lambda r: r["validation"].update(
            complete=False
        ),
        lambda r: r["benchmarks"]["hellaswag"].update(
            examples=20
        ),
        lambda r: r["benchmarks"]["lambada"].update(
            complete=False
        ),
    ],
)
def test_reject_partial_eval(change):
    report = full_report()
    change(report)

    with pytest.raises(ValueError):
        add_stage_d_row(
            base_table(),
            report,
            31104,
        )


def test_reject_mismatched_step():
    with pytest.raises(
        ValueError,
        match="does not match",
    ):
        add_stage_d_row(
            base_table(),
            full_report(),
            31105,
        )
