
"""Tests for checkpoint benchmark publication.

No external datasets, model checkpoints or GPU required.
"""

import copy
import json
import math

import pytest

from evals.publish_checkpoint import (
    COLUMNS,
    make_row,
    merge_manifest,
    publish_row,
)


def full_metrics():
    loss = 3.0

    validation = {
        "complete": True,
        "loss": loss,
        "perplexity": math.exp(loss),
        "tokens_scored": 2047,
        "available_targets": 2047,
    }

    hellaswag = {
        "complete": True,
        "examples": 10042,
        "accuracy": 0.35,
    }

    lambada = {
        "complete": True,
        "examples": 5153,
        "last_word_accuracy": 0.22,
    }

    return validation, hellaswag, lambada


def test_make_row_correct_columns():
    row = make_row(
        "step_10000",
        "Step 10,000",
        *full_metrics(),
    )

    assert row["id"] == "step_10000"
    assert row["label"] == "Step 10,000"
    assert row["val_loss"] == pytest.approx(3.0)
    assert row["perplexity"] == pytest.approx(
        math.exp(3.0)
    )
    assert row["hellaswag_acc_norm"] == 0.35
    assert row["lambada_acc"] == 0.22

    assert set(COLUMNS).issubset(row)


@pytest.mark.parametrize(
    "section",
    ["validation", "hellaswag", "lambada"],
)
def test_incomplete_results_rejected(section):
    reports = list(full_metrics())
    index = {
        "validation": 0,
        "hellaswag": 1,
        "lambada": 2,
    }[section]

    reports[index]["complete"] = False

    with pytest.raises(
        ValueError, match="incomplete"
    ):
        make_row(
            "step_10000",
            "Step 10,000",
            *reports,
        )


def test_partial_validation_rejected():
    reports = list(full_metrics())
    reports[0]["tokens_scored"] = 1024

    with pytest.raises(
        ValueError, match="all targets"
    ):
        make_row(
            "step_10000",
            "Step 10,000",
            *reports,
        )


def test_wrong_benchmark_counts_rejected():
    reports = list(full_metrics())
    reports[1]["examples"] = 20

    with pytest.raises(
        ValueError, match="10042"
    ):
        make_row(
            "step_10000",
            "Step 10,000",
            *reports,
        )


def test_nonfinite_values_rejected():
    reports = list(full_metrics())
    reports[1]["accuracy"] = float("nan")

    with pytest.raises(
        ValueError, match="finite"
    ):
        make_row(
            "step_10000",
            "Step 10,000",
            *reports,
        )


def test_preserves_unrelated_manifest_fields():
    manifest = {
        "model": "PlaygroundLM",
        "checkpoints": [
            {"id": "step_10000", "url": "weights.onnx"}
        ],
        "benchmarks": {
            "columns": list(COLUMNS),
            "rows": [],
            "description": "Existing description",
        },
        "riddle_prompts": [
            "What has a key but no lock?"
        ],
    }

    original = copy.deepcopy(manifest)
    row = make_row(
        "step_10000",
        "Step 10,000",
        *full_metrics(),
    )

    updated = merge_manifest(manifest, row)

    assert manifest == original
    assert updated["model"] == original["model"]
    assert (
        updated["checkpoints"]
        == original["checkpoints"]
    )
    assert (
        updated["riddle_prompts"]
        == original["riddle_prompts"]
    )
    assert (
        updated["benchmarks"]["description"]
        == "Existing description"
    )
    assert updated["benchmarks"]["rows"] == [row]


def test_upsert_does_not_duplicate_rows():
    manifest = {
        "benchmarks": {
            "columns": list(COLUMNS),
            "rows": [],
        }
    }

    row = make_row(
        "step_10000",
        "Step 10,000",
        *full_metrics(),
    )

    once = merge_manifest(manifest, row)
    twice = merge_manifest(once, row)

    assert len(twice["benchmarks"]["rows"]) == 1
    assert twice["benchmarks"]["rows"][0] == row


def test_upsert_preserves_other_checkpoint():
    first = make_row(
        "step_10000",
        "Step 10,000",
        *full_metrics(),
    )
    second = make_row(
        "step_20000",
        "Step 20,000",
        *full_metrics(),
    )

    manifest = {
        "benchmarks": {
            "columns": list(COLUMNS),
            "rows": [first, second],
        }
    }

    revised = dict(first)
    revised["label"] = "Updated Step 10,000"

    updated = merge_manifest(manifest, revised)
    rows = updated["benchmarks"]["rows"]

    assert len(rows) == 2
    assert rows[0]["label"] == revised["label"]
    assert rows[1] == second


def test_atomic_publish_to_file(tmp_path):
    path = tmp_path / "manifest.json"

    original = {
        "model": "PlaygroundLM",
        "checkpoints": [],
        "benchmarks": {
            "columns": list(COLUMNS),
            "rows": [],
        },
        "riddle_prompts": [],
    }
    path.write_text(json.dumps(original))

    row = make_row(
        "step_10000",
        "Step 10,000",
        *full_metrics(),
    )

    publish_row(path, row)
    publish_row(path, row)

    actual = json.loads(path.read_text())

    assert len(actual["benchmarks"]["rows"]) == 1
    assert actual["benchmarks"]["rows"][0] == row
    assert actual["riddle_prompts"] == []
    assert actual["checkpoints"] == []


def test_failed_upsert_leaves_file_unchanged(tmp_path):
    path = tmp_path / "manifest.json"
    original = {
        "benchmarks": {
            "columns": ["some_other_column"],
            "rows": [],
        }
    }
    path.write_text(json.dumps(original))

    row = make_row(
        "step_10000",
        "Step 10,000",
        *full_metrics(),
    )

    before = path.read_bytes()

    with pytest.raises(
        ValueError, match="columns differ"
    ):
        publish_row(path, row)

    assert path.read_bytes() == before
