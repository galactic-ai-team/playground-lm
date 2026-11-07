"""Regression tests for grouped demo benchmark metrics."""

import json
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
DEMO = ROOT / "demo"

COLUMNS = [
    "val_loss",
    "perplexity",
    "hellaswag_acc_norm",
    "lambada_acc",
]


def load_table():
    return json.loads(
        (DEMO / "benchmarks.json").read_text(encoding="utf-8")
    )


def test_demo_benchmark_schema():
    table = load_table()
    assert table["columns"] == COLUMNS
    assert table["source_type"] == "reported_rounded_summary"

    assert [row["id"] for row in table["rows"]] == [
        "step_00002035",
        "step_00010907",
        "step_00019366",
        "gpt2_124m",
    ]

    for row in table["rows"]:
        assert isinstance(row["label"], str)
        for key in COLUMNS:
            assert isinstance(row[key], (int, float))
        assert 0 <= row["hellaswag_acc_norm"] <= 1
        assert 0 <= row["lambada_acc"] <= 1


def test_groups_cover_columns_in_exact_order():
    table = load_table()
    groups = table["column_groups"]

    assert len(groups) == 2
    assert groups[0]["label"] == "FineWeb-Edu validation"
    assert groups[0]["columns"] == [
        "val_loss", "perplexity"
    ]
    assert groups[1]["label"] == "Downstream benchmarks"
    assert groups[1]["columns"] == [
        "hellaswag_acc_norm", "lambada_acc"
    ]

    assert [
        key for group in groups for key in group["columns"]
    ] == COLUMNS


def test_reported_scores_unchanged():
    rows = load_table()["rows"]
    assert [
        (
            row["val_loss"],
            row["perplexity"],
            row["hellaswag_acc_norm"],
            row["lambada_acc"],
        )
        for row in rows
    ] == [
        (3.928, 50.8, 0.257, 0.091),
        (3.442, 31.3, 0.276, 0.185),
        (3.288, 26.8, 0.288, 0.19),
        (3.279, 26.6, 0.309, 0.326),
    ]


def test_validation_splits_are_explicit():
    rows = load_table()["rows"]
    assert [row["validation_split"] for row in rows] == [
        "stage_b",
        "stage_c",
        "stage_c",
        "stage_c",
    ]

    assert "different validation split" in (
        load_table()["notes"]
    )


def test_no_overall_win_claim():
    table = load_table()
    s2, gpt2 = table["rows"][2:4]

    assert s2["val_loss"] > gpt2["val_loss"]
    assert s2["hellaswag_acc_norm"] < gpt2["hellaswag_acc_norm"]
    assert s2["lambada_acc"] < gpt2["lambada_acc"]


def test_frontend_supports_group_headers():
    source = (DEMO / "app.js").read_text(encoding="utf-8")

    assert "source.column_groups" in source
    assert "benchmark-group-row" in source
    assert "th.colSpan = group.columns.length" in source
    assert "modelHeader.rowSpan = 2" in source
    assert "row.validation_split === 'stage_b'" in source
