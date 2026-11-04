"""Static regression checks for the demo benchmark integration."""

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


def test_demo_benchmark_table_schema():
    table = json.loads(
        (DEMO / "benchmarks.json").read_text(encoding="utf-8")
    )

    assert table["columns"] == COLUMNS
    assert table["source_type"] == "reported_rounded_summary"

    rows = table["rows"]
    assert len(rows) == 2
    assert [row["id"] for row in rows] == [
        "step_00002035",
        "gpt2_124m",
    ]

    for row in rows:
        assert isinstance(row["label"], str)
        for key in COLUMNS:
            assert isinstance(row[key], (int, float))
        assert 0 <= row["hellaswag_acc_norm"] <= 1
