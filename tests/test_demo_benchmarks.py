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
    assert len(rows) == 4
    assert [row["id"] for row in rows] == [
        "step_00002035",
        "step_00010907",
        "step_00019366",
        "gpt2_124m",
    ]

    for row in rows:
        assert isinstance(row["label"], str)
        for key in COLUMNS:
            assert isinstance(row[key], (int, float))
        assert 0 <= row["hellaswag_acc_norm"] <= 1
        assert 0 <= row["lambada_acc"] <= 1

    # Stage C s2 near GPT-2 on same Stage C val; do not claim overall win.
    s2 = rows[2]
    gpt2 = rows[3]
    assert s2["val_loss"] == 3.288
    assert gpt2["val_loss"] == 3.279
    assert s2["hellaswag_acc_norm"] < gpt2["hellaswag_acc_norm"]
    assert s2["lambada_acc"] < gpt2["lambada_acc"]
    assert "Stage C val" in gpt2["label"]
