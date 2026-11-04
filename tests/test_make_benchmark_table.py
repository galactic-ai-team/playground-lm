
"""Offline tests for deterministic benchmark table generation."""

import json
import math
import sys

import pytest

from scripts import make_benchmark_table as tables


def report(loss=3.928, hs=0.257, la=0.091):
    return {
        "schema_version": 1,
        "publishable": True,
        "validation": {
            "complete": True,
            "loss": loss,
            "perplexity": math.exp(loss),
            "tokens_scored": 512,
            "available_targets": 512,
        },
        "benchmarks": {
            "hellaswag": {
                "complete": True,
                "examples": 10042,
                "accuracy": hs,
            },
            "lambada": {
                "complete": True,
                "examples": 5153,
                "last_word_accuracy": la,
            },
        },
    }


def test_sorted_checkpoints_and_baseline():
    table = tables.build_table(
        [(2035, report()), (500, report(5.0))],
        report(3.255, 0.309, 0.326),
    )

    assert table["columns"] == list(tables.COLUMNS)
    assert [row["id"] for row in table["rows"]] == [
        "step_00000500",
        "step_00002035",
        "gpt2_124m",
    ]
    assert [row.get("step") for row in table["rows"]] == [
        500, 2035, None,
    ]
    assert table["rows"][1]["hellaswag_acc_norm"] == 0.257
    assert table["rows"][2]["lambada_acc"] == 0.326


def test_markdown_percentage_display():
    table = tables.build_table(
        [(2035, report())],
        report(3.255, 0.309, 0.326),
    )
    text = tables.markdown_table(table)

    assert "PlaygroundLM Step 2,035" in text
    assert "OpenAI GPT-2 124M" in text
    assert "25.70%" in text
    assert "9.10%" in text
    assert "30.90%" in text
    assert "32.60%" in text


@pytest.mark.parametrize(
    "mutation",
    [
        lambda r: r.update(publishable=False),
        lambda r: r["validation"].update(complete=False),
        lambda r: r["validation"].update(tokens_scored=10),
        lambda r: r["benchmarks"]["hellaswag"].update(
            examples=20
        ),
        lambda r: r["benchmarks"]["lambada"].update(
            complete=False
        ),
    ],
)
def test_partial_scores_rejected(mutation):
    data = report()
    mutation(data)

    with pytest.raises(ValueError):
        tables.build_table([(2035, data)], report(3.255))


def test_row_and_raw_metrics_conflict_rejected():
    data = report()
    data["row"] = {
        "val_loss": 999.0,
        "perplexity": math.exp(3.928),
        "hellaswag_acc_norm": 0.257,
        "lambada_acc": 0.091,
    }
    with pytest.raises(ValueError, match="disagrees"):
        tables.build_table([(2035, data)], report(3.255))


def test_duplicate_steps_rejected():
    with pytest.raises(ValueError, match="Duplicate"):
        tables.build_table(
            [(2, report()), (2, report())],
            report(3.255),
        )


def test_cli_files_are_deterministic(
    tmp_path, monkeypatch
):
    stage = tmp_path / "stage.json"
    baseline = tmp_path / "baseline.json"
    json_out = tmp_path / "benchmarks.json"
    md_out = tmp_path / "BENCHMARKS.md"

    stage.write_text(json.dumps(report()))
    baseline.write_text(
        json.dumps(report(3.255, 0.309, 0.326))
    )

    monkeypatch.setattr(sys, "argv", [
        "make_benchmark_table",
        "--checkpoint", f"2035={stage}",
        "--baseline", str(baseline),
        "--json-output", str(json_out),
        "--markdown-output", str(md_out),
    ])
    tables.main()

    first_json = json_out.read_bytes()
    first_md = md_out.read_bytes()

    tables.main()

    assert json_out.read_bytes() == first_json
    assert md_out.read_bytes() == first_md

    result = json.loads(first_json)
    assert len(result["rows"]) == 2
    assert result["rows"][0]["step"] == 2035
    assert result["rows"][1]["id"] == "gpt2_124m"


def test_checkpoint_argument_parsing():
    pairs = tables.parse_checkpoint_pairs([
        "2035=stage_b.json",
        "500=early.json",
    ])
    assert [step for step, _ in pairs] == [500, 2035]

    with pytest.raises(ValueError, match="Duplicate"):
        tables.parse_checkpoint_pairs([
            "2035=a.json",
            "2035=b.json",
        ])
