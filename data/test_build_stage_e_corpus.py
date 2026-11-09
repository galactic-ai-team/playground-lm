"""Tests for data/build_stage_e_corpus.py (Nova). Offline: Dolly input
is a tiny local fixture; no downloads. Run: python -m pytest data/
"""

import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent))

from build_stage_e_corpus import (
    DOLLY_ATTRIBUTION,
    IDENTITY_EXAMPLES,
    arithmetic_records,
    audit_examples,
    build_stage_e,
    load_dolly,
    render_example,
    verify_arithmetic_text,
    write_stage_e,
)
from common import make_record, read_jsonl


def _dolly_fixture(tmp_path: Path) -> Path:
    rows = [
        {"instruction": "Name a primary color.",
         "context": "", "response": "Red is a primary color.",
         "category": "open_qa"},
        {"instruction": "Summarize: The cat sat. It purred.",
         "context": "The cat sat on the mat all afternoon.",
         "response": "A cat sat on a mat and purred.",
         "category": "summarization"},
        {"instruction": "Empty response row", "context": "",
         "response": "  ", "category": "open_qa"},
    ]
    path = tmp_path / "dolly.jsonl"
    path.write_text("\n".join(json.dumps(r) for r in rows))
    return path


def test_dolly_loader_keeps_license_and_skips_empty(tmp_path):
    records, skipped = load_dolly(_dolly_fixture(tmp_path))
    assert skipped == 1 and len(records) == 2
    assert all(r["source"] == "dolly-15k" for r in records)
    assert "Context: The cat sat" in records[1]["text"]
    train, val, stats = build_stage_e(
        dolly_path=_dolly_fixture(tmp_path), arith_count=10, val_pct=0
    )
    assert stats["licenses"]["dolly-15k"]["attribution"] == DOLLY_ATTRIBUTION
    assert stats["licenses"]["dolly-15k"]["license"] == "CC BY-SA 3.0"
    assert stats["per_source_docs"]["dolly-15k"] == 2


def test_arithmetic_generated_answers_all_verify():
    records = arithmetic_records(300, seed=7)
    assert len(records) == 300
    assert all(verify_arithmetic_text(r["text"]) for r in records)


def test_arithmetic_verifier_catches_tampering():
    good = render_example("What is 6 * 7?", "42")
    assert verify_arithmetic_text(good)
    assert not verify_arithmetic_text(render_example("What is 6 * 7?", "43"))
    assert not verify_arithmetic_text(render_example("What is 7 / 2?", "3"))
    assert not verify_arithmetic_text("no structure here")
    # The audit drops a planted bad-arithmetic record and counts it.
    bad = make_record(render_example("What is 2 + 2?", "5"),
                      source="arithmetic-generated", author="nova-generated")
    kept, report = audit_examples([bad])
    assert kept == [] and report["dropped_arithmetic_failed"] == 1


def test_identity_examples_all_carry_source_notes():
    assert len(IDENTITY_EXAMPLES) >= 5
    for ex in IDENTITY_EXAMPLES:
        assert ex["source_note"].strip(), ex["instruction"]
        assert ex["instruction"].strip() and ex["response"].strip()


def test_denylist_guard_exact_and_ngram():
    probe = ("At sunrise, the tiny observatory received a signal that "
             "nobody could explain. The message began")
    exact = make_record(render_example("Repeat the probe", probe),
                        source="dolly-15k", author="databricks-dolly")
    # A dolly row whose response quotes >= 13 consecutive probe words.
    quoting = make_record(
        render_example("Continue this", probe + " with strange static."),
        source="dolly-15k", author="databricks-dolly")
    clean = make_record(render_example("Name a color.", "Blue."),
                        source="dolly-15k", author="databricks-dolly")
    kept, report = audit_examples([exact, quoting, clean],
                                  denylist_texts=[probe])
    assert [r["text"] for r in kept] == [clean["text"]]
    assert report["dropped_denylist_ngram"] == 2


def test_build_split_disjoint_deterministic_and_writes(tmp_path):
    t1, v1, s1 = build_stage_e(arith_count=50, seed=3, val_pct=50)
    t2, v2, _ = build_stage_e(arith_count=50, seed=3, val_pct=50)
    assert [r["id"] for r in t1] == [r["id"] for r in t2]
    assert [r["id"] for r in v1] == [r["id"] for r in v2]
    assert not ({r["id"] for r in t1} & {r["id"] for r in v1})
    assert s1["audit"]["arithmetic_verified"] >= 40  # some arith may dedupe
    out = tmp_path / "stage_e"
    written = write_stage_e(out, t1, v1, s1)
    rows = list(read_jsonl(out / "stage-e-train.jsonl"))
    assert [r["text"] for r in rows] == [r["text"] for r in t1]
    saved = json.loads((out / "stage-e-stats.json").read_text())
    assert saved == written and saved["docs_train"] == len(t1)
    assert "CC BY-SA 3.0" in json.dumps(saved["licenses"])
