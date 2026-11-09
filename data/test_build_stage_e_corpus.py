"""Tests for data/build_stage_e_corpus.py (Nova). Offline: Dolly input
is a tiny local fixture; no downloads. Run: python -m pytest data/
"""

import json
import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).parent))

from build_stage_e_corpus import (
    DOLLY_ATTRIBUTION,
    IDENTITY_EXAMPLES,
    arithmetic_records,
    audit_examples,
    build_stage_e,
    load_denylist,
    load_dolly,
    load_eval_prompts,
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


def _write_json(tmp_path: Path, name: str, payload) -> Path:
    path = tmp_path / name
    path.write_text(payload if isinstance(payload, str) else json.dumps(payload))
    return path


def test_denylist_frozen_behavior_schema_prompts_user(tmp_path):
    # The actual dev-suite schema: prompts[].user (synthetic contents).
    path = _write_json(tmp_path, "frozen.json", {
        "version": 1,
        "prompts": [
            {"id": "b1", "user": "synthetic behavior prompt one",
             "assistant": "not a denylist text"},
            {"id": "b2", "user": "synthetic behavior prompt two"},
            {"id": "b3", "assistant": "no user field, unusable"},
        ],
    })
    texts, schema = load_denylist(path)
    assert schema == "prompts[].user"
    assert texts == ["synthetic behavior prompt one",
                     "synthetic behavior prompt two"]


def test_denylist_legacy_schemas_still_load(tmp_path):
    held = _write_json(tmp_path, "held.json", {
        "held_out_probes": [{"id": "p1", "prompt": "legacy probe text"}]})
    texts, schema = load_denylist(held)
    assert texts == ["legacy probe text"] and schema == "held_out_probes[].prompt"
    probes = _write_json(tmp_path, "probes.json", {
        "probes": [{"prompt": "older probe text"}]})
    assert load_eval_prompts(probes) == ["older probe text"]


def test_denylist_missing_fails_closed_unless_not_required(tmp_path):
    missing = tmp_path / "nope.json"
    with pytest.raises(FileNotFoundError):
        load_denylist(missing)
    assert load_denylist(missing, required=False) == ([], None)


def test_denylist_malformed_and_empty_fail_closed(tmp_path):
    bad_json = _write_json(tmp_path, "bad.json", "{not json")
    with pytest.raises(ValueError):
        load_denylist(bad_json)
    wrong_schema = _write_json(tmp_path, "wrong.json", {"items": [{"x": 1}]})
    with pytest.raises(ValueError):
        load_denylist(wrong_schema)
    empty = _write_json(tmp_path, "empty.json", {"prompts": []})
    with pytest.raises(ValueError):
        load_denylist(empty)
    unusable = _write_json(tmp_path, "unusable.json",
                           {"prompts": [{"assistant": "no user"}]})
    with pytest.raises(ValueError):
        load_denylist(unusable)
    # The same files are tolerated only when explicitly not required.
    assert load_denylist(bad_json, required=False) == ([], None)
    assert load_denylist(empty, required=False) == ([], "prompts[].user")


def test_denylist_loaded_from_frozen_file_actually_excludes(tmp_path):
    probe = ("Synthetic frozen behavior prompt with enough words to "
             "form a thirteen gram window for the guard to catch")
    path = _write_json(tmp_path, "frozen.json",
                       {"prompts": [{"user": probe}]})
    texts, _ = load_denylist(path)
    quoting = make_record(
        render_example("Continue this", probe + " trailing words here."),
        source="dolly-15k", author="databricks-dolly")
    kept, report = audit_examples([quoting], denylist_texts=texts)
    assert kept == [] and report["dropped_denylist_ngram"] == 1


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
