"""Tests for the Stage E Round-2 split/leakage/identity/GSM8K work (Nova).

Mirrors the round-2 lane packet acceptance tests: template-group split
never straddles, G5 leakage exact_user == 0 after the pipeline, identity
answers restricted to the approved PlaygroundLM set (no human personas),
arithmetic answers exact, GSM8K annotations re-evaluated exactly.
Synthetic fixtures only. Run: python -m pytest data/
"""

import json
import re
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent))

from build_stage_e_sft import (
    dedupe_user_norms,
    drop_val_answer_collisions,
    g5_norm,
    gen_arithmetic,
    gen_identity_and_follow,
    group_split,
    leakage_report,
    load_gsm8k,
    template_key,
    validate_identity_rows,
    verify_gsm8k_answer,
)

DOLLY = "databricks/databricks-dolly-15k"
ARITH = "generated-arithmetic"
IDENT = "generated-identity"


def _row(rid, source, user, assistant, category="open_qa"):
    return {"id": rid, "source": source, "license": "x",
            "category": category, "user": user, "assistant": assistant,
            "text": f"User: {user}\nAssistant: {assistant}"}


def test_template_key_arithmetic_uses_skeleton():
    a = _row("a", ARITH, "What is 12 + 30?", "42", "arithmetic-add")
    b = _row("b", ARITH, "What is 4 + 9?", "13", "arithmetic-add")
    c = _row("c", ARITH, "What is 12 - 30?", "-18", "arithmetic-sub")
    assert template_key(a) == template_key(b)
    assert template_key(a) != template_key(c)


def _mixed_rows():
    rows = []
    for i in range(40):
        rows.append(_row(f"d{i}", DOLLY, f"unique dolly question {i}?",
                         f"dolly answer {i}"))
    rows += gen_arithmetic(60, 7)
    rows += gen_identity_and_follow(48, 7)
    return rows


def test_group_split_never_straddles_templates():
    rows = _mixed_rows()
    train, val, info = group_split(rows, 0.25, 42)
    train_keys = {template_key(r) for r in train}
    val_keys = {template_key(r) for r in val}
    assert train_keys.isdisjoint(val_keys)
    assert val and train
    assert info["method"] == "group_by_template"
    train2, val2, _ = group_split(rows, 0.25, 42)
    assert [r["id"] for r in train] == [r["id"] for r in train2]
    assert [r["id"] for r in val] == [r["id"] for r in val2]


def test_pipeline_gates_g5_leakage_to_zero():
    rows = _mixed_rows()
    # Round-1 failure shape: same prompt, different response (dolly-4758).
    rows.append(_row("d0-alt", DOLLY, rows[0]["user"], "a different answer"))
    # Same order as the builder: exact-text dedupe, then user-norm dedupe.
    seen, uniq = set(), []
    for r in rows:
        if r["text"] not in seen:
            seen.add(r["text"])
            uniq.append(r)
    rows, dropped = dedupe_user_norms(uniq)
    assert dropped >= 1
    assert "d0-alt" not in {r["id"] for r in rows}
    train, val, _info = group_split(rows, 0.25, 42)
    val, _drops = drop_val_answer_collisions(train, val)
    report = leakage_report(train, val)
    assert report["exact_user"] == 0, report
    assert report["exact_assistant"] == 0, report


def test_leakage_report_detects_straddle_with_g5_norm():
    train = [_row("t", DOLLY, "Hello, World!", "Yes.")]
    val = [_row("v", DOLLY, "hello world", "yes")]
    report = leakage_report(train, val)
    assert report["exact_user"] == 1  # punctuation/case-insensitive (G5)
    assert report["exact_assistant"] == 1
    assert report["exact_assistant_by_source"] == {DOLLY: 1}


def test_identity_gate_accepts_generated_rows():
    rows = gen_identity_and_follow(48, 3)
    assert validate_identity_rows(rows) == []


def test_identity_gate_rejects_human_persona():
    bad = _row("bad1", IDENT, "What is your name?", "My name is Joanna.")
    claim = _row("bad2", IDENT, "Who are you?", "I am a human named Sam.")
    unapproved = _row("bad3", IDENT, "Who are you?",
                      "I am a large language model from a big lab.")
    story = _row("ok", DOLLY, "Write a story about a sailor.",
                 "My name is Ishmael, and I sailed the seas.")
    violations = validate_identity_rows([bad, claim, unapproved, story])
    assert violations == ["bad1", "bad2", "bad3"]


def test_arithmetic_answers_are_exact():
    """Answers must be exact ints; Round-2 diversified phrasing is allowed."""
    patterns = [
        (re.compile(r"What is (\d+) \+ (\d+)\?"), lambda a, b: a + b),
        (re.compile(r"What is (\d+) - (\d+)\?"), lambda a, b: a - b),
        (re.compile(r"What is (\d+) \* (\d+)\?"), lambda a, b: a * b),
        (re.compile(r"Compute (\d+) plus (\d+)\."), lambda a, b: a + b),
        (re.compile(r"Compute (\d+) minus (\d+)\."), lambda a, b: a - b),
        (re.compile(r"Compute (\d+) times (\d+)\."), lambda a, b: a * b),
        (re.compile(r"Add (\d+) and (\d+)\."), lambda a, b: a + b),
        (re.compile(r"Subtract (\d+) from (\d+)\."), lambda a, b: b - a),
        (re.compile(r"Multiply (\d+) by (\d+)\."), lambda a, b: a * b),
        (re.compile(r"(\d+) \+ (\d+) = \?"), lambda a, b: a + b),
        (re.compile(r"(\d+) - (\d+) = \?"), lambda a, b: a - b),
        (re.compile(r"(\d+) × (\d+) = \?"), lambda a, b: a * b),
        (re.compile(r"Find the sum of (\d+) and (\d+)\."), lambda a, b: a + b),
        (re.compile(r"Find the difference (\d+) − (\d+)\."), lambda a, b: a - b),
        (re.compile(r"Find the product of (\d+) and (\d+)\."), lambda a, b: a * b),
    ]
    seen = set()
    for row in gen_arithmetic(200, 11):
        user = row["user"]
        matched = False
        for pat, fn in patterns:
            m = pat.fullmatch(user)
            if not m:
                continue
            a, b = int(m.group(1)), int(m.group(2))
            assert row["assistant"] == str(fn(a, b)), (user, row["assistant"])
            seen.add(row["assistant"])
            matched = True
            break
        assert matched, user
    # Unique answers so group-by-template val buckets survive collision drop.
    assert len(seen) == 200


def test_gsm8k_verification_is_exact():
    good = "Janet has 2 baskets with 3 apples each: <<2*3=6>> apples.\n#### 6"
    assert verify_gsm8k_answer(good) == "6"
    wrong = "Math: <<2*3=7>> apples.\n#### 7"
    assert verify_gsm8k_answer(wrong) is None
    assert verify_gsm8k_answer("No annotations and no final.") is None
    nested = "Total <<(2+3)*4=20>> items.\n#### 20"
    assert verify_gsm8k_answer(nested) == "20"


def test_load_gsm8k_drops_unverifiable_rows(tmp_path):
    path = tmp_path / "gsm8k.jsonl"
    lines = [
        {"question": "What is 2 times 3?",
         "answer": "2 times 3 is <<2*3=6>>.\n#### 6"},
        {"question": "What is 2 times 3, wrongly?",
         "answer": "It is <<2*3=7>>.\n#### 7"},
    ]
    path.write_text("\n".join(json.dumps(x) for x in lines) + "\n")
    rows, report = load_gsm8k(path, 10)
    assert report == {"read": 2, "kept": 1, "dropped_failed_verification": 1}
    assert rows[0]["source"] == "openai/gsm8k"
    assert rows[0]["license"] == "MIT"
    assert rows[0]["assistant"] == "6"
    assert g5_norm(rows[0]["user"]) == "what is 2 times 3"
