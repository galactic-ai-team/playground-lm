"""Tests for data/build_stage_f_corpus.py (Lumi + ideas from Nova's Stage F pack).

Offline: no HF downloads. TriviaQA/SQuAD loaders are not part of this builder;
those sources stayed skipped in the uploaded Stage F corpus.
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).parent))

from build_stage_e_sft import _safe_eval_expr, g5_norm
from build_stage_f_corpus import (
    filter_new,
    gen_word_and_twostep,
    refuse_gsm8k_test_path,
)


def test_word_and_twostep_answers_are_exact():
    rows = gen_word_and_twostep(200, 20261109)
    assert rows, "generator should keep some rows"
    kinds = {r["category"] for r in rows}
    assert kinds <= {
        "arithmetic-word-add", "arithmetic-word-sub", "arithmetic-word-mul",
        "arithmetic-two-add-mul", "arithmetic-two-sub-add", "arithmetic-two-mul-add",
    }
    # Re-seed is deterministic.
    again = gen_word_and_twostep(200, 20261109)
    assert [r["user"] for r in rows] == [r["user"] for r in again]
    import re
    checked = 0
    for r in rows:
        user, ans = r["user"], int(r["assistant"])
        cat = r["category"]
        nums = list(map(int, re.findall(r"\d+", user)))
        if cat == "arithmetic-word-add":
            assert ans == nums[0] + nums[1]
        elif cat == "arithmetic-word-sub":
            assert ans == nums[0] - nums[1]
        elif cat == "arithmetic-word-mul":
            assert ans == nums[0] * nums[1]
        elif cat == "arithmetic-two-add-mul":
            assert ans == (nums[0] + nums[1]) * nums[2]
        elif cat == "arithmetic-two-sub-add":
            assert ans == nums[0] - nums[1] + nums[2]
        elif cat == "arithmetic-two-mul-add":
            assert ans == nums[0] * nums[1] + nums[2]
        else:
            raise AssertionError(cat)
        checked += 1
    assert checked == len(rows)


def test_filter_new_blocks_val_answers_but_allows_train_answer_reuse():
    users, val_answers, ids = set(), {"the roman empire fell in the west in 476 ce"}, set()
    # collide with val answer
    leaked = [{
        "id": "n1", "user": "When did Rome fall?",
        "assistant": "The Roman Empire fell in the West in 476 CE",
    }]
    kept, dropped = filter_new(leaked, set(users), set(val_answers), set(ids))
    assert kept == []
    assert dropped.get("val_assistant") == 1
    # same user as an existing train/val user
    users = {g5_norm("explain photosynthesis simply")}
    dup = [{
        "id": "n2", "user": "Explain photosynthesis simply!",
        "assistant": "a unique completion",
    }]
    kept, dropped = filter_new(dup, set(users), set(), set())
    assert kept == []
    assert dropped.get("user") == 1
    # repeating a train answer (not in val) is allowed
    kept, dropped = filter_new(
        [{"id": "n3", "user": "What is 7 plus 8?", "assistant": "15"}],
        set(), set(), set(),
    )
    assert [r["id"] for r in kept] == ["n3"]
    assert dropped == {}


def test_gsm8k_test_split_is_refused(tmp_path):
    bad = tmp_path / "gsm8k_test.jsonl"
    bad.write_text("{}\n")
    with pytest.raises(SystemExit, match="test split"):
        refuse_gsm8k_test_path(bad)
    ok = tmp_path / "gsm8k_main_train.jsonl"
    ok.write_text("{}\n")
    refuse_gsm8k_test_path(ok)  # does not raise
