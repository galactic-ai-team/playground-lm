"""Offline tests for the Stage G scratchpad arithmetic generator."""
import json
import random
import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parent))
import gen_scratchpad_arith as g  # noqa: E402


@pytest.mark.parametrize("op,a,b,ans", [
    ("add", 47, 85, 132), ("add", 0, 0, 0), ("add", 9999, 1, 10000),
    ("sub", 302, 47, 255), ("sub", 1000, 999, 1), ("sub", 12, 345, -333), ("sub", 5, 5, 0),
    ("mul", 908, 7, 6356), ("mul", 47, 36, 1692), ("mul", 99, 99, 9801), ("mul", 0, 7, 0),
    ("div", 3584, 7, 512), ("div", 18, 9, 2), ("div", 7, 7, 1),
])
def test_known_problems_verify(op, a, b, ans):
    text = g.solve(op, a, b)
    assert g.extract_answer(text) == ans
    assert text.endswith(f"Answer: {ans}")
    assert g.verify_steps(text) >= 1
    g.check_row(op, a, b, text)


def test_random_fuzz_all_ops_verify():
    rng = random.Random(7)
    for _ in range(2000):
        op = rng.choice(list(g.WRITERS))
        a, b = g.sample_problem(rng, op, 5)
        g.check_row(op, a, b, g.solve(op, a, b))


def test_verifier_catches_a_wrong_step():
    text = g.solve("add", 47, 85).replace("4 + 8 + 1 = 13", "4 + 8 + 1 = 14")
    with pytest.raises(ValueError, match="bad step"):
        g.verify_steps(text)


def test_verifier_catches_wrong_division_step():
    text = g.solve("div", 3584, 7).replace("35 / 7 = 5 remainder 0", "35 / 7 = 4 remainder 0")
    with pytest.raises(ValueError, match="bad division"):
        g.verify_steps(text)


def test_check_row_catches_wrong_final_answer():
    text = g.solve("mul", 47, 36).replace("Answer: 1692", "Answer: 1693")
    with pytest.raises(ValueError, match="final answer"):
        g.check_row("mul", 47, 36, text)


def test_extract_answer_requires_fixed_cue_at_end():
    assert g.extract_answer("Work: ... Answer: -42") == -42
    assert g.extract_answer("Answer: 5 and then more text") is None
    assert g.extract_answer("The answer is 5") is None


def test_invalid_inputs_rejected():
    with pytest.raises(ValueError):
        g.div_steps(10, 3)  # not exact
    with pytest.raises(ValueError):
        g.mul_steps(12, 123)  # 3-digit multiplier
    with pytest.raises(ValueError):
        g.add_steps(-1, 2)


def test_generate_is_deterministic_unique_and_split_by_problem():
    t1, v1 = g.generate(1500, 11)
    t2, v2 = g.generate(1500, 11)
    assert t1 == t2 and v1 == v2
    keys_t = {r["problem_key"] for r in t1}
    keys_v = {r["problem_key"] for r in v1}
    assert len(keys_t) == len(t1) and len(keys_v) == len(v1)
    assert not keys_t & keys_v
    assert 0 < len(v1) < len(t1)
    for r in t1 + v1:
        assert r["license"] == "MIT (this repository)"
        assert r["text"] == f"User: {r['user']}\nAssistant: {r['assistant']}"
        assert "\n" not in r["assistant"]
        op, a, b = r["problem_key"].split(":")
        g.check_row(op, int(a), int(b), r["assistant"])
    cats = {r["category"] for r in t1}
    assert cats == {f"arithmetic-scratchpad-{op}" for op in g.WRITERS}


def test_exclude_keys_never_emitted():
    train, val = g.generate(800, 3, max_digits=3)
    banned = frozenset(r["problem_key"] for r in (train + val)[:100])
    t2, v2 = g.generate(800, 3, max_digits=3, exclude_keys=banned)
    assert not banned & {r["problem_key"] for r in t2 + v2}


def test_cli_writes_manifest_and_refuses_overwrite(tmp_path):
    excl = tmp_path / "keys.txt"
    excl.write_text("# comment\nadd:1:1\n")
    m = g.main(["--n", "200", "--seed", "5", "--out-dir", str(tmp_path / "out"),
                "--exclude-keys", str(excl)])
    assert m["excluded_keys_count"] == 1
    assert m["files"]["train"]["rows"] + m["files"]["val"]["rows"] == 200
    rows = [json.loads(line) for line in (tmp_path / "out" / "scratchpad_train.jsonl").open()]
    assert all(r["problem_key"] != "add:1:1" for r in rows)
    with pytest.raises(FileExistsError):
        g.main(["--n", "200", "--seed", "5", "--out-dir", str(tmp_path / "out")])
