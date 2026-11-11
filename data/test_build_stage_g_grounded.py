"""Stage G grounded-facts builder (Claude). Synthetic fact sheet only."""
import json
import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parent))
import build_stage_g_grounded as gb  # noqa: E402

COUNTRIES = [("France", "Paris"), ("Japan", "Tokyo"), ("Kenya", "Nairobi"), ("Peru", "Lima"),
             ("Egypt", "Cairo"), ("Canada", "Ottawa"), ("Norway", "Oslo"), ("Chile", "Santiago"),
             ("Ghana", "Accra"), ("Spain", "Madrid")]


def fact_sheet(heldout=("Peru", "Ghana")):
    facts = []
    for i, (country, capital) in enumerate(COUNTRIES):
        facts.append({"id": f"geo-{i:03d}", "text": f"{capital} is the capital of {country}.",
                      "keywords": [country, capital, "capital"], "aliases": [capital], "topic": "geography",
                      "verify_url": "https://example.org", "license": "MIT (this repository)",
                      "heldout": country in heldout,
                      "questions": [f"What is the capital of {country}?", f"Which city is {country}'s capital?",
                                    f"Name the capital city of {country}.", f"{country}'s capital is which city?"]})
    for i, (planet, n) in enumerate([("Mars", "two"), ("Earth", "one"), ("Neptune", "sixteen")]):
        facts.append({"id": f"sci-{i:03d}", "text": f"{planet} has {n} known moons in this sheet.",
                      "keywords": [planet], "aliases": [n], "topic": "science", "verify_url": "",
                      "license": "MIT (this repository)", "heldout": False,
                      "questions": [f"How many moons does {planet} have?", f"{planet} has how many moons?"]})
    return facts


def test_heldout_facts_never_appear_in_training_rows():
    facts = fact_sheet()
    rows, stats = gb.build_rows(facts, seed=1)
    heldout = {f["id"] for f in facts if f["heldout"]}
    assert stats["heldout_excluded"] == 2
    for r in rows:
        assert r["answer_fact"] not in heldout
        assert not heldout & set(r["fact_ids"])
        assert "Lima" not in r["user"] + r["assistant"] and "Accra" not in r["user"] + r["assistant"]
    assert gb.audit_rows(rows, facts) == []


def test_distractors_never_contain_the_answer_alias():
    facts = fact_sheet()
    rows, _ = gb.build_rows(facts, seed=2)
    by_id = {f["id"]: f for f in facts}
    grounded = [r for r in rows if r["kind"] == "grounded"]
    assert grounded
    for r in grounded:
        ans = by_id[r["answer_fact"]]
        assert r["fact_ids"].count(ans["id"]) == 1 and len(r["fact_ids"]) == 3
        for fid in r["fact_ids"]:
            if fid != ans["id"]:
                assert not gb.mentions(by_id[fid]["text"], ans["aliases"])
        assert r["assistant"] == ans["aliases"][0] + "."


def test_fixed_formats_shares_and_positions():
    facts = fact_sheet()
    rows, stats = gb.build_rows(facts, seed=3)
    n = len(rows)
    assert stats["not_in_facts"] / n == pytest.approx(0.25, abs=0.03)
    assert stats["closed_book"] / n == pytest.approx(0.10, abs=0.03)
    for r in rows:
        if r["kind"] == "not_in_facts":
            assert r["assistant"] == "I'm not sure; the facts I was given don't say."
            assert gb._FACTS_RE.match(r["user"])
        elif r["kind"] == "closed_book":
            assert not r["user"].startswith("Facts:")
        else:
            assert r["user"].startswith("Facts: [1] ") and " Question: " in r["user"]
    # 2-4 phrasings per usable fact, and the answer fact is not always in the same slot.
    per_fact = {}
    for r in rows:
        if r["kind"] == "grounded":
            per_fact[r["answer_fact"]] = per_fact.get(r["answer_fact"], 0) + 1
    assert all(2 <= v <= 4 for v in per_fact.values())
    slots = {r["fact_ids"].index(r["answer_fact"]) for r in rows if r["kind"] == "grounded"}
    assert slots == {0, 1, 2}


def test_not_in_facts_rows_really_lack_the_answer():
    facts = fact_sheet()
    rows, _ = gb.build_rows(facts, seed=4)
    by_id = {f["id"]: f for f in facts}
    by_question = {q: f for f in facts for q in f["questions"]}
    for r in (r for r in rows if r["kind"] == "not_in_facts"):
        target = by_question[r["user"].split(" Question: ")[1]]
        assert target["id"] not in r["fact_ids"]
        assert not any(gb.mentions(by_id[i]["text"], target["aliases"]) for i in r["fact_ids"])


def test_deterministic_and_skips_facts_without_questions(tmp_path):
    facts = fact_sheet()
    assert gb.build_rows(facts, seed=5)[0] == gb.build_rows(facts, seed=5)[0]
    sheet = tmp_path / "sheet.jsonl"
    entries = facts + [{"id": "x-1", "text": "No questions here.", "aliases": ["x"], "heldout": False}]
    sheet.write_text("".join(json.dumps(f) + "\n" for f in entries))
    loaded, report = gb.load_facts(sheet)
    assert report["skipped_no_questions"] == 1 and report["entries"] == len(entries)


def test_val_split_keeps_val_facts_out_of_training():
    facts = fact_sheet()
    built = gb.build(facts, seed=6, val_frac=0.3)
    train_answers = {r["answer_fact"] for r in built["train"]} | {i for r in built["train"] for i in r["fact_ids"]}
    val_answers = {r["answer_fact"] for r in built.get("val", []) if r["answer_fact"]}
    assert val_answers and not val_answers & train_answers


def test_audit_catches_violations():
    facts = fact_sheet()
    rows, _ = gb.build_rows(facts, seed=7)
    heldout_id = next(f["id"] for f in facts if f["heldout"])
    bad = [dict(r) for r in rows[:3]]
    bad[0]["answer_fact"] = heldout_id
    g = next(dict(r) for r in rows if r["kind"] == "grounded")
    ans = next(f for f in facts if f["id"] == g["answer_fact"])
    other = next(f for f in facts if f["id"] not in g["fact_ids"] and not f["heldout"])
    ans_copy = dict(other, id="dup", text=f"{ans['aliases'][0]} is mentioned here.")
    g["fact_ids"] = [i if i == ans["id"] else "dup" for i in g["fact_ids"]][:3]
    problems = gb.audit_rows(bad + [g], facts + [ans_copy])
    assert any("heldout fact in a training row" in p for p in problems)
    assert any("distractor contains an answer alias" in p or "exactly one answer fact" in p for p in problems)


def test_cli_writes_rows_and_g5_report(tmp_path):
    sheet = tmp_path / "sheet.jsonl"
    sheet.write_text("".join(json.dumps(f) + "\n" for f in fact_sheet()))
    other_val = tmp_path / "chat_val.jsonl"
    other_val.write_text(json.dumps({"user": "What is the capital of France?", "assistant": "Paris.",
                                     "source": "chat"}) + "\n")
    gb.main(["--facts", str(sheet), "--out", str(tmp_path / "out"), "--val-frac", "0.3",
             "--g5-val", str(other_val)])
    report = json.loads((tmp_path / "out" / "grounded_report.json").read_text())
    assert report["audit"] == "clean" and "own_val" in report["g5"] and "chat_val.jsonl" in report["g5"]
    assert (tmp_path / "out" / "train.jsonl").is_file()


def test_cli_bare_answers_and_excluded_fact_ids(tmp_path):
    """Lumi glue: --bare-answers drops the trailing period; --exclude-fact-ids holds facts out."""
    import build_stage_g_grounded as gb
    facts = fact_sheet(heldout=())
    sheet = tmp_path / "sheet.jsonl"
    sheet.write_text("".join(json.dumps(f) + "\n" for f in facts))
    excl_ids = [f["id"] for f in facts[:2]]
    excl = tmp_path / "excl.txt"
    excl.write_text("# owner-only\n" + "\n".join(excl_ids) + "\n")
    out = tmp_path / "out"
    try:
        gb.main(["--facts", str(sheet), "--out", str(out), "--bare-answers", "--exclude-fact-ids", str(excl),
                 "--val-frac", "0"])
    finally:
        gb.BARE_ANSWERS = False
    rep = json.loads((out / "grounded_report.json").read_text())
    assert rep["facts"]["newly_held_out"] == 2 and rep["facts"]["bare_answers"] is True
    rows = [json.loads(line) for line in (out / "train.jsonl").read_text().splitlines()]
    assert rows
    for r in rows:
        assert r["answer_fact"] not in excl_ids and not set(excl_ids) & set(r["fact_ids"])
        if r["kind"] != "not_in_facts":
            assert not r["assistant"].endswith(".")
