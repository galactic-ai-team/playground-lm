"""Rule-based fact questions (Lumi glue): every v1 fact gets 2-4 answerable phrasings."""
import json
import re
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
from fact_questions import questions_for  # noqa: E402

FACTS = [json.loads(line) for line in (HERE / "fact_sheet.jsonl").read_text().splitlines() if line.strip()]


def test_every_fact_gets_two_to_four_unique_questions_without_the_answer_leaking():
    for f in FACTS:
        qs = questions_for(f)
        assert 2 <= len(qs) <= 4, (f["id"], qs)
        assert len({q.casefold() for q in qs}) == len(qs)
        for q in qs:
            if q.startswith(("Fill in the blank", "Complete the sentence")):
                assert "___" in q
            elif q.startswith("What is the ") and not q.startswith(("What is the capital of",
                                                                      "What is the chemical symbol")):
                # Superlative form must not contain its own answer. (Template forms may legitimately
                # share words with the answer: Monaco/Monaco, South Africa/Africa.)
                assert not re.search(rf"\b{re.escape(f['aliases'][0])}\b", q, re.IGNORECASE), (f["id"], q)


def test_templates():
    cap = {"text": "Paris is the capital of France.", "aliases": ["Paris"]}
    assert questions_for(cap)[:2] == ["What is the capital of France?", "Which city is the capital of France?"]
    sym = {"text": "The chemical symbol for gold is Au.", "aliases": ["Au"]}
    assert questions_for(sym)[0] == "What is the chemical symbol for gold?"
    month = {"text": "October has 31 days.", "aliases": ["31"]}
    assert questions_for(month)[0] == "How many days does October have?"
    sup = {"text": "Jupiter is the largest planet in the solar system.", "aliases": ["Jupiter"]}
    assert questions_for(sup)[0] == "What is the largest planet in the solar system?"
    other = {"text": "Bees make honey.", "aliases": ["honey"]}
    assert questions_for(other) == ["Fill in the blank: Bees make ___.", "Complete the sentence: Bees make ___."]
