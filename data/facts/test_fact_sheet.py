"""Tests for data/facts/fact_sheet.jsonl (Stage G, Nova).

Schema and hygiene gates from the Stage G lane packet: unique ids,
unique normalised text, every alias appears in its text (casefold),
keywords non-empty, heldout ratio 15-25%, each line <= 30 words,
file < 200 KB, entry count in the 400-800 v1 range.
Run: python -m pytest data/facts
"""

import json
import re
from pathlib import Path

SHEET = Path(__file__).resolve().parent / "fact_sheet.jsonl"
REQUIRED = {"id", "text", "keywords", "aliases", "topic", "verify_url",
            "license", "heldout"}
TOPICS = {"geography", "science", "calendar", "history", "animals",
          "language", "self"}


def _rows() -> list[dict]:
    return [json.loads(line)
            for line in SHEET.read_text(encoding="utf-8").splitlines()
            if line.strip()]


def test_file_size_and_count():
    assert SHEET.stat().st_size < 200 * 1024
    rows = _rows()
    assert 400 <= len(rows) <= 800


def test_schema_and_unique_ids():
    rows = _rows()
    ids = [r["id"] for r in rows]
    assert len(ids) == len(set(ids))
    for r in rows:
        assert set(r) == REQUIRED, r["id"]
        assert isinstance(r["heldout"], bool)
        assert r["topic"] in TOPICS
        assert r["license"] == "MIT (this repository)"
        assert r["verify_url"].startswith("https://")
        assert isinstance(r["keywords"], list) and r["keywords"]
        assert all(isinstance(k, str) and k for k in r["keywords"])
        assert isinstance(r["aliases"], list) and r["aliases"]


def test_unique_normalised_text():
    rows = _rows()
    norms = [" ".join(re.findall(r"\w+", r["text"].lower())) for r in rows]
    assert len(norms) == len(set(norms))


def test_every_alias_appears_in_text():
    for r in _rows():
        text = r["text"].casefold()
        for alias in r["aliases"]:
            assert alias.casefold() in text, (r["id"], alias)


def test_heldout_ratio_and_topic_spread():
    rows = _rows()
    held = [r for r in rows if r["heldout"]]
    ratio = len(held) / len(rows)
    assert 0.15 <= ratio <= 0.25, ratio
    by_topic: dict[str, list[dict]] = {}
    for r in rows:
        by_topic.setdefault(r["topic"], []).append(r)
    assert set(by_topic) == TOPICS
    for topic, members in by_topic.items():
        topic_held = sum(1 for r in members if r["heldout"])
        assert topic_held > 0, f"no heldout entries in {topic}"


def test_line_lengths():
    for line in SHEET.read_text(encoding="utf-8").splitlines():
        if not line.strip():
            continue
        assert len(line.split()) <= 30
        row = json.loads(line)
        assert len(row["text"].split()) <= 30
