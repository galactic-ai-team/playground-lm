"""Tests for User:/Assistant: denylist glue on the bin+masks SFT builder (Lumi).

Nova's corpus builder audits Instruction:/Response: text; the train path is
data/build_stage_e_sft.py. These tests cover the SFT-side apply_denylist and
fail-closed load wiring without touching live corpora.
"""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from build_stage_e_corpus import load_denylist
from build_stage_e_sft import (
    DEFAULT_DENYLIST_PATH,
    apply_denylist,
    format_example,
)


def _row(user: str, assistant: str = "ok", source: str = "generated-identity", license_: str = "MIT (this repository)") -> dict:
    return {
        "id": f"t-{user[:24]}",
        "source": source,
        "license": license_,
        "category": "test",
        "user": user,
        "assistant": assistant,
        "text": format_example(user, assistant),
    }


def test_apply_denylist_exact_user_schema():
    deny = ["Who are you?", "What is 8 * 7?"]
    rows = [
        _row("Who are you?"),
        _row("What is 8 * 7?"),
        _row("Name three primary colors."),
    ]
    kept, report = apply_denylist(rows, deny)
    assert report["checked"] == 3
    assert report["dropped_denylist_exact"] == 2
    assert report["dropped_denylist_ngram"] == 0
    assert [r["user"] for r in kept] == ["Name three primary colors."]


def test_apply_denylist_ngram_overlap_long_user():
    probe = " ".join(f"word{i}" for i in range(20))
    deny = [probe]
    # Share a 13-gram with the probe
    overlapping = " ".join(f"word{i}" for i in range(5, 20)) + " tail"
    clean = "Completely unrelated short question?"
    kept, report = apply_denylist([_row(overlapping), _row(clean)], deny)
    assert report["dropped_denylist_ngram"] == 1
    assert report["dropped_denylist_exact"] == 0
    assert [r["user"] for r in kept] == [clean]


def test_load_denylist_empty_fail_closed(tmp_path):
    empty = tmp_path / "empty.json"
    empty.write_text(json.dumps({"prompts": []}) + "\n")
    with pytest.raises(ValueError):
        load_denylist(empty, required=True)
    texts, schema = load_denylist(empty, required=False)
    assert texts == []
    assert schema == "prompts[].user"


def test_load_denylist_missing_fail_closed(tmp_path):
    missing = tmp_path / "nope.json"
    with pytest.raises(FileNotFoundError):
        load_denylist(missing, required=True)
    assert load_denylist(missing, required=False) == ([], None)


def test_frozen_behavior_user_prompts_exclude_synthetic_overlap():
    texts, schema = load_denylist(DEFAULT_DENYLIST_PATH, required=True)
    assert schema == "prompts[].user"
    assert "Who are you?" in texts
    rows = [_row(u) for u in ("Who are you?", "Are you ChatGPT?", "Safe unrelated ask about rainfall.")]
    kept, report = apply_denylist(rows, texts)
    assert report["dropped_denylist_exact"] >= 2
    assert all(r["user"] == "Safe unrelated ask about rainfall." for r in kept)


def test_license_metadata_on_generated_helpers():
    """Generated arithmetic/identity rows must carry MIT (repo), not CC0-equivalent."""
    import build_stage_e_sft as sft

    arith = sft.gen_arithmetic(3, seed=1)
    ident = sft.gen_identity_and_follow(3, seed=1)
    assert arith and all(r["license"] == "MIT (this repository)" for r in arith)
    assert ident and all(r["license"] == "MIT (this repository)" for r in ident)
    dolly_note = "CC BY-SA 3.0"
    # provenance template in main sources list uses Dolly CC BY-SA 3.0
    src = Path(sft.__file__).read_text()
    assert 'license": "CC BY-SA 3.0"' in src or "CC BY-SA 3.0" in src
    assert "CC0-1.0-equivalent-generated" not in src
