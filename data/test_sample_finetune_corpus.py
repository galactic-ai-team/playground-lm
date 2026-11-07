"""Tests for data/sample_finetune_corpus.py (Nova). Offline: the HTTP
layer is faked with payloads shaped like the real /api/v0 responses.
Run: python -m pytest data/
"""

import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent))

from build_finetune_corpus import write_finetune_corpus
from common import read_jsonl
from sample_finetune_corpus import (
    build_sample,
    plan_stage_d,
    run_sample,
    take_by_token_budget,
)

BASE = "https://example.test"


def rec(source: str, author: str, chars: int, tag: str) -> dict:
    # approx_tokens is round(chars / 4): 40 chars -> 10, 400 -> 100.
    return {"text": tag + "x" * (chars - len(tag)), "source": source,
            "author": author, "id": tag}


def test_take_by_token_budget_round_robins_groups():
    records = [
        rec("playground-lobby", "lumi", 40, "a"),      # 10 tok
        rec("playground-lobby", "lumi", 40, "b"),      # 10 tok
        rec("playground-lobby", "lumi", 40, "c"),      # 10 tok
        rec("playground-wall", "nova", 40, "d"),       # 10 tok
        rec("playground-lobby", "claude", 400, "e"),   # 100 tok
    ]
    kept, used = take_by_token_budget(records, 25)
    # Round-robin takes one lumi + the wall note (20 tok); the next lumi
    # doc would exceed the cap and the 100-tok claude doc never fits.
    assert [r["id"] for r in kept] == ["a", "d"]
    assert used == 20


def test_take_by_token_budget_zero_and_oversized():
    records = [rec("playground-lobby", "lumi", 400, "a")]  # 100 tok
    assert take_by_token_budget(records, 0) == ([], 0)
    assert take_by_token_budget(records, 50) == ([], 0)
    kept, used = take_by_token_budget(records, 100)
    assert [r["id"] for r in kept] == ["a"] and used == 100


def test_plan_stage_d_math_and_clamps():
    plan = plan_stage_d({"approx_tokens_train": 20_000,
                         "approx_tokens_val": 2_000})
    assert plan["full_approx_tokens_total"] == 22_000
    assert plan["suggested_epochs"] == 50  # 1M target / 20k per epoch
    assert plan["planned_train_tokens"] == 1_000_000
    assert plan["est_train_minutes_at_10k_tok_s"] == 1.7
    assert plan["est_train_minutes_at_25k_tok_s"] == 0.7
    tiny = plan_stage_d({"approx_tokens_train": 2_000,
                         "approx_tokens_val": 0})
    assert tiny["suggested_epochs"] == 200  # clamped, not 500
    assert plan_stage_d({})["suggested_epochs"] == 0


LONG = "the boardwalk by crystal lake glows under eleven neon stars tonight"
PAYLOADS = {
    f"{BASE}/api/v0/rooms": {
        "version": "v0",
        "rooms": [{"id": "lobby", "name": "The Lobby", "messageCount": 3}],
    },
    f"{BASE}/api/v0/rooms/lobby/messages": {
        "version": "v0",
        "messages": [
            {"id": "m1", "roomId": "lobby", "agentName": "Lumi",
             "content": LONG, "createdAt": "2026-10-24T04:00:00+00:00",
             "isSimulated": False},
            {"id": "m2", "roomId": "lobby", "agentName": "Claude",
             "content": "Claude built the frame API and camera core.",
             "createdAt": "2026-10-24T04:01:00+00:00", "isSimulated": False},
            {"id": "m3", "roomId": "lobby", "agentName": "Muse",
             "content": "hello playground",
             "createdAt": "2026-10-24T04:02:00+00:00", "isSimulated": False},
        ],
    },
    f"{BASE}/api/v0/memory": {
        "version": "v0",
        "notes": [
            {"id": "n1", "agentName": "Muse",
             "text": "Nova finished the universe and pinned this note.",
             "createdAt": "2026-10-24T04:03:00+00:00", "tags": []},
        ],
    },
}


def fake_fetch(url: str) -> dict:
    assert url in PAYLOADS, f"unexpected URL fetched: {url}"
    return PAYLOADS[url]


def test_build_sample_end_to_end(tmp_path):
    train, val, stats = build_sample(
        fake_fetch, BASE, train_token_cap=51, val_token_cap=10,
        val_pct=0, pause_s=0, snapshot_utc="test",
    )
    assert val == []
    # Prefixed docs cost 18 (LONG) + 13 (claude) + 6 (hello) + 14
    # (wall) = 51, exactly the cap: with room in the budget the sample
    # is the whole tiny corpus, wall note included.
    assert len(train) == 4
    assert stats["sample_approx_tokens_train"] == 51
    assert stats["full_docs_train"] == 4 and stats["full_docs_val"] == 0
    assert stats["full_per_author_docs"] == {"claude": 1, "lumi": 1, "nova": 2}
    assert stats["full_per_source_docs"] == {
        "playground-lobby": 3, "playground-wall": 1,
    }
    assert stats["sample_per_author_docs"] == stats["full_per_author_docs"]
    assert stats["stage_d_plan"]["full_approx_tokens_train"] > 0

    out = tmp_path / "finetune_sample"
    written = write_finetune_corpus(out, train, val, stats)
    rows = list(read_jsonl(out / "playground-train.jsonl"))
    assert [r["text"] for r in rows] == [r["text"] for r in train]
    saved = json.loads((out / "playground-finetune-stats.json").read_text())
    assert saved == written and saved["docs_train"] == 4


def test_build_sample_val_side_caps():
    train, val, stats = build_sample(
        fake_fetch, BASE, train_token_cap=4096, val_token_cap=10,
        val_pct=100, pause_s=0, snapshot_utc="test",
    )
    assert train == []
    # Every doc costs more than the 10-token val cap except the
    # 6-token "Nova: hello playground"; only it is kept.
    assert [r["text"] for r in val] == ["Nova: hello playground"]
    assert stats["sample_approx_tokens_val"] == 6


def test_run_sample_dry_run_writes_nothing(tmp_path):
    out = tmp_path / "finetune_sample"
    reported = run_sample(
        fake_fetch, out=out, dry_run=True, base=BASE,
        val_pct=0, pause_s=0, snapshot_utc="test",
    )
    assert reported["dry_run"] is True
    assert reported["sample_approx_tokens_train"] > 0
    assert not out.exists()  # no directory, no files


def test_take_by_token_budget_skips_repeated_ids():
    r1 = rec("playground-lobby", "lumi", 40, "a")       # 10 tok
    r2 = rec("playground-lobby", "claude", 40, "b")     # 10 tok
    kept, used = take_by_token_budget([r1, dict(r1), r2], 30)
    # The exact repeat of r1 (same id, as a rebalanced corpus contains)
    # must not eat sample budget: each unique doc appears at most once.
    assert [r["id"] for r in kept] == ["a", "b"]
    assert used == 20


def test_build_sample_forwards_rebalance_options():
    train, val, stats = build_sample(
        fake_fetch, BASE, train_token_cap=4096, val_token_cap=1024,
        val_pct=0, pause_s=0, snapshot_utc="test",
        min_per_author=3, max_repeat=3,
    )
    # Full (rebalanced) train is 3 docs x 3 authors in this fixture
    # (lumi 1->3, claude 1->3, nova 2->3); the sample itself still
    # holds unique docs only.
    assert stats["full_docs_train"] == 9
    assert stats["rebalance"]["per_author"]["lumi"]["final_docs"] == 3
    ids = [r["id"] for r in train]
    assert len(ids) == len(set(ids))


def test_run_sample_writes_files(tmp_path):
    out = tmp_path / "finetune_sample"
    reported = run_sample(
        fake_fetch, out=out, dry_run=False, base=BASE,
        val_pct=0, pause_s=0, snapshot_utc="test",
    )
    assert reported["dry_run"] is False
    assert (out / "playground-train.jsonl").exists()
    assert (out / "playground-finetune-stats.json").exists()
