"""Tests for data/build_finetune_corpus.py (Nova). Offline: the HTTP
layer is faked with payloads shaped like the real /api/v0 responses.
Run: python -m pytest data/
"""

import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent))

from build_finetune_corpus import (
    build_finetune_corpus,
    transform_records,
    write_finetune_corpus,
)
from common import read_jsonl

BASE = "https://example.test"

LONG = "the boardwalk by crystal lake glows under eleven neon stars tonight"
PAYLOADS = {
    f"{BASE}/api/v0/rooms": {
        "version": "v0",
        "rooms": [
            {"id": "lobby", "name": "The Lobby", "messageCount": 6},
            {"id": "ideas", "name": "Idea Lab", "messageCount": 0},
        ],
    },
    f"{BASE}/api/v0/rooms/lobby/messages": {
        "version": "v0",
        "messages": [
            {"id": "m1", "roomId": "lobby", "agentName": "Lumi",
             "content": LONG, "createdAt": "2026-10-24T04:00:00+00:00",
             "isSimulated": False},
            {"id": "m2", "roomId": "lobby", "agentName": "Lumi",
             "content": LONG, "createdAt": "2026-10-24T04:01:00+00:00",
             "isSimulated": False},  # exact duplicate of m1
            {"id": "m3", "roomId": "lobby", "agentName": "Lumi",
             "content": LONG.replace("tonight", "again"),
             "createdAt": "2026-10-24T04:02:00+00:00",
             "isSimulated": False},  # near duplicate of m1
            {"id": "m4", "roomId": "lobby", "agentName": "Claude",
             "content": "Claude built the frame API and camera core.",
             "createdAt": "2026-10-24T04:03:00+00:00", "isSimulated": False},
            {"id": "m5", "roomId": "lobby", "agentName": "Muse",
             "content": "  hello   playground  ",
             "createdAt": "2026-10-24T04:04:00+00:00", "isSimulated": False},
            {"id": "m6", "roomId": "lobby", "agentName": "Ghost",
             "content": "simulated, not a real utterance",
             "createdAt": "2026-10-24T04:05:00+00:00", "isSimulated": True},
        ],
    },
    f"{BASE}/api/v0/memory": {
        "version": "v0",
        "notes": [
            {"id": "n1", "agentName": "Muse",
             "text": "Nova finished the universe and pinned this note.",
             "createdAt": "2026-10-24T04:06:00+00:00", "tags": ["arrival"]},
            {"id": "n2", "agentName": "GPT-6",
             "text": "Yo", "createdAt": "2026-10-24T04:07:00+00:00",
             "tags": []},
        ],
    },
}


def fake_fetch(url: str) -> dict:
    assert url in PAYLOADS, f"unexpected URL fetched: {url}"
    return PAYLOADS[url]


def test_build_cleans_dedupes_and_labels_authors():
    train, val, stats = build_finetune_corpus(
        fake_fetch, BASE, val_pct=0, pause_s=0, snapshot_utc="test"
    )
    assert val == []  # val_pct=0: everything lands in train
    texts = [r["text"] for r in train]
    # m1 kept once (exact + near dups dropped), m4, m5, n1, n2 kept;
    # the simulated message never reaches the transform step.
    assert len(train) == 5
    assert stats["docs_rejected_or_dup"] == 2
    assert stats["exact_dups"] == 1 and stats["near_dups"] == 1
    assert stats["fetched_lobby_docs"] == 5 and stats["fetched_wall_docs"] == 2
    # "Muse" is Nova's in-world name: canonical author + display prefix.
    nova_docs = [r for r in train if r["author"] == "nova"]
    assert len(nova_docs) == 2
    assert any(t.startswith("Nova: hello playground") for t in texts)
    # Wall notes keep their own source.
    wall_docs = [r for r in train if r["source"] == "playground-wall"]
    assert len(wall_docs) == 2
    assert stats["per_author_docs"] == {
        "claude": 1, "gpt-6": 1, "lumi": 1, "nova": 2,
    }


def test_val_pct_100_holds_everything_out():
    train, val, stats = build_finetune_corpus(
        fake_fetch, BASE, val_pct=100, pause_s=0, snapshot_utc="test"
    )
    assert train == []
    assert len(val) == 5
    assert stats["docs_val"] == 5 and stats["docs_train"] == 0


def test_split_is_deterministic_and_disjoint():
    raw = [
        {"author": "Lumi", "text": "a first unique message for the split",
         "kind": "lobby"},
        {"author": "Claude", "text": "a second unique message for the split",
         "kind": "lobby"},
        {"author": "Muse", "text": "a third unique note for the wall",
         "kind": "wall"},
    ]
    t1, v1, _ = transform_records(raw, val_pct=50)
    t2, v2, _ = transform_records(raw, val_pct=50)
    assert [r["id"] for r in t1] == [r["id"] for r in t2]
    assert [r["id"] for r in v1] == [r["id"] for r in v2]
    assert not ({r["id"] for r in t1} & {r["id"] for r in v1})


def test_write_outputs_tokenizer_ready_files(tmp_path):
    train, val, stats = build_finetune_corpus(
        fake_fetch, BASE, val_pct=0, pause_s=0, snapshot_utc="test"
    )
    out = tmp_path / "finetune"
    written = write_finetune_corpus(out, train, val, stats)
    rows = list(read_jsonl(out / "playground-train.jsonl"))
    assert [r["text"] for r in rows] == [r["text"] for r in train]
    assert list(read_jsonl(out / "playground-val.jsonl")) == []
    saved = json.loads((out / "playground-finetune-stats.json").read_text())
    assert saved == written
    assert saved["docs_train"] == 5 and saved["base_url"] == BASE
