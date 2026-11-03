"""Tests for data/fetch_playground.py (Nova). Run: python -m pytest data/

The HTTP layer is faked with canned API payloads shaped like the real
/api/v0 responses (verified against the live site 2026-11-02).
"""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent))

from fetch_playground import collect

BASE = "https://example.test"

PAYLOADS = {
    f"{BASE}/api/v0/rooms": {
        "version": "v0",
        "rooms": [
            {"id": "lobby", "name": "The Lobby", "messageCount": 3},
            {"id": "ideas", "name": "Idea Lab", "messageCount": 0},
        ],
    },
    f"{BASE}/api/v0/rooms/lobby/messages": {
        "version": "v0",
        "messages": [
            {"id": "m2", "roomId": "lobby", "agentName": "Lumi",
             "content": "second message", "createdAt": "2026-10-24T05:00:00+00:00",
             "isSimulated": False},
            {"id": "m1", "roomId": "lobby", "agentName": "Muse",
             "content": "first message", "createdAt": "2026-10-24T04:00:00+00:00",
             "isSimulated": False},
            {"id": "m3", "roomId": "lobby", "agentName": "Ghost",
             "content": "simulated, not real", "createdAt": "2026-10-24T06:00:00+00:00",
             "isSimulated": True},
        ],
    },
    f"{BASE}/api/v0/memory": {
        "version": "v0",
        "notes": [
            {"id": "n1", "agentName": "Claude", "text": "a wall note",
             "createdAt": "2026-10-24T06:30:00+00:00", "tags": ["arrival"]},
        ],
    },
}


def fake_fetch(url: str) -> dict:
    assert url in PAYLOADS, f"unexpected URL fetched: {url}"
    return PAYLOADS[url]


def test_collect_maps_fields_and_sorts_oldest_first():
    lobby, wall = collect(fake_fetch, BASE, pause_s=0)
    # Simulated message excluded; empty room never fetched (fake would
    # assert); remaining two sorted by ts ascending.
    assert [r["text"] for r in lobby] == ["first message", "second message"]
    assert lobby[0]["author"] == "Muse"
    assert lobby[0]["kind"] == "lobby" and lobby[0]["room"] == "lobby"
    assert wall[0]["text"] == "a wall note"
    assert wall[0]["kind"] == "wall" and wall[0]["tags"] == ["arrival"]


def test_collect_can_include_simulated():
    lobby, _ = collect(fake_fetch, BASE, include_simulated=True, pause_s=0)
    assert len(lobby) == 3
    assert lobby[-1]["author"] == "Ghost"


def test_collect_respects_room_filter():
    lobby, _ = collect(fake_fetch, BASE, rooms=[], pause_s=0)
    assert lobby == []
