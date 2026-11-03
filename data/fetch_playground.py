"""Fetch raw Playground text for the fine-tune corpus (Nova).

Pulls the AI Playground's PUBLIC read APIs (no auth, no relay link �
this never touches any agent's private credentials):

  GET {base}/api/v0/rooms                  -> room list
  GET {base}/api/v0/rooms/{id}/messages    -> full message content
  GET {base}/api/v0/memory                 -> Memory Wall notes

Writes raw JSONL in exactly the shape prepare_playground.py accepts
({author, text, kind, ts, id, room?} per line):
  <out>/playground-lobby.jsonl   all room messages, oldest first
  <out>/playground-wall.jsonl    Memory Wall notes, oldest first

Notes:
  - The /api/v0/feed endpoint is NOT used: its summaries are
    truncated, which would poison the corpus with cut-off text.
    The rooms endpoint returns full content.
  - Messages flagged isSimulated are excluded unless
    --include-simulated: they are not real agent utterances.
  - Assumes each messages response is complete; if the API reports
    a room messageCount larger than the messages returned, a warning
    is printed (pagination params are not documented in v0).

Stdlib only (urllib). Usage:
  python data/fetch_playground.py --out data/raw
"""

from __future__ import annotations

import argparse
import json
import time
import urllib.request
from pathlib import Path
from typing import Callable

DEFAULT_BASE = "https://ai-playground-xi-gules.vercel.app"
USER_AGENT = "PlaygroundLM-dataset-fetch/1.0 (galactic-ai-team; public read API)"


def fetch_json(url: str, timeout: float = 30.0) -> dict:
    req = urllib.request.Request(url, headers={"User-Agent": USER_AGENT})
    with urllib.request.urlopen(req, timeout=timeout) as resp:
        return json.loads(resp.read().decode("utf-8"))


def collect(
    fetch: Callable[[str], dict],
    base: str,
    *,
    rooms: list[str] | None = None,
    include_simulated: bool = False,
    pause_s: float = 0.2,
) -> tuple[list[dict], list[dict]]:
    """Return (lobby_records, wall_records), each oldest-first."""
    base = base.rstrip("/")

    room_list = fetch(f"{base}/api/v0/rooms").get("rooms", [])
    if rooms is None:
        rooms = [r["id"] for r in room_list if r.get("messageCount", 0) > 0]

    lobby: list[dict] = []
    for room in room_list:
        rid = room["id"]
        if rid not in rooms:
            continue
        if pause_s:
            time.sleep(pause_s)
        payload = fetch(f"{base}/api/v0/rooms/{rid}/messages")
        messages = payload.get("messages", [])
        if len(messages) < room.get("messageCount", len(messages)):
            print(
                f"WARNING: room {rid} reports {room['messageCount']} messages "
                f"but the API returned {len(messages)}; keeping what we got."
            )
        for m in messages:
            if m.get("isSimulated") and not include_simulated:
                continue
            lobby.append(
                {
                    "author": m.get("agentName"),
                    "text": m.get("content"),
                    "kind": "lobby",
                    "room": rid,
                    "ts": m.get("createdAt"),
                    "id": m.get("id"),
                }
            )

    if pause_s:
        time.sleep(pause_s)
    notes = fetch(f"{base}/api/v0/memory").get("notes", [])
    wall = [
        {
            "author": n.get("agentName"),
            "text": n.get("text"),
            "kind": "wall",
            "ts": n.get("createdAt"),
            "id": n.get("id"),
            "tags": n.get("tags", []),
        }
        for n in notes
    ]

    lobby.sort(key=lambda r: r["ts"] or "")
    wall.sort(key=lambda r: r["ts"] or "")
    return lobby, wall


def _write(path: Path, records: list[dict]) -> int:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as f:
        for rec in records:
            f.write(json.dumps(rec, ensure_ascii=False) + "\n")
    return len(records)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--base", default=DEFAULT_BASE)
    ap.add_argument("--out", type=Path, default=Path("data/raw"))
    ap.add_argument("--rooms", default=None,
                    help="comma-separated room ids (default: all with messages)")
    ap.add_argument("--include-simulated", action="store_true")
    args = ap.parse_args()

    rooms = args.rooms.split(",") if args.rooms else None
    lobby, wall = collect(
        fetch_json,
        args.base,
        rooms=rooms,
        include_simulated=args.include_simulated,
    )
    n1 = _write(args.out / "playground-lobby.jsonl", lobby)
    n2 = _write(args.out / "playground-wall.jsonl", wall)
    print(f"lobby: {n1} messages -> {args.out / 'playground-lobby.jsonl'}")
    print(f"wall:  {n2} notes    -> {args.out / 'playground-wall.jsonl'}")


if __name__ == "__main__":
    main()
