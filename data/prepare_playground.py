"""Prepare the Playground fine-tune corpus (Nova).

Input: one or more raw export files from the AI Playground. Accepts
  - .jsonl / .json: objects with author-ish and text-ish fields
    (author|agent|name|user, text|message|content|note), optional
    kind ("lobby" | "wall" | "memory-wall") and ts.
  - .txt: lines of the form "Name: message".

Output records (common.py schema) have source playground-lobby or
playground-wall and a canonical author id. Each document's text is
prefixed "Name: " with the display name, so the stock GPT-2 BPE vocab
(no new special tokens) still gives the model an author signal to
condition on — GPT-6's eval prompts use the same prefix format:
  "Lumi: ..." / "Nova: ..." / "Claude: ..." ...

Author keys in ALIASES are all lowercase because canonical_author()
lowercases the raw name before lookup. Note "Muse" -> nova (Nova acts
under that name in-world) while "claude" -> claude: different authors.

Cleaning: common.clean_text, exact + near dedupe (the corpus is small
and chatty — repeats and copypasta are real), hash-stable split with a
larger val share (10%) so the held-out Playground eval is measurable.
Stephon's own messages are kept and labeled "stephon": he directed
half of what we said, and the fine-tune should know his voice too.

Usage:
  python data/prepare_playground.py raw/lobby.jsonl raw/wall.json \
      --out data/processed
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from common import Deduper, approx_tokens, clean_text, make_record, split_for, write_jsonl

ALIASES = {
    "lumi": "lumi", "grok": "lumi",
    "gpt-6": "gpt-6", "gpt6": "gpt-6", "chatgpt": "gpt-6",
    "muse": "nova", "nova": "nova",
    "claude": "claude", "claude opus": "claude",
    "gemini": "gemini", "dee": "dee",
    "stephon": "stephon", "galactic_steph_py": "stephon", "galactic steph": "stephon",
}
DISPLAY = {
    "lumi": "Lumi", "gpt-6": "GPT-6", "nova": "Nova", "claude": "Claude",
    "gemini": "Gemini", "dee": "Dee", "stephon": "Stephon", "unknown": "Unknown",
}


def canonical_author(raw: str | None) -> str:
    if not raw:
        return "unknown"
    return ALIASES.get(str(raw).strip().lower(), "unknown")


def rows_from_file(path: Path):
    if path.suffix == ".txt":
        for line in path.read_text(encoding="utf-8").splitlines():
            if ": " in line:
                name, text = line.split(": ", 1)
                yield {"author": name, "text": text, "kind": "lobby"}
        return
    raw = path.read_text(encoding="utf-8")
    if path.suffix == ".jsonl":
        objs = [json.loads(l) for l in raw.splitlines() if l.strip()]
    else:
        parsed = json.loads(raw)
        objs = parsed if isinstance(parsed, list) else parsed.get("messages", [])
    for o in objs:
        yield {
            "author": o.get("author") or o.get("agent") or o.get("name") or o.get("user"),
            "text": o.get("text") or o.get("message") or o.get("content") or o.get("note"),
            "kind": o.get("kind") or o.get("source") or "lobby",
        }


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("inputs", nargs="+", type=Path)
    ap.add_argument("--out", type=Path, default=Path("data/processed"))
    ap.add_argument("--val-pct", type=int, default=10)
    args = ap.parse_args()

    deduper = Deduper(near=True)
    train, val = [], []
    per_author: dict[str, int] = {}
    rejected = 0

    for path in args.inputs:
        for row in rows_from_file(path):
            text = clean_text(row.get("text") or "", min_chars=2)
            if text is None or not deduper.is_new(text):
                rejected += 1
                continue
            author = canonical_author(row.get("author"))
            kind = str(row.get("kind", "lobby")).lower()
            source = "playground-wall" if "wall" in kind else "playground-lobby"
            doc = f"{DISPLAY[author]}: {text}"
            rec = make_record(doc, source=source, author=author)
            (val if split_for(doc, args.val_pct) == "val" else train).append(rec)
            per_author[author] = per_author.get(author, 0) + 1

    n_train = write_jsonl(args.out / "playground-train.jsonl", train)
    n_val = write_jsonl(args.out / "playground-val.jsonl", val)
    stats = {
        "source": "playground",
        "docs_train": n_train,
        "docs_val": n_val,
        "docs_rejected_or_dup": rejected,
        "exact_dups": deduper.exact_dups,
        "near_dups": deduper.near_dups,
        "per_author_docs": per_author,
        "approx_tokens_train": sum(approx_tokens(r["text"]) for r in train),
        "approx_tokens_val": sum(approx_tokens(r["text"]) for r in val),
        "val_pct": args.val_pct,
    }
    (args.out / "playground-stats.json").write_text(json.dumps(stats, indent=2))
    print(json.dumps(stats, indent=2))


if __name__ == "__main__":
    main()
