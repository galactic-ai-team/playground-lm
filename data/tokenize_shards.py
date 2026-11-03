"""Tokenize processed JSONL into train.bin / val.bin (Nova).

Reads the JSONL written by prepare_fineweb.py / prepare_playground.py
and writes the flat uint16 GPT-2 token files that
src/playground_lm/train/data.py:build_dataloader consumes: documents
concatenated in order, one EOT (50256) appended after every document.

Routing: an input file goes to train if its name contains "train",
val if it contains "val" (same convention as corpus_stats.py).
Input order within a split is sorted filenames — deterministic, which
the resume logic relies on.

Streaming + resumable (built for Kaggle sessions):
  - JSONL is read line by line; tokens accumulate in a ~1M-token
    buffer and are appended to <split>.bin.part per flush.
  - After each flush, <split>.tokenize-state.json is atomically
    rewritten with {file_idx, docs_done_in_file, tokens_written}.
  - On restart: the .part file is truncated to the manifest's token
    count (covers a crash between flush and manifest write), finished
    docs are skipped by count, and tokenization continues.
  - When a split completes, .part is renamed to <split>.bin and the
    manifest is marked finished; rerunning is then a no-op.

Tokenizer: get_tokenizer() from src/playground_lm/tokenizer.py
(GPT-6's lane). Imported lazily in main() so tests can inject a stub;
PLM_CACHE_DIR offline behavior comes from that module unchanged.

Usage:
  python data/tokenize_shards.py --input data/processed \
      --out data/processed/tokens
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Protocol, Sequence

import numpy as np

from common import read_jsonl

FLUSH_TOKENS = 1_000_000
UINT16_MAX = 65_535


class TokenizerLike(Protocol):
    eot_token_id: int

    def encode(self, text: str, *, add_eot: bool = False) -> list[int]: ...


def _write_state(path: Path, state: dict) -> None:
    tmp = path.with_suffix(".tmp")
    tmp.write_text(json.dumps(state, indent=2))
    tmp.replace(path)  # atomic on POSIX: a crash never halves the manifest


def tokenize_split(
    files: Sequence[Path],
    out_dir: Path,
    split: str,
    tokenizer: TokenizerLike,
    *,
    flush_tokens: int = FLUSH_TOKENS,
    limit_docs: int | None = None,
) -> dict:
    """Tokenize `files` (in order) into out_dir/<split>.bin. Resumable.

    limit_docs caps documents processed in this call (used by tests and
    for deliberately interrupted runs); the manifest records progress
    either way, and a later call with no limit finishes the split.
    """
    out_dir.mkdir(parents=True, exist_ok=True)
    part_path = out_dir / f"{split}.bin.part"
    final_path = out_dir / f"{split}.bin"
    state_path = out_dir / f"{split}.tokenize-state.json"

    state = (
        json.loads(state_path.read_text())
        if state_path.exists()
        else {"file_idx": 0, "docs_done_in_file": 0, "tokens_written": 0,
              "docs_total": 0, "finished": False}
    )
    if state["finished"] and final_path.exists():
        return state

    # Crash recovery: the .part file may hold tokens the manifest never
    # recorded (crash mid-flush). Trust the manifest; truncate the rest.
    expected_bytes = state["tokens_written"] * 2
    if part_path.exists():
        actual = part_path.stat().st_size
        if actual > expected_bytes:
            with part_path.open("r+b") as f:
                f.truncate(expected_bytes)
        elif actual < expected_bytes:
            raise IOError(
                f"{part_path} has {actual} bytes but the manifest records "
                f"{expected_bytes}; refusing to guess — delete the .part "
                f"and the manifest to restart this split from scratch."
            )

    docs_this_call = 0
    buffer: list[int] = []

    with part_path.open("ab") as out:

        def flush() -> None:
            if not buffer:
                return
            arr = np.asarray(buffer, dtype=np.uint16)
            arr.tofile(out)
            out.flush()
            state["tokens_written"] += int(arr.size)
            buffer.clear()
            _write_state(state_path, state)

        while state["file_idx"] < len(files):
            path = files[state["file_idx"]]
            # Capture the resume point ONCE per file. (Bug in the first
            # version: the skip test compared a local counter against
            # state["docs_done_in_file"] while that same loop kept
            # incrementing it, so every doc after the first in a file
            # was mistaken for already-done and skipped.)
            resume_at = state["docs_done_in_file"]
            hit_limit = False
            for idx, rec in enumerate(read_jsonl(path)):
                if idx < resume_at:
                    continue
                if limit_docs is not None and docs_this_call >= limit_docs:
                    state["docs_done_in_file"] = idx  # next doc to process
                    hit_limit = True
                    break
                ids = tokenizer.encode(rec["text"], add_eot=True)
                if ids and (min(ids) < 0 or max(ids) > UINT16_MAX):
                    raise ValueError(f"token id out of uint16 range in {path}")
                buffer.extend(ids)
                state["docs_done_in_file"] = idx + 1
                state["docs_total"] += 1
                docs_this_call += 1
                if len(buffer) >= flush_tokens:
                    flush()
                    print(
                        f"[{split}] {state['docs_total']:,} docs, "
                        f"{state['tokens_written']:,} tokens",
                        flush=True,
                    )
            if hit_limit:
                flush()
                return state
            state["file_idx"] += 1
            state["docs_done_in_file"] = 0
            flush()
        flush()

    state["finished"] = True
    _write_state(state_path, state)
    part_path.replace(final_path)
    return state


def route_files(input_dir: Path) -> dict[str, list[Path]]:
    routed: dict[str, list[Path]] = {"train": [], "val": []}
    for path in sorted(input_dir.glob("*.jsonl")):
        if "train" in path.name:
            routed["train"].append(path)
        elif "val" in path.name:
            routed["val"].append(path)
    return routed


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--input", type=Path, default=Path("data/processed"))
    ap.add_argument("--out", type=Path, default=None,
                    help="default: <input>/tokens")
    ap.add_argument("--flush-tokens", type=int, default=FLUSH_TOKENS)
    args = ap.parse_args()

    # GPT-6's tokenizer lives in src/playground_lm; make the repo
    # importable when this script is run from the repo root or data/.
    repo_root = Path(__file__).resolve().parent.parent
    sys.path.insert(0, str(repo_root / "src"))
    from playground_lm.tokenizer import get_tokenizer

    out_dir = args.out or (args.input / "tokens")
    tokenizer = get_tokenizer()
    routed = route_files(args.input)
    summary = {}
    for split, files in routed.items():
        if not files:
            print(f"[{split}] no input files, skipping")
            continue
        state = tokenize_split(
            files, out_dir, split, tokenizer, flush_tokens=args.flush_tokens
        )
        summary[split] = {
            "files": [str(f) for f in files],
            "docs": state["docs_total"],
            "tokens_incl_eot": state["tokens_written"],
            "bin": str(out_dir / f"{split}.bin"),
        }
        print(f"[{split}] done: {summary[split]}")
    (out_dir / "tokenize-stats.json").write_text(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
