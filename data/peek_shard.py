"""Peek inside a tokenized .bin shard (Nova).

Sanity-check a train.bin / val.bin before a GPU run:
  - total token count and max token id (must be <= 50256, the GPT-2
    EOT id; anything above it means the shard is corrupt or wasn't
    written by tokenize_shards.py)
  - a few random fixed-length windows, decoded back to text with
    GPT-6's tokenizer, so a human can eyeball that the bytes are
    really our corpus (and spot HTML garbage, mojibake, or a shard
    of pure EOTs before burning Kaggle hours on it).

Usage:
  python data/peek_shard.py data/processed/tokens/train.bin \
      --windows 3 --length 256 --seed 42
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path
from typing import Protocol, Sequence

import numpy as np

MAX_VALID_ID = 50_256  # GPT-2 EOT; the largest id GPT-2 BPE can emit


class DecoderLike(Protocol):
    def decode(self, token_ids: Sequence[int]) -> str: ...


def peek(
    path: Path,
    tokenizer: DecoderLike,
    *,
    windows: int = 3,
    length: int = 256,
    seed: int = 42,
) -> dict:
    """Return {tokens, max_id, windows: [{start, text}, ...]}."""
    data = np.memmap(path, dtype=np.uint16, mode="r")
    n = int(data.shape[0])
    if n == 0:
        raise ValueError(f"{path} is empty")
    max_id = int(data.max())
    win_len = min(length, n)
    rng = np.random.default_rng(seed)
    if n - win_len > 0:
        starts = sorted(int(s) for s in rng.integers(0, n - win_len + 1, size=windows))
    else:
        starts = [0]
    out_windows = []
    for s in starts:
        ids = [int(t) for t in data[s : s + win_len]]
        out_windows.append({"start": s, "text": tokenizer.decode(ids)})
    return {"tokens": n, "max_id": max_id, "windows": out_windows}


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("path", type=Path)
    ap.add_argument("--windows", type=int, default=3)
    ap.add_argument("--length", type=int, default=256)
    ap.add_argument("--seed", type=int, default=42)
    args = ap.parse_args()

    repo_root = Path(__file__).resolve().parent.parent
    sys.path.insert(0, str(repo_root / "src"))
    from playground_lm.tokenizer import get_tokenizer

    result = peek(
        args.path,
        get_tokenizer(),
        windows=args.windows,
        length=args.length,
        seed=args.seed,
    )
    status = "OK" if result["max_id"] <= MAX_VALID_ID else "INVALID — id above EOT!"
    print(f"{args.path}: {result['tokens']:,} tokens, max id {result['max_id']} ({status})")
    for w in result["windows"]:
        print(f"\n--- window @ token {w['start']:,} ---")
        print(w["text"])


if __name__ == "__main__":
    main()
