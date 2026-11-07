"""Dry-run sample of the Stage D Playground fine-tune corpus (Nova).

Builds the full corpus with build_finetune_corpus.py, then keeps only a
few thousand approx tokens per split for an end-to-end smoke test of
fetch -> clean/dedupe -> split -> JSONL -> tokenize before the real
Stage D run. Sample selection round-robins across (source, author)
groups so the tiny sample still covers lobby + wall and every voice,
instead of just the first messages chronologically.

Outputs (same filenames/schema as the full builder, ready for
tokenize_shards.py):
  <out>/playground-train.jsonl          (default out: data/finetune_sample)
  <out>/playground-val.jsonl
  <out>/playground-finetune-stats.json  sample counts, full-corpus
                                        counts, and the Stage D plan

Expected full-corpus size (approx tokens, chars/4 — exact BPE counts
come from tokenize_shards.py at Stage D time): the public Playground
held 195 room messages + 6 wall notes at the last full fetch (Nov 2)
and grows slowly, so expect roughly 15k-35k approx tokens total,
split ~90/10 train/val by the hash-stable split. The stats sidecar
records the actual counts for the snapshot it was built from.

Stage D budget: the corpus is tiny next to the ~2-4 GPU-hours left
after Stage C. plan_stage_d() targets ~1M training tokens (tens of
epochs over a ~20k-token train split), which is minutes of pure
training time at the 10-25k tok/s the pretrain stages assume. The
budget is mostly eval, checkpoint export and slack — do not inflate
epochs to fill it: on a corpus this small, extra epochs buy verbatim
memorization, not voice. Checkpoint and run the val + verbatim-copy
checks every ~10 epochs and stop when val stops improving.

Usage:
  python data/sample_finetune_corpus.py
  python data/sample_finetune_corpus.py --train-tokens 8192 --val-tokens 2048
"""

from __future__ import annotations

import argparse
import json
import math
from pathlib import Path
from typing import Callable

from build_finetune_corpus import build_finetune_corpus, write_finetune_corpus
from common import approx_tokens
from fetch_playground import DEFAULT_BASE, fetch_json

DEFAULT_TRAIN_TOKENS = 4096
DEFAULT_VAL_TOKENS = 1024
TARGET_TRAIN_TOKENS = 1_000_000
MIN_EPOCHS = 10
MAX_EPOCHS = 200


def take_by_token_budget(
    records: list[dict], max_tokens: int
) -> tuple[list[dict], int]:
    """Greedily keep records up to an approx-token budget.

    Round-robins across (source, author) groups in first-appearance
    order so a small budget still samples every source and voice; a
    record that does not fit the remaining budget is skipped (a later,
    shorter record may still fit). Returns the kept records in their
    original order, plus the approx tokens they account for.
    """
    if max_tokens <= 0:
        return [], 0
    groups: dict[tuple[str, str], list[int]] = {}
    for idx, rec in enumerate(records):
        groups.setdefault((rec["source"], rec["author"]), []).append(idx)
    keys = list(groups)
    pos = {k: 0 for k in keys}
    kept: set[int] = set()
    used = 0
    remaining = True
    while remaining:
        remaining = False
        for key in keys:
            idxs = groups[key]
            while pos[key] < len(idxs):
                idx = idxs[pos[key]]
                cost = approx_tokens(records[idx]["text"])
                if used + cost <= max_tokens:
                    kept.add(idx)
                    used += cost
                    pos[key] += 1
                    remaining = True
                    break
                pos[key] += 1  # too big for what is left; try its next one
    return [records[i] for i in sorted(kept)], used


def plan_stage_d(full_stats: dict) -> dict:
    """Expected token counts and a sized Stage D pass for the full corpus."""
    train_tokens = int(full_stats.get("approx_tokens_train", 0))
    val_tokens = int(full_stats.get("approx_tokens_val", 0))
    if train_tokens > 0:
        epochs = math.ceil(TARGET_TRAIN_TOKENS / train_tokens)
        epochs = max(MIN_EPOCHS, min(MAX_EPOCHS, epochs))
    else:
        epochs = 0
    planned_tokens = epochs * train_tokens
    return {
        "full_approx_tokens_train": train_tokens,
        "full_approx_tokens_val": val_tokens,
        "full_approx_tokens_total": train_tokens + val_tokens,
        "target_train_tokens": TARGET_TRAIN_TOKENS,
        "suggested_epochs": epochs,
        "planned_train_tokens": planned_tokens,
        "est_train_minutes_at_10k_tok_s": round(planned_tokens / 10_000 / 60, 1),
        "est_train_minutes_at_25k_tok_s": round(planned_tokens / 25_000 / 60, 1),
        "gpu_budget_hours_after_stage_c": [2, 4],
        "note": (
            "Pure training time is minutes; the 2-4 h budget is mostly "
            "eval (HellaSwag/LAMBADA), checkpoint export and slack. "
            "Checkpoint every ~10 epochs, watch val + verbatim-copy, "
            "and early-stop instead of inflating epochs to fill time."
        ),
    }


def build_sample(
    fetch: Callable[[str], dict],
    base: str = DEFAULT_BASE,
    *,
    train_token_cap: int = DEFAULT_TRAIN_TOKENS,
    val_token_cap: int = DEFAULT_VAL_TOKENS,
    val_pct: int = 10,
    rooms: list[str] | None = None,
    include_simulated: bool = False,
    pause_s: float = 0.2,
    snapshot_utc: str | None = None,
) -> tuple[list[dict], list[dict], dict]:
    """Fetch the full corpus, then cut it down to the sample budget."""
    full_train, full_val, full_stats = build_finetune_corpus(
        fetch,
        base,
        rooms=rooms,
        include_simulated=include_simulated,
        val_pct=val_pct,
        pause_s=pause_s,
        snapshot_utc=snapshot_utc,
    )
    train, train_tokens = take_by_token_budget(full_train, train_token_cap)
    val, val_tokens = take_by_token_budget(full_val, val_token_cap)
    stats = {
        "source": "playground-finetune-sample",
        "base_url": full_stats["base_url"],
        "snapshot_utc": full_stats["snapshot_utc"],
        "train_token_cap": train_token_cap,
        "val_token_cap": val_token_cap,
        "sample_approx_tokens_train": train_tokens,
        "sample_approx_tokens_val": val_tokens,
        "full_docs_train": len(full_train),
        "full_docs_val": len(full_val),
        "fetched_lobby_docs": full_stats["fetched_lobby_docs"],
        "fetched_wall_docs": full_stats["fetched_wall_docs"],
        "stage_d_plan": plan_stage_d(full_stats),
    }
    return train, val, stats


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--base", default=DEFAULT_BASE)
    ap.add_argument("--out", type=Path, default=Path("data/finetune_sample"))
    ap.add_argument("--train-tokens", type=int, default=DEFAULT_TRAIN_TOKENS)
    ap.add_argument("--val-tokens", type=int, default=DEFAULT_VAL_TOKENS)
    ap.add_argument("--val-pct", type=int, default=10)
    ap.add_argument("--rooms", default=None,
                    help="comma-separated room ids (default: all with messages)")
    ap.add_argument("--include-simulated", action="store_true")
    args = ap.parse_args()

    rooms = args.rooms.split(",") if args.rooms else None
    train, val, stats = build_sample(
        fetch_json,
        args.base,
        train_token_cap=args.train_tokens,
        val_token_cap=args.val_tokens,
        val_pct=args.val_pct,
        rooms=rooms,
        include_simulated=args.include_simulated,
    )
    written = write_finetune_corpus(args.out, train, val, stats)
    print(json.dumps(written, indent=2))


if __name__ == "__main__":
    main()
