"""Stream a FineWeb-Edu sample into PlaygroundLM's record format (Nova).


Source: HuggingFaceFW/fineweb-edu, config "sample-10BT" (public, no
account or key needed; the `datasets` library streams it). FineWeb-Edu
is already quality-filtered and deduplicated upstream, so this script
only cleans lightly, exact-dedupes, applies the hash-stable split, and
stops at a token budget (README target: ~2.5B tokens for pretraining).


Usage:
  pip install datasets
  python data/prepare_fineweb.py --target-tokens 2500000000 \
      --out data/processed --val-pct 1
  # smaller smoke run:
  python data/prepare_fineweb.py --target-tokens 2000000 --out data/smoke


Output: fineweb-train-00000.jsonl, fineweb-train-00001.jsonl, ...
(50k docs per shard), fineweb-val.jsonl, fineweb-stats.json.
Resumable: rerunning with the same --out skips docs already written
(tracked in fineweb-manifest.json), so an interrupted stream can
continue instead of restarting.


Shutdown note: streaming datasets keep native IO threads (pyarrow /
fsspec) alive in the background. Letting the interpreter finalize
while they wind down can abort the process at exit with a PyGILState
fatal error — after all output is already safely written. This script
therefore closes the stream iterator explicitly and, once files are
closed and stats are on disk, exits with os._exit(0), skipping the
teardown race entirely. All output is flushed before that point.
"""


from __future__ import annotations


import argparse
import gc
import json
import os
import sys
from pathlib import Path


from common import Deduper, approx_tokens, clean_text, make_record, split_for


SHARD_DOCS = 50_000
MIN_CHARS = 500  # drop stubs/fragments; FineWeb-Edu docs are articles




def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--target-tokens", type=int, default=2_500_000_000)
    ap.add_argument("--out", type=Path, default=Path("data/processed"))
    ap.add_argument("--val-pct", type=int, default=1)
    ap.add_argument("--config", default="sample-10BT")
    args = ap.parse_args()


    # Must be set before `datasets` (and its tokenizers dependency) is
    # imported: parallel tokenizer threads are one source of the
    # exit-time GIL fatal this script avoids.
    os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")
    from datasets import load_dataset  # imported late: heavy dependency


    args.out.mkdir(parents=True, exist_ok=True)
    manifest_path = args.out / "fineweb-manifest.json"
    manifest = (
        json.loads(manifest_path.read_text())
        if manifest_path.exists()
        else {"docs_seen": 0, "tokens_written": 0}
    )


    ds = load_dataset(
        "HuggingFaceFW/fineweb-edu", name=args.config, split="train", streaming=True
    )
    stream = iter(ds)


    deduper = Deduper(near=False)
    tokens = manifest["tokens_written"]
    seen = 0
    kept_train = kept_val = rejected = 0
    shard_idx = manifest["docs_seen"] // SHARD_DOCS
    shard_docs = manifest["docs_seen"] % SHARD_DOCS
    train_f = (args.out / f"fineweb-train-{shard_idx:05d}.jsonl").open("a", encoding="utf-8")
    val_f = (args.out / "fineweb-val.jsonl").open("a", encoding="utf-8")


    try:
        for row in stream:
            seen += 1
            if seen <= manifest["docs_seen"]:
                continue  # resume: skip what a previous run already wrote
            if tokens >= args.target_tokens:
                break
            text = clean_text(row.get("text", ""), min_chars=MIN_CHARS)
            if text is None or not deduper.is_new(text):
                rejected += 1
                continue
            rec = make_record(text, source="fineweb-edu")
            line = json.dumps(rec, ensure_ascii=False) + "\n"
            if split_for(text, args.val_pct) == "val":
                val_f.write(line)
                kept_val += 1
            else:
                train_f.write(line)
                kept_train += 1
                shard_docs += 1
                if shard_docs >= SHARD_DOCS:
                    train_f.close()
                    shard_idx += 1
                    shard_docs = 0
                    train_f = (args.out / f"fineweb-train-{shard_idx:05d}.jsonl").open(
                        "w", encoding="utf-8"
                    )
            tokens += approx_tokens(text)
            if (kept_train + kept_val) % 10_000 == 0:
                print(f"kept {kept_train + kept_val:,} docs, ~{tokens:,} tokens", flush=True)
                manifest.update(
                    docs_seen=manifest["docs_seen"] + kept_train + kept_val + rejected,
                    tokens_written=tokens,
                )
                manifest_path.write_text(json.dumps(manifest, indent=2))
    finally:
        train_f.close()
        val_f.close()
        # Close the stream explicitly so its worker/IO threads stop
        # now, not during interpreter finalization.
        close = getattr(stream, "close", None)
        if callable(close):
            close()
        del stream, ds
        gc.collect()


    stats = {
        "source": "fineweb-edu",
        "config": args.config,
        "docs_kept_train": kept_train,
        "docs_kept_val": kept_val,
        "docs_rejected_or_dup": rejected + deduper.exact_dups,
        "approx_tokens_written_total": tokens,
        "chars_per_token_assumption": 4.0,
        "val_pct": args.val_pct,
    }
    (args.out / "fineweb-stats.json").write_text(json.dumps(stats, indent=2))
    print(json.dumps(stats, indent=2))


    # Everything is written and closed. Exit without running the
    # interpreter teardown that races pyarrow/fsspec background
    # threads into a PyGILState fatal (see module docstring).
    sys.stdout.flush()
    sys.stderr.flush()
    os._exit(0)




if __name__ == "__main__":
    main()
