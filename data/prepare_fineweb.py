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
Resumable: rerunning with the same --out skips stream rows already
consumed (tracked in fineweb-manifest.json, checkpointed every 10k kept
docs and at the end), truncates any lines written after the last
checkpoint, and continues in the same shard. Exact-dedupe state is
per-run (FineWeb-Edu is deduped upstream, so cross-session dupes are rare).


Stage C (5B+ tokens) is built in parallel Kaggle CPU sessions:
  - Partition the config's parquet files with --data-files (a glob the
    datasets library matches inside the config), one --out per session.
  - Give each session a distinct --shard-start so shard filenames never
    collide when outputs are merged. Only applies to a FRESH manifest;
    a resumed run trusts the shard_idx already in its manifest.
  - Bound a session's work with --max-docs (kept docs); the manifest
    resume covers interruptions inside a slice.
  - --val-max-tokens caps the val file (Stage C: ~5M). Once the cap is
    reached, later val-DESIGNATED docs are demoted to train and counted
    in the stats as docs_val_demoted. The written val docs themselves
    never enter train, so the eval set stays clean. val_tokens is
    checkpointed in the manifest, so the cap holds across resumes.
    Parallel assembly: session 0 owns val with the cap; every other
    session passes --val-max-tokens 0 to demote all of its val to train.


Shutdown note: streaming datasets keep native IO threads (pyarrow /
fsspec) alive in the background. Letting the interpreter finalize
while they wind down can abort the process at exit with a PyGILState
fatal error -- after all output is already safely written. This script
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


MANIFEST_DEFAULTS = {
    "docs_seen": 0,          # stream rows consumed (kept + val + rejected)
    "tokens_written": 0,     # approx tokens written (chars / 4)
    "val_tokens": 0,         # approx tokens written to the val file
    "shard_idx": 0,          # current train shard being appended to
    "shard_docs": 0,         # docs already in that shard
    "train_shard_bytes": 0,  # byte size of that shard at the checkpoint
    "val_bytes": 0,          # byte size of fineweb-val.jsonl at the checkpoint
}


def load_manifest(path: Path) -> dict:
    manifest = dict(MANIFEST_DEFAULTS)
    if path.exists():
        manifest.update(json.loads(path.read_text()))
    return manifest


def write_manifest(path: Path, manifest: dict) -> None:
    tmp = path.with_suffix(".tmp")
    tmp.write_text(json.dumps(manifest, indent=2))
    tmp.replace(path)  # atomic: a crash never leaves half a manifest


def truncate_to(path: Path, size: int) -> None:
    """Cut `path` back to `size` bytes (lines written after the checkpoint)."""
    if path.exists() and path.stat().st_size > size:
        with path.open("r+b") as f:
            f.truncate(size)




def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--target-tokens", type=int, default=2_500_000_000)
    ap.add_argument("--out", type=Path, default=Path("data/processed"))
    ap.add_argument("--val-pct", type=int, default=1)
    ap.add_argument("--config", default="sample-10BT")
    ap.add_argument("--data-files", default=None,
                    help="optional glob restricting which parquet files of "
                         "the config are streamed (Stage C partitioning)")
    ap.add_argument("--max-docs", type=int, default=None,
                    help="stop after this many KEPT docs this run")
    ap.add_argument("--shard-start", type=int, default=0,
                    help="starting shard index for a fresh manifest "
                         "(parallel sessions); ignored on resume")
    ap.add_argument("--val-max-tokens", type=int, default=None,
                    help="cap val at ~this many approx tokens; later "
                         "val-designated docs are demoted to train")
    args = ap.parse_args()


    # Must be set before `datasets` (and its tokenizers dependency) is
    # imported: parallel tokenizer threads are one source of the
    # exit-time GIL fatal this script avoids.
    os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")
    from datasets import load_dataset  # imported late: heavy dependency


    args.out.mkdir(parents=True, exist_ok=True)
    manifest_path = args.out / "fineweb-manifest.json"
    fresh = not manifest_path.exists()
    manifest = load_manifest(manifest_path)
    if fresh and args.shard_start:
        manifest["shard_idx"] = args.shard_start
    # Crash recovery: drop any lines written after the last checkpoint so
    # a resumed run never duplicates them (the stream re-yields those docs).
    truncate_to(args.out / f"fineweb-train-{manifest['shard_idx']:05d}.jsonl",
                manifest["train_shard_bytes"])
    truncate_to(args.out / "fineweb-val.jsonl", manifest["val_bytes"])

    load_kwargs: dict = {"name": args.config, "split": "train", "streaming": True}
    if args.data_files:
        load_kwargs["data_files"] = args.data_files
    ds = load_dataset("HuggingFaceFW/fineweb-edu", **load_kwargs)
    stream = iter(ds)

    deduper = Deduper(near=False)
    tokens = manifest["tokens_written"]
    val_tokens = manifest["val_tokens"]
    seen = 0  # stream rows consumed, including ones skipped on resume
    kept_train = kept_val = rejected = demoted = 0
    shard_idx = manifest["shard_idx"]
    shard_docs = manifest["shard_docs"]
    train_f = (args.out / f"fineweb-train-{shard_idx:05d}.jsonl").open("a", encoding="utf-8")
    val_f = (args.out / "fineweb-val.jsonl").open("a", encoding="utf-8")

    def checkpoint() -> None:
        train_f.flush()
        val_f.flush()
        manifest.update(
            docs_seen=seen,
            tokens_written=tokens,
            val_tokens=val_tokens,
            shard_idx=shard_idx,
            shard_docs=shard_docs,
            train_shard_bytes=train_f.tell(),
            val_bytes=val_f.tell(),
        )
        write_manifest(manifest_path, manifest)

    try:
        for row in stream:
            if seen < manifest["docs_seen"]:
                seen += 1
                continue  # resume: skip what a previous run already consumed
            if tokens >= args.target_tokens:
                break
            if args.max_docs is not None and kept_train + kept_val >= args.max_docs:
                break
            seen += 1
            text = clean_text(row.get("text", ""), min_chars=MIN_CHARS)
            if text is None or not deduper.is_new(text):
                rejected += 1
                continue
            rec = make_record(text, source="fineweb-edu")
            line = json.dumps(rec, ensure_ascii=False) + "\n"
            doc_tokens = approx_tokens(text)
            to_val = split_for(text, args.val_pct) == "val"
            if to_val and args.val_max_tokens is not None \
                    and val_tokens >= args.val_max_tokens:
                to_val = False
                demoted += 1
            if to_val:
                val_f.write(line)
                kept_val += 1
                val_tokens += doc_tokens
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
            tokens += doc_tokens
            if (kept_train + kept_val) % 10_000 == 0:
                print(f"kept {kept_train + kept_val:,} docs, ~{tokens:,} tokens", flush=True)
                checkpoint()
        checkpoint()  # final state, so a rerun after completion is a no-op
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
        "data_files": args.data_files,
        "shard_start": args.shard_start,
        "docs_kept_train": kept_train,
        "docs_kept_val": kept_val,
        "docs_val_demoted": demoted,
        "docs_rejected_or_dup": rejected + deduper.exact_dups,
        "approx_tokens_written_total": tokens,
        "approx_val_tokens": val_tokens,
        "chars_per_token_assumption": 4.0,
        "val_pct": args.val_pct,
        "val_max_tokens": args.val_max_tokens,
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
