#!/usr/bin/env python3
"""Stage E Round 3 — replay mixer (Nova, data lane).

Mixes Stage C FineWeb-Edu pretraining tokens ("replay") into the Round-2
leak-free SFT stream so general next-token skill does not drift during
instruction tuning (Claude's FineWeb regression gate: val rise <= 0.02
vs C31101; every Round-2 epochs x LR candidate failed it without replay).

Mask policy per source:
  - SFT rows:    completion-only masks, byte-identical to the Round-2
                 corpus (train.masks.bin prefix is copied, never rebuilt).
  - Replay rows: full-loss — every token supervised (masks = 1 across the
                 whole replay segment). Replay docs are pure LM text.

Guarantees kept from Round 2:
  - The SFT split is not touched: train/val JSONL and val bins pass
    through unchanged; replay goes to TRAIN only.
  - The SFT corpus must pass the G5 leakage gate (exact_user == 0 and
    exact_assistant == 0) before mixing; a leaking corpus is refused.
  - Replay docs are checked against the SFT *val* split with the G5 norm
    (exact user/assistant collision, or any shared 13-gram with a val
    user) and dropped fail-closed. An optional eval-texts file adds a
    HellaSwag/LAMBADA-style 13-gram contamination screen for replay.

No downloads: replay bins are existing Stage C shards, referenced by
path + sha256 in the emitted provenance. Tokenizer is used only to
decode replay docs for the text checks; pass --require-text-check for
release builds so a missing tokenizer fails instead of skipping.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import random
import shutil
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))

from build_stage_e_sft import _g5_ngrams, g5_norm, leakage_report

ROOT = Path(__file__).resolve().parents[1]
EOT_ID = 50256
REPLAY_SOURCE_ID = "fineweb-edu"
REPLAY_LICENSE = {
    "id": REPLAY_SOURCE_ID,
    "url": "https://huggingface.co/datasets/HuggingFaceFW/fineweb-edu",
    "license": "ODC-By v1.0",
    "notes": "Stage C pretraining corpus (FineWeb-Edu sample-10BT lineage); "
             "replay only — no new download; same license notes as Stage C.",
}


def sha256_file(path: Path) -> str:
    h = hashlib.sha256()
    with Path(path).open("rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def doc_spans(tokens: np.ndarray, eot_id: int = EOT_ID) -> list[tuple[int, int]]:
    """(start, end) spans split after each EOT; a trailing span without a
    final EOT is kept as a last document."""
    ends = np.nonzero(tokens == eot_id)[0]
    spans: list[tuple[int, int]] = []
    start = 0
    for end in ends:
        spans.append((start, int(end) + 1))
        start = int(end) + 1
    if start < tokens.size:
        spans.append((start, int(tokens.size)))
    return spans


def replay_budget(sft_train_tokens: int, ratio: float) -> int:
    """Replay tokens so replay is `ratio` of the TOTAL mixed train set."""
    if not 0.0 < ratio < 1.0:
        raise ValueError("ratio must be in (0, 1)")
    return int(math.ceil(sft_train_tokens * ratio / (1.0 - ratio)))


def _read_jsonl(path: Path) -> list[dict]:
    return [json.loads(line) for line in Path(path).read_text().splitlines()
            if line.strip()]


def sample_replay(
    replay_bins: list[Path],
    budget: int,
    seed: int,
    *,
    tokenizer=None,
    val_users: set[str] | None = None,
    val_answers: set[str] | None = None,
    val_user_grams: set | None = None,
    eval_grams: set | None = None,
) -> tuple[np.ndarray, list[dict], dict]:
    """Sample whole EOT-delimited docs (seeded) until `budget` tokens.

    Returns (tokens, kept_doc_records, report). Docs failing a text check
    are dropped and counted; sampling continues past them.
    """
    docs: list[tuple[int, Path, int, int]] = []  # (bin idx, path, start, end)
    sources: list[dict] = []
    arrays: list[np.ndarray] = []
    for idx, path in enumerate(replay_bins):
        arr = np.memmap(path, dtype=np.uint16, mode="r")
        arrays.append(arr)
        spans = doc_spans(arr)
        sources.append({
            "path": str(path),
            "sha256": None,  # filled by caller (hashing is I/O heavy)
            "tokens_available": int(arr.size),
            "documents": len(spans),
        })
        for start, end in spans:
            docs.append((idx, path, start, end))
    rng = random.Random(seed)
    rng.shuffle(docs)

    kept_tokens: list[np.ndarray] = []
    kept_records: list[dict] = []
    total = 0
    drops = {"val_exact": 0, "val_ngram": 0, "eval_ngram": 0}
    text_checked = tokenizer is not None
    for idx, _path, start, end in docs:
        if total >= budget:
            break
        segment = np.asarray(arrays[idx][start:end])
        if text_checked:
            text = tokenizer.decode([int(t) for t in segment])
            norm = g5_norm(text)
            if norm in (val_users or set()) or norm in (val_answers or set()):
                drops["val_exact"] += 1
                continue
            grams = _g5_ngrams(text)
            if val_user_grams and grams & val_user_grams:
                drops["val_ngram"] += 1
                continue
            if eval_grams and grams & eval_grams:
                drops["eval_ngram"] += 1
                continue
        kept_tokens.append(segment.copy())
        kept_records.append({
            "source": REPLAY_SOURCE_ID,
            "source_bin": str(_path),
            "start": start,
            "end": end,
            "tokens": int(segment.size),
            "text": tokenizer.decode([int(t) for t in segment]) if text_checked else None,
        })
        total += int(segment.size)
    tokens = (np.concatenate(kept_tokens) if kept_tokens
              else np.zeros(0, dtype=np.uint16))
    report = {
        "budget_tokens": budget,
        "sampled_tokens": total,
        "budget_met": total >= budget,
        "documents_kept": len(kept_records),
        "documents_available": len(docs),
        "drops": drops,
        "text_checked": text_checked,
        "sources": sources,
    }
    return tokens.astype(np.uint16), kept_records, report


def audit_mixed(out_dir: Path, sft_dir: Path) -> dict:
    """Post-build mask/format audit of a mixed corpus (counts only)."""
    out = Path(out_dir)
    sft = Path(sft_dir)
    train_tokens = np.fromfile(out / "train.bin", dtype=np.uint16)
    train_masks = np.fromfile(out / "train.masks.bin", dtype=np.uint8)
    sft_tokens = np.fromfile(sft / "train.bin", dtype=np.uint16)
    sft_masks = np.fromfile(sft / "train.masks.bin", dtype=np.uint8)
    n = sft_tokens.size
    checks = {
        "train_len_parity": bool(train_tokens.size == train_masks.size),
        "sft_prefix_tokens_identical": bool(
            train_tokens.size >= n and np.array_equal(train_tokens[:n], sft_tokens)),
        "sft_prefix_masks_identical": bool(
            train_masks.size >= n and np.array_equal(train_masks[:n], sft_masks)),
        "replay_segment_full_loss": bool(
            train_masks.size >= n and np.all(train_masks[n:] == 1)),
        "val_bin_identical": sha256_file(out / "val.bin") == sha256_file(sft / "val.bin"),
        "val_masks_identical": sha256_file(out / "val.masks.bin") == sha256_file(sft / "val.masks.bin"),
        "sft_train_tokens": int(n),
        "mixed_train_tokens": int(train_tokens.size),
        "replay_tokens": int(train_tokens.size - n),
        "sft_supervised_tokens": int(sft_masks.sum()),
        "mixed_supervised_tokens": int(train_masks.sum()),
    }
    checks["pass"] = all(v for k, v in checks.items() if isinstance(v, bool))
    return checks


def mix(
    *,
    sft_dir: Path,
    replay_bins: list[Path],
    ratio: float,
    seed: int,
    out_dir: Path,
    tokenizer=None,
    eval_texts: list[str] | None = None,
    require_text_check: bool = False,
) -> dict:
    sft_dir = Path(sft_dir)
    out_dir = Path(out_dir)
    if tokenizer is None and require_text_check:
        raise SystemExit("text check required but no tokenizer available")
    train_rows = _read_jsonl(sft_dir / "train.jsonl")
    val_rows = _read_jsonl(sft_dir / "val.jsonl")

    leak = leakage_report(train_rows, val_rows)
    if leak["exact_user"] != 0 or leak["exact_assistant"] != 0:
        raise SystemExit(
            f"SFT corpus fails G5 gate: {leak} — fix the corpus, not the mix")

    sft_tokens = np.fromfile(sft_dir / "train.bin", dtype=np.uint16)
    sft_masks = np.fromfile(sft_dir / "train.masks.bin", dtype=np.uint8)
    if sft_tokens.size != sft_masks.size:
        raise SystemExit("SFT train bin/masks length mismatch — refusing to mix")

    val_users = {g5_norm(r.get("user") or "") for r in val_rows}
    val_answers = {g5_norm(r.get("assistant") or "") for r in val_rows}
    val_user_grams: set = set()
    for r in val_rows:
        val_user_grams |= _g5_ngrams(r.get("user") or "")
    eval_grams: set = set()
    for text in eval_texts or []:
        eval_grams |= _g5_ngrams(text)

    budget = replay_budget(int(sft_tokens.size), ratio)
    replay_tokens, kept_records, report = sample_replay(
        [Path(p) for p in replay_bins], budget, seed,
        tokenizer=tokenizer, val_users=val_users, val_answers=val_answers,
        val_user_grams=val_user_grams, eval_grams=eval_grams or None)
    for src, path in zip(report["sources"], replay_bins):
        src["sha256"] = sha256_file(Path(path))

    out_dir.mkdir(parents=True, exist_ok=True)
    mixed_tokens = np.concatenate([sft_tokens, replay_tokens]).astype(np.uint16)
    mixed_masks = np.concatenate(
        [sft_masks, np.ones(replay_tokens.size, dtype=np.uint8)]).astype(np.uint8)
    mixed_tokens.tofile(out_dir / "train.bin")
    mixed_masks.tofile(out_dir / "train.masks.bin")
    for name in ("val.bin", "val.masks.bin", "train.jsonl", "val.jsonl"):
        shutil.copyfile(sft_dir / name, out_dir / name)
    if tokenizer is not None:
        with (out_dir / "replay.jsonl").open("w", encoding="utf-8") as f:
            for rec in kept_records:
                f.write(json.dumps(
                    {"source": rec["source"], "tokens": rec["tokens"],
                     "text": rec["text"]}, ensure_ascii=False) + "\n")

    sft_provenance = {}
    prov_path = sft_dir / "provenance.json"
    if prov_path.exists():
        sft_provenance = json.loads(prov_path.read_text())
    checks = audit_mixed(out_dir, sft_dir)
    total = int(mixed_tokens.size)
    provenance = {
        "mixer": "data/mix_stage_e_replay.py",
        "seed": seed,
        "ratio_target": ratio,
        "ratio_actual_replay_share": (int(replay_tokens.size) / total) if total else 0.0,
        "sft_dir": str(sft_dir),
        "sft_provenance_sources": sft_provenance.get("sources", []),
        "sources": (sft_provenance.get("sources", []) + [REPLAY_LICENSE]),
        "mask_policy": {
            "sft": "completion-only (Round-2 masks copied byte-identical)",
            "replay": "full-loss (every replay token supervised)",
        },
        "token_counts": {
            "sft_train": int(sft_tokens.size),
            "replay": int(replay_tokens.size),
            "mixed_train": total,
            "val": int(np.fromfile(sft_dir / 'val.bin', dtype=np.uint16).size),
        },
        "replay_sampling": report,
        "contamination_screen": {
            "eval_texts_provided": bool(eval_texts),
            "eval_ngram_drops": report["drops"]["eval_ngram"],
            "note": "Replay source is the Stage C corpus, already screened "
                    "at pretraining (data/contamination_check.py); this screen "
                    "is an additional fail-closed pass when eval texts are given.",
        },
        "sft_leakage_report": leak,
        "checks": checks,
        "inputs_sha256": {
            "sft_train_bin": sha256_file(sft_dir / "train.bin"),
            "sft_train_masks": sha256_file(sft_dir / "train.masks.bin"),
            "sft_val_bin": sha256_file(sft_dir / "val.bin"),
            "sft_val_masks": sha256_file(sft_dir / "val.masks.bin"),
        },
        "outputs_sha256": {
            "train_bin": sha256_file(out_dir / "train.bin"),
            "train_masks": sha256_file(out_dir / "train.masks.bin"),
            "val_bin": sha256_file(out_dir / "val.bin"),
            "val_masks": sha256_file(out_dir / "val.masks.bin"),
        },
        "val_unchanged": True,
    }
    (out_dir / "provenance.json").write_text(json.dumps(provenance, indent=2) + "\n")
    if not checks["pass"]:
        raise SystemExit(f"mixed corpus failed post-build audit: {checks}")
    return provenance


def _load_tokenizer():
    try:
        sys.path.insert(0, str(ROOT / "src"))
        from playground_lm.tokenizer import get_tokenizer
        return get_tokenizer()
    except Exception:
        return None


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--sft-dir", type=Path, required=True)
    ap.add_argument("--replay-bin", type=Path, action="append", required=True,
                    dest="replay_bins")
    ap.add_argument("--ratio", type=float, required=True,
                    choices=[0.1, 0.25, 0.5])
    ap.add_argument("--seed", type=int, default=20261109)
    ap.add_argument("--out", type=Path, required=True)
    ap.add_argument("--eval-texts", type=Path, default=None,
                    help="JSONL with {'text': ...} rows for the replay "
                         "contamination screen (optional)")
    ap.add_argument("--require-text-check", action="store_true",
                    help="fail if the repo tokenizer cannot be loaded "
                         "(release builds)")
    args = ap.parse_args()

    eval_texts = None
    if args.eval_texts is not None:
        eval_texts = [r.get("text", "") for r in _read_jsonl(args.eval_texts)]
    tokenizer = _load_tokenizer()
    provenance = mix(
        sft_dir=args.sft_dir, replay_bins=args.replay_bins, ratio=args.ratio,
        seed=args.seed, out_dir=args.out, tokenizer=tokenizer,
        eval_texts=eval_texts, require_text_check=args.require_text_check)
    print(json.dumps({
        "ratio_target": provenance["ratio_target"],
        "ratio_actual_replay_share": round(
            provenance["ratio_actual_replay_share"], 4),
        "token_counts": provenance["token_counts"],
        "replay_drops": provenance["replay_sampling"]["drops"],
        "checks_pass": provenance["checks"]["pass"],
    }, indent=2))
    print("wrote", args.out)


if __name__ == "__main__":
    main()
