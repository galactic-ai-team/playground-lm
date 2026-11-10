#!/usr/bin/env python3
"""Build Stage F SFT corpus: facts/arithmetic on top of leak-free Stage E val.

Keeps the Round-2/3 leak-free chat val split unchanged. Adds:
  - remaining verified GSM8K *train* rows (MIT, local JSONL only)
  - new programmatic arithmetic and short word problems (MIT, this repo)
TriviaQA and SQuAD are skipped in this build (see provenance exclusions).

G5: exact_user and exact_assistant against the frozen val must be 0.
"""
from __future__ import annotations

import argparse
import json
import random
import sys
from collections import Counter
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "data"))
sys.path.insert(0, str(ROOT / "src"))

from build_stage_e_corpus import load_denylist  # noqa: E402
from build_stage_e_sft import (  # noqa: E402
    DEFAULT_DENYLIST_PATH,
    apply_denylist,
    format_example,
    g5_norm,
    gen_arithmetic,
    leakage_report,
    load_gsm8k,
    sha256_file,
    validate_identity_rows,
    write_split,
)
from playground_lm.tokenizer import get_tokenizer  # noqa: E402


SEED = 20261109
NAMES = [
    "Maya", "Leo", "Aria", "Omar", "Nina", "Kai", "Elena", "Jonas",
    "Priya", "Noah", "Sofia", "Malik", "Hana", "Ivo", "Tara", "Beau",
]


def load_jsonl(path: Path) -> list[dict]:
    rows = []
    with path.open(encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                rows.append(json.loads(line))
    return rows


def blocked_sets(train_rows: list[dict], val_rows: list[dict]) -> tuple[set[str], set[str], set[str]]:
    users = {g5_norm(r.get("user") or "") for r in train_rows + val_rows}
    answers = {g5_norm(r.get("assistant") or "") for r in train_rows + val_rows}
    ids = {r.get("id") for r in train_rows + val_rows if r.get("id")}
    return users, answers, ids


def filter_new(
    rows: list[dict],
    users: set[str],
    val_answers: set[str],
    ids: set[str],
) -> tuple[list[dict], dict]:
    """Drop identity/user leaks and val-answer collisions (G5 exact_assistant).

    Repeating an already-seen *train* answer is allowed; G5 only compares val to train.
    """
    kept = []
    dropped = Counter()
    for row in rows:
        if row.get("id") in ids:
            dropped["id"] += 1
            continue
        u, a = g5_norm(row.get("user") or ""), g5_norm(row.get("assistant") or "")
        if not u or not a:
            dropped["empty"] += 1
            continue
        if u in users:
            dropped["user"] += 1
            continue
        if a in val_answers:
            dropped["val_assistant"] += 1
            continue
        kept.append(row)
        users.add(u)
        if row.get("id"):
            ids.add(row["id"])
    return kept, dict(dropped)


def gen_word_and_twostep(n: int, seed: int) -> list[dict]:
    rng = random.Random(seed + 101)
    rows: list[dict] = []
    used_answers: set[str] = set()
    used_users: set[str] = set()
    attempts = 0
    i = 0

    def push(user: str, ans, kind: str) -> None:
        ans_s = str(ans)
        if ans_s in used_answers or g5_norm(user) in used_users:
            return
        used_answers.add(ans_s)
        used_users.add(g5_norm(user))
        idx = len(rows)
        rows.append({
            "id": f"arith-f-{kind}-{idx}",
            "source": "generated-arithmetic",
            "license": "MIT (this repository)",
            "category": f"arithmetic-{kind}",
            "user": user,
            "assistant": ans_s,
            "text": format_example(user, ans_s),
        })

    templates = [
        ("word-add", lambda a, b, name, item: (
            f"{name} has {a} {item} and gets {b} more. How many {item} does {name} have now?",
            a + b,
        )),
        ("word-sub", lambda a, b, name, item: (
            f"{name} has {a} {item} and gives away {b}. How many {item} remain?",
            a - b,
        )),
        ("word-mul", lambda a, b, name, item: (
            f"{name} packs {a} {item} into each of {b} boxes. How many {item} is that in all?",
            a * b,
        )),
        ("two-add-mul", lambda a, b, c, *_: (
            f"What is {a} plus {b}, then times {c}?",
            (a + b) * c,
        )),
        ("two-sub-add", lambda a, b, c, *_: (
            f"Start with {a}, subtract {b}, then add {c}. What is the result?",
            a - b + c,
        )),
        ("two-mul-add", lambda a, b, c, *_: (
            f"Compute {a} times {b}, then add {c}.",
            a * b + c,
        )),
    ]
    items = ["apples", "pencils", "stickers", "marbles", "books", "cards"]
    while len(rows) < n and attempts < n * 50:
        attempts += 1
        kind, fn = templates[i % len(templates)]
        name = NAMES[i % len(NAMES)]
        item = items[(i // len(NAMES)) % len(items)]
        if kind.startswith("word-mul"):
            a, b = rng.randint(2, 20), rng.randint(2, 12)
            c = 0
        elif kind.startswith("word-sub"):
            a, b = rng.randint(5, 80), rng.randint(1, 40)
            if b > a:
                a, b = b, a
            c = 0
        elif kind.startswith("word-add"):
            a, b = rng.randint(1, 80), rng.randint(1, 80)
            c = 0
        elif kind == "two-sub-add":
            a, b, c = rng.randint(10, 80), rng.randint(1, 30), rng.randint(1, 40)
            if b > a:
                a, b = b, a
        else:
            a, b, c = rng.randint(1, 30), rng.randint(1, 20), rng.randint(2, 12)
        user, ans = fn(a, b, name, item) if kind.startswith("word") else fn(a, b, c, name, item)
        push(user, ans, kind)
        i += 1
    return rows


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--base-dir", type=Path, required=True, help="Round-2 sft dir with train/val jsonl")
    ap.add_argument("--gsm8k-jsonl", type=Path, required=True)
    ap.add_argument("--out", type=Path, required=True)
    ap.add_argument("--seed", type=int, default=SEED)
    ap.add_argument("--arith-n", type=int, default=4000)
    ap.add_argument("--word-n", type=int, default=1500)
    args = ap.parse_args()

    base_train = load_jsonl(args.base_dir / "train.jsonl")
    base_val = load_jsonl(args.base_dir / "val.jsonl")
    if len(base_val) != 469 or len(base_train) != 9702:
        print(f"warn: unexpected base sizes train={len(base_train)} val={len(base_val)}", flush=True)

    users, answers, ids = blocked_sets(base_train, base_val)
    val_answers = {g5_norm(r.get('assistant') or '') for r in base_val}

    gsm8k_rows, gsm8k_report = load_gsm8k(args.gsm8k_jsonl, limit=0)
    # limit=0 might mean all or none — check load_gsm8k
    print("gsm8k raw", gsm8k_report, "n", len(gsm8k_rows), flush=True)

    arith = gen_arithmetic(args.arith_n, args.seed)
    for row in arith:
        row["id"] = f"arith-f-basic-{row['id']}"
    words = gen_word_and_twostep(args.word_n, args.seed)

    denylist_texts, denylist_schema = load_denylist(DEFAULT_DENYLIST_PATH, required=True)
    added_reports = {}
    new_rows = []
    for label, batch in (("gsm8k", gsm8k_rows), ("arith-basic", arith), ("arith-word", words)):
        batch, deny_rep = apply_denylist(batch, denylist_texts)
        kept, dropped = filter_new(batch, users, val_answers, ids)
        added_reports[label] = {"input": len(batch), "kept": len(kept), "dropped": dropped, "denylist": deny_rep}
        new_rows.extend(kept)
        print(f"added {label}: {added_reports[label]}", flush=True)

    violations = validate_identity_rows(base_train + base_val + new_rows)
    if violations:
        raise SystemExit(f"identity gate failed: {violations[:10]}")

    train_rows = list(base_train) + new_rows
    val_rows = list(base_val)
    leak = leakage_report(train_rows, val_rows)
    print("internal leakage_report", leak, flush=True)
    if leak["exact_user"] != 0 or leak["exact_assistant"] != 0:
        raise SystemExit(
            f"G5 leak: exact_user={leak['exact_user']} exact_assistant={leak['exact_assistant']}"
        )

    out = args.out
    out.mkdir(parents=True, exist_ok=True)
    tokenizer = get_tokenizer()
    train_meta = write_split(out, "train", train_rows, tokenizer)
    val_meta = write_split(out, "val", val_rows, tokenizer)

    by_src_train = Counter(r.get("source") for r in train_rows)
    by_src_val = Counter(r.get("source") for r in val_rows)
    provenance = {
        "stage": "F",
        "seed": args.seed,
        "base_corpus": str(args.base_dir),
        "base_val_frozen": True,
        "base_val_examples": len(base_val),
        "base_train_examples": len(base_train),
        "sources": [
            {
                "id": "databricks/databricks-dolly-15k",
                "url": "https://huggingface.co/datasets/databricks/databricks-dolly-15k",
                "license": "CC BY-SA 3.0",
                "notes": "Carried forward from leak-free Stage E Round-2 SFT; val split frozen.",
            },
            {
                "id": "generated-arithmetic",
                "license": "MIT (this repository)",
                "notes": "Programmatic 1-3 digit +/−/×, 2-step, and short word problems; answers machine-verified. New Stage F rows use seed 20261109.",
            },
            {
                "id": "generated-identity",
                "license": "MIT (this repository)",
                "notes": "Carried forward from Stage E Round-2 SFT (frozen val).",
            },
            {
                "id": "openai/gsm8k",
                "url": "https://huggingface.co/datasets/openai/gsm8k",
                "license": "MIT",
                "notes": "Local train-split JSONL only; calculator annotations re-verified; no test split.",
            },
        ],
        "exclusions": [
            "TriviaQA skipped: official repo claims Apache 2.0 for code and data, but Hugging Face tags license unknown and evidence documents have mixed site terms. Not packed without a re-verified nocontext dump.",
            "SQuAD 1.1/2.0 skipped: official license is CC BY-SA 4.0, but items are passage-conditioned and a poor fit for closed-book chat. Not packed in this build.",
            "No new web scrapes. No NC datasets. No private eval items.",
            "Frozen behavior_eval.prompts.frozen.json user prompts (denylist).",
        ],
        "added_reports": added_reports,
        "gsm8k_ingest": gsm8k_report,
        "denylist_schema": denylist_schema,
        "denylist_path": str(DEFAULT_DENYLIST_PATH),
        "leakage_report": leak,
        "train": train_meta,
        "val": val_meta,
        "counts": {
            "train_examples": len(train_rows),
            "val_examples": len(val_rows),
            "new_train_examples": len(new_rows),
            "by_source_train": dict(by_src_train),
            "by_source_val": dict(by_src_val),
        },
        "format": "User:/Assistant: with <|endoftext|> after assistant; completion-only masks",
    }
    (out / "provenance.json").write_text(json.dumps(provenance, indent=2) + "\n")
    print(json.dumps({"out": str(out), "counts": provenance["counts"], "leak": leak}, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
