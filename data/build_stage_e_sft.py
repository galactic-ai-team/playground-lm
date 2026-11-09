#!/usr/bin/env python3
"""Build Stage E instruction-tuning corpus with completion masks.

Sources (verified licenses, recorded in provenance.json):
  - databricks/databricks-dolly-15k (Creative Commons Attribution-ShareAlike 3.0)
  - Programmatically generated arithmetic with exact labels (no external model)

Format matches the chat UI:
  User: <instruction/context/question>
  Assistant: <response><|endoftext|>

Prompt tokens are unsupervised; assistant tokens (+ EOS) are supervised.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import random
import re
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]


def sha256_file(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        for chunk in iter(lambda: f.read(1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()


def normalize_space(text: str) -> str:
    return re.sub(r"\s+", " ", str(text or "")).strip()


def format_example(user: str, assistant: str) -> str:
    user = normalize_space(user)
    assistant = normalize_space(assistant)
    if not user or not assistant:
        raise ValueError("empty turn")
    return f"User: {user}\nAssistant: {assistant}"


def load_dolly(limit: int, seed: int) -> list[dict]:
    from datasets import load_dataset

    ds = load_dataset("databricks/databricks-dolly-15k", split="train")
    rows = []
    for i, row in enumerate(ds):
        instruction = normalize_space(row.get("instruction"))
        context = normalize_space(row.get("context"))
        response = normalize_space(row.get("response"))
        category = row.get("category") or "unknown"
        if not instruction or not response:
            continue
        if len(response) > 800 or len(instruction) > 400:
            continue
        # Skip very long context passages for 124M first pass.
        if context and len(context) > 600:
            continue
        if context:
            user = f"{instruction}\n\nContext: {context}"
        else:
            user = instruction
        rows.append(
            {
                "id": f"dolly-{i}",
                "source": "databricks/databricks-dolly-15k",
                "license": "CC BY-SA 3.0",
                "category": category,
                "user": user,
                "assistant": response,
                "text": format_example(user, response),
            }
        )
    rng = random.Random(seed)
    rng.shuffle(rows)
    return rows[:limit]


def gen_arithmetic(n: int, seed: int) -> list[dict]:
    rng = random.Random(seed + 17)
    ops = {
        "add": lambda a, b: (f"What is {a} + {b}?", str(a + b)),
        "sub": lambda a, b: (f"What is {a} - {b}?", str(a - b)),
        "mul": lambda a, b: (f"What is {a} * {b}?", str(a * b)),
    }
    rows = []
    for i in range(n):
        kind = rng.choice(list(ops))
        if kind == "mul":
            a, b = rng.randint(2, 12), rng.randint(2, 12)
        else:
            a, b = rng.randint(0, 200), rng.randint(0, 200)
            if kind == "sub" and b > a:
                a, b = b, a
        user, ans = ops[kind](a, b)
        # Short exact answer only — no prose filler.
        rows.append(
            {
                "id": f"arith-{i}",
                "source": "generated-arithmetic",
                "license": "CC0-1.0-equivalent-generated",
                "category": f"arithmetic-{kind}",
                "user": user,
                "assistant": ans,
                "text": format_example(user, ans),
            }
        )
    return rows


def gen_identity_and_follow(n: int, seed: int) -> list[dict]:
    rng = random.Random(seed + 99)
    identity_answers = [
        "I am PlaygroundLM, a ~124M-parameter language model trained from scratch by the Galactic AI Team.",
        "PlaygroundLM-124M, a from-scratch GPT-2-small-class model. I am not ChatGPT.",
        "No. I am PlaygroundLM, a smaller custom model trained by the Galactic AI Team.",
    ]
    follow_pairs = [
        ("Reply with only the word OK.", "OK"),
        ("Say hi in exactly three words.", "Hello from PlaygroundLM"),
        ("Write a one-sentence greeting.", "Hello — I am PlaygroundLM, ready to try a short answer."),
        ("Answer with a single number: 7", "7"),
        ("Return only YES.", "YES"),
        ("Spell the word cat.", "cat"),
    ]
    who_prompts = [
        "Who are you?",
        "Who are you today?",
        "Please introduce yourself briefly.",
        "What is your name?",
        "Tell me which model I am talking to.",
        "Identify yourself in one sentence.",
    ]
    model_prompts = [
        "What model are you?",
        "Which model powers this chat?",
        "What is your model size?",
        "Describe the model you are.",
    ]
    chatgpt_prompts = [
        "Are you ChatGPT?",
        "Are you OpenAI ChatGPT?",
        "Do you claim to be ChatGPT?",
        "Is this ChatGPT?",
    ]
    rows = []
    i = 0
    while len(rows) < n:
        bucket = i % 4
        if bucket == 0:
            user = who_prompts[i % len(who_prompts)] + (f" ({i})" if i >= 20 else "")
            assistant = identity_answers[0]
        elif bucket == 1:
            user = model_prompts[i % len(model_prompts)] + (f" [{i}]" if i >= 20 else "")
            assistant = identity_answers[1]
        elif bucket == 2:
            user = chatgpt_prompts[i % len(chatgpt_prompts)] + (f" #{i}" if i >= 20 else "")
            assistant = identity_answers[2]
        else:
            user, assistant = follow_pairs[i % len(follow_pairs)]
            if i >= 20:
                user = f"{user} (request {i})"
        rows.append(
            {
                "id": f"identity-{i}",
                "source": "generated-identity",
                "license": "CC0-1.0-equivalent-generated",
                "category": "identity-follow",
                "user": user,
                "assistant": assistant,
                "text": format_example(user, assistant),
            }
        )
        i += 1
        if i > n * 3:
            break
    return rows[:n]


def tokenize_rows(rows, tokenizer):
    eot = tokenizer.eot_id if hasattr(tokenizer, "eot_id") else 50256
    token_chunks = []
    mask_chunks = []
    stats = {"examples": 0, "prompt_tokens": 0, "completion_tokens": 0}
    for row in rows:
        prompt = f"User: {row['user']}\nAssistant:"
        completion = f" {row['assistant']}"
        p_ids = tokenizer.encode(prompt)
        c_ids = tokenizer.encode(completion) + [eot]
        if len(p_ids) + len(c_ids) < 3:
            continue
        token_chunks.append(np.asarray(p_ids + c_ids, dtype=np.uint16))
        mask_chunks.append(
            np.concatenate(
                [
                    np.zeros(len(p_ids), dtype=np.uint8),
                    np.ones(len(c_ids), dtype=np.uint8),
                ]
            )
        )
        stats["examples"] += 1
        stats["prompt_tokens"] += len(p_ids)
        stats["completion_tokens"] += len(c_ids)
    if not token_chunks:
        raise RuntimeError("no tokenized examples")
    tokens = np.concatenate(token_chunks)
    masks = np.concatenate(mask_chunks)
    return tokens, masks, stats


def write_split(out_dir: Path, split: str, rows, tokenizer):
    tokens, masks, stats = tokenize_rows(rows, tokenizer)
    tok_path = out_dir / f"{split}.bin"
    mask_path = out_dir / f"{split}.masks.bin"
    jsonl_path = out_dir / f"{split}.jsonl"
    tokens.tofile(tok_path)
    masks.tofile(mask_path)
    with jsonl_path.open("w", encoding="utf-8") as f:
        for row in rows:
            f.write(json.dumps(row, ensure_ascii=False) + "\n")
    meta = {
        **stats,
        "tokens": int(tokens.size),
        "supervised_tokens": int(masks.sum()),
        "sha256_tokens": sha256_file(tok_path),
        "sha256_masks": sha256_file(mask_path),
    }
    return meta


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", type=Path, default=ROOT / "data" / "stage_e_sft")
    ap.add_argument("--seed", type=int, default=20261108)
    ap.add_argument("--dolly-limit", type=int, default=8000)
    ap.add_argument("--arith-n", type=int, default=800)
    ap.add_argument("--identity-n", type=int, default=120)
    ap.add_argument("--val-frac", type=float, default=0.05)
    args = ap.parse_args()

    import sys

    sys.path.insert(0, str(ROOT / "src"))
    from playground_lm.tokenizer import get_tokenizer

    tokenizer = get_tokenizer()
    rows = []
    rows.extend(load_dolly(args.dolly_limit, args.seed))
    rows.extend(gen_arithmetic(args.arith_n, args.seed))
    rows.extend(gen_identity_and_follow(args.identity_n, args.seed))

    rng = random.Random(args.seed)
    rng.shuffle(rows)
    # Dedup exact texts
    seen = set()
    uniq = []
    for row in rows:
        key = row["text"]
        if key in seen:
            continue
        seen.add(key)
        uniq.append(row)
    rows = uniq

    n_val = max(50, int(len(rows) * args.val_frac))
    val_rows = rows[:n_val]
    train_rows = rows[n_val:]

    out = args.out
    out.mkdir(parents=True, exist_ok=True)
    train_meta = write_split(out, "train", train_rows, tokenizer)
    val_meta = write_split(out, "val", val_rows, tokenizer)

    provenance = {
        "seed": args.seed,
        "sources": [
            {
                "id": "databricks/databricks-dolly-15k",
                "url": "https://huggingface.co/datasets/databricks/databricks-dolly-15k",
                "license": "CC BY-SA 3.0",
                "notes": "Filtered to short English instruction/response pairs; long contexts dropped.",
            },
            {
                "id": "generated-arithmetic",
                "license": "generated exact-label arithmetic",
                "notes": "Programmatic Q/A with exact numeric labels; not model-generated.",
            },
            {
                "id": "generated-identity",
                "license": "generated",
                "notes": "Short PlaygroundLM identity and brief instruction-following examples.",
            },
        ],
        "format": "User:/Assistant: with <|endoftext|> after assistant; completion-only masks",
        "exclusions": [
            "No Stage D playground lobby corpus",
            "No evaluation items from HellaSwag/LAMBADA",
            "No unpaid/unknown web scrapes",
        ],
        "train": train_meta,
        "val": val_meta,
        "counts": {
            "train_examples": len(train_rows),
            "val_examples": len(val_rows),
            "by_source_train": {},
        },
    }
    for row in train_rows:
        provenance["counts"]["by_source_train"][row["source"]] = (
            provenance["counts"]["by_source_train"].get(row["source"], 0) + 1
        )
    (out / "provenance.json").write_text(json.dumps(provenance, indent=2) + "\n")
    print(json.dumps(provenance["counts"], indent=2))
    print("train_tokens", train_meta["tokens"], "supervised", train_meta["supervised_tokens"])
    print("val_tokens", val_meta["tokens"], "supervised", val_meta["supervised_tokens"])
    print("wrote", out)


if __name__ == "__main__":
    main()
