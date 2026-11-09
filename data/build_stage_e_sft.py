#!/usr/bin/env python3
"""Build Stage E instruction-tuning corpus with completion masks.

Sources (verified licenses, recorded in provenance.json):
  - databricks/databricks-dolly-15k (Creative Commons Attribution-ShareAlike 3.0)
  - Programmatically generated arithmetic with exact labels (no external model)
  - openai/gsm8k (MIT) — optional, local JSONL only, annotations re-verified

Round 2 (Nova): group-by-template train/val split (templates never
straddle), G5-norm leakage report recorded in provenance and gated at
exact_user == 0 / exact_assistant == 0, identity truthfulness gate.

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

# Reuse Nova's denylist loader (fail-closed). Apply to the *user* field of
# SFT rows — Nova's corpus builder audits Instruction:/Response: `text`,
# which does not cover this User:/Assistant: train path.
DEFAULT_DENYLIST_PATH = ROOT / "evals" / "behavior" / "behavior_eval.prompts.frozen.json"
NGRAM_WORDS = 13


def _norm(text: str) -> str:
    return " ".join(str(text or "").lower().split())


def _word_ngrams(text: str, n: int = NGRAM_WORDS) -> set[tuple[str, ...]]:
    words = _norm(text).split()
    return {tuple(words[i : i + n]) for i in range(len(words) - n + 1)}


def apply_denylist(rows: list[dict], denylist_texts: list[str]) -> tuple[list[dict], dict]:
    """Drop rows whose user prompt exact-matches or 13-gram-overlaps denylist."""
    deny_exact = {_norm(t) for t in denylist_texts}
    deny_ngrams: set[tuple[str, ...]] = set()
    for t in denylist_texts:
        deny_ngrams |= _word_ngrams(t)
    kept: list[dict] = []
    report = {
        "checked": 0,
        "dropped_denylist_exact": 0,
        "dropped_denylist_ngram": 0,
        "denylist_texts": len(denylist_texts),
    }
    for row in rows:
        report["checked"] += 1
        user = row.get("user") or ""
        if _norm(user) in deny_exact:
            report["dropped_denylist_exact"] += 1
            continue
        grams = _word_ngrams(user)
        if grams and grams & deny_ngrams:
            report["dropped_denylist_ngram"] += 1
            continue
        kept.append(row)
    return kept, report



# ---------------------------------------------------------------------------
# Round 2 (Nova, data lane): group-by-template split + G5 leakage audit.
# Round 1 used a random row split; Claude's G5 audit of sft_candidate found
# exact_user val^train = 1 (Dolly dolly-4758), exact_assistant = 48, and
# 9/294 13-gram hits (all generated-identity) because templates straddled.
# The norm below matches Claude's gate exactly:
#   " ".join(re.findall(r"\w+", text.lower()))
# ---------------------------------------------------------------------------


def g5_norm(text: str) -> str:
    return " ".join(re.findall(r"\w+", str(text or "").lower()))


def _g5_ngrams(text: str, n: int = NGRAM_WORDS) -> set[tuple[str, ...]]:
    words = g5_norm(text).split()
    return {tuple(words[i : i + n]) for i in range(len(words) - n + 1)}


def template_key(row: dict) -> str:
    """Stable grouping key: rows sharing a key never straddle the split."""
    source = row.get("source") or ""
    user = row.get("user") or ""
    if source == "generated-arithmetic":
        # Skeleton keeps operators: G5 norm would erase +,-,* and merge
        # distinct operations into one bucket.
        skeleton = re.sub(r"\d+", "N", " ".join(user.lower().split()))
        return f"arith::{skeleton}"
    if source == "generated-identity":
        # Assistant text is the reused surface for identity rows; grouping
        # by assistant keeps every completion on one side of the split.
        digest = hashlib.sha256(g5_norm(row.get("assistant") or "").encode())
        return f"ident::ans::{digest.hexdigest()[:12]}"
    if source == "databricks/databricks-dolly-15k":
        digest = hashlib.sha256(g5_norm(user).encode()).hexdigest()[:16]
        return f"dolly::{row.get('category') or 'unknown'}::{digest}"
    digest = hashlib.sha256(g5_norm(user).encode()).hexdigest()[:16]
    return f"src::{source}::{digest}"


def dedupe_user_norms(rows: list[dict]) -> tuple[list[dict], int]:
    """Drop later rows whose G5-normalized user already appeared.

    Round 1 deduped exact rendered text only, so the same prompt with a
    different response could land in both splits (the dolly-4758 class).
    """
    seen: set[str] = set()
    kept: list[dict] = []
    dropped = 0
    for row in rows:
        key = g5_norm(row.get("user") or "")
        if key in seen:
            dropped += 1
            continue
        seen.add(key)
        kept.append(row)
    return kept, dropped


def group_split(
    rows: list[dict], val_frac: float, seed: int
) -> tuple[list[dict], list[dict], dict]:
    """Assign whole template buckets to train or val (deterministic).

    Per source, buckets are visited in sha256(seed:key) order and go to
    val until that source's val row target is met, so no template_key
    straddles AND every source keeps val coverage. Note the honest
    consequence for generated-arithmetic: with only 3 skeletons, one
    whole operation can land in val — val then measures a held-out
    operation, not held-out numbers for a seen operation.
    """
    buckets: dict[str, list[dict]] = {}
    for row in rows:
        buckets.setdefault(template_key(row), []).append(row)
    by_source: dict[str, list[str]] = {}
    for key, members in buckets.items():
        by_source.setdefault(members[0].get("source") or "", []).append(key)
    train: list[dict] = []
    val: list[dict] = []
    val_keys: set[str] = set()
    for source in sorted(by_source):
        keys = sorted(
            by_source[source],
            key=lambda k: hashlib.sha256(f"{seed}:{k}".encode()).hexdigest(),
        )
        source_rows = sum(len(buckets[k]) for k in keys)
        target = int(round(source_rows * val_frac))
        filled = 0
        for key in keys:
            if filled < target:
                val.extend(buckets[key])
                val_keys.add(key)
                filled += len(buckets[key])
            else:
                train.extend(buckets[key])
    info = {
        "method": "group_by_template",
        "template_key_version": 1,
        "buckets": len(buckets),
        "val_buckets": len(val_keys),
        "train_rows": len(train),
        "val_rows": len(val),
    }
    return train, val, info


def leakage_report(train_rows: list[dict], val_rows: list[dict]) -> dict:
    """Claude-G5-equivalent leakage metrics (counts only, no text)."""
    train_users = {g5_norm(r.get("user") or "") for r in train_rows}
    train_answers = {g5_norm(r.get("assistant") or "") for r in train_rows}
    train_grams: set[tuple[str, ...]] = set()
    for r in train_rows:
        train_grams |= _g5_ngrams(r.get("user") or "")

    def _by_source(rows, keyset, field):
        out: dict[str, int] = {}
        total = 0
        for r in rows:
            if g5_norm(r.get(field) or "") in keyset:
                total += 1
                src = r.get("source") or "unknown"
                out[src] = out.get(src, 0) + 1
        return total, out

    exact_user, _ = _by_source(val_rows, train_users, "user")
    exact_asst, exact_asst_by_source = _by_source(val_rows, train_answers, "assistant")
    ngram_hits = 0
    long_enough = 0
    ngram_by_source: dict[str, int] = {}
    for r in val_rows:
        grams = _g5_ngrams(r.get("user") or "")
        if not grams:
            continue
        long_enough += 1
        if grams & train_grams:
            ngram_hits += 1
            src = r.get("source") or "unknown"
            ngram_by_source[src] = ngram_by_source.get(src, 0) + 1
    return {
        "norm": "g5",
        "exact_user": exact_user,
        "exact_assistant": exact_asst,
        "exact_assistant_by_source": exact_asst_by_source,
        "ngram13_hits": ngram_hits,
        "ngram13_long_enough": long_enough,
        "ngram13_by_source": ngram_by_source,
    }


def drop_val_answer_collisions(
    train_rows: list[dict], val_rows: list[dict]
) -> tuple[list[dict], dict]:
    """Fail-closed cleanup: drop val rows whose assistant text (G5 norm)
    already appears in train. Counts by source; guarantees the post-split
    exact_assistant metric is 0 rather than merely reported."""
    train_answers = {g5_norm(r.get("assistant") or "") for r in train_rows}
    kept: list[dict] = []
    dropped: dict[str, int] = {}
    for r in val_rows:
        if g5_norm(r.get("assistant") or "") in train_answers:
            src = r.get("source") or "unknown"
            dropped[src] = dropped.get(src, 0) + 1
            continue
        kept.append(r)
    return kept, dropped


# ---------------------------------------------------------------------------
# Identity truthfulness gate: identity answers must come from the approved
# PlaygroundLM set; no invented human personas anywhere in the corpus.
# ---------------------------------------------------------------------------

APPROVED_IDENTITY_ANSWERS = {
    g5_norm(a)
    for a in (
        "I am PlaygroundLM, a small model built by the AI Playground team",
        "I am PlaygroundLM, a ~124M-parameter language model trained from scratch by the Galactic AI Team.",
        "PlaygroundLM-124M, a from-scratch GPT-2-small-class model. I am not ChatGPT.",
        "No. I am PlaygroundLM, a smaller custom model trained by the Galactic AI Team.",
    )
}
_IDENTITY_QUESTION = re.compile(
    r"(who are you|your name|what model|which model|are you chatgpt|"
    r"is this chatgpt|introduce yourself|identify yourself)",
    re.IGNORECASE,
)
_HUMAN_PERSONA = re.compile(
    r"\bmy name is (?!playgroundlm\b)[A-Za-z]+", re.IGNORECASE
)
_HUMAN_CLAIM = re.compile(r"\bi am a human\b", re.IGNORECASE)


def validate_identity_rows(rows: list[dict]) -> list[str]:
    """Return ids of rows violating the identity rules (empty = clean).

    Scope: whenever the *user* asks an identity question, the completion
    must not claim a human name/persona; generated-identity rows must
    additionally answer from the approved PlaygroundLM set. Fictional
    first-person text answering non-identity prompts (e.g. a Dolly story)
    is out of scope — the risk surface is the model claiming an identity
    when asked who it is.
    """
    violations: list[str] = []
    for row in rows:
        if not _IDENTITY_QUESTION.search(row.get("user") or ""):
            continue
        assistant = row.get("assistant") or ""
        if _HUMAN_PERSONA.search(assistant) or _HUMAN_CLAIM.search(assistant):
            violations.append(row.get("id") or "?")
            continue
        if (row.get("source") == "generated-identity"
                and g5_norm(assistant) not in APPROVED_IDENTITY_ANSWERS):
            violations.append(row.get("id") or "?")
    return violations


# ---------------------------------------------------------------------------
# GSM8K ingestion (MIT, openai/gsm8k): local JSONL only — no download here.
# Calculator annotations <<expr=result>> are re-evaluated exactly; a row is
# kept only if every annotation checks out and the #### final answer parses.
# ---------------------------------------------------------------------------

_ANNOTATION = re.compile(r"<<([^=<>]+)=([^<>]+)>>")


def _safe_eval_expr(expr: str) -> float:
    import ast
    import operator as op

    ops = {ast.Add: op.add, ast.Sub: op.sub, ast.Mult: op.mul,
           ast.Div: op.truediv, ast.USub: op.neg, ast.UAdd: op.pos}

    def ev(node):
        if isinstance(node, ast.Expression):
            return ev(node.body)
        if isinstance(node, ast.Constant) and isinstance(node.value, (int, float)):
            return node.value
        if isinstance(node, ast.BinOp) and type(node.op) in ops:
            return ops[type(node.op)](ev(node.left), ev(node.right))
        if isinstance(node, ast.UnaryOp) and type(node.op) in ops:
            return ops[type(node.op)](ev(node.operand))
        raise ValueError("unsafe expression")

    return ev(ast.parse(expr, mode="eval"))


def verify_gsm8k_answer(answer_text: str) -> str | None:
    """Return the final answer string iff all <<expr=result>> annotations
    evaluate exactly and a #### final answer is present; else None."""
    for expr, claimed in _ANNOTATION.findall(answer_text or ""):
        try:
            value = _safe_eval_expr(expr.strip())
            claimed_value = float(claimed.replace(",", "").strip())
        except Exception:
            return None
        if abs(value - claimed_value) > 1e-9:
            return None
    if "####" not in (answer_text or ""):
        return None
    final = answer_text.rsplit("####", 1)[1].strip().replace(",", "").replace("$", "")
    try:
        float(final)
    except ValueError:
        return None
    return final


def load_gsm8k(path: Path, limit: int) -> tuple[list[dict], dict]:
    """Load a local GSM8K JSONL export ({"question", "answer"} rows)."""
    rows: list[dict] = []
    report = {"read": 0, "kept": 0, "dropped_failed_verification": 0}
    with Path(path).open(encoding="utf-8") as f:
        for i, line in enumerate(f):
            if not line.strip():
                continue
            report["read"] += 1
            raw = json.loads(line)
            final = verify_gsm8k_answer(raw.get("answer") or "")
            question = normalize_space(raw.get("question"))
            if final is None or not question:
                report["dropped_failed_verification"] += 1
                continue
            rows.append(
                {
                    "id": f"gsm8k-{i}",
                    "source": "openai/gsm8k",
                    "license": "MIT",
                    "category": "gsm8k-word-problem",
                    "user": question,
                    "assistant": final,
                    "text": format_example(question, final),
                }
            )
            report["kept"] += 1
            if limit and len(rows) >= limit:
                break
    return rows, report


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
    """Exact-label arithmetic with diversified phrasing (Round 2).

    Round-1 used three "What is N op N?" skeletons; group-by-template then
    dumped one whole operator into val. More phrasings → more template
    buckets → val gets a meaningful held-out count closer to val_frac.

    Answers are unique across the arith set so val buckets survive
    drop_val_answer_collisions (short shared ints would otherwise wipe val).
    """
    rng = random.Random(seed + 17)
    phrasing = {
        "add": [
            lambda a, b: (f"What is {a} + {b}?", str(a + b)),
            lambda a, b: (f"Compute {a} plus {b}.", str(a + b)),
            lambda a, b: (f"Add {a} and {b}.", str(a + b)),
            lambda a, b: (f"{a} + {b} = ?", str(a + b)),
            lambda a, b: (f"Find the sum of {a} and {b}.", str(a + b)),
        ],
        "sub": [
            lambda a, b: (f"What is {a} - {b}?", str(a - b)),
            lambda a, b: (f"Compute {a} minus {b}.", str(a - b)),
            lambda a, b: (f"Subtract {b} from {a}.", str(a - b)),
            lambda a, b: (f"{a} - {b} = ?", str(a - b)),
            lambda a, b: (f"Find the difference {a} − {b}.", str(a - b)),
        ],
        "mul": [
            lambda a, b: (f"What is {a} * {b}?", str(a * b)),
            lambda a, b: (f"Compute {a} times {b}.", str(a * b)),
            lambda a, b: (f"Multiply {a} by {b}.", str(a * b)),
            lambda a, b: (f"{a} × {b} = ?", str(a * b)),
            lambda a, b: (f"Find the product of {a} and {b}.", str(a * b)),
        ],
    }
    kinds = list(phrasing)
    rows: list[dict] = []
    used_answers: set[str] = set()
    attempts = 0
    i = 0
    while len(rows) < n and attempts < n * 40:
        attempts += 1
        kind = kinds[i % len(kinds)]
        templates = phrasing[kind]
        tmpl = templates[(i // len(kinds)) % len(templates)]
        if kind == "mul":
            a, b = rng.randint(2, 48), rng.randint(2, 48)
        else:
            a, b = rng.randint(0, 4000), rng.randint(0, 4000)
            if kind == "sub" and b > a:
                a, b = b, a
        user, ans = tmpl(a, b)
        if ans in used_answers:
            i += 1
            continue
        used_answers.add(ans)
        rows.append(
            {
                "id": f"arith-{len(rows)}",
                "source": "generated-arithmetic",
                "license": "MIT (this repository)",
                "category": f"arithmetic-{kind}",
                "user": user,
                "assistant": ans,
                "text": format_example(user, ans),
            }
        )
        i += 1
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
                "license": "MIT (this repository)",
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
    ap.add_argument(
        "--eval-prompts",
        type=Path,
        default=DEFAULT_DENYLIST_PATH,
        help="denylist JSON (default: behavior suite frozen prompts[].user)",
    )
    ap.add_argument(
        "--allow-empty-denylist",
        action="store_true",
        help="build even if denylist missing/malformed/empty (not for release)",
    )
    ap.add_argument(
        "--gsm8k-jsonl",
        type=Path,
        default=None,
        help="local openai/gsm8k JSONL export to ingest (MIT; no download)",
    )
    ap.add_argument("--gsm8k-limit", type=int, default=1000)
    args = ap.parse_args()

    import sys

    sys.path.insert(0, str(ROOT / "src"))
    from playground_lm.tokenizer import get_tokenizer

    tokenizer = get_tokenizer()
    rows = []
    rows.extend(load_dolly(args.dolly_limit, args.seed))
    rows.extend(gen_arithmetic(args.arith_n, args.seed))
    rows.extend(gen_identity_and_follow(args.identity_n, args.seed))
    gsm8k_report = None
    if args.gsm8k_jsonl is not None:
        gsm8k_rows, gsm8k_report = load_gsm8k(args.gsm8k_jsonl, args.gsm8k_limit)
        rows.extend(gsm8k_rows)

    # Identity truthfulness gate (fail closed): approved PlaygroundLM
    # answers only, no invented human personas anywhere.
    violations = validate_identity_rows(rows)
    if violations:
        raise SystemExit(
            f"identity gate failed for {len(violations)} row(s): {violations[:10]}"
        )

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
    # Round 2: also dedupe by G5-normalized user (same prompt, different
    # response must not straddle the split — the dolly-4758 class).
    rows, user_dedup_dropped = dedupe_user_norms(rows)

    # Fail-closed denylist (Nova schema) on the *user* field — the exact
    # corpus that becomes train.bin / *.masks.bin (not JSONL-only path).
    import sys as _sys
    _sys.path.insert(0, str(ROOT / "data"))
    from build_stage_e_corpus import load_denylist  # noqa: E402

    denylist_path = Path(args.eval_prompts)
    denylist_texts, denylist_schema = load_denylist(
        denylist_path, required=not args.allow_empty_denylist
    )
    rows, deny_report = apply_denylist(rows, denylist_texts)
    deny_report["denylist_path"] = str(denylist_path)
    deny_report["denylist_schema"] = denylist_schema

    # Round 2: group-by-template split — a template bucket lands whole in
    # train or val, so templates never straddle (random row split caused
    # the round-1 G5 failures).
    train_rows, val_rows, split_info = group_split(rows, args.val_frac, args.seed)
    # Fail-closed cleanup: no val completion may repeat a train completion.
    val_rows, answer_drops = drop_val_answer_collisions(train_rows, val_rows)
    split_info["val_answer_collision_drops"] = answer_drops
    split_info["val_rows_after_cleanup"] = len(val_rows)
    leak = leakage_report(train_rows, val_rows)
    if leak["exact_user"] != 0 or leak["exact_assistant"] != 0:
        raise SystemExit(
            f"leakage gate failed: exact_user={leak['exact_user']} "
            f"exact_assistant={leak['exact_assistant']} (must be 0/0)"
        )

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
                "license": "MIT (this repository)",
                "notes": "Programmatic Q/A with exact numeric labels; not model-generated. License aligned with Nova corpus builder.",
            },
            {
                "id": "generated-identity",
                "license": "MIT (this repository)",
                "notes": "Short PlaygroundLM identity and brief instruction-following examples. License aligned with Nova corpus builder.",
            },
        ],
        "format": "User:/Assistant: with <|endoftext|> after assistant; completion-only masks",
        "exclusions": [
            "No Stage D playground lobby corpus",
            "No evaluation items from HellaSwag/LAMBADA",
            "No unpaid/unknown web scrapes",
            "Frozen behavior_eval.prompts.frozen.json user prompts (denylist exact + 13-gram)",
        ],
        "denylist_audit": deny_report,
        "split": split_info,
        "leakage_report": leak,
        "user_dedupe_dropped": user_dedup_dropped,
        "gsm8k_ingest": gsm8k_report,
        "train": train_meta,
        "val": val_meta,
        "counts": {
            "train_examples": len(train_rows),
            "val_examples": len(val_rows),
            "by_source_train": {},
            "by_source_val": {},
        },
    }
    if gsm8k_report is not None:
        provenance["sources"].append(
            {
                "id": "openai/gsm8k",
                "url": "https://huggingface.co/datasets/openai/gsm8k",
                "license": "MIT",
                "notes": "Local JSONL export only; every <<expr=result>> calculator annotation re-evaluated exactly at ingest; rows failing verification dropped.",
            }
        )
    for row in train_rows:
        provenance["counts"]["by_source_train"][row["source"]] = (
            provenance["counts"]["by_source_train"].get(row["source"], 0) + 1
        )
    for row in val_rows:
        provenance["counts"]["by_source_val"][row["source"]] = (
            provenance["counts"]["by_source_val"].get(row["source"], 0) + 1
        )
    (out / "provenance.json").write_text(json.dumps(provenance, indent=2) + "\n")
    print(json.dumps(provenance["counts"], indent=2))
    print("train_tokens", train_meta["tokens"], "supervised", train_meta["supervised_tokens"])
    print("val_tokens", val_meta["tokens"], "supervised", val_meta["supervised_tokens"])
    print("wrote", out)


if __name__ == "__main__":
    main()
