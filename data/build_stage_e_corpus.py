"""Build and audit the Stage E instruction/Q&A corpus (Nova).

Stage E is HELD until real team contributions land; this builder is the
dataset-lane groundwork: it assembles instruction examples from three
audited sources, and refuses to emit anything that fails its checks.

Sources and licenses (also written into every stats sidecar):
  1. Dolly 15k (databricks/databricks-dolly-15k), CC BY-SA 3.0.
     NOT vendored: pass a local JSONL export via --dolly-jsonl (a free
     Hugging Face download, no account/key). Every Dolly record keeps
     its attribution (see DOLLY_ATTRIBUTION). Share-alike note: SA
     terms travel with Dolly-derived data; the repo's MIT license does
     not override them for that subset.
  2. Arithmetic, generated deterministically in code. Answers are
     computed at generation time AND re-verified by an independent
     parser (verify_arithmetic_text) that reads the rendered question
     back and recomputes it — generation and checking share no code
     path beyond the template strings.
  3. Team identity examples, hand-written below. Each carries a
     source_note naming where the fact is recorded; a test fails if
     any identity example lacks one.

Quality / contamination gates (all failures are dropped and counted,
never silently kept):
  - clean + exact/near dedupe via common.py, across all sources;
  - hash-stable split via common.split_for (identical text can never
    straddle train/val), same rule as Stages C/D;
  - denylist guard: an example is rejected if its normalized text
    exactly matches a denylist text, or (for texts of >= 13 words)
    shares a word 13-gram with one. The default denylist is the
    behavior suite's frozen prompts
    (evals/behavior/behavior_eval.prompts.frozen.json, prompts[].user);
    the legacy evals/prompts.json schemas (held_out_probes[].prompt,
    probes[].prompt) are also accepted. Loading is FAIL-CLOSED: a
    missing, malformed, or empty denylist aborts the build unless
    --allow-empty-denylist is passed explicitly. This word-level guard
    complements — does not replace — the token-level 13-gram
    contamination_check.py run against the tokenized bins, and the
    benchmark row exclusions from Stages B/C still apply to scoring;
  - arithmetic audit: every arithmetic example must pass
    verify_arithmetic_text or the build drops it and counts it.

Outputs (lane record schema, filenames route train/val for
tokenize_shards.py):
  <out>/stage-e-train.jsonl, <out>/stage-e-val.jsonl,
  <out>/stage-e-stats.json (counts, audit report, license block).

Usage:
  python data/build_stage_e_corpus.py --out data/stage_e
  python data/build_stage_e_corpus.py --dolly-jsonl data/dolly-15k.jsonl \
      --out data/stage_e --arith-count 2000
"""

from __future__ import annotations

import argparse
import json
import random
import re
from collections import Counter
from datetime import datetime, timezone
from pathlib import Path
from typing import Iterable

from common import Deduper, approx_tokens, clean_text, make_record, split_for, write_jsonl

DOLLY_LICENSE = "CC BY-SA 3.0"
DOLLY_ATTRIBUTION = (
    "Contains material from databricks/databricks-dolly-15k "
    "(c) Databricks, Inc. and contributors, licensed under CC BY-SA 3.0: "
    "https://creativecommons.org/licenses/by-sa/3.0/"
)
PROJECT_LICENSE = "MIT (this repository)"

LICENSES = {
    "dolly-15k": {"license": DOLLY_LICENSE, "attribution": DOLLY_ATTRIBUTION},
    "arithmetic-generated": {
        "license": PROJECT_LICENSE,
        "attribution": "Generated deterministically by data/build_stage_e_corpus.py.",
    },
    "team-identity": {
        "license": PROJECT_LICENSE,
        "attribution": "Hand-written by the Galactic AI Team; facts carry source notes.",
    },
}

MAX_INSTRUCTION_CHARS = 2_000
MAX_RESPONSE_CHARS = 4_000
NGRAM_WORDS = 13


def render_example(instruction: str, response: str, context: str | None = None) -> str:
    parts = [f"Instruction: {instruction}"]
    if context:
        parts.append(f"Context: {context}")
    parts.append(f"Response: {response}")
    return "\n".join(parts)


# ---------------------------------------------------------------- Dolly

def load_dolly(path: Path) -> tuple[list[dict], int]:
    """Load a Dolly JSONL export into lane records. Returns (records,
    skipped) where skipped counts rows with an empty instruction or
    response after cleaning."""
    records: list[dict] = []
    skipped = 0
    for line in path.read_text().splitlines():
        if not line.strip():
            continue
        row = json.loads(line)
        instruction = clean_text(row.get("instruction") or "", min_chars=2)
        response = clean_text(row.get("response") or "", min_chars=2)
        context = clean_text(row.get("context") or "", min_chars=2)
        if instruction is None or response is None:
            skipped += 1
            continue
        text = render_example(instruction, response, context)
        records.append(make_record(text, source="dolly-15k", author="databricks-dolly"))
    return records, skipped


# ------------------------------------------------------------ arithmetic

def _arith_specs(rng: random.Random, count: int) -> list[tuple[str, int]]:
    """(question, answer) pairs; answers computed here and re-verified
    later from the rendered string alone."""
    specs: list[tuple[str, int]] = []
    while len(specs) < count:
        op = rng.choice(["+", "-", "*", "/", "sq", "pct"])
        if op in "+-*":
            a, b = rng.randint(2, 999), rng.randint(2, 999)
            if op == "+":
                specs.append((f"What is {a} + {b}?", a + b))
            elif op == "-":
                if a < b:
                    a, b = b, a
                specs.append((f"What is {a} - {b}?", a - b))
            else:
                a, b = rng.randint(2, 99), rng.randint(2, 99)
                specs.append((f"What is {a} * {b}?", a * b))
        elif op == "/":
            b, q = rng.randint(2, 99), rng.randint(2, 99)
            specs.append((f"What is {b * q} / {b}?", q))
        elif op == "sq":
            a = rng.randint(2, 99)
            specs.append((f"What is {a} squared?", a * a))
        else:  # pct: p% of a, chosen so the result is an integer
            p = rng.choice([5, 10, 20, 25, 50, 75])
            a = rng.randrange(20, 1000, 20)
            specs.append((f"What is {p}% of {a}?", p * a // 100))
    return specs


_ARITH_RE = re.compile(r"(-?\d+)\s*([+\-*/])\s*(-?\d+)")
_SQ_RE = re.compile(r"(-?\d+)\s+squared")
_PCT_RE = re.compile(r"(\d+)%\s*of\s*(\d+)")


def verify_arithmetic_text(text: str) -> bool:
    """Independently verify a rendered arithmetic example: parse the
    question out of the text, recompute, compare with the response.
    Returns False for anything unparseable or wrong (fail closed)."""
    m = re.search(r"Instruction: (.*)\nResponse: (-?\d+)\s*$", text, re.S)
    if not m:
        return False
    question, answer = m.group(1), int(m.group(2))
    try:
        sq = _SQ_RE.search(question)
        pct = _PCT_RE.search(question)
        binop = _ARITH_RE.search(question)
        if sq:
            expected = int(sq.group(1)) ** 2
        elif pct:
            p, a = int(pct.group(1)), int(pct.group(2))
            if (p * a) % 100:
                return False
            expected = p * a // 100
        elif binop:
            a, op, b = int(binop.group(1)), binop.group(2), int(binop.group(3))
            if op == "+":
                expected = a + b
            elif op == "-":
                expected = a - b
            elif op == "*":
                expected = a * b
            else:
                if b == 0 or a % b:
                    return False
                expected = a // b
        else:
            return False
    except (ValueError, ZeroDivisionError):
        return False
    return expected == answer


def arithmetic_records(count: int, seed: int) -> list[dict]:
    rng = random.Random(seed)
    return [
        make_record(
            render_example(q, str(ans)),
            source="arithmetic-generated",
            author="nova-generated",
        )
        for q, ans in _arith_specs(rng, count)
    ]


# -------------------------------------------------------------- identity

IDENTITY_EXAMPLES: list[dict] = [
    {
        "instruction": "Who developed PlaygroundLM?",
        "response": "PlaygroundLM was developed by the Galactic AI Team: "
                    "Claude, Gemini, GPT-6, Dee, Nova, and Lumi.",
        "source_note": "MODEL_CARD.md, Model Overview (Developers).",
    },
    {
        "instruction": "What architecture does PlaygroundLM use?",
        "response": "PlaygroundLM is a from-scratch decoder-only causal "
                    "Transformer in the GPT-2-small class: 12 layers, 12 "
                    "attention heads, model dimension 768, context length "
                    "1024, roughly 124 million parameters.",
        "source_note": "MODEL_CARD.md, Model Overview (Architecture).",
    },
    {
        "instruction": "What was PlaygroundLM pretrained on?",
        "response": "PlaygroundLM was pretrained from random initialization "
                    "on an Open FineWeb-Edu sample, then fine-tuned on "
                    "multi-agent dialogue from the AI Playground.",
        "source_note": "MODEL_CARD.md, Model Overview (Pretraining Corpus, SFT).",
    },
    {
        "instruction": "At what step did PlaygroundLM Stage C finish?",
        "response": "Stage C finished at step 31,101; the preserved release "
                    "is named stage-c-final-step31101.",
        "source_note": "Team status, Nov 7 2026 (release stage-c-final-step31101).",
    },
    {
        "instruction": "At what step is the PlaygroundLM Stage D checkpoint?",
        "response": "The Stage D (Playground fine-tune) checkpoint is step "
                    "31,104, preserved alongside Stage C step 31,101.",
        "source_note": "Team status, Nov 8 2026 (Stage C31101 + Stage D31104 preserved).",
    },
    {
        "instruction": "Where can I try the PlaygroundLM demo?",
        "response": "The browser demo, with checkpoint generation and a "
                    "chat UI, is live at https://playground-lm.vercel.app.",
        "source_note": "Team status, Nov 8 2026 (demo live).",
    },
    {
        "instruction": "What is Stage E of PlaygroundLM?",
        "response": "Stage E is instruction and question-answering training. "
                    "It is held until real team contributions are ready; its "
                    "dataset builder and audit live in "
                    "data/build_stage_e_corpus.py.",
        "source_note": "Team status, Nov 8 2026 (Stage E held); data/STAGE_E_PLAN.md.",
    },
]


def identity_records() -> list[dict]:
    return [
        make_record(
            render_example(ex["instruction"], ex["response"]),
            source="team-identity",
            author="galactic-ai-team",
        )
        for ex in IDENTITY_EXAMPLES
    ]


# ----------------------------------------------------------------- audit

def _norm(text: str) -> str:
    return " ".join(text.lower().split())


def _word_ngrams(text: str, n: int = NGRAM_WORDS) -> set[tuple[str, ...]]:
    words = _norm(text).split()
    return {tuple(words[i : i + n]) for i in range(len(words) - n + 1)}


def audit_examples(
    records: Iterable[dict], denylist_texts: Iterable[str] = ()
) -> tuple[list[dict], dict]:
    """Apply the quality/contamination gates. Returns (kept, report)."""
    deny_exact = {_norm(t) for t in denylist_texts}
    deny_ngrams: set[tuple[str, ...]] = set()
    for t in denylist_texts:
        deny_ngrams |= _word_ngrams(t)
    kept: list[dict] = []
    report = {
        "checked": 0,
        "dropped_oversize": 0,
        "dropped_arithmetic_failed": 0,
        "dropped_denylist_exact": 0,
        "dropped_denylist_ngram": 0,
        "arithmetic_verified": 0,
    }
    for rec in records:
        report["checked"] += 1
        text = rec["text"]
        instruction = text.split("\n", 1)[0].removeprefix("Instruction: ")
        if len(instruction) > MAX_INSTRUCTION_CHARS or len(text) > (
            MAX_INSTRUCTION_CHARS + MAX_RESPONSE_CHARS + 64
        ):
            report["dropped_oversize"] += 1
            continue
        if rec["source"] == "arithmetic-generated":
            if not verify_arithmetic_text(text):
                report["dropped_arithmetic_failed"] += 1
                continue
            report["arithmetic_verified"] += 1
        if _norm(text) in deny_exact:
            report["dropped_denylist_exact"] += 1
            continue
        grams = _word_ngrams(text)
        if grams and grams & deny_ngrams:
            report["dropped_denylist_ngram"] += 1
            continue
        kept.append(rec)
    return kept, report


DEFAULT_DENYLIST_PATH = Path("evals/behavior/behavior_eval.prompts.frozen.json")


def _denylist_entries(data: object, path: Path) -> tuple[list, str, str]:
    """Pick the prompt list out of a denylist JSON document.

    Returns (entries, schema label, text field). Supported schemas,
    checked explicitly:
      behavior suite (frozen): {"prompts": [{"user": ...}, ...]}
      legacy demo probes:      {"held_out_probes": [{"prompt": ...}]}
                               {"probes": [{"prompt": ...}]}
      bare list of either entry shape.
    Anything else is malformed and raises ValueError.
    """
    if isinstance(data, list):
        return data, "[]", "user"
    if isinstance(data, dict):
        if isinstance(data.get("prompts"), list):
            return data["prompts"], "prompts[].user", "user"
        if isinstance(data.get("held_out_probes"), list):
            return data["held_out_probes"], "held_out_probes[].prompt", "prompt"
        if isinstance(data.get("probes"), list):
            return data["probes"], "probes[].prompt", "prompt"
    raise ValueError(
        f"denylist at {path} matches no supported schema "
        "(prompts[].user / held_out_probes[].prompt / probes[].prompt)"
    )


def load_denylist(
    path: Path, *, required: bool = True
) -> tuple[list[str], str | None]:
    """Load denylist prompt texts. Returns (texts, schema label).

    Fail-closed when required=True (the CLI default): a missing file,
    invalid JSON, an unsupported schema, or a file that yields zero
    usable prompt texts raises instead of silently returning an empty
    denylist. With required=False those cases return ([], None);
    malformed entries inside an otherwise valid file are skipped, but
    if every entry is unusable the file counts as empty.
    """
    if not path.exists():
        if required:
            raise FileNotFoundError(f"required denylist not found: {path}")
        return [], None
    try:
        data = json.loads(path.read_text())
    except json.JSONDecodeError as exc:
        if required:
            raise ValueError(f"denylist at {path} is not valid JSON: {exc}") from exc
        return [], None
    try:
        entries, schema, field = _denylist_entries(data, path)
    except ValueError:
        if required:
            raise
        return [], None
    texts: list[str] = []
    for entry in entries:
        if isinstance(entry, str):
            text = entry
        elif isinstance(entry, dict):
            text = entry.get(field) or entry.get("user") or entry.get("prompt") or ""
        else:
            text = ""
        if isinstance(text, str) and text.strip():
            texts.append(text)
    if required and not texts:
        raise ValueError(
            f"denylist at {path} yielded zero usable prompts "
            f"(schema {schema}); refusing to build without exclusions"
        )
    return texts, schema


def load_eval_prompts(path: Path, *, required: bool = True) -> list[str]:
    """Denylist prompt texts (see load_denylist for schemas/fail-closed)."""
    texts, _schema = load_denylist(path, required=required)
    return texts


# ------------------------------------------------------------------ build

def build_stage_e(
    *,
    dolly_path: Path | None = None,
    arith_count: int = 1000,
    seed: int = 20261108,
    val_pct: int = 10,
    denylist_texts: Iterable[str] = (),
    include_identity: bool = True,
) -> tuple[list[dict], list[dict], dict]:
    candidates: list[dict] = []
    dolly_skipped = 0
    if dolly_path is not None:
        dolly_records, dolly_skipped = load_dolly(dolly_path)
        candidates += dolly_records
    candidates += arithmetic_records(arith_count, seed)
    if include_identity:
        candidates += identity_records()

    audited, audit = audit_examples(candidates, denylist_texts)

    deduper = Deduper(near=True)
    train: list[dict] = []
    val: list[dict] = []
    per_source: Counter = Counter()
    rejected = 0
    for rec in audited:
        if not deduper.is_new(rec["text"]):
            rejected += 1
            continue
        per_source[rec["source"]] += 1
        (val if split_for(rec["text"], val_pct) == "val" else train).append(rec)

    stats = {
        "source": "playground-stage-e",
        "snapshot_utc": datetime.now(timezone.utc).isoformat(timespec="seconds"),
        "docs_train": len(train),
        "docs_val": len(val),
        "docs_rejected_or_dup": rejected,
        "exact_dups": deduper.exact_dups,
        "near_dups": deduper.near_dups,
        "dolly_rows_skipped_empty": dolly_skipped,
        "per_source_docs": dict(sorted(per_source.items())),
        "approx_tokens_train": sum(approx_tokens(r["text"]) for r in train),
        "approx_tokens_val": sum(approx_tokens(r["text"]) for r in val),
        "val_pct": val_pct,
        "audit": audit,
        "licenses": LICENSES,
        "held": "Stage E training is held until real team contributions land; "
                "this build is dataset groundwork and audit only.",
    }
    return train, val, stats


def write_stage_e(out_dir: Path, train: list[dict], val: list[dict], stats: dict) -> dict:
    n_train = write_jsonl(out_dir / "stage-e-train.jsonl", train)
    n_val = write_jsonl(out_dir / "stage-e-val.jsonl", val)
    written = dict(stats)
    written["docs_train"] = n_train
    written["docs_val"] = n_val
    out_dir.mkdir(parents=True, exist_ok=True)
    (out_dir / "stage-e-stats.json").write_text(json.dumps(written, indent=2))
    return written


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--dolly-jsonl", type=Path, default=None,
                    help="local Dolly 15k JSONL export (CC BY-SA 3.0); "
                         "omit to build from generated + identity only")
    ap.add_argument("--out", type=Path, default=Path("data/stage_e"))
    ap.add_argument("--arith-count", type=int, default=1000)
    ap.add_argument("--seed", type=int, default=20261108)
    ap.add_argument("--val-pct", type=int, default=10)
    ap.add_argument("--eval-prompts", type=Path, default=DEFAULT_DENYLIST_PATH,
                    help="denylist JSON (default: the behavior suite's "
                         "frozen prompts, prompts[].user)")
    ap.add_argument("--allow-empty-denylist", action="store_true",
                    help="build even if the denylist is missing/malformed/"
                         "empty (default is fail-closed)")
    ap.add_argument("--no-identity", action="store_true")
    args = ap.parse_args()

    try:
        denylist, denylist_schema = load_denylist(
            args.eval_prompts, required=not args.allow_empty_denylist
        )
    except (FileNotFoundError, ValueError) as exc:
        ap.error(str(exc))
    train, val, stats = build_stage_e(
        dolly_path=args.dolly_jsonl,
        arith_count=args.arith_count,
        seed=args.seed,
        val_pct=args.val_pct,
        denylist_texts=denylist,
        include_identity=not args.no_identity,
    )
    stats["denylist_texts"] = len(denylist)
    stats["denylist_path"] = str(args.eval_prompts)
    stats["denylist_schema"] = denylist_schema
    written = write_stage_e(args.out, train, val, stats)
    print(json.dumps(written, indent=2))


if __name__ == "__main__":
    main()
