#!/usr/bin/env python3
"""Stage G: grounded-facts SFT rows from the fact sheet (Claude).

Teaches the model to answer FROM the facts in its prompt (the demo retrieves
the top facts with BM25 and adds them), in the fixed formats:

    User: Facts: [1] Paris is the capital of France. [2] <fact> [3] <fact> Question: What is the capital of France?
    Assistant: Paris.

    (none of the given facts answers)  Assistant: I'm not sure; the facts I was given don't say.

Input: the fact sheet JSONL (data/facts/fact_sheet.jsonl, Nova), one entry per
line with id, text, keywords, aliases (accepted answer forms, the first is the
canonical reply), topic, verify_url, license, heldout. This builder also needs
`questions`: 2-4 self-written phrasings that the fact answers. Entries
without questions are skipped and counted (we don't invent questions from
fact text).

Rows (shares are of the final row count):
  grounded      ~65%  one row per chosen phrasing (2-4 per fact): the answer
                      fact plus 2 distractor facts, positions shuffled;
  not_in_facts  ~25%  a question whose fact is NOT among the 3 given facts
                      (and no given fact mentions its answer) -> the not-sure line;
  closed_book   ~10%  the question with no Facts: block -> the short answer,
                      to keep v0.5's plain chat form working when retrieval
                      finds nothing.

Held-out facts (heldout: true) are kept out of training rows ENTIRELY: never
the answer, never a distractor, never the subject of a not-sure or
closed-book row. They exist to test reading from context on unseen facts.
Distractors never contain any alias of the answer. `audit_rows` re-checks all
of this; `--g5-val` runs the Stage E leakage report against a val file.

Usage:
  python data/build_stage_g_grounded.py --facts data/facts/fact_sheet.jsonl --out data/stage_g/grounded \\
      [--val-frac 0.1] [--g5-val other_val.jsonl] [--tokenize]
"""
from __future__ import annotations

import argparse
import hashlib
import json
import random
import re
import sys
from pathlib import Path

SOURCE = "grounded-facts"
NOT_SURE = "I'm not sure; the facts I was given don't say."
FACTS_PER_ROW = 3
REQUIRED = ("id", "text", "aliases", "heldout")
LICENSE = "MIT (this repository) rows; fact texts per fact-sheet license field"


def normalize_space(text: str) -> str:
    return re.sub(r"\s+", " ", str(text or "")).strip()


def reply_for(fact: dict) -> str:
    answer = normalize_space(fact["aliases"][0])
    return answer if answer[-1:] in ".!?" else answer + "."


def mentions(text: str, aliases) -> bool:
    """Whole-word, case-insensitive: does `text` contain any alias?"""
    low = normalize_space(text).lower()
    for a in aliases:
        a = normalize_space(a).lower()
        if a and re.search(r"(?<!\w)" + re.escape(a) + r"(?!\w)", low):
            return True
    return False


def facts_prompt(texts: list[str], question: str) -> str:
    listed = " ".join(f"[{i}] {normalize_space(t)}" for i, t in enumerate(texts, 1))
    return f"Facts: {listed} Question: {normalize_space(question)}"


def load_facts(path: Path) -> tuple[list[dict], dict]:
    facts, report = [], {"entries": 0, "skipped_no_questions": 0, "skipped_invalid": 0}
    seen = set()
    for line in Path(path).read_text(encoding="utf-8").splitlines():
        if not line.strip():
            continue
        report["entries"] += 1
        f = json.loads(line)
        if any(k not in f for k in REQUIRED) or not f["aliases"] or f["id"] in seen \
                or not isinstance(f["heldout"], bool):
            report["skipped_invalid"] += 1
            continue
        seen.add(f["id"])
        if not f.get("questions"):
            report["skipped_no_questions"] += 1
            f["questions"] = []
        facts.append(f)
    return facts, report


def _fact_split(fact_id: str, val_frac: float, salt: str = "stage-g-grounded") -> bool:
    h = int(hashlib.sha256(f"{salt}:{fact_id}".encode()).hexdigest()[:8], 16)
    return h / 0xFFFFFFFF < val_frac


def _distractors(rng: random.Random, pool: list[dict], answer: dict, k: int) -> list[dict] | None:
    ok = [f for f in pool if f["id"] != answer["id"] and not mentions(f["text"], answer["aliases"])]
    same = [f for f in ok if f.get("topic") and f.get("topic") == answer.get("topic")]
    picks = rng.sample(same, k) if len(same) >= k else rng.sample(ok, k) if len(ok) >= k else None
    return picks


def build_rows(facts: list[dict], seed: int = 20261110, *, not_in_facts_frac: float = 0.25,
               closed_book_frac: float = 0.10, phrasings: tuple[int, int] = (2, 4),
               distractor_pool: list[dict] | None = None) -> tuple[list[dict], dict]:
    """Rows answering from `facts` (already split). Held-out facts are dropped
    first, from the answers and from the distractor pool (default: the same
    facts; val passes the train facts too, which only ever appear as
    distractors there)."""
    rng = random.Random(seed)
    usable = [f for f in facts if not f["heldout"] and f["questions"]]
    pool_all = [f for f in (distractor_pool if distractor_pool is not None else facts) if not f["heldout"]]
    stats = {"facts_used": len(usable), "heldout_excluded": sum(f["heldout"] for f in facts),
             "grounded": 0, "not_in_facts": 0, "closed_book": 0, "no_distractors": 0}
    rows: list[dict] = []

    def row(kind, user, assistant, answer_id, fact_ids):
        rows.append({"user": normalize_space(user), "assistant": assistant, "source": SOURCE, "kind": kind,
                     "answer_fact": answer_id, "fact_ids": fact_ids, "license": LICENSE})
        stats[kind] += 1

    for fact in usable:
        qs = list(fact["questions"])
        rng.shuffle(qs)
        for q in qs[: rng.randint(min(phrasings[0], len(qs)), min(phrasings[1], len(qs)))]:
            distract = _distractors(rng, pool_all, fact, FACTS_PER_ROW - 1)
            if distract is None:
                stats["no_distractors"] += 1
                continue
            shown = distract + [fact]
            rng.shuffle(shown)
            row("grounded", facts_prompt([f["text"] for f in shown], q), reply_for(fact), fact["id"],
                [f["id"] for f in shown])

    grounded = stats["grounded"]
    other = 1.0 - not_in_facts_frac - closed_book_frac
    if grounded and other > 0:
        n_total = grounded / other
        for _ in range(round(n_total * not_in_facts_frac)):
            target = rng.choice(usable)
            pool = [f for f in pool_all if f["id"] != target["id"] and not mentions(f["text"], target["aliases"])]
            if len(pool) < FACTS_PER_ROW:
                continue
            shown = rng.sample(pool, FACTS_PER_ROW)
            row("not_in_facts", facts_prompt([f["text"] for f in shown], rng.choice(target["questions"])),
                NOT_SURE, None, [f["id"] for f in shown])
        for _ in range(round(n_total * closed_book_frac)):
            target = rng.choice(usable)
            row("closed_book", rng.choice(target["questions"]), reply_for(target), target["id"], [])
    rng.shuffle(rows)
    return rows, stats


_FACTS_RE = re.compile(r"^Facts: \[1\] (.+?) \[2\] (.+?) \[3\] (.+?) Question: (.+)$")


def audit_rows(rows: list[dict], facts: list[dict]) -> list[str]:
    """Violations (empty list = clean). Counts by kind; never echoes row text."""
    by_id = {f["id"]: f for f in facts}
    heldout = {f["id"] for f in facts if f["heldout"]}
    heldout_texts = {normalize_space(f["text"]) for f in facts if f["heldout"]}
    problems: dict[str, int] = {}

    def bad(kind):
        problems[kind] = problems.get(kind, 0) + 1

    for r in rows:
        ids = r.get("fact_ids") or []
        if r.get("answer_fact") in heldout or heldout & set(ids):
            bad("heldout fact in a training row")
        m = _FACTS_RE.match(r["user"])
        if m and any(normalize_space(t) in heldout_texts for t in m.groups()[:3]):
            bad("heldout fact text in a prompt")
        if r["kind"] == "grounded":
            ans = by_id[r["answer_fact"]]
            if not m or len(ids) != FACTS_PER_ROW or ids.count(ans["id"]) != 1:
                bad("grounded row without exactly one answer fact among 3")
            elif any(mentions(by_id[i]["text"], ans["aliases"]) for i in ids if i != ans["id"]):
                bad("distractor contains an answer alias")
            if r["assistant"] != reply_for(ans):
                bad("grounded reply is not the canonical alias")
        elif r["kind"] == "not_in_facts":
            if not m or r["assistant"] != NOT_SURE:
                bad("malformed not-in-facts row")
        elif r["kind"] == "closed_book":
            if r["user"].startswith("Facts:") or ids:
                bad("closed-book row has facts")
        else:
            bad("unknown row kind")
    return [f"{k}: {v}" for k, v in sorted(problems.items())]


def build(facts: list[dict], *, seed: int = 20261110, val_frac: float = 0.0, **kw) -> dict:
    train_facts = [f for f in facts if not _fact_split(f["id"], val_frac)] if val_frac else facts
    val_facts = [f for f in facts if _fact_split(f["id"], val_frac)] if val_frac else []
    train, tstats = build_rows(train_facts, seed, **kw)
    out = {"train": train, "train_stats": tstats}
    if val_facts:
        # Val rows answer only from val facts (never an answer, or even shown,
        # in train); train facts may appear in val prompts as distractors.
        val, vstats = build_rows(val_facts, seed + 1, distractor_pool=val_facts + train_facts, **kw)
        out.update(val=val, val_stats=vstats)
    return out


def main(argv=None) -> None:
    p = argparse.ArgumentParser(description="Stage G grounded-facts SFT builder")
    p.add_argument("--facts", required=True, type=Path)
    p.add_argument("--out", required=True, type=Path)
    p.add_argument("--seed", type=int, default=20261110)
    p.add_argument("--val-frac", type=float, default=0.1, help="share of non-heldout facts kept for val")
    p.add_argument("--not-in-facts", type=float, default=0.25)
    p.add_argument("--closed-book", type=float, default=0.10)
    p.add_argument("--g5-val", type=Path, action="append", help="val JSONL to run the G5 leakage report against")
    p.add_argument("--tokenize", action="store_true", help="also write .bin/.masks.bin (needs tiktoken)")
    args = p.parse_args(argv)

    facts, load_report = load_facts(args.facts)
    built = build(facts, seed=args.seed, val_frac=args.val_frac,
                  not_in_facts_frac=args.not_in_facts, closed_book_frac=args.closed_book)
    problems = audit_rows(built["train"], facts) + audit_rows(built.get("val", []), facts)
    if problems:
        raise SystemExit("audit failed: " + "; ".join(problems))
    args.out.mkdir(parents=True, exist_ok=True)
    sys.path.insert(0, str(Path(__file__).resolve().parent))
    from build_stage_e_sft import leakage_report  # noqa: E402

    report = {"facts": load_report, "train": built["train_stats"], "val": built.get("val_stats"),
              "audit": "clean", "g5": {}}
    if built.get("val"):
        report["g5"]["own_val"] = leakage_report(built["train"], built["val"])
    for path in args.g5_val or []:
        other = [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]
        report["g5"][path.name] = leakage_report(built["train"], other)
    for split in ("train", "val"):
        rows = built.get(split)
        if not rows:
            continue
        with (args.out / f"{split}.jsonl").open("w", encoding="utf-8") as f:
            for r in rows:
                f.write(json.dumps(r, ensure_ascii=False) + "\n")
        if args.tokenize:
            from build_stage_e_sft import write_split  # noqa: E402
            from playground_lm.tokenizer import get_tokenizer  # noqa: E402
            report[f"{split}_tokens"] = write_split(args.out, split, rows, get_tokenizer())
    (args.out / "grounded_report.json").write_text(json.dumps(report, indent=2) + "\n")
    print(json.dumps({k: report[k] for k in ("facts", "train", "val")}, indent=2))


if __name__ == "__main__":
    main()
