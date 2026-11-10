"""Stage G S1/S2 scoring and paired receipt compatibility (public code only).

No private items, training data, model weights, or network access in this module.
S1 = strict whole completion, greedy max_new=48.
S2 = strict final Answer: integer extraction (otherwise whole completion),
     greedy max_new=256. Both use strip+casefold exact match; numeric
     answers additionally ignore digit-group separators (spaces/commas).

Run inference with evals.run_behavior_eval.generate under the specified
settings. This module scores only outputs supplied by the caller.
"""
from __future__ import annotations

import re
from typing import Any, Callable, Mapping

from evals.stage_e_review_gates import mcnemar_exact

S1_MAX_NEW = 48
S2_MAX_NEW = 256
FORMATS = {"closed-book", "pipeline"}
METRICS = {"s1", "s2"}
ANSWER_AT_END = re.compile(r"(?<![\w])Answer:\s*([+-]?\d[\d, ]*)\s*$")
INTEGER = re.compile(r"^[+-]?\d[\d, ]*$")


def extract_final(output: str) -> str:
    """Last terminal Answer: integer, or unchanged output.

    Must occur at end of output; prose after the number invalidates extraction.
    The fixed Stage G generator emits Answer: inline, not on a new line.
    """
    if not isinstance(output, str):
        raise TypeError("output must be a string")
    matches = list(ANSWER_AT_END.finditer(output))
    return matches[-1].group(1).strip() if matches else output


def normalize_answer(value: str) -> str:
    if not isinstance(value, str):
        raise TypeError("answer must be text")
    s = value.strip().casefold()
    if INTEGER.fullmatch(s) and re.sub(r"[\s,]", "", s) not in ("", "+", "-"):
        try:
            return str(int(re.sub(r"[\s,]", "", s)))
        except ValueError:
            pass
    return s


def _accepted(expectation: Mapping[str, Any]) -> tuple[str, ...]:
    typ = expectation.get("score_type")
    if typ == "exact":
        answers = [expectation.get("expected")]
    elif typ == "one_of_exact":
        answers = expectation.get("expected_any")
    else:
        raise ValueError("only exact/one_of_exact are supported")
    if not isinstance(answers, list) or not answers or any(
        not isinstance(a, str) or not a.strip() for a in answers
    ):
        raise ValueError("missing valid expected answer(s)")
    return tuple(normalize_answer(a) for a in answers)


def score_s1(output: str, expectation: Mapping[str, Any]) -> bool:
    """Legacy strict whole-completion exact match (48-token protocol)."""
    return normalize_answer(output) in _accepted(expectation)


def score_s2(output: str, expectation: Mapping[str, Any]) -> bool:
    """Stage G final-answer exact match (256-token protocol)."""
    return normalize_answer(extract_final(output)) in _accepted(expectation)


def build_eval_prompt(
    item: Mapping[str, Any], *, mode: str = "closed-book",
    retriever: Callable[[str], list[str]] | None = None,
    fact_sheet_sha256: str | None = None,
) -> str:
    """Format a probe using the established User:/Assistant: chat formatter.

    A pipeline caller supplies a deterministic fact-sheet-bound retriever;
    it returns a list of factual sentences, not external instructions.
    """
    from evals.run_behavior_eval import build_prompt

    if mode not in FORMATS:
        raise ValueError("invalid evaluation mode")
    if mode == "closed-book":
        if retriever is not None or fact_sheet_sha256 is not None:
            raise ValueError("closed-book cannot provide retrieval settings")
        return build_prompt(item)
    if not isinstance(fact_sheet_sha256, str) or not re.fullmatch(
        r"[0-9a-f]{64}", fact_sheet_sha256
    ) or retriever is None:
        raise ValueError("pipeline requires retriever and fact-sheet SHA256")
    if "messages" in item:
        raise ValueError("pipeline probes must be single-turn user prompts")
    question = item.get("user")
    if not isinstance(question, str) or not question.strip():
        raise ValueError("missing question")
    facts = retriever(question)
    if not isinstance(facts, list) or len(facts) > 3 or any(
        not isinstance(s, str) or not s.strip() or "\n" in s or "\r" in s
        or "Assistant:" in s or "User:" in s for s in facts
    ):
        raise ValueError("retriever must return 0-3 one-line fact strings")
    indexed = " ".join(f"[{i}] {fact.strip()}" for i, fact in enumerate(facts, 1))
    prompt = f"Facts: {indexed} Question: {question.strip()}" if facts else f"Facts: (none) Question: {question.strip()}"
    return build_prompt({"user": prompt})


def make_receipt(
    results: list[Mapping[str, Any]], *, scorer: str, mode: str,
    suite_sha256: str, fact_sheet_sha256: str | None = None,
    model_tag: str, checkpoint_sha256: str, max_new: int | None = None,
    seed_base: int = 42, device: str = "cpu",
) -> dict[str, Any]:
    """Receipt containing per-item outcomes; keep private-suite receipts owner-only."""
    if not isinstance(seed_base, int) or isinstance(seed_base, bool) or not isinstance(device, str) or not device:
        raise ValueError("invalid deterministic generation settings")
    if scorer not in METRICS or mode not in FORMATS:
        raise ValueError("unsupported scorer/mode")
    expected_max = S1_MAX_NEW if scorer == "s1" else S2_MAX_NEW
    max_new = expected_max if max_new is None else max_new
    if max_new != expected_max:
        raise ValueError("scorer/max_new mismatch")
    if mode == "pipeline" and not (isinstance(fact_sheet_sha256, str) and re.fullmatch(r"[0-9a-f]{64}", fact_sheet_sha256)):
        raise ValueError("pipeline receipt needs fact-sheet SHA256")
    if mode == "closed-book" and fact_sheet_sha256 is not None:
        raise ValueError("closed-book receipt must not carry a fact-sheet SHA")
    for name, value in (("suite_sha256", suite_sha256), ("checkpoint_sha256", checkpoint_sha256)):
        if not isinstance(value, str) or not re.fullmatch(r"[0-9a-f]{64}", value):
            raise ValueError(f"invalid {name}")
    ids: set[str] = set()
    by_category: dict[str, dict[str, int]] = {}
    rows = []
    for r in results:
        rid, cat, passed = r.get("id"), r.get("category"), r.get("pass")
        if not isinstance(rid, str) or not rid or rid in ids or not isinstance(cat, str) or not cat or type(passed) is not bool:
            raise ValueError("invalid/duplicate receipt item")
        ids.add(rid)
        if "output" in r:  # Do not inadvertently include raw private outputs in public receipts.
            raise ValueError("pass summary only; raw outputs are not accepted")
        by_category.setdefault(cat, {"n": 0, "passed": 0})["n"] += 1
        by_category[cat]["passed"] += int(passed)
        rows.append({"id": rid, "category": cat, "pass": passed})
    if not rows:
        raise ValueError("cannot issue an empty receipt")
    passed = sum(r["pass"] for r in rows)
    return {
        "schema_version": 1, "kind": "stage_g_behavior_receipt", "suite_sha256": suite_sha256,
        "scorer": scorer, "max_new": max_new, "decoding": "greedy", "temperature": 0.0,
        "top_k": 0, "seed_base": seed_base, "device": device,
        "mode": mode, "fact_sheet_sha256": fact_sheet_sha256,
        "model_tag": model_tag, "checkpoint_sha256": checkpoint_sha256,
        "n": len(rows), "passed": passed, "pass_rate": passed / len(rows),
        "by_category": by_category, "rows": rows,
        "note": "Private suite and output text are not embedded. Not promotion evidence alone.",
    }


def _check_pair(a: Mapping[str, Any], b: Mapping[str, Any]) -> None:
    if a.get("kind") != "stage_g_behavior_receipt" or b.get("kind") != "stage_g_behavior_receipt":
        raise ValueError("invalid receipt kind")
    for key in ("schema_version", "suite_sha256", "scorer", "max_new", "decoding", "temperature", "top_k", "seed_base", "device", "mode", "fact_sheet_sha256"):
        if a.get(key) != b.get(key):
            raise ValueError(f"incomparable receipts: {key}")
    if a.get("scorer") not in METRICS or a.get("max_new") != (S1_MAX_NEW if a["scorer"] == "s1" else S2_MAX_NEW):
        raise ValueError("invalid scorer settings")
    if a.get("decoding") != "greedy" or a.get("temperature") != 0.0 or a.get("top_k") != 0:
        raise ValueError("invalid deterministic decoding settings")
    if a.get("mode") not in FORMATS:
        raise ValueError("invalid mode")
    if a.get("mode") == "pipeline" and not a.get("fact_sheet_sha256"):
        raise ValueError("pipeline fact-sheet SHA missing")
    for r in (a, b):
        rows = r.get("rows")
        if not isinstance(rows, list) or not rows or r.get("n") != len(rows):
            raise ValueError("invalid rows")
        ids = [item.get("id") for item in rows]
        if len(ids) != len(set(ids)) or any(type(item.get("pass")) is not bool for item in rows):
            raise ValueError("invalid/duplicate item results")
        if sum(item["pass"] for item in rows) != r.get("passed"):
            raise ValueError("incorrect pass total")
    if {(r["id"], r["category"]) for r in a["rows"]} != {(r["id"], r["category"]) for r in b["rows"]}:
        raise ValueError("incomparable receipts: item IDs/categories")


def compare_receipts(incumbent: Mapping[str, Any], candidate: Mapping[str, Any], *, alpha: float = 0.05, min_net: int = 5) -> dict[str, Any]:
    """Paired exact McNemar; keep final promotion decision outside scorer."""
    _check_pair(incumbent, candidate)
    if not (0 < alpha < 1) or min_net < 1:
        raise ValueError("invalid significance thresholds")
    a = {r["id"]: r["pass"] for r in incumbent["rows"]}
    b = {r["id"]: r["pass"] for r in candidate["rows"]}
    gained = sum(b[k] and not a[k] for k in a)
    lost = sum(a[k] and not b[k] for k in a)
    p = mcnemar_exact(gained, lost)
    return {
        "incumbent": incumbent["model_tag"], "candidate": candidate["model_tag"],
        "n": len(a), "incumbent_passed": sum(a.values()), "candidate_passed": sum(b.values()),
        "gained": gained, "lost": lost, "net": gained - lost,
        "mcnemar_exact_p": p, "alpha": alpha, "min_net": min_net,
        "significant_win": p < alpha and gained - lost >= min_net,
        "promotion": "requires independent regression, leakage gates, and Stephon approval",
    }


def evaluate_suite(
    model: Any, tokenizer: Any, items: list[Mapping[str, Any]], *,
    scorer: str, mode: str, suite_sha256: str, model_tag: str,
    checkpoint_sha256: str, retriever: Callable[[str], list[str]] | None = None,
    fact_sheet_sha256: str | None = None, seed_base: int = 42, device: str = "cpu",
) -> dict[str, Any]:
    """Run the accepted greedy harness once per private probe; return pass-only receipt.

    The calling operator must independently hash-verify frozen fixtures and
    audit training overlap. This does not write output or promote anything.
    """
    from evals.run_behavior_eval import generate

    if scorer not in METRICS:
        raise ValueError("invalid scorer")
    max_new = S1_MAX_NEW if scorer == "s1" else S2_MAX_NEW
    score_fn = score_s1 if scorer == "s1" else score_s2
    results = []
    for index, item in enumerate(items):
        prompt = build_eval_prompt(item, mode=mode, retriever=retriever,
                                   fact_sheet_sha256=fact_sheet_sha256)
        text, _token_ids = generate(model, tokenizer, prompt, max_new=max_new,
                                    device=device, seed=seed_base + index)
        results.append({"id": item["id"], "category": item["category"],
                        "pass": score_fn(text, item["expect"])})
    return make_receipt(results, scorer=scorer, mode=mode,
                        suite_sha256=suite_sha256, model_tag=model_tag,
                        checkpoint_sha256=checkpoint_sha256,
                        fact_sheet_sha256=fact_sheet_sha256, max_new=max_new,
                        seed_base=seed_base, device=device)
