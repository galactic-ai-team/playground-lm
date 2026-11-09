#!/usr/bin/env python3
"""Stage E frozen behavior evaluation (eval lane only).

The public frozen development suite measures predefined behavior; it is not
blind evidence of generalization. Keep expectation fixtures out of SFT data.

Runs deterministic greedy continuations using evals.run_eval.generate_tokens,
never model.generate()'s sampling path. Fail closed on broken frozen fixtures.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import re
from pathlib import Path
from typing import Any, Mapping

FROZEN_PROMPTS_SHA256 = "e0f7feb4af2cf6854f852318cbeb983d45a0c7dc85785ae5f46df16f4cf67d39"
FROZEN_EXPECT_SHA256 = "caf5a86f823f3918eaebb827963fec16e2eaa220b0245e57bf3d8eb235f8c8b6"
SUITE_DIR = Path(__file__).resolve().parent / "behavior"
DEFAULT_PROMPTS = SUITE_DIR / "behavior_eval.prompts.frozen.json"
DEFAULT_EXPECT = SUITE_DIR / "behavior_eval.expect.frozen.json"
VALID_SCORERS = {"nonempty", "contains", "contains_any", "exact_or_startswith"}


def _read_json(path: Path) -> dict[str, Any]:
    doc = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(doc, dict):
        raise ValueError(f"Expected JSON object: {path}")
    return doc


def _verify_frozen(path: Path, default: Path, expected_sha256: str) -> str:
    raw = path.read_bytes()
    actual = hashlib.sha256(raw).hexdigest()
    if path.resolve() == default.resolve() and actual != expected_sha256:
        raise ValueError(f"Frozen evaluation file modified: {path}")
    return actual


def _validate_expect(expect: Any, probe_id: str) -> None:
    if not isinstance(expect, dict) or expect.get("type") not in VALID_SCORERS:
        raise ValueError(f"Unsupported scorer for {probe_id}")
    kind = expect["type"]
    if kind in ("contains", "exact_or_startswith"):
        if not isinstance(expect.get("value"), str) or not expect["value"].strip():
            raise ValueError(f"Missing nonempty scorer value: {probe_id}")
    if kind == "contains_any":
        vals = expect.get("values")
        if not isinstance(vals, list) or not vals or any(
            not isinstance(v, str) or not v.strip() for v in vals
        ):
            raise ValueError(f"Invalid scorer values: {probe_id}")


def _contains(text: str, needle: str) -> bool:
    """Text match, but don't mistake '145' for arithmetic answer '45'."""
    # Require lexical boundaries for short answers and names: "No" must
    # not match "Nobody", and 45 must not match 145 or 456.
    prefix = r"(?<!\w)" if needle[0].isalnum() else ""
    suffix = r"(?!\w)" if needle[-1].isalnum() else ""
    pattern = prefix + re.escape(needle) + suffix
    return re.search(pattern, text, flags=re.IGNORECASE) is not None


def score(text: str, expect: Mapping[str, Any]) -> bool:
    if not isinstance(text, str):
        raise TypeError("Model output must be a string")
    _validate_expect(expect, "score")
    t = text.strip()
    kind = expect["type"]
    if kind == "nonempty":
        return bool(t)
    if kind == "contains":
        return _contains(t, expect["value"])
    if kind == "contains_any":
        return any(_contains(t, x) for x in expect["values"])
    # The original prefix scorer accepted 'OKay', 'YESterday'. Require a
    # word boundary while permitting natural following punctuation or text.
    value = expect["value"]
    return bool(re.match(re.escape(value) + r"(?=$|\W)", t, flags=re.IGNORECASE))


def build_prompt(item: Mapping[str, Any], create_message_format: bool = True) -> str:
    """Reproduce SFT's User:/Assistant: prefix; validate turn order."""
    if not isinstance(item, Mapping):
        raise ValueError("Probe must be an object")
    if "messages" in item:
        messages = item["messages"]
        if not isinstance(messages, list) or not messages:
            raise ValueError("messages must be nonempty")
        lines = []
        last_role = None
        for m in messages:
            if not isinstance(m, dict) or m.get("role") not in ("user", "assistant"):
                raise ValueError("Only user/assistant messages supported")
            role = m["role"]
            content = m.get("content")
            if not isinstance(content, str) or not content.strip() or "\nUser:" in content or "\nAssistant:" in content:
                raise ValueError("Invalid message content")
            if role == last_role or (last_role is None and role != "user"):
                raise ValueError("Messages must alternate, starting with user")
            lines.append(f"{'User' if role == 'user' else 'Assistant'}: {content.strip()}")
            last_role = role
        if last_role != "user":
            raise ValueError("Final conversation message must be user")
        return "\n".join(lines + ["Assistant:"])
    user = item.get("user")
    if not isinstance(user, str) or not user.strip():
        raise ValueError("Missing nonempty user message")
    if "\nAssistant:" in user or "\nUser:" in user:
        raise ValueError("Role delimiter embedded in user message")
    return f"User: {user.strip()}\nAssistant:"


def load_suite(prompts_path: Path, expect_path: Path) -> dict[str, Any]:
    """Validate frozen evaluation split, exact ID alignment and scorer schema."""
    prompts_path, expect_path = Path(prompts_path), Path(expect_path)
    prompt_sha = _verify_frozen(prompts_path, DEFAULT_PROMPTS, FROZEN_PROMPTS_SHA256)
    expect_sha = _verify_frozen(expect_path, DEFAULT_EXPECT, FROZEN_EXPECT_SHA256)
    prompts_doc, expect_doc = _read_json(prompts_path), _read_json(expect_path)
    prompts, expects = prompts_doc.get("prompts"), expect_doc.get("expect_by_id")
    if not isinstance(prompts, list) or not prompts:
        raise ValueError("Prompt suite must be a nonempty list")
    if not isinstance(expects, dict):
        raise ValueError("Expectations must be keyed by probe ID")
    if prompts_doc.get("seed") != expect_doc.get("seed"):
        raise ValueError("Prompt and expectation seeds differ")
    if prompts_doc.get("frozen_at_et") != expect_doc.get("frozen_at_et"):
        raise ValueError("Frozen timestamps differ")
    items, ids = [], set()
    for item in prompts:
        if not isinstance(item, dict):
            raise ValueError("Probe must be a JSON object")
        eid = item.get("id")
        if not isinstance(eid, str) or not eid.strip() or eid in ids:
            raise ValueError(f"Invalid/duplicate probe ID: {eid}")
        if "expect" in item:
            raise ValueError(f"Expected scorer leaked into prompt file: {eid}")
        if not isinstance(item.get("category"), str) or not item["category"]:
            raise ValueError(f"Invalid probe category: {eid}")
        build_prompt(item)
        ids.add(eid)
        if eid not in expects:
            raise ValueError(f"Missing expectation: {eid}")
        _validate_expect(expects[eid], eid)
        items.append({**item, "expect": expects[eid]})
    if ids != set(expects):
        raise ValueError(f"Unexpected expectation IDs: {sorted(set(expects) - ids)}")
    return {
        "frozen_at_et": prompts_doc.get("frozen_at_et"),
        "seed": prompts_doc.get("seed"),
        "prompts_sha256": prompt_sha,
        "expect_sha256": expect_sha,
        "prompts": items,
    }


def generate(model: Any, tokenizer: Any, prompt: str, max_new: int = 48,
             device: str = "cpu", seed: int = 42) -> tuple[str, list[int]]:
    """Use shared greedy token generation; truncate on EOT and next User turn."""
    from evals.run_eval import generate_tokens
    if not isinstance(max_new, int) or max_new <= 0:
        raise ValueError("max_new must be positive")
    ids = tokenizer.encode(prompt)
    if not ids:
        raise ValueError("Tokenized prompt is empty")
    eot = tokenizer.eot_token_id  # GPT2Tokenizer's actual API (not eot_id).
    context_len = getattr(model, "context_len", 1024)
    if not isinstance(context_len, int) or context_len <= 0:
        raise ValueError("Invalid model context_len")
    continuation = generate_tokens(
        model, ids, max_new_tokens=max_new, context_len=context_len,
        device=device, eot_id=eot, temperature=0.0, top_k=0, seed=seed,
    )
    decoded = tokenizer.decode(continuation)
    # Do not score any model-invented subsequent conversation turn.
    decoded = re.split(r"\n(?:User|Assistant):", decoded, maxsplit=1)[0]
    return decoded.strip(), continuation


def has_long_repetition(token_ids: list[int], run_length: int = 8) -> bool:
    if run_length <= 1:
        raise ValueError("run_length must exceed 1")
    run = 1
    for previous, current in zip(token_ids, token_ids[1:]):
        run = run + 1 if previous == current else 1
        if run >= run_length:
            return True
    return False


def evaluate(model: Any, tokenizer: Any, suite: Mapping[str, Any], *,
             max_new: int = 48, device: str = "cpu") -> dict[str, Any]:
    rows = []
    for index, item in enumerate(suite["prompts"]):
        text, token_ids = generate(
            model, tokenizer, build_prompt(item), max_new=max_new,
            device=device, seed=int(suite.get("seed") or 42) + index,
        )
        rows.append({
            "id": item["id"], "category": item["category"],
            "output": text, "pass": score(text, item["expect"]),
            "long_repetition": has_long_repetition(token_ids),
            "n_tokens": len(token_ids),
        })
    by_category: dict[str, dict[str, int]] = {}
    for row in rows:
        item = by_category.setdefault(row["category"], {"n": 0, "passed": 0})
        item["n"] += 1
        item["passed"] += int(row["pass"])
    passed = sum(row["pass"] for row in rows)
    return {
        "schema_version": 2,
        "suite_type": "public_frozen_development_only",
        "seed": suite.get("seed"),
        "prompts_sha256": suite.get("prompts_sha256"),
        "expect_sha256": suite.get("expect_sha256"),
        "n": len(rows), "passed": passed, "pass_rate": passed / len(rows),
        "repetition_flags": sum(row["long_repetition"] for row in rows),
        "by_category": by_category, "rows": rows,
        "limitations": (
            "Public frozen development probes are not a private blind holdout. "
            "String heuristics are imperfect and do not prove generalization. "
            "Do not tune training examples against these expectations."
        ),
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--config", required=True)
    parser.add_argument("--prompts-json", type=Path, default=DEFAULT_PROMPTS)
    parser.add_argument("--expect-json", type=Path, default=DEFAULT_EXPECT)
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--max-new", type=int, default=48)
    parser.add_argument("--device", default="cpu")
    args = parser.parse_args()
    if args.max_new <= 0:
        parser.error("--max-new must be positive")
    import sys
    sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
    from playground_lm.tokenizer import get_tokenizer
    from evals.run_eval import load_model

    suite = load_suite(args.prompts_json, args.expect_json)
    model = load_model(
        "playground", args.checkpoint,
        factory="playground_lm.model:PlaygroundLM",
        device=args.device, config_path=args.config,
    ).eval()
    report = evaluate(model, get_tokenizer(), suite, max_new=args.max_new, device=args.device)
    report["checkpoint"] = args.checkpoint
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(report, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    print(json.dumps({k: report[k] for k in ("n", "passed", "pass_rate", "repetition_flags", "by_category")}, indent=2))


if __name__ == "__main__":
    main()
