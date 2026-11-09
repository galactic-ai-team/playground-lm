#!/usr/bin/env python3
"""Frozen behavior eval for PlaygroundLM chat-style generation.

Loads prompts + expects as separate frozen files so training/data lanes can
read prompts without shipping scorers in the same document they design from.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import torch


def score(text: str, expect: dict) -> bool:
    t = (text or "").strip()
    et = expect["type"]
    if et == "nonempty":
        return bool(t)
    if et == "contains":
        return expect["value"] in t
    if et == "contains_any":
        return any(v in t for v in expect["values"])
    if et == "exact_or_startswith":
        return t == expect["value"] or t.startswith(expect["value"])
    raise ValueError(et)


def build_prompt(item, create_message_format=True):
    if "messages" in item:
        lines = []
        for m in item["messages"]:
            role = "User" if m["role"] == "user" else "Assistant"
            lines.append(f"{role}: {m['content']}")
        lines.append("Assistant:")
        return "\n".join(lines)
    return f"User: {item['user']}\nAssistant:"


def load_suite(prompts_path: Path, expect_path: Path) -> dict:
    prompts_doc = json.loads(prompts_path.read_text())
    expect_doc = json.loads(expect_path.read_text())
    by_id = expect_doc["expect_by_id"]
    items = []
    for item in prompts_doc["prompts"]:
        eid = item["id"]
        if eid not in by_id:
            raise KeyError(f"missing expect for {eid}")
        merged = dict(item)
        merged["expect"] = by_id[eid]
        items.append(merged)
    return {
        "frozen_at_et": prompts_doc.get("frozen_at_et"),
        "seed": prompts_doc.get("seed"),
        "notes": prompts_doc.get("notes"),
        "prompts": items,
    }


@torch.no_grad()
def generate(model, tokenizer, prompt, max_new=48, device="cpu"):
    ids = tokenizer.encode(prompt)
    if not ids:
        ids = [tokenizer.eot_id]
    x = torch.tensor([ids[-1024:]], dtype=torch.long, device=device)
    out = model.generate(x, max_new_tokens=max_new, temperature=0.0, top_k=1)
    new_ids = out[0].tolist()[x.shape[1] :]
    text_ids = []
    for tid in new_ids:
        if tid == tokenizer.eot_id:
            break
        text_ids.append(tid)
    text = tokenizer.decode(text_ids)
    if "\nUser:" in text:
        text = text.split("\nUser:")[0]
    return text.strip(), text_ids


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--checkpoint", required=True)
    ap.add_argument("--config", required=True)
    ap.add_argument(
        "--prompts-json",
        default=str(Path(__file__).resolve().parent / "behavior" / "behavior_eval.prompts.frozen.json"),
    )
    ap.add_argument(
        "--expect-json",
        default=str(Path(__file__).resolve().parent / "behavior" / "behavior_eval.expect.frozen.json"),
        help="EVAL LANE ONLY. Do not pass expect file into training/data design workflows.",
    )
    ap.add_argument("--out", required=True)
    ap.add_argument("--max-new", type=int, default=48)
    ap.add_argument("--device", default="cpu")
    args = ap.parse_args()

    import sys

    sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
    from playground_lm.tokenizer import get_tokenizer
    from evals.run_eval import load_model

    tok = get_tokenizer()
    model = load_model(
        "playground",
        args.checkpoint,
        factory="playground_lm.model:PlaygroundLM",
        device=args.device,
        config_path=args.config,
    ).eval()

    suite = load_suite(Path(args.prompts_json), Path(args.expect_json))
    rows = []
    for item in suite["prompts"]:
        prompt = build_prompt(item)
        text, ids = generate(model, tok, prompt, max_new=args.max_new, device=args.device)
        ok = score(text, item["expect"])
        rep = False
        if ids:
            run = 1
            for a, b in zip(ids, ids[1:]):
                run = run + 1 if a == b else 1
                if run >= 8:
                    rep = True
                    break
        rows.append(
            {
                "id": item["id"],
                "category": item["category"],
                "prompt": prompt,
                "output": text,
                "pass": ok,
                "long_repetition": rep,
                "n_tokens": len(ids),
            }
        )
        print(item["id"], "PASS" if ok else "FAIL", repr(text[:120]), flush=True)

    summary = {
        "checkpoint": args.checkpoint,
        "n": len(rows),
        "passed": sum(r["pass"] for r in rows),
        "pass_rate": sum(r["pass"] for r in rows) / max(1, len(rows)),
        "repetition_flags": sum(r["long_repetition"] for r in rows),
        "by_category": {},
        "rows": rows,
        "limitations": "124M completion model; not instruction-tuned ChatGPT-quality. Exact string matches may fail even when answer is partially right.",
    }
    for r in rows:
        c = summary["by_category"].setdefault(r["category"], {"n": 0, "passed": 0})
        c["n"] += 1
        c["passed"] += int(r["pass"])
    Path(args.out).write_text(json.dumps(summary, indent=2) + "\n")
    print(
        json.dumps(
            {k: summary[k] for k in ("n", "passed", "pass_rate", "repetition_flags", "by_category")},
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
