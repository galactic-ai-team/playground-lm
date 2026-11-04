
#!/usr/bin/env python3
"""Audit held-out generations for exact training-token copying.

Uses run_eval.load_model(), evaluate_prompts() and copy_matches().
The default threshold is 50 consecutive GPT-2 tokens.

A flag is evidence of a verbatim overlap, not proof of why it occurred.
No matched training text is included in the output.

Example:
    python -m scripts.check_verbatim_copy \
      --checkpoint runs/plm-c/checkpoints/step_00010000.safetensors \
      --train-bin data/train.bin \
      --prompts evals/prompts.json \
      --max-new-tokens 100 \
      --device cuda \
      --output evals/results/stage_c_copy_check.json
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any, Sequence

from evals.run_eval import (
    COPY_THRESHOLD,
    copy_matches,
    evaluate_prompts,
    load_model,
)
from playground_lm.tokenizer import get_tokenizer

FACTORY = "playground_lm.model:PlaygroundLM"


def audit_token_sequences(
    train_bin: str | Path,
    generations: Sequence[Sequence[int]],
    *,
    threshold: int = COPY_THRESHOLD,
    prompt_ids: Sequence[str] | None = None,
) -> dict[str, Any]:
    """Audit continuation tokens using the project's exact matcher.

    Designed to be independently unit-testable without a model.
    """
    if threshold <= 0:
        raise ValueError("threshold must be positive")

    sequences = [list(ids) for ids in generations]

    if prompt_ids is None:
        prompt_ids = [
            f"generation_{index}"
            for index in range(len(sequences))
        ]

    if len(prompt_ids) != len(sequences):
        raise ValueError(
            "Prompt IDs and generation counts must match"
        )

    if len(set(prompt_ids)) != len(prompt_ids):
        raise ValueError("Duplicate prompt IDs")

    matches = copy_matches(
        train_bin,
        sequences,
        threshold=threshold,
    )

    results = []
    flagged = 0
    short = 0

    for prompt_id, tokens, match in zip(
        prompt_ids, sequences, matches
    ):
        is_short = len(tokens) < threshold
        short += int(is_short)
        flagged += int(match["flagged"])

        results.append({
            "id": prompt_id,
            "generated_tokens": len(tokens),
            "checked": not is_short,
            "flagged": bool(match["flagged"]),
            "match_start_in_generation": match[
                "match_start_in_generation"
            ],
            "match_start_in_training": match[
                "match_start_in_training"
            ],
        })

    return {
        "schema_version": 1,
        "metric": "exact_contiguous_training_token_overlap",
        "threshold_tokens": threshold,
        "generations": len(sequences),
        "checked_generations": len(sequences) - short,
        "short_generations": short,
        "flagged_generations": flagged,
        "any_flagged": flagged > 0,
        "results": results,
    }


def run_check(
    *,
    checkpoint: str | Path,
    train_bin: str | Path,
    prompts: str | Path,
    device: str = "cpu",
    config: str | Path | None = None,
    context_len: int | None = None,
    max_new_tokens: int = 100,
    threshold: int = COPY_THRESHOLD,
    temperature: float = 0.0,
    top_k: int = 0,
    seed: int = 42,
) -> dict[str, Any]:
    """Generate once from held-out prompts, then audit each output."""
    if max_new_tokens <= 0:
        raise ValueError("--max-new-tokens must be positive")
    if threshold <= 0:
        raise ValueError("--threshold must be positive")
    if temperature < 0 or top_k < 0:
        raise ValueError("Invalid sampling parameters")

    train_path = Path(train_bin)
    if not train_path.is_file():
        raise FileNotFoundError(train_path)
    if train_path.stat().st_size == 0:
        raise ValueError("Training data is empty")

    model = load_model(
        "playground",
        str(checkpoint),
        factory=FACTORY,
        config_path=(
            str(config) if config is not None else None
        ),
        device=device,
    )

    model_context = getattr(model, "context_len", None)
    if context_len is None:
        if not isinstance(model_context, int):
            raise ValueError("Model has no context_len")
        context_len = model_context

    if context_len <= 0:
        raise ValueError("context_len must be positive")
    if (
        model_context is not None
        and context_len > model_context
    ):
        raise ValueError(
            "context_len exceeds model context"
        )

    tokenizer = get_tokenizer()

    records, generated_tokens = evaluate_prompts(
        model,
        prompts,
        tokenizer=tokenizer,
        context_len=context_len,
        max_new_tokens=max_new_tokens,
        device=device,
        temperature=temperature,
        top_k=top_k,
        seed=seed,
    )

    audit = audit_token_sequences(
        train_path,
        generated_tokens,
        threshold=threshold,
        prompt_ids=[record["id"] for record in records],
    )

    audit.update({
        "model": str(checkpoint),
        "backend": "playground",
        "tokenizer": "gpt2",
        "train_bin": str(train_path),
        "prompts": str(prompts),
        "context_len": context_len,
        "max_new_tokens": max_new_tokens,
        "temperature": temperature,
        "top_k": top_k,
        "seed": seed,
    })

    # Full raw generation strings are intentionally excluded.
    return audit


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--checkpoint", type=Path, required=True
    )
    parser.add_argument(
        "--config", type=Path
    )
    parser.add_argument(
        "--train-bin", type=Path, required=True
    )
    parser.add_argument(
        "--prompts",
        type=Path,
        default=Path("evals/prompts.json"),
    )
    parser.add_argument(
        "--device", default="cpu"
    )
    parser.add_argument(
        "--context-len", type=int
    )
    parser.add_argument(
        "--max-new-tokens",
        type=int,
        default=100,
    )
    parser.add_argument(
        "--threshold",
        type=int,
        default=COPY_THRESHOLD,
    )
    parser.add_argument(
        "--temperature",
        type=float,
        default=0.0,
    )
    parser.add_argument(
        "--top-k",
        type=int,
        default=0,
    )
    parser.add_argument(
        "--seed", type=int, default=42
    )
    parser.add_argument(
        "--output",
        type=Path,
        required=True,
    )
    args = parser.parse_args()

    report = run_check(
        checkpoint=args.checkpoint,
        config=args.config,
        train_bin=args.train_bin,
        prompts=args.prompts,
        device=args.device,
        context_len=args.context_len,
        max_new_tokens=args.max_new_tokens,
        threshold=args.threshold,
        temperature=args.temperature,
        top_k=args.top_k,
        seed=args.seed,
    )

    # Reuse the project's atomic JSON writer.
    from evals.publish_checkpoint import write_json_atomic

    write_json_atomic(args.output, report)

    print(json.dumps({
        "output": str(args.output),
        "threshold_tokens": report["threshold_tokens"],
        "checked_generations": report["checked_generations"],
        "short_generations": report["short_generations"],
        "flagged_generations": report["flagged_generations"],
        "any_flagged": report["any_flagged"],
    }, indent=2))


if __name__ == "__main__":
    main()
