#!/usr/bin/env python3
"""HellaSwag and OpenAI-LAMBADA benchmarks for causal LMs.

HellaSwag: validation, 10,042 four-choice examples.
    Choose smallest mean NLL over completion tokens.

LAMBADA: EleutherAI/lambada_openai, default/test, 5,153 examples.
    Greedy exact last-word token accuracy and target-token perplexity.

Reuses the model adapters from evals/run_eval.py.
The first dataset load may download from Hugging Face.
"""

from __future__ import annotations

import argparse
import json
import math
import re
from pathlib import Path
from typing import Any, Iterable, Mapping, Sequence

import torch
import torch.nn.functional as F

from playground_lm.tokenizer import get_tokenizer
from evals.run_eval import get_logits, load_model

HELLASWAG_DATASET = "Rowan/hellaswag"
LAMBADA_DATASET = "EleutherAI/lambada_openai"


def load_rows(
    dataset: str,
    *,
    offline: bool = False,
) -> Iterable[Mapping[str, Any]]:
    """Load the official split, optionally using local files only."""
    from datasets import load_dataset

    if dataset == "hellaswag":
        name, config, split = (
            HELLASWAG_DATASET, None, "validation"
        )
    elif dataset == "lambada":
        name, config, split = (
            LAMBADA_DATASET, "default", "test"
        )
    else:
        raise ValueError(dataset)

    if offline:
        from datasets import DownloadConfig

        return load_dataset(
            name,
            config,
            split=split,
            download_config=DownloadConfig(
                local_files_only=True
            ),
        )

    return load_dataset(name, config, split=split)


def _encode(tokenizer: Any, text: str) -> list[int]:
    ids = tokenizer.encode(text)
    if not isinstance(ids, list):
        ids = list(ids)
    if not ids:
        raise ValueError("Empty token sequence")
    return ids


def _completion_ids(
    tokenizer: Any,
    prefix: str,
    suffix: str,
) -> tuple[list[int], list[int]]:
    """Tokenize the complete string, tracking completion tokens.

    Fail if the BPE boundary changes across the join rather
    than silently scoring a different tokenization.
    """
    context = _encode(tokenizer, prefix)
    joined = _encode(tokenizer, prefix + suffix)

    if (
        joined[:len(context)] != context
        or len(joined) <= len(context)
    ):
        raise ValueError(
            "BPE boundary changed across context/completion"
        )

    return joined, list(
        range(len(context), len(joined))
    )


@torch.inference_mode()
def _completion_nll(
    model: torch.nn.Module,
    sequences: Sequence[list[int]],
    target_positions: Sequence[list[int]],
    *,
    context_len: int,
    device: str,
) -> tuple[list[float], list[int]]:
    """Return total target NLL and target count for each sequence.

    Target token at index p is predicted by logits at p-1.
    When a sequence exceeds the model context, use rolling windows.
    """
    sums: list[float] = []
    counts: list[int] = []

    for ids, positions in zip(
        sequences, target_positions
    ):
        if (
            not positions
            or any(
                p < 1 or p >= len(ids)
                for p in positions
            )
        ):
            raise ValueError(
                "Completion requires a target after context"
            )

        total = 0.0

        if len(ids) - 1 <= context_len:
            x = torch.tensor(
                [ids[:-1]],
                dtype=torch.long,
                device=device,
            )
            logits = get_logits(model, x)[0].float()
            logp = F.log_softmax(logits, dim=-1)

            for pos in positions:
                total -= float(
                    logp[pos - 1, ids[pos]].item()
                )

        else:
            for pos in positions:
                start = max(0, pos - context_len)

                x = torch.tensor(
                    [ids[start:pos]],
                    dtype=torch.long,
                    device=device,
                )
                logits = get_logits(
                    model, x
                )[0, -1].float()

                target = torch.tensor(
                    [ids[pos]],
                    device=device,
                )

                total += float(
                    F.cross_entropy(
                        logits[None], target
                    ).item()
                )

        sums.append(total)
        counts.append(len(positions))

    return sums, counts


def _hellaswag_context(
    row: Mapping[str, Any],
) -> str:
    """Construct the completion-style HellaSwag context."""
    a = str(row["ctx_a"])
    b = str(row["ctx_b"])

    return (
        a + " " + b.capitalize()
    ).strip()


@torch.inference_mode()
def score_hellaswag(
    model: torch.nn.Module,
    rows: Iterable[Mapping[str, Any]],
    tokenizer: Any,
    *,
    context_len: int = 1024,
    device: str = "cpu",
    limit: int | None = None,
) -> dict[str, Any]:
    """Choose the ending with lowest mean completion-token NLL."""
    if limit is not None and limit <= 0:
        raise ValueError("limit must be positive")

    correct = 0
    count = 0

    for row in rows:
        if limit is not None and count >= limit:
            break

        context = _hellaswag_context(row)
        endings = row["endings"]

        if len(endings) != 4:
            raise ValueError(
                "HellaSwag row must have four endings"
            )

        sequences = []
        targets = []

        for ending in endings:
            ids, positions = _completion_ids(
                tokenizer,
                context,
                " " + str(ending),
            )
            sequences.append(ids)
            targets.append(positions)

        nll, lengths = _completion_nll(
            model,
            sequences,
            targets,
            context_len=context_len,
            device=device,
        )

        predicted = min(
            range(4),
            key=lambda i: nll[i] / lengths[i],
        )

        correct += predicted == int(row["label"])
        count += 1

    if not count:
        raise ValueError(
            "No HellaSwag examples evaluated"
        )

    return {
        "dataset": HELLASWAG_DATASET,
        "split": "validation",
        "metric": "accuracy_normalized_completion_nll",
        "accuracy": correct / count,
        "correct": correct,
        "examples": count,
        "complete": limit is None and count == 10042,
        "expected_full_examples": 10042,
    }


def _lambada_parts(
    text: str,
) -> tuple[str, str]:
    """Split before whitespace preceding the final word.

    Include whitespace in the completion so GPT-2's
    leading-space BPE tokens are evaluated correctly.
    """
    clean = text.rstrip()
    match = re.search(r"\s+\S+$", clean)

    if not match or not clean[:match.start()]:
        raise ValueError(
            "LAMBADA passage must contain "
            "context and final word"
        )

    return (
        clean[:match.start()],
        clean[match.start():],
    )


@torch.inference_mode()
def _greedy_last_word(
    model: torch.nn.Module,
    prefix_ids: list[int],
    *,
    tokens_to_generate: int,
    context_len: int,
    device: str,
) -> list[int]:
    """Greedy-decode exactly the reference token count."""
    generated: list[int] = []

    for _ in range(tokens_to_generate):
        context = (
            prefix_ids + generated
        )[-context_len:]

        x = torch.tensor(
            [context],
            dtype=torch.long,
            device=device,
        )

        next_id = int(
            get_logits(model, x)[0, -1]
            .argmax()
            .item()
        )

        generated.append(next_id)

    return generated


@torch.inference_mode()
def score_lambada(
    model: torch.nn.Module,
    rows: Iterable[Mapping[str, Any]],
    tokenizer: Any,
    *,
    context_len: int = 1024,
    device: str = "cpu",
    limit: int | None = None,
) -> dict[str, Any]:
    """Exact greedy last-word accuracy and target-token PPL."""
    if limit is not None and limit <= 0:
        raise ValueError("limit must be positive")

    correct = 0
    count = 0
    token_count = 0
    total_nll = 0.0

    for row in rows:
        if limit is not None and count >= limit:
            break

        prefix, word = _lambada_parts(
            str(row["text"])
        )

        ids, positions = _completion_ids(
            tokenizer,
            prefix,
            word,
        )

        nll, lengths = _completion_nll(
            model,
            [ids],
            [positions],
            context_len=context_len,
            device=device,
        )

        total_nll += nll[0]
        token_count += lengths[0]

        prefix_ids = ids[:positions[0]]

        greedy = _greedy_last_word(
            model,
            prefix_ids,
            tokens_to_generate=lengths[0],
            context_len=context_len,
            device=device,
        )

        reference = ids[
            positions[0]:positions[-1] + 1
        ]

        correct += greedy == reference
        count += 1

    if not count:
        raise ValueError(
            "No LAMBADA examples evaluated"
        )

    mean_nll = total_nll / token_count

    return {
        "dataset": LAMBADA_DATASET,
        "config": "default",
        "split": "test",
        "metric": "exact_last_word_token_accuracy",
        "last_word_accuracy": correct / count,
        "correct": correct,
        "examples": count,
        "target_tokens": token_count,
        "target_token_nll": mean_nll,
        "target_word_perplexity": (
            math.exp(mean_nll)
            if mean_nll < 709
            else None
        ),
        "complete": limit is None and count == 5153,
        "expected_full_examples": 5153,
    }


def main() -> None:
    parser = argparse.ArgumentParser(
        description=__doc__
    )

    parser.add_argument(
        "--backend",
        choices=["playground", "hf"],
        required=True,
    )
    parser.add_argument("--model", required=True)
    parser.add_argument("--factory")
    parser.add_argument("--config")
    parser.add_argument("--device", default="cpu")
    parser.add_argument(
        "--context-len",
        type=int,
        default=1024,
    )
    parser.add_argument(
        "--limit",
        type=int,
        help="Smoke-test limit per benchmark",
    )
    parser.add_argument(
        "--offline",
        action="store_true",
        help="Use cached datasets only",
    )
    parser.add_argument("--output", required=True)

    args = parser.parse_args()

    if args.context_len <= 0:
        parser.error(
            "--context-len must be positive"
        )

    model = load_model(
        args.backend,
        args.model,
        factory=args.factory,
        config_path=args.config,
        device=args.device,
    )

    tokenizer = get_tokenizer()

    hellaswag = score_hellaswag(
        model,
        load_rows(
            "hellaswag",
            offline=args.offline,
        ),
        tokenizer,
        context_len=args.context_len,
        device=args.device,
        limit=args.limit,
    )

    lambada = score_lambada(
        model,
        load_rows(
            "lambada",
            offline=args.offline,
        ),
        tokenizer,
        context_len=args.context_len,
        device=args.device,
        limit=args.limit,
    )

    report = {
        "schema_version": 1,
        "model": args.model,
        "backend": args.backend,
        "tokenizer": "gpt2",
        "context_len": args.context_len,
        "benchmarks": {
            "hellaswag": hellaswag,
            "lambada": lambada,
        },
    }

    destination = Path(args.output)
    destination.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    destination.write_text(
        json.dumps(
            report,
            indent=2,
            ensure_ascii=False,
            allow_nan=False,
        ) + "\n",
        encoding="utf-8",
    )

    print(
        json.dumps(
            {
                "model": args.model,
                "hellaswag_accuracy": hellaswag[
                    "accuracy"
                ],
                "lambada_last_word_accuracy": lambada[
                    "last_word_accuracy"
                ],
                "lambada_target_word_perplexity": lambada[
                    "target_word_perplexity"
                ],
                "output": str(destination),
            },
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
