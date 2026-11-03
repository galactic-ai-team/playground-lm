#!/usr/bin/env python3
"""HellaSwag and OpenAI-LAMBADA evaluation for causal LMs.

HellaSwag: Rowan/hellaswag validation, 10,042 examples.
  Accuracy by highest length-normalized completion loglikelihood.

LAMBADA: EleutherAI/lambada_openai default/test, 5,153 examples.
  Accuracy: every reference final-word token must be greedy.
  Target-word PPL: exp(mean negative loglikelihood per example).
  Target-token PPL is also reported separately.

Uses the same load_model/get_logits adapters as run_eval.py.
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

from evals.run_eval import get_logits, load_model
from playground_lm.tokenizer import get_tokenizer

HELLASWAG_DATASET = "Rowan/hellaswag"
LAMBADA_DATASET = "EleutherAI/lambada_openai"
HELLASWAG_SIZE = 10042
LAMBADA_SIZE = 5153


def load_rows(
    dataset: str, *, offline: bool = False
) -> Iterable[Mapping[str, Any]]:
    """Load the official dataset split."""
    from datasets import load_dataset

    if dataset == "hellaswag":
        name, config, split = HELLASWAG_DATASET, None, "validation"
    elif dataset == "lambada":
        name, config, split = LAMBADA_DATASET, "default", "test"
    else:
        raise ValueError(f"Unknown benchmark: {dataset}")

    if offline:
        from datasets import DownloadConfig

        return load_dataset(
            name,
            config,
            split=split,
            download_config=DownloadConfig(local_files_only=True),
        )

    return load_dataset(name, config, split=split)


def _encode(tokenizer: Any, text: str) -> list[int]:
    ids = list(tokenizer.encode(text))
    if not ids:
        raise ValueError("Empty token sequence")
    return ids


def _completion_ids(
    tokenizer: Any,
    prefix: str,
    suffix: str,
) -> tuple[list[int], list[int]]:
    """Tokenize a context/continuation like the causal LM Eval Harness.

    Move trailing context whitespace into the continuation, tokenize
    the entire joined string, then take the suffix AFTER len(encoded
    context). Do not require the prefix IDs to match: BPE can change
    merges across the string boundary.

    Returns complete token sequence and absolute target positions.
    """
    if not prefix or not suffix:
        raise ValueError("Context and continuation must be nonempty")

    n_spaces = len(prefix) - len(prefix.rstrip())
    if n_spaces:
        suffix = prefix[-n_spaces:] + suffix
        prefix = prefix[:-n_spaces]
        if not prefix:
            raise ValueError("Context cannot be only whitespace")

    context_ids = _encode(tokenizer, prefix)
    joined = _encode(tokenizer, prefix + suffix)

    split = len(context_ids)
    if split >= len(joined):
        # Rare pathological boundary merges can consume an entire
        # continuation. Keep at least one token for scoring while
        # retaining the joint tokenization; never silently skip an
        # example. Normal GPT-2/HellaSwag completions do not need this.
        split = len(joined) - 1

    if split < 1:
        raise ValueError(
            "Combined text has no predictable completion token"
        )

    return joined, list(range(split, len(joined)))


@torch.inference_mode()
def _completion_stats(
    model: torch.nn.Module,
    sequences: Sequence[list[int]],
    target_positions: Sequence[list[int]],
    *,
    context_len: int,
    device: str,
) -> tuple[list[float], list[int], list[bool]]:
    """Return NLL sums, token counts, and full greedy-match flags.

    A target at position p is predicted by logits at p - 1.
    Greedy correctness checks EVERY reference target token using
    teacher forcing. Full-word greedy correctness is equivalent
    to generating those exact tokens autoregressively from context.

    Long examples use a rolling context for each target position.
    """
    if context_len <= 0:
        raise ValueError("context_len must be positive")
    if len(sequences) != len(target_positions):
        raise ValueError("Sequences and targets must have equal lengths")

    sums: list[float] = []
    counts: list[int] = []
    greedy_flags: list[bool] = []

    for ids, positions in zip(sequences, target_positions):
        if not positions or any(
            p < 1 or p >= len(ids) for p in positions
        ):
            raise ValueError("Invalid completion target positions")

        if positions != list(range(positions[0], positions[-1] + 1)):
            raise ValueError("Completion positions must be contiguous")

        total_nll = 0.0
        all_greedy = True

        if len(ids) - 1 <= context_len:
            # One forward pass covers all targets.
            x = torch.tensor(
                [ids[:-1]], dtype=torch.long, device=device
            )
            logits = get_logits(model, x)[0].float()

            positions_tensor = torch.tensor(
                positions, dtype=torch.long, device=device
            )
            target = torch.tensor(
                [ids[p] for p in positions],
                dtype=torch.long,
                device=device,
            )
            selected = logits.index_select(
                0, positions_tensor - 1
            )

            total_nll = float(
                F.cross_entropy(
                    selected,
                    target,
                    reduction="sum",
                ).item()
            )
            all_greedy = bool(
                torch.equal(selected.argmax(dim=-1), target)
            )

        else:
            # Score each reference token with the longest available
            # left context, without exceeding the model context length.
            for p in positions:
                start = max(0, p - context_len)
                x = torch.tensor(
                    [ids[start:p]],
                    dtype=torch.long,
                    device=device,
                )
                logits = get_logits(model, x)[0, -1].float()
                target_id = ids[p]

                total_nll += float(
                    F.cross_entropy(
                        logits.unsqueeze(0),
                        torch.tensor(
                            [target_id],
                            dtype=torch.long,
                            device=device,
                        ),
                    ).item()
                )

                if int(logits.argmax().item()) != target_id:
                    all_greedy = False

        sums.append(total_nll)
        counts.append(len(positions))
        greedy_flags.append(all_greedy)

    return sums, counts, greedy_flags


def _completion_nll(
    model: torch.nn.Module,
    sequences: Sequence[list[int]],
    target_positions: Sequence[list[int]],
    *,
    context_len: int,
    device: str,
) -> tuple[list[float], list[int]]:
    """Backward-compatible NLL interface used by existing tests."""
    sums, counts, _ = _completion_stats(
        model,
        sequences,
        target_positions,
        context_len=context_len,
        device=device,
    )
    return sums, counts


def _hellaswag_preprocess(text: str) -> str:
    """Match EleutherAI's HellaSwag WikiHow cleanup."""
    text = text.strip()
    text = text.replace(" [title]", ". ")
    text = re.sub(r"\[.*?\]", "", text)
    return text.replace("  ", " ")


def _hellaswag_context(row: Mapping[str, Any]) -> str:
    """Use activity label and ctx_a/ctx_b as in the harness."""
    ctx = (
        str(row["ctx_a"])
        + " "
        + str(row["ctx_b"]).capitalize()
    )

    activity = str(row.get("activity_label") or "")
    if activity:
        ctx = activity + ": " + ctx

    return _hellaswag_preprocess(ctx)


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
    """Choose lowest mean completion NLL over four endings."""
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
                "HellaSwag example must have four endings"
            )

        sequences = []
        targets = []

        for ending in endings:
            continuation = " " + _hellaswag_preprocess(
                str(ending)
            )
            ids, positions = _completion_ids(
                tokenizer,
                context,
                continuation,
            )
            sequences.append(ids)
            targets.append(positions)

        nll_sums, lengths = _completion_nll(
            model,
            sequences,
            targets,
            context_len=context_len,
            device=device,
        )

        predicted = min(
            range(4),
            key=lambda i: nll_sums[i] / lengths[i],
        )

        correct += int(predicted == int(row["label"]))
        count += 1

    if not count:
        raise ValueError("No HellaSwag examples evaluated")

    return {
        "dataset": HELLASWAG_DATASET,
        "split": "validation",
        "metric": "accuracy_normalized_completion_nll",
        "accuracy": correct / count,
        "correct": correct,
        "examples": count,
        "complete": limit is None and count == HELLASWAG_SIZE,
        "expected_full_examples": HELLASWAG_SIZE,
    }


def _lambada_parts(text: str) -> tuple[str, str]:
    """Match lambada_openai.yaml's literal-space word split.

    Reference task:
        context = ' '.join(text.split(' ')[:-1])
        target  = ' ' + text.split(' ')[-1]
    """
    parts = text.split(" ")
    if len(parts) < 2 or not parts[-1]:
        raise ValueError(
            "LAMBADA passage requires a nonempty final word"
        )

    prefix = " ".join(parts[:-1])
    if not prefix.strip():
        raise ValueError(
            "LAMBADA passage requires preceding context"
        )

    return prefix, " " + parts[-1]


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
    """Harness-style last-word greedy accuracy and perplexity."""
    if limit is not None and limit <= 0:
        raise ValueError("limit must be positive")

    correct = 0
    examples = 0
    target_tokens = 0
    total_nll = 0.0

    for row in rows:
        if limit is not None and examples >= limit:
            break

        prefix, word = _lambada_parts(str(row["text"]))
        ids, positions = _completion_ids(
            tokenizer,
            prefix,
            word,
        )

        nll_sums, lengths, greedy_flags = _completion_stats(
            model,
            [ids],
            [positions],
            context_len=context_len,
            device=device,
        )

        total_nll += nll_sums[0]
        target_tokens += lengths[0]
        correct += int(greedy_flags[0])
        examples += 1

    if not examples:
        raise ValueError("No LAMBADA examples evaluated")

    word_nll = total_nll / examples
    token_nll = total_nll / target_tokens

    return {
        "dataset": LAMBADA_DATASET,
        "config": "default",
        "split": "test",
        "metric": "exact_last_word_token_accuracy",
        "last_word_accuracy": correct / examples,
        "correct": correct,
        "examples": examples,
        "target_tokens": target_tokens,
        # Retain existing output keys for downstream consumers.
        "target_token_nll": token_nll,
        # Standard harness perplexity aggregation is per example/word.
        "target_word_perplexity": (
            math.exp(word_nll) if word_nll < 709 else None
        ),
        # Explicit token-normalized alternative, for diagnostics.
        "target_token_perplexity": (
            math.exp(token_nll) if token_nll < 709 else None
        ),
        "complete": limit is None and examples == LAMBADA_SIZE,
        "expected_full_examples": LAMBADA_SIZE,
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
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
        "--context-len", type=int, default=1024
    )
    parser.add_argument(
        "--limit",
        type=int,
        help="Smoke-test limit; results will be incomplete",
    )
    parser.add_argument(
        "--offline", action="store_true"
    )
    parser.add_argument("--output", required=True)
    args = parser.parse_args()

    if args.context_len <= 0:
        parser.error("--context-len must be positive")

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
            "hellaswag", offline=args.offline
        ),
        tokenizer,
        context_len=args.context_len,
        device=args.device,
        limit=args.limit,
    )

    lambada = score_lambada(
        model,
        load_rows(
            "lambada", offline=args.offline
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
        parents=True, exist_ok=True
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
                "hellaswag_accuracy": hellaswag["accuracy"],
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
