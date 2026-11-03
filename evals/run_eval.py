#!/usr/bin/env python3
"""Held-out evaluation for PlaygroundLM and GPT-2.

Supports combined PyTorch checkpoints and Dee's split safetensors
checkpoints without conversion.
"""

from __future__ import annotations

import argparse
import importlib
import json
import math
import mmap
import random
from pathlib import Path
from typing import Any, Callable, Mapping

import numpy as np
import torch
import torch.nn.functional as F

from playground_lm.tokenizer import get_tokenizer, VOCAB_SIZE

COPY_THRESHOLD = 50


def load_factory(spec: str) -> Callable[..., Any]:
    if ":" not in spec:
        raise ValueError("Factory must have format module:callable")
    module_name, name = spec.split(":", 1)
    obj = importlib.import_module(module_name)
    for part in name.split("."):
        obj = getattr(obj, part)
    if not callable(obj):
        raise TypeError(f"{spec} is not callable")
    return obj


def model_config(config: Any) -> dict[str, Any]:
    """Accept flat kwargs or Dee's nested config.model mapping."""
    if not isinstance(config, Mapping):
        raise ValueError("Model config must be a mapping")
    if "model" in config:
        config = config["model"]
        if not isinstance(config, Mapping):
            raise ValueError("config.model must be a mapping")
    if not config:
        raise ValueError("Empty model configuration")
    return dict(config)


def read_model_config(path: str | Path) -> dict[str, Any]:
    document = json.loads(
        Path(path).read_text(encoding="utf-8")
    )
    if not isinstance(document, dict):
        raise ValueError("Model config JSON must be an object")
    return model_config(document.get("config", document))


def split_checkpoint_config(
    weights_path: str | Path,
    config_path: str | Path | None = None,
) -> dict[str, Any]:
    """Discover model kwargs for Dee's split safetensors format.

    Resolution:
    1. Explicit --config JSON
    2. JSON next to weights, or run-level manifest.json
    3. Paired step_XXXXXXXX_state.pt, read with weights_only=True

    The optimizer state is never loaded into the inference model.
    """
    path = Path(weights_path)
    if config_path is not None:
        return read_model_config(config_path)

    candidates = [
        path.with_name(path.stem + "_model_config.json"),
        path.parent / "model-config.json",
        path.parent / "model_config.json",
        path.parent / "config.json",
        path.parent / "manifest.json",
        path.parent.parent / "manifest.json",
    ]
    for candidate in candidates:
        if candidate.is_file():
            document = json.loads(
                candidate.read_text(encoding="utf-8")
            )
            if not isinstance(document, dict):
                continue
            data = document.get("config", document)
            if (
                isinstance(data, dict)
                and (
                    "model" in data
                    or "vocab_size" in data
                )
            ):
                return model_config(data)

    # Dee's CheckpointManager pairs these two files.
    state_path = path.with_name(path.stem + "_state.pt")
    if state_path.is_file():
        training_state = torch.load(
            state_path,
            map_location="cpu",
            weights_only=True,
        )
        if (
            isinstance(training_state, Mapping)
            and "config" in training_state
        ):
            return model_config(training_state["config"])

    raise FileNotFoundError(
        "No model config found for split checkpoint "
        f"{path}. Supply --config with a model-config JSON, "
        "or include the run manifest / paired _state.pt file."
    )


def checkpoint_weights(
    checkpoint: Any,
) -> Mapping[str, torch.Tensor]:
    """Extract tensor state dict from supported PyTorch formats."""
    if not isinstance(checkpoint, Mapping):
        raise ValueError("Checkpoint must be a mapping")

    for key in ("model", "model_state_dict", "state_dict"):
        candidate = checkpoint.get(key)
        if isinstance(candidate, Mapping):
            if candidate and all(
                isinstance(value, torch.Tensor)
                for value in candidate.values()
            ):
                return candidate

    if checkpoint and all(
        isinstance(value, torch.Tensor)
        for value in checkpoint.values()
    ):
        return checkpoint

    raise ValueError("Checkpoint contains no model state dict")


def load_model(
    backend: str,
    model_ref: str,
    *,
    device: str,
    factory: str | None = None,
    config_path: str | None = None,
) -> torch.nn.Module:
    """Load HF GPT-2, combined .pt, or split .safetensors.

    For safetensors, architecture kwargs come from JSON, a run
    manifest, or Dee's paired _state.pt.
    """
    if backend == "hf":
        from transformers import AutoModelForCausalLM

        model = AutoModelForCausalLM.from_pretrained(
            model_ref,
            trust_remote_code=False,
        )

    elif backend == "playground":
        if factory is None:
            raise ValueError("--factory is required for playground")

        path = Path(model_ref)
        if path.suffix == ".safetensors":
            from safetensors.torch import load_file

            state = load_file(str(path), device="cpu")
            config = split_checkpoint_config(
                path, config_path
            )
        else:
            checkpoint = torch.load(
                path,
                map_location="cpu",
                weights_only=True,
            )
            state = checkpoint_weights(checkpoint)

            if config_path is not None:
                config = read_model_config(config_path)
            elif (
                isinstance(checkpoint, Mapping)
                and "config" in checkpoint
            ):
                config = model_config(checkpoint["config"])
            else:
                raise ValueError(
                    "Combined checkpoint has no model config; "
                    "supply --config"
                )

        make_model = load_factory(factory)
        model = make_model(**config)
        model.load_state_dict(state, strict=True)

    else:
        raise ValueError(f"Unsupported backend: {backend}")

    model.to(device)
    model.eval()
    return model


def get_logits(
    model: torch.nn.Module,
    x: torch.Tensor,
) -> torch.Tensor:
    output = model(x)

    if isinstance(output, torch.Tensor):
        logits = output
    elif isinstance(output, (tuple, list)):
        logits = output[0]
    elif hasattr(output, "logits"):
        logits = output.logits
    else:
        raise TypeError(
            "Model must return logits, tuple, or .logits"
        )

    if not isinstance(logits, torch.Tensor):
        raise TypeError("Model logits must be a tensor")
    if logits.ndim != 3:
        raise ValueError("Expected logits shape [B, T, V]")
    if logits.shape[:2] != x.shape:
        raise ValueError("Logits and input lengths differ")

    return logits


@torch.inference_mode()
def validation_metrics(
    model: torch.nn.Module,
    val_bin: str | Path,
    *,
    context_len: int,
    batch_size: int,
    device: str,
    max_batches: int | None = None,
) -> dict[str, Any]:
    """Token-weighted held-out loss over nonoverlapping windows."""
    if context_len <= 0 or batch_size <= 0:
        raise ValueError(
            "context_len and batch_size must be positive"
        )
    if max_batches is not None and max_batches <= 0:
        raise ValueError("max_batches must be positive")

    path = Path(val_bin)
    if path.stat().st_size % 2:
        raise ValueError("Validation file is not uint16-aligned")

    tokens = np.memmap(path, dtype="<u2", mode="r")
    if len(tokens) < 2:
        raise ValueError(
            "Validation requires at least two tokens"
        )
    if int(tokens.max()) >= VOCAB_SIZE:
        raise ValueError(
            "Validation contains out-of-vocabulary IDs"
        )

    target_count = len(tokens) - 1
    total_loss = 0.0
    scored = 0
    batches = 0

    for batch_start in range(
        0, target_count, context_len * batch_size
    ):
        if max_batches is not None and batches >= max_batches:
            break

        items = []
        for offset in range(batch_size):
            start = batch_start + offset * context_len
            if start >= target_count:
                break
            stop = min(
                start + context_len, target_count
            )
            x = np.asarray(
                tokens[start:stop], dtype=np.int64
            )
            y = np.asarray(
                tokens[start + 1:stop + 1],
                dtype=np.int64,
            )
            items.append((x, y))

        if not items:
            break

        longest = max(len(x) for x, _ in items)
        x_batch = torch.zeros(
            (len(items), longest), dtype=torch.long
        )
        y_batch = torch.full(
            (len(items), longest),
            -100,
            dtype=torch.long,
        )

        for row, (x, y) in enumerate(items):
            x_batch[row, :len(x)] = torch.from_numpy(x)
            y_batch[row, :len(y)] = torch.from_numpy(y)

        x_batch = x_batch.to(device)
        y_batch = y_batch.to(device)
        logits = get_logits(model, x_batch)

        loss_sum = F.cross_entropy(
            logits.float().reshape(
                -1, logits.shape[-1]
            ),
            y_batch.reshape(-1),
            ignore_index=-100,
            reduction="sum",
        )
        count = int(
            (y_batch != -100).sum().item()
        )
        total_loss += float(loss_sum.item())
        scored += count
        batches += 1

    if scored == 0:
        raise ValueError("No validation tokens scored")

    mean_loss = total_loss / scored
    return {
        "loss": mean_loss,
        "perplexity": (
            math.exp(mean_loss)
            if mean_loss < 709 else None
        ),
        "tokens_scored": scored,
        "available_targets": target_count,
        "coverage": scored / target_count,
        "batches": batches,
        "context_len": context_len,
        "complete": scored == target_count,
    }


@torch.inference_mode()
def generate_tokens(
    model: torch.nn.Module,
    prompt_ids: list[int],
    *,
    max_new_tokens: int,
    context_len: int,
    device: str,
    eot_id: int,
    temperature: float = 0.0,
    top_k: int = 0,
    seed: int = 42,
) -> list[int]:
    """Generate continuation tokens only."""
    if not prompt_ids:
        raise ValueError("Empty tokenized prompt")
    if max_new_tokens < 0 or context_len <= 0:
        raise ValueError("Invalid generation dimensions")
    if temperature < 0 or top_k < 0:
        raise ValueError(
            "temperature/top_k must be nonnegative"
        )

    ids = list(prompt_ids)
    generated: list[int] = []
    generator = torch.Generator(device=device)
    generator.manual_seed(seed)

    for _ in range(max_new_tokens):
        x = torch.tensor(
            [ids[-context_len:]],
            dtype=torch.long,
            device=device,
        )
        logits = get_logits(
            model, x
        )[0, -1].float()

        if temperature == 0:
            next_id = int(
                torch.argmax(logits).item()
            )
        else:
            logits = logits / temperature
            if top_k:
                k = min(top_k, logits.numel())
                values, indices = torch.topk(
                    logits, k
                )
                probabilities = torch.softmax(
                    values, dim=-1
                )
                choice = torch.multinomial(
                    probabilities, 1,
                    generator=generator,
                )
                next_id = int(
                    indices[choice].item()
                )
            else:
                probabilities = torch.softmax(
                    logits, dim=-1
                )
                next_id = int(
                    torch.multinomial(
                        probabilities, 1,
                        generator=generator,
                    ).item()
                )

        if next_id == eot_id:
            break
        ids.append(next_id)
        generated.append(next_id)

    return generated


def copy_matches(
    train_bin: str | Path,
    generations: list[list[int]],
    *,
    threshold: int = COPY_THRESHOLD,
) -> list[dict[str, Any]]:
    """Exact aligned uint16 token matches in training data."""
    if threshold <= 0:
        raise ValueError("threshold must be positive")

    path = Path(train_bin)
    if path.stat().st_size % 2:
        raise ValueError(
            "Training file is not uint16-aligned"
        )

    results: list[dict[str, Any]] = [
        {
            "flagged": False,
            "threshold_tokens": threshold,
            "match_start_in_generation": None,
            "match_start_in_training": None,
        }
        for _ in generations
    ]

    windows: dict[
        bytes, list[tuple[int, int]]
    ] = {}

    for index, ids in enumerate(generations):
        if any(
            token < 0 or token >= VOCAB_SIZE
            for token in ids
        ):
            raise ValueError(
                "Generation contains invalid GPT-2 IDs"
            )
        if len(ids) < threshold:
            continue

        data = np.asarray(
            ids, dtype="<u2"
        ).tobytes()

        for start in range(
            len(ids) - threshold + 1
        ):
            key = data[
                start * 2:(start + threshold) * 2
            ]
            windows.setdefault(key, []).append(
                (index, start)
            )

    if not windows or path.stat().st_size == 0:
        return results

    with path.open("rb") as source:
        with mmap.mmap(
            source.fileno(), 0,
            access=mmap.ACCESS_READ,
        ) as corpus:
            for needle, occurrences in windows.items():
                offset = 0
                found_at = -1
                while True:
                    position = corpus.find(
                        needle, offset
                    )
                    if position < 0:
                        break
                    if position % 2 == 0:
                        found_at = position
                        break
                    offset = position + 1

                if found_at < 0:
                    continue

                for index, start in occurrences:
                    result = results[index]
                    if not result["flagged"]:
                        result["flagged"] = True
                        result[
                            "match_start_in_generation"
                        ] = start
                        result[
                            "match_start_in_training"
                        ] = found_at // 2

    return results


def evaluate_prompts(
    model: torch.nn.Module,
    prompts_file: str | Path,
    *,
    tokenizer: Any,
    context_len: int,
    max_new_tokens: int,
    device: str,
    temperature: float,
    top_k: int,
    seed: int,
) -> tuple[
    list[dict[str, Any]],
    list[list[int]],
]:
    document = json.loads(
        Path(prompts_file).read_text(
            encoding="utf-8"
        )
    )
    records = []
    continuations = []

    for index, probe in enumerate(
        document["held_out_probes"]
    ):
        prompt = probe["prompt"]
        prompt_ids = tokenizer.encode(prompt)
        if not prompt_ids:
            raise ValueError(
                f"Empty prompt: {probe['id']}"
            )
        if len(prompt_ids) > context_len:
            raise ValueError(
                f"Prompt exceeds context: {probe['id']}"
            )

        generated = generate_tokens(
            model,
            prompt_ids,
            max_new_tokens=max_new_tokens,
            context_len=context_len,
            device=device,
            eot_id=tokenizer.eot_token_id,
            temperature=temperature,
            top_k=top_k,
            seed=seed + index,
        )
        continuations.append(generated)
        records.append({
            "id": probe["id"],
            "category": probe.get("category"),
            "evaluation": probe.get("evaluation"),
            "prompt": prompt,
            "reference": probe.get("reference"),
            "generation": tokenizer.decode(
                generated
            ),
            "generated_tokens": len(generated),
        })

    return records, continuations


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Evaluate PlaygroundLM or GPT-2"
    )
    parser.add_argument(
        "--backend",
        choices=["playground", "hf"],
        required=True,
    )
    parser.add_argument(
        "--model", required=True
    )
    parser.add_argument("--factory")
    parser.add_argument("--config")
    parser.add_argument(
        "--val-bin", required=True
    )
    parser.add_argument("--train-bin")
    parser.add_argument(
        "--prompts", default="evals/prompts.json"
    )
    parser.add_argument(
        "--output", required=True
    )
    parser.add_argument(
        "--device", default="cpu"
    )
    parser.add_argument(
        "--context-len",
        type=int, default=1024,
    )
    parser.add_argument(
        "--batch-size", type=int, default=4
    )
    parser.add_argument("--max-batches", type=int)
    parser.add_argument(
        "--max-new-tokens",
        type=int, default=80,
    )
    parser.add_argument(
        "--temperature", type=float,
        default=0.0,
    )
    parser.add_argument(
        "--top-k", type=int, default=0
    )
    parser.add_argument(
        "--seed", type=int, default=42
    )
    args = parser.parse_args()

    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(args.seed)

    tokenizer = get_tokenizer()
    model = load_model(
        args.backend,
        args.model,
        device=args.device,
        factory=args.factory,
        config_path=args.config,
    )

    validation = validation_metrics(
        model,
        args.val_bin,
        context_len=args.context_len,
        batch_size=args.batch_size,
        device=args.device,
        max_batches=args.max_batches,
    )
    records, continuations = evaluate_prompts(
        model,
        args.prompts,
        tokenizer=tokenizer,
        context_len=args.context_len,
        max_new_tokens=args.max_new_tokens,
        device=args.device,
        temperature=args.temperature,
        top_k=args.top_k,
        seed=args.seed,
    )

    if args.train_bin:
        findings = copy_matches(
            args.train_bin, continuations
        )
        for record, finding in zip(
            records, findings
        ):
            record["verbatim_copy"] = finding
        copy_summary = {
            "checked": True,
            "threshold_tokens": COPY_THRESHOLD,
            "flagged_generations": sum(
                finding["flagged"]
                for finding in findings
            ),
        }
    else:
        copy_summary = {
            "checked": False,
            "reason": "--train-bin was not provided",
        }

    report = {
        "schema_version": 1,
        "model": args.model,
        "backend": args.backend,
        "tokenizer": "gpt2",
        "seed": args.seed,
        "temperature": args.temperature,
        "top_k": args.top_k,
        "max_new_tokens": args.max_new_tokens,
        "validation": validation,
        "copy_check": copy_summary,
        "generations": records,
    }

    destination = Path(args.output)
    destination.parent.mkdir(
        parents=True, exist_ok=True
    )
    destination.write_text(
        json.dumps(
            report, indent=2,
            ensure_ascii=False,
        ) + "\n",
        encoding="utf-8",
    )
    print(json.dumps({
        "model": args.model,
        "loss": validation["loss"],
        "perplexity": validation["perplexity"],
        "coverage": validation["coverage"],
        "copy_flags": copy_summary.get(
            "flagged_generations"
        ),
        "output": str(destination),
    }, indent=2))


if __name__ == "__main__":
    main()
