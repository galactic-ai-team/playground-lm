"""Export PlaygroundLM weights to a portable safetensors bundle.

Accepts:
- combined PyTorch .pt checkpoint
- Dee's split .safetensors checkpoint

Example:
    python -m playground_lm.export_checkpoint \
        checkpoints/step_00000003.safetensors \
        exports/step_00000003
"""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
from typing import Any, Mapping

import torch
from safetensors.torch import load_file, save_file

from .tokenizer import (
    CONTEXT_LENGTH,
    EOT_ID,
    VOCAB_SIZE,
)

MODEL_KEYS = (
    "model",
    "model_state_dict",
    "state_dict",
)


def unwrap_state_dict(
    checkpoint: Any,
) -> Mapping[str, Any]:
    if not isinstance(checkpoint, Mapping):
        raise TypeError(
            "Checkpoint must be a mapping"
        )

    for key in MODEL_KEYS:
        candidate = checkpoint.get(key)
        if (
            isinstance(candidate, Mapping)
            and candidate
            and all(
                isinstance(value, torch.Tensor)
                for value in candidate.values()
            )
        ):
            return candidate

    if checkpoint and all(
        isinstance(value, torch.Tensor)
        for value in checkpoint.values()
    ):
        return checkpoint

    raise ValueError(
        "No tensor state dict found"
    )


def prepare_tensors(
    state: Mapping[str, Any],
) -> dict[str, torch.Tensor]:
    result = {}

    for name, value in state.items():
        if not isinstance(name, str) or not name:
            raise ValueError(
                "State keys must be nonempty strings"
            )
        if not isinstance(value, torch.Tensor):
            raise TypeError(
                f"{name} is not a tensor"
            )

        result[name] = (
            value.detach()
            .to("cpu")
            .contiguous()
            .clone()
        )

    if not result:
        raise ValueError(
            "Cannot export an empty model"
        )
    return result


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as source:
        for chunk in iter(
            lambda: source.read(1024 * 1024),
            b"",
        ):
            digest.update(chunk)
    return digest.hexdigest()


def export_checkpoint(
    source: str | Path,
    destination: str | Path,
    *,
    model_config: dict[str, Any] | None = None,
) -> dict[str, Any]:
    source = Path(source)
    destination = Path(destination)

    if not source.is_file():
        raise FileNotFoundError(source)

    if source.suffix == ".safetensors":
        checkpoint = None
        state = prepare_tensors(
            load_file(str(source), device="cpu")
        )
        # Obtain architecture config from the same discovery path
        # used by the evaluator.
        from evals.run_eval import split_checkpoint_config

        try:
            config = split_checkpoint_config(source)
        except FileNotFoundError:
            if model_config is None:
                raise
            config = {}

    else:
        checkpoint = torch.load(
            source,
            map_location="cpu",
            weights_only=True,
        )
        state = prepare_tensors(
            unwrap_state_dict(checkpoint)
        )
        config = {}

        if isinstance(checkpoint, Mapping):
            raw = checkpoint.get("config", {})
            if isinstance(raw, Mapping):
                if isinstance(
                    raw.get("model"), Mapping
                ):
                    raw = raw["model"]
                for key, value in raw.items():
                    if not isinstance(key, str):
                        continue
                    try:
                        json.dumps(
                            value, allow_nan=False
                        )
                    except (
                        TypeError, ValueError
                    ):
                        continue
                    config[key] = value

    if model_config is not None:
        config.update(model_config)

    destination.mkdir(
        parents=True, exist_ok=True
    )
    weights_path = (
        destination / "model.safetensors"
    )

    save_file(
        state,
        str(weights_path),
        metadata={
            "format": "pt",
            "model": "PlaygroundLM",
            "tokenizer": "gpt2",
        },
    )

    stored_elements = sum(
        tensor.numel()
        for tensor in state.values()
    )

    manifest: dict[str, Any] = {
        "format_version": 1,
        "architecture": "PlaygroundLM",
        "training": "random_initialization",
        "artifact_type": "inference_weights",
        "weights": weights_path.name,
        "weights_sha256": sha256_file(
            weights_path
        ),
        "source_checkpoint_sha256": sha256_file(
            source
        ),
        "stored_tensor_elements": stored_elements,
        "tensor_count": len(state),
        "tokenizer": {
            "encoding": "gpt2",
            "vocab_size": VOCAB_SIZE,
            "eot_token_id": EOT_ID,
            "context_length": CONTEXT_LENGTH,
        },
        "model_config": config,
    }

    (destination / "manifest.json").write_text(
        json.dumps(
            manifest,
            indent=2,
            sort_keys=True,
            allow_nan=False,
        ) + "\n",
        encoding="utf-8",
    )
    return manifest


def main() -> None:
    parser = argparse.ArgumentParser(
        description=__doc__
    )
    parser.add_argument(
        "checkpoint", type=Path
    )
    parser.add_argument(
        "destination", type=Path
    )
    parser.add_argument(
        "--config-json", type=Path
    )
    args = parser.parse_args()

    config = None
    if args.config_json:
        config = json.loads(
            args.config_json.read_text(
                encoding="utf-8"
            )
        )
        if not isinstance(config, dict):
            parser.error(
                "--config-json must contain an object"
            )
        if isinstance(
            config.get("model"), dict
        ):
            config = config["model"]

    manifest = export_checkpoint(
        args.checkpoint,
        args.destination,
        model_config=config,
    )

    print(
        f"Exported {manifest['tensor_count']} "
        f"tensors to {args.destination}"
    )


if __name__ == "__main__":
    main()
