"""Token-weighted masked validation loss over one exact pass, as a receipt (Claude).

Replaces ad-hoc "mean of N batch means" smoke numbers with a figure two
checkpoints can be compared on:
  * one deterministic pass over the SFT val split, using the same
    example-aligned windows as training's val loader (each supervisable
    completion token scored exactly once; nothing repeated);
  * token-weighted: sum of completion-token NLL / number of completion tokens;
  * a receipt with the val/mask sha256, checkpoint sha256, window and token
    counts, so evals/stage_e_review_gates.py can check both checkpoints were
    scored on identical windows.
CPU is fine for the Stage E val split. What it measures is fit to the SFT
corpus; it is not a generalization or benchmark result.

Usage:
  python evals/masked_val_receipt.py --checkpoint candidate_31128/step_00031128.safetensors \\
      --factory playground_lm.model:PlaygroundLM --corpus sft_candidate --name E --out e_val.json
"""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F

from playground_lm.model import IGNORE_INDEX
from playground_lm.train.sft_data import _eval_windows, audit_sft_arrays, window_targets


def _sha256(path: Path) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for block in iter(lambda: f.read(1 << 20), b""):
            h.update(block)
    return h.hexdigest()


@torch.no_grad()
def masked_val(model: torch.nn.Module, tokens: np.ndarray, masks: np.ndarray, context_len: int,
               batch_size: int = 8, device: str = "cpu", **audit_kwargs) -> dict:
    """audit_kwargs go to audit_sft_arrays (eot_id / vocab_size for tests)."""
    audit, layout = audit_sft_arrays(tokens, masks, context_len, **audit_kwargs)
    windows = _eval_windows(layout, context_len, int(tokens.shape[0]) - context_len - 1)
    model.eval()
    nll_sum, count = 0.0, 0
    for i in range(0, len(windows), batch_size):
        rows = [window_targets(tokens, masks, layout.starts, s, context_len, (lo, hi))
                for s, lo, hi in windows[i : i + batch_size]]
        x = torch.from_numpy(np.stack([r[0] for r in rows])).to(device)
        y = torch.from_numpy(np.stack([r[1] for r in rows])).to(device)
        out = model(x)
        logits = out[0] if isinstance(out, tuple) else getattr(out, "logits", out)
        nll_sum += float(F.cross_entropy(logits.float().reshape(-1, logits.size(-1)), y.reshape(-1),
                                         ignore_index=IGNORE_INDEX, reduction="sum"))
        count += int((y != IGNORE_INDEX).sum())
    return {
        "windows": len(windows),
        "supervised_tokens": count,
        "expected_supervised_tokens": audit.supervised_tokens - audit.unsupervisable_tokens,
        "nll_sum": nll_sum,
        "token_weighted_loss": nll_sum / count if count else None,
        "single_pass": True,
        "context_len": context_len,
    }


def main(argv=None) -> None:
    p = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    p.add_argument("--checkpoint", required=True)
    p.add_argument("--factory", default="playground_lm.model:PlaygroundLM")
    p.add_argument("--config", help="model-config JSON (else sidecar / header / manifest)")
    p.add_argument("--corpus", required=True, help="dir with val.bin + val.masks.bin")
    p.add_argument("--name", required=True, help="label, e.g. C or E")
    p.add_argument("--context-len", type=int, default=1024)
    p.add_argument("--batch-size", type=int, default=8)
    p.add_argument("--device", default="cpu")
    p.add_argument("--out", required=True)
    args = p.parse_args(argv)

    from evals.run_eval import load_model

    corpus = Path(args.corpus)
    tokens = np.memmap(corpus / "val.bin", dtype=np.uint16, mode="r")
    masks = np.memmap(corpus / "val.masks.bin", dtype=np.uint8, mode="r")
    model = load_model("playground", args.checkpoint, device=args.device,
                       factory=args.factory, config_path=args.config)
    result = masked_val(model, tokens, masks, args.context_len, args.batch_size, args.device)
    receipt = {
        "schema_version": 1,
        "name": args.name,
        "checkpoint": Path(args.checkpoint).name,
        "checkpoint_sha256": _sha256(Path(args.checkpoint)),
        "val_sha256": _sha256(corpus / "val.bin"),
        "val_masks_sha256": _sha256(corpus / "val.masks.bin"),
        **result,
        "claim_scope": "fit to the Stage E SFT val split only; not generalization or a benchmark",
    }
    Path(args.out).write_text(json.dumps(receipt, indent=2) + "\n")
    print(json.dumps({k: receipt[k] for k in ("name", "windows", "supervised_tokens", "token_weighted_loss")}))


if __name__ == "__main__":
    main()
