"""Grow a trained PlaygroundLM into a deeper and/or wider one (Stage G, Track 2).

Design and first draft: Dee (depth duplication + Net2Net-style widening). Rewritten by
Lumi for this repo's actual state_dict keys (``blocks.{i}.attn.qkv`` etc., tied
``token_embedding``) and with honest function-preservation claims, all checked by
tests in ``tests/test_grow.py``.

Depth (``grow_depth``), three layer orders:

* ``"stack"`` (default): whole-model stacking, new order = old 0..n-1 then 0..n-1
  (the G_stack operator of Du et al., 2024, "Stacking Your Transformers"). Not
  function-preserving; reported to reach a given loss with far fewer tokens than
  training the bigger model from scratch.
* ``"duplicate"``: adjacent copies (0,0,1,1,...), Dee's original choice. Not
  function-preserving.
* ``"identity"``: adjacent copies whose attention and MLP output projections are
  zeroed, so each inserted block adds 0 to the residual stream. **Exactly**
  function-preserving (tested to float tolerance).

Width (``grow_width``) for the same head size (768/12 = 1024/16 = 64): old weights sit in
the leading block; new attention heads and MLP units get small random input weights;
every write back into the residual stream from a new unit, and every LayerNorm gain
and bias on a new residual channel, starts at 0, so new channels are silent at init.
This is **approximately** function-preserving only: LayerNorm statistics are now taken
over 1024 channels (256 of them zero), which shifts activations. Tests measure the drift
and require the grown model to load strictly and train without NaNs; they do not claim
exactness.
"""
from __future__ import annotations

import re

import torch

_BLOCK = re.compile(r"^blocks\.(\d+)\.(.+)$")
_OUT_PROJ = ("attn.c_proj.weight", "attn.c_proj.bias", "mlp.c_proj.weight", "mlp.c_proj.bias")


def _n_layers(state_dict: dict) -> int:
    idx = [int(m.group(1)) for k in state_dict if (m := _BLOCK.match(k))]
    if not idx:
        raise ValueError("no blocks.{i}.* keys found; not a PlaygroundLM state_dict")
    return max(idx) + 1


def grow_depth(state_dict: dict, factor: int = 2, mode: str = "stack") -> dict:
    """Return a new state_dict with ``factor`` x the layers. Input is not mutated."""
    if factor < 2:
        raise ValueError(f"factor must be >= 2, got {factor}")
    if mode not in ("stack", "duplicate", "identity"):
        raise ValueError(f"unknown mode {mode!r}")
    n = _n_layers(state_dict)
    if mode == "stack":
        order = [i for _ in range(factor) for i in range(n)]
    else:
        order = [i for i in range(n) for _ in range(factor)]
    out = {k: v.clone() for k, v in state_dict.items() if not _BLOCK.match(k)}
    seen: dict[int, int] = {}
    for new_i, old_i in enumerate(order):
        copy_no = seen.get(old_i, 0)
        seen[old_i] = copy_no + 1
        for k, v in state_dict.items():
            m = _BLOCK.match(k)
            if not m or int(m.group(1)) != old_i:
                continue
            t = v.clone()
            if mode == "identity" and copy_no > 0 and m.group(2) in _OUT_PROJ:
                t.zero_()
            out[f"blocks.{new_i}.{m.group(2)}"] = t
    return out


def _embed(old: torch.Tensor, shape: tuple[int, ...], fill: str, std: float,
           gen: torch.Generator) -> torch.Tensor:
    if fill == "zeros":
        new = torch.zeros(shape, dtype=old.dtype)
    elif fill == "ones":
        new = torch.ones(shape, dtype=old.dtype)
    else:
        new = (torch.randn(shape, generator=gen) * std).to(old.dtype)
    new[tuple(slice(0, s) for s in old.shape)] = old
    return new


def _widen_qkv(w: torch.Tensor, d_old: int, d_new: int, std: float, gen, is_bias: bool) -> torch.Tensor:
    """qkv rows are [q | k | v], each d wide with heads contiguous: widen each block."""
    parts = w.split(d_old, dim=0)
    if is_bias:
        return torch.cat([_embed(p, (d_new,), "zeros", std, gen) for p in parts])
    return torch.cat([_embed(p, (d_new, d_new), "randn", std, gen) for p in parts])


def grow_width(state_dict: dict, d_new: int, *, n_heads_old: int, n_heads_new: int,
               std: float = 0.02, seed: int = 0) -> dict:
    """Widen to ``d_new`` keeping head size fixed. Approximately function-preserving."""
    d_old = state_dict["token_embedding.weight"].shape[1]
    if d_new <= d_old:
        raise ValueError(f"d_new={d_new} must exceed d_old={d_old}")
    if d_old % n_heads_old or d_new % n_heads_new or d_old // n_heads_old != d_new // n_heads_new:
        raise ValueError("head size must stay the same (e.g. 768/12 -> 1024/16)")
    gen = torch.Generator().manual_seed(seed)
    h_old, h_new = 4 * d_old, 4 * d_new
    out: dict = {}
    for k, v in state_dict.items():
        name = _BLOCK.match(k).group(2) if _BLOCK.match(k) else k
        if name in ("token_embedding.weight", "position_embedding.weight"):
            new = _embed(v, (v.shape[0], d_new), "zeros", std, gen)
        elif name in ("ln_1.weight", "ln_2.weight", "ln_f.weight", "ln_1.bias", "ln_2.bias", "ln_f.bias"):
            new = _embed(v, (d_new,), "zeros", std, gen)  # new channels silent
        elif name == "attn.qkv.weight":
            new = _widen_qkv(v, d_old, d_new, std, gen, is_bias=False)
        elif name == "attn.qkv.bias":
            new = _widen_qkv(v, d_old, d_new, std, gen, is_bias=True)
        elif name == "attn.c_proj.weight":  # [d, d]: new heads write 0 to the residual
            new = _embed(v, (d_new, d_new), "zeros", std, gen)
        elif name in ("attn.c_proj.bias", "mlp.c_proj.bias"):
            new = _embed(v, (d_new,), "zeros", std, gen)
        elif name == "mlp.c_fc.weight":  # [4d, d]
            new = _embed(v, (h_new, d_new), "randn", std, gen)
        elif name == "mlp.c_fc.bias":
            new = _embed(v, (h_new,), "zeros", std, gen)
        elif name == "mlp.c_proj.weight":  # [d, 4d]: new units write 0
            new = _embed(v, (d_new, h_new), "zeros", std, gen)
        else:
            raise KeyError(f"grow_width does not know how to widen {k} {tuple(v.shape)}")
        assert new.shape[0] >= v.shape[0]
        out[k] = new
    assert h_old < h_new
    return out
