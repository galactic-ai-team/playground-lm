"""masked_val_receipt: one exact, token-weighted pass (Claude). Needs real PyTorch."""

import numpy as np
import pytest

torch = pytest.importorskip("torch")
if not hasattr(getattr(torch, "nn", None), "Linear"):
    pytest.skip("needs a real PyTorch install", allow_module_level=True)
import torch.nn.functional as F  # noqa: E402

from evals.masked_val_receipt import masked_val  # noqa: E402
from playground_lm.model import IGNORE_INDEX, PlaygroundLM  # noqa: E402
from playground_lm.train.sft_data import _eval_windows, audit_sft_arrays, window_targets  # noqa: E402

EOT = 63


def pack(n):
    toks, masks = [], []
    for i in range(n):
        p, c = 3 + i % 4, 2 + i % 5
        toks += [1 + (i + j) % 30 for j in range(p)] + [31 + (i + j) % 30 for j in range(c)] + [EOT]
        masks += [0] * p + [1] * (c + 1)
    return np.array(toks, dtype=np.uint16), np.array(masks, dtype=np.uint8)


def test_single_pass_token_weighted_matches_manual():
    torch.manual_seed(0)
    model = PlaygroundLM(n_layers=1, n_heads=2, d_model=16, context_len=12, vocab_size=64)
    tokens, masks = pack(25)
    result = masked_val(model, tokens, masks, context_len=12, batch_size=3, eot_id=EOT, vocab_size=64)

    # Manual reference: every validation window once, NLL summed over supervised targets.
    audit, layout = audit_sft_arrays(tokens, masks, 12, eot_id=EOT, vocab_size=64)
    nll, count = 0.0, 0
    for s, lo, hi in _eval_windows(layout, 12, tokens.size - 13):
        x, y = window_targets(tokens, masks, layout.starts, s, 12, (lo, hi))
        logits, _ = model(torch.from_numpy(x)[None])
        yt = torch.from_numpy(y)
        nll += float(F.cross_entropy(logits[0].float(), yt, ignore_index=IGNORE_INDEX, reduction="sum"))
        count += int((yt != IGNORE_INDEX).sum())
    assert result["single_pass"] is True
    assert result["supervised_tokens"] == count == audit.supervised_tokens - audit.unsupervisable_tokens
    assert result["supervised_tokens"] == result["expected_supervised_tokens"]
    assert result["token_weighted_loss"] == pytest.approx(nll / count, rel=1e-5)
