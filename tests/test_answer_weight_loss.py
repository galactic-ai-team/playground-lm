"""Answer-token loss weight (Stage G, Claude): default path unchanged; weighted path correct. Needs real PyTorch."""

import pytest

torch = pytest.importorskip("torch")
if not hasattr(getattr(torch, "nn", None), "Linear"):
    pytest.skip("needs a real PyTorch install", allow_module_level=True)
import torch.nn.functional as F  # noqa: E402

from playground_lm.model import IGNORE_INDEX, PlaygroundLM, chunked_cross_entropy  # noqa: E402


def batch():
    torch.manual_seed(0)
    logits = torch.randn(3, 10, 50, requires_grad=True)
    y = torch.randint(0, 50, (3, 10))
    y[:, :3] = IGNORE_INDEX
    return logits, y


def test_no_weights_is_bit_identical_to_the_current_loss():
    logits, y = batch()
    assert torch.equal(chunked_cross_entropy(logits, y, 7), chunked_cross_entropy(logits, y, 7, weights=None))
    model = PlaygroundLM(n_layers=1, n_heads=2, d_model=16, context_len=10, vocab_size=50)
    x = torch.randint(0, 50, (3, 10))
    assert torch.equal(model(x, y)[1], model(x, y, loss_weights=None)[1])


def test_all_ones_weights_match_the_plain_mean():
    logits, y = batch()
    plain = chunked_cross_entropy(logits, y, 7)
    ones = chunked_cross_entropy(logits, y, 7, weights=torch.ones_like(y, dtype=torch.float32))
    torch.testing.assert_close(ones, plain)


def test_weighted_loss_is_weighted_mean_over_supervised_targets():
    logits, y = batch()
    w = torch.ones_like(y, dtype=torch.float32)
    w[:, -2:] = 2.0
    got = chunked_cross_entropy(logits, y, 7, weights=w)
    ce = F.cross_entropy(logits.reshape(-1, 50), y.reshape(-1), reduction="none", ignore_index=IGNORE_INDEX)
    keep = (y.reshape(-1) != IGNORE_INDEX).float()
    want = (ce * w.reshape(-1)).sum() / (w.reshape(-1) * keep).sum()
    torch.testing.assert_close(got, want)
    got.backward()
    assert logits.grad is not None and torch.isfinite(logits.grad).all()
