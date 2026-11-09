import numpy as np
import pytest

torch = pytest.importorskip("torch")
from torch.nn import functional as F

from playground_lm.model import IGNORE_INDEX, PlaygroundLM, chunked_cross_entropy
from playground_lm.train.sft_data import build_sft_dataloader


def test_chunked_loss_ignores_masked_targets():
    torch.manual_seed(0)
    logits = torch.randn(2, 5, 20, requires_grad=True)
    targets = torch.randint(0, 20, (2, 5))
    targets[:, :2] = IGNORE_INDEX
    plain = F.cross_entropy(
        logits.reshape(-1, 20),
        targets.reshape(-1),
        ignore_index=IGNORE_INDEX,
    )
    chunked = chunked_cross_entropy(logits, targets, chunk_tokens=3)
    torch.testing.assert_close(chunked, plain)
    g1 = torch.autograd.grad(plain, logits, retain_graph=True)[0]
    g2 = torch.autograd.grad(chunked, logits)[0]
    torch.testing.assert_close(g1, g2)


def test_all_ignored_batch_returns_zero_connected_loss():
    logits = torch.randn(1, 4, 10, requires_grad=True)
    targets = torch.full((1, 4), IGNORE_INDEX)
    loss = chunked_cross_entropy(logits, targets, chunk_tokens=2)
    assert float(loss.detach()) == 0.0
    loss.backward()
    assert torch.isfinite(logits.grad).all()


def test_sft_dataloader_masks_prompt_targets(tmp_path):
    tokens = np.arange(40, dtype=np.uint16)
    masks = np.zeros(40, dtype=np.uint8)
    masks[10:] = 1
    (tmp_path / "train.bin").write_bytes(tokens.tobytes())
    (tmp_path / "train.masks.bin").write_bytes(masks.tobytes())
    loader = build_sft_dataloader(
        tmp_path / "train.bin",
        tmp_path / "train.masks.bin",
        context_len=8,
        batch_size=2,
        shuffle=False,
        drop_last=True,
    )
    x, y = next(loader)
    assert x.shape == (2, 8)
    assert y.shape == (2, 8)
    # For start=0, predicted tokens are positions 1..8; mask[1..8] are 0 => ignored
    assert (y[0] == IGNORE_INDEX).all()


def test_model_forward_with_ignore_index():
    model = PlaygroundLM(n_layers=1, n_heads=1, d_model=16, context_len=8, vocab_size=50)
    x = torch.randint(0, 50, (2, 8))
    y = torch.randint(0, 50, (2, 8))
    y[:, :3] = IGNORE_INDEX
    logits, loss = model(x, y)
    assert logits.shape == (2, 8, 50)
    assert torch.isfinite(loss)
    loss.backward()
