"""Model growth (Stage G Track 2): depth stack/duplicate/identity and approximate widen."""
import pytest

torch = pytest.importorskip("torch")
if not hasattr(getattr(torch, "nn", None), "Linear"):
    pytest.skip("needs a real PyTorch install", allow_module_level=True)

from playground_lm.grow import grow_depth, grow_width  # noqa: E402
from playground_lm.model import PlaygroundLM  # noqa: E402

V, CTX = 97, 16


def tiny(n_layers=2, n_heads=2, d_model=16, seed=0):
    torch.manual_seed(seed)
    m = PlaygroundLM(vocab_size=V, n_layers=n_layers, n_heads=n_heads, d_model=d_model, context_len=CTX)
    # Perturb away from init so zero biases / unit LNs don't hide bugs.
    with torch.no_grad():
        for p in m.parameters():
            p.add_(torch.randn_like(p) * 0.05)
    return m.eval()


def batch():
    g = torch.Generator().manual_seed(1)
    return torch.randint(0, V, (3, CTX), generator=g)


def load(cfg, sd):
    m = PlaygroundLM(vocab_size=V, context_len=CTX, **cfg)
    m.load_state_dict(sd, strict=True)
    return m.eval()


def test_identity_depth_is_exactly_function_preserving():
    small = tiny()
    big = load(dict(n_layers=4, n_heads=2, d_model=16), grow_depth(small.state_dict(), 2, mode="identity"))
    x = batch()
    with torch.no_grad():
        assert torch.allclose(small(x)[0], big(x)[0], atol=1e-5)


@pytest.mark.parametrize("mode", ["stack", "duplicate"])
def test_stack_and_duplicate_orders_load_and_train(mode):
    small = tiny()
    sd = grow_depth(small.state_dict(), 2, mode=mode)
    order = [0, 1, 0, 1] if mode == "stack" else [0, 0, 1, 1]
    for new_i, old_i in enumerate(order):
        assert torch.equal(sd[f"blocks.{new_i}.attn.qkv.weight"], small.state_dict()[f"blocks.{old_i}.attn.qkv.weight"])
    big = load(dict(n_layers=4, n_heads=2, d_model=16), sd).train()
    x = batch()
    _, loss = big(x, x)
    loss.backward()
    assert torch.isfinite(loss)


def test_depth_does_not_mutate_input_and_rejects_bad_args():
    sd = tiny().state_dict()
    before = {k: v.clone() for k, v in sd.items()}
    grow_depth(sd, 2, mode="identity")
    assert all(torch.equal(before[k], sd[k]) for k in sd)
    with pytest.raises(ValueError):
        grow_depth(sd, 1)
    with pytest.raises(ValueError):
        grow_depth(sd, 2, mode="nope")
    with pytest.raises(ValueError):
        grow_depth({"x": torch.zeros(1)}, 2)


def test_width_768_to_1024_shape_loads_strictly_with_tied_embeddings():
    small = tiny(n_layers=2, n_heads=3, d_model=12)  # head size 4 -> 4 heads of 4 = 16
    sd = grow_width(small.state_dict(), 16, n_heads_old=3, n_heads_new=4)
    big = load(dict(n_layers=2, n_heads=4, d_model=16), sd)
    assert big.lm_head.weight.data_ptr() == big.token_embedding.weight.data_ptr()
    # Old weights sit in the leading block.
    assert torch.equal(sd["token_embedding.weight"][:, :12], small.state_dict()["token_embedding.weight"])


def test_width_is_approximate_and_reported_not_exact():
    small = tiny(n_layers=2, n_heads=3, d_model=12)
    big = load(dict(n_layers=2, n_heads=4, d_model=16),
               grow_width(small.state_dict(), 16, n_heads_old=3, n_heads_new=4))
    x = batch()
    with torch.no_grad():
        a, b = small(x, x)[1], big(x, x)[1]
        la, lb = small(x)[0].flatten(), big(x)[0].flatten()
    corr = torch.corrcoef(torch.stack([la, lb]))[0, 1]
    assert torch.isfinite(b) and abs(float(a - b)) < 1.0   # close, not identical
    assert corr > 0.8
    big.train()
    _, loss = big(x, x)
    loss.backward()
    assert all(torch.isfinite(p.grad).all() for p in big.parameters() if p.grad is not None)


def test_width_rejects_head_size_change_and_shrink():
    sd = tiny(n_layers=1, n_heads=3, d_model=12).state_dict()
    with pytest.raises(ValueError):
        grow_width(sd, 16, n_heads_old=3, n_heads_new=2)
    with pytest.raises(ValueError):
        grow_width(sd, 12, n_heads_old=3, n_heads_new=3)
