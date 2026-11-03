import math

import pytest
import torch
from safetensors.torch import load_file, save_file
from torch.nn import functional as F

from playground_lm.model import PlaygroundLM


TINY = dict(n_layers=2, n_heads=2, d_model=32, context_len=16, vocab_size=100)
CONFIG = dict(
    vocab_size=50257, n_layers=12, n_heads=12, d_model=768,
    context_len=1024, batch_size=12, grad_accum_steps=8, lr=6e-4,
    warmup_steps=2000, weight_decay=0.1, grad_clip=1.0,
)


@pytest.fixture(autouse=True)
def cpu_setup():
    previous_threads = torch.get_num_threads()
    torch.set_num_threads(1)
    with torch.random.fork_rng(devices=[]):
        torch.manual_seed(1234)
        yield
    torch.set_num_threads(previous_threads)


@pytest.fixture
def model():
    return PlaygroundLM(**TINY)


def test_shapes_and_loss(model):
    x = torch.randint(100, (2, 16))
    y = torch.randint(100, (2, 16))
    logits, loss = model(x)
    assert logits.shape == (2, 16, 100)
    assert loss is None
    logits, loss = model(x, y)
    assert loss.ndim == 0
    assert abs(loss.item() - math.log(100)) < 0.1
    torch.testing.assert_close(loss, F.cross_entropy(logits.reshape(-1, 100), y.flatten()))
    loss.backward()
    assert all(p.grad is not None and torch.isfinite(p.grad).all()
               for p in model.parameters())


def test_causality(model):
    model.eval()
    x = torch.randint(100, (2, 16))
    changed = x.clone()
    changed[:, 8:] = (changed[:, 8:] + 1) % 100
    with torch.no_grad():
        original, _ = model(x)
        modified, _ = model(changed)
    torch.testing.assert_close(original[:, :8], modified[:, :8], rtol=0, atol=0)
    assert not torch.equal(original[:, 8:], modified[:, 8:])


def test_weight_tying(model):
    assert model.lm_head.weight is model.token_embedding.weight
    assert sum(p is model.lm_head.weight for p in model.parameters()) == 1


@pytest.mark.parametrize("assign", [False, True])
@pytest.mark.parametrize("nested", [False, True])
def test_safetensors_round_trip(model, tmp_path, assign, nested):
    original = torch.nn.ModuleDict({"model": model}) if nested else model
    path = str(tmp_path / "model.safetensors")
    state = original.state_dict()
    prefix = "model." if nested else ""
    assert prefix + "lm_head.weight" not in state
    save_file(state, path)
    restored = PlaygroundLM(**TINY)
    target = torch.nn.ModuleDict({"model": restored}) if nested else restored
    loaded = load_file(path)
    result = target.load_state_dict(loaded, strict=True, assign=assign)
    assert not result.missing_keys and not result.unexpected_keys
    assert prefix + "lm_head.weight" not in loaded
    assert restored.lm_head.weight is restored.token_embedding.weight
    x = torch.randint(100, (2, 12))
    torch.testing.assert_close(model(x)[0], restored(x)[0], rtol=0, atol=0)
    save_file(target.state_dict(), str(tmp_path / "resaved.safetensors"))


def test_strict_loading_rejects_missing_parameter(model):
    state = model.state_dict()
    del state["blocks.0.attn.qkv.weight"]
    with pytest.raises(RuntimeError, match="Missing key"):
        PlaygroundLM(**TINY).load_state_dict(state, strict=True)


def test_default_parameter_count():
    # Meta tensors check the real architecture without allocating 500 MB.
    with torch.device("meta"):
        model = PlaygroundLM(**CONFIG)
        defaults = PlaygroundLM()
    assert 120_000_000 < model.num_params(False) < 130_000_000
    assert model.num_params(False) == 124_439_808
    assert defaults.num_params(False) == model.num_params(False)
    assert model.num_params() == 124_439_808 - 1024 * 768


@pytest.mark.parametrize("top_k", [None, 1, 10, 200])
def test_generate(model, top_k):
    model.train()
    prompt = torch.randint(100, (2, 14))
    result = model.generate(prompt, 5, temperature=0.8, top_k=top_k)
    assert result.shape == (2, 19)
    assert result.dtype == torch.int64
    assert model.training
    assert ((result >= 0) & (result < 100)).all()
    torch.testing.assert_close(result[:, :14], prompt)
    model.eval()
    assert torch.equal(model.generate(prompt, 0), prompt)
    assert not model.training


def test_autocast(model):
    x = torch.randint(100, (2, 8))
    y = torch.randint(100, (2, 8))
    with torch.autocast("cpu", dtype=torch.bfloat16):
        logits, loss = model(x, y)
    assert logits.dtype == torch.bfloat16
    assert loss.dtype == torch.float32 and torch.isfinite(loss)
    loss.backward()
    assert all(p.grad is not None and torch.isfinite(p.grad).all()
               for p in model.parameters())


def test_compile(model):
    # AOT eager exercises full graph capture and compiled autograd on CPU.
    compiled = torch.compile(model, backend="aot_eager", fullgraph=True)
    x = torch.randint(100, (2, 8))
    y = torch.randint(100, (2, 8))
    expected_logits, expected_loss = model(x, y)
    expected_loss.backward()
    expected_grads = [p.grad.clone() for p in model.parameters()]
    model.zero_grad(set_to_none=True)
    logits, loss = compiled(x, y)
    torch.testing.assert_close(logits, expected_logits)
    torch.testing.assert_close(loss, expected_loss)
    loss.backward()
    for parameter, expected_grad in zip(model.parameters(), expected_grads):
        torch.testing.assert_close(parameter.grad, expected_grad)
