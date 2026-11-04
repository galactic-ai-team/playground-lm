
"""ONNX export tests, including real PlaygroundLM architecture.

CPU-only. No pretrained weights or external datasets.
"""

import json

import numpy as np
import pytest

torch = pytest.importorskip("torch")
pytest.importorskip("onnx")
pytest.importorskip("onnxruntime")

from evals.export_onnx import (
    LogitsOnly,
    artifact_info,
    check_onnx,
    effective_vocab_size,
    export_bundle,
    export_fp32,
    valid_example_ids,
)


class TinyLM(torch.nn.Module):
    def __init__(self, vocab_size=32, dim=16):
        super().__init__()
        self.embedding = torch.nn.Embedding(vocab_size, dim)
        self.projection = torch.nn.Linear(dim, vocab_size)
        self.vocab_size = vocab_size

    def forward(self, x, y=None):
        logits = self.projection(self.embedding(x))
        return logits, None


def test_logits_only_wrapper():
    model = TinyLM()
    x = torch.tensor([[1, 2, 3]])

    actual = LogitsOnly(model)(x)
    expected, _ = model(x)

    assert actual.shape == (1, 3, 32)
    torch.testing.assert_close(actual, expected)


def test_input_ids_respect_vocabulary():
    ids = valid_example_ids(length=100, vocab_size=32)
    assert ids.shape == (1, 100)
    assert ids.min() >= 0
    assert ids.max() < 32
    assert ids[0, 32] == 0


def test_vocab_mismatch_rejected():
    model = TinyLM(vocab_size=32)

    assert effective_vocab_size(model) == 32

    with pytest.raises(ValueError, match="Vocabulary mismatch"):
        effective_vocab_size(model, 50257)


def test_fp32_export_dynamic_sequences(tmp_path):
    torch.manual_seed(123)
    model = TinyLM().eval()
    path = tmp_path / "tiny.onnx"

    export_fp32(model, path, sample_length=4)

    assert path.is_file()
    result = check_onnx(
        path,
        model,
        lengths=(1, 4, 7, 40),
        vocab_size=32,
    )
    assert result["passed"]
    assert result["sequence_lengths"] == [1, 4, 7, 40]
    assert result["pytorch_compared"]


def test_bundle_and_manifest(tmp_path):
    torch.manual_seed(42)
    model = TinyLM().eval()
    output_dir = tmp_path / "export"

    manifest = export_bundle(
        model,
        output_dir,
        context_len=16,
        sample_length=4,
        quantize=False,
    )

    assert (output_dir / "model.onnx").is_file()
    assert not (output_dir / "model.int8.onnx").exists()
    assert manifest["runtime"]["opset"] == 17
    assert manifest["runtime"]["input_name"] == "input_ids"
    assert manifest["runtime"]["context_length"] == 16
    assert manifest["tokenizer"]["vocab_size"] == 32

    info = artifact_info(output_dir / "model.onnx")
    assert len(info["sha256"]) == 64
    assert info["bytes"] > 0

    saved = json.loads(
        (output_dir / "manifest.json").read_text()
    )
    assert saved == manifest


def test_int8_export_smoke(tmp_path):
    torch.manual_seed(7)
    model = TinyLM().eval()

    manifest = export_bundle(
        model,
        tmp_path / "quantized",
        context_len=16,
        sample_length=4,
        quantize=True,
    )

    artifact = tmp_path / "quantized" / "model.int8.onnx"

    assert artifact.is_file()
    assert artifact.stat().st_size > 0
    assert manifest["artifacts"]["int8"]["validation"]["passed"]
    assert manifest["artifacts"]["int8"]["browser_verified"] is False


def test_invalid_sample_length(tmp_path):
    with pytest.raises(ValueError, match="Invalid sample_length"):
        export_bundle(
            TinyLM(),
            tmp_path,
            context_len=4,
            sample_length=8,
            quantize=False,
        )


def test_real_playgroundlm_sdpa_dynamic_export(tmp_path):
    """Exercise Codex's actual attention, embeddings and weight tying."""
    from playground_lm.model import PlaygroundLM

    torch.manual_seed(2026)
    model = PlaygroundLM(
        vocab_size=32,
        n_layers=1,
        n_heads=2,
        d_model=16,
        context_len=16,
        dropout=0.0,
    ).eval()

    path = tmp_path / "playground_sdpa.onnx"

    export_fp32(
        model,
        path,
        sample_length=4,
    )

    result = check_onnx(
        path,
        model,
        lengths=(1, 4, 7, 16),
        vocab_size=32,
    )

    assert result["passed"]
    assert result["max_abs_error"] is not None
    assert np.isfinite(result["max_abs_error"])


def test_real_playgroundlm_export_rejects_long_context(tmp_path):
    from playground_lm.model import PlaygroundLM

    model = PlaygroundLM(
        vocab_size=32,
        n_layers=1,
        n_heads=2,
        d_model=16,
        context_len=8,
    )

    with pytest.raises(ValueError, match="model context_len"):
        export_bundle(
            model,
            tmp_path,
            context_len=16,
            sample_length=4,
            quantize=False,
        )


def _tiny_playground(d_model=32):
    from playground_lm.model import PlaygroundLM

    torch.manual_seed(11)
    return PlaygroundLM(
        vocab_size=64,
        n_layers=1,
        n_heads=2,
        d_model=d_model,
        context_len=16,
        dropout=0.0,
    ).eval()


def test_int8_keeps_attention_matmuls_fp32(tmp_path):
    import onnx

    from evals.export_onnx import weight_matmul_nodes

    model = _tiny_playground()
    manifest = export_bundle(
        model, tmp_path, context_len=16, sample_length=4, quantize=True
    )
    weight, act_act = weight_matmul_nodes(tmp_path / "model.onnx")
    assert len(act_act) == 2  # Q@K^T and P@V in the single block
    assert weight

    graph = onnx.load(str(tmp_path / "model.int8.onnx")).graph
    remaining = {n.name for n in graph.node if n.op_type == "MatMul"}
    assert set(act_act) <= remaining
    assert any(n.op_type == "MatMulInteger" for n in graph.node)
    # Embedding Gather is quantized too: no FP32 [vocab, d_model] table.
    assert not any(
        list(i.dims) == [64, 32] and i.data_type == onnx.TensorProto.FLOAT
        for i in graph.initializer
    )

    art = manifest["artifacts"]["int8"]
    assert art["quantization"] == "dynamic_qint8_per_channel"
    assert art["excluded_activation_matmuls"] == 2
    assert 0.0 <= art["quality"]["top1_agreement"] <= 1.0
    assert np.isfinite(art["quality"]["mean_kl_nats"])


def test_q4_matmul_nbits_mode(tmp_path):
    import onnx

    pytest.importorskip("onnxruntime.quantization.matmul_nbits_quantizer")
    model = _tiny_playground()
    manifest = export_bundle(
        model,
        tmp_path,
        context_len=16,
        sample_length=4,
        quantize=True,
        quant_mode="q4",
    )
    graph = onnx.load(str(tmp_path / "model.int8.onnx")).graph
    assert any(n.op_type == "MatMulNBits" for n in graph.node)
    art = manifest["artifacts"]["int8"]
    assert art["validation"]["passed"]
    assert art["quantization"].startswith("matmul_nbits_4bit")
    assert not list(tmp_path.glob("*.tmp"))


def test_unknown_quant_mode(tmp_path):
    from evals.export_onnx import quantize_model

    with pytest.raises(ValueError, match="mode must be"):
        quantize_model(tmp_path / "a.onnx", tmp_path / "b.onnx", mode="fp8")
