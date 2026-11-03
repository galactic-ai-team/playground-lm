"""ONNX export tests; CPU-only and no pretrained weights."""

import json

import pytest

torch = pytest.importorskip("torch")
pytest.importorskip("onnx")
pytest.importorskip("onnxruntime")

from evals.export_onnx import (
    LogitsOnly,
    artifact_info,
    check_onnx,
    export_bundle,
    export_fp32,
)


class TinyLM(torch.nn.Module):
    """Small causal LM for export-contract testing."""

    def __init__(self, vocab_size=32, dim=16):
        super().__init__()
        self.embedding = torch.nn.Embedding(
            vocab_size, dim
        )
        self.projection = torch.nn.Linear(
            dim, vocab_size
        )
        self.vocab_size = vocab_size

    def forward(self, x, y=None):
        hidden = self.embedding(x)
        logits = self.projection(hidden)
        return logits, None


def test_logits_only_wrapper():
    model = TinyLM()
    wrapper = LogitsOnly(model)

    x = torch.tensor([[1, 2, 3]], dtype=torch.long)

    output = wrapper(x)
    expected, _ = model(x)

    assert output.shape == (1, 3, 32)
    torch.testing.assert_close(output, expected)


def test_fp32_export_dynamic_sequences(tmp_path):
    torch.manual_seed(123)
    model = TinyLM().eval()

    path = tmp_path / "tiny.onnx"

    export_fp32(
        model,
        path,
        sample_length=4,
    )

    assert path.is_file()
    assert path.stat().st_size > 0

    result = check_onnx(
        path,
        model,
        lengths=(1, 4, 7),
        vocab_size=32,
    )

    assert result["passed"] is True
    assert result["sequence_lengths"] == [1, 4, 7]
    assert result["pytorch_compared"] is True


def test_bundle_and_manifest(tmp_path):
    torch.manual_seed(42)
    model = TinyLM().eval()

    output_dir = tmp_path / "export"

    manifest = export_bundle(
        model,
        output_dir,
        context_len=16,
        vocab_size=32,
        sample_length=4,
        quantize=False,
    )

    assert (
        output_dir / "model.onnx"
    ).is_file()

    assert (
        output_dir / "model.int8.onnx"
    ).exists() is False

    assert manifest["runtime"]["opset"] == 17
    assert manifest["runtime"]["input_name"] == "input_ids"
    assert manifest["runtime"]["context_length"] == 16

    info = artifact_info(
        output_dir / "model.onnx"
    )

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
        vocab_size=32,
        sample_length=4,
        quantize=True,
    )

    artifact = tmp_path / "quantized" / "model.int8.onnx"

    assert artifact.is_file()
    assert artifact.stat().st_size > 0
    assert manifest["artifacts"]["int8"]["validation"]["passed"]
    assert (
        manifest["artifacts"]["int8"]["browser_verified"]
        is False
    )


def test_invalid_sample_length(tmp_path):
    with pytest.raises(
        ValueError,
        match="Invalid sample_length",
    ):
        export_bundle(
            TinyLM(),
            tmp_path,
            context_len=4,
            sample_length=8,
            quantize=False,
        )
