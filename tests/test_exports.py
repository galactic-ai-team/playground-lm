"""Offline tokenizer cache and safetensors export tests."""

import hashlib
import json

import pytest

from playground_lm import tokenizer as tokenizer_module


def test_offline_cache_missing(
    tmp_path, monkeypatch
) -> None:
    monkeypatch.setenv("PLM_CACHE_DIR", str(tmp_path))
    monkeypatch.delenv("TIKTOKEN_CACHE_DIR", raising=False)

    with pytest.raises(FileNotFoundError, match="Missing offline"):
        tokenizer_module.configure_cache()


def test_offline_cache_valid(
    tmp_path, monkeypatch
) -> None:
    url = "https://example.test/asset"
    payload = b"test-tokenizer-asset"
    expected = hashlib.sha256(payload).hexdigest()

    monkeypatch.setattr(
        tokenizer_module,
        "ASSETS",
        {url: expected},
    )

    filename = tokenizer_module.cache_filename(url)
    (tmp_path / filename).write_bytes(payload)

    monkeypatch.setenv("PLM_CACHE_DIR", str(tmp_path))
    monkeypatch.delenv("TIKTOKEN_CACHE_DIR", raising=False)

    result = tokenizer_module.configure_cache()

    assert result == tmp_path.resolve()
    assert (
        tokenizer_module.os.environ["TIKTOKEN_CACHE_DIR"]
        == str(tmp_path.resolve())
    )


def test_offline_cache_corrupt(
    tmp_path, monkeypatch
) -> None:
    url = "https://example.test/asset"

    monkeypatch.setattr(
        tokenizer_module,
        "ASSETS",
        {url: hashlib.sha256(b"correct").hexdigest()},
    )

    (tmp_path / tokenizer_module.cache_filename(url)).write_bytes(
        b"incorrect"
    )

    monkeypatch.setenv("PLM_CACHE_DIR", str(tmp_path))

    with pytest.raises(ValueError, match="Corrupt"):
        tokenizer_module.configure_cache()


def test_safetensors_roundtrip(tmp_path) -> None:
    torch = pytest.importorskip("torch")
    safetensors = pytest.importorskip("safetensors.torch")

    from playground_lm.export_checkpoint import export_checkpoint

    source = tmp_path / "checkpoint.pt"
    output = tmp_path / "export"

    original = {
        "embedding.weight": torch.arange(
            24, dtype=torch.float32
        ).reshape(6, 4),
        "projection.weight": torch.ones(4, 4),
    }

    torch.save(
        {
            "model": original,
            "config": {
                "n_layer": 12,
                "n_head": 12,
                "n_embd": 768,
            },
            "step": 100,
        },
        source,
    )

    manifest = export_checkpoint(source, output)

    restored = safetensors.load_file(
        str(output / "model.safetensors")
    )

    assert set(restored) == set(original)

    for name, tensor in original.items():
        assert torch.equal(restored[name], tensor)

    assert manifest["tensor_count"] == 2
    assert manifest["stored_tensor_elements"] == 40
    assert manifest["tokenizer"]["vocab_size"] == 50_257
    assert manifest["model_config"]["n_layer"] == 12

    saved = json.loads(
        (output / "manifest.json").read_text()
    )
    assert saved == manifest


def test_export_rejects_invalid_checkpoint(tmp_path) -> None:
    torch = pytest.importorskip("torch")
    pytest.importorskip("safetensors")

    from playground_lm.export_checkpoint import export_checkpoint

    source = tmp_path / "invalid.pt"
    torch.save({"optimizer": {"lr": 0.001}}, source)

    with pytest.raises(ValueError, match="No tensor state dict"):
        export_checkpoint(source, tmp_path / "out")


def test_prompt_file_schema() -> None:
    from pathlib import Path

    prompts_path = (
        Path(__file__).resolve().parents[1]
        / "evals"
        / "prompts.json"
    )
    data = json.loads(prompts_path.read_text(encoding="utf-8"))

    assert data["tokenizer"] == "gpt2"
    assert len(data["held_out_probes"]) >= 8
    assert len(data["demo_riddles"]) >= 6

    ids = [
        entry["id"]
        for section in ("held_out_probes", "demo_riddles")
        for entry in data[section]
    ]
    assert len(ids) == len(set(ids))

    for section in ("held_out_probes", "demo_riddles"):
        for entry in data[section]:
            assert entry["prompt"].strip()
            assert len(entry["prompt"]) < 1024

    for riddle in data["demo_riddles"]:
        assert riddle["answer"].strip()
        assert riddle["answer"].lower() not in riddle["prompt"].lower()
