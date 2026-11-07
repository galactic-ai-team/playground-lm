"""Stage D training launcher: resolve flat or nested finetune bins (Lumi)."""

from __future__ import annotations

import importlib.util
import sys
from pathlib import Path

import pytest

LAUNCHER = (
    Path(__file__).resolve().parent.parent
    / "kaggle"
    / "stage_d_train"
    / "stage_d_train.py"
)


def _load_launcher():
    spec = importlib.util.spec_from_file_location("stage_d_train_launcher", LAUNCHER)
    assert spec is not None and spec.loader is not None
    mod = importlib.util.module_from_spec(spec)
    sys.modules["stage_d_train_launcher"] = mod
    spec.loader.exec_module(mod)
    return mod


@pytest.fixture(scope="module")
def launcher():
    return _load_launcher()


def _touch_bin(path: Path, n_tokens: int = 2048) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_bytes(b"\x00\x01" * n_tokens)
    return path


def test_resolve_finetune_bins_prefers_env(launcher, tmp_path, monkeypatch):
    train = _touch_bin(tmp_path / "custom" / "train.bin")
    val = _touch_bin(tmp_path / "custom" / "val.bin")
    monkeypatch.setenv("PLM_FINETUNE_BIN", str(train))
    monkeypatch.setenv("PLM_FINETUNE_VAL_BIN", str(val))
    got_train, got_val = launcher.resolve_finetune_bins()
    assert got_train == train
    assert got_val == val


def test_resolve_finetune_bins_flat_mount(launcher, tmp_path, monkeypatch):
    monkeypatch.delenv("PLM_FINETUNE_BIN", raising=False)
    monkeypatch.delenv("PLM_FINETUNE_VAL_BIN", raising=False)
    monkeypatch.setattr(launcher, "INPUT", tmp_path)
    train = _touch_bin(tmp_path / "plm-stage-d-data" / "train.bin")
    _touch_bin(tmp_path / "plm-stage-d-data" / "val.bin")
    got_train, got_val = launcher.resolve_finetune_bins()
    assert got_train == train
    assert got_val == train.with_name("val.bin")


def test_resolve_finetune_bins_nested_finetune_dir(launcher, tmp_path, monkeypatch):
    monkeypatch.delenv("PLM_FINETUNE_BIN", raising=False)
    monkeypatch.delenv("PLM_FINETUNE_VAL_BIN", raising=False)
    monkeypatch.setattr(launcher, "INPUT", tmp_path)
    train = _touch_bin(tmp_path / "plm-stage-d-data" / "finetune" / "train.bin")
    _touch_bin(tmp_path / "plm-stage-d-data" / "finetune" / "val.bin")
    got_train, got_val = launcher.resolve_finetune_bins()
    assert got_train == train
    assert got_val == train.with_name("val.bin")


def test_resolve_finetune_bins_missing_raises(launcher, tmp_path, monkeypatch):
    monkeypatch.delenv("PLM_FINETUNE_BIN", raising=False)
    monkeypatch.delenv("PLM_FINETUNE_VAL_BIN", raising=False)
    monkeypatch.setattr(launcher, "INPUT", tmp_path)
    with pytest.raises(FileNotFoundError, match="plm-stage-d-data"):
        launcher.resolve_finetune_bins()
