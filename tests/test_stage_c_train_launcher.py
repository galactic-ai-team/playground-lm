"""Stage C training launcher: /tmp concat dir + train.bin size gate (Lumi)."""

from __future__ import annotations

import importlib.util
import sys
from pathlib import Path

import pytest

LAUNCHER = (
    Path(__file__).resolve().parent.parent
    / "kaggle"
    / "stage_c_train"
    / "stage_c_train.py"
)


def _load_launcher():
    spec = importlib.util.spec_from_file_location("stage_c_train_launcher", LAUNCHER)
    assert spec is not None and spec.loader is not None
    mod = importlib.util.module_from_spec(spec)
    # Avoid executing main on import — module only defines helpers/constants.
    sys.modules["stage_c_train_launcher"] = mod
    spec.loader.exec_module(mod)
    return mod


@pytest.fixture(scope="module")
def launcher():
    return _load_launcher()


def test_concat_dir_defaults_under_tmp(launcher):
    assert launcher.CONCAT_DIR == Path("/tmp/plm_stage_c")
    assert launcher.RUNS_DIR == Path("/kaggle/working/runs")
    assert str(launcher.CONCAT_DIR).startswith("/tmp")
    assert launcher.ENV_OVERRIDES["PLM_DATA_DIR"] == str(launcher.CONCAT_DIR)
    assert launcher.ENV_OVERRIDES["PLM_RUN_DIR"] == str(launcher.RUNS_DIR)


def test_expected_train_size_constants(launcher):
    assert launcher.EXPECTED_TRAIN_TOKENS == 2_857_393_718
    assert launcher.EXPECTED_TRAIN_BYTES == launcher.EXPECTED_TRAIN_TOKENS * 2


def test_verify_train_bin_aborts_on_size_mismatch(launcher, tmp_path, monkeypatch, capsys):
    monkeypatch.setattr(launcher, "log", lambda msg: print(msg))
    bad = tmp_path / "train.bin"
    bad.write_bytes(b"\x00\x01" * 100)  # 200 bytes != expected
    with pytest.raises(RuntimeError, match="size mismatch"):
        launcher.verify_train_bin(bad)


def test_verify_train_bin_accepts_expected_size_and_records_sha(
    launcher, tmp_path, monkeypatch
):
    monkeypatch.setattr(launcher, "log", lambda msg: None)
    monkeypatch.delenv("PLM_TRAIN_SHA256", raising=False)
    # Don't write 5.7 GB — temporarily shrink the expected size for the test.
    monkeypatch.setattr(launcher, "EXPECTED_TRAIN_TOKENS", 4)
    monkeypatch.setattr(launcher, "EXPECTED_TRAIN_BYTES", 8)
    good = tmp_path / "train.bin"
    good.write_bytes(b"abcdefgh")  # 8 bytes = 4 uint16 tokens
    digest = launcher.verify_train_bin(good)
    assert isinstance(digest, str) and len(digest) == 64

    monkeypatch.setenv("PLM_TRAIN_SHA256", "deadbeef")
    with pytest.raises(RuntimeError, match="sha256 mismatch"):
        launcher.verify_train_bin(good)

    monkeypatch.setenv("PLM_TRAIN_SHA256", digest)
    assert launcher.verify_train_bin(good) == digest
