# kaggle/stage_e_train/test_stage_e_launcher.py
# Tests for the Stage E launcher. Contract: paired {train,val}.{bin,masks.bin},
# PLM_STAGE="E", seed 31101, atomic resume, dry-run path.

import importlib.util
import json
import os
from pathlib import Path
from unittest.mock import MagicMock

import numpy as np
import pytest

_LAUNCHER = Path(__file__).resolve().parents[1] / "kaggle" / "stage_e_train" / "stage_e_train.py"
_spec = importlib.util.spec_from_file_location("stage_e_train", _LAUNCHER)
stage_e = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(stage_e)


# --- Fixtures ------------------------------------------------------------

def _write_pair(dir_: Path, n_tokens: int = 4096, n_supervised: int = 1024) -> None:
    """Write a minimal Stage E pair: tokens uint16 + masks uint8 of equal length."""
    rng = np.random.default_rng(0)
    tokens = rng.integers(0, 50257, size=n_tokens, dtype=np.uint16)
    masks = np.zeros(n_tokens, dtype=np.uint8)
    masks[-n_supervised:] = 1
    dir_.mkdir(parents=True, exist_ok=True)
    tokens.tofile(dir_ / "train.bin")
    masks.tofile(dir_ / "train.masks.bin")
    tokens[: n_tokens // 2].tofile(dir_ / "val.bin")
    masks[: n_tokens // 2].tofile(dir_ / "val.masks.bin")


# --- Config / stage identity --------------------------------------------

def test_stage_is_e():
    """PLM_STAGE must be E, not D (that was the draft bug)."""
    # main() builds the env inline; assert on the literal in the source
    src = Path(stage_e.__file__).read_text()
    assert '"PLM_STAGE": "E"' in src
    assert '"PLM_STAGE": "D"' not in src


def test_seed_is_stage_c_31101():
    assert stage_e.STAGE_C_FINAL_STEP == 31101
    assert stage_e.STAGE_C_FINAL_STEP != 31104


# --- Data contract (paired .bin + .masks.bin) ---------------------------

def test_stage_data_copies_paired_files(tmp_path, monkeypatch):
    src = tmp_path / "in"; _write_pair(src)
    data_dir = tmp_path / "data"
    monkeypatch.setattr(stage_e, "DATA_DIR", data_dir)
    monkeypatch.setenv("PLM_FINETUNE_BIN", str(src / "train.bin"))
    monkeypatch.setenv("PLM_FINETUNE_VAL_BIN", str(src / "val.bin"))

    n = stage_e.stage_data()
    assert n == 4096
    for name in ("train.bin", "train.masks.bin", "val.bin", "val.masks.bin"):
        assert (data_dir / name).is_file(), name


def test_stage_data_rejects_mask_length_mismatch(tmp_path, monkeypatch):
    src = tmp_path / "in"; _write_pair(src)
    # Truncate train mask by one byte to break alignment.
    (src / "train.masks.bin").write_bytes((src / "train.masks.bin").read_bytes()[:-1])
    monkeypatch.setattr(stage_e, "DATA_DIR", tmp_path / "data")
    monkeypatch.setenv("PLM_FINETUNE_BIN", str(src / "train.bin"))
    monkeypatch.setenv("PLM_FINETUNE_VAL_BIN", str(src / "val.bin"))
    with pytest.raises(ValueError, match="train mask/token mismatch"):
        stage_e.stage_data()


def test_resolve_files_finds_flat_layout(tmp_path, monkeypatch):
    root = tmp_path / "plm-stage-e-data"; _write_pair(root)
    monkeypatch.setattr(stage_e, "INPUT", tmp_path)
    monkeypatch.delenv("PLM_FINETUNE_BIN", raising=False)
    monkeypatch.delenv("PLM_FINETUNE_VAL_BIN", raising=False)
    tb, tm, vb, vm = stage_e.resolve_finetune_files()
    assert tb.name == "train.bin" and tm.name == "train.masks.bin"
    assert vb.name == "val.bin" and vm.name == "val.masks.bin"


# --- Disk / quota preflight ---------------------------------------------

def test_disk_preflight_refuses_when_low(tmp_path, monkeypatch):
    # Lumi glue: sandbox has no /kaggle/working; point WORKING at tmp.
    monkeypatch.setattr(stage_e, "WORKING", tmp_path)
    monkeypatch.setattr(stage_e, "MIN_DISK_GB", 999999.0)
    assert stage_e.preflight_disk() is False


def test_disk_preflight_passes_with_default(tmp_path, monkeypatch):
    monkeypatch.setattr(stage_e, "WORKING", tmp_path)
    monkeypatch.setattr(stage_e, "MIN_DISK_GB", 0.001)
    assert stage_e.preflight_disk() is True


def test_quota_guard_off_by_default(monkeypatch):
    monkeypatch.setattr(stage_e, "QUOTA_GUARD", False)
    assert stage_e.preflight_quota() is True


def test_quota_guard_blocks_when_estimate_too_high(monkeypatch):
    monkeypatch.setattr(stage_e, "QUOTA_GUARD", True)
    monkeypatch.setattr(stage_e, "EST_SESSION_H", 99.0)
    monkeypatch.setattr(stage_e, "QUOTA_LIMIT_H", 30.0)
    assert stage_e.preflight_quota() is False


# --- Atomic resume safety (the critical fix) ----------------------------

def test_seed_from_stage_c_does_not_destroy_runs_on_missing_files(tmp_path, monkeypatch):
    """If Stage C mount lacks the step files, existing RUNS_DIR is preserved."""
    # Existing runs dir with a sentinel file.
    runs = tmp_path / "runs"; runs.mkdir()
    (runs / "manifest.json").write_text('{"latest_step": 999}')
    (runs / "sentinel.txt").write_text("keep me")
    monkeypatch.setattr(stage_e, "RUNS_DIR", runs)

    # Empty Stage C mount so needed files are missing.
    c_mount = tmp_path / "c" / "runs" / "checkpoints"
    c_mount.mkdir(parents=True)
    monkeypatch.setattr(stage_e, "INPUT", tmp_path)
    monkeypatch.setattr(
        stage_e, "find_mount", lambda slug: tmp_path / "c" if slug == stage_e.STAGE_C_SLUG else None
    )

    with pytest.raises(FileNotFoundError, match="Stage C final checkpoint incomplete"):
        stage_e.seed_from_stage_c()

    # RUNS_DIR untouched.
    assert (runs / "manifest.json").read_text() == '{"latest_step": 999}'
    assert (runs / "sentinel.txt").read_text() == "keep me"
    # No staging leftovers.
    assert not (tmp_path / "runs.tmp").exists()
    assert not (tmp_path / "runs.old").exists()


def test_seed_from_stage_c_installs_atomically(tmp_path, monkeypatch):
    """Happy path: seed stages to .tmp, then atomically installs."""
    runs = tmp_path / "runs"  # does not exist yet
    monkeypatch.setattr(stage_e, "RUNS_DIR", runs)

    c_root = tmp_path / "c"
    ckpt = c_root / "runs" / "checkpoints"
    ckpt.mkdir(parents=True)
    stem = f"step_{stage_e.STAGE_C_FINAL_STEP:08d}"
    (ckpt / f"{stem}.safetensors").write_bytes(b"W")
    (ckpt / f"{stem}_state.pt").write_bytes(b"S")
    (ckpt / f"{stem}_model_config.json").write_text("{}")
    (c_root / "runs" / "manifest.json").write_text(json.dumps({
        "checkpoints": [{"step": stage_e.STAGE_C_FINAL_STEP, "tokens_seen": 2_857_393_718,
                         "wall_time_s": 1.0, "is_milestone": True,
                         "created_at": 0, "path": f"checkpoints/{stem}.safetensors"}],
        "config": {},
    }))
    monkeypatch.setattr(
        stage_e, "find_mount", lambda slug: c_root if slug == stage_e.STAGE_C_SLUG else None
    )

    stage_e.seed_from_stage_c()
    assert (runs / "manifest.json").is_file()
    m = json.loads((runs / "manifest.json").read_text())
    assert m["latest_step"] == 31101
    assert (runs / "checkpoints" / f"{stem}.safetensors").read_bytes() == b"W"
    assert (runs / "checkpoints" / f"{stem}_state.pt").read_bytes() == b"S"


def test_resume_previous_stage_e_preserves_runs_on_bad_copy(tmp_path, monkeypatch):
    """If the previous mount's copy fails mid-way, existing RUNS_DIR is preserved."""
    runs = tmp_path / "runs"; runs.mkdir()
    (runs / "sentinel.txt").write_text("keep")
    monkeypatch.setattr(stage_e, "RUNS_DIR", runs)

    # Previous mount exists but has no manifest -> resume returns False without touching runs.
    prev = tmp_path / "prev" / "runs"; prev.mkdir(parents=True)
    (prev / "not_manifest.txt").write_text("junk")
    monkeypatch.setattr(
        stage_e, "find_mount", lambda slug: tmp_path / "prev" if slug == stage_e.STAGE_E_SLUG else None
    )

    assert stage_e.resume_previous_stage_e() is False
    assert (runs / "sentinel.txt").read_text() == "keep"
    assert not (tmp_path / "runs.tmp").exists()


def test_resume_previous_stage_e_atomic_install(tmp_path, monkeypatch):
    runs = tmp_path / "runs"
    monkeypatch.setattr(stage_e, "RUNS_DIR", runs)

    prev = tmp_path / "prev" / "runs"; prev.mkdir(parents=True)
    (prev / "manifest.json").write_text(json.dumps({"latest_step": 31205}))
    monkeypatch.setattr(
        stage_e, "find_mount", lambda slug: tmp_path / "prev" if slug == stage_e.STAGE_E_SLUG else None
    )

    assert stage_e.resume_previous_stage_e() is True
    m = json.loads((runs / "manifest.json").read_text())
    assert m["latest_step"] == 31205


# --- Dry-run path -------------------------------------------------------

def test_dry_run_sets_cpu_and_one_step(tmp_path, monkeypatch):
    """PLM_DRY_RUN forces CUDA_VISIBLE_DEVICES='' and one update."""
    captured = {}

    def fake_run(cmd, env=None, **kw):
        captured["env"] = env
        r = MagicMock(); r.returncode = 0; return r

    # Minimal valid layout so main() reaches the launch call.
    data_src = tmp_path / "in"; _write_pair(data_src)
    monkeypatch.setattr(stage_e, "DATA_DIR", tmp_path / "data")
    monkeypatch.setenv("PLM_FINETUNE_BIN", str(data_src / "train.bin"))
    monkeypatch.setenv("PLM_FINETUNE_VAL_BIN", str(data_src / "val.bin"))

    monkeypatch.setattr(stage_e, "preflight_disk", lambda: True)
    monkeypatch.setattr(stage_e, "preflight_quota", lambda: True)
    monkeypatch.setattr(stage_e, "resume_previous_stage_e", lambda: True)
    monkeypatch.setattr(stage_e, "seed_from_stage_c", lambda: None)
    monkeypatch.setattr(stage_e, "clone_repo", lambda: None)
    monkeypatch.setattr(stage_e, "REPO_DIR", tmp_path)
    (tmp_path / "notebooks").mkdir()
    (tmp_path / "notebooks" / "playground_lm_train.py").write_text("# stub")
    (tmp_path / "src" / "playground_lm" / "train").mkdir(parents=True)
    (tmp_path / "src" / "playground_lm" / "train" / "__init__.py").write_text("")
    (tmp_path / "src" / "playground_lm" / "train" / "stage_config.py").write_text(
        "def finetune_max_steps(n, epochs=2): return 31101 + max(1, n // 4096)\n"
    )
    monkeypatch.setattr(stage_e, "DRY_RUN", True)
    monkeypatch.setattr(stage_e.subprocess, "run", fake_run)

    rc = stage_e.main()
    assert rc == 0
    assert captured["env"]["CUDA_VISIBLE_DEVICES"] == ""
    assert captured["env"]["PLM_MAX_STEPS"] == str(stage_e.STAGE_C_FINAL_STEP + 1)
    assert captured["env"]["PLM_STAGE"] == "E"

def test_atomic_preserves_old_backup_when_replacing_runs(tmp_path, monkeypatch):
    """Lumi glue: prior RUNS_DIR becomes runs.old and is not deleted."""
    runs = tmp_path / "runs"
    runs.mkdir()
    (runs / "manifest.json").write_text('{"latest_step": 1}')
    (runs / "keep.txt").write_text("prior")
    # Pre-existing .old must be archived, not deleted.
    old = tmp_path / "runs.old"
    old.mkdir()
    (old / "ancient.txt").write_text("ancient")
    monkeypatch.setattr(stage_e, "RUNS_DIR", runs)

    prev = tmp_path / "prev" / "runs"
    prev.mkdir(parents=True)
    (prev / "manifest.json").write_text('{"latest_step": 31205}')
    monkeypatch.setattr(
        stage_e, "find_mount",
        lambda slug: tmp_path / "prev" if slug == stage_e.STAGE_E_SLUG else None,
    )
    assert stage_e.resume_previous_stage_e() is True
    assert (runs / "manifest.json").is_file()
    assert json.loads((runs / "manifest.json").read_text())["latest_step"] == 31205
    assert (tmp_path / "runs.old" / "keep.txt").read_text() == "prior"
    archives = list(tmp_path.glob("runs.old.bak.*"))
    assert archives, "expected archived previous .old"
    assert (archives[0] / "ancient.txt").read_text() == "ancient"
