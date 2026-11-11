"""Offline tests for the Stage G Kaggle launcher and throughput bench (Lumi glue)."""
import importlib.util
import sys
from pathlib import Path

import numpy as np
import pytest

ROOT = Path(__file__).resolve().parents[1]


def _load(rel, name):
    spec = importlib.util.spec_from_file_location(name, ROOT / rel)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


L = _load("kaggle/stage_g_train/stage_g_train.py", "stage_g_train")
B = _load("kaggle/stage_g_bench/stage_g_bench.py", "stage_g_bench")


def test_training_refused_until_stage_g_suite_frozen():
    with pytest.raises(SystemExit, match="frozen Stage G suite"):
        L.require_frozen_suite({})
    with pytest.raises(SystemExit):
        L.require_frozen_suite({"PLM_STAGE_G_SUITE_SHA256": "partial"})
    L.require_frozen_suite({"PLM_STAGE_G_SUITE_SHA256": "a" * 64})
    L.require_frozen_suite({"PLM_DRY_RUN": "1"})


def test_defaults_and_ratio_checks():
    env = {"PLM_GROUNDED_RATIO": "0.15", "PLM_CHAT_RATIO": "0.10"}
    L.apply_defaults(env)
    assert env["PLM_GIT_BRANCH"] == "feat/stage-g-scratchpad"
    assert env["PLM_DATA_SLUG"] == "plm-stage-g-data"
    assert env["PLM_RELEASE_TAG"] == "stage-g-checkpoints-candidate"
    env2 = {"PLM_GROUNDED_RATIO": "0.15", "PLM_CHAT_RATIO": "0.1", "PLM_RUN_ID": "keep"}
    L.apply_defaults(env2)
    assert env2["PLM_RUN_ID"] == "keep"
    with pytest.raises(SystemExit, match="GROUNDED"):
        L.apply_defaults({"PLM_CHAT_RATIO": "0.1"})
    with pytest.raises(SystemExit, match="CHAT"):
        L.apply_defaults({"PLM_GROUNDED_RATIO": "0.1"})


def test_stage_f_launcher_reused_and_retargeted():
    stage_f = L.load_stage_f(ROOT)
    env = {"PLM_DATA_SLUG": "plm-stage-g-data", "PLM_GIT_BRANCH": "feat/stage-g-scratchpad"}
    L.configure(stage_f, env)
    assert stage_f.STAGE_F_DATA_SLUG == "plm-stage-g-data"
    assert stage_f.GIT_BRANCH == "feat/stage-g-scratchpad"
    src = (ROOT / "kaggle/stage_f_train/stage_f_train.py").read_text()
    assert 'os.environ.get("PLM_BATCH_SIZE", "6")' in src  # Stage G passes 5; Stage F default unchanged
    assert 'os.environ.get("PLM_RELEASE_TAG", "stage-f-checkpoints-candidate")' in src


@pytest.mark.parametrize("flat", [True, False])
def test_stage_grounded_flat_or_subdir(tmp_path, flat):
    root = tmp_path / "ds"
    d = root if flat else root / "grounded"
    d.mkdir(parents=True)
    stem = "grounded_train" if flat else "train"
    np.arange(2048, dtype=np.uint16).tofile(d / f"{stem}.bin")
    np.ones(2048, dtype=np.uint8).tofile(d / f"{stem}.masks.bin")
    stage_f = L.load_stage_f(ROOT)
    env = {}
    dest = L.stage_grounded(root, stage_f._stage_pair, tmp_path / "out", env)
    assert env["PLM_GROUNDED_DIR"] == str(dest)
    assert (dest / "train.bin").stat().st_size == 4096 and (dest / "train.masks.bin").is_file()


def test_stage_grounded_missing_stream_fails(tmp_path):
    with pytest.raises(SystemExit, match="grounded"):
        L.stage_grounded(tmp_path, lambda *a: None, tmp_path / "o", {})


def test_no_secrets_in_stage_g_kaggle_files():
    for rel in ("kaggle/stage_g_train/stage_g_train.py", "kaggle/stage_g_bench/stage_g_bench.py",
                "kaggle/stage_g_bench/kernel-metadata.json"):
        text = (ROOT / rel).read_text()
        for bad in ("ghp_", "hf_", "github_pat_", "KAGGLE_KEY"):
            assert bad not in text, (rel, bad)


def test_bench_param_counts_match_real_models():
    torch = pytest.importorskip("torch")
    if not hasattr(getattr(torch, "nn", None), "Linear"):
        pytest.skip("needs real torch")
    from playground_lm.model import PlaygroundLM
    for n_layers, d in [(2, 16), (3, 24)]:
        m = PlaygroundLM(vocab_size=101, n_layers=n_layers, n_heads=2, d_model=d, context_len=8)
        assert B.count_params(n_layers, d, vocab=101, ctx=8) == sum(p.numel() for p in m.parameters())
    assert B.count_params(24, 1024) == 354_823_168
    assert B.count_params(12, 768) == 124_439_808


def test_bench_summary_math():
    r = B.summarize({"micro_batch": 6, "s_per_step": 4.46}, world=2)
    assert r["tok_per_s"] == pytest.approx(98_304 / 4.46, rel=1e-4)  # Stage C: ~22k tok/s
    assert r["hours_for"]["1B"] == pytest.approx(1e9 / (r["tok_per_s"] * 3600), abs=0.1)
    assert B.tokens_per_step(2, 3) == 49_152
    assert B.summarize({"micro_batch": 1, "s_per_step": None}, 2).get("tok_per_s") is None


def test_bench_configs_and_caps():
    names = [c["name"] for c in B.CONFIGS]
    assert names == ["124M_control_12x768", "G_depth_24x768", "G_full_24x1024"]
    assert B.CONFIGS[0]["micro_batches"] == [6]  # calibrates against Stage C
    assert B.HARD_CAP_SECONDS <= 75 * 60
    for c in B.CONFIGS:
        assert c["d_model"] // c["n_heads"] == 64


def test_bench_dry_run_cpu():
    torch = pytest.importorskip("torch")
    if not hasattr(getattr(torch, "nn", None), "Linear"):
        pytest.skip("needs real torch")
    out = B.dry_run(ROOT)
    assert out["dry_run"] and all(r["finite"] for r in out["results"])
