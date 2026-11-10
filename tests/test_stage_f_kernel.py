"""Static/offline invariants for Stage F Kaggle regression adapter (GPT-6)."""
import importlib.util
import json
from pathlib import Path
import pytest

HERE=Path(__file__).resolve().parents[1]
DIR=HERE/'kaggle'/'stage_f_regression'
SPEC=importlib.util.spec_from_file_location('stage_f_kernel',DIR/'stage_f_regression.py')
m=importlib.util.module_from_spec(SPEC);SPEC.loader.exec_module(m)


def test_slug_matches_code_file():
    k=json.loads((DIR/'kernel-metadata.json').read_text())
    assert k['id']=='galacticaiteam/plm-stage-f-regression'
    assert (DIR/k['code_file']).exists()
    assert k['kernel_type']=='script'
    assert k['enable_gpu'] is True
    assert k['is_private'] is True


def test_exact_models_and_sha():
    assert [x[0] for x in m.MODELS]==['C31101','D31104','E31140','StageF']
    assert m.BASELINE_SHA=='4604d3a1b105cc70601352fbd83d6d61b34435b3cf09151b0562592a7b076275'


def test_official_candidate_defaults(monkeypatch):
    monkeypatch.delenv('PLM_BENCHMARK_LIMIT', raising=False)
    monkeypatch.delenv('PLM_MAX_VAL_BATCHES', raising=False)
    monkeypatch.delenv('PLM_SEED', raising=False)
    args=m.eval_args(Path('/kaggle/input/stage-c/val.bin'))
    assert args[args.index('--benchmark-limit')+1]=='200'
    assert args[args.index('--max-val-batches')+1]=='20'
    assert args[args.index('--seed')+1]=='20261108'


def test_limits_explicit_and_identical(monkeypatch):
    monkeypatch.setenv('PLM_BENCHMARK_LIMIT','128')
    monkeypatch.setenv('PLM_MAX_VAL_BATCHES','8')
    args=m.eval_args(Path('/kaggle/input/stage-c/val.bin'))
    assert args[args.index('--benchmark-limit')+1]=='128'
    assert args[args.index('--max-val-batches')+1]=='8'
    monkeypatch.setenv('PLM_BENCHMARK_LIMIT','full')
    monkeypatch.setenv('PLM_MAX_VAL_BATCHES','full')
    full=m.eval_args(Path('/kaggle/input/stage-c/val.bin'))
    assert '--benchmark-limit' not in full and '--max-val-batches' not in full


def test_bad_limits_and_mutable_refs_rejected(monkeypatch):
    monkeypatch.setenv('PLM_BENCHMARK_LIMIT','0')
    with pytest.raises(ValueError):m.eval_args(Path('/kaggle/input/val.bin'))
    monkeypatch.setenv('PLM_GIT_COMMIT','main')
    with pytest.raises(ValueError):m.reviewed_ref()


def test_kaggle_mount_refuses_host_disk(monkeypatch,tmp_path):
    path=tmp_path/'weights.safetensors';path.write_bytes(b'weights')
    monkeypatch.setenv('PLM_C_WEIGHTS',str(path))
    with pytest.raises(ValueError,match='kaggle/input'):m.mounted('PLM_C_WEIGHTS')
