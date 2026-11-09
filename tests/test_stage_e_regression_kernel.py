"""Metadata and runner contract; no external services accessed."""
import json
import sys
from pathlib import Path
import pytest

ROOT=Path(__file__).resolve().parents[1]
KERNEL=ROOT/'kaggle'/'stage_e_regression'
sys.path.insert(0,str(KERNEL))
import stage_e_regression as k


def test_kaggle_code_file_slug_matches():
    meta=json.loads((KERNEL/'kernel-metadata.json').read_text())
    assert meta['id']=='galacticaiteam/plm-stage-e-regression'
    assert meta['code_file']=='stage_e_regression.py'
    assert (KERNEL/meta['code_file']).is_file()
    assert meta['kernel_type']=='script'
    assert meta['is_private'] is True
    assert meta['model_sources']==[]
    assert meta['dataset_sources']==[]  # operator attaches explicitly later


def test_fail_closed_missing_mount(monkeypatch):
    monkeypatch.delenv('PLM_C_WEIGHTS',raising=False)
    with pytest.raises(ValueError,match='PLM_C_WEIGHTS required'):
        k.mounted('PLM_C_WEIGHTS')


def test_no_training_or_private_suite_in_kernel_source():
    source=(KERNEL/'stage_e_regression.py').read_text()
    assert "evals.run_regression_receipt" in source
    assert "--repeat-check" in source
    assert "evals.bridge_regression_receipts" in source
    assert "train.py" not in source
    assert "behavior_eval.expect" not in source
