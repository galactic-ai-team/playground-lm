#!/usr/bin/env python3
"""Free Kaggle script kernel: two comparable regression receipts and a Claude bridge.

ATTACH IN KAGGLE UI BEFORE RUNNING:
  * C31101 frozen baseline weights (.safetensors) and model_config.json;
  * Round-3 frozen candidate weights and model_config.json;
  * ORIGINAL Stage C FineWeb-Edu val.bin (NOT Stage D/E validation);
  * cached HellaSwag and LAMBADA datasets OR enable Internet for first fetch.

Set environment variables PLM_C_WEIGHTS, PLM_C_CONFIG, PLM_E_WEIGHTS,
PLM_E_CONFIG, PLM_STAGE_C_VAL (absolute /kaggle/input/... paths).
No large model downloads, no credentials, no training, no model promotion.
Only small JSON outputs in /kaggle/working/regression_receipts.
"""
from __future__ import annotations

import json
import os
from pathlib import Path
import subprocess
import sys

ROOT = Path('/kaggle/working')
REPO = ROOT / 'playground-lm'
OUTPUT = ROOT / 'regression_receipts'
# Immutable repo ref: override to a reviewed SHA, not a mutable branch, if needed.
REF = os.environ.get('PLM_GIT_COMMIT', '').strip()
URL = 'https://github.com/galactic-ai-team/playground-lm.git'


def mounted(env: str) -> Path:
    text = os.environ.get(env, '').strip()
    if not text:
        raise ValueError(f'{env} required: set to an attached /kaggle/input/... path')
    p = Path(text).resolve()
    if not p.is_file() or not str(p).startswith('/kaggle/input/'):
        raise ValueError(f'{env} must be an existing Kaggle input mount file: {p}')
    return p


def checkout() -> None:
    if not REF or len(REF) != 40 or any(c not in '0123456789abcdef' for c in REF.lower()):
        raise ValueError('PLM_GIT_COMMIT must be a reviewed full 40-character commit SHA')
    if REPO.exists():
        if not (REPO / '.git').is_dir():
            raise ValueError('Existing repo directory is not a git checkout; refusing to overwrite')
    else:
        subprocess.run(['git', 'clone', '--filter=blob:none', '--no-checkout', URL, str(REPO)], check=True)
    subprocess.run(['git', '-C', str(REPO), 'fetch', '--depth', '1', 'origin', REF], check=True)
    subprocess.run(['git', '-C', str(REPO), 'checkout', '--detach', 'FETCH_HEAD'], check=True)
    required = ['evals/run_regression_receipt.py', 'evals/bridge_regression_receipts.py',
                'evals/regression_gate.py']
    for name in required:
        if not (REPO / name).is_file():
            raise FileNotFoundError(f'Reviewed repository lacks {name}; commit these helpers first')


def main() -> None:
    baseline = mounted('PLM_C_WEIGHTS')
    candidate = mounted('PLM_E_WEIGHTS')
    val_bin = mounted('PLM_STAGE_C_VAL')
    base_cfg = mounted('PLM_C_CONFIG')
    candidate_cfg = mounted('PLM_E_CONFIG')
    if baseline == candidate:
        raise ValueError('Candidate and baseline weights cannot be the same path')
    if val_bin.stat().st_size % 2 or val_bin.stat().st_size == 0:
        raise ValueError('Stage C validation must be nonempty uint16 bin')
    checkout()
    OUTPUT.mkdir(parents=True, exist_ok=True)

    limit = os.environ.get('PLM_BENCHMARK_LIMIT', '128')
    val_batches = os.environ.get('PLM_MAX_VAL_BATCHES', '8')
    args = ['--val-bin', str(val_bin), '--device', os.environ.get('PLM_DEVICE', 'cuda'),
            '--context-len', os.environ.get('PLM_CONTEXT_LEN', '1024'),
            '--batch-size', os.environ.get('PLM_BATCH_SIZE', '4'),
            '--seed', os.environ.get('PLM_SEED', '42')]
    if limit.lower() != 'full':
        args += ['--benchmark-limit', str(int(limit))]
    if val_batches.lower() != 'full':
        args += ['--max-val-batches', str(int(val_batches))]
    if os.environ.get('PLM_DATASETS_OFFLINE', '0') == '1':
        args += ['--offline']

    env = dict(os.environ)
    env['PYTHONPATH'] = f'{REPO / "src"}:{REPO}'
    results = {}
    for name, weight, cfg in [('C31101', baseline, base_cfg), ('candidate', candidate, candidate_cfg)]:
        out = OUTPUT / f'{name}.json'
        command = [sys.executable, '-m', 'evals.run_regression_receipt',
                   '--checkpoint', str(weight), '--config', str(cfg),
                   '--out', str(out), *args]
        if name == 'C31101':
            command += ['--repeat-check']
        subprocess.run(command, cwd=REPO, env=env, check=True)
        results[name] = out
        if out.stat().st_size >= 100_000:
            raise ValueError('Receipt exceeds 100KB cap')

    combined = OUTPUT / 'stage_e_regression.json'
    bridge_command = [sys.executable, '-m', 'evals.bridge_regression_receipts',
                      '--baseline', str(results['C31101']), '--candidate', str(results['candidate']),
                      '--out', str(combined), '--candidate-name',
                      os.environ.get('PLM_CANDIDATE_NAME', 'round3_candidate'),
                      '--expected-baseline-sha256',
                      '4604d3a1b105cc70601352fbd83d6d61b34435b3cf09151b0562592a7b076275']
    subprocess.run(bridge_command, cwd=REPO, env=env, check=True)
    if combined.stat().st_size >= 100_000:
        raise ValueError('Bridged receipt exceeds 100KB cap')
    summary = json.loads(combined.read_text())
    print(json.dumps({'receipt': str(combined), 'publishable': summary['publishable'],
                      'models': list(summary['models']), 'warning': 'public subset only; NO PROMOTION'}, indent=2))


if __name__ == '__main__':
    main()
