"""CPU-only structural tests. No private suites, downloads, or model weights."""
import hashlib
import json
import math

import pytest

from evals.run_regression_receipt import (
    make_receipt, metric_delta, sha256_path, validate_metrics,
)


def result(full=True):
    return {
        'validation': {'loss': 3.0, 'perplexity': math.exp(3), 'tokens_scored': 80,
                       'available_targets': 80 if full else 100, 'complete': full},
        'hellaswag': {'accuracy': .3, 'examples': 10042 if full else 10,
                      'correct': 3012 if full else 3, 'complete': full},
        'lambada': {'last_word_accuracy': .2, 'examples': 5153 if full else 10,
                    'correct': 1031 if full else 2, 'complete': full},
        'metrics': {'val_loss': 3.0, 'perplexity': math.exp(3),
                    'hellaswag_acc_norm': .3, 'lambada_acc': .2},
    }


def test_full_receipt_schema_and_sha(tmp_path):
    ckpt, val = tmp_path / 'weights.safetensors', tmp_path / 'val.bin'
    ckpt.write_bytes(b'weights')
    val.write_bytes(b'\x01\x00' * 8)
    r = result()
    receipt = make_receipt(checkpoint=ckpt, val_bin=val, config=None,
                           context_len=1024, batch_size=4, benchmark_limit=None,
                           max_val_batches=None, seed=42, device='cpu', offline=True,
                           validation=r['validation'], hellaswag=r['hellaswag'],
                           lambada=r['lambada'], now='2026-11-09T00:00:00+00:00',
                           repeat_deltas={k: 0.0 for k in r['metrics']})
    assert receipt['receipt_type'] == 'regression_receipt'
    assert receipt['checkpoint']['sha256'] == hashlib.sha256(b'weights').hexdigest()
    assert receipt['datasets']['validation']['sha256'] == sha256_path(val)
    assert receipt['publishable'] and receipt['complete']
    assert receipt['repeat_check']['performed']
    assert 'prompts' not in json.dumps(receipt)
    assert receipt['settings']['benchmark_limit'] is None


def test_subset_receipt_nonpublishable(tmp_path):
    ckpt, val = tmp_path/'ckpt', tmp_path/'val.bin'
    ckpt.write_bytes(b'w')
    val.write_bytes(b'\x00\x00')
    r = result(full=False)
    receipt = make_receipt(checkpoint=ckpt, val_bin=val, config=None, context_len=64,
                           batch_size=2, benchmark_limit=10, max_val_batches=1,
                           seed=1, device='cpu', offline=True,
                           validation=r['validation'], hellaswag=r['hellaswag'], lambada=r['lambada'])
    assert not receipt['complete'] and not receipt['publishable']
    assert receipt['coverage']['hellaswag']['examples'] == 10


def test_unexpected_partial_full_run_rejected(tmp_path):
    r = result(full=False)
    with pytest.raises(ValueError, match='incomplete'):
        validate_metrics(r['validation'], r['hellaswag'], r['lambada'], full_requested=True)


def test_repeat_identical_passes():
    r = result()
    assert all(v == 0 for v in metric_delta(r, result(), tolerance=1e-8).values())


def test_repeat_numeric_drift_rejected():
    first, second = result(), result()
    second['metrics']['val_loss'] += 0.01
    with pytest.raises(ValueError, match='Repeat-check failed'):
        metric_delta(first, second, tolerance=1e-5)


def test_repeat_count_drift_rejected():
    first, second = result(), result()
    second['hellaswag']['examples'] = 10000
    with pytest.raises(ValueError, match='Nondeterministic'):
        metric_delta(first, second, tolerance=1e-5)


def test_bad_metrics_rejected():
    r = result()
    r['lambada']['last_word_accuracy'] = 1.2
    with pytest.raises(ValueError, match='outside'):
        validate_metrics(r['validation'], r['hellaswag'], r['lambada'], full_requested=True)


def test_bad_perplexity_rejected():
    r = result()
    r['validation']['perplexity'] = 25
    with pytest.raises(ValueError, match='inconsistent'):
        validate_metrics(r['validation'], r['hellaswag'], r['lambada'], full_requested=True)
