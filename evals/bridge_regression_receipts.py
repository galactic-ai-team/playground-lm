#!/usr/bin/env python3
"""Bridge TWO immutable public regression receipts into Claude's stage_e_regression v1.

No model computation, private probes, uploads, or training. Fail closed on
non-identical eval settings, validation bytes and sample coverage. Subsets
are non-publishable. The public receipt is NOT promotion authorization.
"""
from __future__ import annotations

import argparse
import json
import math
from pathlib import Path
from typing import Any, Mapping

REQUIRED_METRICS = ('val_loss', 'perplexity', 'hellaswag_acc_norm', 'lambada_acc')
COMPARABLE_SETTINGS = ('context_len', 'batch_size', 'benchmark_limit', 'max_val_batches', 'seed')


def _finite(value: Any, name: str) -> float:
    if isinstance(value, bool) or not isinstance(value, (int, float)) or not math.isfinite(value):
        raise ValueError(f'{name} must be finite numeric')
    return float(value)


def _validate_receipt(r: Mapping[str, Any]) -> None:
    if r.get('receipt_type') != 'regression_receipt' or r.get('schema_version') != 1:
        raise ValueError('Not a regression_receipt v1')
    if r.get('suite_type') != 'public_general_ability_only':
        raise ValueError('Receipt must be public general ability only')
    for key in REQUIRED_METRICS:
        _finite(r['metrics'][key], key)
    if not isinstance(r['checkpoint']['sha256'], str) or len(r['checkpoint']['sha256']) != 64:
        raise ValueError('Invalid weights SHA-256')
    if not isinstance(r['datasets']['validation']['sha256'], str) or len(r['datasets']['validation']['sha256']) != 64:
        raise ValueError('Invalid val.bin SHA-256')
    if r.get('publishable') and not r.get('complete'):
        raise ValueError('Incomplete receipt cannot be publishable')
    for category in ('validation', 'hellaswag', 'lambada'):
        if category not in r['coverage']:
            raise ValueError(f'Missing {category} coverage')


def bridge(baseline: Mapping[str, Any], candidate: Mapping[str, Any], *,
           baseline_name: str = 'C31101', candidate_name: str = 'round3_candidate',
           expected_baseline_sha256: str | None = None) -> dict[str, Any]:
    """Create a comparison receipt with Claude's expected fields."""
    if not baseline_name or not candidate_name or baseline_name == candidate_name:
        raise ValueError('Unique model names required')
    _validate_receipt(baseline)
    _validate_receipt(candidate)
    if expected_baseline_sha256 and baseline['checkpoint']['sha256'] != expected_baseline_sha256:
        raise ValueError('Stage C baseline weight hash mismatch')
    if baseline['datasets']['validation']['sha256'] != candidate['datasets']['validation']['sha256']:
        raise ValueError('Validation corpus differs between checkpoints')
    for key in COMPARABLE_SETTINGS:
        if baseline['settings'].get(key) != candidate['settings'].get(key):
            raise ValueError(f'Incomparable setting: {key}')
    if baseline['datasets']['hellaswag'] != candidate['datasets']['hellaswag'] or baseline['datasets']['lambada'] != candidate['datasets']['lambada']:
        raise ValueError('Benchmark dataset identifiers differ')
    for category, keys in (('validation', ('tokens_scored','available_targets')), ('hellaswag', ('examples',)), ('lambada', ('examples',))):
        for key in keys:
            if baseline['coverage'][category].get(key) != candidate['coverage'][category].get(key):
                raise ValueError(f'Incomparable {category} coverage: {key}')

    def model_fields(r: Mapping[str, Any]) -> dict[str, Any]:
        m, c = r['metrics'], r['coverage']
        return {
            'weights_sha256': r['checkpoint']['sha256'],
            'fineweb': {'loss': m['val_loss'], 'perplexity': m['perplexity'],
                        'tokens_scored': c['validation']['tokens_scored']},
            'hellaswag': {'accuracy': m['hellaswag_acc_norm'],
                          'examples': c['hellaswag']['examples']},
            'lambada': {'last_word_accuracy': m['lambada_acc'],
                        'examples': c['lambada']['examples']},
        }

    s = baseline['settings']
    full = bool(baseline['complete'] and candidate['complete'] and baseline['publishable'] and candidate['publishable'])
    return {
        'kind': 'stage_e_regression', 'schema_version': 1,
        'baseline': baseline_name,
        'models': {baseline_name: model_fields(baseline), candidate_name: model_fields(candidate)},
        'eval': {'fineweb_val_sha256': baseline['datasets']['validation']['sha256'],
                 'val_max_batches': s['max_val_batches'], 'hellaswag_limit': s['benchmark_limit'],
                 'lambada_limit': s['benchmark_limit'], 'context_len': s['context_len'],
                 'batch_size': s['batch_size'], 'seed': s['seed']},
        'publishable': full,
        'limitations': 'Public regression only; subset scores are nonpublishable. No private held-out evaluation or automatic promotion.',
    }


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--baseline', type=Path, required=True)
    p.add_argument('--candidate', type=Path, required=True)
    p.add_argument('--out', type=Path, required=True)
    p.add_argument('--candidate-name', default='round3_candidate')
    p.add_argument('--expected-baseline-sha256')
    args = p.parse_args()
    b = json.loads(args.baseline.read_text())
    c = json.loads(args.candidate.read_text())
    result = bridge(b, c, candidate_name=args.candidate_name,
                    expected_baseline_sha256=args.expected_baseline_sha256)
    try:
        from evals.regression_gate import validate_receipt
    except ImportError:
        pass
    else:
        issues = validate_receipt(result)
        if issues:
            raise ValueError(f'Claude regression gate rejected receipt: {issues}')
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(result, indent=2, allow_nan=False) + '\n')
    print(json.dumps({'out': str(args.out), 'publishable': result['publishable'], 'models': list(result['models'])}))


if __name__ == '__main__':
    main()
