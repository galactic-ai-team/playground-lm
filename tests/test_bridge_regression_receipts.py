"""Offline synthetic contracts; no checkpoints, GPU, downloads or private probes."""
import copy
import pytest
from evals.bridge_regression_receipts import bridge


def receipt(sha='a'*64):
    return {
        'schema_version': 1, 'receipt_type': 'regression_receipt',
        'suite_type': 'public_general_ability_only',
        'checkpoint': {'sha256': sha},
        'datasets': {'validation': {'sha256': 'b'*64},
                     'hellaswag': 'Rowan/hellaswag:validation',
                     'lambada': 'EleutherAI/lambada_openai:default:test'},
        'settings': {'context_len':1024, 'batch_size':4,'benchmark_limit':128,
                     'max_val_batches':8,'seed':42,'device':'cuda'},
        'metrics': {'val_loss':3.17,'perplexity':23.807,
                    'hellaswag_acc_norm':.3,'lambada_acc':.23},
        'coverage': {'validation':{'tokens_scored':32768,'available_targets':100000,'complete':False},
                     'hellaswag':{'examples':128,'correct':38,'complete':False},
                     'lambada':{'examples':128,'correct':29,'complete':False}},
        'complete':False,'publishable':False,
    }


def test_bridge_schema_and_nonpublishable():
    a=receipt()
    b=receipt('c'*64)
    result=bridge(a,b,expected_baseline_sha256='a'*64)
    assert result['kind']=='stage_e_regression'
    assert result['schema_version']==1
    assert result['baseline']=='C31101'
    assert result['eval']['fineweb_val_sha256']=='b'*64
    assert result['eval']['val_max_batches']==8
    assert result['eval']['hellaswag_limit']==128
    assert result['eval']['lambada_limit']==128
    assert result['models']['C31101']['fineweb']['tokens_scored']==32768
    assert result['models']['round3_candidate']['weights_sha256']=='c'*64
    assert result['publishable'] is False
    assert a==receipt()  # immutable inputs


def test_reject_different_validation_hash():
    b=receipt('c'*64)
    b['datasets']['validation']['sha256']='d'*64
    with pytest.raises(ValueError,match='Validation corpus'):
        bridge(receipt(),b)

@pytest.mark.parametrize('key', ['context_len','batch_size','benchmark_limit','max_val_batches','seed'])
def test_reject_mismatched_settings(key):
    b=receipt('c'*64)
    b['settings'][key]=999
    with pytest.raises(ValueError,match='Incomparable setting'):
        bridge(receipt(),b)

@pytest.mark.parametrize('category,field', [('validation','tokens_scored'),('validation','available_targets'),('hellaswag','examples'),('lambada','examples')])
def test_reject_different_coverage(category,field):
    b=receipt('c'*64)
    b['coverage'][category][field]+=1
    with pytest.raises(ValueError,match='Incomparable'):
        bridge(receipt(),b)


def test_reject_baseline_hash_mismatch():
    with pytest.raises(ValueError,match='baseline weight hash mismatch'):
        bridge(receipt(),receipt('c'*64),expected_baseline_sha256='e'*64)


def test_reject_invalid_metrics_and_public_schema():
    b=receipt('c'*64)
    b['metrics']['lambada_acc']=float('nan')
    with pytest.raises(ValueError,match='finite'):
        bridge(receipt(),b)
    b=receipt('c'*64)
    b['suite_type']='private_holdout'
    with pytest.raises(ValueError,match='public general ability'):
        bridge(receipt(),b)


def test_full_only_if_both_full():
    a,b=receipt(),receipt('c'*64)
    a['complete']=a['publishable']=True
    assert bridge(a,b)['publishable'] is False
    b['complete']=b['publishable']=True
    assert bridge(a,b)['publishable'] is True
