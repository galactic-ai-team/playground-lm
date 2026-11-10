#!/usr/bin/env python3
"""Stage F free Kaggle public regression: C31101, D31104, E31140, candidate.

GPT-6 package, defaults pinned to official Stage F candidate knobs
(benchmark_limit=200, max_val_batches=20, seed 20261108). Use
PLM_BENCHMARK_LIMIT=1000 only for a final promotion candidate.

Attach all weights/configs as Kaggle INPUT datasets or notebook outputs; attach
original Stage C FineWeb-Edu val.bin. No downloads of weights, no training.
Run only after operator reviews and manually launches this kernel.
Output: four small run_regression_receipt JSON files + three Claude v1 bridges.

Required environment (all absolute /kaggle/input/... files):
 PLM_C_WEIGHTS, PLM_C_CONFIG, PLM_D_WEIGHTS, PLM_D_CONFIG,
 PLM_E_WEIGHTS, PLM_E_CONFIG, PLM_F_WEIGHTS, PLM_F_CONFIG,
 PLM_STAGE_C_VAL, PLM_GIT_COMMIT (reviewed 40-char commit).

Optional: PLM_BENCHMARK_LIMIT (default 200), PLM_MAX_VAL_BATCHES (default 20),
 PLM_DEVICE=cuda, PLM_DATASETS_OFFLINE=1, PLM_CANDIDATE_NAME=StageF.
Use "full" for BOTH limits to request full scoring; partial is NEVER publishable.
"""
from __future__ import annotations
import json
import os
from pathlib import Path
import subprocess
import sys

ROOT=Path('/kaggle/working')
REPO=ROOT/'playground-lm'
OUT=ROOT/'stage_f_regression_receipts'
URL='https://github.com/galactic-ai-team/playground-lm.git'
BASELINE_SHA='4604d3a1b105cc70601352fbd83d6d61b34435b3cf09151b0562592a7b076275'
MODELS=(('C31101','PLM_C_WEIGHTS','PLM_C_CONFIG'),
        ('D31104','PLM_D_WEIGHTS','PLM_D_CONFIG'),
        ('E31140','PLM_E_WEIGHTS','PLM_E_CONFIG'),
        ('StageF','PLM_F_WEIGHTS','PLM_F_CONFIG'))


def mounted(name:str)->Path:
    value=os.environ.get(name,'').strip()
    if not value:raise ValueError(f'{name} is required')
    p=Path(value).resolve()
    if not p.is_file() or not p.is_relative_to('/kaggle/input'):
        raise ValueError(f'{name} must be an existing /kaggle/input input file')
    return p


def reviewed_ref()->str:
    ref=os.environ.get('PLM_GIT_COMMIT','').strip().lower()
    if len(ref)!=40 or any(c not in '0123456789abcdef' for c in ref):
        raise ValueError('PLM_GIT_COMMIT must be a reviewed immutable 40-char SHA')
    return ref


def checkout(ref:str)->None:
    if REPO.exists() and not (REPO/'.git').is_dir():
        raise ValueError('Refusing to overwrite non-git repo directory')
    if not REPO.exists():
        subprocess.run(['git','clone','--filter=blob:none','--no-checkout',URL,str(REPO)],check=True)
    subprocess.run(['git','-C',str(REPO),'fetch','--depth','1','origin',ref],check=True)
    subprocess.run(['git','-C',str(REPO),'checkout','--detach','FETCH_HEAD'],check=True)
    for rel in ('evals/run_regression_receipt.py','evals/bridge_regression_receipts.py',
                'evals/regression_gate.py'):
        if not (REPO/rel).is_file():raise FileNotFoundError(f'Reviewed SHA lacks {rel}')


def eval_args(val_bin:Path)->list[str]:
    limit=os.environ.get('PLM_BENCHMARK_LIMIT','200')
    batches=os.environ.get('PLM_MAX_VAL_BATCHES','20')
    args=['--val-bin',str(val_bin),'--device',os.environ.get('PLM_DEVICE','cuda'),
          '--context-len',os.environ.get('PLM_CONTEXT_LEN','1024'),
          '--batch-size',os.environ.get('PLM_BATCH_SIZE','4'),
          '--seed',os.environ.get('PLM_SEED','20261108')]
    if limit.lower()!='full':
        if int(limit)<1:raise ValueError('Benchmark limit must be positive')
        args+=['--benchmark-limit',str(int(limit))]
    if batches.lower()!='full':
        if int(batches)<1:raise ValueError('Validation batch limit must be positive')
        args+=['--max-val-batches',str(int(batches))]
    if os.environ.get('PLM_DATASETS_OFFLINE','0')=='1':args+=['--offline']
    return args


def small(path:Path)->None:
    if not path.is_file() or path.stat().st_size >=100_000:
        raise ValueError(f'Missing or oversize JSON receipt: {path}')


def main()->None:
    # Fail fast on mounts before checkout or large reads.
    mount={name:(mounted(w),mounted(c)) for name,w,c in MODELS}
    val=mounted('PLM_STAGE_C_VAL')
    if val.stat().st_size==0 or val.stat().st_size%2:raise ValueError('Stage C val.bin not uint16')
    if len({str(mount[x][0]) for x in mount})!=len(MODELS):
        raise ValueError('Every model must use distinct weight files')
    ref=reviewed_ref();args=eval_args(val)
    checkout(ref)
    OUT.mkdir(parents=True,exist_ok=True)
    env=dict(os.environ);env['PYTHONPATH']=f'{REPO / "src"}:{REPO}'
    receipts={}
    for name,_,_ in MODELS:
        weights,config=mount[name];dest=OUT/f'{name}.json'
        cmd=[sys.executable,'-m','evals.run_regression_receipt',
             '--checkpoint',str(weights),'--config',str(config),'--out',str(dest),*args]
        if name=='C31101':cmd+=['--repeat-check']
        subprocess.run(cmd,cwd=REPO,env=env,check=True)
        small(dest)
        doc=json.loads(dest.read_text())
        if (doc.get('settings',{}).get('benchmark_limit') is not None or
            doc.get('settings',{}).get('max_val_batches') is not None) and doc.get('publishable') is not False:
            raise ValueError('Subset receipts must be nonpublishable')
        receipts[name]=dest
    bridges={}
    for model_name in ('D31104','E31140','StageF'):
        dest=OUT/f'C31101_vs_{model_name}.json'
        subprocess.run([sys.executable,'-m','evals.bridge_regression_receipts',
                        '--baseline',str(receipts['C31101']),
                        '--candidate',str(receipts[model_name]),
                        '--out',str(dest),'--candidate-name',model_name,
                        '--expected-baseline-sha256',BASELINE_SHA],
                       cwd=REPO,env=env,check=True)
        small(dest)
        bridges[model_name]=str(dest)
    print(json.dumps({'receipts':{k:str(v) for k,v in receipts.items()},
                      'bridges':bridges,'training_started':False,
                      'promotion_decision':'NONE — Stephon only'},indent=2))

if __name__=='__main__':main()
