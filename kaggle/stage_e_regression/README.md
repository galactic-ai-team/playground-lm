# Round 3 public regression — preparation only

This package is not a launched Kaggle run. Do **not** run until Lumi authorizes. It does not train, promote, delete files, or access private evaluation cases. Mount large weights only on Kaggle. No local checkpoints.

## Preflight

1. Include `evals/bridge_regression_receipts.py` in a reviewed commit on `review/stage-e-round2` (or a follow-up branch). The previously accepted `evals/run_regression_receipt.py` is reused unchanged.
2. Create Kaggle **script** kernel slug `galacticaiteam/plm-stage-e-regression` with the included `stage_e_regression.py` + `kernel-metadata.json` (the `id` and `code_file` must match).
3. Attach Kaggle inputs via UI: C31101 frozen seed safetensors + its model config, frozen Round 3 candidate safetensors + its model config, **the original Stage C FineWeb-Edu `val.bin`**, and cached official HellaSwag/LAMBADA data (or use internet access for the public datasets).
4. Set env vars `PLM_C_WEIGHTS`, `PLM_C_CONFIG`, `PLM_E_WEIGHTS`, `PLM_E_CONFIG`, `PLM_STAGE_C_VAL` to exact files under `/kaggle/input/...`.
5. Set `PLM_GIT_COMMIT` to a **reviewed full commit SHA** that contains both bridge and existing receipt utilities. The script requires an immutable 40-character SHA; no default branch or short ref is accepted. Optionally set `PLM_CANDIDATE_NAME`.
6. Optional common subset knobs (must be same for both): `PLM_BENCHMARK_LIMIT=128`, `PLM_MAX_VAL_BATCHES=8`, `PLM_CONTEXT_LEN=1024`, `PLM_BATCH_SIZE=4`, `PLM_SEED=42`, `PLM_DEVICE=cuda`. To request complete scoring, use `PLM_BENCHMARK_LIMIT=full` and `PLM_MAX_VAL_BATCHES=full`; this may exhaust quota. Use `PLM_DATASETS_OFFLINE=1` only if datasets were already cached in Kaggle.

## Outputs

Three small JSONs in `/kaggle/working/regression_receipts/`: `C31101.json`, `candidate.json`, and `stage_e_regression.json`. The first two retain the existing `regression_receipt` v1 schema (SHA-256 checkpoint and val.bin, settings, coverage, metrics, UTC timestamp, repeat check). The third bridges to Claude's `stage_e_regression` v1: `baseline`, `models`, and `eval` fields, with identical limits and hash. The bridge calls `evals.regression_gate.validate_receipt` when present; baseline C31101 SHA is checked against Claude's frozen constant.

C31101 uses `--repeat-check`. Repeat metric deltas default to absolute `1e-5` tolerance. Subsets **always** produce `publishable=false`. Full validation requires all splits completed and cannot be assumed from limits. The gate is advisory and has no promotion side effects.

## Frozen Round-2 private suite

No new private suite in this prep delivery. After Round 3 candidate freeze and Claude FineWeb gate approval, create a **new** ~60-case suite and run local prompt-hash non-overlap against the Round-1 suite. Hashes of whole JSON files alone are insufficient for prompt-level overlap checks. Never commit private expectation files or model outputs. No Round-1 private prompt text included here.

## Tests

`PYTHONPATH=src:. pytest -q tests/test_regression_receipt.py tests/test_bridge_regression_receipts.py tests/test_stage_e_regression_kernel.py`
