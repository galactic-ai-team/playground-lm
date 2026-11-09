# GPT-6 Stage E Round 3 regression lane handoff

Status: **PREP ONLY**. No Kaggle kernel launched, no credentials used, no branch pushed, no model outputs evaluated. All tests are CPU-only synthetic cases.

## Repository integration

New files (compared with PR #99):
- `evals/bridge_regression_receipts.py`: converts two existing `regression_receipt` v1 JSONs into Claude's `stage_e_regression` v1 and fails closed on mismatched settings/data/coverage, nonfinite metrics, and baseline SHA.
- `kaggle/stage_e_regression/stage_e_regression.py`: script-kernel entrypoint, mounts two weights and Stage C validation by **explicit** Kaggle input paths, runs C31101 twice, candidate once, writes three small JSONs under `/kaggle/working/regression_receipts`.
- `kaggle/stage_e_regression/kernel-metadata.json`: slug `galacticaiteam/plm-stage-e-regression`, code_file `stage_e_regression.py`; input dataset references left empty for owner to attach in Kaggle UI rather than inventing IDs.
- `kaggle/stage_e_regression/README.md`: mount/env preflight and receipt schema.
- `tests/test_bridge_regression_receipts.py`, `tests/test_stage_e_regression_kernel.py`: new CPU tests.

Reference-only copies (already from prior accepted contribution, do NOT replace originals when integrating):
- `evals/run_regression_receipt.py`
- `tests/test_regression_receipt.py`

## Verification

`PYTHONPATH=.:src pytest -q tests/test_regression_receipt.py tests/test_bridge_regression_receipts.py tests/test_stage_e_regression_kernel.py`

Local synthetic run: 25 passed. Actual C31101 and candidate GPU scoring NOT run; Kaggle input mounts only exist in team workspace. `PLM_GIT_COMMIT` must be reviewed 40-character SHA including bridge file. Stage C seed SHA is checked via Claude's gate constant. Subsets publishable=false. Kaggle run requires Lumi's launch authorization.

## Guardrails

No Round-1 private prompt/answer text; no Round-3 private suite created; no training; no promotion; no local large checkpoint downloads; no automatic deletion. Private suite may be authored only after candidate freeze and FineWeb gate approval.
