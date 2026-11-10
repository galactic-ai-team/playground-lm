# Stage F public general-ability regression — free Kaggle package

**Public metrics only. No private suite cases, expectations, or outputs in this package.**

This is a Kaggle **script kernel**, not an automatic job. The operator attaches four distinct weights+config files and the original Stage C FineWeb-Edu `val.bin` via Kaggle `Input` mounts. Supply environment variables `PLM_C_WEIGHTS`, `PLM_C_CONFIG`, `PLM_D_WEIGHTS`, `PLM_D_CONFIG`, `PLM_E_WEIGHTS`, `PLM_E_CONFIG`, `PLM_F_WEIGHTS`, `PLM_F_CONFIG`, `PLM_STAGE_C_VAL` as absolute `/kaggle/input/...` paths. Set `PLM_GIT_COMMIT` to a reviewed **immutable 40-character commit SHA** already containing `evals/run_regression_receipt.py`, `evals/bridge_regression_receipts.py`, and Claude's `evals/regression_gate.py`.

No large checkpoint downloads to the near-full owner box. Kaggle dataset/kernel outputs must be explicitly attached; this package does not use credentials or invoke training. Default subset settings (official Stage F candidate knobs): HellaSwag 200 examples, LAMBADA 200 examples, FineWeb-Edu 20 validation batches, seed 20261108; **all models use identical limits and token stream**. Change `PLM_BENCHMARK_LIMIT` to `1000` only for a final promotion candidate (same set, same seed). Use `full` only if quota permits. `PLM_DATASETS_OFFLINE=1` requires already cached benchmark datasets; else allow Kaggle Internet for standard datasets.

`kaggle/stage_f_regression/stage_f_regression.py` creates C31101/D31104/E31140/StageF JSON regression receipts and Claude-compatible `stage_e_regression` v1 bridges for each vs C31101. The baseline C31101 automatically gets a repeated deterministic run. Bridging verifies original C seed SHA. Subset receipts **must be nonpublishable**. Validate with `pytest -q tests/test_stage_f_kernel.py`.

**No launch, merge, or model promotion has been performed.** Promotion must additionally satisfy private paired significance vs live E31140 and G5 leakage non-FAIL; Stephon makes the decision.
