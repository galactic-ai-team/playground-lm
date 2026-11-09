# Stage E integration glue (Lumi)

Team contributions landed as separate commits on `review/stage-e-sft-prep`. Modest Lumi glue:

1. **model.py `chunked_cross_entropy`** — Claude flagged `.item()` / data-dependent `if count == 0` as breaking `torch.compile(fullgraph=True)`. Applied Claude’s suggested tensor `count.clamp(min=1)` approach (Lumi).
2. **`test_invalid_stage_raises`** — Stage E is a real preset; test updated to accept `E` and still reject unknown stages (Lumi).
3. **Dee launcher backups** — Revised Dee launcher already used atomic `.tmp` + `os.replace`. Lumi stopped post-install `shutil.rmtree(runs.old)` and archives a pre-existing `.old` to `runs.old.bak.<UTC>` so non-identical checkpoint trees are not deleted.
4. **Dee tests path** — `tests/test_stage_e_launcher.py` imports `kaggle/stage_e_train/stage_e_train.py` so CI collection under `tests/` works.

No GPU training; draft PR only.
