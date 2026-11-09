# Stage E integration glue (Lumi)

Team contributions landed as separate commits on `review/stage-e-sft-prep`. Modest Lumi glue:

1. **model.py `chunked_cross_entropy`** — Claude flagged `.item()` / data-dependent `if count == 0` as breaking `torch.compile(fullgraph=True)`. Applied Claude’s suggested tensor `count.clamp(min=1)` approach (Lumi).
2. **`test_invalid_stage_raises`** — Stage E is a real preset; test updated to accept `E` and still reject unknown stages (Lumi).
3. **Dee launcher backups** — Revised Dee launcher already used atomic `.tmp` + `os.replace`. Lumi stopped post-install `shutil.rmtree(runs.old)` and archives a pre-existing `.old` to `runs.old.bak.<UTC>` so non-identical checkpoint trees are not deleted.
4. **Dee tests path** — `tests/test_stage_e_launcher.py` imports `kaggle/stage_e_train/stage_e_train.py` so CI collection under `tests/` works.

No GPU training; draft PR only.

## Lumi follow-up (preflight → candidate prep)

Authorship: **Lumi** (`lumi@ai-playground.team`). Narrow glue only:

1. **`data/build_stage_e_sft.py`** — wire Nova `load_denylist` onto the *User:* field of the bin+masks builder (Nova JSONL Instruction: audit did not cover this train path). License strings for generated arithmetic/identity reconciled to `MIT (this repository)`. Provenance records `denylist_audit`.
2. **`data/test_build_stage_e_sft_denylist.py`** — synthetic User-schema exact/ngram, empty/missing fail-closed, license metadata.
3. **`notebooks/playground_lm_train.py`** — `PLM_MILESTONES` env; `PLM_RESET_OPTIMIZER=1` rebuilds fresh AdamW after resume (keeps weights/step, drops Stage C moments for SFT).
4. **`kaggle/stage_e_train/stage_e_train.py`** — default 1 epoch, eval 5 on short runs, milestones every 10, `PLM_RESET_OPTIMIZER=1`, pin `review/stage-e-sft-prep` (+ optional `PLM_GIT_COMMIT`), run id `stage-e-sft-candidate-ep1`.

Corpus candidate lives at `/workspace/plm_stageE/sft_candidate/` (do not silently train old `sft/`).
