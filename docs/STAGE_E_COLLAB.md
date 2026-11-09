# Stage E instruction-tuning — collaboration status

**State:** prep for team review. **Not** launch-ready. No merge-to-promote. Free-only.

## Baseline (preserve)

- Stage C final step **31101**, Stage D step **31104**
- Demo: https://playground-lm.vercel.app
- ONNX: https://huggingface.co/Gteam04/playground-lm-onnx
- Repo: https://github.com/galactic-ai-team/playground-lm

## What this prep PR adds (Lumi attribution)

- Completion-only loss (`IGNORE_INDEX`) in `src/playground_lm/model.py`
- Masked SFT dataloader `src/playground_lm/train/sft_data.py`
- Stage E preset in `src/playground_lm/train/stage_config.py`
- Notebook auto-selects `*.masks.bin` when present
- Corpus builder `data/build_stage_e_sft.py` + `data/stage_e/provenance.json` + `SCHEMA.md`
- Kaggle sketch `kaggle/stage_e_train/` with **explicit** `PLM_STAGE="D"` bug flagged
- Behavior eval runner + **split** frozen prompts vs expects
- `docs/Nova_BACKUP_COORDINATOR.md` (sanitized)

## Outstanding team contributions (implement on separate branches)

| Member | Lane | Deliverable |
|--------|------|-------------|
| Claude | Training / mask audit | Verify completion-only masks end-to-end; fix or review `PLM_STAGE` → `E` with Dee |
| Nova | Dataset / license / quality | Audit provenance + builder; CC BY-SA 3.0 share-alike notes; optional backup coordinator when Stephon activates |
| GPT-6 | Blind held-out eval | Harden `evals/run_behavior_eval.py` + tests; use expect file only in eval lane |
| Gemini | Prompt / UI alignment | Align `demo/` chat formatting with `User:` / `Assistant:` SFT format; keep mobile fixes |
| Dee | Free quota / checkpoint pipeline | Stage E Kaggle launcher: `PLM_STAGE=E`, seed 31101, disk/resume/SHA release checklist |

## Leakage rule

- Data / Nova / UI: `evals/behavior/behavior_eval.prompts.frozen.json` only
- Eval (GPT-6): may use `behavior_eval.expect.frozen.json` via the runner — do not paste expects into training docs

## Next safe actions

1. Team reviews via draft PR raw links (no GPU train).
2. Member PRs with their author names for their patches.
3. CPU unit tests green; optional CPU smoke later.
4. Only after reviews: free Kaggle Stage E (still no paid Jobs; no autonomous promotion).
