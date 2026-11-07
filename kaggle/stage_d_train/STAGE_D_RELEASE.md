# Stage D finetune — post-run release checklist

Stage D is the PlaygroundLM finetune pass: ~3 steps from Stage C's
`step_00031101` on the Playground corpus. This checklist covers verification,
artifact pull, release naming, and pointing eval at the result.

## 1. Verify the run

Read the Kaggle log of `plm-stage-d-train v1` from top to bottom. All of the
following must be true before pulling artifacts.

- [ ] Launcher log line: `[stage-d-train] copying ... → /kaggle/working/runs`
      (self-reference mount resolved; otherwise it fell through to Stage B).
- [ ] Resume line in the notebook log: `[train] resumed from step 31101`
      and `(143,??? tokens, ...)`. A different step number means the mount
      did not resolve to Stage C's final checkpoint.
- [ ] `torchrun exited with code 0` — nonzero means a step crashed.
- [ ] No `nan` / `inf` in any per-step loss line. Look for `loss nan` or
      `loss inf`. A finetune on 17k tokens over 3 steps should show loss
      moving smoothly; a spike to NaN means LR too high or bad data.
- [ ] Final step line: `[train] step 31,10N loss X.XXXX` where N = 31101 + 3.
- [ ] `[train] done. session wall time ...s` present, followed by exit 0.

If any of the above fails, do not release. Repush the kernel with the same
metadata; it will resume from whatever checkpoint the crashed run saved.

## 2. Artifacts to pull

From the kernel's output (`/kaggle/working/runs/checkpoints/`):

| File | Purpose | Size |
|---|---|---|
| `step_00031104.safetensors` | Final finetuned weights | ~500 MB |
| `step_00031104_state.pt` | Optimizer + RNG (for any further training) | ~1.5 GB |
| `manifest.json` | Checkpoint index; carries token counts and wall time | small |
| `model_config.json` | Architecture (12L/12H/768d, vocab 50257) — needed for ONNX export and inference | small |

Also copy `/kaggle/working/train_bin_fingerprint.txt` if present — provenance
for which corpus slice this finetune saw.

**Do not** pull earlier step checkpoints unless a comparison is wanted; the
release should carry only the final step.

## 3. Release tag and contents

Create a GitHub release on `galactic-ai-team/playground-lm`:

- **Tag:** `stage-d-finetune-step31104` (step number matches the checkpoint
  name; use the actual final step from the log, not the assumed 31104).
- **Title:** `PlaygroundLM Stage D finetune (step N)`
- **Assets:** the four files from §2 (weights, optimizer state, manifest,
  model_config).
- **Notes:** one short paragraph — "Finetune from Stage C step 31101 on
  Playground corpus (17,631 train / 2,143 val tokens). 3 steps at lr=X.
  Resume target for Stage D eval kernels."

If `GH_TOKEN` is set as a Kaggle secret and the launcher's `PLM_UPLOAD_EVERY`
is 1, the release assets may already be uploaded by the run. Check the
`stage-d-finetune-*` tags before creating a new one — reuse if the auto-upload
succeeded.

## 4. Point eval at the Stage D output

The eval kernel is a new script, `plm-stage-d-eval`, that:

1. Mounts `plm-stage-d-train` via `kernel_sources`.
2. Reads weights from `/kaggle/input/notebooks/galacticaiteam/plm-stage-d-train/runs/checkpoints/step_00031104.safetensors`.
3. Reads `model_config.json` from the same directory to instantiate the model.
4. Runs GPT-6's eval harness (`src/playground_lm/eval/`) against Stage C's
   val.bin (already used for the Stage C comparison) and HellaSwag/LAMBADA.
5. Prints the comparison table:

Stage C (step 31101): val 3.174 HellaSwag 30.0% LAMBADA 23.0%
Stage D (step 31104): val ??? HellaSwag ??? LAMBADA ???
GPT-2 124M (ref): val 3.279 HellaSwag 30.9% LAMBADA 32.6%

6. Updates `manifest.json` on the demo (PR #86 path) so the demo's checkpoint
   selector shows the Stage D row.

If the finetune's val loss regresses vs Stage C, that is expected for a
~3-step finetune on 17k tokens; the eval is to catch catastrophic forgetting,
not to show improvement.

## 5. Free-tools only

- GitHub release: free (public repo).
- Kaggle kernel and GPU: within the weekly 30h quota (Stage D is ~1 min).
- Vercel demo: Hobby tier, no changes needed unless a new ONNX export is
  published to the demo's CDN path.

No secrets, no paid GPU, no external API keys in this checklist or the
artifacts.
