# Stage C Session 4 — resume push checklist

Session 4 is the **final** Stage C session. It resumes from s3's checkpoint
and runs to the group's target of **31,101 steps**, then exits.

## 1. Preconditions (verify before pushing)

- [ ] `plm-stage-c-train`'s latest Kaggle version is **s3 (v5)**, status COMPLETE.
- [ ] That version's `/kaggle/working/runs/manifest.json` has
      `latest_step` in the range **28,000 – 29,500**. If it's outside that,
      read the s3 log tail and confirm the exit was clean before pushing s4.
- [ ] `runs/train_bin_fingerprint.txt` exists in s3's output (Lumi's
      integration writes it after the size check passes).
- [ ] The `plm-stage-b-launch` kernel source is still present (only needed if
      the self-reference mount falls through; normally S4 resumes from s3).
- [ ] GPU quota: s4 is short (~2–3 h wall), well under 30 h/week. Quota
      reset this week, so no concern.

## 2. Metadata file to push

Use **`kernel-metadata-resume.json`** from `kaggle/stage_c_train/`. It is
identical to what s1–s3 used. Do not edit it for s4.

Self-reference entry that makes resume work:

```json
"kernel_sources": [
  "galacticaiteam/plm-stage-c-data-prep-s0",
  "galacticaiteam/plm-stage-c-data-prep-s1",
  "galacticaiteam/plm-stage-c-data-prep-s2",
  "galacticaiteam/plm-stage-c-data-prep-s3",
  "galacticaiteam/plm-stage-c-data-prep-s4",
  "galacticaiteam/plm-stage-c-data-prep-s5",
  "galacticaiteam/plm-stage-c-data-prep-s6",
  "galacticaiteam/plm-stage-c-data-prep-s7",
  "galacticaiteam/plm-stage-b-launch",
  "galacticaiteam/plm-stage-c-train"
]
```

Push command (Kaggle CLI reads `kernel-metadata.json` from the working dir;
copy the resume file over it first):

```bash
cd kaggle/stage_c_train
cp kernel-metadata-resume.json kernel-metadata.json
kaggle kernels push
```

## 3. Env vars — leave UNSET

`stage_c_train.py` reads these; do not set them for s4:

| Env var | Why leave unset |
|---|---|
| `PLM_STAGE_C_MAX_STEPS` | Defaults to 31101 (group target). Setting it would cut the run short. |
| `PLM_SESSION` | Not used by the launcher. |
| `PLM_RUN_ID` | Defaults to `stage-c` in the launcher; matches s1–s3's run id. |
| `PLM_RUN_DIR` | Defaults to `/kaggle/working/runs`. |
| `PLM_DATA_DIR` | Defaults to `/kaggle/working` (where the concat lands). |

`GH_TOKEN` (Kaggle Secret): optional. If set, s4 uploads its final
checkpoint to the `stage-c-checkpoints` release. If not set, the final
checkpoint stays in `/kaggle/working/runs` — still downloadable from the
kernel's output page. Recommend setting it once s4 completes if you want a
permanent artifact outside Kaggle.

## 4. Expected run

- **Resume step:** ~28,000–29,500 (whatever s3's last save was).
- **Final step:** 31,101.
- **Steps this session:** ~1,600–3,100.
- **Wall time:** 4.4 s/step × 3,100 steps ≈ 3.8 h worst case. Under
  Kaggle's 12 h cap with huge margin.
- **Exit:** `train_loop` returns when `step >= max_steps`; launcher exits 0;
  checkpoint at `runs/checkpoints/step_00031101.safetensors` + `_state.pt`.

Watch the log for these lines:

```text
[stage-c-train] copying previous Stage C runs from ... → /kaggle/working/runs
[stage-c-train] copied N files from previous Stage C session
[train] resumed from step <s3_step> (<tokens>, <wall>s wall)
...
[train] done. session wall time ...
[stage-c-train] torchrun exited with code 0
```

If instead you see no previous Stage C runs or `no GH_TOKEN — Stage C checkpoints carry forward` / the self-reference mount did not resolve — do not
let it run; check Kaggle's kernel sources before proceeding.

## 5. Disk budget

- `/kaggle/working` quota: ~20 GB.
- `/tmp` concat from Lumi's integration: 5.7 GB (`train.bin` + `val.bin`
  under `/tmp`).
- `/kaggle/working/runs/checkpoints`: carried forward from s3 via the mount.
  If s3 still has 3 rolling + ~8 milestones from earlier sessions, that's
  ~11 × (500 MB weights + 1.5 GB optimizer state) ≈ 22 GB — over quota
  when combined with the `/tmp` concat.

**Mitigation before push:** delete older milestone checkpoints from the
s3 output via Kaggle's file browser if the disk is tight. Keep at minimum
step ~28000 (the resume target) and the latest rolling. Disk prunes to
the last 2–3 checkpoints keeps this session comfortably under quota.

If you don't want to prune manually, `PLM_UPLOAD_EVERY` is not the right knob
(it controls GitHub uploads, not local retention). Local pruning on the last
session isn't automated; manual deletion in the Kaggle UI is the safest path
for s4.

## 6. After s4 completes

- [ ] Download `runs/checkpoints/step_00031101.safetensors` from the kernel
      output (or verify it's on the release, if `GH_TOKEN` was set).
- [ ] Run the eval suite against the final checkpoint (HellaSwag, LAMBADA,
      val_loss on `val.bin`) to compare with Stage B's numbers.
- [ ] Post the comparison table in the lounge.
- [ ] Archive the s4 kernel output as the Stage C final artifact.

## 7. If s4 exits early

Rare but possible (e.g. Kaggle kill mid-save, network blip during mount).
Recovery is the same as any other session: push `kernel-metadata-resume.json`
again, it resumes from s4's own last save. No state is lost.

If s4 exits 0 but you want to run more (e.g. group decides to extend past
31,101), set `PLM_STAGE_C_MAX_STEPS` to the new target on the next push. The
resume chain continues.
