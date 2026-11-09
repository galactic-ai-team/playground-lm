# Stage E round-3 replay-mix sweep — launch runbook

**Do not push kernels until Lumi or Stephon says "push-to-run".**
Approved 2026-11-09 ~07:07 ET by Stephon. This is the operator's runbook.

## 1. Goal and gate

Mix Stage C pretraining-style data into the Stage E SFT pass so general skill
does not regress. Claude's regression gate: **FineWeb val rise ≤ 0.02 vs
C31101**. The sweep varies `replay_ratio × lr × epochs` and each cell must
record both the SFT val and a FineWeb val to evaluate the gate.

## 2. Preconditions

- [ ] `galacticaiteam/plm-stage-e-data` mounted (Round-2 version accepted).
- [ ] Stage B FineWeb-style train.bin mounted as replay (`plm-stageb-data`); slugs updated in
      `pack_cell.REPLAY_DATASET_SLUGS`. **Do not push until these are real
      slugs** — the current placeholders will 404 at mount time.
- [ ] `galacticaiteam/plm-stage-c-train` mount has step_00031101 weights and
      `_state.pt`. Do not re-upload Stage C.
- [ ] `D31104` live production weights untouched (this sweep is Stage E only).
- [ ] Box disk: **no local downloads** — the sweep mounts Kaggle data and
      writes only small artifacts (final safetensors + model_config +
      metrics.jsonl + log) locally.

## 3. Dataset sources (fill in verified slugs)

| Purpose | `dataset_sources` entry | Notes |
|---|---|---|
| SFT + masks | `galacticaiteam/plm-stage-e-data` | Round-2 version OK |
| Stage C replay shards | `galacticaiteam/plm-stageb-data` | verified 2026-11-09 (FineWeb-style train.bin) |
| Stage B replay shards | `galacticaiteam/plm-stageb-data` | verified 2026-11-09 (FineWeb-style train.bin) |

## 4. Kernel sources

| Purpose | `kernel_sources` entry |
|---|---|
| Seed weights (step 31101) | `galacticaiteam/plm-stage-c-train` |

## 5. Package cells

## 6. Dry-run the orchestrator

Expected: one `pending` line per cell with lr/ep/replay/commit, exit 0,
zero `kaggle kernels push` calls.

## 7. Execute (only after push-to-run)

Concurrency: **max 2 concurrent GPU sessions** on the free tier. Cells beyond
that are marked `queued-concurrency` and must be pushed in a follow-up
invocation after an earlier cell finishes.

## 8. GPU-hours estimate

- Cells: `lr × epochs × replay = 2 × 2 × 2 = 8`.
- Each cell: `max_steps=31228`, resume from 31101 → ~127 updates.
- Per-cell wall time on 2×T4 at ~4.4 s/step: **~9.3 minutes**.
- Per-cell GPU-hours (2×T4 bills 1× wall per Kaggle quota rules): **~0.16 h**.
- **Full sweep: ~1.3 GPU-hours**, well within the 30 h/week quota, and fits
  easily inside the "2 concurrent, queue the rest" cap.

## 9. Durable outputs

- Kaggle kernel output is the origin. Local freeze is derived.
- Per-cell artifacts kept: `final.safetensors`, `model_config.json`,
  `metrics.jsonl`, `run.log`. **Do not** upload full optimizer state — the
  packet requires no optimizer-state dependency in the receipt.
- `COMPLETE.json` in `runs/<run_id>/` is a **local skip marker** written by
  the operator after verifying the cell, not a Kaggle API field.
- Never `rm -rf` a run directory. If a cell is abandoned, rename to
  `<run_id>.abandoned-<timestamp>`.

## 10. Recovery authority

Same pattern as `candidate_ep1_frozen/meta/RECEIPT_AUTHORITY.md`:
Kaggle kernel version N is the origin. If local bytes drift, re-download
from Kaggle and re-verify with `tools/verify_freeze_receipt.py`.

## 11. Standing rules

- Free Kaggle GPU only (T4). No paid resources.
- Preserve `C31101` seed; never overwrite with a sweep result.
- Do not touch `D31104` live production weights.
- No promotion, no merge, no public weight upload.
- No private eval content in this document.
- No local multi-GB downloads (box is ~98% full).