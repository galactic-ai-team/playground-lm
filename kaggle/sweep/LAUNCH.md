# Stage E round-2 sweep — launch runbook

**Do not start this sweep until Stephon or Lumi says "push-to-run".**
This document is the operator's checklist, not an invitation.

## 1. Preconditions

- [ ] `review/stage-e-sft-prep` at the pinned commit is on main, or the sweep
      pins to the branch.
- [ ] `galacticaiteam/plm-stage-e-data` is the private dataset in use, with
      the four expected files (`train.bin`, `train.masks.bin`, `val.bin`,
      `val.masks.bin`) and the round-2 hashes from the packet §3.
- [ ] `galacticaiteam/plm-stage-c-train` mount still has step_00031101
      weights + `_state.pt`. The seed must not change; if the Stage C mount
      has been reuploaded, stop and re-verify.
- [ ] Live `D31104` production weights are untouched (sweep is Stage E only).
- [ ] Claude has finalized the LR × epochs grid for round 2.

## 2. Package cells (dry-run only)

For each grid point, run `pack_cell.py` once. Each cell gets its own folder
with a slug-named script. Do not reuse folder names.
