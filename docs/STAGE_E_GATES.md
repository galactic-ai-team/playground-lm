# Stage E review gates (Claude's lane)

All gates run on saved receipts; none trains or promotes anything.

| Gate | Checks | Can PASS? |
|---|---|---|
| G1 checksums | `meta/SHA256SUMS.txt` matches the freeze | yes |
| G2 same-corpus | masked-val receipts for both checkpoints use the same val file and windows | yes |
| G3 masks | SFT audit: 0/1 masks, in-vocab ids, completions end in EOT after a prompt | yes |
| G4 shift | supervised `y[t] == x[t+1]` on real val windows | yes |
| G5 leakage | train/val overlap (exact user = FAIL; exact answer or 13-word overlap = WARN) | yes |
| G6 lr-schedule | logged LR = schedule-local warmup + cosine (`--sweep-run` for round-2 runs) | yes |
| G7 val-claim | masked-val delta is worded as corpus fit; flags batch-weighted or repeated windows | yes |
| **G8 public-dev** | paired exact McNemar on the public development probes | **no: WARN-only** |
| G9 promotion | see below | only with every requirement |

**G8 is WARN-only by design.** The public development probes are frozen and public, not a blind holdout. Their pass rate is descriptive and never counts toward promotion, however large the gain. Don't tune training data against them.

**G9 blocks promotion unless all of these hold:**
1. A valid regression receipt from `evals/regression_receipt.py`: the baseline is the Stage C 31101 seed (by sha256), every model was scored on the same FineWeb-Edu val tokens and the same HellaSwag/LAMBADA examples, and the candidate's overall verdict is PASS.
2. The private blind suite was hash-frozen before scoring.
3. G5 leakage did not FAIL in the same gate run; re-run G5 on every new corpus.
4. Stephon approved (`--approved-by Stephon`).

**Regression verdicts** (`evals/regression_gate.py`, defaults recorded in each receipt):
FineWeb-Edu val loss FAILs on a rise above 0.02. HellaSwag (tolerance 1 point) and LAMBADA (1.5 points) PASS within tolerance, WARN when the drop exceeds tolerance but is within sampling noise (1.96 × standard error of the difference), and FAIL when it exceeds both. The worst metric decides. Only PASS counts for G9.

**Round-2 sweep** (`src/playground_lm/train/sweep_config.py`): epochs {1, 3} × peak LR {3e-5, 1e-5} from Stage C 31101, optimizer reset, schedule restarting at 31101, warmup = a quarter of the run capped at 20. `run_env(run)` gives the launcher its environment; `SweepRun.to_dict()` is the `--sweep-run` file for G6.

**Order for each round-2 candidate:**
1. `masked_val_receipt.py` for C31101 and the candidate (G2/G7).
2. `regression_receipt.py --baseline C31101 --candidate NAME=...` (G9).
3. `stage_e_review_gates.py` with `--corpus` (current corpus), `--metrics`, `--sweep-run`, both receipts, the behavior reports, and `--regression-receipt`.
