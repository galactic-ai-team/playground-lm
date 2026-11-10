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

## Round 3: hold FineWeb-Edu within the gate with replay

**Goal:** every round-2 cell raised FineWeb-Edu val loss by more than the regression gate's 0.02 against C31101. Round 3 tries to stay within that gate by mixing Stage C pretraining rows into the SFT batches, at a gentle learning rate. The gate itself is unchanged: FineWeb val rise ≤ 0.02, HellaSwag drop ≤ 1 point, LAMBADA drop ≤ 1.5 points, with the noise rule above.

**Grid** (`build_sweep_round3`): replay_ratio {0.25, 0.5} × peak LR 1e-5 × 1 SFT epoch, from Stage C 31101 only, optimizer reset, schedule restarting at 31101. An optional third cell, 0.5 × 3e-5, runs only if quota allows (`include_optional=True`).

**What replay_ratio means:** the share of rows (1024-token windows) drawn from Stage C pretraining data (`replay_mix.py`). Each rank's optimizer step gets exactly that share (3 of 12 rows at 0.25, 6 of 12 at 0.5). Replay rows supervise every token and SFT rows only their completions, so replay's share of the loss is larger than its share of rows. "Epochs" counts passes over the SFT data, so with replay one epoch takes more updates (with round 1's corpus size: 35 at 0.25, 53 at 0.5).

**Receipts per round-3 candidate:**
- masked SFT val receipt on the leak-free round-2 val split (instruction half), for C31101 and the candidate (G2/G7);
- regression receipt on Stage C `val.bin`, same subsets for every model (G9);
- G6 with the run's `--sweep-run` file;
- G5 on the round-3 training corpus.

G9 still blocks without a PASS regression verdict, the frozen blind suite and Stephon's approval.

## Stage F: facts + simple math

**The regression gate is still against C31101,** with unchanged tolerances (FineWeb-Edu val rise ≤ 0.02, HellaSwag drop ≤ 1 point, LAMBADA drop ≤ 1.5 points, same noise rule, worst metric decides). Live v0.5 Chat (E31140) already sits at +0.007 on the round-3 subset, so a Stage F run seeded from E31140 has about 0.013 of headroom. That's why every cell replays FineWeb.

**Promotion also requires a clear private-suite win over E31140 (G10),** on top of G9:
- a **fresh** private suite, hash-frozen before scoring (pass previously used suite hashes with `--used-suite-sha`);
- candidate and E31140 scored on the same items; G10 PASSes only if exact paired McNemar p < 0.05 **and** net cases (gained − lost) are ≥ **+5**. A 1-item bump is not a win; net +5 with p ≥ 0.05 is not a win either. (`--private-alpha`, `--private-min-net`).
- This is Stephon's confirmed Stage F bar (2026-11-09), not only a p-value reading.

The CLI's overall `promotion` is PASS only when both G9 and G10 pass.

**Grid** (`build_sweep_stage_f`), all at peak LR 1e-5, one epoch over the new fact/arith data, optimizer reset, schedule restarting at the seed's step:

| Cell | Seed | Rows per 24-row update: facts / chat / FineWeb replay |
|---|---|---|
| primary | E31140 | 8 / 4 / 12 (replay 0.50, chat 1/6) |
| second | E31140 | 12 / 6 / 6 (replay 0.25, chat 0.25) |
| control (optional) | C31101 | 8 / 4 / 12 |

Each rank's optimizer step gets exact whole rows from each source (4/2/6 or 6/3/3 of 12). Fact/arith and chat rows keep completion-only masks; replay rows use full loss. The chat share keeps v0.5 Chat's behaviour from fading while it learns the new facts.

**Regression subsets:** reuse the round-3 knobs (`max_val_batches=20`, `benchmark_limit=200`, seed 20261108, the same Stage C `val.bin`) for every model so the comparison stays like for like. `--regression-checkpoint-receipt NAME=path` combines the per-checkpoint receipts and refuses mismatched settings. **Caveat:** at 200 examples the noise rule can only FAIL HellaSwag or LAMBADA drops of about 9 to 10 points, so in practice those two metrics can only WARN. Use `benchmark_limit=200` for every sweep candidate. For a final promotion candidate only, rerun C31101, E31140 and the finalist on the same set with `benchmark_limit=1000` (detectable drop about 4 points). That changes the sample size, not the thresholds, and still needs Stephon's go-ahead before any promotion.
