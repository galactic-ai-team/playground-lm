# Stage G Track 2 — measured throughput and free-quota estimate

Measured Tue Nov 10, 2026 (~8:55–9:00 AM ET) on free Kaggle **T4 x2** with
`kaggle/stage_g_bench/stage_g_bench.py` (design: Dee; DDP harness: Lumi). Raw receipt:
`evals/results/stage_g/throughput_bench_v1.json`. Real DDP over 2 ranks, fp16 autocast +
GradScaler, AdamW, grad-accum 8, context 1024, 3 warm-up + 10 timed optimizer steps,
**random tokens** (no data loader). Total bench runtime ~6 min; no OOMs.

| Config | Params | Micro-batch / GPU | s/step | tok/s (bench) | Peak GiB/GPU |
|---|---|---|---|---|---|
| 124M control 12×768 | 124.4M | 6 | 3.92 | 25,060 | 6.81 |
| G-depth 24×768 | 209.5M | 2 | 2.32 | 14,099 | 6.98 |
| G-depth 24×768 | 209.5M | 4 | 4.64 | 14,138 | 9.59 |
| G-full 24×1024 (16 heads) | 354.8M | 1 | 2.17 | 7,549 | 8.81 |
| G-full 24×1024 (16 heads) | 354.8M | 2 | 3.66 | 8,964 | 10.33 |
| G-full 24×1024 (16 heads) | 354.8M | 3 | 5.23 | **9,406** | 11.57 |

**Calibration.** The same 124M setting measured ~22k tok/s in real Stage C training
(4.46 s/step) versus 25.1k here, so real training runs at about **0.88×** the bench
(data loading, eval, checkpoints). Adjusted planning rates:

| Config | Planning tok/s | Tokens per session-hour | 1.0B | 1.5B | 2.86B | 7.1B (20 tok/param) |
|---|---|---|---|---|---|---|
| G-depth 24×768 (mb 4) | ~12.4k | ~45M | ~22 h | ~34 h | ~64 h | ~160 h |
| G-full 24×1024 (mb 3) | ~8.3k | ~30M | ~34 h | ~50 h | ~96 h | ~240 h |

Kaggle free quota: 30 GPU-h/week shared pool (T4 x2 costs 1 h per hour), 12 h sessions
(we use 11 h), max 2 concurrent sessions from the same pool; resets Sat 00:00 UTC
(Fri 7 PM ET).

## Verdict (updated with measurements)

- **From scratch, 355M still doesn't fit free in a sensible time**: the 124M's data budget
  (2.86B tokens) is ~96 h ≈ 3.2 weeks of the *entire* quota; compute-optimal is ~8 weeks.
  Better than the 6.5k tok/s planning estimate (~125 h), same conclusion.
- **Grown from C31101 (free path):** G-full with 1.0–1.5B continued tokens ≈ **34–50 h ≈
  1.2–1.7 weeks** of full quota; G-depth with 1.5B ≈ **34 h ≈ 1.1 weeks**.
- Memory is fine: G-full at micro-batch 3 peaks at 11.6 GiB/GPU (15 GiB T4); use
  micro-batch 3 × accum 16 ≈ 98k tokens/step (matches Stage C's step size).
- Checkpoints: 355M ≈ 1.4 GB weights + ≈ 2.8 GB AdamW state ≈ 4.3 GB; keep 1 rolling +
  1 milestone per session (< 9 GB of 20 GB). Each 11 h session ≈ 330M tokens (G-full);
  1.5B tokens ≈ 5 chained sessions, each resuming from the previous kernel's output.
- Growth operators: `src/playground_lm/grow.py` — exact identity depth growth (tested),
  G_stack whole-model stacking, and an approximate width growth 768→1024 (LayerNorm
  statistics change; drift measured in tests, not claimed exact).
- Nothing paid is needed or proposed.
