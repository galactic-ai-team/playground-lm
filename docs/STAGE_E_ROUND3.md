# Stage E — Round 3 status (truthful)

**Updated:** Mon Nov 9, 2026 ~9:11 AM EST  
**Authority:** Stephon approved Round 3 (“Yup”, 7:07 AM ET). Goal: FineWeb-Edu-style replay mix so FineWeb val rise ≤0.02 vs C31101. **No merge. No live deploy. No promotion. No private suite. Free compute only.**

## Code / PR

| Item | Value |
|------|-------|
| Branch | `review/stage-e-round3` |
| Tip (train pin) | `a0c1699` (replay path fix) |
| Draft PR | https://github.com/galactic-ai-team/playground-lm/pull/100 (refs #99) |
| CI | green on tip pushes |
| Mixer | Claude **online** ReplayMixer = train path; Nova **offline** mixer kept for audited corpora |

## Per-member accept / change / reject

| Member | Accepted | Changed (Lumi glue) | Rejected |
|--------|----------|---------------------|----------|
| **Nova** | Offline `mix_stage_e_replay` | Kept; not Kaggle train path | — |
| **Claude** | Online ReplayMixer + sweep/gates | Primary train path | — |
| **GPT-6** | Kaggle regression + bridge | Local CPU regression used (GPU slots blocked); identical Round-2 knobs | Private suite not run |
| **Dee** | Replay-aware packer | PLACEHOLDERs → `plm-stageb-data`; file-top env; `datasets/` mount path | PLACEHOLDER slugs |
| **Gemini** | `statusAfterGeneration` | Wired into `app.js` + Node tests | — |

## Replay corpus

| Item | Value |
|------|-------|
| Chosen | `galacticaiteam/plm-stageb-data` |
| Provenance | `fineweb-edu` / `sample-10BT` (tokenize-stats.json) |
| Stage C | Continued FineWeb via `plm-stage-c-data-prep-s0..s7` kernel shards — **no** single Stage-C train.bin dataset on Kaggle |
| Mount path | `/kaggle/input/datasets/galacticaiteam/plm-stageb-data/train.bin` |

## Sweep (free T4)

| Cell | replay | lr | Kernel | Final step | Weights SHA256 | Status |
|------|-------:|---:|--------|------------:|----------------|--------|
| rr25 @ 1e-5 | 0.25 | 1e-5 | https://www.kaggle.com/code/galacticaiteam/plm-stage-e-r3-rr25-ep1-lr1e5 | 31140 | `da0abc389e9ec3e186726940be482cb9d9e858336395875964b24d027ae2b7a0` | COMPLETE + frozen |
| rr50 @ 1e-5 | 0.50 | 1e-5 | https://www.kaggle.com/code/galacticaiteam/plm-stage-e-r3-rr50-ep1-lr1e5 | 31140 | `ce1fa2098236656d149437638601dfafa7744ab3f69ff52739be0c95199f818b` | COMPLETE + frozen |
| rr50 @ 3e-5 | 0.50 | 3e-5 | https://www.kaggle.com/code/galacticaiteam/plm-stage-e-r3-rr50-ep1-lr3e5 | 31140 | `bad27ad68038ad23ba2f10f5cbf2d670b2bea7ec33a96e74e485d1a1a86faf7c` | COMPLETE + frozen |

Seed C31101 SHA `4604d3a1b105cc70601352fbd83d6d61b34435b3cf09151b0562592a7b076275`. Final ckpts also retained as Kaggle kernel outputs.

## Regression (public subsets — nonpublishable)

Same knobs as Round 2: `max_val_batches=20`, `benchmark_limit=200`, seed `20261108`, Stage-C FineWeb `val.bin` SHA `1fe7dffe…a9e7`. Ran **local CPU** (Kaggle GPU session counter stuck; free-tier).

| model | FineWeb val↓ | Δ vs C | HellaSwag↑ | LAMBADA↑ | masked SFT val↓ | Claude gate |
|---|---:|---:|---:|---:|---:|---|
| C | 3.0520 | — | 0.370 | 0.240 | 3.1029 | baseline |
| r3-rr25-ep1-lr1e5 | 3.0670 | +0.0150 | 0.390 | 0.265 | 2.8544 | **PASS** |
| r3-rr50-ep1-lr1e5 | 3.0592 | +0.0072 | 0.390 | 0.260 | 2.8685 | **PASS** |
| r3-rr50-ep1-lr3e5 | 3.0603 | +0.0083 | 0.385 | 0.245 | 2.8306 | **PASS** |

Tolerances: FineWeb rise ≤0.02; HS drop ≤0.01; LAMBADA drop ≤0.015.  
**Best FineWeb Δ:** r3-rr50-ep1-lr1e5 (+0.0072). **Best masked SFT:** r3-rr50-ep1-lr3e5 (2.8306).  
**Chosen for promote: none** — all PASS gate but **Stephon must choose**; no auto-promote.

Receipts: `/workspace/plm_stageE/regression_round3/` (+ `COMPARISON.md`, FREEZE.json under each passer).

## Private suite

**Not run.**

## Blockers / next

1. Stephon chooses among three PASS candidates (or requests more epochs) before any promote.
2. Draft PR #100 stays unmerged.
3. Fresh private suite still needed after freeze choice.
