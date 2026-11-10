# Stage E — Round 3 status (truthful)

**Updated:** Mon Nov 9, 2026 ~9:55 AM EST  
**Authority:** Stephon approved promotion 9:35 AM ET Nov 9 — swap live Stage D 31104 for `r3-rr50-ep1-lr1e5` (50% replay, lr 1e-5). **D31104 retained for rollback.**

## Promotion (live)

| Item | Value |
|------|-------|
| Live demo | https://playground-lm.vercel.app |
| Live step | **31140** (Stage E Round 3; friendly name **v0.5 Chat**) |
| Weights SHA | `ce1fa2098236656d149437638601dfafa7744ab3f69ff52739be0c95199f818b` |
| ONNX q8 SHA | `0fb5504d0e98e4eeb2032a3c450ffd2b0d996bb11c8d27912be1d79b821fdee8` |
| HF path | `Gteam04/playground-lm-onnx` / `step_00031140.q8.onnx` @ rev `27778a045bee172529170330ada959131037da64` |
| GH release | https://github.com/galactic-ai-team/playground-lm/releases/tag/stage-e-r3-step31140 |
| Runtime | ORT-Web 1.24.3 WASM |
| Rollback | One-line: restore D31104 `onnx_url` `…/resolve/9732b2c8318190b43b5f97c9f7299559efd46b26/step_00031104.q8.onnx` (file still on HF) |

CPU ONNX parity: 2110 positions, top1≈0.935, KL≈0.0045, **6/6** greedy exact vs PyTorch. Browser verification by owner after deploy.


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
**Chosen for promote:** `r3-rr50-ep1-lr1e5` (Stephon 9:35 AM ET Nov 9).

Receipts: `/workspace/plm_stageE/regression_round3/` (+ `COMPARISON.md`, FREEZE.json under each passer).

## Private suite (Round 3 — owner-only)

**Location:** `/workspace/plm_stageE/private_eval_r3/` (chmod 700; never in git/Drive as fixtures).  
**Frozen SHA:** prompts `e65ae055…0dd0`, expect `47f756f1…be58`. **60 cases** (instruction_format 10, reading_context 12, unseen_numeracy 14, model_identity 10, conversation_consistency 14).  
**Self-tests:** 8 passed. **Contamination audit:** exact 0 / 13-gram 0 vs Round-3 SFT train+val, Round-1 private, HellaSwag/LAMBADA samples, FineWeb jsonl sample — **not blocking**.  
**Protocol:** greedy, max_new 48, exact strip+casefold scorer (same class as Round 1); seed_base `202611090913` per AUTHORING.

| model | exact pass / 60 | vs C improved/worsened/same | notes |
|---|---:|---|---|
| C31101 | 0 | — | baseline |
| D31104 | 0 | 0/0/60 | |
| E31128 (R1) | 0 | 0/0/60 | |
| r3-rr25-ep1-lr1e5 | 1 | 1/0/59 | unseen_numeracy +1 |
| r3-rr50-ep1-lr1e5 | 1 | 1/0/59 | unseen_numeracy +1 |
| r3-rr50-ep1-lr3e5 | 1 | 1/0/59 | unseen_numeracy +1 |

**Recommend (not promote):** `r3-rr50-ep1-lr1e5` — best FineWeb Δ among gate-PASS (+0.0072) with equal private exact-score to other R3 cells.  
**Caveats:** N=60; strict exact scorer floors all models (R1 C was also 0/60 on R1 suite); private ≠ promotion evidence. Aggregate-only summary on Drive receipts; **no prompt/output upload**.


## Blockers / next

1. Owner browser check on https://playground-lm.vercel.app (ORT MatMulNBits, CORS, chat).
2. Keep D31104 HF files forever for one-line rollback.
3. Private suite remains owner-only under `/workspace/plm_stageE/private_eval_r3/` (not in git).
