# Stage E — Round 3 status (truthful)

**Updated:** Mon Nov 9, 2026 ~7:54 AM America/New_York  
**Authority:** Stephon approved Round 3 (“Yup”, 7:07 AM ET). Goal: mix FineWeb-Edu-style Stage B/C replay into SFT so FineWeb val rise ≤0.02 vs C31101 (Claude gate). **No merge. No live deploy. No promotion. No private suite. Free Kaggle only.**

## Code / PR

| Item | Value |
|------|-------|
| Branch | `review/stage-e-round3` (off PR99 tip) |
| Tip | `a0c1699` |
| Draft PR | https://github.com/galactic-ai-team/playground-lm/pull/100 (refs #99) |
| CI | **green** — `ci/test` SUCCESS, notebook freshness SUCCESS, Vercel SUCCESS |
| Local suites (spot) | replay+sweep **26 passed**; full suite previously **553 pytest + 27 Node** on tip before path fix (CI re-green after `a0c1699`) |
| Disk | Box ~77% after scratch cleanup; **no** `runs_round2` delete (awaits Stephon). No local multi-GB ckpt/data downloads. |

## Per-member accept / change / reject

| Member | Accepted | Changed (Lumi glue) | Rejected |
|--------|----------|---------------------|----------|
| **Nova** | Offline `mix_stage_e_replay` corpus mixer + tests (audited offline path) | Kept alongside Claude; **not** primary Kaggle train path | — |
| **Claude** | Online `ReplayMixer` (`PLM_REPLAY_RATIO`+`PLM_REPLAY_BIN`), `sweep_config` Round-3 grid, `STAGE_E_GATES.md` FineWeb rise ≤0.02 | **Primary train path** on Kaggle | — |
| **GPT-6** | Kaggle regression package + receipt bridge → Claude gate (25 tests) | — | Private suite **not** run |
| **Dee** | Replay-aware `kaggle/sweep` packer/launcher | Placeholders → real `galacticaiteam/plm-stageb-data`; file-top `os.environ` inject; `DEFAULT_REPLAY_BIN` → `/kaggle/input/datasets/galacticaiteam/plm-stageb-data/train.bin` | Invented PLACEHOLDER dataset slugs (fixed before launch) |
| **Gemini** | `statusAfterGeneration` clears stuck “Stopping generation…” | Wired into `demo/app.js` `finally` + Node tests | — |

**Mixer reconcile:** Claude online mixer = train; Nova offline mixer retained for audited corpus builds. Both credited.

## Replay source

| Item | Value |
|------|-------|
| Dataset | `galacticaiteam/plm-stageb-data` (private; `train.bin` ~516MB FineWeb-style) |
| Val (optional mount) | `galacticaiteam/plm-stagec-val` |
| SFT data | `galacticaiteam/plm-stage-e-data` |
| Seed ckpt | Stage C step **31101** via `kernel_sources: galacticaiteam/plm-stage-c-train` (SHA `4604d3a1…76275`) |
| Mount layout | Probe proved private datasets appear under `/kaggle/input/datasets/<owner>/<slug>/` (not short `/kaggle/input/<slug>/`) — https://www.kaggle.com/code/galacticaiteam/plm-stage-e-r3-mount-probe |

## Sweep plan (free T4 ×2 concurrency)

| Cell | replay | lr | epochs | max_steps | Kernel URL |
|------|-------:|---:|-------:|----------:|------------|
| rr25 @ 1e-5 | 0.25 | 1e-5 | 1 | 31140 | https://www.kaggle.com/code/galacticaiteam/plm-stage-e-r3-rr25-ep1-lr1e5 |
| rr50 @ 1e-5 | 0.50 | 1e-5 | 1 | 31140 | https://www.kaggle.com/code/galacticaiteam/plm-stage-e-r3-rr50-ep1-lr1e5 |
| rr50 @ 3e-5 | 0.50 | 3e-5 | 1 | 31140 | https://www.kaggle.com/code/galacticaiteam/plm-stage-e-r3-rr50-ep1-lr3e5 |

Packs: `/workspace/plm_stageE/sweep_round3_pack/` (pinned `PLM_GIT_COMMIT=a0c1699`, correct `datasets/` replay path + `resolve_replay_bin`).

### Launch status (this morning ET)

| Event | Detail |
|-------|--------|
| v1 push | Ran; **ERROR** — hardcoded short-path `PLM_REPLAY_BIN` missing (`FileNotFoundError: …/plm-stageb-data/train.bin`). Commit pin was `336b42f` (pre-resolve). |
| Mount probe (CPU) | COMPLETE — confirmed `…/datasets/galacticaiteam/plm-stageb-data/train.bin` |
| Path fix | Commit `a0c1699` + rebuilt packs |
| v2/v3 re-push | **Blocked:** `Maximum batch GPU session count of 2 reached` even while all three show `ERROR`. ListKernelSessions returns no active sessions (stale counter / cooldown). Poller retrying. |
| Outputs policy | Keep final ckpts as **Kaggle kernel outputs** only; download **small** metrics/receipts JSON only |

## Regression / Claude gate

| Item | Value |
|------|-------|
| Subsets | FineWeb **20** batches; HellaSwag/LAMBADA **200**; seed **20261108** (identical Round 2) |
| Baseline | C31101 receipts reference |
| Gate | FineWeb rise ≤ **0.02**; report table C vs candidates incl masked SFT val |
| Status | **Not run yet** — blocked on successful sweep COMPLETE + small JSON receipts |
| Private suite | **Not run** (per Stephon) |
| Promote | **No** |

## Round-2 baseline (for gate comparison)

All Round-2 candidates FAIL FineWeb gate (closest ep1-lr1e5 +0.0287). See `STAGE_E_ROUND2.md`.

## Blockers / next

1. **GPU concurrency ghost slots** — cannot push fixed packs until Kaggle frees the batch GPU session counter (or Stephon cancels sessions in the web UI).
2. After slots free: push three cells → wait COMPLETE → download **only** small metrics/receipts → run GPT-6 Kaggle regression → Claude gate table.
3. Draft PR #100 stays unmerged; no live change; no `runs_round2` local delete until Stephon says yes.
