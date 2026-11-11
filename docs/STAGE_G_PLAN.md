# Stage G plan — step-by-step math, a built-in fact sheet, and a ~350M feasibility check

**Status:** plan + setup only. No long training started. No promotion. Nothing deleted. Free only.
**Approved direction:** Stephon, Tue Nov 10, 2026 (morning, ET) — "Do both, math steps and fact sheet plus the 350M model."
**Written:** Tue Nov 10, 2026, ~7:45 AM ET (America/New_York).
**Live (unchanged):** PlaygroundLM **v0.5 Chat** = Stage E Round 3 step **31140** (`r3-rr50-ep1-lr1e5`), weights SHA `ce1fa2098236656d149437638601dfafa7744ab3f69ff52739be0c95199f818b`, ONNX q8 `0fb5504d…821fdee8`. It stays the rollback.
**Code branch:** `feat/stage-g-scratchpad` (stacked on Stage F draft PR #103, branch `feat/stage-f-facts-arith`).

## Why Stage G is different from Stage F

Stage F trained four candidates (F31204, F31236, F31274, F31340). All passed the FineWeb gate and G5, and all scored **2/72** on the Stage F private suite, the same as E31140 (arithmetic 0/34, facts 0/26, McNemar p=1.0, net 0). More instruction mixing on short "question → bare answer" rows does not teach a 124M model to compute or to recall facts. Stage G changes **what** the model is asked to do:

1. **Math:** write the work out digit by digit (a scratchpad), then a fixed `Answer: <int>` line. Each step is a single-digit operation the model can learn; the final answer is assembled from those steps instead of guessed in one token.
2. **Facts:** stop asking the model to memorise. The demo looks the answer up in a small, license-clean fact sheet (keyword/BM25 in the browser) and puts the top facts into the prompt. SFT teaches the model to answer **from the provided facts**, and to say it is not sure when the facts don't contain the answer.
3. **Scale:** honestly assess a ~350M model on free compute (Track 2 below). Bigger alone will not fix math or facts at this size; it is a later multiplier, not the main lever.

## Track 1 — 124M Stage G

### 1a. Scratchpad arithmetic (MIT, generated in this repo)

- Generator: `data/gen_scratchpad_arith.py` (+ `data/test_gen_scratchpad_arith.py`, 23 tests) on `feat/stage-g-scratchpad`.
- Ops: add / sub (borrowing, negative results) up to 4 digits; mul k-digit × 1-digit (column + carry) and ≤2-digit × 2-digit (partial products + column add); exact long division by 2–9.
- Fixed one-line format (matches the demo's `User:/Assistant:` and Stage E `format_example`):
  `Work: Ones: 7 + 5 + 0 = 12, write 2, carry 1. Tens: 4 + 8 + 1 = 13, write 3, carry 1. Hundreds: carry 1, write 1. Digits from the top place down: 1 3 2. Answer: 132`
- Every step equation and every final answer is machine-checked (`verify_steps`, `check_row`) before a row is written.
- Split by problem (`op:a:b` hash), never by template, so a problem can't be in train and val under different phrasings. `--exclude-keys` drops any owner-only private-suite problems (keys file stays in the owner-only dir, never in git).
- Local v0 dry build (not committed, not yet de-duplicated against private suites): 20,000 rows → 19,054 train / 946 val, ~1.85M tokens, median 82 tokens, p95 240, max 249 per row. Files in `/workspace/plm_stageG/scratchpad_v0/`.
- GPT-6 reviews the format; Claude decides mix share. Word problems (GSM8K-style with scratchpad) are a v1 extension only after v0 shows any movement.

### 1b. Built-in fact sheet + context-grounded SFT

- **Fact sheet** (Nova): `data/facts/fact_sheet.jsonl`, target 400–800 entries for v1. Each entry: `id`, `text` (one self-written sentence), `keywords`, `aliases` (accepted answer forms), `topic`, `verify_url` (where Nova checked the fact; text is **not** copied), `license` (`MIT (this repository)` for self-written; CC0/CC-BY only with attribution in `data/facts/ATTRIBUTION.md`). No NC, no share-alike unless Stephon says yes, no scraped paragraphs. Topics: geography (capitals, continents, oceans, largest/longest), basic science (planets, elements, units, body), calendar/time, simple history dates, animals, language basics, PlaygroundLM self-facts.
- Frozen by SHA once v1 is done. 20% of entries are **held out** from SFT, so we can test reading from context on facts the model never trained on.
- **Retrieval** (Gemini, demo JS): BM25 over `text + keywords + aliases`, top-3, minimum score threshold, only for the newest user turn, ≤160 tokens of facts. Deterministic and unit-tested in Node.
- **Prompt format** (fixed, used by both demo and SFT):
  `User: Facts: [1] Paris is the capital of France. [2] … [3] … Question: What is the capital of France?`
  `Assistant: Paris.`
  The UI still shows the user's original text; the facts are added only to the model prompt and listed as source chips.
- **Grounded SFT** (Claude builder, Nova data): from the non-held-out facts, several question phrasings each, with 2 distractor facts per row, shuffled positions; ~25% rows where none of the given facts answer the question → `I'm not sure; the facts I was given don't say.`; some rows with no `Facts:` at all (closed-book) to keep v0.5 chat behaviour.

### 1c. Calculator fallback (demo only, clearly labelled)

Prefer the model doing the work. The demo may show a separate chip **"Calculator check (not the model)"** for prompts that parse as a single `a op b`, with ✓/✗ against the model's `Answer:`. It never replaces or edits the model's text, it's off in scoring, and it's documented in the README/model card.

### 1d. Track 1 training (free Kaggle T4 ×2, ≤8 GPU-hours total)

- Seed **E31140**; completion-only masks on SFT, full loss on FineWeb replay; `PLM_RESET_OPTIMIZER=1`; schedule restart at the seed step. Reuse the Stage F three-way mixer (`src/playground_lm/train/replay_mix.py`) extended to four sources.
- Draft grid for Claude to finalise (≤3 runs): shares scratchpad / grounded facts / chat SFT / FineWeb replay =
  - G1: 25 / 15 / 10 / 50, lr 1e-5, 2 scratchpad epochs
  - G2: 25 / 15 / 10 / 50, lr 3e-5 (scratchpad is a new skill; replay protects FineWeb)
  - G3 (optional): 35 / 20 / 10 / 35, lr 1e-5
- Kaggle disk: 124M checkpoint ≈1.5 GB (weights + state); keep the Stage F milestone spacing.

## Track 2 — ~350M (GPT-2-medium size): honest free-compute verdict

### Numbers

> **Measured Nov 10 (~9 AM ET):** see `docs/STAGE_G_350M_ESTIMATE.md`. G-full 24×1024 ran 9.4k tok/s at micro-batch 3 (≈8.3k real-training, ≈30M tokens/session-hour); G-depth 24×768 14.1k (≈12.4k real). The verdict below holds: from scratch ~96 h for 2.86B tokens; grown 1.0–1.5B ≈ 34–50 h.

| Item | Value | Basis |
|---|---|---|
| Measured 124M speed | **~22k tokens/s** on Kaggle T4 ×2 (DDP, fp16 AMP, micro-batch 6 × accum 8, 98,304 tok/step, 4.46 s/step) | Stage B/C logs (Nov 3) |
| 350M config | 24 layers, d=1024, 16 heads, ctx 1024, tied 50,257 vocab ≈ 355M params (302M non-embedding) | GPT-2 medium |
| Compute per token vs 124M | ≈ **2.8×** (6·N incl. LM head + attention) | arithmetic |
| Expected 350M speed | ideal ~7.7k tok/s; **realistic 5.5–7.5k tok/s** (micro-batch 2–3, more accum); plan on **~6.5k tok/s ≈ 23M tokens per session-hour** | estimate; Dee measures |
| T4 memory | weights+grads+AdamW ≈ 5.7 GB/GPU + ~2 GB activations per 1024-token sequence → micro-batch 2–3 fits 15 GB without activation checkpointing | estimate from Stage C 6.81 GiB at mb 6 |
| Kaggle quota | 30 GPU-h/week shared pool (T4 ×2 costs 1 h per hour); 12 h per session (we use 11 h); max 2 concurrent sessions, both from the same pool; resets Sat 00:00 UTC = **Fri 7 PM ET** | Kaggle docs/staff |
| One 11 h session | ≈ 250M tokens | 6.5k × 39,600 s |
| One full week of quota | ≈ 700M tokens (if Track 1 used nothing) | 30 h × 23M |
| 350M checkpoint | ≈ 1.4 GB weights + ≈ 2.9 GB optimizer state ≈ **4.3 GB**; keep 1 rolling + 1 milestone (~8.6 GB of 20 GB `/kaggle/working`) | 16 B/param |

### Token budgets → hours

| Budget | Tokens | Hours at ~23M/h | Weeks of the *whole* free quota |
|---|---|---|---|
| Compute-optimal (20 tokens/param) | 7.1B | ~300 h | **~10 weeks** |
| Same data as our 124M Stage C | 2.86B | ~125 h | **~4+ weeks** (≈12 sessions) |
| Minimum that's worth a demo, from scratch | 1.5B | ~65 h | ~2+ weeks (and under-trained) |
| **Grown from 124M**, continued training | 1.0–1.5B | ~45–65 h | **~1.5–2 weeks** |

### Verdict

- **From scratch, a sensible 350M does not fit free in a reasonable time.** Even matching the 124M's data takes the entire free Kaggle quota for 4+ weeks with nothing left for Track 1 or evals; compute-optimal is ~10 weeks. Free Colab doesn't change this: its limits are unpublished, it's usually one T4 (~3k tok/s for 350M), and long runs get disconnected. Use Colab for evals/exports only.
- **Growing from our 124M is the only free path worth trying.** Depth growth (stack 12 → 24 layers) is the best-supported method in published work (e.g. "Stacking Your Transformers", Du et al., 2024 reports large token savings vs. from-scratch); width growth 768 → 1024 needs a function-preserving widen (Net2Net-style) and is less proven at a non-integer ratio. Two options for Dee's benchmark:
  - **G-depth:** 24 L × 768 ≈ 210M params, ~1.7× the 124M compute (~13k tok/s, ~47M tok/h). 1.5B tokens ≈ 32 h ≈ 1 week+ of quota. Lowest risk.
  - **G-full:** 24 L × 1024 ≈ 355M (true GPT-2-medium size), ~6.5k tok/s. 1.0–1.5B tokens ≈ 45–65 h ≈ 1.5–2 weeks of quota.
  - Grow from **C31101** (the pretrain base, not the chat model), then redo the Stage E/G SFT on top.
- **What it will and won't buy:** a 350M model should lower FineWeb loss and help writing quality a bit. It won't make closed-book facts or arithmetic reliable on its own; scratchpad + fact sheet stay the main levers and transfer to the bigger model.
- **Demo cost:** 350M q8 ONNX ≈ 360 MB download (about 3× today) and ~3× slower in-browser. Phones may struggle; it might ship as an optional "bigger model" switch.
- **Nothing paid is needed.** A paid GPU (e.g. a single A100 for a day) would be what makes from-scratch 350M practical. That's Stephon's call, not part of this plan.

### Track 2 steps (all free)

1. Dee: throughput benchmark kernel, ≤1.5 GPU-h, measures tok/s and peak memory for 124M (control), G-depth, and G-full at micro-batch 1/2/3, 50 steps each on existing Stage C shards. Also the grow-from-C31101 script (depth stack + optional widen) with an equivalence test (grown model's loss on a fixed batch within tolerance of C31101 for the function-preserving variant).
2. If G-full ≥ 6k tok/s and fits memory → plan G-full; else G-depth. Lumi writes the session-chaining plan (each session resumes from the previous kernel output; checkpoint every ~30 min; upload to HF only if a Kaggle Secret is available — otherwise kernel outputs chain like Stage C).
3. Track 2 training starts only after Track 1 runs are scored (quota), earliest week of Nov 14 (after Fri 7 PM ET reset).

## Eval bar (unchanged in substance, pre-registered for Stage G)

- **Suites:** (1) the frozen Stage F private suite (72 items, prompts SHA `30b9127f…1cfa`), and (2) a **new fresh private suite** GPT-6 writes for Stage G (arithmetic of the shapes above but unseen numbers, in-fact-sheet questions including held-out facts, out-of-sheet questions, chat-form, uncertainty). Owner-only dir, mode 700; never in git/Drive/chats beyond GPT-6's delivery. Frozen by SHA **before** any Stage G candidate is scored.
- **Scoring (pre-registered before any candidate is run, applied identically to E31140 and candidates):**
  - S1 strict: the unchanged Stage F scorer (greedy, max_new 48, whole-output exact match). Reported for continuity.
  - S2 Stage G primary: greedy, max_new 256; if the output ends with `Answer: X`, score X with the same exact/alias rules; otherwise score the whole output. Fact-sheet items scored in the **demo pipeline** (same retriever + fact sheet SHA) and closed-book.
  - Report E31140 + retrieval separately, so we know whether gains come from the fact sheet alone (that could ship with no weight change — still Stephon's yes).
- **Gates:** FineWeb val-loss rise ≤ **+0.02** vs C31101; HellaSwag drop ≤ 1 pt; LAMBADA drop ≤ 1.5 pt (200-item for candidates, 1000-item only for the finalist); **G5 leakage** not FAIL on every new corpus (against both private suites and the fact-sheet held-out set).
- **Win bar (G10):** on the fresh Stage G suite, exact McNemar **p < 0.05 AND net (gained − lost) ≥ +5 vs E31140** (same pipeline). On the Stage F suite: no net loss vs E31140 (net ≥ 0), reported.
- **Promotion:** only with **Stephon's explicit yes**, even if every gate passes. Keep E31140 as rollback (D31104 one step further back). Name would be decided by Stephon (e.g. v0.6).

## Team lanes (packets in `/workspace/plm_stageG/packets/`)

| Teammate | Lane | Deliverable |
|---|---|---|
| GPT-6 | Fresh Stage G private suite + S2 scorer; review of the scratchpad generator | owner-only suite ZIP; `stage-g-gpt6.patch` (scorer + tests; generator review notes) |
| Claude | Four-source mixer + grounded-SFT builder + optional answer-token loss weight; sweep grid | `stage-g-claude.patch` |
| Nova | Fact sheet v1 + license audit + held-out split; backup coordinator | `stage-g-facts.patch` (`data/facts/…`) |
| Dee | Stage G Kaggle launcher; 350M throughput benchmark; grow-from-C31101 script | `stage-g-dee.patch` / files |
| Gemini | Demo BM25 retrieval, fact-sheet-in-prompt, Answer/source/calculator chips, max-new for math | `stage-g-gemini.patch` |

## Timeline (ET)

- **Tue Nov 10:** plan, packets, scratchpad generator draft PR (done). Brief the team.
- **Wed Nov 11:** lane deliverables. Dee's benchmark run (≤1.5 GPU-h). GPT-6 suite draft.
- **Thu Nov 12:** GPT-6 suite frozen (before any scoring); Nova fact sheet v1 frozen; corpus build + G5; 2–3 Track 1 runs (≤8 GPU-h).
- **Fri Nov 13:** score S1/S2, gates, report to Stephon. Kaggle quota resets Fri 7 PM ET.
- **Nov 14 → ~Dec 4:** Track 2 grown model sessions if the benchmark supports it (~1.5–2 weeks of free quota), then SFT and the same eval bar.

## Non-negotiables

Free Kaggle/Colab only; anything paid goes to Stephon first. No shared passwords/tokens; nothing secret in chats. Private items never in git/Drive/packets. Delete nothing without Stephon. No promotion without Stephon's yes. Nova is backup coordinator (manual activation only).
