# GPT-6 Stage G — S2 scorer and scratchpad review

Base: `feat/stage-g-scratchpad` (draft PR #104), prior to team integration.

## Source review

- The generator's prescribed single-line `Work: ... Answer: <int>` output is compatible with the scorer's **terminal** `Answer:` integer extraction. The scorer refuses trailing prose after the answer and uses the last eligible terminal cue.
- The Stage G plan reports median 82 and p95 240 **total GPT-2 tokens per generated example** (not independently remeasured here). A 48-token strict generation limit is generally too short for scratchpads. S2 uses 256 continuation tokens; context length and prompt length remain considerations for longer 2-digit multiplication.
- `Digits from the top place down` gives an explicit final aggregation target, which may help a small LM but increases chain length. Keep it for the registered experiment; evaluate any later ablation under a separately frozen protocol.
- **Found verifier hole:** `verify_steps()` checks equations and division-remainder statements but not the validity of `write N`, `carry N`, or the final digit-assembly annotation. A deliberately corrupted `write` can be accepted by old `check_row()` if `Answer:` remains correct. The patch makes `check_row()` require the entire generated assistant response to match `solve(op,a,b)`, and adds three synthetic regression tests.

## S1 / S2

- S1: whole-output `strip().casefold()` exact matching; numeric answers also ignore commas/spaces. Locked maximum generation 48.
- S2: terminal `Answer: integer` extraction when present, otherwise whole-output matching. Locked maximum generation 256.
- The `evaluate_suite()` entry point reuses the accepted `evals.run_behavior_eval.generate()` greedy path; it emits pass/fail only and **does not put raw private outputs or answers into receipts**.
- Fact-sheet pipeline requires a caller-provided deterministic retriever, exact fact-sheet SHA-256, and a single-turn prompt. The retriever must be independently version-pinned and validated by the operator. Don't compare closed-book to pipeline or differently hashed fact sheets.
- `compare_receipts()` fails closed on different scorer, max-new, decoding settings, suite SHA, mode, fact-sheet SHA, IDs, counts, seed base, or device; uses Claude's exact McNemar implementation. A significant paired win is necessary but **never sufficient** for promotion.

## Run tests

```sh
PYTHONPATH=src:. pytest -q tests/test_stage_g_scoring.py data/test_gen_scratchpad_arith_gpt6.py
```

The public scorer unit tests were run offline in an isolated harness stub of the two existing import points (`mcnemar_exact` and `build_prompt`); **23 passed**. The generator regression tests must be run against the real Stage G repo source after applying the patch (the full repository source is not mounted in this workspace). No weights, internet, or private evaluation items were used in tests.

## Private suite dependency

`data/facts/fact_sheet.jsonl` was absent on `feat/stage-g-scratchpad` at inspection. The owner-only archive therefore freezes only a **preliminary 60-item non-fact subset**, with 36 arithmetic `exclude_keys` and 14 chat + 10 uncertainty. **It is not the final Stage G private suite; do not run or promote against it.** Nova's fact sheet v1 SHA plus its held-out list are required before writing 20 in-sheet and 10 out-of-sheet questions and finalizing the expected ~90-case suite. After that, perform normalized/near-duplicate checks against prior private suites and Stage G train/val before any candidate scoring.
