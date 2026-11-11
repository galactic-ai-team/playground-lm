# PlaygroundLM v0.6 Chat — Stage G G2_31505 promotion

**Authority:** Stephon approved at 4:54 PM ET on Nov 11, 2026: put G2 live with fact lookup and keep v0.5 Chat (E31140) for rollback.

| | |
|--|--|
| Live name | **PlaygroundLM v0.6 Chat** |
| Candidate | G2_31505 (Stage G Track 1, lr 3e-5) |
| Weights SHA256 | `a4710eb328e0ac7ca240ce02c5d8711586a01f405660d690053f86e0427bb6be` |
| ONNX q8 | `live/v0.6_g2_31505/step_00031505_g2.q8.onnx` on `Gteam04/playground-lm-onnx` @ `d9e1629d821e7a9550fac9c54c56b132cd9327b5` |
| ONNX SHA256 | `f7a76210291f00d9eaca0fe12f4c74fdc3be6ad71475af447777c9a3fd5792de` (179,148,599 bytes) |
| Fact lookup | BM25 top-3 from `demo/facts/fact_sheet.jsonl` (fact sheet v1 sha256 `f6ec9c14c29d55dc50cbd4c855a6de9e9654b65c765d1449a02d039e24c2ed33`). Prompt: `Facts: [1] … Question: …`, or `Facts: (none)` when nothing matches. Only for checkpoints with `fact_lookup: true`. |
| Rollback | v0.5 Chat (E31140). See `docs/ROLLBACK.md`. |
| Math | Still weak (0/36 on the Stage G arithmetic split). |

This PR ships the demo slice only (fact sheet, BM25 lookup, labels, manifest pin). Draft [PR #104](https://github.com/galactic-ai-team/playground-lm/pull/104) stays open for remaining Stage G / Track 2 training infrastructure and is not required to run the live site.

## Public regression (full, vs C31101, seed 20261108)

| | FineWeb val | Δ vs C | HellaSwag | LAMBADA |
|--|--:|--:|--:|--:|
| C31101 | 3.174208 | — | 3016/10042 (30.03%) | 1184/5153 (22.98%) |
| **G2_31505** | **3.178433** | **+0.00423** | **3007/10042 (29.94%)** | **1179/5153 (22.88%)** |

All gates PASS (FineWeb limit +0.02, HS drop ≤ 1 pt, LAMBADA drop ≤ 1.5 pt; G5 not FAIL).

## Private suites (aggregates only)

- Stage G 90-item suite, facts in the prompt: G2 **22/90** vs E31140 **2/90** (net +20, McNemar p≈1.1e-5). Facts 13/30, chat 3/14, uncertainty 6/10, **math 0/36**.
- Same suite without the fact sheet: 2/90 (net 0).
- Stage F 72-item suite: G2 **10/72** vs E31140 **2/72** (8 wins, 0 losses, p=0.0078).

Item text and model outputs from those suites are owner-only and are not in this repository.
