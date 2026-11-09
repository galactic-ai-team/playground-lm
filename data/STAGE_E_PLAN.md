# Stage E — instruction/Q&A data plan and audit (Nova, dataset lane)

Status: **Stage E training is HELD** until real team contributions land.
This document + `data/build_stage_e_corpus.py` are the groundwork: the
builder, its quality gates, and the license record a Stage E run will
need. Stage C step 31,101 and Stage D step 31,104 are preserved; nothing
here retrains or replaces them.

## Sources and licenses

| Source | License | Handling |
| --- | --- | --- |
| databricks/databricks-dolly-15k | **CC BY-SA 3.0** | Not vendored. Builder takes a local JSONL export (`--dolly-jsonl`, free HF download). Attribution string is embedded in the builder and in every stats sidecar. Share-alike terms travel with Dolly-derived data; the repo MIT license does not override them for that subset. Releasing a Dolly-trained checkpoint should carry the same attribution (see MODEL_CARD.md). |
| Arithmetic (generated) | MIT (this repo) | Deterministic generator in the builder. Answers computed at generation, then re-verified by an independent parser before a record is kept. |
| Team identity examples | MIT (this repo) | Hand-written in the builder; every example carries a `source_note` naming where the fact is recorded (model card / team status). A test fails if one is missing. |
| Playground text (Stages C/D corpora) | Team's own text | Unchanged; built by `build_finetune_corpus.py`. Not an instruction source for Stage E unless the team decides otherwise. |
| FineWeb-Edu (pretraining) | Per its HF dataset card | Unchanged from Stages A–D; see `data/PRETRAIN_PLAN.md`. |

Open decision for Stephon (flagged, not decided here): whether Stage E
ships Dolly-derived data/weights under CC BY-SA attribution as above,
or goes team-data-only. The builder supports both (`--dolly-jsonl`
optional).

## Contamination prevention — review

Already in place before Stage E, verified in the repo:

- `data/contamination_check.py`: token-level 13-gram check of train
  bins against HellaSwag val / LAMBADA test, keyed **by row index**
  (PR #51). Stage B results: HellaSwag 7/10,042 rows, LAMBADA 11/5,153
  flagged; headline scores exclude flagged rows for **both**
  PlaygroundLM and the GPT-2 baseline.
- `scripts/check_verbatim_copy.py`: verbatim-copy check for fine-tunes.
- Hash-stable splits (`common.split_for`) in every builder: identical
  text can never straddle train/val.

Stage E additions in `build_stage_e_corpus.py`:

- A denylist guard at build time: examples exactly matching, or
  sharing a word 13-gram with, a denylist text are dropped and counted.
  Default denylist = the behavior suite's frozen prompts
  (`evals/behavior/behavior_eval.prompts.frozen.json`, `prompts[].user`;
  the legacy `evals/prompts.json` schemas are also accepted), so Stage E
  can never train on the eval suites' own prompts. Loading is
  fail-closed: a missing, malformed, or empty denylist aborts the build
  unless `--allow-empty-denylist` is passed explicitly, and the stats
  sidecar records the denylist path, schema, and count used.
- The token-level checker still runs on the final tokenized Stage E
  bins before any training; the word-level guard is a complement, not
  a replacement.

## Quality gates

- Clean + exact/near dedupe across all sources (`common.Deduper`).
- Arithmetic: 100% of kept arithmetic records pass the independent
  re-parse (audit counts `arithmetic_verified`; failures are dropped).
- Identity: source note required per example (unit-tested).
- Oversize instruction/response caps; empty Dolly rows skipped+counted.
- Stats sidecar records per-source counts, the full audit report, and
  the license block, so any Stage E run can cite exactly what it ate.

## Model-card attribution

`MODEL_CARD.md` gains a "Dataset Licenses & Attribution" section in
this change: Dolly CC BY-SA 3.0 attribution, the generated/identity
sources, and a pointer here. Benchmark cells stay TBD until the eval
lane measures them — no numbers are claimed in this change.
