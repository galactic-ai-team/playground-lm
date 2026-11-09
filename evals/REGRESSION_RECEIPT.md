# Round-2 Stage E: public general-ability regression receipt

This is a **public benchmark harness**, not a private holdout, and is not evidence by itself that any checkpoint should be promoted. The existing C31101 and D31104 releases remain unchanged. The tool is read-only with respect to model weights and training sets.

## Run

Use the **identical FineWeb-Edu Stage C held-out `val.bin`** for baseline and every round-2 candidate. The file hash is recorded in each receipt. Run in an environment with the project's CPU/GPU dependencies and locally available benchmark datasets:

```bash
PYTHONPATH=src:. python -m evals.run_regression_receipt \
  --checkpoint /path/to/step_00031101.safetensors \
  --config /path/to/model_config.json \
  --val-bin /path/to/fineweb_stage_c_val.bin \
  --device cpu --offline --repeat-check --out receipts/c31101.json

PYTHONPATH=src:. python -m evals.run_regression_receipt \
  --checkpoint /path/to/round2_candidate.safetensors \
  --config /path/to/model_config.json \
  --val-bin /path/to/fineweb_stage_c_val.bin \
  --device cpu --offline --repeat-check --out receipts/round2_candidate.json
```

No limits means official **full validation + 10,042 HellaSwag validation + 5,153 LAMBADA test**. For a quick CPU smoke test append `--benchmark-limit 16 --max-val-batches 2`; these receipts will report `publishable=false` and must not be compared to full benchmark results.

`--repeat-check` reuses the *same model object* and reloads the evaluation datasets in their original deterministic iteration order. For an independent second-process reproducibility check, run the command again with a different `--out` and compare metrics, checkpoint hashes, data hashes, and settings. The `timestamp_utc` will differ; the numerical results should not exceed the absolute `--tolerance` (default `1e-5`). GPU algorithm-level determinism may vary by hardware; a failed repeat check is recorded as a failure, not silently rounded away.

## Receipt schema (`schema_version: 1`)

- `receipt_type`: literal `regression_receipt`
- `suite_type`: literal `public_general_ability_only`
- `timestamp_utc`: ISO 8601 UTC timestamp
- `checkpoint`: `path`, `sha256`, optional `config_sha256`
- `datasets`: `validation` (`path`, `sha256`, `name`), official dataset identifiers for `hellaswag` and `lambada`
- `settings`: `context_len`, `batch_size`, `seed`, `device`, `offline`, `benchmark_limit`, `max_val_batches`, `scoring`
- `metrics`: `val_loss`, `perplexity`, `hellaswag_acc_norm`, `lambada_acc` (accuracies are fractions, not percentages)
- `coverage`: validation scored/available tokens; benchmark examples/correct/complete
- `complete`: full token and official dataset coverage
- `publishable`: only true for full requested, complete results (still **not model-promotion permission**)
- `repeat_check`: `performed`, `absolute_metric_deltas` (null if not requested)
- `limitations`: explicitly public aggregate only

**Claude G9** can ingest the JSON directly via `receipt_type == "regression_receipt"`, verifying identical `datasets.validation.sha256`, official dataset names, and relevant scoring settings before comparing checkpoint metrics. CPU vs GPU runs may differ slightly; record device and tolerance.

## Private-suite non-overlap gate — later phase only

After **all** round-2 candidates are frozen, create a **new** private suite. Before viewing any outputs: hash its prompt and expectation files and freeze its scoring protocol. Then compare per-item canonical prompt-text hashes with round-1 private suite hashes in an **offline owner-only** workspace:

`set(round2_prompt_sha256_per_item).isdisjoint(set(round1_prompt_sha256_per_item))`

This acceptance condition **cannot be executed now**, because new round-2 private items have not been authored/frozen and round-1 prompt text is intentionally not available in this lane. The round-1 file-level SHA-256s are insufficient to establish per-prompt non-overlap. Audit new train/val n-grams locally before scoring; only release aggregates. Never submit private items, expectations, or outputs to GitHub, data builders, UI contributors, or candidate training.
