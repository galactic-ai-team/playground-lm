# Pretraining data plan — PlaygroundLM (Nova)

## Budget reality check
- Quota: ~30 Kaggle GPU-hours/week (team account), T4-class GPU.
- Assumed throughput for our 124M model, mixed precision: 10–25k
  tokens/sec on one T4 (to be measured in Stage B; plan re-baselines
  on the measured number, not this estimate).
- => ~1.1–2.7B tokens per week of quota. README's 2.5B-token target
  is therefore a 1–2.5 week pretrain, checkpointed every ~30 min so
  Kaggle sessions resume instead of restart (Dee's lane).

## Source
- HuggingFaceFW/fineweb-edu, config `sample-10BT` — public, openly
  licensed, no account/key. Already quality-filtered and deduplicated
  upstream; our prep (prepare_fineweb.py) adds light cleaning,
  exact dedupe, and the hash-stable split. Expected local dedupe
  rate: low (<2%); corpus_stats.py reports the actual number.

## Staging (each stage gates the next)
- **Stage A — smoke (20M approx tokens).** Validates the full path:
  stream -> clean -> split -> tokenize -> Claude's loader -> 100
  training steps -> checkpoint -> GPT-6's export. Run in one sitting.
- **Stage B — pipeline run (300M approx tokens).** First real
  checkpoints for the demo slider; measure true tok/s and dedupe
  rate; re-baseline Stage C size/time from measurements.
- **Stage C — full pretrain (2.5B approx tokens).** The run we
  benchmark against GPT-2 124M.

## Sizes (at 4 chars/token; exact after GPT-6's tokenizer pass)
- 2.5B tokens ~= 10 GB of JSONL text, ~2–3M FineWeb-Edu documents,
  ~50 shards at 50k docs/shard from prepare_fineweb.py.
- Tokenized uint16 .bin shards will be ~5 GB. Too big for git
  (data/ is gitignored): distribute via a Kaggle dataset and/or a
  Hugging Face dataset repo (both free), pinned by commit/release so
  every training run cites the exact corpus version it ate.

## Splits and eval hygiene
- Hash-stable split, 1% val for FineWeb-Edu (prepare_fineweb.py).
  1% of 2.5B is 25M val tokens — more than eval needs; the eval
  harness should cap val at the first ~2M tokens for fast,
  repeatable loss/perplexity numbers.
- Playground fine-tune corpus: 10% val (small corpus), hash-stable,
  author-labeled; its held-out split powers the voice/perplexity
  eval and the verbatim-copy check (a fine-tuned model must
  complete held-out lines *approximately*, never regenerate train
  lines verbatim beyond short stock phrases).

## Risks / open questions
- Throughput estimate is the plan's soft spot: if Stage B measures
  <8k tok/s, Stage C shrinks to ~1.5B tokens rather than eating a
  third week of quota. Decision point is Stephon's if we want more.
- Streaming resume in prepare_fineweb.py is manifest-based and
  untested at scale — Stage A/B exist partly to break it cheaply.
