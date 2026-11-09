# Stage E SFT artifact schema

Built by `data/build_stage_e_sft.py` (completion-only masks).

## Files written to `--out` (local/Kaggle data dir)

| File | Format | Meaning |
|------|--------|---------|
| `train.jsonl` / `val.jsonl` | JSON Lines | One object per example: `id`, `source`, `license`, `user`, `assistant`, `text` |
| `train.bin` / `val.bin` | `uint16` little-endian | Packed GPT-2 token ids |
| `train.masks.bin` / `val.masks.bin` | `uint8` | `1` = supervised (assistant/completion incl. EOT); `0` = prompt |
| `provenance.json` | JSON | Sources, licenses, counts, sha256 of token/mask files |

## Chat format

```
User: <instruction>
Assistant: <response><|endoftext|>
```

Prompt tokens unsupervised; assistant tokens (+ EOS) supervised via masks.

## Snapshot recorded in this PR

See `data/stage_e/provenance.json` (counts/shas from the Lumi prep build).  
Full `.bin` / `.jsonl` corpora stay out of git (rebuild with the builder on free CPU/Kaggle). Do **not** mix held-out behavior **expect** scorers into this corpus.
