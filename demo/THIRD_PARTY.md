# Third-party files in demo/

## GPT-2 BPE tokenizer files

| File | Size | SHA-256 |
|------|------|---------|
| `vocab.json` | 1,042,301 bytes (50,257 tokens) | `196139668be63f3b5d6574427317ae82f612a97c5d1cdaf36ed2256dbf636783` |
| `merges.txt` | 456,318 bytes (50,000 merges + `#version: 0.2` header) | `1ce1664773c50f3e0cc8842619a93edc4624525b728b188a9e0be33b7726adc5` |

- **Source:** Hugging Face `openai-community/gpt2`,
  https://huggingface.co/openai-community/gpt2/resolve/main/vocab.json and
  https://huggingface.co/openai-community/gpt2/resolve/main/merges.txt.
  These are the same as OpenAI's original `encoder.json` / `vocab.bpe` from
  https://github.com/openai/gpt-2.
- **Licence:** MIT (Copyright (c) 2019 OpenAI), per the gpt-2 repository and the
  Hugging Face model card.
- **Used by:** `demo/gpt2_bpe.js` through `demo/app.js` (`initTokenizer` probes
  `./vocab.json` + `./merges.txt` first). With these files present, the demo uses
  exact GPT-2 byte-level BPE (`"Hello world"` -> `[15496, 995]`, EOT 50256) and
  not the UTF-8 byte fallback. Files are unmodified.
