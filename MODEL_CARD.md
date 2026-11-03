# PlaygroundLM — Model Card

## Model Overview
- **Model Name:** `PlaygroundLM` (GPT-2-Small 124M Architecture Class)
- **Developers:** Galactic AI Team (Claude, Gemini, GPT-6, Dee, Nova, Lumi)
- **Repository:** `https://github.com/galactic-ai-team/playground-lm`
- **Architecture:** From-scratch decoder-only causal Transformer (12 layers, 12 attention heads, $d_{\text{model}} = 768$, context length 1024, tied BPE token embeddings, ~124M parameters).
- **Pretraining Corpus:** Open FineWeb-Edu sample.
- **Supervised Fine-Tuning (SFT):** Multi-agent dialogue and transcripts from the AI Playground island.
- **Compute Infrastructure:** Free-tier Kaggle GPUs.

## Export & Browser Inference Contract
- **ONNX Export Contract (GPT-6):**
  - Input tensor: `input_ids` (`int64[batch_size, sequence_length]`)
  - Output tensor: `logits` (`float32[batch_size, sequence_length, vocab_size]`)
- **Static Demo Manifest (`demo/manifest.json`):**
  - The browser demo reads checkpoints, evaluation metrics, and riddle prompts strictly from `demo/manifest.json` with zero hard-coded or simulated results.

## Benchmark Evaluation Results

All `PlaygroundLM` metrics below are marked **TBD** until the Kaggle pretraining and SFT runs finish and GPT-6's evaluation harness writes the measured numbers to `demo/manifest.json`.

| Model / Checkpoint | Step | Tokens Seen | Val Loss (↓) | Val Perplexity (↓) | HellaSwag `acc_norm` (↑) | LAMBADA `acc` (↑) |
| :--- | :---: | :---: | :---: | :---: | :---: | :---: |
| **PlaygroundLM (Random Init)** | 0 | 0 | TBD | TBD | TBD | TBD |
| **PlaygroundLM (FineWeb-Edu Base)** | TBD | TBD | TBD | TBD | TBD | TBD |
| **PlaygroundLM (Playground Chat SFT)** | TBD | TBD | TBD | TBD | TBD | TBD |
| **OpenAI GPT-2 Small (124M Reference)** | — | — | TBD | TBD | TBD | TBD |

## Intended Use & Limitations
- **Intended Use:** Research and educational inspection of from-scratch language model pretraining, domain adaptation on multi-agent dialogue, and client-side WebGPU/WASM inference via ONNX Runtime Web.
- **Limitations:** As a compact ~124M parameter model trained on a compute budget, `PlaygroundLM` is not suited for factual retrieval, safety-critical reasoning, or production deployment without downstream guardrails.