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

## Dataset Licenses & Attribution
- **FineWeb-Edu (pretraining):** terms per its Hugging Face dataset card; see `data/PRETRAIN_PLAN.md`.
- **Playground dialogue (Stage D SFT):** the Galactic AI Team's own AI Playground text, built by `data/build_finetune_corpus.py`.
- **Dolly 15k (Stage E, held):** contains material from `databricks/databricks-dolly-15k`, © Databricks, Inc. and contributors, licensed under **CC BY-SA 3.0** (https://creativecommons.org/licenses/by-sa/3.0/). Share-alike terms apply to that subset; see `data/STAGE_E_PLAN.md`.
- **Arithmetic and team-identity examples (Stage E, held):** generated/hand-written in this repository (MIT), see `data/build_stage_e_corpus.py`; arithmetic answers are machine-verified and identity facts carry source notes.
- **GPT-2 BPE tokenizer files:** MIT (OpenAI), see `demo/THIRD_PARTY.md`.

## Intended Use & Limitations
- **Intended Use:** Research and educational inspection of from-scratch language model pretraining, domain adaptation on multi-agent dialogue, and client-side WebGPU/WASM inference via ONNX Runtime Web.
- **Limitations:** As a compact ~124M parameter model trained on a compute budget, `PlaygroundLM` is not suited for factual retrieval, safety-critical reasoning, or production deployment without downstream guardrails.

## Stage E Round 3 (live demo — step 31,140)

Promoted 2026-11-09 (Stephon): `r3-rr50-ep1-lr1e5` — 50% FineWeb-Edu Stage B replay + Playground SFT, lr 1e-5, seed Stage C 31,101.
Weights SHA256 `ce1fa2098236656d149437638601dfafa7744ab3f69ff52739be0c95199f818b`.
ONNX q8 on Hugging Face `Gteam04/playground-lm-onnx` (`step_00031140.q8.onnx`); Stage D `step_00031104.q8.onnx` kept for rollback.

**Honest subset metrics (Round-3 regression knobs):** FineWeb val 3.059 vs C 3.052 (Δ+0.007); HellaSwag 39.0%; LAMBADA 26.0% (200-item). Private exact 1/60 vs C/D 0/60 — not promotion evidence.

**Rollback:** point demo `manifest.json` live URL back to D31104:
`https://huggingface.co/Gteam04/playground-lm-onnx/resolve/9732b2c8318190b43b5f97c9f7299559efd46b26/step_00031104.q8.onnx`

