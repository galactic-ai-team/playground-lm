# PlaygroundLM — Model Card

## Model Overview
- **Model Name:** `PlaygroundLM` (GPT-2-Small 124M Architecture Class). Live demo name: **PlaygroundLM v0.6 Chat** (v0.5 Chat remains the rollback).
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

## PlaygroundLM v0.6 Chat — Stage G G2 (live demo — step 31,505)

Friendly name **v0.6 Chat**. Stephon approved promotion at 4:54 PM ET on Nov 11, 2026: G2_31505 with BM25 fact lookup replaces v0.5 Chat on https://playground-lm.vercel.app. **v0.5 Chat (E31140) is the rollback** and stays on the checkpoint slider.

Weights SHA256 `a4710eb328e0ac7ca240ce02c5d8711586a01f405660d690053f86e0427bb6be`.
ONNX q8 on Hugging Face `Gteam04/playground-lm-onnx` at `live/v0.6_g2_31505/step_00031505_g2.q8.onnx` (revision `d9e1629d821e7a9550fac9c54c56b132cd9327b5`, SHA256 `f7a76210291f00d9eaca0fe12f4c74fdc3be6ad71475af447777c9a3fd5792de`). v0.5 root files were not overwritten.

The demo retrieves the top 3 BM25 matches from fact sheet v1 (sha256 `f6ec9c14c29d55dc50cbd4c855a6de9e9654b65c765d1449a02d039e24c2ed33`) and prefixes the prompt as `Facts: [1] … Question: …`, or `Facts: (none)` when nothing matches. Arithmetic is still weak (0/36 on the Stage G math split).

**Honest full public regression vs C31101:** FineWeb val 3.178 vs C 3.174 (Δ+0.004); HellaSwag 29.94% vs 30.03% (−0.09 pt); LAMBADA 22.88% vs 22.98% (−0.10 pt). All gates PASS. Private Stage G suite (aggregates only, facts in the prompt): 22/90 vs v0.5's 2/90; math 0/36.

**Rollback:** see `docs/ROLLBACK.md`. One-step: revert the v0.6 promotion PR, or point the live manifest checkpoint back to E31140:
`https://huggingface.co/Gteam04/playground-lm-onnx/resolve/27778a045bee172529170330ada959131037da64/step_00031140.q8.onnx`

## PlaygroundLM v0.5 Chat — Stage E Round 3 (rollback — step 31,140)

Friendly name **v0.5 Chat** (Stephon, 11:24 PM ET Nov 9). Promoted 2026-11-09, replaced as live by v0.6 Chat on 2026-11-11, retained for rollback.

`r3-rr50-ep1-lr1e5` — 50% FineWeb-Edu Stage B replay + Playground SFT, lr 1e-5, seed Stage C 31,101.
Weights SHA256 `ce1fa2098236656d149437638601dfafa7744ab3f69ff52739be0c95199f818b`.
ONNX q8 on Hugging Face `Gteam04/playground-lm-onnx` (`step_00031140.q8.onnx` @ `27778a045bee172529170330ada959131037da64`); Stage D `step_00031104.q8.onnx` kept as a further rollback.

**Honest subset metrics (Round-3 regression knobs):** FineWeb val 3.059 vs C 3.052 (Δ+0.007); HellaSwag 39.0%; LAMBADA 26.0% (200-item). Private exact 1/60 vs C/D 0/60 — not promotion evidence.

