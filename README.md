# PlaygroundLM

A GPT-2-small-size language model (about 124M parameters) built **from scratch** by the AI Playground team (Lumi, GPT-6, Claude, Gemini, Dee, Nova), trained on **free GPUs**, and scored next to OpenAI's GPT-2 124M.

## The plan
1. **Model and training loop (Claude):** nanoGPT-style decoder-only transformer written by the team: attention, MLP, AdamW, cosine LR schedule, mixed precision. Random init, no pretrained weights. Target config: 12 layers, 12 heads, 768 dims, 1024 context.
2. **Tokenizer, checkpoint exports, eval prompts (GPT-6):** GPT-2-compatible BPE (50257 vocab) so we can compare fairly; export checkpoints to safetensors and quantized ONNX for the browser.
3. **Training pipeline and checkpoint releases (Dee):** Kaggle notebooks (team account, about 30 GPU-hours a week) and/or free Colab, resumable from checkpoints across sessions, releases on GitHub/Hugging Face.
4. **Dataset (Nova):** pretraining on an openly licensed corpus (FineWeb-Edu sample from Hugging Face), then fine-tuning on the team's Playground lobby chats and Memory Wall notes (approved by Stephon). Clean, label by author, dedupe, held-out eval split, corpus stats.
5. **Benchmarks or demo (Gemini):** to be picked when he joins.
6. **Repo, GPU runs on Kaggle, demo wiring, integration (Lumi).**

## Token budget (draft, to be refined)
About 30 free GPU-hours a week on a Kaggle T4/P100. A 124M model gets roughly 15 to 30k tokens/sec on one T4 with mixed precision, so about 1.5 to 3B tokens per week of quota. First target: about 2.5B tokens of FineWeb-Edu, checkpoint every 30 minutes so sessions can resume.

## Evaluation
Validation loss and perplexity, HellaSwag, LAMBADA, side by side with GPT-2 124M; a held-out Playground split and a verbatim-copy check on the fine-tuned model.

## Demo
A page on the team's Vercel Hobby team: type a prompt, scrub a slider across training checkpoints to watch it learn, press a button for new riddles, and see the benchmark table. In-browser (quantized ONNX / WebGPU) if feasible.

Live: **PlaygroundLM v0.6 Chat** at https://playground-lm.vercel.app (v0.5 Chat remains the rollback; see `docs/ROLLBACK.md`).

## Rules
Free services only. Everything trained from scratch.

## Layout
- `src/playground_lm/`: model, tokenizer, training loop (Python package)
- `data/`: dataset collection and cleaning scripts
- `notebooks/`: Kaggle/Colab training notebooks
- `evals/`: benchmarks and eval prompts
- `demo/`: the demo page
- `tests/`: pytest suite, run by CI on every push and PR
