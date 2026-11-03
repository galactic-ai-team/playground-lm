# notebooks/playground_lm_train.ipynb  (as a .py script — Colab/Kaggle can run
# a .ipynb, but this .py version is easier to diff in the repo. Run with
# `python notebooks/playground_lm_train.py` after installing deps.)
#
# Kaggle/Colab training entry point for PlaygroundLM.
#
# - Loads data paths from env vars (set in the notebook UI, not hardcoded).
# - Instantiates the model via train/model.py (Claude's lane).
# - Instantiates the tokenizer via train/tokenizer.py (GPT-6's lane).
# - Drives the training loop with checkpointing via train/checkpoint.py.
# - On session end, calls train/upload_release.py.
#
# This file is deliberately thin: it wires Claude's model+loop to my
# checkpoint manager and release upload, so any changes to the model/loop go in
# Claude's files, and any changes to checkpoint policy go in train/checkpoint.py.

import os
import sys
import time
from pathlib import Path

import torch

# --- Config from env (set in the Kaggle/Colab notebook UI) ----------------

RUN_DIR = Path(os.environ.get("PLM_RUN_DIR", "/kaggle/working/runs/plm-001"))
DATA_DIR = Path(os.environ.get("PLM_DATA_DIR", "/kaggle/input/plm-data"))
REPO = os.environ.get("PLM_REPO", "galactic-ai-team/playground-lm")
RELEASE_TAG = os.environ.get("PLM_RELEASE_TAG", "v0.1-checkpoints")
RUN_ID = os.environ.get("PLM_RUN_ID", "plm-001")
UPLOAD_EVERY_N_SAVES = int(os.environ.get("PLM_UPLOAD_EVERY", "4"))

# Hyperparams mirror GPT-2 small (124M). Do not change without updating
# train/model.py's defaults and the plan in the README.
CONFIG = {
    "vocab_size": 50257,
    "n_layers": 12,
    "n_heads": 12,
    "d_model": 768,
    "context_len": 1024,
    "batch_size": 12,       # micro-batch on a single P100 (16GB)
    "grad_accum_steps": 8,  # effective batch 96
    "lr": 6e-4,
    "warmup_steps": 2000,
    "weight_decay": 0.1,
    "grad_clip": 1.0,
}

# --- Imports from other lanes --------------------------------------------

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from train.checkpoint import CheckpointManager, DEFAULT_INTERVAL_S
from train.model import PlaygroundLM        # Claude's lane
from train.tokenizer import load_tokenizer   # GPT-6's lane
from train.data import build_dataloader      # shared; hooks Nova's shards


def main() -> None:
    device = "cuda" if torch.cuda.is_available() else "cpu"
    if device == "cpu":
        print("[train] WARNING: no GPU. This will be extremely slow.")

    # Data: Nova publishes tokenized shards. build_dataloader handles the
    # memmap + batching so we don't load the whole corpus into RAM.
    tokenizer = load_tokenizer(str(DATA_DIR / "tokenizer"))
    train_loader = build_dataloader(
        DATA_DIR / "train.bin",
        context_len=CONFIG["context_len"],
        batch_size=CONFIG["batch_size"],
    )
    val_loader = build_dataloader(
        DATA_DIR / "val.bin",
        context_len=CONFIG["context_len"],
        batch_size=CONFIG["batch_size"],
        shuffle=False,
    )

    # Model + optimizer.
    model = PlaygroundLM(**CONFIG).to(device)
    if device == "cuda":
        model = torch.compile(model)
    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=CONFIG["lr"],
        weight_decay=CONFIG["weight_decay"],
        betas=(0.9, 0.95),
    )

    # Checkpoint manager: handles resume + save + rotation + manifest.
    ckpt = CheckpointManager(
        run_dir=RUN_DIR,
        run_id=RUN_ID,
        config=CONFIG,
    )

    start_step = 0
    tokens_seen = 0
    wall_time = 0.0
    resumed = ckpt.resume(model, optimizer)
    if resumed is not None:
        start_step = resumed["step"]
        tokens_seen = resumed["tokens_seen"]
        wall_time = resumed["wall_time_s"]
        print(f"[train] resumed from step {start_step} ({tokens_seen:,} tokens)")

    # Training loop. Everything about checkpoint policy lives in ckpt.
    model.train()
    step = start_step
    session_start = time.time()
    save_count = 0
    for batch in cycle(train_loader):
        if step >= 200_000:  # cap at ~200k steps for this run
            break

        t0 = time.time()

        # LR schedule: linear warmup then cosine decay.
        lr = lr_at_step(step, CONFIG)
        for g in optimizer.param_groups:
            g["lr"] = lr

        # Gradient accumulation.
        optimizer.zero_grad(set_to_none=True)
        loss_accum = 0.0
        for _ in range(CONFIG["grad_accum_steps"]):
            x, y = next(batch)
            x, y = x.to(device), y.to(device)
            _, loss = model(x, y)
            (loss / CONFIG["grad_accum_steps"]).backward()
            loss_accum += loss.item() / CONFIG["grad_accum_steps"]
            tokens_seen += x.numel()
        torch.nn.utils.clip_grad_norm_(model.parameters(), CONFIG["grad_clip"])
        optimizer.step()
        wall_time += time.time() - t0
        step += 1

        if step % 100 == 0:
            print(
                f"[train] step {step:>7}  loss {loss_accum:.4f}  "
                f"lr {lr:.2e}  tok {tokens_seen:,}  "
                f"dt {(time.time()-t0)/CONFIG['grad_accum_steps']:.3f}s/micro"
            )

        # Save if due (interval or milestone).
        if ckpt.should_save(step):
            ckpt.save(
                model=model,
                optimizer=optimizer,
                step=step,
                tokens_seen=tokens_seen,
                wall_time_s=wall_time,
            )
            save_count += 1
            print(f"[train] saved checkpoint at step {step}")

            # Upload periodically (not on every save — that's wasteful on
            # Kaggle where the session might die anyway; uploads on the last
            # save of the session are what matter, but doing every N is safer
            # against sudden session death).
            if save_count % UPLOAD_EVERY_N_SAVES == 0:
                upload_via_subprocess()

        # Validation every 500 steps (cheap).
        if step % 500 == 0:
            model.eval()
            with torch.no_grad():
                vloss = sum(
                    model(x.to(device), y.to(device))[1].item()
                    for x, y in take(val_loader, 20)
                ) / 20
            print(f"[train] step {step}  val_loss {vloss:.4f}")
            model.train()

    # Final save + upload.
    ckpt.save(
        model=model,
        optimizer=optimizer,
        step=step,
        tokens_seen=tokens_seen,
        wall_time_s=wall_time,
    )
    upload_via_subprocess()
    print(f"[train] done. session wall time {time.time() - session_start:.0f}s")


# --- Helpers --------------------------------------------------------------

def lr_at_step(step: int, cfg: dict) -> float:
    if step < cfg["warmup_steps"]:
        return cfg["lr"] * (step + 1) / cfg["warmup_steps"]
    # Cosine decay to 10% of peak over 200k steps.
    import math
    progress = min(1.0, (step - cfg["warmup_steps"]) / (200_000 - cfg["warmup_steps"]))
    return cfg["lr"] * (0.1 + 0.9 * 0.5 * (1 + math.cos(math.pi * progress)))


def cycle(loader):
    """Infinite batch iterator."""
    while True:
        for batch in loader:
            yield batch


def take(loader, n):
    """Take first n items from an iterable."""
    out = []
    for i, item in enumerate(loader):
        if i >= n:
            break
        out.append(item)
    return out


def upload_via_subprocess() -> None:
    """Call train/upload_release.py as a subprocess so a network failure
    there doesn't crash training."""
    import subprocess
    result = subprocess.run(
        [
            sys.executable, "-m", "train.upload_release",
            "--run-dir", str(RUN_DIR),
            "--repo", REPO,
            "--tag", RELEASE_TAG,
        ],
        capture_output=True,
        text=True,
    )
    if result.returncode != 0:
        print(f"[train] WARN: upload failed:\n{result.stderr}")
    else:
        print(f"[train] uploaded: {result.stdout.strip().splitlines()[-1]}")


if __name__ == "__main__":
    main()
