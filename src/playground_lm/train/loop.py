"""Training for notebook-created models, optimizers and infinite batch iterators."""

import json
import logging
import math
import time
from contextlib import nullcontext
from pathlib import Path

import torch

logger = logging.getLogger(__name__)


def _learning_rate(step, peak, warmup, total):
    # Updates are numbered 1..total. Runs ending in warmup only warm up.
    if step <= warmup:
        return peak * step / warmup
    progress = (step - warmup) / (total - warmup)
    return peak * (0.1 + 0.45 * (1.0 + math.cos(math.pi * progress)))


def train(*, model, optimizer, train_loader, val_loader, checkpoint_manager,
          config, start_step=0, max_steps=200_000, device="cpu", on_save=None,
          start_tokens_seen: int = 0, start_wall_time_s: float = 0.0):
    """Train from completed update start_step through max_steps (inclusive).

    Append one metrics record per update, with val_loss every 500 updates.
    Throughput excludes validation, logging and saving. wall_time_s starts at
    start_wall_time_s; the one-based callback save_count covers this invocation
    only. Token counts start at start_tokens_seen and assume full batches of
    the configured size. FP16 overflows retry the same update number with
    fresh micro-batches.
    """
    if not 0 <= start_step <= max_steps:
        raise ValueError("require 0 <= start_step <= max_steps")
    for key in ("batch_size", "grad_accum_steps", "context_len"):
        if not isinstance(config[key], int) or config[key] <= 0:
            raise ValueError(f"{key} must be a positive integer")
    warmup = config["warmup_steps"]
    if not isinstance(warmup, int) or warmup < 0:
        raise ValueError("warmup_steps must be a nonnegative integer")
    peak, clip = float(config["lr"]), float(config["grad_clip"])
    if not math.isfinite(peak) or peak <= 0:
        raise ValueError("lr must be finite and positive")
    if not math.isfinite(clip) or clip <= 0:
        raise ValueError("grad_clip must be finite and positive")
    accum = config["grad_accum_steps"]
    tokens_per_step = config["batch_size"] * accum * config["context_len"]
    device = torch.device(device)
    if device.type not in ("cpu", "cuda"):
        raise ValueError("this loop supports CPU and CUDA")
    amp_dtype, scaler = None, None
    if device.type == "cuda":
        with torch.cuda.device(device):
            amp_dtype = (torch.bfloat16 if torch.cuda.is_bf16_supported()
                         else torch.float16)
        if amp_dtype == torch.float16:
            scaler = torch.cuda.amp.GradScaler()

    def autocast():
        return (nullcontext() if amp_dtype is None else
                torch.autocast(device_type="cuda", dtype=amp_dtype))

    def synchronize():
        if device.type == "cuda":
            torch.cuda.synchronize(device)

    step, save_count, last_saved_step = start_step, 0, None
    tokens_seen = start_tokens_seen
    started = time.perf_counter()
    run_dir = Path(checkpoint_manager.run_dir)
    run_dir.mkdir(parents=True, exist_ok=True)

    def save():
        nonlocal save_count, last_saved_step
        if last_saved_step == step:
            return
        synchronize()
        checkpoint_manager.save(
            getattr(model, "_orig_mod", model), optimizer, step,
            tokens_seen, start_wall_time_s + (time.perf_counter() - started),
        )
        last_saved_step = step
        save_count += 1
        if on_save is not None:
            on_save(step, save_count)

    model.train()
    try:
        with (run_dir / "metrics.jsonl").open("a", encoding="utf-8") as metrics:
            while step < max_steps:
                lr = _learning_rate(step + 1, peak, warmup, max_steps)
                for group in optimizer.param_groups:
                    group["lr"] = lr
                synchronize()
                update_started = time.perf_counter()
                optimizer.zero_grad(set_to_none=True)
                loss_sum = torch.zeros((), device=device, dtype=torch.float32)
                for _ in range(accum):
                    x, y = next(train_loader)
                    with autocast():
                        _, loss = model(x.to(device), y.to(device))
                        micro_loss = loss / accum
                    loss_sum += loss.detach().float()
                    if scaler is None:
                        micro_loss.backward()
                    else:
                        scaler.scale(micro_loss).backward()
                if scaler is not None:
                    scaler.unscale_(optimizer)
                torch.nn.utils.clip_grad_norm_(model.parameters(), clip)
                if scaler is None:
                    optimizer.step()
                else:
                    old_scale = scaler.get_scale()
                    scaler.step(optimizer)
                    scaler.update()
                    if scaler.get_scale() < old_scale:
                        logger.warning("FP16 overflow; retrying update %d", step + 1)
                        continue
                step += 1
                tokens_seen += tokens_per_step
                synchronize()
                elapsed = time.perf_counter() - update_started
                record = dict(step=step, loss=loss_sum.item() / accum,
                              lr=lr, tokens_per_sec=tokens_per_step / elapsed,
                              tokens_seen=tokens_seen)
                if step % 500 == 0:
                    model.eval()
                    try:
                        val_sum = torch.zeros((), device=device, dtype=torch.float32)
                        with torch.no_grad():
                            for _ in range(20):
                                x, y = next(val_loader)
                                with autocast():
                                    _, val_loss = model(x.to(device), y.to(device))
                                val_sum += val_loss.detach().float()
                        record["val_loss"] = val_sum.item() / 20
                    finally:
                        model.train()
                record["wall_time_s"] = (
                    start_wall_time_s + (time.perf_counter() - started)
                )
                metrics.write(json.dumps(record) + "\n")
                metrics.flush()
                logger.info("%s", json.dumps(record))
                if checkpoint_manager.should_save(step):
                    save()
            save()
    except KeyboardInterrupt:
        optimizer.zero_grad(set_to_none=True)
        try:
            save()
        except Exception:
            logger.exception("Checkpoint save failed during KeyboardInterrupt")
        raise