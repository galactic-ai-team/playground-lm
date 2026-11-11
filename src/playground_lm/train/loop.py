"""Training for notebook-created models, optimizers and infinite batch iterators.

Review fixes (Claude):
  * Mixed precision is chosen from the GPU's compute capability. Recent
    PyTorch reports torch.cuda.is_bf16_supported() == True on a T4 and a P100
    because it counts *emulated* bf16, which is far slower there. Native bf16
    needs compute capability 8.0+ (A100, L4, ...); older GPUs now get FP16 +
    GradScaler. config["amp_dtype"] = "auto" | "bf16" | "fp16" | "fp32"
    overrides, and the choice is logged and written to the first metrics row.
  * FP16 overflow retries are capped (config["max_overflow_retries"], default
    25 in a row); before, a persistent overflow would retry forever silently.
  * Validation uses the same batches every time: the first eval reads
    config["eval_batches"] (default 20) batches and later evals reuse them, so
    val_loss is comparable across the run instead of drifting with the
    iterator. Interval: config["eval_interval"] (default 500).
  * Optional config["max_session_s"]: save and return cleanly when a Kaggle
    or Colab session is about to run out.
  * The FP16 GradScaler state is checkpointed: saves pass
    extra_state={"grad_scaler": ...} to the checkpoint manager, and
    start_scaler_state restores it on resume (pass
    resumed["extra"].get("grad_scaler") from CheckpointManager.resume).
  * Multi-GPU (DDP via torchrun, see train/distributed.py): when a process
    group is running, gradients are all-reduced once per optimizer step
    (no_sync on the other micro-batches), tokens_seen and tokens_per_sec
    count every GPU, logged and validation losses are averaged over GPUs,
    the session-budget stop is agreed by all ranks, and only rank 0 writes
    metrics, saves checkpoints and calls on_save. Checkpoints hold the plain
    model, so the format is unchanged. Single-GPU and CPU runs don't touch
    torch.distributed at all.
  * Health metrics in every metrics.jsonl row:
      grad_norm        total gradient norm before clipping (the value
                       clip_grad_norm_ already computes); a rising trend or
                       spikes usually come before a loss blow-up.
      loss_scale       FP16 GradScaler scale after this update (FP16 runs only);
                       a scale that keeps halving means frequent overflows.
      overflow_retries FP16 overflows skipped before this update succeeded
                       (only present when > 0).
      peak_mem_gib     peak GPU memory allocated by this process since the
                       session started, in GiB (CUDA only; per GPU under DDP,
                       rank 0's value is logged).
  * Per-stage LR schedule: config["schedule_start_step"] (default 0) is the
    update the current schedule starts from. Warmup and cosine decay run
    from there to max_steps, so a stage that continues from an earlier
    stage's checkpoint (Stage C from Stage B's step 2035) warms up again
    from 0 instead of jumping straight from the old stage's final LR (10% of
    peak) to ~peak. It is a fixed number, not "wherever this session
    resumed", so later sessions of the same stage continue the schedule.
"""

import json
import logging
import math
import time
from contextlib import nullcontext
from pathlib import Path

import torch
import torch.distributed as dist

from .distributed import unwrap

logger = logging.getLogger(__name__)


def _learning_rate(step, peak, warmup, total):
    # Updates are numbered 1..total. Runs ending in warmup only warm up.
    if step <= warmup:
        return peak * step / warmup
    progress = (step - warmup) / (total - warmup)
    return peak * (0.1 + 0.45 * (1.0 + math.cos(math.pi * progress)))


def _pick_amp_dtype(device, requested="auto"):
    """None (FP32), torch.bfloat16 or torch.float16 for this device."""
    names = {"fp32": None, "float32": None, "bf16": torch.bfloat16,
             "bfloat16": torch.bfloat16, "fp16": torch.float16, "float16": torch.float16}
    if device.type != "cuda":
        return None
    if requested != "auto":
        if requested not in names:
            raise ValueError(f"amp_dtype must be auto, bf16, fp16 or fp32, got {requested!r}")
        return names[requested]
    # Native bf16 needs compute capability 8.0+ (Ampere: A100, A10, L4, ...).
    # is_bf16_supported() is True on a T4/P100 too, via slow emulation.
    major, _minor = torch.cuda.get_device_capability(device)
    return torch.bfloat16 if major >= 8 else torch.float16


def _make_grad_scaler():
    try:
        return torch.amp.GradScaler("cuda")  # PyTorch 2.3+
    except (AttributeError, TypeError):
        return torch.cuda.amp.GradScaler()  # older 2.x


class _NullMetrics:
    """Stands in for metrics.jsonl on ranks other than 0."""

    def write(self, _text):
        pass

    def flush(self):
        pass


def _peak_mem_gib(device):
    """Peak memory allocated on a CUDA device since the process started (or
    the last reset), in GiB; None on CPU."""
    if device.type != "cuda":
        return None
    return torch.cuda.max_memory_allocated(device) / 2**30


def _world():
    """(rank, world_size) of a running process group, else (0, 1)."""
    if dist.is_available() and dist.is_initialized() and dist.get_world_size() > 1:
        return dist.get_rank(), dist.get_world_size()
    return 0, 1


def _mean_over_ranks(value, world):
    # SUM then divide: gloo (CPU tests) has no AVG reduce op.
    if world > 1:
        dist.all_reduce(value, op=dist.ReduceOp.SUM)
        value /= world
    return value


def train(*, model, optimizer, train_loader, val_loader, checkpoint_manager,
          config, start_step=0, max_steps=200_000, device="cpu", on_save=None,
          start_tokens_seen: int = 0, start_wall_time_s: float = 0.0,
          start_scaler_state=None):
    """Train from completed update start_step through max_steps (inclusive).

    Append one metrics record per update, with val_loss every eval_interval
    updates (default 500) on the same eval_batches batches each time.
    Throughput excludes validation, logging and saving. wall_time_s starts at
    start_wall_time_s; the one-based callback save_count covers this invocation
    only. Token counts start at start_tokens_seen and assume full batches of
    the configured size. FP16 overflows retry the same update number with
    fresh micro-batches. start_scaler_state, if given, is a GradScaler
    state_dict from a previous session; it is ignored when this run doesn't
    use FP16 (e.g. resuming a T4 checkpoint on an A100 with bf16).
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
    schedule_start = config.get("schedule_start_step", 0)
    if not isinstance(schedule_start, int) or not 0 <= schedule_start <= start_step:
        raise ValueError(
            f"schedule_start_step must be an int in [0, start_step={start_step}], "
            f"got {schedule_start!r}")
    if max_steps > 0 and schedule_start >= max_steps:
        raise ValueError(f"schedule_start_step {schedule_start} must be below max_steps {max_steps}")
    accum = config["grad_accum_steps"]
    rank, world = _world()
    is_main = rank == 0
    # Global tokens per optimizer step: every GPU processes its own batches.
    tokens_per_step = config["batch_size"] * accum * config["context_len"] * world
    device = torch.device(device)
    if device.type not in ("cpu", "cuda"):
        raise ValueError("this loop supports CPU and CUDA")
    amp_dtype = _pick_amp_dtype(device, config.get("amp_dtype", "auto"))
    scaler = _make_grad_scaler() if amp_dtype == torch.float16 else None
    if scaler is not None and start_scaler_state:
        scaler.load_state_dict(start_scaler_state)
        logger.info("resumed GradScaler at loss scale %s", scaler.get_scale())
    amp_name = {None: "fp32", torch.bfloat16: "bf16", torch.float16: "fp16"}[amp_dtype]
    logger.info("mixed precision: %s%s", amp_name, " + GradScaler" if scaler else "")
    max_overflow_retries = int(config.get("max_overflow_retries", 25))
    eval_interval = int(config.get("eval_interval", 500))
    eval_batches = int(config.get("eval_batches", 20))
    max_session_s = config.get("max_session_s")
    if eval_interval <= 0 or eval_batches <= 0:
        raise ValueError("eval_interval and eval_batches must be positive")

    def autocast():
        return (nullcontext() if amp_dtype is None else
                torch.autocast(device_type="cuda", dtype=amp_dtype))

    def synchronize():
        if device.type == "cuda":
            torch.cuda.synchronize(device)

    step, save_count, last_saved_step = start_step, 0, None
    tokens_seen = start_tokens_seen
    consecutive_overflows = 0
    val_cache = []  # first eval's batches, reused so val_loss is comparable
    first_record = True
    session_started = time.monotonic()  # separate clock: perf_counter is for throughput
    started = time.perf_counter()
    run_dir = Path(checkpoint_manager.run_dir)
    run_dir.mkdir(parents=True, exist_ok=True)

    def save():
        nonlocal save_count, last_saved_step
        if last_saved_step == step:
            return
        if not is_main:  # weights and optimizer state are identical on every rank
            last_saved_step = step
            return
        synchronize()
        args = (unwrap(model), optimizer, step,
                tokens_seen, start_wall_time_s + (time.perf_counter() - started))
        if scaler is None:
            checkpoint_manager.save(*args)
        else:
            checkpoint_manager.save(*args, extra_state={"grad_scaler": scaler.state_dict()})
        last_saved_step = step
        save_count += 1
        if on_save is not None:
            on_save(step, save_count)

    eval_model = model.module if hasattr(model, "no_sync") and hasattr(model, "module") else model
    no_sync = getattr(model, "no_sync", None) if world > 1 else None
    model.train()
    try:
        metrics_file = ((run_dir / "metrics.jsonl").open("a", encoding="utf-8") if is_main
                        else nullcontext(_NullMetrics()))
        with metrics_file as metrics:
            while step < max_steps:
                # Steps within the current stage's schedule (1-based).
                lr = _learning_rate(step + 1 - schedule_start, peak, warmup,
                                    max_steps - schedule_start)
                for group in optimizer.param_groups:
                    group["lr"] = lr
                synchronize()
                update_started = time.perf_counter()
                optimizer.zero_grad(set_to_none=True)
                loss_sum = torch.zeros((), device=device, dtype=torch.float32)
                for micro in range(accum):
                    batch = next(train_loader)
                    x, y = batch[0], batch[1]
                    # Optional per-target loss weights (Stage G answer weighting).
                    w = batch[2] if len(batch) > 2 else None
                    # DDP: all-reduce gradients only on the last micro-batch.
                    sync = (no_sync() if no_sync is not None and micro < accum - 1
                            else nullcontext())
                    with sync:
                        with autocast():
                            if w is None:
                                _, loss = model(x.to(device), y.to(device))
                            else:
                                _, loss = model(x.to(device), y.to(device), loss_weights=w.to(device))
                            micro_loss = loss / accum
                        loss_sum += loss.detach().float()
                        if scaler is None:
                            micro_loss.backward()
                        else:
                            scaler.scale(micro_loss).backward()
                if scaler is not None:
                    scaler.unscale_(optimizer)
                # Total norm before clipping; identical on every rank under DDP.
                grad_norm = torch.nn.utils.clip_grad_norm_(model.parameters(), clip)
                if scaler is None:
                    optimizer.step()
                else:
                    old_scale = scaler.get_scale()
                    scaler.step(optimizer)
                    scaler.update()
                    if scaler.get_scale() < old_scale:
                        consecutive_overflows += 1
                        if consecutive_overflows > max_overflow_retries:
                            raise FloatingPointError(
                                f"FP16 overflow on {consecutive_overflows} attempts in a row at "
                                f"update {step + 1}; stopping (last checkpoint is intact)")
                        logger.warning("FP16 overflow; retrying update %d", step + 1)
                        continue
                overflow_retries = consecutive_overflows
                consecutive_overflows = 0
                step += 1
                tokens_seen += tokens_per_step
                _mean_over_ranks(loss_sum, world)
                synchronize()
                elapsed = time.perf_counter() - update_started
                record = dict(step=step, loss=loss_sum.item() / accum,
                              lr=lr, tokens_per_sec=tokens_per_step / elapsed,
                              tokens_seen=tokens_seen)
                record["grad_norm"] = float(grad_norm)
                if scaler is not None:
                    record["loss_scale"] = float(scaler.get_scale())
                if overflow_retries:
                    record["overflow_retries"] = overflow_retries
                peak_mem = _peak_mem_gib(device)  # not `peak`: that's the LR peak above
                if peak_mem is not None:
                    record["peak_mem_gib"] = round(peak_mem, 3)
                if first_record:
                    record["amp_dtype"] = amp_name
                    first_record = False
                if step % eval_interval == 0:
                    model.eval()
                    try:
                        val_sum = torch.zeros((), device=device, dtype=torch.float32)
                        with torch.no_grad():
                            for i in range(eval_batches):
                                if i == len(val_cache):
                                    val_cache.append(next(val_loader))
                                x, y = val_cache[i]
                                with autocast():
                                    _, val_loss = eval_model(x.to(device), y.to(device))
                                val_sum += val_loss.detach().float()
                        record["val_loss"] = _mean_over_ranks(val_sum, world).item() / eval_batches
                    finally:
                        model.train()
                record["wall_time_s"] = (
                    start_wall_time_s + (time.perf_counter() - started)
                )
                metrics.write(json.dumps(record) + "\n")
                metrics.flush()
                if is_main:
                    logger.info("%s", json.dumps(record))
                if checkpoint_manager.should_save(step):
                    save()
                if max_session_s is not None:
                    out_of_time = time.monotonic() - session_started >= float(max_session_s)
                    if world > 1:  # every rank must stop on the same update
                        flag = torch.tensor([1.0 if out_of_time else 0.0], device=device)
                        dist.all_reduce(flag, op=dist.ReduceOp.MAX)
                        out_of_time = bool(flag.item())
                    if out_of_time:
                        logger.info("session time budget reached at update %d; saving", step)
                        break
            save()
    except KeyboardInterrupt:
        optimizer.zero_grad(set_to_none=True)
        try:
            save()
        except Exception:
            logger.exception("Checkpoint save failed during KeyboardInterrupt")
        raise
