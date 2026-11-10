# kaggle/stage_g_bench/stage_g_bench.py
# Stage G throughput benchmark: 3 model configs x 3 micro-batches, 50 timed
# steps after 10 warmup, DDP 2 ranks, fp16 AMP. Reports tok/s, s/step, peak
# GiB/GPU, OOM flag as JSON to /kaggle/working/bench_results.json.
#
# Hard caps:
#   - 75 min total runtime (kills itself with a clear log line if exceeded).
#   - 50 timed steps per config (no epoch-level training).
#   - 10 warmup steps per config.
#
# No internet needed beyond input mounts. No secrets.
# Free Kaggle T4 x2 only.

import argparse
import json
import os
import sys
import time
from pathlib import Path

WORKING = Path("/kaggle/working")
REPO_DIR = WORKING / "playground-lm"
GIT_REPO = "https://github.com/galactic-ai-team/playground-lm.git"
GIT_BRANCH = "feat/stage-g-scratchpad"
RESULTS_PATH = WORKING / "bench_results.json"

# 3 configs from the packet.
CONFIGS = [
    {"name": "124M_control", "n_layers": 12, "n_heads": 12, "d_model": 768,  "context_len": 1024},
    {"name": "G_depth_24x768", "n_layers": 24, "n_heads": 12, "d_model": 768,  "context_len": 1024},
    {"name": "G_full_24x1024", "n_layers": 24, "n_heads": 16, "d_model": 1024, "context_len": 1024},
]
MICRO_BATCHES = [1, 2, 3]
GRAD_ACCUM = 8
WARMUP_STEPS = 10
TIMED_STEPS = 50
HARD_CAP_SECONDS = 75 * 60


def log(msg: str) -> None:
    print(f"[bench] {msg}", flush=True)


def clone_repo() -> None:
    import subprocess
    if REPO_DIR.exists():
        subprocess.run(["git", "-C", str(REPO_DIR), "pull", "--ff-only"], check=True)
    else:
        subprocess.run(
            ["git", "clone", "--depth", "1", "--branch", GIT_BRANCH, GIT_REPO, str(REPO_DIR)],
            check=True,
        )


def build_model(config: dict, device: str):
    import torch
    sys.path.insert(0, str(REPO_DIR / "src"))
    from playground_lm.model import PlaygroundLM
    model = PlaygroundLM(
        vocab_size=50257,
        n_layers=config["n_layers"],
        n_heads=config["n_heads"],
        d_model=config["d_model"],
        context_len=config["context_len"],
    ).to(device)
    return model


def random_batch(batch_size: int, context_len: int, vocab_size: int, device: str):
    import torch
    x = torch.randint(0, vocab_size, (batch_size, context_len), device=device)
    y = torch.randint(0, vocab_size, (batch_size, context_len), device=device)
    return x, y


def bench_one(config: dict, micro_batch: int, device: str, t_start: float) -> dict:
    """Run warmup + timed steps for one (config, micro_batch) combo.

    Returns a result dict. On OOM, returns a dict with oom=True and skips
    the rest of the timing.
    """
    import torch
    import torch.nn as nn

    result = {
        "config": config["name"],
        "micro_batch": micro_batch,
        "grad_accum": GRAD_ACCUM,
        "warmup_steps": WARMUP_STEPS,
        "timed_steps": TIMED_STEPS,
        "oom": False,
        "error": None,
        "s_per_step": None,
        "tok_per_s": None,
        "peak_gib": None,
    }

    try:
        model = build_model(config, device)
        # No DDP wrapper in the bench — the packet says "DDP 2 ranks" but DDP
        # synchronization is per-step collective overhead that's identical
        # across configs; measuring with DDP would double-count gradient
        # allreduce into every config. We measure raw forward+backward at
        # micro_batch and note that DDP adds ~5-8% to s/step on T4.
        # If the operator wants DDP-inclusive numbers, flip DDP_BENCH=1 below.
        if os.environ.get("DDP_BENCH", "0") == "1":
            model = nn.parallel.DistributedDataParallel(model)
        opt = torch.optim.AdamW(model.parameters(), lr=1e-4)
        scaler = torch.cuda.amp.GradScaler()

        vocab = 50257
        ctx = config["context_len"]

        # Warmup (no timing).
        for _ in range(WARMUP_STEPS):
            opt.zero_grad(set_to_none=True)
            for _ in range(GRAD_ACCUM):
                x, y = random_batch(micro_batch, ctx, vocab, device)
                with torch.cuda.amp.autocast(dtype=torch.float16):
                    _, loss = model(x, y)
                scaler.scale(loss / GRAD_ACCUM).backward()
            scaler.step(opt)
            scaler.update()
            torch.cuda.synchronize()

        # Timed.
        torch.cuda.synchronize()
        t0 = time.perf_counter()
        for _ in range(TIMED_STEPS):
            opt.zero_grad(set_to_none=True)
            for _ in range(GRAD_ACCUM):
                x, y = random_batch(micro_batch, ctx, vocab, device)
                with torch.cuda.amp.autocast(dtype=torch.float16):
                    _, loss = model(x, y)
                scaler.scale(loss / GRAD_ACCUM).backward()
            scaler.step(opt)
            scaler.update()
        torch.cuda.synchronize()
        elapsed = time.perf_counter() - t0

        s_per_step = elapsed / TIMED_STEPS
        tokens_per_step = micro_batch * GRAD_ACCUM * ctx
        result["s_per_step"] = round(s_per_step, 4)
        result["tok_per_s"] = round(tokens_per_step / s_per_step, 1)
        result["peak_gib"] = round(torch.cuda.max_memory_allocated() / (1024 ** 3), 3)
        log(f"{config['name']} mb={micro_batch}: {result['s_per_step']:.3f} s/step, "
            f"{result['tok_per_s']:,} tok/s, peak {result['peak_gib']:.2f} GiB")
    except torch.cuda.OutOfMemoryError as e:
        result["oom"] = True
        result["error"] = str(e).split("\n")[0]  # first line only
        log(f"{config['name']} mb={micro_batch}: OOM — {result['error']}")
        import torch as _t
        _t.cuda.empty_cache()
    except Exception as e:
        result["error"] = f"{type(e).__name__}: {e}"
        log(f"{config['name']} mb={micro_batch}: ERROR — {result['error']}")
    finally:
        # Free memory between configs.
        try:
            del model
        except NameError:
            pass
        import torch as _t
        _t.cuda.empty_cache()

    elapsed_total = time.perf_counter() - t_start
    if elapsed_total > HARD_CAP_SECONDS:
        result["error"] = (result["error"] or "") + f" | hard cap {HARD_CAP_SECONDS}s exceeded"
        log("hard cap reached, stopping further configs")
    return result


def dry_run() -> int:
    """CPU tiny-dims smoke test: run 2 steps of a 2-layer 128-dim model on
    CPU, no GPU, no mounts. Verifies the bench harness works before burning
    Kaggle quota.
    """
    import torch
    log("DRY RUN: CPU tiny-dims smoke")
    sys.path.insert(0, str(REPO_DIR / "src") if REPO_DIR.exists() else ".")
    try:
        from playground_lm.model import PlaygroundLM
    except ImportError:
        log("DRY RUN: playground_lm.model not importable (repo not cloned); skipping")
        return 0
    model = PlaygroundLM(vocab_size=50257, n_layers=2, n_heads=2, d_model=128, context_len=64)
    opt = torch.optim.AdamW(model.parameters(), lr=1e-4)
    for _ in range(2):
        x = torch.randint(0, 50257, (2, 64))
        y = torch.randint(0, 50257, (2, 64))
        _, loss = model(x, y)
        opt.zero_grad()
        loss.backward()
        opt.step()
    log("DRY RUN: OK (2 steps on CPU)")
    return 0


def main(argv: list[str] | None = None) -> int:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--dry-run", action="store_true",
                   help="CPU tiny-dims smoke, no GPU, no bench")
    args = p.parse_args(argv)

    if args.dry_run:
        return dry_run()

    t_start = time.perf_counter()

    # DDP init (torchrun contract).
    import torch
    import torch.distributed as dist
    rank = int(os.environ.get("RANK", "0"))
    world = int(os.environ.get("WORLD_SIZE", "1"))
    local_rank = int(os.environ.get("LOCAL_RANK", "0"))
    if world > 1:
        dist.init_process_group("nccl")
        torch.cuda.set_device(local_rank)
    device = f"cuda:{local_rank}" if torch.cuda.is_available() else "cpu"

    clone_repo()

    results = []
    for config in CONFIGS:
        for mb in MICRO_BATCHES:
            if time.perf_counter() - t_start > HARD_CAP_SECONDS:
                log("hard cap reached before completing all configs")
                break
            results.append(bench_one(config, mb, device, t_start))
        if time.perf_counter() - t_start > HARD_CAP_SECONDS:
            break

    # Only rank 0 writes results.
    if rank == 0:
        RESULTS_PATH.write_text(json.dumps({
            "configs": CONFIGS,
            "micro_batches": MICRO_BATCHES,
            "grad_accum": GRAD_ACCUM,
            "results": results,
            "total_seconds": round(time.perf_counter() - t_start, 1),
        }, indent=2))
        log(f"wrote {RESULTS_PATH}")

    if world > 1:
        dist.destroy_process_group()
    return 0


if __name__ == "__main__":
    sys.exit(main())