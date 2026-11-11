# kaggle/stage_g_bench/stage_g_bench.py
# Stage G throughput benchmark (design: Dee; DDP/torchrun harness: Lumi glue).
#
# Measures training throughput on free Kaggle T4 x2 for:
#   124M control (12 x 768, micro-batch 6 = the Stage C setting, to calibrate vs the
#   measured ~22k tok/s), G-depth 24 x 768 (~210M), G-full 24 x 1024 / 16 heads (~355M).
# Real DDP over 2 ranks (torchrun), fp16 AMP + GradScaler, AdamW(0.9, 0.95), grad accum 8
# with no_sync on non-final micro-steps, random tokens (throughput only; no data needed).
# A per-rank OOM probe (one full optimizer step without DDP) runs first so one rank
# can't OOM inside an all-reduce and hang the other.
#
# Output: /kaggle/working/bench_results.json, rewritten after every combo (partial
# results survive a timeout). Hard cap 75 min wall. Free Kaggle only. No secrets.
#
#   python stage_g_bench.py            # Kaggle entry: clone repo, torchrun workers
#   python stage_g_bench.py --dry-run  # CPU, tiny dims, no GPU, no network

from __future__ import annotations

import argparse
import json
import os
import subprocess
import sys
import time
from pathlib import Path

WORKING = Path(os.environ.get("PLM_WORKING", "/kaggle/working"))
REPO_DIR = WORKING / "playground-lm"
GIT_REPO = "https://github.com/galactic-ai-team/playground-lm.git"
GIT_BRANCH = os.environ.get("PLM_GIT_BRANCH", "feat/stage-g-scratchpad")
RESULTS_PATH = WORKING / "bench_results.json"

CONFIGS = [
    {"name": "124M_control_12x768", "n_layers": 12, "n_heads": 12, "d_model": 768, "micro_batches": [6]},
    {"name": "G_depth_24x768", "n_layers": 24, "n_heads": 12, "d_model": 768, "micro_batches": [2, 4]},
    {"name": "G_full_24x1024", "n_layers": 24, "n_heads": 16, "d_model": 1024, "micro_batches": [1, 2, 3]},
]
CONTEXT_LEN = 1024
VOCAB = 50257
GRAD_ACCUM = 8
WARMUP_STEPS = int(os.environ.get("PLM_BENCH_WARMUP", "3"))
TIMED_STEPS = int(os.environ.get("PLM_BENCH_STEPS", "10"))
HARD_CAP_SECONDS = 75 * 60
MEM_LIMIT_GIB = 14.5  # T4 has ~14.7 GiB usable; leave headroom for DDP buckets


def log(msg: str) -> None:
    print(f"[bench] {msg}", flush=True)


def count_params(n_layers: int, d_model: int, vocab: int = VOCAB, ctx: int = CONTEXT_LEN) -> int:
    """Exact PlaygroundLM parameter count (tied embeddings)."""
    per_block = 12 * d_model * d_model + 13 * d_model  # qkv+proj+fc+proj weights/biases + 2 LN
    return vocab * d_model + ctx * d_model + n_layers * per_block + 2 * d_model


def tokens_per_step(world: int, micro_batch: int, accum: int = GRAD_ACCUM, ctx: int = CONTEXT_LEN) -> int:
    return world * micro_batch * accum * ctx


def summarize(result: dict, world: int) -> dict:
    s = result.get("s_per_step")
    if s:
        tps = tokens_per_step(world, result["micro_batch"]) / s
        result["tok_per_s"] = round(tps, 1)
        result["tokens_per_session_hour"] = round(tps * 3600)
        result["hours_for"] = {f"{b / 1e9:g}B": round(b / (tps * 3600), 1) for b in (1.0e9, 1.5e9, 2.86e9, 7.1e9)}
    return result


def _write(results: list[dict], meta: dict) -> None:
    RESULTS_PATH.parent.mkdir(parents=True, exist_ok=True)
    RESULTS_PATH.write_text(json.dumps({**meta, "results": results}, indent=2) + "\n")


def _step(model, opt, scaler, micro_batch, device, ddp_model=None, autocast=None):
    import torch
    m = ddp_model or model
    opt.zero_grad(set_to_none=True)
    for i in range(GRAD_ACCUM):
        x = torch.randint(0, VOCAB, (micro_batch, CONTEXT_LEN), device=device)
        y = torch.randint(0, VOCAB, (micro_batch, CONTEXT_LEN), device=device)
        last = i == GRAD_ACCUM - 1
        ctx = m.no_sync() if (ddp_model is not None and not last) else _null()
        with ctx:
            with autocast():
                _, loss = m(x, y)
            scaler.scale(loss / GRAD_ACCUM).backward()
    scaler.step(opt)
    scaler.update()


class _null:
    def __enter__(self):
        return self

    def __exit__(self, *a):
        return False


def worker() -> int:
    import torch
    import torch.distributed as dist
    from torch.nn.parallel import DistributedDataParallel as DDP

    sys.path.insert(0, str(REPO_DIR / "src"))
    from playground_lm.model import PlaygroundLM

    local_rank = int(os.environ.get("LOCAL_RANK", "0"))
    world = int(os.environ.get("WORLD_SIZE", "1"))
    rank = int(os.environ.get("RANK", "0"))
    torch.cuda.set_device(local_rank)
    device = torch.device("cuda", local_rank)
    try:
        dist.init_process_group("nccl", device_id=device)
    except TypeError:  # torch < 2.3
        dist.init_process_group("nccl")
    t_start = time.perf_counter()
    meta = {"kind": "stage_g_throughput_bench", "world_size": world, "grad_accum": GRAD_ACCUM,
            "context_len": CONTEXT_LEN, "warmup_steps": WARMUP_STEPS, "timed_steps": TIMED_STEPS,
            "gpu": torch.cuda.get_device_name(device), "torch": torch.__version__,
            "precision": "fp16 autocast + GradScaler", "data": "random tokens"}
    results: list[dict] = []
    autocast = lambda: torch.autocast("cuda", dtype=torch.float16)  # noqa: E731

    for cfg in CONFIGS:
        for mb in cfg["micro_batches"]:
            if time.perf_counter() - t_start > HARD_CAP_SECONDS - 300:
                if rank == 0:
                    log("hard cap near; stopping")
                break
            r = {"config": cfg["name"], "n_layers": cfg["n_layers"], "d_model": cfg["d_model"],
                 "n_heads": cfg["n_heads"], "params": count_params(cfg["n_layers"], cfg["d_model"]),
                 "micro_batch": mb, "tokens_per_step": tokens_per_step(world, mb),
                 "oom": False, "error": None, "s_per_step": None, "peak_gib_max_rank": None}
            model = opt = ddp = None
            torch.cuda.empty_cache()
            torch.cuda.reset_peak_memory_stats(device)
            ok = torch.ones(1, device=device)
            try:
                model = PlaygroundLM(vocab_size=VOCAB, n_layers=cfg["n_layers"], n_heads=cfg["n_heads"],
                                     d_model=cfg["d_model"], context_len=CONTEXT_LEN).to(device)
                opt = torch.optim.AdamW(model.parameters(), lr=1e-4, weight_decay=0.1, betas=(0.9, 0.95))
                scaler = torch.amp.GradScaler("cuda")
                _step(model, opt, scaler, mb, device, autocast=autocast)  # local OOM probe
                torch.cuda.synchronize(device)
                probe = torch.cuda.max_memory_allocated(device) / 2**30
                bucket = 4 * sum(p.numel() for p in model.parameters()) / 2**30  # DDP fp32 buckets
                if probe + bucket > MEM_LIMIT_GIB:
                    raise torch.cuda.OutOfMemoryError(f"predicted {probe + bucket:.2f} GiB > {MEM_LIMIT_GIB}")
            except torch.cuda.OutOfMemoryError as e:
                ok.zero_()
                r["oom"], r["error"] = True, str(e).split("\n")[0][:200]
            except Exception as e:  # noqa: BLE001
                ok.zero_()
                r["error"] = f"{type(e).__name__}: {e}"[:200]
            dist.all_reduce(ok, op=dist.ReduceOp.MIN)
            if ok.item() == 1:
                try:
                    ddp = DDP(model, device_ids=[local_rank])
                    for _ in range(WARMUP_STEPS):
                        _step(model, opt, scaler, mb, device, ddp_model=ddp, autocast=autocast)
                    torch.cuda.synchronize(device)
                    dist.barrier()
                    t0 = time.perf_counter()
                    for _ in range(TIMED_STEPS):
                        _step(model, opt, scaler, mb, device, ddp_model=ddp, autocast=autocast)
                    torch.cuda.synchronize(device)
                    dist.barrier()
                    r["s_per_step"] = round((time.perf_counter() - t0) / TIMED_STEPS, 4)
                except torch.cuda.OutOfMemoryError as e:
                    r["oom"], r["error"] = True, str(e).split("\n")[0][:200]
            elif rank == 0 and not r["error"]:
                r["oom"], r["error"] = True, "another rank failed the OOM probe"
            peak = torch.tensor([torch.cuda.max_memory_allocated(device) / 2**30], device=device)
            dist.all_reduce(peak, op=dist.ReduceOp.MAX)
            r["peak_gib_max_rank"] = round(peak.item(), 2)
            summarize(r, world)
            del ddp, opt, model
            torch.cuda.empty_cache()
            if rank == 0:
                results.append(r)
                log(f"{r['config']} mb={mb}: s/step={r['s_per_step']} tok/s={r.get('tok_per_s')} "
                    f"peak={r['peak_gib_max_rank']} GiB oom={r['oom']} {r['error'] or ''}")
                _write(results, {**meta, "elapsed_s": round(time.perf_counter() - t_start, 1)})
    dist.barrier()
    dist.destroy_process_group()
    return 0


def clone_repo() -> None:
    if REPO_DIR.exists():
        subprocess.run(["git", "-C", str(REPO_DIR), "fetch", "--depth", "1", "origin", GIT_BRANCH], check=True)
        subprocess.run(["git", "-C", str(REPO_DIR), "checkout", "--force", f"origin/{GIT_BRANCH}"], check=True)
    else:
        subprocess.run(["git", "clone", "--depth", "1", "--branch", GIT_BRANCH, GIT_REPO, str(REPO_DIR)], check=True)


def dry_run(repo: Path) -> dict:
    """CPU smoke with tiny dims: builds each config shape scaled down, 1 step each."""
    import torch
    sys.path.insert(0, str(repo / "src"))
    from playground_lm.model import PlaygroundLM
    out = []
    for cfg in CONFIGS:
        heads = cfg["n_heads"] // 4  # 3 / 3 / 4
        m = PlaygroundLM(vocab_size=101, n_layers=cfg["n_layers"] // 6, n_heads=heads,
                         d_model=8 * heads, context_len=16)
        opt = torch.optim.AdamW(m.parameters(), lr=1e-3)
        x = torch.randint(0, 101, (2, 16))
        _, loss = m(x, x)
        loss.backward()
        opt.step()
        out.append({"config": cfg["name"], "loss": float(loss.detach()), "finite": bool(torch.isfinite(loss.detach()))})
    log(f"DRY RUN OK: {out}")
    return {"dry_run": True, "results": out}


def main(argv: list[str] | None = None) -> int:
    p = argparse.ArgumentParser(description="Stage G throughput benchmark")
    p.add_argument("--dry-run", action="store_true", help="CPU tiny-dims smoke; no GPU, no network")
    p.add_argument("--worker", action="store_true", help=argparse.SUPPRESS)
    args = p.parse_args(argv)
    if args.dry_run:
        dry_run(Path(__file__).resolve().parents[2])
        return 0
    if args.worker:
        return worker()
    clone_repo()
    import torch
    nproc = max(1, torch.cuda.device_count())
    log(f"GPUs: {nproc}; launching torchrun")
    cmd = [sys.executable, "-m", "torch.distributed.run", "--standalone", f"--nproc_per_node={nproc}",
           str(Path(__file__).resolve()), "--worker"]
    try:
        rc = subprocess.run(cmd, timeout=HARD_CAP_SECONDS).returncode
    except subprocess.TimeoutExpired:
        log("hard cap hit; partial results (if any) are in bench_results.json")
        rc = 124
    if RESULTS_PATH.exists():
        print(RESULTS_PATH.read_text())
    return rc


if __name__ == "__main__":
    sys.exit(main())
