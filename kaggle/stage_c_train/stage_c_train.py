# stage_c_train.py
# Kaggle GPU script kernel: Stage C PlaygroundLM training.
#
# Assumes kernel_sources contains:
#   - galacticaiteam/plm-stage-c-data-prep-s0 through -s7
#     (each producing stage-c/session-<N>/final/train.bin; s0 also val.bin)
#   - galacticaiteam/plm-stage-b-launch
#     (its /runs/checkpoints/ has step_00002035.safetensors + _state.pt,
#      for full optimizer/RNG continuity into Stage C)
#   - galacticaiteam/plm-stage-c-train (self-reference, resume metadata only)
#
# Flow:
#   1. Concatenate the 8 session train.bin files into /tmp/plm_stage_c/train.bin
#      (keeps ~5.7 GB off Kaggle's 20 GB /kaggle/working quota).
#   2. Copy session 0's val.bin to /tmp/plm_stage_c/val.bin.
#   3. Copy Stage B's step_00002035 checkpoint (weights + optimizer state) into
#      /kaggle/working/runs/ so CheckpointManager.resume() picks it up with
#      full continuity — no GH_TOKEN, no synthetic state.
#   4. Clone main, launch notebooks/playground_lm_train.py via torchrun with
#      PLM_STAGE=C, PLM_MAX_STEPS (default 31101 = 2035 + 29066, one pass over
#      2.857B), PLM_SCHEDULE_START=2035 (LR re-warms instead of jumping ~10x;
#      Claude) and PLM_SAMPLING=permute (windows without replacement, so one
#      pass covers every window instead of ~63%; Lumi, Claude's note).
#   5. Delete the concatenated train.bin/val.bin (rebuilt every session) so
#      they don't count against Kaggle's 20 GB saved output.
#
# Free-tools only: Kaggle 2x T4, no paid services.

import hashlib
import json
import os
import shutil
import subprocess
import sys
import time
from pathlib import Path

# --- Paths ----------------------------------------------------------------

WORKING = Path("/kaggle/working")
INPUT = Path("/kaggle/input")
REPO_DIR = WORKING / "playground-lm"
RUNS_DIR = WORKING / "runs"
# Joined train/val live under /tmp so the ~5.7 GB concat does not compete with
# checkpoints for Kaggle's 20 GB /kaggle/working quota. RUNS_DIR stays on
# /kaggle/working (persisted via the plm-stage-c-train self-reference mount).
CONCAT_DIR = Path("/tmp/plm_stage_c")

# One FineWeb pass: 2,857,393,718 uint16 tokens. Permutation resume depends on
# an identical joined train.bin across sessions — size is the hard gate;
# sha256 is recorded (and enforced when PLM_TRAIN_SHA256 is set).
EXPECTED_TRAIN_TOKENS = 2_857_393_718
EXPECTED_TRAIN_BYTES = EXPECTED_TRAIN_TOKENS * 2  # uint16

GIT_REPO = "https://github.com/galactic-ai-team/playground-lm.git"
GIT_BRANCH = "main"

KAGGLE_OWNER = "galacticaiteam"

# Stage B checkpoint mount (kernel_sources). NOTE (Lumi integration fix):
# Kaggle now mounts kernel_sources at /kaggle/input/notebooks/<owner>/<slug>/
# (confirmed in the Stage B launch log), not /kaggle/input/<slug>/. Paths are
# resolved at runtime by find_source_root() / find_in_source().
STAGE_B_SLUG = "plm-stage-b-launch"
STAGE_B_RESUME_STEP = 2035

# Previous version of THIS kernel (self-reference, resume metadata only).
STAGE_C_TRAIN_SLUG = "plm-stage-c-train"

# --- Config ---------------------------------------------------------------

N_SESSIONS = 8

# FINAL step number (absolute; the run resumes at Stage B's 2035). One pass:
# 2,857,393,718 tokens -> 2,790,423 non-overlapping 1024-token windows / 96
# sequences per step = 29,066 steps -> 2035 + 29,066 = 31,101 (matches the
# Stage C preset). Override via PLM_STAGE_C_MAX_STEPS at push time (e.g. 2535
# for a 500-step smoke).
STAGE_C_MAX_STEPS = int(os.environ.get("PLM_STAGE_C_MAX_STEPS", "31101"))

ENV_OVERRIDES = {
    "PLM_STAGE": "C",
    "PLM_MAX_STEPS": str(STAGE_C_MAX_STEPS),
    "PLM_SCHEDULE_START": str(STAGE_B_RESUME_STEP),  # re-warm LR (Claude)
    "PLM_SAMPLING": "permute",  # without replacement (Claude's coverage note)
    "PLM_BATCH_SIZE": "6",              # T4-friendly; grad_accum → 16
    # GPU quota: 2xT4 bills at 1x wall time (Kaggle product update, 2022).
    # 1 hour wall on 2xT4 = 1 hour GPU quota. 30h/week quota ≈ 30h wall
    # on 2xT4. Session max is 12h. At ~4.4s/step, 11h ≈ 9,000 steps.
    "PLM_MAX_SESSION_S": str(11 * 3600),  # under Kaggle's 12h GPU kill
    "PLM_DATA_DIR": str(CONCAT_DIR),      # holds train.bin / val.bin
    "PLM_RUN_DIR": str(RUNS_DIR),
    "PLM_RUN_ID": "stage-c",
    "PLM_REPO": "galactic-ai-team/playground-lm",
    "PLM_RELEASE_TAG": "stage-c-checkpoints",
    "PLM_UPLOAD_EVERY": "1",
}

# --- Helpers --------------------------------------------------------------

def log(msg: str) -> None:
    print(f"[stage-c-train] {msg}", flush=True)


def log_input_layout(max_depth: int = 4) -> None:
    """Log the /kaggle/input tree (dirs only, shallow) so mount-layout
    changes are visible in the kernel log."""
    if not INPUT.exists():
        log(f"{INPUT} does not exist")
        return
    log(f"{INPUT} layout (dirs, depth<={max_depth}):")
    base_depth = len(INPUT.parts)
    for root, dirs, _files in os.walk(INPUT):
        depth = len(Path(root).parts) - base_depth
        dirs[:] = sorted(d for d in dirs if d not in (".git", "playground-lm", "proc"))
        if depth >= max_depth:
            dirs[:] = []
        log(f"  {root}")


def find_source_root(slug: str) -> Path | None:
    """Locate the mount root of a kernel_sources entry. Tries the known
    layouts first, then walks /kaggle/input for a directory named <slug>."""
    candidates = [
        INPUT / "notebooks" / KAGGLE_OWNER / slug,
        INPUT / slug,
    ]
    for c in candidates:
        if c.is_dir():
            return c
    for root, dirs, _files in os.walk(INPUT):
        if len(Path(root).parts) - len(INPUT.parts) > 4:
            dirs[:] = []
            continue
        dirs[:] = [d for d in dirs if d not in (".git", "playground-lm")]
        if slug in dirs:
            return Path(root) / slug
    return None


def find_in_source(slug: str, filename: str, prefer: str = "") -> Path | None:
    """Find <filename> inside kernel source <slug>'s mount, skipping the
    playground-lm git clone that every launcher leaves in its output. If
    several match, prefer paths containing <prefer>."""
    root = find_source_root(slug)
    if root is None:
        return None
    hits = [
        p for p in root.rglob(filename)
        if "playground-lm" not in p.relative_to(root).parts and ".git" not in p.parts
    ]
    if not hits:
        return None
    hits.sort(key=lambda p: (prefer not in str(p), len(p.parts), str(p)))
    return hits[0]


def load_gh_token() -> str | None:
    """Optional GH_TOKEN for uploading Stage C checkpoints to the release.
    Training itself doesn't need it — Stage B resume reads from kernel_sources."""
    try:
        from kaggle_secrets import UserSecretsClient
        token = UserSecretsClient().get_secret("GH_TOKEN")
        if token:
            log("GH_TOKEN loaded from Kaggle Secrets")
            return token
    except Exception as e:
        log(f"Kaggle Secrets unavailable or secret not set ({e})")
    token = os.environ.get("GH_TOKEN")
    if token:
        log("GH_TOKEN loaded from env")
    return token


def _sha256_file(path: Path, chunk_size: int = 8 * 1024 * 1024) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        while True:
            chunk = f.read(chunk_size)
            if not chunk:
                break
            h.update(chunk)
    return h.hexdigest()


def _log_df(target: str = "/tmp") -> None:
    """Best-effort `df -h <target>` into the kernel log."""
    try:
        out = subprocess.check_output(["df", "-h", target], text=True)
        for line in out.strip().splitlines():
            log(f"df -h {target}: {line}")
    except Exception as e:
        log(f"df -h {target} failed: {e}")


def verify_train_bin(train_path: Path) -> str:
    """Hard-gate train.bin size; optionally enforce PLM_TRAIN_SHA256.
    Returns the computed sha256 hex digest."""
    size = train_path.stat().st_size
    tokens = size // 2
    log(f"train.bin size check: {size:,} bytes ({tokens:,} tokens) at {train_path}")
    if size != EXPECTED_TRAIN_BYTES or tokens != EXPECTED_TRAIN_TOKENS:
        raise RuntimeError(
            f"train.bin size mismatch: got {size:,} bytes / {tokens:,} tokens, "
            f"expected {EXPECTED_TRAIN_BYTES:,} bytes / {EXPECTED_TRAIN_TOKENS:,} "
            f"tokens. Permutation resume depends on an identical joined file — aborting."
        )
    log("train.bin size OK (matches EXPECTED_TRAIN_TOKENS)")

    digest = _sha256_file(train_path)
    log(f"train.bin sha256: {digest}")

    expected_sha = os.environ.get("PLM_TRAIN_SHA256", "").strip()
    if expected_sha:
        if digest.lower() != expected_sha.lower():
            raise RuntimeError(
                f"train.bin sha256 mismatch: got {digest}, "
                f"expected PLM_TRAIN_SHA256={expected_sha}. Aborting."
            )
        log("train.bin sha256 OK (matches PLM_TRAIN_SHA256)")
    else:
        log("PLM_TRAIN_SHA256 unset; size gate only (sha256 recorded for later sessions)")
    return digest


def write_train_fingerprint(digest: str) -> Path:
    """Persist sha256 under RUNS_DIR so later sessions can set PLM_TRAIN_SHA256."""
    RUNS_DIR.mkdir(parents=True, exist_ok=True)
    out = RUNS_DIR / "train_bin_fingerprint.txt"
    out.write_text(digest + "\n", encoding="utf-8")
    log(f"wrote {out} ({digest})")
    return out


def concat_session_bins() -> tuple[Path, Path]:
    """Concatenate all N sessions' train.bin into one file under CONCAT_DIR
    (/tmp). Copy session 0's val.bin. Returns (train_path, val_path)."""
    CONCAT_DIR.mkdir(parents=True, exist_ok=True)
    out_train = CONCAT_DIR / "train.bin"
    out_val = CONCAT_DIR / "val.bin"

    if out_train.exists():
        out_train.unlink()
    if out_val.exists():
        out_val.unlink()

    total_tokens = 0
    with open(out_train, "wb") as f_out:
        for n in range(N_SESSIONS):
            slug = f"plm-stage-c-data-prep-s{n}"
            src = find_in_source(slug, "train.bin", prefer=f"session-{n}/final")
            if src is None or not src.exists():
                raise FileNotFoundError(
                    f"session {n} train.bin not found under kernel source {slug} "
                    f"(mount root: {find_source_root(slug)}); see layout above"
                )
            n_tok = src.stat().st_size // 2
            log(f"concat s{n}: {n_tok:,} tokens from {src}")
            with open(src, "rb") as f_in:
                while True:
                    chunk = f_in.read(8 * 1024 * 1024)
                    if not chunk:
                        break
                    f_out.write(chunk)
            total_tokens += n_tok

    log(f"train.bin: {total_tokens:,} tokens at {out_train}")
    _log_df("/tmp")
    size = out_train.stat().st_size
    log(f"train.bin bytes={size:,} tokens={size // 2:,}")

    val_src = find_in_source(
        "plm-stage-c-data-prep-s0", "val.bin", prefer="session-0/final"
    )
    if val_src is None or not val_src.exists():
        raise FileNotFoundError(
            "session 0 val.bin not found under kernel source plm-stage-c-data-prep-s0"
        )
    log(f"val.bin source: {val_src}")
    shutil.copy(val_src, out_val)
    log(f"val.bin: {out_val.stat().st_size // 2:,} tokens copied from s0")

    return out_train, out_val


def copy_previous_train_runs() -> bool:
    """Copy previous Stage C session's runs/ into RUNS_DIR. Enables torchrun
    resume across Kaggle GPU sessions."""
    root = find_source_root(STAGE_C_TRAIN_SLUG)
    PREV_TRAIN_RUNS = (root / "runs") if root else INPUT / STAGE_C_TRAIN_SLUG / "runs"
    if not PREV_TRAIN_RUNS.exists():
        log(f"no previous Stage C runs at {PREV_TRAIN_RUNS}")
        return False
    if not (PREV_TRAIN_RUNS / "manifest.json").exists():
        log("previous Stage C runs dir exists but no manifest.json")
        return False

    log(f"copying previous Stage C runs from {PREV_TRAIN_RUNS} → {RUNS_DIR}")
    if RUNS_DIR.exists():
        shutil.rmtree(RUNS_DIR)
    RUNS_DIR.parent.mkdir(parents=True, exist_ok=True)
    shutil.copytree(PREV_TRAIN_RUNS, RUNS_DIR)
    n_files = sum(1 for _ in RUNS_DIR.rglob("*") if _.is_file())
    log(f"copied {n_files} files from previous Stage C session")
    return True


def stage_b_checkpoint_paths() -> tuple[Path, Path]:
    """Return (weights_path, state_path) for the Stage B final checkpoint
    from the kernel_sources mount."""
    name = f"step_{STAGE_B_RESUME_STEP:08d}.safetensors"
    found = find_in_source(STAGE_B_SLUG, name, prefer="runs/checkpoints")
    if found is not None:
        weights = found
    else:
        root = find_source_root(STAGE_B_SLUG) or (INPUT / STAGE_B_SLUG)
        weights = root / "runs" / "checkpoints" / name
    state = weights.parent / f"step_{STAGE_B_RESUME_STEP:08d}_state.pt"
    log(f"Stage B checkpoint lookup: {weights}")
    return weights, state


def copy_stage_b_checkpoint() -> bool:
    """Copy Stage B's final checkpoint (weights + optimizer state) into
    RUNS_DIR so CheckpointManager.resume() picks it up with full continuity.
    Reads from the plm-stage-b-launch kernel mount — no GH_TOKEN needed.
    """
    if (RUNS_DIR / "manifest.json").exists():
        log("Stage C resume state already present; skipping Stage B copy")
        return True

    weights, state = stage_b_checkpoint_paths()
    STAGE_B_RUNS = weights.parent.parent
    if not weights.exists():
        log(f"FATAL: Stage B weights not found at {weights}")
        log(f"  available under {STAGE_B_RUNS / 'checkpoints'}:")
        ckpt_dir = STAGE_B_RUNS / "checkpoints"
        if ckpt_dir.exists():
            for p in sorted(ckpt_dir.iterdir()):
                log(f"    {p.name}")
        return False
    if not state.exists():
        log(f"FATAL: Stage B optimizer state not found at {state}")
        return False

    ckpt_dir = RUNS_DIR / "checkpoints"
    ckpt_dir.mkdir(parents=True, exist_ok=True)
    shutil.copy(weights, ckpt_dir / weights.name)
    shutil.copy(state, ckpt_dir / state.name)
    log(f"copied {weights.name} ({weights.stat().st_size // 1024 // 1024} MB)")
    log(f"copied {state.name} ({state.stat().st_size // 1024 // 1024} MB)")

    # Build a CheckpointManager-shaped manifest pointing at the copied files.
    # If Stage B had a manifest in its output, use its token/wall_time values;
    # otherwise fall back to known figures (200M tokens, milestone).
    tokens_seen = 200_000_000
    wall_time_s = 0.0
    stage_b_manifest = STAGE_B_RUNS / "manifest.json"
    if stage_b_manifest.exists():
        try:
            with open(stage_b_manifest) as f:
                bm = json.load(f)
            for entry in bm.get("checkpoints", []):
                if entry.get("step") == STAGE_B_RESUME_STEP:
                    tokens_seen = entry.get("tokens_seen", tokens_seen)
                    wall_time_s = entry.get("wall_time_s", wall_time_s)
                    break
        except Exception as e:
            log(f"WARN: could not read Stage B manifest ({e}); using defaults")

    synthetic = {
        "run_id": "stage-b-final",
        "config": {},
        "checkpoints": [{
            "step": STAGE_B_RESUME_STEP,
            "path": f"checkpoints/{weights.name}",
            "tokens_seen": tokens_seen,
            "wall_time_s": wall_time_s,
            "created_at": time.time(),
            "is_milestone": True,
        }],
        "latest_step": STAGE_B_RESUME_STEP,
        "latest_tokens_seen": tokens_seen,
        "latest_wall_time_s": wall_time_s,
    }
    with open(RUNS_DIR / "manifest.json", "w") as f:
        json.dump(synthetic, f, indent=2)

    log(
        f"Stage B checkpoint staged at {RUNS_DIR}: step {STAGE_B_RESUME_STEP}, "
        f"{tokens_seen:,} tokens, full optimizer/RNG continuity"
    )
    return True


def _release_has_asset(tag: str, asset_name: str, token: str | None) -> bool:
    """Return True if `gh release view` shows <asset_name> on <tag>.
    Skips (returns False) when gh/network/token unavailable."""
    if not token:
        return False
    env = os.environ.copy()
    env["GH_TOKEN"] = token
    try:
        proc = subprocess.run(
            ["gh", "release", "view", tag, "--repo", ENV_OVERRIDES["PLM_REPO"], "--json", "assets"],
            capture_output=True, text=True, env=env, timeout=60,
        )
    except FileNotFoundError:
        log("gh not installed; cannot confirm release assets")
        return False
    except Exception as e:
        log(f"gh release view failed ({e}); skipping delete")
        return False
    if proc.returncode != 0:
        log(f"gh release view {tag} failed: {(proc.stderr or proc.stdout).strip()[:200]}")
        return False
    try:
        assets = json.loads(proc.stdout).get("assets") or []
    except Exception as e:
        log(f"could not parse gh release view JSON ({e})")
        return False
    names = {a.get("name") for a in assets if isinstance(a, dict)}
    return asset_name in names


def free_working_disk_if_tight(token: str | None) -> None:
    """Optionally delete older milestone .safetensors already on GH releases
    stage-b-checkpoints / stage-c-s1-step10907 so /kaggle/working has room for
    new checkpoints. Keeps the newest ckpt + its _state.pt + manifest.json +
    metrics. Never deletes without confirming the release asset exists."""
    ckpt_dir = RUNS_DIR / "checkpoints"
    if not ckpt_dir.is_dir():
        log("no checkpoints dir yet; skipping disk free")
        return

    # Rough free-space check on /kaggle/working.
    try:
        usage = shutil.disk_usage(str(WORKING))
        free_gib = usage.free / (1024 ** 3)
        log(f"/kaggle/working free ≈ {free_gib:.2f} GiB")
        if free_gib >= 8.0:
            log("working disk has ≥8 GiB free; no milestone prune")
            return
    except Exception as e:
        log(f"disk_usage failed ({e}); attempting conservative prune anyway")

    if not token:
        log("no GH_TOKEN — cannot confirm release assets; skipping milestone deletes")
        return

    weights = sorted(ckpt_dir.glob("step_*.safetensors"))
    if len(weights) <= 1:
        log(f"only {len(weights)} weight file(s); nothing to prune")
        return

    # Newest by step number embedded in the filename.
    def step_of(p: Path) -> int:
        try:
            return int(p.stem.split("_")[1])
        except Exception:
            return -1

    weights.sort(key=step_of)
    newest = weights[-1]
    keep = {newest.name, newest.name.replace(".safetensors", "_state.pt")}
    log(f"keeping newest checkpoint set: {sorted(keep)}")

    # Known release homes for older milestones.
    release_for = {
        2035: "stage-b-checkpoints",
        10907: "stage-c-s1-step10907",
    }

    for w in weights[:-1]:
        step = step_of(w)
        tag = release_for.get(step)
        if tag is None:
            log(f"skip delete {w.name}: no known release tag for step {step}")
            continue
        if not _release_has_asset(tag, w.name, token):
            log(f"skip delete {w.name}: not confirmed on release {tag}")
            continue
        for victim in (w, w.parent / w.name.replace(".safetensors", "_state.pt")):
            if victim.name in keep:
                continue
            if victim.exists():
                try:
                    victim.unlink()
                    log(f"deleted {victim} (confirmed on {tag})")
                except Exception as e:
                    log(f"failed to delete {victim}: {e}")


def clone_repo() -> None:
    if REPO_DIR.exists():
        log(f"{REPO_DIR} exists; pulling")
        subprocess.run(["git", "-C", str(REPO_DIR), "pull", "--ff-only"], check=True)
        return
    log(f"cloning {GIT_REPO} ({GIT_BRANCH})")
    subprocess.run(
        ["git", "clone", "--depth", "1", "--branch", GIT_BRANCH, GIT_REPO, str(REPO_DIR)],
        check=True,
    )


def _count_gpus() -> int:
    try:
        import torch
        return max(1, torch.cuda.device_count())
    except Exception:
        return 1


# --- Main -----------------------------------------------------------------

def main() -> int:
    log(f"starting Stage C training launch (max_steps={STAGE_C_MAX_STEPS})")
    log(f"CONCAT_DIR={CONCAT_DIR} (train/val under /tmp); RUNS_DIR={RUNS_DIR}")
    log_input_layout()

    # 1. Concatenate the 8 sessions under /tmp and hard-gate size (+ optional sha).
    train_path, _val_path = concat_session_bins()
    try:
        digest = verify_train_bin(train_path)
    except RuntimeError as e:
        log(f"FATAL: {e}")
        return 1

    # 2. Resume: prefer a previous Stage C session, else Stage B.
    resumed_c = copy_previous_train_runs()
    if not resumed_c:
        if not copy_stage_b_checkpoint():
            log("FATAL: could not prepare resume state from Stage B")
            return 1

    # Record sha256 after runs/ is in place (survives via self-ref mount).
    write_train_fingerprint(digest)

    # 3. GH_TOKEN for uploading Stage C checkpoints (optional).
    token = load_gh_token()
    if token:
        ENV_OVERRIDES["GH_TOKEN"] = token
    else:
        log("no GH_TOKEN — Stage C checkpoints stay in /kaggle/working/runs and "
            "are carried forward via the plm-stage-c-train self-reference mount")

    # 3b. If /kaggle/working is tight, drop older milestones already on GH releases.
    free_working_disk_if_tight(token)

    # 4. Clone repo.
    clone_repo()

    # 5. Launch torchrun.
    env = os.environ.copy()
    env.update(ENV_OVERRIDES)

    notebook = REPO_DIR / "notebooks" / "playground_lm_train.py"
    if not notebook.exists():
        log(f"FATAL: {notebook} not found")
        return 1

    nproc = _count_gpus()
    log(f"launching torchrun --nproc_per_node={nproc}")
    cmd = [
        sys.executable, "-m", "torch.distributed.run",
        "--standalone",
        f"--nproc_per_node={nproc}",
        str(notebook),
    ]
    result = subprocess.run(cmd, env=env, cwd=str(REPO_DIR))
    log(f"torchrun exited with code {result.returncode}")

    # 6. The concatenated data is rebuilt every session from the data-prep
    #    mounts; drop it so it isn't left on /tmp either.
    for p in (CONCAT_DIR / "train.bin", CONCAT_DIR / "val.bin"):
        try:
            p.unlink()
            log(f"removed {p} (rebuilt next session)")
        except FileNotFoundError:
            pass
    return result.returncode


if __name__ == "__main__":
    sys.exit(main())
