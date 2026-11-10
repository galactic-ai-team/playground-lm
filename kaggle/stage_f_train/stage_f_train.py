# kaggle/stage_f_train/stage_f_train.py
# Stage F: fact/arith SFT continuation from Stage E step 31140 (v0.5 Chat).
# Seeds from the plm-e31140-seed dataset (weights + manifest, no optimizer state).
# SFT data: plm-stage-f-data (facts/ + chat/ streams for the three-way mixer;
# combined train.bin remains at the dataset root). Replay: plm-stageb-data.
# Free T4 only.
#
# Contract (from data/stage_e/SCHEMA.md):
#   PLM_DATA_DIR holds {train,val}.bin (uint16 tokens) and
#   {train,val}.masks.bin (uint8, 1=supervised). Lengths must match.
#
# Resume safety: never delete an existing RUNS_DIR before a verified copy.
# Copy to RUNS_DIR.tmp, verify manifest, then os.replace the staging dir
# into place. If any step fails, RUNS_DIR is untouched.
#
# Stage E preset comes from stage_config.STAGE_PRESETS["E"] (lr 3e-5,
# schedule_start 31101, effective_batch 24, eval_interval 25).
#
# Free Kaggle only. PLM_DRY_RUN=1 runs a single CPU step with no GPU quota.
# No real GPU training in this file — it is launch-ready but not launched.

import json
import os
import shutil
import subprocess
import sys
import time
from pathlib import Path


WORKING = Path("/kaggle/working")
INPUT = Path("/kaggle/input")
REPO_DIR = WORKING / "playground-lm"
RUNS_DIR = WORKING / "runs"
DATA_DIR = Path("/tmp/plm_stage_e")
CHAT_DIR = Path("/tmp/plm_stage_f_chat")
GIT_REPO = "https://github.com/galactic-ai-team/playground-lm.git"
GIT_BRANCH = os.environ.get("PLM_GIT_BRANCH", "feat/stage-f-facts-arith")
GIT_COMMIT = os.environ.get("PLM_GIT_COMMIT", "").strip()  # immutable pin when set

STAGE_C_SLUG = "plm-stage-c-train"
STAGE_E_SLUG = "plm-stage-e-train"
STAGE_F_DATA_SLUG = "plm-stage-f-data"
STAGE_F_SEED_SLUG = "plm-e31140-seed"
STAGE_C_FINAL_STEP = 31101
STAGE_F_SEED_STEP = int(os.environ.get("PLM_SEED_STEP", "31140"))
STAGE_F_SEED_TOKENS = int(os.environ.get("PLM_SEED_TOKENS", "3058311168"))
STAGE_F_SEED_SHA256 = os.environ.get(
    "PLM_SEED_SHA256",
    "ce1fa2098236656d149437638601dfafa7744ab3f69ff52739be0c95199f818b",
)

FINETUNE_EPOCHS = float(os.environ.get("PLM_FINETUNE_EPOCHS", "1"))
DRY_RUN = os.environ.get("PLM_DRY_RUN", "0") == "1"
MIN_DISK_GB = float(os.environ.get("PLM_MIN_DISK_GB", "8.0"))
QUOTA_GUARD = os.environ.get("PLM_QUOTA_GUARD", "0") == "1"
QUOTA_LIMIT_H = float(os.environ.get("PLM_QUOTA_LIMIT_H", "30"))
EST_SESSION_H = float(os.environ.get("PLM_EST_SESSION_H", "1.0"))


def log(msg: str) -> None:
    print(f"[stage-e] {msg}", flush=True)



def _archive_existing_old(old: Path) -> None:
    """Preserve a prior .old tree as a durable timestamped backup (Lumi glue).

    Dee's revised launcher rotated runs -> runs.old then deleted .old. Deleting
    non-identical checkpoint trees risks data loss; keep archives instead.
    """
    if not old.exists():
        return
    stamp = time.strftime("%Y%m%dT%H%M%SZ", time.gmtime())
    archived = old.parent / f"{old.name}.bak.{stamp}"
    n = 0
    while archived.exists():
        n += 1
        archived = old.parent / f"{old.name}.bak.{stamp}.{n}"
    os.replace(old, archived)
    log(f"preserved previous backup as {archived}")


def find_mount(slug: str) -> Path | None:
    for root in (
        INPUT / "datasets" / "galacticaiteam" / slug,
        INPUT / "notebooks" / "galacticaiteam" / slug,
        INPUT / slug,
    ):
        if root.is_dir():
            return root
    return None


def find_stage_f_root() -> Path:
    """Dataset root that contains val.bin (and optionally facts/ and chat/)."""
    found = find_mount(STAGE_F_DATA_SLUG)
    candidates = []
    if found:
        candidates.append(found)
    candidates.extend([
        INPUT / "datasets" / "galacticaiteam" / STAGE_F_DATA_SLUG,
        INPUT / STAGE_F_DATA_SLUG,
    ])
    for root in candidates:
        if (root / "val.bin").is_file():
            return root
    raise FileNotFoundError(f"{STAGE_F_DATA_SLUG} root with val.bin not found")


def _stage_pair(src_bin: Path, src_mask: Path, dest_dir: Path, stem: str = "train") -> None:
    dest_dir.mkdir(parents=True, exist_ok=True)
    dest_bin, dest_mask = dest_dir / f"{stem}.bin", dest_dir / f"{stem}.masks.bin"
    shutil.copy(src_bin, dest_bin)
    shutil.copy(src_mask, dest_mask)
    n = token_count(dest_bin)
    if dest_mask.stat().st_size != n:
        raise ValueError(f"{dest_mask}: {dest_mask.stat().st_size} masks vs {n} tokens")
    log(f"staged {src_bin} -> {dest_bin} ({n:,} tokens / masks aligned)")


def stage_three_way_data() -> int:
    """Facts train into DATA_DIR, chat train into CHAT_DIR, frozen val at DATA_DIR."""
    root = find_stage_f_root()
    # Prefer facts/ and chat/ subdirs; fall back to flat root names (Kaggle
    # dataset --dir-mode skip does not upload subdirectories).
    if (root / "facts" / "train.bin").is_file():
        facts_bin, facts_mask = root / "facts" / "train.bin", root / "facts" / "train.masks.bin"
    else:
        facts_bin, facts_mask = root / "facts_train.bin", root / "facts_train.masks.bin"
    if (root / "chat" / "train.bin").is_file():
        chat_bin, chat_mask = root / "chat" / "train.bin", root / "chat" / "train.masks.bin"
    else:
        chat_bin, chat_mask = root / "chat_train.bin", root / "chat_train.masks.bin"
    if not facts_bin.is_file() or not facts_mask.is_file():
        raise SystemExit(f"PLM_CHAT_RATIO set but facts stream missing under {root}")
    if not chat_bin.is_file() or not chat_mask.is_file():
        raise SystemExit(f"PLM_CHAT_RATIO set but chat stream missing under {root}")
    val_bin, val_mask = root / "val.bin", root / "val.masks.bin"
    if not val_bin.is_file() or not val_mask.is_file():
        raise SystemExit(f"frozen val missing under {root}")
    _stage_pair(facts_bin, facts_mask, DATA_DIR, "train")
    _stage_pair(val_bin, val_mask, DATA_DIR, "val")
    _stage_pair(chat_bin, chat_mask, CHAT_DIR, "train")
    os.environ["PLM_CHAT_DIR"] = str(CHAT_DIR)
    log(f"three-way mix: facts {DATA_DIR}, chat {CHAT_DIR}, val from {root}")
    return token_count(DATA_DIR / "train.bin")


def _resolve_pair(dir_hint: Path) -> tuple[Path, Path, Path, Path]:
    """Return (train.bin, train.masks.bin, val.bin, val.masks.bin) under dir_hint.

    Raises FileNotFoundError if any are missing.
    """
    train_bin = dir_hint / "train.bin"
    train_mask = dir_hint / "train.masks.bin"
    val_bin = dir_hint / "val.bin"
    val_mask = dir_hint / "val.masks.bin"
    missing = [p for p in (train_bin, train_mask, val_bin, val_mask) if not p.is_file()]
    if missing:
        raise FileNotFoundError(f"missing Stage E data files under {dir_hint}: {missing}")
    return train_bin, train_mask, val_bin, val_mask


def resolve_finetune_files() -> tuple[Path, Path, Path, Path]:
    """Locate the paired Stage E artifacts from env or the data mount.

    Accepted layouts, in priority order:
      1. PLM_FINETUNE_BIN / PLM_FINETUNE_VAL_BIN env (both must be set).
      2. /kaggle/input/plm-stage-e-data[/finetune]/
      3. /kaggle/input/datasets/galacticaiteam/plm-stage-e-data[/finetune]/
      4. rglob under any plm-stage-e-data mount that contains train.bin
    """
    env_train = os.environ.get("PLM_FINETUNE_BIN")
    env_val = os.environ.get("PLM_FINETUNE_VAL_BIN")
    if env_train and env_val:
        return _resolve_pair(Path(env_train).parent)

    candidates = [
        INPUT / "plm-stage-f-data",
        INPUT / "datasets" / "galacticaiteam" / "plm-stage-f-data",
        INPUT / "plm-stage-e-data" / "finetune",
        INPUT / "datasets" / "galacticaiteam" / "plm-stage-e-data" / "finetune",
        INPUT / "plm-stage-e-data",
        INPUT / "datasets" / "galacticaiteam" / "plm-stage-e-data",
    ]
    for c in candidates:
        if (c / "train.bin").is_file():
            return _resolve_pair(c)

    for root in list(INPUT.rglob("plm-stage-f-data")) + list(INPUT.rglob("plm-stage-e-data")):
        if not root.is_dir():
            continue
        for hit in root.rglob("train.bin"):
            try:
                return _resolve_pair(hit.parent)
            except FileNotFoundError:
                continue
    raise FileNotFoundError(
        "Stage E paired data not found (need train.bin, train.masks.bin, "
        "val.bin, val.masks.bin)"
    )


def token_count(path: Path, context_len: int = 1024) -> int:
    size = path.stat().st_size
    if size % 2:
        raise ValueError(f"{path}: {size} bytes is not a uint16 token file")
    n = size // 2
    if n <= context_len:
        raise ValueError(f"{path}: only {n} tokens")
    return n


def stage_data() -> int:
    """Copy the paired train/val bins+masks into DATA_DIR and verify masks
    align with tokens byte-for-byte. Returns train token count.

    When PLM_CHAT_RATIO is set, DATA_DIR gets the facts stream (plus frozen
    val) and CHAT_DIR gets the chat stream for the three-way mixer.
    """
    if float(os.environ.get("PLM_CHAT_RATIO") or 0) > 0:
        return stage_three_way_data()
    train_bin, train_mask, val_bin, val_mask = resolve_finetune_files()
    DATA_DIR.mkdir(parents=True, exist_ok=True)
    for src in (train_bin, train_mask, val_bin, val_mask):
        shutil.copy(src, DATA_DIR / src.name)
        log(f"staged {src.name} ({src.stat().st_size} bytes)")

    # Mask/token alignment: train.masks.bin must have exactly one uint8 per
    # uint16 token. Same for val.
    train_tok = token_count(DATA_DIR / "train.bin")
    val_tok = token_count(DATA_DIR / "val.bin")
    train_mask_bytes = (DATA_DIR / "train.masks.bin").stat().st_size
    val_mask_bytes = (DATA_DIR / "val.masks.bin").stat().st_size
    if train_mask_bytes != train_tok:
        raise ValueError(
            f"train mask/token mismatch: {train_mask_bytes} masks vs {train_tok} tokens"
        )
    if val_mask_bytes != val_tok:
        raise ValueError(
            f"val mask/token mismatch: {val_mask_bytes} masks vs {val_tok} tokens"
        )
    log(f"train {train_tok:,} tokens / masks aligned; val {val_tok:,} tokens / masks aligned")
    return train_tok


def preflight_disk() -> bool:
    free = shutil.disk_usage(WORKING).free / (1024 ** 3)
    log(f"disk free: {free:.1f} GB (need >= {MIN_DISK_GB:.1f})")
    if free < MIN_DISK_GB:
        log("FATAL: insufficient disk")
        return False
    return True


def preflight_quota() -> bool:
    if not QUOTA_GUARD:
        log("quota guard off (set PLM_QUOTA_GUARD=1 to enforce)")
        return True
    log(f"quota guard: {EST_SESSION_H:.1f} h vs limit {QUOTA_LIMIT_H:.1f} h")
    if EST_SESSION_H > QUOTA_LIMIT_H:
        log("FATAL: estimated session exceeds quota limit")
        return False
    return True


def _atomic_stage_copy(src: Path, dst: Path) -> None:
    """Copy src -> dst atomically. If dst exists, it is preserved on failure.

    Procedure:
      1. Copy src into dst.parent / (dst.name + ".tmp"), removing any stale tmp.
      2. Verify tmp has a manifest.json (caller's contract).
      3. os.replace(dst -> dst.parent / (dst.name + ".old")) if dst exists,
         then os.replace(tmp -> dst), then remove .old.
    """
    tmp = dst.parent / (dst.name + ".tmp")
    old = dst.parent / (dst.name + ".old")
    if tmp.exists():
        shutil.rmtree(tmp)
    shutil.copytree(src, tmp)
    if not (tmp / "manifest.json").is_file():
        shutil.rmtree(tmp)
        raise RuntimeError(f"staged copy at {tmp} has no manifest.json; existing {dst} preserved")
    if dst.exists():
        if old.exists():
            _archive_existing_old(old)
        os.replace(dst, old)
        os.replace(tmp, dst)
        # Keep dst.old as durable backup (Lumi glue; do not rmtree).
    else:
        os.replace(tmp, dst)
    log(f"atomically installed {src} -> {dst}")


def sha256_file(path: Path) -> str:
    import hashlib
    h = hashlib.sha256()
    with path.open("rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def _find_seed_checkpoint() -> tuple[Path, Path]:
    """Return (weights, optional model_config) for the E31140 seed."""
    roots = []
    found = find_mount(STAGE_F_SEED_SLUG)
    if found:
        roots.append(found)
    roots.extend([
        INPUT / "datasets" / "galacticaiteam" / STAGE_F_SEED_SLUG,
        INPUT / STAGE_F_SEED_SLUG,
    ])
    stem = f"step_{STAGE_F_SEED_STEP:08d}"
    for root in roots:
        for cand in (
            root / "runs" / "checkpoints" / f"{stem}.safetensors",
            root / "checkpoints" / f"{stem}.safetensors",
            root / f"{stem}.safetensors",
        ):
            if cand.is_file():
                cfg = cand.with_name(f"{stem}_model_config.json")
                return cand, cfg
    raise FileNotFoundError(f"E31140 seed weights not found under {STAGE_F_SEED_SLUG}")


def seed_from_e31140() -> None:
    """Install a runs/ tree with E31140 weights + manifest (no optimizer state)."""
    weights, cfg = _find_seed_checkpoint()
    digest = sha256_file(weights)
    if digest != STAGE_F_SEED_SHA256:
        raise SystemExit(
            f"seed sha256 mismatch: got {digest}, expected {STAGE_F_SEED_SHA256}"
        )
    tmp = RUNS_DIR.parent / (RUNS_DIR.name + ".tmp")
    old = RUNS_DIR.parent / (RUNS_DIR.name + ".old")
    if tmp.exists():
        shutil.rmtree(tmp)
    (tmp / "checkpoints").mkdir(parents=True)
    shutil.copy(weights, tmp / "checkpoints" / weights.name)
    if cfg.is_file():
        shutil.copy(cfg, tmp / "checkpoints" / cfg.name)
    stem = f"step_{STAGE_F_SEED_STEP:08d}"
    manifest = {
        "run_id": "stage-e-r3-31140-seed",
        "config": {},
        "checkpoints": [{
            "step": STAGE_F_SEED_STEP,
            "path": f"checkpoints/{stem}.safetensors",
            "tokens_seen": STAGE_F_SEED_TOKENS,
            "wall_time_s": 0.0,
            "created_at": time.time(),
            "is_milestone": True,
        }],
        "latest_step": STAGE_F_SEED_STEP,
        "latest_tokens_seen": STAGE_F_SEED_TOKENS,
        "latest_wall_time_s": 0.0,
    }
    (tmp / "manifest.json").write_text(json.dumps(manifest, indent=2))
    if RUNS_DIR.exists():
        if old.exists():
            _archive_existing_old(old)
        os.replace(RUNS_DIR, old)
        os.replace(tmp, RUNS_DIR)
    else:
        os.replace(tmp, RUNS_DIR)
    log(f"seeded from E{STAGE_F_SEED_STEP} sha256={digest} ({STAGE_F_SEED_TOKENS:,} tokens)")


def seed_from_stage_c() -> None:
    """Copy Stage C step 31101 (weights + state + model_config) into a
    freshly-staged RUNS_DIR.tmp, write a manifest, and atomically install."""
    root = find_mount(STAGE_C_SLUG)
    if root is None:
        raise FileNotFoundError(f"kernel source {STAGE_C_SLUG} is not mounted")
    src = root / "runs" / "checkpoints"
    stem = f"step_{STAGE_C_FINAL_STEP:08d}"
    needed = [src / f"{stem}.safetensors", src / f"{stem}_state.pt"]
    missing = [p for p in needed if not p.is_file()]
    if missing:
        raise FileNotFoundError(f"Stage C final checkpoint incomplete, missing {missing}")

    tmp = RUNS_DIR.parent / (RUNS_DIR.name + ".tmp")
    old = RUNS_DIR.parent / (RUNS_DIR.name + ".old")
    if tmp.exists():
        shutil.rmtree(tmp)
    (tmp / "checkpoints").mkdir(parents=True)

    for p in needed:
        shutil.copy(p, tmp / "checkpoints" / p.name)
    model_cfg = src / f"{stem}_model_config.json"
    if model_cfg.is_file():
        shutil.copy(model_cfg, tmp / "checkpoints" / model_cfg.name)

    c_manifest = json.loads((root / "runs" / "manifest.json").read_text())
    entry = next(
        (e for e in c_manifest["checkpoints"] if e["step"] == STAGE_C_FINAL_STEP),
        None,
    )
    if entry is None:
        shutil.rmtree(tmp)
        raise ValueError(
            f"Stage C manifest has no step {STAGE_C_FINAL_STEP}; has Stage C finished?"
        )
    manifest = {
        "run_id": "stage-c-final",
        "config": c_manifest.get("config", {}),
        "checkpoints": [dict(entry, path=f"checkpoints/{stem}.safetensors",
                             created_at=time.time(), is_milestone=True)],
        "latest_step": STAGE_C_FINAL_STEP,
        "latest_tokens_seen": entry["tokens_seen"],
        "latest_wall_time_s": entry["wall_time_s"],
    }
    (tmp / "manifest.json").write_text(json.dumps(manifest, indent=2))

    if RUNS_DIR.exists():
        if old.exists():
            _archive_existing_old(old)
        os.replace(RUNS_DIR, old)
        os.replace(tmp, RUNS_DIR)
        # Keep RUNS_DIR.old as durable backup (Lumi glue; do not rmtree).
    else:
        os.replace(tmp, RUNS_DIR)
    log(f"seeded from Stage C step {STAGE_C_FINAL_STEP} ({entry['tokens_seen']:,} tokens)")


def gh_token() -> str | None:
    try:
        from kaggle_secrets import UserSecretsClient
        return UserSecretsClient().get_secret("GH_TOKEN") or None
    except Exception:
        return os.environ.get("GH_TOKEN")


def clone_repo() -> None:
    if REPO_DIR.exists():
        subprocess.run(["git", "-C", str(REPO_DIR), "fetch", "--depth", "1", "origin", GIT_COMMIT or GIT_BRANCH], check=True)
        target = GIT_COMMIT or f"origin/{GIT_BRANCH}"
        subprocess.run(["git", "-C", str(REPO_DIR), "checkout", "--force", target], check=True)
    else:
        if GIT_COMMIT:
            subprocess.run(
                ["git", "clone", "--depth", "1", GIT_REPO, str(REPO_DIR)],
                check=True,
            )
            subprocess.run(["git", "-C", str(REPO_DIR), "fetch", "--depth", "1", "origin", GIT_COMMIT], check=True)
            subprocess.run(["git", "-C", str(REPO_DIR), "checkout", "--force", GIT_COMMIT], check=True)
        else:
            subprocess.run(
                ["git", "clone", "--depth", "1", "--branch", GIT_BRANCH, GIT_REPO, str(REPO_DIR)],
                check=True,
            )
    log(f"repo at {REPO_DIR} branch={GIT_BRANCH} commit={GIT_COMMIT or '(branch tip)'}")


def resolve_replay_bin() -> str | None:
    """Locate FineWeb-style train.bin on Kaggle mounts (Round 3).

    Kaggle may mount private datasets under either
    /kaggle/input/<slug>/ or /kaggle/input/datasets/<owner>/<slug>/.
    Prefer Stage B / FineWeb corpus bins; never pick Stage E finetune/train.bin.
    """
    import os
    from pathlib import Path
    env = os.environ.get("PLM_REPLAY_BIN")
    if env and Path(env).is_file():
        return env
    root = Path("/kaggle/input")
    if root.is_dir():
        log(f"kaggle input mounts: {[p.name for p in sorted(root.iterdir())]}")
    prefer = ("plm-stageb-data", "plm-stage-c", "stageb", "stage-c", "fineweb")
    hits = list(root.rglob("train.bin")) if root.is_dir() else []
    # Exclude Stage E SFT paired finetune bins from replay candidates.
    hits = [h for h in hits if "finetune" not in str(h).lower()
            and "plm-stage-f-data" not in str(h).lower()
            and "plm-stage-e-data" not in str(h).lower()]
    log(f"train.bin candidates: {[str(h) for h in hits[:12]]}")
    if not hits:
        return None  # do not return a missing env path
    def score(p: Path) -> tuple:
        s = str(p).lower()
        pref = 0 if any(k in s for k in prefer) else 1
        try:
            sz = -p.stat().st_size  # larger first
        except OSError:
            sz = 0
        return (pref, sz, len(s))
    hits.sort(key=score)
    return str(hits[0])


def main() -> int:
    if not preflight_disk():
        return 1
    if not preflight_quota():
        return 1

    n_train = stage_data()

    # Stage F always seeds from the pinned E31140 weights (not a prior F run).
    seed_from_e31140()

    clone_repo()
    sys.path.insert(0, str(REPO_DIR / "src"))
    from playground_lm.train.stage_config import finetune_max_steps

    # Prefer sweep-injected PLM_MAX_STEPS (Round-3 replay lengthens the schedule).
    if os.environ.get("PLM_MAX_STEPS"):
        max_steps = int(os.environ["PLM_MAX_STEPS"])
        log(f"max_steps {max_steps} from PLM_MAX_STEPS env (sweep pin)")
    else:
        max_steps = finetune_max_steps(n_train, epochs=FINETUNE_EPOCHS)
    seed_step = STAGE_F_SEED_STEP
    span = max_steps - seed_step
    log(f"max_steps {max_steps} ({span} updates, ~{FINETUNE_EPOCHS:g} epochs, "
        f"replay_ratio={os.environ.get('PLM_REPLAY_RATIO', '0')}, "
        f"chat_ratio={os.environ.get('PLM_CHAT_RATIO', '0')})")
    if n_train < 24 * 1024:
        log(f"WARN: train set ({n_train:,} tokens) is smaller than one 24-sequence batch")

    env = dict(os.environ)
    ms = list(range(seed_step + 10, max_steps, 10))
    if not ms or ms[-1] != max_steps:
        ms.append(max_steps)
    # Conservative short-run defaults: eval every 5 (or 1 if tiny).
    if "PLM_EVAL_INTERVAL" in os.environ:
        eval_iv = os.environ["PLM_EVAL_INTERVAL"]
    elif span <= 15:
        eval_iv = "1"
    elif span <= 80:
        eval_iv = "5"
    else:
        eval_iv = "25"
    warmup = int(os.environ["PLM_WARMUP"]) if os.environ.get("PLM_WARMUP") else max(1, min(20, span // 4))
    replay_bin = resolve_replay_bin()
    if os.environ.get("PLM_REPLAY_RATIO") and float(os.environ.get("PLM_REPLAY_RATIO") or 0) > 0:
        if not replay_bin:
            raise SystemExit("PLM_REPLAY_RATIO set but no train.bin found under /kaggle/input")
        os.environ["PLM_REPLAY_BIN"] = replay_bin
        env["PLM_REPLAY_BIN"] = replay_bin
        log(f"replay bin -> {replay_bin}")
    env.update({
        "PLM_STAGE": "E",
        "PLM_MAX_STEPS": str(max_steps),
        "PLM_WARMUP": str(warmup),
        "PLM_EVAL_INTERVAL": eval_iv,
        "PLM_BATCH_SIZE": "6",
        "PLM_SCHEDULE_START": os.environ.get("PLM_SCHEDULE_START", str(seed_step)),
        "PLM_DATA_DIR": str(DATA_DIR),
        "PLM_CHAT_DIR": os.environ.get("PLM_CHAT_DIR", ""),
        "PLM_CHAT_RATIO": os.environ.get("PLM_CHAT_RATIO", ""),
        "PLM_RUN_DIR": str(RUNS_DIR),
        "PLM_RUN_ID": os.environ.get("PLM_RUN_ID", "stage-f-sft"),
        "PLM_REPO": "galactic-ai-team/playground-lm",
        "PLM_RELEASE_TAG": "stage-f-checkpoints-candidate",
        "PLM_UPLOAD_EVERY": "1",
        "PLM_MAX_SESSION_S": str(11 * 3600),
        "PLM_RESET_OPTIMIZER": os.environ.get("PLM_RESET_OPTIMIZER", "1"),
        "PLM_WEIGHTS_ONLY_RESUME": os.environ.get("PLM_WEIGHTS_ONLY_RESUME", "1"),
        "PLM_MILESTONES": ",".join(str(x) for x in ms),
        "PLM_CKPT_INTERVAL_S": os.environ.get("PLM_CKPT_INTERVAL_S", "60"),
    })
    token = gh_token()
    if token:
        env["GH_TOKEN"] = token

    if DRY_RUN:
        env["CUDA_VISIBLE_DEVICES"] = ""
        env["PLM_MAX_STEPS"] = str(seed_step + 1)   # one update
        log("DRY RUN: 1 CPU update, no GPU quota consumed")
        nproc = 1
    else:
        try:
            import torch
            nproc = max(1, torch.cuda.device_count())
        except Exception:
            nproc = 1

    cmd = [
        sys.executable, "-m", "torch.distributed.run", "--standalone",
        f"--nproc_per_node={nproc}",
        str(REPO_DIR / "notebooks" / "playground_lm_train.py"),
    ]
    log(f"launching torchrun on {nproc} proc(s)")
    return subprocess.run(cmd, env=env, cwd=str(REPO_DIR)).returncode


if __name__ == "__main__":
    sys.exit(main())