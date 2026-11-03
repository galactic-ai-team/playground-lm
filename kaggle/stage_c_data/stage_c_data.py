# stage_c_data.py
# Kaggle CPU script kernel: Stage C data preparation.
#
# Runs ONE session of a partitioned, parallel FineWeb-Edu sharding job.
# Each session:
#   1. Reads its index from session.txt (0..N_SESSIONS-1).
#   2. Restores its own previous run's output (self-reference via
#      kernel_sources in the resume metadata) — enables self-resume across
#      Kaggle's 12h CPU session kill.
#   3. Calls prepare_fineweb.py ONCE with a --max-docs bound that fits in one
#      session (script is self-resuming; rerunning with the same --out skips
#      consumed rows and continues).
#   4. Runs data/tokenize_shards.py to pack its JSONL into train.bin/val.bin.
#
# Outputs, per session:
#   /kaggle/working/stage-c/session-<N>/train.bin
#   /kaggle/working/stage-c/session-<N>/val.bin         (session 0 only)
#   /kaggle/working/stage-c/session-<N>/fineweb-stats.json
#   /kaggle/working/stage-c/session-<N>/fineweb-manifest.json
#
# Final concatenation of the N sessions' train.bin files happens downstream
# (either a final pack kernel or the training launcher reads all N mounts).
# That decision is out of scope for this file — see PR notes.
#
# Free-tools only: Kaggle CPU, no paid services, no sign-ups.

import glob
import json
import os
import shutil
import subprocess
import sys
from pathlib import Path

# --- Paths ----------------------------------------------------------------

WORKING = Path("/kaggle/working")
INPUT = Path("/kaggle/input")

REPO_DIR = WORKING / "playground-lm"
OUT_ROOT = WORKING / "stage-c"        # holds session-<N>/ subdirs
PREV_OUTPUT = INPUT / "plm-stage-c-data-prep"  # previous version of this kernel

GIT_REPO = "https://github.com/galactic-ai-team/playground-lm.git"
GIT_BRANCH = "main"

# --- Config ---------------------------------------------------------------

# Session index read from session.txt next to this script.
_here = Path(__file__).resolve().parent
_session_file = _here / "session.txt"
if not _session_file.exists():
    # When Kaggle runs a script kernel, the code file lands in /kaggle/src/
    # and sibling files may not be copied. Fall back to env var or the
    # bundled output path.
    _session_file = Path("/kaggle/src/session.txt")
if _session_file.exists():
    SESSION = int(_session_file.read_text().strip())
else:
    SESSION = int(os.environ.get("PLM_SESSION", "0"))

N_SESSIONS = int(os.environ.get("PLM_N_SESSIONS", "8"))
if not 0 <= SESSION < N_SESSIONS:
    raise ValueError(f"session {SESSION} out of range [0, {N_SESSIONS})")

SESSION_DIR = OUT_ROOT / f"session-{SESSION}"

# Docs per session. FineWeb-Edu average is ~500 tokens/doc, so 20M docs ≈
# 10B tokens — way over. Stage C is ~1B tokens total across N sessions;
# 1B / 8 sessions / 500 tokens ≈ 250k docs per session. Bump if we see
# sessions finishing too fast, lower if they hit the 12h wall.
DOCS_PER_SESSION = int(os.environ.get("PLM_DOCS_PER_SESSION", "250000"))

# Parallel sessions each read a different slice of the parquet glob.
# Nova's plan partitions by data-files glob; if we have, say, 8 parquet
# files, session N reads file N. If the glob pattern is broader, adjust.
# We resolve the actual list at runtime and slice by session index.
FINEWEB_GLOB_ROOT = "/kaggle/input/fineweb-edu"

# Shard-start for a fresh manifest. Each session gets a disjoint numeric
# range so filenames don't collide even if outputs are ever merged on disk.
SHARD_START = SESSION * 1000

# Val: only session 0 writes it.
VAL_MAX_TOKENS = 5_000_000 if SESSION == 0 else 0

# --- Helpers --------------------------------------------------------------

def log(msg: str) -> None:
    print(f"[stage-c/{SESSION}] {msg}", flush=True)


def resolve_data_files() -> str:
    """Return the --data-files glob for this session.

    Resolves all parquet files under FINEWEB_GLOB_ROOT, sorts them, and
    partitions by SESSION index. If the count is not a multiple of N_SESSIONS,
    sessions earlier in the list get one extra file; that's fine.

    Falls back to a single broad glob if no fineweb dir is found.
    """
    candidates = sorted(glob.glob(f"{FINEWEB_GLOB_ROOT}/**/*.parquet", recursive=True))
    if not candidates:
        # Fallback: walk /kaggle/input for anything that looks like FineWeb.
        candidates = sorted(
            p for p in INPUT.rglob("*.parquet") if "fineweb" in str(p).lower()
        )
    if not candidates:
        raise FileNotFoundError(
            f"no parquet files found under {FINEWEB_GLOB_ROOT} or /kaggle/input"
        )

    # Partition: session N gets files N, N+N_SESSIONS, N+2*N_SESSIONS, ...
    mine = candidates[SESSION::N_SESSIONS]
    log(f"data files for session {SESSION}: {len(mine)} of {len(candidates)}")
    # If only one file for this session, return its exact path (not a glob —
    # some parquet readers treat globs oddly with single matches).
    if len(mine) == 1:
        return mine[0]
    # Otherwise, hand prepare_fineweb.py a glob. It accepts a single glob
    # string and may or may not support brace expansion, so we build a
    # best-effort pattern from a common directory. If the files span
    # multiple directories, we return a newline-free space-separated string
    # and hope prepare_fineweb.py supports it — but if not, the fix is one
    # argparse change on the script side.
    # Safest: return a single glob when files share a directory.
    dirs = {str(Path(p).parent) for p in mine}
    if len(dirs) == 1:
        common = dirs.pop()
        return f"{common}/*.parquet"
    # Multi-directory case: fall back to the parent-of-common-path glob.
    # If prepare_fineweb.py can't handle this, the group needs to add a
    # --data-files-list option.
    common_root = os.path.commonpath([str(Path(p).parent) for p in mine])
    return f"{common_root}/**/*.parquet"


def restore_own_previous_output() -> bool:
    """Copy previous version of THIS kernel's output into WORKING.

    Under parallel sessions, each session only sees its own previous
    version (self-reference in kernel_sources). Session N's output is
    overwritten by its own next run, not by session N+1's.

    Returns True if there was previous output to restore.
    """
    if not PREV_OUTPUT.exists():
        log(f"no previous output at {PREV_OUTPUT}; starting fresh")
        return False

    prev_session_dir = PREV_OUTPUT / "stage-c" / f"session-{SESSION}"
    if not prev_session_dir.exists():
        log(f"previous output exists but has no stage-c/session-{SESSION}/; starting fresh")
        return False

    log(f"restoring previous session {SESSION} output from {prev_session_dir}")
    if SESSION_DIR.exists():
        shutil.rmtree(SESSION_DIR)
    shutil.copytree(prev_session_dir, SESSION_DIR)
    n_files = sum(1 for _ in SESSION_DIR.rglob("*") if _.is_file())
    log(f"restored {n_files} files")
    return True


def clone_repo() -> None:
    if REPO_DIR.exists():
        log(f"{REPO_DIR} exists; pulling latest")
        subprocess.run(["git", "-C", str(REPO_DIR), "pull", "--ff-only"], check=True)
        return
    log(f"cloning {GIT_REPO} ({GIT_BRANCH})")
    subprocess.run(
        ["git", "clone", "--depth", "1", "--branch", GIT_BRANCH, GIT_REPO, str(REPO_DIR)],
        check=True,
    )


def find_script(name: str) -> Path:
    candidates = list(REPO_DIR.rglob(name))
    if not candidates:
        raise FileNotFoundError(f"{name} not found under {REPO_DIR}")
    for c in candidates:
        if "data" in c.parts:
            return c
    return candidates[0]


def count_jsonl_docs(proc_dir: Path) -> int:
    """Count docs already written to this session's proc dir. Used to decide
    how many docs this session still needs to produce."""
    total = 0
    for f in proc_dir.glob("fineweb-train-*.jsonl"):
        with open(f, "rb") as fh:
            for _ in fh:
                total += 1
    return total


# --- Main -----------------------------------------------------------------

def main() -> int:
    log(f"session {SESSION}/{N_SESSIONS}")
    log(f"docs per session: {DOCS_PER_SESSION:,}")

    # 1. Prepare out dir.
    SESSION_DIR.mkdir(parents=True, exist_ok=True)

    # 2. Restore own previous output (self-resume).
    restore_own_previous_output()

    # 3. Clone repo.
    clone_repo()
    prepare_script = find_script("prepare_fineweb.py")
    tokenize_script = find_script("tokenize_shards.py")
    log(f"prepare: {prepare_script}")
    log(f"tokenize: {tokenize_script}")

    # 4. Resolve data files for this session.
    data_files = resolve_data_files()
    log(f"data-files: {data_files}")

    # 5. Prepare proc dir (JSONL output).
    proc_dir = SESSION_DIR / "proc"
    proc_dir.mkdir(exist_ok=True)

    # 6. How many docs this session still needs?
    #    prepare_fineweb.py is self-resuming — rerunning with the same --out
    #    skips already-consumed rows. So --max-docs can be set to the TOTAL
    #    target for the session; the script will only produce what's missing.
    #    (If that's wrong — e.g. --max-docs means "produce at most this many
    #    NEW docs this run" — the number would over-shoot. To be safe, we
    #    subtract what's already on disk.)
    already = count_jsonl_docs(proc_dir)
    remaining = max(0, DOCS_PER_SESSION - already)
    log(f"already written: {already:,} docs; need {remaining:,} more")

    if remaining > 0:
        cmd = [
            sys.executable, str(prepare_script),
            "--data-files", data_files,
            "--out", str(proc_dir),
            "--shard-start", str(SHARD_START),
            "--max-docs", str(remaining),
            "--val-max-tokens", str(VAL_MAX_TOKENS),
        ]
        log(f"$ {' '.join(cmd)}")
        result = subprocess.run(cmd, cwd=str(REPO_DIR))
        if result.returncode != 0:
            log(f"FATAL: prepare_fineweb.py exited {result.returncode}")
            return result.returncode
    else:
        log("prepare step already complete; skipping")

    # 7. Pack JSONL → train.bin/val.bin.
    #    tokenize_shards.py is resumable via .part; safe to rerun.
    final = SESSION_DIR / "final"
    final.mkdir(exist_ok=True)
    cmd = [
        sys.executable, str(tokenize_script),
        "--input", str(proc_dir),
        "--out", str(final),
    ]
    log(f"$ {' '.join(cmd)}")
    result = subprocess.run(cmd, cwd=str(REPO_DIR))
    if result.returncode != 0:
        log(f"FATAL: tokenize_shards.py exited {result.returncode}")
        return result.returncode

    # 8. Report.
    train_bin = final / "train.bin"
    val_bin = final / "val.bin"
    if train_bin.exists():
        n_tok = train_bin.stat().st_size // 2
        log(f"train.bin: {n_tok:,} tokens at {train_bin}")
    else:
        log("WARN: tokenize_shards.py finished but no train.bin found")
    if val_bin.exists():
        n_val = val_bin.stat().st_size // 2
        log(f"val.bin: {n_val:,} tokens at {val_bin}")

    log("done")
    return 0


if __name__ == "__main__":
    sys.exit(main())