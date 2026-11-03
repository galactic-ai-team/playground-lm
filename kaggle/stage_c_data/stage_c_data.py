# stage_c_data.py
# Kaggle CPU script kernel: Stage C data preparation.
#
# Runs prepare_fineweb.py for one SESSION (0..N-1) of a partitioned FineWeb-Edu
# sharding job, then packs per-shard .bin outputs into train.bin / val.bin.
#
# Session partition (N = 8 by default, override via PLM_SESSION / PLM_N_SESSIONS):
#   - Each session gets a disjoint slice of the parquet glob (data_files pattern)
#     and a disjoint --shard-start value.
#   - Session 0 also writes val (--val-max-tokens 5000000); others pass 0.
#   - All sessions write to the SAME train.bin on the final pack — but each
#     session only produces its own shards, and packing happens once, at the
#     very end, on the session that owns the last shard range.
#
# Resume across Kaggle's 12h CPU session kill:
#   - Each shard is written to its own .bin file in /kaggle/working/shards/.
#   - A state file /kaggle/working/session_state.json records completed shards.
#   - On restart with the same SESSION, the script skips completed shards.
#   - Previous session's output is mounted via kernel_sources (self-reference).
#
# Free-tools only: Kaggle CPU, no paid services, no sign-ups.

import glob
import json
import os
import shutil
import subprocess
import sys
from pathlib import Path

# --- Config ---------------------------------------------------------------

WORKING = Path("/kaggle/working")
INPUT = Path("/kaggle/input")

REPO_DIR = WORKING / "playground-lm"
SHARDS_DIR = WORKING / "shards"
STATE_FILE = WORKING / "session_state.json"
MANIFEST_FILE = WORKING / "session_manifest.json"

# Previous session's output, if resuming. Same slug as this kernel.
PREVIOUS_OUTPUT = INPUT / "plm-stagec-data-prep"

GIT_REPO = "https://github.com/galactic-ai-team/playground-lm.git"
GIT_BRANCH = "main"

# Which parquet dataset on Kaggle holds the FineWeb-Edu sample. This must
# match whatever dataset slug the group is using. Override via env if needed.
FINEWEB_DATASET_GLOB = "/kaggle/input/fineweb-edu/sample/10BT/*.parquet"
if not glob.glob(FINEWEB_DATASET_GLOB):
    # Fallback: walk /kaggle/input looking for parquet files under any dir
    # whose name contains "fineweb".
    candidates = [
        p for p in INPUT.rglob("*.parquet")
        if "fineweb" in str(p).lower()
    ]
    if candidates:
        # Use the directory of the first match; assume siblings.
        FINEWEB_DATASET_GLOB = str(candidates[0].parent / "*.parquet")

# Total sessions in the Stage C partition.
N_SESSIONS = int(os.environ.get("PLM_N_SESSIONS", "8"))

# Session index for THIS run. Set via env at push time, or override here.
SESSION = int(os.environ.get("PLM_SESSION", "0"))
if not 0 <= SESSION < N_SESSIONS:
    raise ValueError(
        f"PLM_SESSION={SESSION} out of range for PLM_N_SESSIONS={N_SESSIONS}"
    )

# Shards per session. Stage C is ~1B tokens; at ~10M tokens per shard (a
# reasonable size for a CPU session to produce in a few hours), that's ~100
# shards total, ~13 per session. Adjust once we see actual throughput.
SHARDS_PER_SESSION = int(os.environ.get("PLM_SHARDS_PER_SESSION", "13"))

# Shard index range for this session.
SHARD_START = SESSION * SHARDS_PER_SESSION
SHARD_END = SHARD_START + SHARDS_PER_SESSION  # exclusive

# Val: only session 0 writes it.
VAL_MAX_TOKENS = 5_000_000 if SESSION == 0 else 0

# --- Helpers --------------------------------------------------------------

def log(msg: str) -> None:
    print(f"[stage-c/{SESSION}] {msg}", flush=True)


def restore_previous_output() -> bool:
    """Copy previous session's shards + state into /kaggle/working.

    Returns True if resume state was restored. On the very first run of any
    session, PREVIOUS_OUTPUT doesn't exist and this returns False.
    """
    if not PREVIOUS_OUTPUT.exists():
        log(f"no previous output at {PREVIOUS_OUTPUT}; starting fresh")
        return False

    prev_state = PREVIOUS_OUTPUT / "session_state.json"
    prev_manifest = PREVIOUS_OUTPUT / "session_manifest.json"
    prev_shards = PREVIOUS_OUTPUT / "shards"

    # Only resume if the previous session was the SAME session index. A
    # session-N state file must not be applied to session-N+1.
    if prev_manifest.exists():
        with open(prev_manifest) as f:
            prev = json.load(f)
        if prev.get("session") != SESSION:
            log(
                f"previous output was session {prev.get('session')}, "
                f"not {SESSION}; starting fresh for this session"
            )
            return False

    if not prev_shards.exists():
        log("previous output has no shards/ dir; starting fresh")
        return False

    log(f"restoring shards from {prev_shards}")
    if SHARDS_DIR.exists():
        shutil.rmtree(SHARDS_DIR)
    shutil.copytree(prev_shards, SHARDS_DIR)

    if prev_state.exists():
        shutil.copy(prev_state, STATE_FILE)
        with open(STATE_FILE) as f:
            state = json.load(f)
        log(f"restored state: {len(state.get('completed_shards', []))} shards done")

    return True


def load_state() -> dict:
    """Load or initialize the per-session state file."""
    if STATE_FILE.exists():
        with open(STATE_FILE) as f:
            return json.load(f)
    return {
        "session": SESSION,
        "completed_shards": [],
        "total_tokens": 0,
        "val_written": False,
    }


def save_state(state: dict) -> None:
    tmp = STATE_FILE.with_suffix(".json.tmp")
    with open(tmp, "w") as f:
        json.dump(state, f, indent=2)
    os.replace(tmp, STATE_FILE)


def save_manifest() -> None:
    with open(MANIFEST_FILE, "w") as f:
        json.dump(
            {
                "session": SESSION,
                "n_sessions": N_SESSIONS,
                "shard_start": SHARD_START,
                "shard_end": SHARD_END,
                "shards_per_session": SHARDS_PER_SESSION,
                "val_max_tokens": VAL_MAX_TOKENS,
            },
            f,
            indent=2,
        )


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


def find_prepare_script() -> Path:
    """Locate prepare_fineweb.py in the cloned repo."""
    candidates = list(REPO_DIR.rglob("prepare_fineweb.py"))
    if not candidates:
        raise FileNotFoundError(
            f"prepare_fineweb.py not found under {REPO_DIR}"
        )
    # Prefer data/prepare_fineweb.py if it exists
    for c in candidates:
        if "data" in c.parts:
            return c
    return candidates[0]


# --- Main -----------------------------------------------------------------

def main() -> int:
    log(f"session {SESSION}/{N_SESSIONS}, shards [{SHARD_START}, {SHARD_END})")
    log(f"fineweb glob: {FINEWEB_DATASET_GLOB}")

    if not glob.glob(FINEWEB_DATASET_GLOB):
        log(f"FATAL: no parquet files matched {FINEWEB_DATASET_GLOB}")
        return 1

    SHARDS_DIR.mkdir(parents=True, exist_ok=True)

    # 1. Restore previous session's progress (if any).
    restore_previous_output()
    save_manifest()

    # 2. Clone repo (has prepare_fineweb.py).
    clone_repo()
    prepare_script = find_prepare_script()
    log(f"prepare script: {prepare_script}")

    # 3. Load or init state.
    state = load_state()
    completed = set(state["completed_shards"])
    log(f"already completed shards: {sorted(completed)}")

    # 4. Run prepare_fineweb.py for each shard not yet complete.
    # The script's --shard-start is a global index; we do one invocation per
    # shard so a session kill mid-shard loses at most one shard's work.
    for shard_idx in range(SHARD_START, SHARD_END):
        if shard_idx in completed:
            log(f"shard {shard_idx} already done, skipping")
            continue

        log(f"generating shard {shard_idx}")
        shard_path = SHARDS_DIR / f"shard_{shard_idx:04d}.bin"

        cmd = [
            sys.executable, str(prepare_script),
            "--data-files", FINEWEB_DATASET_GLOB,
            "--shard-start", str(shard_idx),
            "--max-docs", str(_docs_per_shard()),
            "--val-max-tokens", str(VAL_MAX_TOKENS if (SESSION == 0 and not state["val_written"]) else 0),
            "--out-dir", str(SHARDS_DIR),
        ]

        log(f"$ {' '.join(cmd)}")
        result = subprocess.run(cmd, cwd=str(REPO_DIR))
        if result.returncode != 0:
            log(f"FATAL: prepare_fineweb.py exited with {result.returncode}")
            return result.returncode

        # The script writes to out-dir; we don't know the exact filename
        # pattern yet. Verify at least one new .bin appeared.
        new_bins = sorted(SHARDS_DIR.glob(f"*{shard_idx}*.bin"))
        if not new_bins:
            log(f"FATAL: no output for shard {shard_idx} — check prepare_fineweb.py args")
            return 1

        completed.add(shard_idx)
        state["completed_shards"] = sorted(completed)
        if SESSION == 0:
            state["val_written"] = True  # val max tokens only matters once
        save_state(state)
        log(f"shard {shard_idx} done ({len(new_bins)} file(s))")

    # 5. Pack shards into train.bin / val.bin. Only the LAST session does
    # this, once all sessions' shards are present — but on Kaggle each kernel
    # only sees its own output + the previous version of itself, not other
    # sessions. So the actual "pack" runs as a separate post-step, or the
    # session that writes the final shard pack passes its train.bin forward.
    #
    # For now: this session packs ITS shards into a session-local train.bin
    # that the next session's restore step will see via kernel_sources and
    # concatenate with its own. Since sessions run sequentially (session N
    # mounts session N-1's output), the last session's output is the full
    # packed train.bin.
    log("packing shards into session-local train.bin")
    pack_shards(SHARDS_DIR, state)

    log(f"done. total tokens this session: {state['total_tokens']:,}")
    return 0


def _docs_per_shard() -> int:
    """Approximate doc count per shard. Tune when we see actual throughput.

    At ~1B tokens / 100 shards = ~10M tokens per shard. FineWeb-Edu avg doc
    is ~500 tokens, so ~20K docs per shard.
    """
    return int(os.environ.get("PLM_DOCS_PER_SHARD", "20000"))


def pack_shards(shards_dir: Path, state: dict) -> None:
    """Concatenate this session's shard .bin files into a session-local
    train.bin and (if session 0) val.bin. Later sessions restore the prior
    session's train.bin and append their own shards to it.
    """
    import numpy as np

    # Restore previous session's train.bin if it was carried forward.
    prev_train = PREVIOUS_OUTPUT / "train.bin"
    prev_val = PREVIOUS_OUTPUT / "val.bin"
    out_train = WORKING / "train.bin"
    out_val = WORKING / "val.bin"

    if prev_train.exists():
        log(f"carrying forward previous train.bin ({prev_train.stat().st_size // 2:,} tokens)")
        shutil.copy(prev_train, out_train)
    else:
        out_train.write_bytes(b"")

    if prev_val.exists() and not out_val.exists():
        shutil.copy(prev_val, out_val)

    # Append this session's shards to train.bin.
    total_added = 0
    with open(out_train, "ab") as f_out:
        for shard in sorted(shards_dir.glob("*.bin")):
            # Skip shards that belong to other sessions (defensive).
            try:
                idx = int(shard.stem.split("_")[-1])
            except ValueError:
                continue
            if not (SHARD_START <= idx < SHARD_END):
                continue
            n_tokens = shard.stat().st_size // 2
            with open(shard, "rb") as f_in:
                while True:
                    chunk = f_in.read(1024 * 1024 * 8)
                    if not chunk:
                        break
                    f_out.write(chunk)
            total_added += n_tokens

    state["total_tokens"] = state.get("total_tokens", 0) + total_added
    save_state(state)
    log(f"train.bin: +{total_added:,} tokens, total {(out_train.stat().st_size // 2):,}")

    # Session 0 also writes val.bin — prepare_fineweb.py writes val to its
    # own file, we just rename it into place if it exists.
    val_shard = shards_dir / "val.bin"
    if val_shard.exists() and not out_val.exists():
        shutil.copy(val_shard, out_val)
        log(f"val.bin: {out_val.stat().st_size // 2:,} tokens")


if __name__ == "__main__":
    sys.exit(main())