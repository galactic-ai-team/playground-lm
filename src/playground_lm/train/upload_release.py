# src/playground_lm/train/upload_release.py
# Package and upload checkpoints to a GitHub release.
#
# Public API (importable from the training notebook):
#   upload(run_dir, repo, tag, gh_token) -> UploadResult
#
# CLI (for ad-hoc use from a shell):
#   python -m playground_lm.train.upload_release \
#       --run-dir /kaggle/working/runs/plm-001 \
#       --repo galactic-ai-team/playground-lm \
#       --tag v0.1-checkpoints
#
# Uses the GitHub CLI (`gh`), preinstalled on Kaggle/Colab. Authenticates via
# the token passed in or read from the GH_TOKEN env var. No extra deps, no
# extra sign-ups — just a token from Stephon's account with `repo` scope.
#
# Idempotency:
#   - Checkpoints already on the release are skipped by filename.
#   - manifest.json is always re-uploaded (--clobber) so the demo frontend
#     sees the latest set even when nothing new was added.
#   - Only `.safetensors` weights are uploaded, NOT optimizer state — that's
#     needed locally for resume but is useless (and huge) for the demo.

import argparse
import json
import os
import subprocess
import sys
import tempfile
from dataclasses import dataclass, field
from pathlib import Path

from playground_lm.train.sidecar import sidecar_path


@dataclass
class UploadResult:
    uploaded: list[str] = field(default_factory=list)
    skipped: list[str] = field(default_factory=list)
    manifest_uploaded: bool = False
    release_created: bool = False


def _run(cmd: list[str], check: bool = True) -> subprocess.CompletedProcess:
    return subprocess.run(cmd, check=check, capture_output=True, text=True)


def _ensure_release(repo: str, tag: str) -> bool:
    """Create the release if it doesn't exist. Returns True if created."""
    result = _run(["gh", "release", "view", tag, "--repo", repo], check=False)
    if result.returncode == 0:
        return False
    print(f"[upload] creating release {tag} in {repo}")
    _run([
        "gh", "release", "create", tag,
        "--repo", repo,
        "--title", f"PlaygroundLM checkpoints {tag}",
        "--notes",
        "Rolling training checkpoints (safetensors only). "
        "See manifest.json for step and tokens_seen.",
        "--latest",
    ])
    return True


def _existing_assets(repo: str, tag: str) -> set[str]:
    result = _run(
        ["gh", "release", "view", tag, "--repo", repo, "--json", "assets"],
    )
    data = json.loads(result.stdout)
    return {a["name"] for a in data.get("assets", [])}


def _upload_one(repo: str, tag: str, file_path: Path) -> None:
    """Upload a single file, overwriting if it exists."""
    _run([
        "gh", "release", "upload", tag, str(file_path),
        "--repo", repo, "--clobber",
    ])


def upload(
    run_dir: str | Path,
    repo: str,
    tag: str,
    gh_token: str | None = None,
) -> UploadResult:
    """
    Upload all not-yet-present checkpoints from `run_dir` to the GitHub
    release `tag` in `repo`, plus a refreshed manifest.json.

    Args:
        run_dir:  Directory containing manifest.json and checkpoints/.
        repo:     GitHub repo slug, e.g. "galactic-ai-team/playground-lm".
        tag:      Release tag, e.g. "v0.1-checkpoints".
        gh_token: GitHub token with `repo` scope. If None, reads from GH_TOKEN
                  env var. Raises ValueError if neither is available.

    Returns:
        UploadResult listing what was uploaded / skipped.

    Raises:
        ValueError:  no token available.
        SystemExit:  manifest missing or `gh` upload fails.
    """
    # Resolve token: explicit arg wins over env var. `gh` picks up GH_TOKEN
    # from the process env automatically, so setting it here is sufficient.
    token = gh_token or os.environ.get("GH_TOKEN")
    if not token:
        raise ValueError(
            "upload(): no GitHub token — pass gh_token= or set GH_TOKEN env var"
        )
    os.environ["GH_TOKEN"] = token

    run_dir = Path(run_dir)
    manifest_path = run_dir / "manifest.json"
    if not manifest_path.exists():
        raise SystemExit(f"[upload] no manifest at {manifest_path}")

    with open(manifest_path) as f:
        manifest = json.load(f)

    result = UploadResult()
    result.release_created = _ensure_release(repo, tag)

    present = _existing_assets(repo, tag)

    # Always refresh manifest.json, even if nothing else changed. The demo
    # reads this to discover what checkpoints exist.
    tmp_manifest = Path(tempfile.mkdtemp()) / "manifest.json"
    with open(tmp_manifest, "w") as f:
        json.dump(manifest, f, indent=2)
    _upload_one(repo, tag, tmp_manifest)
    result.manifest_uploaded = True
    print(f"[upload] manifest.json -> {tag}")

    # Upload each checkpoint weight that isn't already there.
    for entry in manifest["checkpoints"]:
        name = Path(entry["path"]).name
        weights = run_dir / entry["path"]
        # The small model-config JSON beside the weights (Claude) goes up too,
        # so evals of a released checkpoint never need its _state.pt. It is
        # also uploaded when only the weights are already released (backfill).
        side = sidecar_path(weights)
        if side.exists() and side.name not in present:
            _upload_one(repo, tag, side)
            print(f"[upload] step {entry['step']:>8} -> {side.name}")
            result.uploaded.append(side.name)
        if name in present:
            result.skipped.append(name)
            continue
        if not weights.exists():
            print(f"[upload] WARN: manifest references missing {weights}, skipping")
            result.skipped.append(name)
            continue
        _upload_one(repo, tag, weights)
        print(f"[upload] step {entry['step']:>8} -> {name}")
        result.uploaded.append(name)

    print(
        f"[upload] done. {len(result.uploaded)} new, "
        f"{len(result.skipped)} already present."
    )
    return result


def _cli_main() -> None:
    p = argparse.ArgumentParser(
        description="Upload PlaygroundLM checkpoints to a GitHub release.",
    )
    p.add_argument("--run-dir", required=True, type=Path)
    p.add_argument("--repo", required=True)
    p.add_argument("--tag", required=True)
    args = p.parse_args()

    try:
        upload(args.run_dir, args.repo, args.tag)
    except ValueError as e:
        sys.exit(str(e))


if __name__ == "__main__":
    _cli_main()