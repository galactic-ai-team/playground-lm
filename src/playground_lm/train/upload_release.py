# train/upload_release.py
# Package and upload checkpoints to a GitHub release. Uses the GitHub CLI
# (`gh`) which is preinstalled on Kaggle/Colab and authenticates via
# GH_TOKEN env var. No extra sign-ups — just a token from Stephon's account
# with `repo` scope, set as a Colab/Kaggle secret.
#
# Usage:
#   python train/upload_release.py --run-dir /content/runs/plm-001 \
#       --tag v0.1-checkpoints --repo galactic-ai-team/playground-lm
#
# Behavior:
#   - Reads manifest.json from the run dir.
#   - For each checkpoint not already on the release, uploads the .safetensors
#     (NOT the optimizer state — that's only needed locally for resume).
#   - Also uploads manifest.json itself.
#   - Idempotent: re-running only uploads new checkpoints.

import argparse
import json
import os
import subprocess
import sys
import tempfile
from pathlib import Path


def run(cmd: list[str], check: bool = True) -> subprocess.CompletedProcess:
    return subprocess.run(cmd, check=check, capture_output=True, text=True)


def ensure_release(repo: str, tag: str) -> None:
    """Create the release if it doesn't exist."""
    result = run(["gh", "release", "view", tag, "--repo", repo], check=False)
    if result.returncode != 0:
        print(f"[upload] creating release {tag} in {repo}")
        run(
            [
                "gh", "release", "create", tag,
                "--repo", repo,
                "--title", f"PlaygroundLM checkpoints {tag}",
                "--notes", "Rolling training checkpoints (safetensors only). "
                           "See manifest.json for step and tokens_seen.",
                "--latest",
            ]
        )


def existing_assets(repo: str, tag: str) -> set[str]:
    result = run(
        ["gh", "release", "view", tag, "--repo", repo, "--json", "assets"],
    )
    data = json.loads(result.stdout)
    return {a["name"] for a in data.get("assets", [])}


def upload(run_dir: Path, repo: str, tag: str) -> None:
    manifest_path = run_dir / "manifest.json"
    if not manifest_path.exists():
        sys.exit(f"[upload] no manifest at {manifest_path}")

    with open(manifest_path) as f:
        manifest = json.load(f)

    ensure_release(repo, tag)
    present = existing_assets(repo, tag)

    # Always upload/refresh manifest.json.
    tmp_manifest = Path(tempfile.mkdtemp()) / "manifest.json"
    with open(tmp_manifest, "w") as f:
        json.dump(manifest, f, indent=2)
    # Delete existing manifest so the refresh overwrites.
    if "manifest.json" in present:
        run(["gh", "release", "delete-asset", tag, "manifest.json", "--repo", repo, "-y"])
    run(
        [
            "gh", "release", "upload", tag, str(tmp_manifest),
            "--repo", repo, "--clobber",
        ]
    )
    print(f"[upload] manifest.json -> {tag}")

    uploaded = 0
    for entry in manifest["checkpoints"]:
        name = Path(entry["path"]).name
        if name in present:
            continue
        weights = run_dir / entry["path"]
        if not weights.exists():
            print(f"[upload] WARN: manifest references missing {weights}, skipping")
            continue
        run(
            ["gh", "release", "upload", tag, str(weights), "--repo", repo, "--clobber"]
        )
        print(f"[upload] step {entry['step']:>8} -> {name}")
        uploaded += 1

    print(f"[upload] done. {uploaded} new file(s) uploaded.")


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--run-dir", required=True, type=Path)
    p.add_argument("--repo", required=True)
    p.add_argument("--tag", required=True)
    args = p.parse_args()

    if not os.environ.get("GH_TOKEN"):
        sys.exit("[upload] GH_TOKEN env var not set. Set it to a token with repo scope.")

    upload(args.run_dir, args.repo, args.tag)


if __name__ == "__main__":
    main()
