# src/playground_lm/train/test_upload_release.py
# Tests for upload_release.upload(). The `gh` CLI is mocked via subprocess.run
# monkeypatching so these run without network or a real GitHub token.
#
# Two gotchas in the mocking:
#   1. `gh release upload <tag> <file> --repo <r> --clobber` — the file is at
#      argv[3], not argv[-1] (the last arg is "--clobber").
#   2. `gh release view` is called twice: once bare (existence check) and once
#      with `--json assets`. The bare call must be distinguished from the
#      JSON call BEFORE deciding what to return, or json.loads fails on "".

import json
import os
import tempfile
from pathlib import Path
from unittest.mock import patch, MagicMock

import pytest

from playground_lm.train.upload_release import upload


def _make_run_dir(tmp: Path, steps: list[int]) -> Path:
    """Create a run_dir with a manifest and dummy checkpoint files."""
    ckpt_dir = tmp / "checkpoints"
    ckpt_dir.mkdir(parents=True)
    entries = []
    for step in steps:
        name = f"step_{step:08d}.safetensors"
        (ckpt_dir / name).write_bytes(b"fake weights")
        entries.append({
            "step": step,
            "path": f"checkpoints/{name}",
            "tokens_seen": step * 1000,
            "wall_time_s": step * 0.5,
            "created_at": 1000.0 + step,
            "is_milestone": False,
        })
    manifest = {
        "run_id": "test",
        "config": {},
        "checkpoints": entries,
        "latest_step": max(steps) if steps else -1,
        "latest_tokens_seen": 0,
        "latest_wall_time_s": 0.0,
    }
    (tmp / "manifest.json").write_text(json.dumps(manifest))
    return tmp


def _make_gh_mock(
    existing_assets: set[str],
    release_exists: bool = True,
    calls: list[list[str]] | None = None,
):
    """
    Return a side_effect for subprocess.run that simulates the gh CLI.

    Distinguishes the two `gh release view` invocations by argv shape:
      - `gh release view <tag> --repo <repo>`             → existence check
      - `gh release view <tag> --repo <repo> --json assets` → asset listing

    Records all argv lists into `calls` if provided.
    """
    def fake_run(cmd, check=True, capture_output=True, text=True):
        if calls is not None:
            calls.append(cmd)

        result = MagicMock()
        result.returncode = 0
        result.stdout = ""
        result.stderr = ""

        # gh release view ... --json assets  → return asset list
        if (
            len(cmd) >= 4
            and cmd[0] == "gh"
            and cmd[1] == "release"
            and cmd[2] == "view"
            and "--json" in cmd
        ):
            result.stdout = json.dumps(
                {"assets": [{"name": n} for n in existing_assets]}
            )
            return result

        # gh release view ... (bare) → existence check
        if (
            len(cmd) >= 3
            and cmd[0] == "gh"
            and cmd[1] == "release"
            and cmd[2] == "view"
        ):
            if not release_exists:
                result.returncode = 1
            return result

        # gh release create / upload / delete-asset → success
        return result

    return fake_run


def _uploaded_filenames(calls: list[list[str]]) -> list[str]:
    """Extract filenames from `gh release upload` invocations.
    argv shape: ['gh', 'release', 'upload', <tag>, <file>, '--repo', <repo>, '--clobber']
    File is at index 4."""
    names = []
    for cmd in calls:
        if (
            len(cmd) >= 5
            and cmd[0] == "gh"
            and cmd[1] == "release"
            and cmd[2] == "upload"
        ):
            names.append(Path(cmd[4]).name)
    return names


# --- Tests ----------------------------------------------------------------

def test_upload_requires_token(monkeypatch):
    """No gh_token arg and no GH_TOKEN env → ValueError."""
    monkeypatch.delenv("GH_TOKEN", raising=False)
    with tempfile.TemporaryDirectory() as tmp:
        run_dir = _make_run_dir(Path(tmp), steps=[1])
        with pytest.raises(ValueError, match="no GitHub token"):
            upload(run_dir, "owner/repo", "v1", gh_token=None)


def test_upload_uses_explicit_token():
    """Explicit gh_token arg is set on GH_TOKEN env for the gh subprocess."""
    with tempfile.TemporaryDirectory() as tmp:
        run_dir = _make_run_dir(Path(tmp), steps=[1])
        with patch.dict(os.environ, {}, clear=True):
            with patch("subprocess.run", side_effect=_make_gh_mock(set())):
                upload(run_dir, "owner/repo", "v1", gh_token="ghp_test")
                assert os.environ.get("GH_TOKEN") == "ghp_test"


def test_upload_uploads_new_checkpoints():
    """Checkpoints not already on the release are uploaded."""
    with tempfile.TemporaryDirectory() as tmp:
        run_dir = _make_run_dir(Path(tmp), steps=[100, 200, 300])
        calls: list[list[str]] = []

        with patch("subprocess.run", side_effect=_make_gh_mock(set(), calls=calls)):
            result = upload(run_dir, "owner/repo", "v1", gh_token="ghp_test")

        uploaded = _uploaded_filenames(calls)
        assert "manifest.json" in uploaded
        assert "step_00000100.safetensors" in uploaded
        assert "step_00000200.safetensors" in uploaded
        assert "step_00000300.safetensors" in uploaded
        assert len(result.uploaded) == 3
        assert result.manifest_uploaded is True


def test_upload_skips_existing_checkpoints():
    """Checkpoints already on the release are not re-uploaded."""
    with tempfile.TemporaryDirectory() as tmp:
        run_dir = _make_run_dir(Path(tmp), steps=[100, 200])
        existing = {"step_00000100.safetensors"}
        calls: list[list[str]] = []

        with patch(
            "subprocess.run",
            side_effect=_make_gh_mock(existing, calls=calls),
        ):
            result = upload(run_dir, "owner/repo", "v1", gh_token="ghp_test")

        uploaded = _uploaded_filenames(calls)
        assert "step_00000100.safetensors" not in uploaded
        assert "step_00000200.safetensors" in uploaded
        assert result.uploaded == ["step_00000200.safetensors"]
        assert result.skipped == ["step_00000100.safetensors"]


def test_manifest_always_refreshed():
    """Even when all checkpoints already exist, manifest.json is re-uploaded."""
    with tempfile.TemporaryDirectory() as tmp:
        run_dir = _make_run_dir(Path(tmp), steps=[100])
        existing = {"step_00000100.safetensors"}
        calls: list[list[str]] = []

        with patch(
            "subprocess.run",
            side_effect=_make_gh_mock(existing, calls=calls),
        ):
            upload(run_dir, "owner/repo", "v1", gh_token="ghp_test")

        uploaded = _uploaded_filenames(calls)
        assert "manifest.json" in uploaded


def test_creates_release_if_missing():
    """If `gh release view` (bare) fails, `gh release create` is called."""
    with tempfile.TemporaryDirectory() as tmp:
        run_dir = _make_run_dir(Path(tmp), steps=[100])
        calls: list[list[str]] = []

        # release_exists=False makes the bare view return rc=1.
        with patch(
            "subprocess.run",
            side_effect=_make_gh_mock(set(), release_exists=False, calls=calls),
        ):
            result = upload(run_dir, "owner/repo", "v1", gh_token="ghp_test")

        create_calls = [
            c for c in calls
            if len(c) >= 3 and c[0] == "gh" and c[1] == "release" and c[2] == "create"
        ]
        assert len(create_calls) == 1
        assert result.release_created is True


def test_missing_manifest_raises():
    """No manifest.json → SystemExit, not silent success."""
    with tempfile.TemporaryDirectory() as tmp:
        with patch("subprocess.run", side_effect=_make_gh_mock(set())):
            with pytest.raises(SystemExit, match="no manifest"):
                upload(Path(tmp), "owner/repo", "v1", gh_token="ghp_test")


def test_skips_checkpoint_missing_on_disk():
    """A manifest entry pointing to a missing file is warned about and skipped."""
    with tempfile.TemporaryDirectory() as tmp:
        run_dir = _make_run_dir(Path(tmp), steps=[100, 200])
        # Delete one of the files behind the manifest's back.
        (run_dir / "checkpoints" / "step_00000100.safetensors").unlink()
        calls: list[list[str]] = []

        with patch("subprocess.run", side_effect=_make_gh_mock(set(), calls=calls)):
            result = upload(run_dir, "owner/repo", "v1", gh_token="ghp_test")

        assert "step_00000200.safetensors" in result.uploaded
        assert "step_00000100.safetensors" in result.skipped