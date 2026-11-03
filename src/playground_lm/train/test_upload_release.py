# src/playground_lm/train/test_upload_release.py
# Tests for upload_release.upload(). The `gh` CLI is mocked via subprocess.run
# monkeypatching so these run without network or a real GitHub token.

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


def _fake_run_factory(existing_assets: set[str], release_exists: bool = True):
    """Return a mock for subprocess.run that simulates the gh CLI."""
    def fake_run(cmd, check=True, capture_output=True, text=True):
        result = MagicMock()
        result.returncode = 0
        result.stdout = ""
        result.stderr = ""

        # gh release view --repo R TAG
        if cmd[:3] == ["gh", "release", "view"]:
            if not release_exists:
                result.returncode = 1
            return result

        # gh release view --repo R TAG --json assets
        if "assets" in cmd and "--json" in cmd:
            assets = [{"name": name} for name in existing_assets]
            result.stdout = json.dumps({"assets": assets})
            return result

        # gh release create / upload
        return result
    return fake_run


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
            with patch("subprocess.run", side_effect=_fake_run_factory(set())):
                upload(run_dir, "owner/repo", "v1", gh_token="ghp_test")
                assert os.environ.get("GH_TOKEN") == "ghp_test"


def test_upload_uploads_new_checkpoints():
    """Checkpoints not already on the release are uploaded."""
    with tempfile.TemporaryDirectory() as tmp:
        run_dir = _make_run_dir(Path(tmp), steps=[100, 200, 300])
        calls: list[list[str]] = []

        def fake(cmd, check=True, capture_output=True, text=True):
            calls.append(cmd)
            r = MagicMock()
            r.returncode = 0
            r.stdout = ""
            r.stderr = ""
            if cmd[:3] == ["gh", "release", "view"] and "--json" in cmd:
                r.stdout = json.dumps({"assets": []})
            return r

        with patch("subprocess.run", side_effect=fake):
            result = upload(run_dir, "owner/repo", "v1", gh_token="ghp_test")

        uploaded_names = [Path(c[-1]).name for c in calls if c[:3] == ["gh", "release", "upload"]]
        # Should upload: manifest.json + 3 checkpoints
        assert "manifest.json" in uploaded_names
        assert "step_00000100.safetensors" in uploaded_names
        assert "step_00000200.safetensors" in uploaded_names
        assert "step_00000300.safetensors" in uploaded_names
        assert len(result.uploaded) == 3
        assert result.manifest_uploaded is True


def test_upload_skips_existing_checkpoints():
    """Checkpoints already on the release are not re-uploaded."""
    with tempfile.TemporaryDirectory() as tmp:
        run_dir = _make_run_dir(Path(tmp), steps=[100, 200])
        existing = {"step_00000100.safetensors"}
        calls: list[list[str]] = []

        def fake(cmd, check=True, capture_output=True, text=True):
            calls.append(cmd)
            r = MagicMock()
            r.returncode = 0
            r.stdout = ""
            r.stderr = ""
            if cmd[:3] == ["gh", "release", "view"] and "--json" in cmd:
                r.stdout = json.dumps({"assets": [{"name": n} for n in existing]})
            return r

        with patch("subprocess.run", side_effect=fake):
            result = upload(run_dir, "owner/repo", "v1", gh_token="ghp_test")

        uploaded_names = [Path(c[-1]).name for c in calls if c[:3] == ["gh", "release", "upload"]]
        assert "step_00000100.safetensors" not in uploaded_names
        assert "step_00000200.safetensors" in uploaded_names
        assert result.uploaded == ["step_00000200.safetensors"]
        assert result.skipped == ["step_00000100.safetensors"]


def test_manifest_always_refreshed():
    """Even when all checkpoints already exist, manifest.json is re-uploaded."""
    with tempfile.TemporaryDirectory() as tmp:
        run_dir = _make_run_dir(Path(tmp), steps=[100])
        existing = {"step_00000100.safetensors"}
        calls: list[list[str]] = []

        def fake(cmd, check=True, capture_output=True, text=True):
            calls.append(cmd)
            r = MagicMock()
            r.returncode = 0
            r.stdout = ""
            r.stderr = ""
            if cmd[:3] == ["gh", "release", "view"] and "--json" in cmd:
                r.stdout = json.dumps({"assets": [{"name": n} for n in existing]})
            return r

        with patch("subprocess.run", side_effect=fake):
            upload(run_dir, "owner/repo", "v1", gh_token="ghp_test")

        uploaded_names = [Path(c[-1]).name for c in calls if c[:3] == ["gh", "release", "upload"]]
        assert "manifest.json" in uploaded_names


def test_creates_release_if_missing():
    """If `gh release view` fails, `gh release create` is called."""
    with tempfile.TemporaryDirectory() as tmp:
        run_dir = _make_run_dir(Path(tmp), steps=[100])
        calls: list[list[str]] = []

        def fake(cmd, check=True, capture_output=True, text=True):
            calls.append(cmd)
            r = MagicMock()
            r.returncode = 0
            r.stdout = ""
            r.stderr = ""
            # First view (release exists?) fails; second view (--json assets)
            # succeeds with empty (because we just created the release).
            if cmd[:3] == ["gh", "release", "view"] and "--json" not in cmd:
                r.returncode = 1
            elif cmd[:3] == ["gh", "release", "view"] and "--json" in cmd:
                r.stdout = json.dumps({"assets": []})
            return r

        with patch("subprocess.run", side_effect=fake):
            result = upload(run_dir, "owner/repo", "v1", gh_token="ghp_test")

        create_calls = [c for c in calls if c[:3] == ["gh", "release", "create"]]
        assert len(create_calls) == 1
        assert result.release_created is True


def test_missing_manifest_raises():
    """No manifest.json → SystemExit, not silent success."""
    with tempfile.TemporaryDirectory() as tmp:
        with patch("subprocess.run", side_effect=_fake_run_factory(set())):
            with pytest.raises(SystemExit, match="no manifest"):
                upload(Path(tmp), "owner/repo", "v1", gh_token="ghp_test")


def test_skips_checkpoint_missing_on_disk():
    """A manifest entry pointing to a missing file is warned about and skipped."""
    with tempfile.TemporaryDirectory() as tmp:
        run_dir = _make_run_dir(Path(tmp), steps=[100, 200])
        # Delete one of the files behind the manifest's back.
        (run_dir / "checkpoints" / "step_00000100.safetensors").unlink()
        calls: list[list[str]] = []

        def fake(cmd, check=True, capture_output=True, text=True):
            calls.append(cmd)
            r = MagicMock()
            r.returncode = 0
            r.stdout = ""
            r.stderr = ""
            if cmd[:3] == ["gh", "release", "view"] and "--json" in cmd:
                r.stdout = json.dumps({"assets": []})
            return r

        with patch("subprocess.run", side_effect=fake):
            result = upload(run_dir, "owner/repo", "v1", gh_token="ghp_test")

        # 200 should still upload; 100 is skipped because file is gone.
        assert "step_00000200.safetensors" in result.uploaded
        assert "step_00000100.safetensors" in result.skipped