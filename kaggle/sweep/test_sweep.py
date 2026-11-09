# kaggle/sweep/test_sweep.py
# Acceptance tests from the packet:
#   1. Packaging emits <slug>.py + kernel-metadata.json with code_file matching.
#   2. Launcher refuses to delete existing output dirs.
#   3. Manifest lists run_id → artifact paths; resume skips COMPLETE runs.
#   4. Recovery authority documented.

import importlib.util
import json
import sys
from pathlib import Path

import pytest


# Import pack_cell and run_sweep by path.
_here = Path(__file__).parent


def _load(name: str, fname: str):
    spec = importlib.util.spec_from_file_location(name, _here / fname)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


pack_cell = _load("pack_cell", "pack_cell.py")
run_sweep = _load("run_sweep", "run_sweep.py")


# --- Test 1: packaging invariants ----------------------------------------

def test_pack_emits_slug_named_script(tmp_path):
    template = tmp_path / "template.py"
    template.write_text("# stub launcher\n")
    out_root = tmp_path / "cells"

    cell_dir = pack_cell.package_cell(
        template, out_root,
        slug="plm-stage-e-train-sweep-a",
        run_id="sweep-a",
        epochs=2, lr=3e-5, max_steps=31228,
    )

    assert (cell_dir / "plm-stage-e-train-sweep-a.py").is_file()
    meta = json.loads((cell_dir / "kernel-metadata.json").read_text())
    assert meta["code_file"] == "plm-stage-e-train-sweep-a.py"
    assert meta["id"] == "galacticaiteam/plm-stage-e-train-sweep-a"


def test_pack_rejects_mismatched_slug(tmp_path):
    template = tmp_path / "template.py"
    template.write_text("# stub\n")
    with pytest.raises(ValueError, match="invalid slug"):
        pack_cell.package_cell(
            template, tmp_path / "cells",
            slug="Has-Uppercase-And-Spaces",
            run_id="x", epochs=1, lr=3e-5, max_steps=31102,
        )


def test_pack_refuses_to_overwrite(tmp_path):
    template = tmp_path / "template.py"
    template.write_text("# stub\n")
    out = tmp_path / "cells"
    pack_cell.package_cell(template, out, "plm-stage-e-train-a",
                           "run-a", 1, 3e-5, 31102)
    with pytest.raises(FileExistsError):
        pack_cell.package_cell(template, out, "plm-stage-e-train-a",
                               "run-a", 1, 3e-5, 31102)


def test_pack_force_overwrites_only_that_cell(tmp_path):
    """--force rewrites the cell; it never touches sibling cells."""
    template = tmp_path / "template.py"
    template.write_text("# stub\n")
    out = tmp_path / "cells"

    pack_cell.package_cell(template, out, "plm-stage-e-train-a", "a", 1, 3e-5, 31102)
    pack_cell.package_cell(template, out, "plm-stage-e-train-b", "b", 1, 3e-5, 31102)

    # Marker inside cell a to prove it survives.
    (out / "plm-stage-e-train-a" / "SENTINEL").write_text("keep")
    (out / "plm-stage-e-train-b" / "SENTINEL").write_text("keep")

    pack_cell.package_cell(template, out, "plm-stage-e-train-b", "b", 1, 3e-5, 31102, force=True)

    # cell-a untouched.
    assert (out / "plm-stage-e-train-a" / "SENTINEL").is_file()
    # cell-b's sentinel was preserved (force only overwrites via copy, no rmtree).
    assert (out / "plm-stage-e-train-b" / "SENTINEL").is_file()


# --- Test 2: launcher never deletes -------------------------------------

def test_launcher_preserves_existing_output_dirs(tmp_path):
    """A pre-existing run_id directory must not be touched during dry-run."""
    pack_root = tmp_path / "cells"
    out_root = tmp_path / "runs"
    template = tmp_path / "template.py"; template.write_text("# stub\n")

    pack_cell.package_cell(template, pack_root, "plm-stage-e-train-a", "run-a", 1, 3e-5, 31102)

    # Pre-existing run dir with a marker file.
    rd = out_root / "run-a"; rd.mkdir(parents=True)
    (rd / "keep.txt").write_text("preserve me")

    # Dry-run: launcher should not touch rd.
    rc = run_sweep.main([
        "--pack-root", str(pack_root),
        "--out-root", str(out_root),
    ])
    assert rc == 0
    assert (rd / "keep.txt").read_text() == "preserve me"


def test_launcher_skips_complete_runs(tmp_path):
    """A run with COMPLETE.json must be skipped."""
    pack_root = tmp_path / "cells"
    out_root = tmp_path / "runs"
    template = tmp_path / "template.py"; template.write_text("# stub\n")
    pack_cell.package_cell(template, pack_root, "plm-stage-e-train-a", "run-a", 1, 3e-5, 31102)

    rd = out_root / "run-a"; rd.mkdir(parents=True)
    (rd / "COMPLETE.json").write_text(json.dumps({"status": "complete", "step": 31128}))

    summary_out = []
    import io
    from contextlib import redirect_stdout
    buf = io.StringIO()
    with redirect_stdout(buf):
        rc = run_sweep.main([
            "--pack-root", str(pack_root),
            "--out-root", str(out_root),
        ])
    assert rc == 0
    out = buf.getvalue()
    assert "skipped-complete" in out
    # Summary JSON should record it.
    assert '"skipped-complete"' in out


# --- Test 3: manifest lists run_id → artifact paths ---------------------

def test_sweep_grid_builds_one_cell_per_combo():
    grid = run_sweep.build_sweep_grid(
        base_slug="plm-stage-e-train-sweep",
        lr_values=[3e-5, 6e-5],
        epochs_values=[1, 2],
        max_steps=31228,
    )
    assert len(grid) == 4
    slugs = {c["slug"] for c in grid}
    assert "plm-stage-e-train-sweep-lr3e5-ep1" in slugs
    assert "plm-stage-e-train-sweep-lr6e5-ep2" in slugs
    for c in grid:
        assert c["run_id"] == c["slug"]
        assert c["max_steps"] == 31228


def test_pack_writes_cell_json_with_run_id(tmp_path):
    template = tmp_path / "template.py"; template.write_text("# stub\n")
    cell = pack_cell.package_cell(
        template, tmp_path / "cells",
        "plm-stage-e-train-a", "run-a", 2, 3e-5, 31228,
    )
    cell_data = json.loads((cell / "cell.json").read_text())
    assert cell_data == {
        "slug": "plm-stage-e-train-a",
        "run_id": "run-a",
        "epochs": 2,
        "lr": 3e-5,
        "max_steps": 31228,
    }


# --- Test 4: refusal to execute without consent -------------------------

def test_execute_requires_launch_md_ack(tmp_path, monkeypatch):
    pack_root = tmp_path / "cells"
    out_root = tmp_path / "runs"
    template = tmp_path / "template.py"; template.write_text("# stub\n")
    pack_cell.package_cell(template, pack_root, "plm-stage-e-train-a", "run-a", 1, 3e-5, 31102)

    import io
    from contextlib import redirect_stdout
    buf = io.StringIO()
    with redirect_stdout(buf):
        rc = run_sweep.main([
            "--pack-root", str(pack_root),
            "--out-root", str(out_root),
            "--execute",  # no --i-have-read-launch-md
        ])
    assert rc == 2
    assert "REFUSED" in buf.getvalue()


def test_dry_run_never_calls_kaggle_push(tmp_path, monkeypatch):
    """Dry-run must not invoke `kaggle kernels push`."""
    pack_root = tmp_path / "cells"
    out_root = tmp_path / "runs"
    template = tmp_path / "template.py"; template.write_text("# stub\n")
    pack_cell.package_cell(template, pack_root, "plm-stage-e-train-a", "run-a", 1, 3e-5, 31102)

    called = []
    def fake_run(cmd, *a, **kw):
        called.append(cmd)
        class R: returncode = 0
        return R()
    monkeypatch.setattr(run_sweep.subprocess, "run", fake_run)

    import io
    from contextlib import redirect_stdout
    with redirect_stdout(io.StringIO()):
        rc = run_sweep.main([
            "--pack-root", str(pack_root),
            "--out-root", str(out_root),
        ])
    assert rc == 0
    assert called == []  # no kaggle calls in dry-run