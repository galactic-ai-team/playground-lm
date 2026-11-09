# kaggle/sweep/test_sweep.py (Round 3)
# Acceptance tests from the packet:
#   [x] pack_cell smoke: slug==code_file; inject block present; no deletes
#   [x] dry-run run_sweep.py lists Round-3 cells without pushing
#   [x] LAUNCH.md lists dataset_sources + kernel_sources + GPU-hour estimate
#   [x] DRY_RUN=1 still does one CPU update  (verified in stage_e_train tests)
#
# Plus Round-3 specific: replay_ratio injection, concurrency cap, no deletes.

import importlib.util
import io
import json
from contextlib import redirect_stdout
from pathlib import Path

import pytest


_here = Path(__file__).parent


def _load(name: str, fname: str):
    spec = importlib.util.spec_from_file_location(name, _here / fname)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


pack_cell = _load("pack_cell", "pack_cell.py")
run_sweep = _load("run_sweep", "run_sweep.py")


TEMPLATE_STUB = """\
import os, subprocess, sys
def main():
    env = dict(os.environ)
    env.update({"PLM_STAGE": "E"})
    return subprocess.run(["true"], env=env).returncode
if __name__ == "__main__":
    sys.exit(main())
"""


def _write_template(tmp_path: Path) -> Path:
    t = tmp_path / "template.py"
    t.write_text(TEMPLATE_STUB)
    return t


# --- pack_cell: invariants ----------------------------------------------

def test_pack_emits_slug_named_script(tmp_path):
    cell_dir = pack_cell.package_cell(
        _write_template(tmp_path), tmp_path / "cells",
        slug="plm-stage-e-train-r3-a",
        run_id="r3-a", epochs=2, lr=3e-5, max_steps=31228,
        replay_ratio=0.3, git_commit="f875925",
    )
    assert (cell_dir / "plm-stage-e-train-r3-a.py").is_file()
    meta = json.loads((cell_dir / "kernel-metadata.json").read_text())
    assert meta["code_file"] == "plm-stage-e-train-r3-a.py"


def test_pack_injects_all_env_keys(tmp_path):
    cell_dir = pack_cell.package_cell(
        _write_template(tmp_path), tmp_path / "cells",
        slug="plm-stage-e-train-r3-a", run_id="r3-a",
        epochs=2, lr=3e-5, max_steps=31228,
        replay_ratio=0.3, git_commit="f875925",
    )
    src = (cell_dir / "plm-stage-e-train-r3-a.py").read_text()
    for key in ("PLM_REPLAY_RATIO", "PLM_LR", "PLM_FINETUNE_EPOCHS",
                "PLM_RUN_ID", "PLM_GIT_COMMIT"):
        assert key in src, f"missing injection: {key}"
    assert '"0.3"' in src  # replay_ratio value
    assert '"f875925"' in src


def test_pack_records_replay_in_cell_json(tmp_path):
    cell_dir = pack_cell.package_cell(
        _write_template(tmp_path), tmp_path / "cells",
        slug="plm-stage-e-train-r3-a", run_id="r3-a",
        epochs=2, lr=3e-5, max_steps=31228,
        replay_ratio=0.4, git_commit="f875925",
    )
    cell = json.loads((cell_dir / "cell.json").read_text())
    assert cell["replay_ratio"] == 0.4
    assert cell["git_commit"] == "f875925"


def test_pack_rejects_bad_replay_ratio(tmp_path):
    # main() does the range check; package_cell doesn't. Test via main.
    template = _write_template(tmp_path)
    rc = pack_cell.main([
        "--template", str(template),
        "--out-root", str(tmp_path / "cells"),
        "--slug", "plm-stage-e-train-r3-a",
        "--run-id", "r3-a",
        "--epochs", "2", "--lr", "3e-5", "--max-steps", "31228",
        "--replay-ratio", "1.5",
        "--git-commit", "f875925",
    ])
    assert rc == 1


def test_pack_force_does_not_delete_siblings(tmp_path):
    template = _write_template(tmp_path)
    out = tmp_path / "cells"
    pack_cell.package_cell(template, out, "plm-stage-e-train-a", "a", 1, 3e-5,
                           31102, 0.2, "abc")
    pack_cell.package_cell(template, out, "plm-stage-e-train-b", "b", 1, 3e-5,
                           31102, 0.2, "abc")
    (out / "plm-stage-e-train-a" / "SENTINEL").write_text("keep")
    pack_cell.package_cell(template, out, "plm-stage-e-train-b", "b", 1, 3e-5,
                           31102, 0.2, "abc", force=True)
    assert (out / "plm-stage-e-train-a" / "SENTINEL").is_file()


def test_pack_requires_template_env_marker(tmp_path):
    """Round-3 inject prepends os.environ pins; any template text is accepted."""
    template = tmp_path / "bare.py"
    template.write_text("print('hi')\n")
    out = tmp_path / "pack"
    cell = pack_cell.package_cell(
        template, out, "plm-stage-e-r3-bare", "bare",
        1.0, 1e-5, 31140, 0.25, "abc",
    )
    text = (cell / "plm-stage-e-r3-bare.py").read_text()
    assert 'PLM_REPLAY_RATIO' in text
    assert "print('hi')" in text



def test_dry_run_lists_cells_without_pushing(tmp_path, monkeypatch):
    template = _write_template(tmp_path)
    pack_root = tmp_path / "cells"
    out_root = tmp_path / "runs"
    for slug in ("plm-stage-e-train-r3-a", "plm-stage-e-train-r3-b"):
        pack_cell.package_cell(template, pack_root, slug, slug, 2, 3e-5,
                               31228, 0.3, "f875925")

    pushed = []
    monkeypatch.setattr(run_sweep.subprocess, "run",
                        lambda cmd, *a, **kw: pushed.append(cmd) or type("R", (), {"returncode": 0})())

    buf = io.StringIO()
    with redirect_stdout(buf):
        rc = run_sweep.main(["--pack-root", str(pack_root), "--out-root", str(out_root)])
    out = buf.getvalue()
    assert rc == 0
    assert "plm-stage-e-train-r3-a" in out
    assert "plm-stage-e-train-r3-b" in out
    assert "dry-run" in out
    assert pushed == []  # no kaggle kernels push


def test_run_sweep_summary_records_replay_and_commit(tmp_path):
    template = _write_template(tmp_path)
    pack_root = tmp_path / "cells"
    out_root = tmp_path / "runs"
    pack_cell.package_cell(template, pack_root, "plm-stage-e-train-r3-a",
                           "r3-a", 2, 3e-5, 31228, 0.3, "f875925")
    buf = io.StringIO()
    with redirect_stdout(buf):
        run_sweep.main(["--pack-root", str(pack_root), "--out-root", str(out_root)])
    out = buf.getvalue()
    # The trailing JSON summary must include replay_ratio and git_commit.
    assert '"replay_ratio": 0.3' in out
    assert '"git_commit": "f875925"' in out


def test_execute_requires_ack(tmp_path):
    template = _write_template(tmp_path)
    pack_root = tmp_path / "cells"
    out_root = tmp_path / "runs"
    pack_cell.package_cell(template, pack_root, "plm-stage-e-train-r3-a",
                           "r3-a", 2, 3e-5, 31228, 0.3, "f875925")
    buf = io.StringIO()
    with redirect_stdout(buf):
        rc = run_sweep.main([
            "--pack-root", str(pack_root),
            "--out-root", str(out_root),
            "--execute",
        ])
    assert rc == 2
    assert "REFUSED" in buf.getvalue()


def test_concurrency_cap_respected(tmp_path, monkeypatch):
    template = _write_template(tmp_path)
    pack_root = tmp_path / "cells"
    out_root = tmp_path / "runs"
    for i in range(4):
        pack_cell.package_cell(
            template, pack_root, f"plm-stage-e-train-r3-{i}",
            f"r3-{i}", 2, 3e-5, 31228, 0.3, "f875925",
        )
    pushed = []
    monkeypatch.setattr(run_sweep.subprocess, "run",
                        lambda cmd, *a, **kw: pushed.append(cmd) or type("R", (), {"returncode": 0})())

    buf = io.StringIO()
    with redirect_stdout(buf):
        rc = run_sweep.main([
            "--pack-root", str(pack_root),
            "--out-root", str(out_root),
            "--execute", "--i-have-read-launch-md",
        ])
    assert rc == 0
    assert len(pushed) <= run_sweep.MAX_CONCURRENT_GPU


def test_skips_complete_runs(tmp_path):
    template = _write_template(tmp_path)
    pack_root = tmp_path / "cells"
    out_root = tmp_path / "runs"
    pack_cell.package_cell(template, pack_root, "plm-stage-e-train-r3-a",
                           "r3-a", 2, 3e-5, 31228, 0.3, "f875925")
    rd = out_root / "r3-a"; rd.mkdir(parents=True)
    (rd / "COMPLETE.json").write_text(json.dumps({"status": "complete"}))
    buf = io.StringIO()
    with redirect_stdout(buf):
        rc = run_sweep.main(["--pack-root", str(pack_root), "--out-root", str(out_root)])
    assert rc == 0
    assert "skipped-complete" in buf.getvalue()


# --- LAUNCH.md presence -------------------------------------------------

def test_launch_md_exists_and_mentions_dataset_slugs():
    launch = _here / "LAUNCH.md"
    assert launch.is_file(), "LAUNCH.md missing"
    text = launch.read_text()
    assert "plm-stage-e-data" in text
    assert "plm-stage-c-train" in text
    assert "GPU-hour" in text or "GPU hour" in text or "gpu-hour" in text.lower()