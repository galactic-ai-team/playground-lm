# scripts/test_make_notebook.py
# Tests for make_notebook.py. Focus on the marker parsing and the --check
# staleness contract, since that's what CI depends on.

import json
import subprocess
import sys
import tempfile
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parent))
from make_notebook import build_notebook, parse_cells


def test_empty_source_yields_zero_cells():
    assert parse_cells("") == []


def test_pure_code_single_cell():
    cells = parse_cells("x = 1\ny = 2\n")
    assert len(cells) == 1
    assert cells[0]["cell_type"] == "code"
    assert cells[0]["source"] == ["x = 1\n", "y = 2"]


def test_pure_markdown_single_cell():
    src = "# %% [markdown]\n# # Title\n#\n# Body text.\n"
    cells = parse_cells(src)
    assert len(cells) == 1
    assert cells[0]["cell_type"] == "markdown"
    # "# # Title" → "Title after stripping '# '; "#" → ""; "# Body text." → "Body text."
    assert cells[0]["source"][0] == "# Title\n"


def test_mixed_code_and_markdown():
    src = (
        "# %% [markdown]\n"
        "# # Setup\n"
        "# %%\n"
        "import torch\n"
        "# %%\n"
        "print('hello')\n"
    )
    cells = parse_cells(src)
    assert len(cells) == 3
    assert cells[0]["cell_type"] == "markdown"
    assert cells[1]["cell_type"] == "code"
    assert cells[2]["cell_type"] == "code"
    assert cells[1]["source"] == ["import torch"]
    assert cells[2]["source"] == ["print('hello')"]


def test_marker_only_comment_lines_not_in_code():
    """`# %%` lines themselves should not appear in cell bodies."""
    src = "# %%\nx = 1\n# %%\ny = 2\n"
    cells = parse_cells(src)
    for cell in cells:
        joined = "".join(cell["source"])
        assert "# %%" not in joined


def test_build_notebook_metadata():
    with tempfile.TemporaryDirectory() as tmp:
        p = Path(tmp) / "test.py"
        p.write_text("# %%\nx = 1\n")
        nb = build_notebook(p)
        assert nb["nbformat"] == 4
        assert nb["nbformat_minor"] == 5
        assert nb["metadata"]["kernelspec"]["name"] == "python3"
        assert len(nb["cells"]) == 1


def test_deterministic_output():
    """Same source → byte-identical output (no random cell IDs)."""
    with tempfile.TemporaryDirectory() as tmp:
        p = Path(tmp) / "src.py"
        p.write_text("# %%\nx = 1\n# %% [markdown]\n# Note\n")

        a = json.dumps(build_notebook(p), indent=1, sort_keys=True)
        b = json.dumps(build_notebook(p), indent=1, sort_keys=True)
        assert a == b


def test_check_mode_detects_stale():
    """--check exits 1 when the .ipynb doesn't match what the .py would generate."""
    with tempfile.TemporaryDirectory() as tmp:
        src = Path(tmp) / "src.py"
        out = Path(tmp) / "out.ipynb"
        src.write_text("# %%\nx = 1\n")

        script = Path(__file__).resolve().parent / "make_notebook.py"

        # Missing output → stale.
        r = subprocess.run(
            [sys.executable, str(script), "--src", str(src), "--out", str(out), "--check"],
            capture_output=True, text=True,
        )
        assert r.returncode == 1
        assert "STALE" in r.stderr

        # Generate, then --check should pass.
        subprocess.run(
            [sys.executable, str(script), "--src", str(src), "--out", str(out)],
            check=True, capture_output=True,
        )
        r = subprocess.run(
            [sys.executable, str(script), "--src", str(src), "--out", str(out), "--check"],
            capture_output=True, text=True,
        )
        assert r.returncode == 0

        # Modify source → --check should fail again.
        src.write_text("# %%\nx = 2\n")
        r = subprocess.run(
            [sys.executable, str(script), "--src", str(src), "--out", str(out), "--check"],
            capture_output=True, text=True,
        )
        assert r.returncode == 1


def test_check_mode_after_regenerate():
    """Regenerating makes --check pass again."""
    with tempfile.TemporaryDirectory() as tmp:
        src = Path(tmp) / "src.py"
        out = Path(tmp) / "out.ipynb"
        script = Path(__file__).resolve().parent / "make_notebook.py"

        src.write_text("# %%\nx = 1\n")
        subprocess.run(
            [sys.executable, str(script), "--src", str(src), "--out", str(out)],
            check=True, capture_output=True,
        )

        src.write_text("# %%\nx = 2\n")
        r = subprocess.run(
            [sys.executable, str(script), "--src", str(src), "--out", str(out), "--check"],
            capture_output=True,
        )
        assert r.returncode == 1

        subprocess.run(
            [sys.executable, str(script), "--src", str(src), "--out", str(out)],
            check=True, capture_output=True,
        )
        r = subprocess.run(
            [sys.executable, str(script), "--src", str(src), "--out", str(out), "--check"],
            capture_output=True,
        )
        assert r.returncode == 0