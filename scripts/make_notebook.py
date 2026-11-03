#!/usr/bin/env python3
# scripts/make_notebook.py
# Convert a Jupytext-style .py script (with `# %%` cell markers) into an
# .ipynb notebook. Single source of truth is the .py; the .ipynb is generated
# and committed so Kaggle can open it directly without a Jupytext install.
#
# Marker conventions (Jupytext / VS Code compatible):
#   # %%              — start of a code cell
#   # %% [markdown]   — start of a markdown cell; the rest of the cell's lines
#                       are the markdown text (with `# ` prefix stripped if
#                       present, so the .py still reads as a valid .py file)
#
# Usage:
#   python scripts/make_notebook.py                        # regenerate default paths
#   python scripts/make_notebook.py --check                # exit nonzero if stale
#   python scripts/make_notebook.py --src X.py --out Y.ipynb
#
# Determinism: no timestamps, no random cell IDs. Same input → byte-identical
# output. This is what makes `--check` safe to run in CI.

import argparse
import json
import sys
from pathlib import Path


DEFAULT_SRC = "notebooks/playground_lm_train.py"
DEFAULT_OUT = "notebooks/playground_lm_train.ipynb"

# Fixed kernel metadata so Kaggle opens this with the right interpreter.
KERNEL_METADATA = {
    "kernelspec": {
        "display_name": "Python 3",
        "language": "python",
        "name": "python3",
    },
    "language_info": {
        "name": "python",
        "version": "3.11",
    },
}


def parse_cells(src: str) -> list[dict]:
    """
    Parse a .py file with Jupytext markers into a list of notebook cells.

    Returns a list of dicts in nbformat v4 cell format.
    """
    cells: list[dict] = []
    current_kind = "code"
    current_lines: list[str] = []

    def flush():
        nonlocal current_lines, current_kind
        if not current_lines:
            current_kind = "code"
            return
        if current_kind == "markdown":
            # Strip a leading "# " or "#" prefix from each line so the .py
            # file still reads as valid Python comments.
            text_lines = []
            for line in current_lines:
                if line.startswith("# "):
                    text_lines.append(line[2:])
                elif line == "#":
                    text_lines.append("")
                elif line.startswith("#"):
                    text_lines.append(line[1:])
                else:
                    text_lines.append(line)
            cells.append(_md_cell("\n".join(text_lines).rstrip() + "\n"))
        else:
            code = "\n".join(current_lines)
            # Remove the trailing marker line if present (handled below).
            cells.append(_code_cell(code.rstrip() + "\n"))
        current_lines = []
        current_kind = "code"

    for line in src.splitlines():
        stripped = line.strip()
        if stripped == "# %%":
            flush()
            current_kind = "code"
            continue
        if stripped == "# %% [markdown]":
            flush()
            current_kind = "markdown"
            continue
        current_lines.append(line)

    flush()
    return cells


def _md_cell(source: str) -> dict:
    return {
        "cell_type": "markdown",
        "metadata": {},
        "source": _split_source(source),
    }


def _code_cell(source: str) -> dict:
    return {
        "cell_type": "code",
        "execution_count": None,
        "metadata": {},
        "outputs": [],
        "source": _split_source(source),
    }


def _split_source(source: str) -> list[str]:
    """Notebook source is a list of lines (each ending in \\n except possibly
    the last)."""
    lines = source.splitlines(keepends=True)
    if not lines:
        return []
    # Ensure the last line has no trailing newline (nbformat accepts either,
    # but keeping it consistent avoids --check false positives).
    if lines[-1].endswith("\n"):
        lines[-1] = lines[-1][:-1]
    return lines


def build_notebook(src_path: Path) -> dict:
    src = src_path.read_text(encoding="utf-8")
    return {
        "cells": parse_cells(src),
        "metadata": KERNEL_METADATA,
        "nbformat": 4,
        "nbformat_minor": 5,
    }


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Convert a Jupytext-style .py file to .ipynb."
    )
    parser.add_argument("--src", default=DEFAULT_SRC, help="source .py path")
    parser.add_argument("--out", default=DEFAULT_OUT, help="output .ipynb path")
    parser.add_argument(
        "--check",
        action="store_true",
        help="exit nonzero if the .ipynb is stale (do not write)",
    )
    args = parser.parse_args()

    src_path = Path(args.src)
    out_path = Path(args.out)

    if not src_path.exists():
        print(f"[make_notebook] source not found: {src_path}", file=sys.stderr)
        return 2

    nb = build_notebook(src_path)
    text = json.dumps(nb, indent=1, sort_keys=True) + "\n"

    if args.check:
        if not out_path.exists():
            print(
                f"[make_notebook] STALE: {out_path} does not exist",
                file=sys.stderr,
            )
            return 1
        existing = out_path.read_text(encoding="utf-8")
        if existing != text:
            print(
                f"[make_notebook] STALE: {out_path} differs from what "
                f"{src_path} would generate.\n"
                f"  Regenerate with: python {sys.argv[0]}",
                file=sys.stderr,
            )
            return 1
        print(f"[make_notebook] up to date: {out_path}")
        return 0

    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(text, encoding="utf-8")
    print(f"[make_notebook] wrote {out_path} ({len(nb['cells'])} cells)")
    return 0


if __name__ == "__main__":
    sys.exit(main())