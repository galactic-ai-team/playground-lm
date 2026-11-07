#!/usr/bin/env python3
"""Publish locally staged ONNX model references into the demo manifest.

This script does NOT upload files. Every checkpoint entry is checked against
its local ONNX file and given a SHA-256 digest. Existing benchmark scores,
riddle prompts and other manifest fields are retained unchanged.

Example:
    python -m scripts.publish_onnx_manifest \
        --staging-dir /workspace/plm_onnx/hf_staging \
        --index exports/onnx_checkpoints.json \
        --repo galactic-ai-team/playground-lm-onnx \
        --output demo/manifest.json

Index JSON (list or {"checkpoints": [...]}):
    [{"step": 10907, "tokens_seen": 1072164864,
      "filename": "step_10907_q4.onnx", "quant": "q4"}]

Alternatively repeat:
    --checkpoint STEP TOKENS_SEEN FILENAME QUANT

To stage before a Hugging Face repo exists, use --dry-run --repo OWNER/REPO.
Use --revision COMMIT_SHA for immutable resolve URLs after upload.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import re
import tempfile
from pathlib import Path, PurePosixPath
from typing import Any, Mapping
from urllib.parse import quote, urlsplit

DEFAULT_TEMPLATE = "https://huggingface.co/{repo}/resolve/{revision}/{filename}"
ALLOWED_QUANTS = frozenset({"q4", "q8", "int8"})
REPO_PATTERN = re.compile(r"^[A-Za-z0-9][A-Za-z0-9_.-]*/[A-Za-z0-9][A-Za-z0-9_.-]*$")
REVISION_PATTERN = re.compile(r"^[A-Za-z0-9][A-Za-z0-9_.\-/]*$")


def _integer(value: Any, field: str, *, allow_zero: bool = False) -> int:
    if isinstance(value, bool) or not isinstance(value, int) or value < (0 if allow_zero else 1):
        raise ValueError(f"{field} must be a {'nonnegative' if allow_zero else 'positive'} integer")
    return value


def _local_file(staging_dir: Path, filename: str) -> tuple[Path, str]:
    """Reject traversal/symlinks outside staging and normalize to a POSIX path."""
    if not isinstance(filename, str) or not filename or "\\" in filename:
        raise ValueError("filename must be a nonempty POSIX-relative path")
    posix = PurePosixPath(filename)
    if posix.is_absolute() or any(part in (".", "..") for part in filename.split("/")):
        raise ValueError(f"Unsafe staged filename: {filename}")
    if posix.suffix.lower() != ".onnx":
        raise ValueError(f"Not an ONNX file: {filename}")
    root = staging_dir.resolve(strict=True)
    source = (root / Path(*posix.parts)).resolve(strict=True)
    if not source.is_relative_to(root) or not source.is_file():
        raise ValueError(f"Staged file leaves directory or is not a file: {filename}")
    return source, posix.as_posix()


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _format_url(template: str, *, repo: str, revision: str, filename: str) -> str:
    if not REPO_PATTERN.fullmatch(repo):
        raise ValueError("--repo must be OWNER/REPO")
    if not REVISION_PATTERN.fullmatch(revision) or ".." in revision:
        raise ValueError("Invalid revision")
    if "{repo}" not in template or "{filename}" not in template:
        raise ValueError("URL template must include {repo} and {filename}")
    # Never accept arbitrary interpolation fields.
    try:
        url = template.format(
            repo=quote(repo, safe="/"),
            revision=quote(revision, safe="/"),
            filename=quote(filename, safe="/"),
        )
    except (KeyError, ValueError, IndexError) as exc:
        raise ValueError("Invalid URL template") from exc
    parsed = urlsplit(url)
    if parsed.scheme != "https" or not parsed.netloc or parsed.username or parsed.password:
        raise ValueError("ONNX URL must be an absolute HTTPS URL")
    if not parsed.path.endswith(".onnx") or parsed.query or parsed.fragment:
        raise ValueError("ONNX URL must point directly to an .onnx file")
    return url


def load_index(index: Path | None, inline: list[list[str]]) -> list[dict[str, Any]]:
    entries: list[Any] = []
    if index is not None:
        data = json.loads(index.read_text(encoding="utf-8"))
        if isinstance(data, dict):
            data = data.get("checkpoints")
        if not isinstance(data, list):
            raise ValueError("Index must be a list or {checkpoints: [...]} object")
        entries.extend(data)
    for step, tokens, filename, quant in inline:
        entries.append({"step": int(step), "tokens_seen": int(tokens),
                        "filename": filename, "quant": quant})
    if not entries:
        raise ValueError("Supply --index or at least one --checkpoint")
    normalized = []
    for entry in entries:
        if not isinstance(entry, Mapping):
            raise ValueError("Checkpoint entries must be objects")
        step = _integer(entry.get("step"), "step")
        tokens = _integer(entry.get("tokens_seen"), "tokens_seen", allow_zero=True)
        quant = entry.get("quant")
        if quant not in ALLOWED_QUANTS:
            raise ValueError(f"Unsupported quant for step {step}: {quant!r}")
        normalized.append({"step": step, "tokens_seen": tokens,
                           "filename": entry.get("filename"), "quant": quant})
    normalized.sort(key=lambda x: x["step"])
    steps = [e["step"] for e in normalized]
    if len(steps) != len(set(steps)):
        raise ValueError("Duplicate checkpoint step; choose one slider variant per step")
    return normalized


def build_rows(staging_dir: Path, entries: list[Mapping[str, Any]], *,
               repo: str, revision: str = "main",
               url_template: str = DEFAULT_TEMPLATE) -> list[dict[str, Any]]:
    """Build deterministic rows with verified local SHA-256 values."""
    rows = []
    for entry in entries:
        step = _integer(entry["step"], "step")
        tokens = _integer(entry["tokens_seen"], "tokens_seen", allow_zero=True)
        quant = entry["quant"]
        if quant not in ALLOWED_QUANTS:
            raise ValueError(f"Unsupported quant: {quant}")
        path, name = _local_file(staging_dir, entry["filename"])
        url = _format_url(url_template, repo=repo, revision=revision, filename=name)
        rows.append({
            "step": step,
            "tokens_seen": tokens,
            "url": url,          # Current demo/app.js slider compatibility.
            "onnx_url": url,     # Explicit field for future demo consumers.
            "quant": quant,
            "sha256": _sha256(path),
        })
    rows.sort(key=lambda x: x["step"])
    if len({r["step"] for r in rows}) != len(rows):
        raise ValueError("Duplicate checkpoint steps")
    return rows


def update_manifest(existing: Mapping[str, Any], rows: list[dict[str, Any]]) -> dict[str, Any]:
    """Replace checkpoints only, retaining all other manifest properties."""
    if not isinstance(existing, Mapping):
        raise ValueError("Existing manifest must be a JSON object")
    updated = dict(existing)
    updated["checkpoints"] = rows
    return updated


def write_atomic(path: Path, document: Mapping[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    serialized = json.dumps(document, indent=2, ensure_ascii=False, allow_nan=False) + "\n"
    temp = None
    try:
        with tempfile.NamedTemporaryFile(
            mode="w", encoding="utf-8", dir=path.parent,
            prefix=f".{path.name}.", suffix=".tmp", delete=False,
        ) as stream:
            temp = Path(stream.name)
            stream.write(serialized)
            stream.flush()
            os.fsync(stream.fileno())
        os.replace(temp, path)
    finally:
        if temp is not None:
            temp.unlink(missing_ok=True)


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--staging-dir", required=True, type=Path)
    parser.add_argument("--index", type=Path, help="JSON checkpoint index")
    parser.add_argument("--checkpoint", nargs=4, action="append", default=[],
                        metavar=("STEP", "TOKENS", "FILENAME", "QUANT"))
    parser.add_argument("--repo", required=True, help="Hugging Face OWNER/REPO")
    parser.add_argument("--revision", default="main", help="Commit SHA preferred after upload")
    parser.add_argument("--url-template", default=DEFAULT_TEMPLATE,
                        help="Must contain {repo} and {filename}; optional {revision}")
    parser.add_argument("--output", type=Path, default=Path("demo/manifest.json"))
    parser.add_argument("--dry-run", action="store_true", help="Print proposed manifest without writing")
    args = parser.parse_args(argv)
    try:
        entries = load_index(args.index, args.checkpoint)
        rows = build_rows(args.staging_dir, entries, repo=args.repo,
                          revision=args.revision, url_template=args.url_template)
        existing = json.loads(args.output.read_text(encoding="utf-8"))
        updated = update_manifest(existing, rows)
    except (ValueError, TypeError, KeyError, OSError, json.JSONDecodeError) as exc:
        parser.error(str(exc))
    if args.dry_run:
        print(json.dumps(updated, indent=2, ensure_ascii=False))
    else:
        write_atomic(args.output, updated)
        print(json.dumps({"manifest": str(args.output), "checkpoints": len(rows),
                          "quants": {q: sum(r["quant"] == q for r in rows)
                                     for q in sorted({r["quant"] for r in rows})}}, indent=2))


if __name__ == "__main__":
    main()
