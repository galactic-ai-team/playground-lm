#!/usr/bin/env python3
"""Build demo tokenizer assets and a deterministic checkpoint manifest.

Input checkpoints JSON: a list of {step, tokens_seen, url} objects,
where url points to an exported model.int8.onnx. You may also pass
--checkpoint STEP TOKENS_SEEN URL repeatedly.

Attach reports with repeated --report STEP=FILE arguments. One report
may contain validation and benchmarks, or two reports may supply them
separately (run_eval.py and benchmarks.py). Only complete, full-split
scores enter the table. --baseline points to baseline_gpt2.py JSON.

Examples:
  python -m scripts.build_demo_manifest \\
    --checkpoints exports/checkpoints.json \\
    --report 2035=evals/results/stage_b_2035.json \\
    --baseline evals/results/gpt2_124m.json

  python -m scripts.build_demo_manifest \\
    --checkpoint 2035 200000000 https://example.org/model.int8.onnx \\
    --report 2035=evals/results/validation.json \\
    --report 2035=evals/results/benchmarks.json

Tokenizer assets are copied from the exact SHA256-verified GPT-2 assets
used by get_tokenizer(). The demo frontend must separately implement
real GPT-2 byte-level BPE to consume encoder.json and merges.txt.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import os
import tempfile
from pathlib import Path
from typing import Any, Mapping
from urllib.parse import urlsplit

from playground_lm.tokenizer import ASSETS, EOT_ID, VOCAB_SIZE, get_tokenizer
from evals.publish_checkpoint import COLUMNS, make_row, write_json_atomic


def _positive_int(value: Any, name: str, *, zero: bool = False) -> int:
    if isinstance(value, bool) or not isinstance(value, int) or value < (0 if zero else 1):
        raise ValueError(f"{name} must be a {'nonnegative' if zero else 'positive'} integer")
    return value


def _checkpoint(item: Mapping[str, Any]) -> dict[str, Any]:
    step = _positive_int(item["step"], "step")
    tokens = _positive_int(item["tokens_seen"], "tokens_seen", zero=True)
    url = item["url"]
    if not isinstance(url, str) or not url.strip() or url != url.strip():
        raise ValueError("checkpoint url must be a nonempty string")
    parsed = urlsplit(url)
    if parsed.scheme and parsed.scheme not in ("https", "http"):
        raise ValueError("checkpoint url must use http(s) or a relative path")
    if parsed.scheme and not parsed.netloc:
        raise ValueError("Invalid absolute checkpoint URL")
    if not parsed.path.endswith(".int8.onnx"):
        raise ValueError("checkpoint url must reference an INT8 .int8.onnx artifact")
    return {"step": step, "tokens_seen": tokens, "url": url}


def read_checkpoints(path: Path | None, inline: list[list[str]]) -> list[dict[str, Any]]:
    items: list[Mapping[str, Any]] = []
    if path is not None:
        raw = json.loads(path.read_text(encoding="utf-8"))
        if isinstance(raw, dict):
            raw = raw.get("checkpoints")
        if not isinstance(raw, list):
            raise ValueError("--checkpoints must contain a JSON array or {checkpoints: [...]} object")
        items.extend(raw)
    for step, tokens, url in inline:
        items.append({"step": int(step), "tokens_seen": int(tokens), "url": url})

    result = [_checkpoint(item) for item in items]
    result.sort(key=lambda item: item["step"])
    steps = [item["step"] for item in result]
    if len(steps) != len(set(steps)):
        raise ValueError("Duplicate checkpoint step")
    return result


def read_report_pairs(pairs: list[str]) -> dict[int, dict[str, Any]]:
    """Merge complementary run_eval and benchmark outputs by step."""
    merged: dict[int, dict[str, Any]] = {}
    for pair in pairs:
        if "=" not in pair:
            raise ValueError("--report must have format STEP=FILE")
        step_text, file_text = pair.split("=", 1)
        step = _positive_int(int(step_text), "report step")
        path = Path(file_text)
        document = json.loads(path.read_text(encoding="utf-8"))
        if not isinstance(document, dict):
            raise ValueError(f"Report must be a JSON object: {path}")
        if document.get("publishable") is False or document.get("status") == "partial":
            raise ValueError(f"Incomplete report cannot be published: {path}")
        report = merged.setdefault(step, {})
        for key in ("validation", "benchmarks"):
            if key not in document:
                continue
            value = document[key]
            if key == "benchmarks":
                if not isinstance(value, dict):
                    raise ValueError("benchmarks must be an object")
                existing = report.setdefault("benchmarks", {})
                for bench in ("hellaswag", "lambada"):
                    if bench in value:
                        if bench in existing and existing[bench] != value[bench]:
                            raise ValueError(f"Conflicting {bench} reports for step {step}")
                        existing[bench] = value[bench]
            else:
                if key in report and report[key] != value:
                    raise ValueError(f"Conflicting validation reports for step {step}")
                report[key] = value
    return merged


def _row_from_report(
    checkpoint_id: str, label: str, report: Mapping[str, Any]
) -> dict[str, Any]:
    validation = report["validation"]
    benchmarks = report["benchmarks"]
    return make_row(
        checkpoint_id, label, validation,
        benchmarks["hellaswag"], benchmarks["lambada"],
    )


def build_manifest(
    checkpoints: list[dict[str, Any]],
    reports: Mapping[int, Mapping[str, Any]],
    baseline: Mapping[str, Any] | None = None,
    riddles: list[str] | None = None,
) -> dict[str, Any]:
    """Pure builder. Missing reports yield no row, never invented scores."""
    ordered = sorted((_checkpoint(c) for c in checkpoints), key=lambda c: c["step"])
    steps = [c["step"] for c in ordered]
    if len(steps) != len(set(steps)):
        raise ValueError("Duplicate checkpoint step")
    extra = set(reports) - set(steps)
    if extra:
        raise ValueError(f"Reports refer to missing checkpoint steps: {sorted(extra)}")
    rows = []
    for checkpoint in ordered:
        step = checkpoint["step"]
        if step in reports:
            row = _row_from_report(
                f"step_{step:08d}", f"PlaygroundLM Step {step:,}", reports[step]
            )
            row["step"] = step
            rows.append(row)
    if baseline is not None:
        if baseline.get("publishable") is False or baseline.get("status") == "partial":
            raise ValueError("Incomplete GPT-2 baseline report")
        rows.append(_row_from_report(
            "gpt2_124m", "OpenAI GPT-2 124M", baseline
        ))
    if riddles is None:
        riddles = []
    if not isinstance(riddles, list) or any(not isinstance(s, str) for s in riddles):
        raise ValueError("riddle_prompts must be a list of strings")
    return {
        "model": "PlaygroundLM",
        "checkpoints": ordered,
        "benchmarks": {"columns": list(COLUMNS), "rows": rows},
        "riddle_prompts": list(riddles),
    }


def load_gpt2_assets() -> tuple[bytes, bytes]:
    """Fetch cached, SHA-256-checked GPT-2 encoder and merges bytes.

    `get_tokenizer()` selects the official GPT-2 encoding and enforces
    PLM_CACHE_DIR offline settings. `read_file_cached` uses that same
    tiktoken cache and verifies the expected source hashes.
    """
    get_tokenizer()
    from tiktoken.load import read_file_cached

    assets = {}
    for url, expected_sha256 in ASSETS.items():
        data = read_file_cached(url, expected_hash=expected_sha256)
        if hashlib.sha256(data).hexdigest() != expected_sha256:
            raise ValueError(f"Invalid GPT-2 tokenizer asset: {url}")
        assets[Path(urlsplit(url).path).name] = data

    encoder = assets["encoder.json"]
    merges = assets["vocab.bpe"]
    vocab = json.loads(encoder.decode("utf-8"))
    if not isinstance(vocab, dict) or len(vocab) != VOCAB_SIZE or vocab.get("<|endoftext|>") != EOT_ID:
        raise ValueError("GPT-2 encoder.json does not match our tokenizer")
    if not merges.startswith(b"#version: 0.2"):
        raise ValueError("Unexpected GPT-2 BPE merges header")
    return encoder, merges


def _write_bytes_atomic(path: Path, data: bytes) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = None
    try:
        with tempfile.NamedTemporaryFile(dir=path.parent, prefix=f".{path.name}.",
                                         suffix=".tmp", delete=False) as handle:
            temporary = Path(handle.name)
            handle.write(data)
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(temporary, path)
    finally:
        if temporary is not None:
            temporary.unlink(missing_ok=True)


def read_riddles(path: Path | None, existing_manifest: Path) -> list[str]:
    if path is None:
        if not existing_manifest.exists():
            return []
        document = json.loads(existing_manifest.read_text(encoding="utf-8"))
        return document.get("riddle_prompts", [])
    document = json.loads(path.read_text(encoding="utf-8"))
    return document.get("riddle_prompts", []) if isinstance(document, dict) else document


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--checkpoints", type=Path, help="JSON list of exported checkpoint records")
    parser.add_argument("--checkpoint", nargs=3, metavar=("STEP", "TOKENS_SEEN", "INT8_URL"),
                        action="append", default=[])
    parser.add_argument("--report", action="append", default=[], metavar="STEP=FILE",
                        help="Full eval report; repeat to join validation and benchmark reports")
    parser.add_argument("--baseline", type=Path, help="Full GPT-2 baseline_gpt2.py report")
    parser.add_argument("--riddles", type=Path, help="Optional JSON list or {riddle_prompts: [...]} file")
    parser.add_argument("--output", type=Path, default=Path("demo/manifest.json"))
    args = parser.parse_args()

    try:
        checkpoints = read_checkpoints(args.checkpoints, args.checkpoint)
        reports = read_report_pairs(args.report)
        baseline = None
        if args.baseline is not None:
            baseline = json.loads(args.baseline.read_text(encoding="utf-8"))
        riddles = read_riddles(args.riddles, args.output)
        manifest = build_manifest(checkpoints, reports, baseline, riddles)
        encoder, merges = load_gpt2_assets()
    except (ValueError, KeyError, TypeError, OSError, json.JSONDecodeError) as exc:
        parser.error(str(exc))

    output_dir = args.output.parent
    _write_bytes_atomic(output_dir / "encoder.json", encoder)
    _write_bytes_atomic(output_dir / "merges.txt", merges)
    write_json_atomic(args.output, manifest)
    print(json.dumps({
        "manifest": str(args.output),
        "tokenizer_assets": [str(output_dir / "encoder.json"), str(output_dir / "merges.txt")],
        "checkpoints": len(manifest["checkpoints"]),
        "benchmark_rows": len(manifest["benchmarks"]["rows"]),
    }, indent=2))


if __name__ == "__main__":
    main()
