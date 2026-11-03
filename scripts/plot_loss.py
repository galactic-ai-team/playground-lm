#!/usr/bin/env python3
"""Plot training/validation loss curves and export downsampled JSON for the web demo.

Usage:
    python scripts/plot_loss.py runs/*/metrics.jsonl --out-dir reports/
"""

from __future__ import annotations

import argparse
import json
import math
import sys
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Sequence, Tuple

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402


def load_and_merge_metrics(
    paths: Sequence[Path | str],
) -> Tuple[List[Dict[str, Any]], int]:
    """Read one or more metrics.jsonl files, deduplicate by step (keeping last), and sort by step.

    Returns:
        (sorted_rows, skipped_warning_count)
    """
    by_step: Dict[int, Dict[str, Any]] = {}
    warnings_count = 0

    for raw_path in paths:
        path = Path(raw_path)
        if not path.is_file():
            warnings_count += 1
            print(f"WARNING: metrics file not found: {path}", file=sys.stderr)
            continue

        with path.open("r", encoding="utf-8", errors="replace") as handle:
            for line_no, raw_line in enumerate(handle, start=1):
                stripped = raw_line.strip()
                if not stripped:
                    continue
                try:
                    record = json.loads(stripped)
                except json.JSONDecodeError:
                    warnings_count += 1
                    continue

                if not isinstance(record, dict):
                    warnings_count += 1
                    continue

                step_val = record.get("step")
                loss_val = record.get("loss")
                if (
                    step_val is None
                    or isinstance(step_val, bool)
                    or not isinstance(step_val, (int, float))
                ):
                    warnings_count += 1
                    continue
                if (
                    loss_val is None
                    or isinstance(loss_val, bool)
                    or not isinstance(loss_val, (int, float))
                    or not math.isfinite(float(loss_val))
                ):
                    warnings_count += 1
                    continue

                step_int = int(step_val)
                normalized: Dict[str, Any] = dict(record)
                normalized["step"] = step_int
                normalized["loss"] = float(loss_val)
                normalized["tokens_seen"] = int(record.get("tokens_seen", 0) or 0)

                if "val_loss" in record and record["val_loss"] is not None:
                    try:
                        val_f = float(record["val_loss"])
                        if math.isfinite(val_f):
                            normalized["val_loss"] = val_f
                        else:
                            normalized.pop("val_loss", None)
                    except (TypeError, ValueError):
                        normalized.pop("val_loss", None)
                else:
                    normalized.pop("val_loss", None)

                # Keep the last occurrence across files and lines (handles resume overlap)
                by_step[step_int] = normalized

    if warnings_count > 0:
        print(
            f"WARNING: skipped {warnings_count} malformed or unreadable metric line(s).",
            file=sys.stderr,
        )

    sorted_rows = [by_step[k] for k in sorted(by_step.keys())]
    return sorted_rows, warnings_count


def compute_ema(values: Sequence[float], smooth: float = 0.9) -> List[float]:
    """Compute exponential moving average with weight `smooth` in [0.0, 1.0)."""
    if not values:
        return []
    weight = max(0.0, min(float(smooth), 0.9999))
    ema_vals: List[float] = []
    running = values[0]
    for v in values:
        running = weight * running + (1.0 - weight) * v
        ema_vals.append(running)
    return ema_vals


def compute_summary(rows: Sequence[Dict[str, Any]]) -> Dict[str, Any]:
    """Build the summary dictionary from sorted deduplicated metric rows."""
    if not rows:
        return {
            "final_step": None,
            "final_train_loss": None,
            "best_val_loss": None,
            "best_val_step": None,
            "tokens_seen": 0,
        }

    last = rows[-1]
    best_val_loss: Optional[float] = None
    best_val_step: Optional[int] = None

    for r in rows:
        if "val_loss" in r and r["val_loss"] is not None:
            v = float(r["val_loss"])
            if best_val_loss is None or v < best_val_loss:
                best_val_loss = v
                best_val_step = int(r["step"])

    return {
        "final_step": int(last["step"]),
        "final_train_loss": float(last["loss"]),
        "best_val_loss": best_val_loss,
        "best_val_step": best_val_step,
        "tokens_seen": int(last.get("tokens_seen", 0)),
    }


def downsample_points(
    rows: Sequence[Dict[str, Any]], max_points: int = 500
) -> List[Dict[str, Any]]:
    """Downsample sorted metric rows to at most `max_points` while always preserving:
    - every row containing `val_loss`
    - the final row (`rows[-1]`)
    """
    n = len(rows)
    if n == 0:
        return []

    def _to_point(r: Dict[str, Any]) -> Dict[str, Any]:
        pt: Dict[str, Any] = {
            "step": int(r["step"]),
            "tokens_seen": int(r.get("tokens_seen", 0)),
            "loss": float(r["loss"]),
        }
        if "val_loss" in r and r["val_loss"] is not None:
            pt["val_loss"] = float(r["val_loss"])
        return pt

    if max_points <= 0:
        max_points = 1

    if n <= max_points:
        return [_to_point(r) for r in rows]

    mandatory_indices = {
        idx
        for idx, r in enumerate(rows)
        if ("val_loss" in r and r["val_loss"] is not None)
    }
    mandatory_indices.add(n - 1)

    # Always include the first row if budget permits
    if len(mandatory_indices) < max_points:
        mandatory_indices.add(0)

    remaining_budget = max_points - len(mandatory_indices)
    if remaining_budget > 0:
        candidate_indices = [i for i in range(n) if i not in mandatory_indices]
        if len(candidate_indices) <= remaining_budget:
            mandatory_indices.update(candidate_indices)
        else:
            # Evenly spaced selection across candidate_indices
            for k in range(remaining_budget):
                pos = int(round(k * (len(candidate_indices) - 1) / max(1, remaining_budget - 1)))
                mandatory_indices.add(candidate_indices[pos])

    selected_sorted = sorted(mandatory_indices)
    return [_to_point(rows[i]) for i in selected_sorted]


def plot_loss_figure(
    rows: Sequence[Dict[str, Any]],
    out_png: Path,
    smooth: float = 0.9,
    x_axis: str = "tokens",
    gpt2_baseline: Optional[float] = None,
) -> None:
    """Render loss.png with raw train loss (faint), EMA train loss, val_loss, and optional GPT-2 baseline."""
    fig, ax = plt.subplots(figsize=(9.5, 5.5), dpi=150)

    if rows:
        if x_axis == "step":
            xs = [float(r["step"]) for r in rows]
            x_label = "Training Step"
        else:
            xs = [float(r.get("tokens_seen", 0)) / 1e9 for r in rows]
            x_label = "Tokens Seen (Billions)"

        losses = [float(r["loss"]) for r in rows]
        ema_losses = compute_ema(losses, smooth=smooth)

        ax.plot(
            xs,
            losses,
            color="#60a5fa",
            alpha=0.25,
            linewidth=1.0,
            label="Train Loss (raw)",
        )
        ax.plot(
            xs,
            ema_losses,
            color="#2563eb",
            linewidth=2.0,
            label=f"Train Loss (EMA {smooth:g})",
        )

        val_rows = [r for r in rows if "val_loss" in r and r["val_loss"] is not None]
        if val_rows:
            val_xs = (
                [float(r["step"]) for r in val_rows]
                if x_axis == "step"
                else [float(r.get("tokens_seen", 0)) / 1e9 for r in val_rows]
            )
            val_ys = [float(r["val_loss"]) for r in val_rows]
            ax.plot(
                val_xs,
                val_ys,
                color="#f97316",
                marker="o",
                markersize=4.5,
                linewidth=1.8,
                label="Validation Loss",
            )
    else:
        x_label = "Training Step" if x_axis == "step" else "Tokens Seen (Billions)"

    if gpt2_baseline is not None and math.isfinite(float(gpt2_baseline)):
        ax.axhline(
            float(gpt2_baseline),
            color="#dc2626",
            linestyle="--",
            linewidth=1.5,
            label=f"GPT-2 124M ({float(gpt2_baseline):.3f})",
        )

    ax.set_title("PlaygroundLM (124M) Training & Validation Loss")
    ax.set_xlabel(x_label)
    ax.set_ylabel("Cross-Entropy Loss")
    ax.grid(True, linestyle=":", alpha=0.5)
    handles, labels = ax.get_legend_handles_labels()
    if handles:
        ax.legend(loc="upper right", frameon=True)

    out_png.parent.mkdir(parents=True, exist_ok=True)
    fig.tight_layout()
    fig.savefig(out_png)
    plt.close(fig)


def generate_loss_artifacts(
    metric_files: Sequence[Path | str],
    out_dir: Path | str,
    smooth: float = 0.9,
    x_axis: str = "tokens",
    max_points: int = 500,
    gpt2_baseline: Optional[float] = None,
) -> Dict[str, Any]:
    """End-to-end helper: parse, merge, write loss.png and loss_curve.json, and return the JSON payload."""
    out_path = Path(out_dir)
    out_path.mkdir(parents=True, exist_ok=True)

    rows, _ = load_and_merge_metrics(metric_files)
    summary = compute_summary(rows)
    points = downsample_points(rows, max_points=max_points)

    payload = {
        "points": points,
        "summary": summary,
    }

    json_path = out_path / "loss_curve.json"
    with json_path.open("w", encoding="utf-8") as handle:
        json.dump(payload, handle, indent=2)
        handle.write("\n")

    png_path = out_path / "loss.png"
    plot_loss_figure(
        rows=rows,
        out_png=png_path,
        smooth=smooth,
        x_axis=x_axis,
        gpt2_baseline=gpt2_baseline,
    )

    return payload


def parse_args(argv: Optional[Sequence[str]] = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Merge PlaygroundLM metrics.jsonl logs and produce loss.png + loss_curve.json."
    )
    parser.add_argument(
        "inputs",
        nargs="+",
        help="One or more metrics.jsonl files.",
    )
    parser.add_argument(
        "--out-dir",
        type=Path,
        default=Path("reports"),
        help="Output directory for loss.png and loss_curve.json (default: reports/).",
    )
    parser.add_argument(
        "--smooth",
        type=float,
        default=0.9,
        help="EMA smoothing factor in [0, 1) for training loss (default: 0.9).",
    )
    parser.add_argument(
        "--x",
        choices=["tokens", "step"],
        default="tokens",
        help="X-axis unit for loss.png: 'tokens' (in billions) or 'step' (default: tokens).",
    )
    parser.add_argument(
        "--max-points",
        type=int,
        default=500,
        help="Maximum points in loss_curve.json (always preserves val_loss rows and last row; default: 500).",
    )
    parser.add_argument(
        "--gpt2-baseline",
        type=float,
        default=None,
        help="Optional GPT-2 124M baseline loss to plot as a horizontal dashed line.",
    )
    return parser.parse_args(argv)


def main(argv: Optional[Sequence[str]] = None) -> int:
    args = parse_args(argv)
    generate_loss_artifacts(
        metric_files=[Path(p) for p in args.inputs],
        out_dir=args.out_dir,
        smooth=args.smooth,
        x_axis=args.x,
        max_points=args.max_points,
        gpt2_baseline=args.gpt2_baseline,
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
