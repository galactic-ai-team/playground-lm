"""Write step_XXXXXXXX_model_config.json sidecars for checkpoints saved before
CheckpointManager wrote them (Claude).

Reads a run's manifest.json (local file, e.g. downloaded from the release)
and writes one sidecar per checkpoint entry into --out. Needs no weights, no
_state.pt and no torch. Then upload them next to the weights, e.g.:

    python scripts/backfill_model_configs.py --manifest manifest.json --out sidecars
    gh release upload v0.1-checkpoints sidecars/*_model_config.json \\
        --repo galactic-ai-team/playground-lm --clobber

(or put them in the run's checkpoints/ directory; upload_release() uploads any
sidecar that isn't in the release yet).
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))

from playground_lm.train.sidecar import model_config_document, write_model_config_sidecar  # noqa: E402


def backfill(manifest_path: Path, out_dir: Path, steps: set[int] | None = None) -> list[Path]:
    manifest = json.loads(Path(manifest_path).read_text(encoding="utf-8"))
    config = manifest.get("config") or {}
    if "model" not in config:
        raise ValueError(f"{manifest_path}: config has no 'model' section; nothing to describe")
    out_dir.mkdir(parents=True, exist_ok=True)
    written = []
    for entry in manifest["checkpoints"]:
        if steps is not None and entry["step"] not in steps:
            continue
        weights_name = Path(entry["path"]).name
        document = model_config_document(
            manifest["run_id"], config, entry["step"], entry["tokens_seen"], entry["wall_time_s"]
        )
        written.append(write_model_config_sidecar(out_dir / weights_name, document))
    return written


def main(argv: list[str] | None = None) -> None:
    p = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    p.add_argument("--manifest", required=True, type=Path)
    p.add_argument("--out", required=True, type=Path)
    p.add_argument("--steps", type=lambda s: {int(v) for v in s.split(",") if v},
                   help="only these steps, e.g. 10907,19366")
    args = p.parse_args(argv)
    for path in backfill(args.manifest, args.out, args.steps):
        print(f"[backfill] wrote {path}")


if __name__ == "__main__":
    main()