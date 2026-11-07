"""Schema check for demo/manifest.json so training exports can't break the demo page."""
import json
from pathlib import Path

MANIFEST = Path(__file__).resolve().parents[1] / "demo" / "manifest.json"


def _checkpoint_url(c: dict):
    """Prefer onnx_url, then url, then path (matches demo/checkpoint_loader.js)."""
    for key in ("onnx_url", "url", "path"):
        if key in c:
            return c[key]
    return None


def test_manifest_schema():
    m = json.loads(MANIFEST.read_text())
    assert isinstance(m["checkpoints"], list)
    for c in m["checkpoints"]:
        assert isinstance(c["step"], int) and c["step"] >= 0
        assert isinstance(c["tokens_seen"], int) and c["tokens_seen"] >= 0
        # Metadata-only entries (onnx_url/url/path null or "") are allowed so the
        # demo slider can scrub training steps before ONNX weights are hosted.
        raw = _checkpoint_url(c)
        if raw is None or raw == "":
            continue
        assert isinstance(raw, str) and raw.strip() == raw and raw
    steps = [c["step"] for c in m["checkpoints"]]
    assert steps == sorted(steps), "checkpoints must be in training order"
    b = m["benchmarks"]
    assert isinstance(b["columns"], list) and b["columns"]
    for r in b["rows"]:
        assert isinstance(r["model"], str)
        assert set(r) - {"model"} <= set(b["columns"])
    assert all(isinstance(p, str) for p in m["riddle_prompts"])


def test_demo_files_exist():
    d = MANIFEST.parent
    for name in ("index.html", "app.js", "style.css"):
        assert (d / name).is_file()
