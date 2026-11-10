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
    for name in ("index.html", "app.js", "style.css", "favicon.svg"):
        assert (d / name).is_file()
    svg = (d / "favicon.svg").read_text()
    assert "<svg" in svg and "PL" in svg


def test_live_checkpoint_v05_chat_label_preserves_weights():
    m = json.loads(MANIFEST.read_text())
    live = next(c for c in m["checkpoints"] if c["step"] == 31140)
    assert live["label"] == "v0.5 Chat (Stage E)"
    assert live["tokens_seen"] == 3058311168
    assert live["quant"] == "q8"
    assert live["bytes"] == 179148599
    assert live["onnx_url"] == (
        "https://huggingface.co/Gteam04/playground-lm-onnx/resolve/"
        "27778a045bee172529170330ada959131037da64/step_00031140.q8.onnx"
    )
    assert live["sha256"] == "0fb5504d0e98e4eeb2032a3c450ffd2b0d996bb11c8d27912be1d79b821fdee8"
    assert m["model_hosting"]["live_step"] == 31140
    assert m["model_hosting"]["rollback"]["step"] == 31104
