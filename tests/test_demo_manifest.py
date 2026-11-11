"""Schema check for demo/manifest.json so training exports can't break the demo page."""
import json
from pathlib import Path

MANIFEST = Path(__file__).resolve().parents[1] / "demo" / "manifest.json"
FACT_SHEET = Path(__file__).resolve().parents[1] / "demo" / "facts" / "fact_sheet.jsonl"

V05_ONNX = (
    "https://huggingface.co/Gteam04/playground-lm-onnx/resolve/"
    "27778a045bee172529170330ada959131037da64/step_00031140.q8.onnx"
)
V06_ONNX = (
    "https://huggingface.co/Gteam04/playground-lm-onnx/resolve/"
    "d9e1629d821e7a9550fac9c54c56b132cd9327b5/live/v0.6_g2_31505/step_00031505_g2.q8.onnx"
)


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


def test_v05_chat_rollback_weights_unchanged():
    m = json.loads(MANIFEST.read_text())
    v05 = next(c for c in m["checkpoints"] if c["step"] == 31140)
    assert v05["label"] == "v0.5 Chat (Stage E)"
    assert v05["tokens_seen"] == 3058311168
    assert v05["quant"] == "q8"
    assert v05["bytes"] == 179148599
    assert v05["onnx_url"] == V05_ONNX
    assert v05["sha256"] == "0fb5504d0e98e4eeb2032a3c450ffd2b0d996bb11c8d27912be1d79b821fdee8"
    assert v05.get("fact_lookup") is not True


def test_live_checkpoint_v06_chat_with_fact_lookup():
    m = json.loads(MANIFEST.read_text())
    live = next(c for c in m["checkpoints"] if c["step"] == 31505)
    assert live["label"] == "v0.6 Chat (Stage G)"
    assert live["tokens_seen"] == 3073261568
    assert live["quant"] == "q8"
    assert live["bytes"] == 179148599
    assert live["onnx_url"] == V06_ONNX
    assert live["sha256"] == "f7a76210291f00d9eaca0fe12f4c74fdc3be6ad71475af447777c9a3fd5792de"
    assert live["fact_lookup"] is True
    assert live["fact_sheet_sha256"] == "f6ec9c14c29d55dc50cbd4c855a6de9e9654b65c765d1449a02d039e24c2ed33"
    assert "preview_only" not in live
    assert m["model_hosting"]["live_step"] == 31505
    assert m["model_hosting"]["revision"] == "d9e1629d821e7a9550fac9c54c56b132cd9327b5"
    assert m["model_hosting"]["rollback"]["step"] == 31140
    assert m["model_hosting"]["rollback"]["onnx_url"] == V05_ONNX
    # Default slider pick is the last checkpoint.
    assert m["checkpoints"][-1]["step"] == 31505
    assert FACT_SHEET.is_file()
    import hashlib
    sha = hashlib.sha256(FACT_SHEET.read_bytes()).hexdigest()
    assert sha == "f6ec9c14c29d55dc50cbd4c855a6de9e9654b65c765d1449a02d039e24c2ed33"
