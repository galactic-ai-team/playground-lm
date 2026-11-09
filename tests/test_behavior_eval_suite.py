import json
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
PROMPTS = ROOT / "evals" / "behavior" / "behavior_eval.prompts.frozen.json"
EXPECT = ROOT / "evals" / "behavior" / "behavior_eval.expect.frozen.json"


def test_prompts_and_expects_align():
    prompts = json.loads(PROMPTS.read_text())
    expect = json.loads(EXPECT.read_text())
    ids = [p["id"] for p in prompts["prompts"]]
    assert len(ids) == len(set(ids))
    assert set(ids) == set(expect["expect_by_id"])
    for item in prompts["prompts"]:
        assert "expect" not in item  # leakage guard: prompts file stays scorer-free
        assert "user" in item or "messages" in item
    for eid, scorer in expect["expect_by_id"].items():
        assert "type" in scorer


def test_builder_format_and_mask_helpers():
    # Import pure helpers without downloading Dolly.
    import importlib.util
    spec = importlib.util.spec_from_file_location(
        "build_stage_e_sft", ROOT / "data" / "build_stage_e_sft.py"
    )
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    text = mod.format_example("Hello", "World")
    assert text == "User: Hello\nAssistant: World"
