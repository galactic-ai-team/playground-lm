"""Offline regression tests for GPT-6 Stage E behavior evaluation lane."""
import json
from pathlib import Path
from unittest.mock import patch
import sys
import types

import pytest

from evals.run_behavior_eval import (
    DEFAULT_EXPECT, DEFAULT_PROMPTS, build_prompt, evaluate, generate,
    has_long_repetition, load_suite, score,
)


def _fixture(tmp_path):
    prompts = tmp_path / "prompts.json"
    expects = tmp_path / "expects.json"
    prompts.write_text(json.dumps({
        "seed": 7, "frozen_at_et": "frozen", "prompts": [
            {"id": "one", "category": "format", "user": "Say OK"},
        ],
    }))
    expects.write_text(json.dumps({
        "seed": 7, "frozen_at_et": "frozen", "expect_by_id": {
            "one": {"type": "exact_or_startswith", "value": "OK"},
        },
    }))
    return prompts, expects


def test_real_frozen_fixtures_validated():
    suite = load_suite(DEFAULT_PROMPTS, DEFAULT_EXPECT)
    assert len(suite["prompts"]) >= 10
    assert len(suite["prompts_sha256"]) == 64
    assert len(suite["expect_sha256"]) == 64


def test_scorer_boundaries_and_case():
    assert score("OK!", {"type": "exact_or_startswith", "value": "OK"})
    assert not score("OKay", {"type": "exact_or_startswith", "value": "OK"})
    assert not score("YESterday", {"type": "exact_or_startswith", "value": "YES"})
    assert score("The answer is 45.", {"type": "contains", "value": "45"})
    assert not score("The answer is 145.", {"type": "contains", "value": "45"})
    assert score("Blue", {"type": "contains_any", "values": ["blue"]})
    assert not score("Nobody", {"type": "contains_any", "values": ["No"]})
    assert not score("  ", {"type": "nonempty"})


def test_prompt_message_order_enforced():
    item = {"messages": [
        {"role": "user", "content": "Hi"},
        {"role": "assistant", "content": "Hello"},
        {"role": "user", "content": "Now?"},
    ]}
    assert build_prompt(item) == "User: Hi\nAssistant: Hello\nUser: Now?\nAssistant:"
    with pytest.raises(ValueError, match="Final conversation"):
        build_prompt({"messages": item["messages"][:-1]})
    with pytest.raises(ValueError, match="alternate"):
        build_prompt({"messages": [item["messages"][0], item["messages"][0]]})


def test_duplicate_and_extra_expectations_rejected(tmp_path):
    prompts, expects = _fixture(tmp_path)
    document = json.loads(prompts.read_text())
    document["prompts"].append(document["prompts"][0])
    prompts.write_text(json.dumps(document))
    with pytest.raises(ValueError, match="duplicate"):
        load_suite(prompts, expects)
    document["prompts"].pop()
    prompts.write_text(json.dumps(document))
    extras = json.loads(expects.read_text())
    extras["expect_by_id"]["extra"] = {"type": "nonempty"}
    expects.write_text(json.dumps(extras))
    with pytest.raises(ValueError, match="Unexpected expectation"):
        load_suite(prompts, expects)


def test_prompt_leak_rejected(tmp_path):
    prompts, expects = _fixture(tmp_path)
    data = json.loads(prompts.read_text())
    data["prompts"][0]["expect"] = {"type": "nonempty"}
    prompts.write_text(json.dumps(data))
    with pytest.raises(ValueError, match="leaked"):
        load_suite(prompts, expects)


def test_frozen_checksum_rejects_modified_canonical_file(tmp_path, monkeypatch):
    # Simulate a canonical fixture path relocated into an isolated workspace.
    import evals.run_behavior_eval as module
    modified = tmp_path / "frozen.json"
    modified.write_text(DEFAULT_PROMPTS.read_text() + " ")
    monkeypatch.setattr(module, "DEFAULT_PROMPTS", modified)
    with pytest.raises(ValueError, match="Frozen evaluation file modified"):
        module.load_suite(modified, DEFAULT_EXPECT)


def test_generation_uses_shared_greedy_and_real_eot_api():
    class Tokenizer:
        eot_token_id = 50256
        def encode(self, text):
            return [1, 2]
        def decode(self, ids):
            return "OK\nUser: spurious continuation"
    class Model:
        context_len = 16
    fake = lambda *args, **kwargs: [3, 4]
    # Development ZIP lacks the optional tiktoken dependency; generation is mocked.
    with patch.dict(sys.modules, {"tiktoken": types.ModuleType("tiktoken")}), patch(
        "evals.run_eval.generate_tokens", side_effect=fake
    ) as gen:
        result, ids = generate(Model(), Tokenizer(), "User: hi\nAssistant:")
    assert (result, ids) == ("OK", [3, 4])
    assert gen.call_args.kwargs["eot_id"] == 50256
    assert gen.call_args.kwargs["temperature"] == 0.0
    assert gen.call_args.kwargs["context_len"] == 16


def test_repetition_flags():
    assert has_long_repetition([5] * 8)
    assert not has_long_repetition([5] * 7)
    assert not has_long_repetition([1, 1, 2, 2] * 4)


def test_evaluate_counts_categories_without_mutating_suite(monkeypatch):
    import evals.run_behavior_eval as module
    suite = {
        "seed": 7, "prompts_sha256": "p", "expect_sha256": "e",
        "prompts": [
            {"id": "a", "category": "format", "user": "OK", "expect": {"type": "nonempty"}},
            {"id": "b", "category": "format", "user": "YES", "expect": {"type": "nonempty"}},
        ],
    }
    monkeypatch.setattr(module, "generate", lambda *args, **kwargs: ("hi", [1] * 8))
    report = evaluate(object(), object(), suite)
    assert report["n"] == 2 and report["passed"] == 2
    assert report["repetition_flags"] == 2
    assert report["by_category"]["format"] == {"n": 2, "passed": 2}
    assert report["suite_type"] == "public_frozen_development_only"
    assert "blind" in report["limitations"]
