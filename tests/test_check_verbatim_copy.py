
"""Offline tests for the verbatim-copy audit."""

import json
import sys

import numpy as np
import pytest

torch = pytest.importorskip("torch")

from scripts import check_verbatim_copy as copycheck


def write_tokens(path, tokens):
    np.asarray(tokens, dtype="<u2").tofile(path)
    return path


def test_exact_50_token_match_is_flagged(tmp_path):
    sequence = list(range(100, 150))
    training = [5] * 20 + sequence + [6] * 20
    corpus = write_tokens(tmp_path / "train.bin", training)

    report = copycheck.audit_token_sequences(
        corpus,
        [sequence],
        threshold=50,
        prompt_ids=["copied"],
    )

    assert report["any_flagged"] is True
    assert report["flagged_generations"] == 1
    assert report["checked_generations"] == 1

    match = report["results"][0]
    assert match["flagged"] is True
    assert match["match_start_in_generation"] == 0
    assert match["match_start_in_training"] == 20


def test_sliding_window_match(tmp_path):
    sequence = list(range(100, 150))
    corpus = write_tokens(
        tmp_path / "train.bin",
        [999] * 7 + sequence + [999] * 7,
    )
    generated = [1, 2, 3] + sequence + [4, 5]

    report = copycheck.audit_token_sequences(
        corpus,
        [generated],
        threshold=50,
    )

    assert report["results"][0]["flagged"]
    assert (
        report["results"][0]["match_start_in_generation"]
        == 3
    )


def test_one_token_difference_not_flagged(tmp_path):
    sequence = list(range(100, 150))
    corpus = write_tokens(
        tmp_path / "train.bin",
        sequence,
    )

    changed = sequence.copy()
    changed[25] = 999

    report = copycheck.audit_token_sequences(
        corpus, [changed], threshold=50
    )

    assert report["flagged_generations"] == 0


def test_short_generation_is_not_checked(tmp_path):
    corpus = write_tokens(
        tmp_path / "train.bin",
        list(range(100)),
    )

    report = copycheck.audit_token_sequences(
        corpus,
        [list(range(49))],
        threshold=50,
    )

    assert report["checked_generations"] == 0
    assert report["short_generations"] == 1
    assert report["results"][0]["checked"] is False
    assert report["results"][0]["flagged"] is False


def test_unaligned_byte_match_is_ignored(tmp_path):
    # 258 = 0x0102, stored in little-endian as 02 01.
    # 513 = 0x0201, stored as 01 02.
    # A byte-offset match must not count as an aligned token match.
    corpus = write_tokens(
        tmp_path / "train.bin",
        [258] * 60,
    )
    generated = [513] * 50

    report = copycheck.audit_token_sequences(
        corpus, [generated], threshold=50
    )

    assert report["flagged_generations"] == 0


def test_duplicate_prompt_ids_rejected(tmp_path):
    corpus = write_tokens(
        tmp_path / "train.bin",
        [1] * 100,
    )

    with pytest.raises(ValueError, match="Duplicate"):
        copycheck.audit_token_sequences(
            corpus,
            [[1] * 50, [1] * 50],
            prompt_ids=["same", "same"],
        )


def test_runner_uses_real_copy_matcher(
    tmp_path, monkeypatch
):
    sequence = list(range(100, 150))
    corpus = write_tokens(
        tmp_path / "train.bin",
        [8] * 10 + sequence + [9] * 10,
    )

    prompts = tmp_path / "prompts.json"
    prompts.write_text(json.dumps({
        "held_out_probes": [
            {"id": "probe_1", "prompt": "Hi"}
        ]
    }))

    class MockModel:
        context_len = 128

    def fake_loader(backend, model_ref, **kwargs):
        assert backend == "playground"
        return MockModel()

    def fake_evaluate(
        model,
        prompts_file,
        **kwargs,
    ):
        assert kwargs["max_new_tokens"] == 60
        return (
            [{"id": "probe_1", "generation": "mocked"}],
            [sequence],
        )

    monkeypatch.setattr(
        copycheck, "load_model", fake_loader
    )
    monkeypatch.setattr(
        copycheck, "evaluate_prompts", fake_evaluate
    )
    monkeypatch.setattr(
        copycheck, "get_tokenizer", lambda: object()
    )

    report = copycheck.run_check(
        checkpoint="tiny.safetensors",
        train_bin=corpus,
        prompts=prompts,
        max_new_tokens=60,
        threshold=50,
    )

    assert report["flagged_generations"] == 1
    assert report["results"][0]["id"] == "probe_1"
    assert report["results"][0]["match_start_in_training"] == 10
    assert "generation" not in report["results"][0]
