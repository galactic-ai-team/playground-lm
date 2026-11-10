"""Stage G (Claude): four-source mixer, sweep grid, answer-token weights, gate continuity.

Offline and synthetic: no weights, no private items.
"""

import math

import numpy as np
import pytest

from evals.regression_gate import STAGE_C_SEED_SHA256, validate_receipt, verdict
from playground_lm.model import IGNORE_INDEX
from playground_lm.train.replay_mix import MultiSourceMixer, build_stage_g_mixed_loader, step_weights
from playground_lm.train.sft_data import answer_weights, build_sft_dataloader
from playground_lm.train.sweep_config import (
    STAGE_G_BATCH,
    build_sweep_stage_g,
    expected_lr,
    run_env,
    source_passes,
)

TOKENS = {"scratch": 1_000_000, "grounded": 200_000, "chat": 640_047}  # example counts
E_SHA = "ce1fa2098236656d149437638601dfafa7744ab3f69ff52739be0c95199f818b"


# ---- shares -----------------------------------------------------------------------

@pytest.mark.parametrize("shares, per_rank", [((0.25, 0.15, 0.10, 0.50), [5, 3, 2, 10]),
                                             ((0.35, 0.20, 0.10, 0.35), [7, 4, 2, 7])])
def test_packet_shares_are_exact_whole_rows(shares, per_rank):
    assert step_weights(list(shares), 20) == per_rank          # per rank on 2 x T4 (micro 5 x accum 4)
    assert step_weights(list(shares), STAGE_G_BATCH) == [2 * w for w in per_rank]  # 1 GPU / whole update
    assert sum(per_rank) == 20


@pytest.mark.parametrize("bad", [[0.25, 0.15, 0.10, 0.49], [0.3, 0.3, 0.3, 0.3], [-0.1, 0.4, 0.2, 0.5],
                                 [0.25, 0.15, 0.10, 0.50]])
def test_bad_shares_raise(bad):
    rows = 24 if bad == [0.25, 0.15, 0.10, 0.50] else 20  # the draft shares aren't whole at 24 rows
    with pytest.raises(ValueError):
        step_weights(bad, rows)


def rows(tag, weighted=False):
    while True:
        x = np.full((1, 4), tag)
        yield (x, x, np.full((1, 4), 2.0, dtype=np.float32)) if weighted else (x, x)


def test_four_source_mixer_exact_per_step_and_deterministic():
    def mixer():
        return MultiSourceMixer([rows(0), rows(1), rows(2), rows(3)], [0.25, 0.15, 0.10, 0.50],
                                batch_size=5, rows_per_step=20)
    m = mixer()
    for _ in range(6):  # 6 optimizer steps of 4 micro-batches
        tags = np.concatenate([next(m)[0][:, 0] for _ in range(4)])
        assert [int((tags == k).sum()) for k in range(4)] == [5, 3, 2, 10]
    a, b = mixer(), mixer()
    assert all(np.array_equal(next(a)[0], next(b)[0]) for _ in range(8))


def test_mixer_fills_weight_one_for_unweighted_sources():
    m = MultiSourceMixer([rows(0, weighted=True), rows(1), rows(2), rows(3)], [0.25, 0.15, 0.10, 0.50],
                         batch_size=5, rows_per_step=20)
    x, y, w = next(m)
    assert w.shape == y.shape
    assert np.all(w[x[:, 0] == 0] == 2.0) and np.all(w[x[:, 0] != 0] == 1.0)


# ---- sweep --------------------------------------------------------------------------

def test_stage_g_grid():
    runs = build_sweep_stage_g(TOKENS, include_optional=True)
    assert [r.run_id for r in runs] == ["stage-g-e31140-rr50-ch10-gf15-ep2-lr1e-5",
                                        "stage-g-e31140-rr50-ch10-gf15-ep2-lr3e-5",
                                        "stage-g-e31140-rr35-ch10-gf20-ep2-lr1e-5"]
    assert len(build_sweep_stage_g(TOKENS)) == 2
    assert all(r.seed == "E31140" and r.schedule_start == 31140 and r.effective_batch == 40 for r in runs)
    # 2 scratchpad epochs at 10 scratch rows/update (G1/G2) and 14 (G3), 1024 tokens a row.
    assert runs[0].updates == math.ceil(2 * 1_000_000 / (10 * 1024)) == 196
    assert runs[2].updates == math.ceil(2 * 1_000_000 / (14 * 1024))


def test_stage_g_run_env():
    env = run_env(build_sweep_stage_g(TOKENS)[0])
    assert env["PLM_SEED_STEP"] == "31140" and env["PLM_SEED_SHA256"] == E_SHA
    assert env["PLM_RESET_OPTIMIZER"] == "1" and env["PLM_SCHEDULE_START"] == "31140"
    assert env["PLM_REPLAY_RATIO"] == "0.5" and env["PLM_CHAT_RATIO"] == "0.1"
    assert env["PLM_GROUNDED_RATIO"] == "0.15" and env["PLM_LR"] == "1e-05"
    assert env["PLM_EFFECTIVE_BATCH"] == "40" and env["PLM_BATCH_SIZE"] == "5"
    assert env["PLM_MAX_STEPS"] == str(31140 + 196)


def test_stage_g_expected_lr_and_source_passes():
    run = build_sweep_stage_g(TOKENS)[0]
    assert run.warmup == 20
    assert expected_lr(run, 31141) == pytest.approx(1e-5 / 20)
    assert expected_lr(run, 31160) == pytest.approx(1e-5)
    assert expected_lr(run, run.max_steps) == pytest.approx(1e-6)
    passes = source_passes(run, TOKENS)
    assert passes["scratch"] == pytest.approx(2.0, abs=0.01)
    assert passes["grounded"] == pytest.approx(196 * 6 * 1024 / 200_000, abs=0.01)


def test_stage_g_needs_scratch_tokens():
    with pytest.raises(ValueError):
        build_sweep_stage_g({"grounded": 1})


# ---- answer-token weights ---------------------------------------------------------------

MARK = (90, 91)  # stands in for the tokens of " Answer:"
EOT = 50256


def test_after_marker_weights_only_the_final_answer():
    # [prompt 10 11][work 20 21 22][marker 90 91][answer 33 34][EOT]
    x = np.array([[10, 11, 20, 21, 22, 90, 91, 33, 34]])
    y = np.array([[IGNORE_INDEX, 20, 21, 22, 90, 91, 33, 34, EOT]])
    w = answer_weights(x, y, "after_marker", 2.0, MARK, EOT)
    # targets predicting 33, 34 and the EOT are weighted; the worked steps and the marker are not
    assert w.tolist() == [[1, 1, 1, 1, 1, 1, 2, 2, 2]]


def test_completion_mode_weights_every_supervised_target():
    y = np.array([[IGNORE_INDEX, IGNORE_INDEX, 7, 8, EOT]])
    w = answer_weights(np.zeros_like(y), y, "completion", 2.0)
    assert w.tolist() == [[1, 1, 2, 2, 2]]


def _sft(tmp_path, name, base, n=30, answer=False):
    toks, masks = [], []
    for i in range(n):
        prompt = [base + j for j in range(3)]
        completion = [base + 50 + j for j in range(2)] + (list(MARK) + [base + 80] if answer else []) + [EOT]
        toks += prompt + completion
        masks += [0] * len(prompt) + [1] * len(completion)
    np.array(toks, dtype=np.uint16).tofile(tmp_path / f"{name}.bin")
    np.array(masks, dtype=np.uint8).tofile(tmp_path / f"{name}.masks.bin")
    return tmp_path / f"{name}.bin", tmp_path / f"{name}.masks.bin"


def test_weight_one_loader_is_exactly_the_current_loader(tmp_path):
    t, m = _sft(tmp_path, "s", 100, answer=True)
    plain = build_sft_dataloader(t, m, context_len=12, batch_size=3, seed=5)
    one = build_sft_dataloader(t, m, context_len=12, batch_size=3, seed=5, answer_weight=1.0,
                               answer_mode="after_marker", answer_marker=MARK)
    for _ in range(10):
        a, b = next(plain), next(one)
        assert len(b) == 2 and all(np.array_equal(np.asarray(p), np.asarray(q)) for p, q in zip(a, b))


def test_four_source_real_loaders(tmp_path):
    sb, sm = _sft(tmp_path, "scratch", 100, answer=True)
    gb_, gm = _sft(tmp_path, "grounded", 1000)
    cb, cm = _sft(tmp_path, "chat", 2000)
    np.arange(5000, 9000, dtype=np.uint16).tofile(tmp_path / "replay.bin")
    common = dict(grounded_ratio=0.15, chat_ratio=0.10, replay_ratio=0.50, context_len=12, batch_size=5,
                  rows_per_step=20)
    loader = build_stage_g_mixed_loader(sb, sm, gb_, gm, cb, cm, tmp_path / "replay.bin", **common)
    batches = [next(loader) for _ in range(4)]
    assert all(len(b) == 2 for b in batches)  # weight 1.0: plain (x, y)
    x = np.concatenate([np.asarray(b[0]) for b in batches])
    y = np.concatenate([np.asarray(b[1]) for b in batches])
    src = np.select([x[:, 0] >= 5000, x[:, 0] >= 2000, x[:, 0] >= 1000], [3, 2, 1], 0)
    assert [int((src == k).sum()) for k in range(4)] == [5, 3, 2, 10]
    assert np.all(y[src == 3] != IGNORE_INDEX)                # replay: full loss
    assert np.all((y[src < 3] == IGNORE_INDEX).sum(axis=1) > 0)  # SFT sources: prompts masked

    weighted = build_stage_g_mixed_loader(sb, sm, gb_, gm, cb, cm, tmp_path / "replay.bin",
                                          answer_weight=2.0, answer_marker=MARK, **common)
    xw, yw, w = (np.asarray(v) for v in next(weighted))
    src = np.select([xw[:, 0] >= 5000, xw[:, 0] >= 2000, xw[:, 0] >= 1000], [3, 2, 1], 0)
    assert np.all(w[src >= 2] == 1.0)  # chat and replay never weighted
    assert np.any(w[src <= 1] == 2.0)  # scratch answers / grounded replies are


# ---- gate continuity ----------------------------------------------------------------------

def test_verdict_still_fails_on_fineweb_rise_over_002():
    def m(loss, sha):
        return {"weights": "w", "weights_sha256": sha, "fineweb": {"loss": loss, "tokens_scored": 81920},
                "hellaswag": {"accuracy": 0.39, "examples": 200}, "lambada": {"last_word_accuracy": 0.26, "examples": 200}}
    r = {"schema_version": 1, "kind": "stage_e_regression", "baseline": "C",
         "eval": {"fineweb_val_sha256": "a", "val_max_batches": 20, "hellaswag_limit": 200, "lambada_limit": 200},
         "models": {"C": m(3.052, STAGE_C_SEED_SHA256), "G1": m(3.0721, "g" * 64), "G2": m(3.0719, "h" * 64)}}
    assert validate_receipt(r) == []
    assert verdict(r, "G1")["status"] == "FAIL"   # +0.0201
    assert verdict(r, "G2")["status"] == "PASS"   # +0.0199
