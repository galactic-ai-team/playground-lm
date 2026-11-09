"""Stage E round 3: replay mixing and the replay-ratio sweep (Claude). Synthetic data only."""

import numpy as np
import pytest

from playground_lm.train.replay_mix import ReplayMixer, build_replay_mixed_loader, check_exact, replay_count
from playground_lm.train.sweep_config import (
    SEED_SHA256,
    build_sweep_round3,
    expected_lr,
    make_run,
    run_env,
)

TOKENS = 640_047  # round-1 corpus size, as an example; the launcher passes the real count


# ---- mixer ------------------------------------------------------------------------

def rows(tag):
    """Endless one-row (x, y) batches whose values say where they came from."""
    while True:
        yield np.full((1, 4), tag), np.full((1, 4), tag)


@pytest.mark.parametrize("ratio, per_step", [(0.25, 3), (0.5, 6)])
def test_every_optimizer_step_gets_exactly_ratio_rows(ratio, per_step):
    mixer = ReplayMixer(rows(0), rows(1), ratio, batch_size=6, rows_per_step=12)  # 6 per micro x accum 2
    for _ in range(20):  # 20 optimizer steps on one rank
        a, b = next(mixer), next(mixer)
        assert int(a[0][:, 0].sum() + b[0][:, 0].sum()) == per_step
    assert mixer.replay_rows_served == 20 * per_step


def test_micro_batches_get_the_nearest_whole_numbers():
    mixer = ReplayMixer(rows(0), rows(1), 0.25, batch_size=6)
    counts = [int(next(mixer)[0][:, 0].sum()) for _ in range(8)]
    assert set(counts) <= {1, 2} and sum(counts) == 12  # 48 rows -> 12 replay


def test_mixer_is_deterministic_and_keeps_x_y_paired():
    def run():
        m = ReplayMixer(rows(0), rows(1), 0.5, batch_size=6)
        return [next(m) for _ in range(3)]
    for (xa, ya), (xb, yb) in zip(run(), run()):
        assert np.array_equal(xa, xb) and np.array_equal(xa, ya)  # each row's y came with its x
        assert xa.shape == (6, 4)


def test_ratio_validation():
    with pytest.raises(ValueError):
        ReplayMixer(rows(0), rows(1), 1.0, 6)
    with pytest.raises(ValueError):
        ReplayMixer(rows(0), rows(1), -0.1, 6)
    with pytest.raises(ValueError, match="whole number"):
        ReplayMixer(rows(0), rows(1), 0.3, 6, rows_per_step=12)
    assert check_exact(0.25, 12) == 3 and replay_count(4, 0.25) == 1 and replay_count(3, 0.25) == 0


def _pack(n, eot=50256):
    toks, masks = [], []
    for i in range(n):
        p, c = 3 + i % 4, 2 + i % 5
        toks += [100 + j for j in range(p)] + [500 + j for j in range(c)] + [eot]
        masks += [0] * p + [1] * (c + 1)
    return np.array(toks, dtype=np.uint16), np.array(masks, dtype=np.uint8)


def test_real_loaders_replay_rows_fully_supervised_sft_rows_masked(tmp_path):
    from playground_lm.model import IGNORE_INDEX

    t, m = _pack(40)
    t.tofile(tmp_path / "train.bin")
    m.tofile(tmp_path / "train.masks.bin")
    np.arange(1000, 3000, dtype=np.uint16).tofile(tmp_path / "replay.bin")  # stand-in pretraining tokens
    loader = build_replay_mixed_loader(tmp_path / "train.bin", tmp_path / "train.masks.bin", tmp_path / "replay.bin",
                                       ratio=0.5, context_len=16, batch_size=6, rows_per_step=12)
    x, y = (np.asarray(v) for v in next(loader))
    assert x.shape == y.shape == (6, 16)
    replay = x[:, 0] >= 1000
    assert replay.sum() == 3
    assert np.all(y[replay] != IGNORE_INDEX) and np.array_equal(y[replay][:, :-1], x[replay][:, 1:])
    assert np.all((y[~replay] == IGNORE_INDEX).sum(axis=1) > 0)  # SFT rows keep their prompt masks


def test_ratio_zero_is_the_plain_sft_loader(tmp_path):
    t, m = _pack(20)
    t.tofile(tmp_path / "train.bin")
    m.tofile(tmp_path / "train.masks.bin")
    loader = build_replay_mixed_loader(tmp_path / "train.bin", tmp_path / "train.masks.bin", tmp_path / "none.bin",
                                       ratio=0.0, context_len=16, batch_size=4)
    assert np.asarray(next(loader)[0]).shape == (4, 16)  # replay file never opened


# ---- sweep ------------------------------------------------------------------------

def test_round3_primary_grid():
    runs = build_sweep_round3(TOKENS)
    assert [(r.replay_ratio, r.peak_lr, r.epochs) for r in runs] == [(0.25, 1e-5, 1.0), (0.5, 1e-5, 1.0)]
    assert [r.run_id for r in runs] == ["stage-e-r3-rr25-ep1-lr1e-5", "stage-e-r3-rr50-ep1-lr1e-5"]
    # One SFT epoch: 18 SFT rows per update at 0.25, 12 at 0.5.
    assert [r.updates for r in runs] == [35, 53]
    assert all(r.schedule_start == 31101 for r in runs)


def test_round3_optional_cell_and_unique_ids():
    runs = build_sweep_round3(TOKENS, include_optional=True)
    assert len(runs) == 3 and len({r.run_id for r in runs}) == 3
    assert runs[-1].run_id == "stage-e-r3-rr50-ep1-lr3e-5" and runs[-1].updates == 53


@pytest.mark.parametrize("lr", [1e-5, 3e-5])
def test_expected_lr_for_round3_runs(lr):
    run = make_run(TOKENS, 1, lr, prefix="stage-e-r3", replay_ratio=0.5)
    assert run.warmup == 13
    assert expected_lr(run, 31102) == pytest.approx(lr / 13)
    assert expected_lr(run, 31101 + 13) == pytest.approx(lr)
    assert expected_lr(run, run.max_steps) == pytest.approx(0.1 * lr)
    lrs = [expected_lr(run, s) for s in range(31101 + 13, run.max_steps + 1)]
    assert all(a >= b for a, b in zip(lrs, lrs[1:]))


def test_run_env_carries_the_replay_ratio():
    env = run_env(build_sweep_round3(TOKENS)[0])
    assert env["PLM_REPLAY_RATIO"] == "0.25" and env["PLM_LR"] == "1e-05"
    assert env["PLM_MAX_STEPS"] == str(31101 + 35) and env["PLM_RUN_ID"] == "stage-e-r3-rr25-ep1-lr1e-5"
    assert env["PLM_RESET_OPTIMIZER"] == "1" and env["PLM_SCHEDULE_START"] == "31101"
    assert "PLM_REPLAY_RATIO" not in run_env(make_run(TOKENS, 1, 1e-5))  # round-2 runs unchanged


def test_replay_ratio_must_fit_the_batch():
    with pytest.raises(ValueError, match="whole number"):
        make_run(TOKENS, 1, 1e-5, replay_ratio=0.3)  # 7.2 of 24 rows
    with pytest.raises(ValueError):
        make_run(TOKENS, 1, 1e-5, replay_ratio=1.0)


def test_seed_is_still_stage_c_31101():
    assert SEED_SHA256.startswith("4604d3a1") and SEED_SHA256.endswith("76275")
