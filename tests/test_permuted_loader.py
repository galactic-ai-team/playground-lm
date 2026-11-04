"""Without-replacement Stage C sampling (Lumi, for Claude's 63%-coverage note)."""

import numpy as np
import pytest

torch = pytest.importorskip("torch")

from playground_lm.train.data import build_permuted_dataloader  # noqa: E402

CTX = 4


def _file(tmp_path, n_tokens):
    p = tmp_path / "train.bin"
    np.arange(n_tokens, dtype=np.uint16).tofile(p)
    return p


def _starts(batches):
    return [int(x[0]) for xb, _ in batches for x in xb]


@pytest.mark.parametrize("n_tokens", [4 * 97 + 1, 4 * 100 + 3, 4 * 128 + 1])
def test_one_pass_covers_every_window_once(tmp_path, n_tokens):
    path = _file(tmp_path, n_tokens)
    n = (n_tokens - 1) // CTX
    it = build_permuted_dataloader(path, CTX, batch_size=1)
    starts = _starts(next(it) for _ in range(n))
    assert sorted(starts) == [w * CTX for w in range(n)]


def test_targets_are_inputs_shifted(tmp_path):
    it = build_permuted_dataloader(_file(tmp_path, 401), CTX, batch_size=3)
    x, y = next(it)
    assert x.shape == y.shape == (3, CTX)
    assert torch.equal(y[:, :-1], x[:, 1:])
    assert torch.equal(y[:, -1], x[:, -1] + 1)


def test_ranks_split_the_stream_and_resume_is_exact(tmp_path):
    path = _file(tmp_path, 4 * 103 + 1)
    whole = _starts(next(build_permuted_dataloader(path, CTX, batch_size=4)) for _ in range(1))
    single = build_permuted_dataloader(path, CTX, batch_size=2)
    one = _starts([next(single), next(single), next(single), next(single)])
    # Two "ranks", batch 2 each, stride 4: interleave to the single stream.
    r0 = build_permuted_dataloader(path, CTX, batch_size=2, first_sample=0, sample_stride=4)
    r1 = build_permuted_dataloader(path, CTX, batch_size=2, first_sample=2, sample_stride=4)
    a0, a1 = _starts([next(r0), next(r0)]), _starts([next(r1), next(r1)])
    assert a0[:2] + a1[:2] + a0[2:] + a1[2:] == one
    assert one[:4] == whole
    # Resuming at sample 4 continues where the first two batches stopped.
    resumed = build_permuted_dataloader(path, CTX, batch_size=2, first_sample=4)
    assert _starts([next(resumed), next(resumed)]) == one[4:]


def test_seed_changes_order(tmp_path):
    path = _file(tmp_path, 4 * 101 + 1)
    a = _starts([next(build_permuted_dataloader(path, CTX, batch_size=8, seed=1))])
    b = _starts([next(build_permuted_dataloader(path, CTX, batch_size=8, seed=2))])
    assert a != b


@pytest.mark.parametrize("kwargs", [dict(first_sample=-1), dict(sample_stride=1), dict(batch_size=0)])
def test_rejects_bad_args(tmp_path, kwargs):
    args = dict(batch_size=2) | kwargs
    with pytest.raises(ValueError):
        next(build_permuted_dataloader(_file(tmp_path, 401), CTX, **args))
