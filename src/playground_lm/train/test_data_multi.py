# src/playground_lm/train/test_data_multi.py
# build_dataloader with several token files (Claude), for Stage C's 8 data-prep outputs.
#
# Each test file holds a disjoint, strictly increasing range of token ids, so
# a window's file and contiguity can be read straight off its values.

from pathlib import Path

import numpy as np
import pytest

from playground_lm.train.data import build_dataloader

A0, B0 = 1000, 20000  # first token id in file A / file B


def _write(path: Path, first: int, n: int) -> Path:
    np.arange(first, first + n, dtype=np.uint16).tofile(path)
    return path


@pytest.fixture
def two_files(tmp_path):
    a = _write(tmp_path / "a.bin", A0, 300)  # 290 valid starts at context_len 10
    b = _write(tmp_path / "b.bin", B0, 100)  # 90 valid starts
    return a, b


def _rows(batch):
    x, y = batch
    return np.asarray(x), np.asarray(y)


def _file_of(row):
    return "a" if row[0] < B0 else "b"


def test_windows_never_cross_a_file_boundary(two_files):
    loader = build_dataloader(list(two_files), context_len=10, batch_size=16, shuffle=True, seed=1)
    for _ in range(200):
        x, y = _rows(next(loader))
        for xr, yr in zip(x, y):
            assert np.all(np.diff(xr) == 1)  # contiguous ids: one file, no join inside
            assert np.array_equal(yr, xr + 1)  # targets are the next tokens
            if _file_of(xr) == "a":
                assert A0 <= xr[0] and yr[-1] < A0 + 300
            else:
                assert B0 <= xr[0] and yr[-1] < B0 + 100


def test_files_are_sampled_in_proportion_to_their_windows(two_files):
    loader = build_dataloader(list(two_files), context_len=10, batch_size=100, shuffle=True, seed=7)
    picks = [_file_of(r) for _ in range(200) for r in _rows(next(loader))[0]]
    share_a = picks.count("a") / len(picks)
    assert share_a == pytest.approx(290 / 380, abs=0.02)  # 20,000 windows
    assert picks.count("b") > 0


def test_every_window_of_both_files_is_reachable(two_files):
    loader = build_dataloader(list(two_files), context_len=10, batch_size=200, shuffle=True, seed=3)
    seen = set()
    for _ in range(100):
        seen.update(int(r[0]) for r in _rows(next(loader))[0])
    expected = set(range(A0, A0 + 290)) | set(range(B0, B0 + 90))
    assert seen == expected  # every valid start, and nothing past a file's last one


def test_one_element_list_matches_a_single_path(two_files):
    a, _ = two_files
    single = build_dataloader(a, context_len=10, batch_size=4, shuffle=True, seed=5, seed_offset=2)
    listed = build_dataloader([a], context_len=10, batch_size=4, shuffle=True, seed=5, seed_offset=2)
    for _ in range(20):
        (xs, ys), (xl, yl) = _rows(next(single)), _rows(next(listed))
        assert np.array_equal(xs, xl) and np.array_equal(ys, yl)


def test_seed_and_offset_reproduce_and_differ(two_files):
    def first(seed_offset):
        loader = build_dataloader(list(two_files), 10, 8, shuffle=True, seed=9, seed_offset=seed_offset)
        return _rows(next(loader))[0]

    assert np.array_equal(first(0), first(0))
    assert not np.array_equal(first(0), first(1))


def test_sequential_eval_covers_each_file_once_then_stops(two_files):
    loader = build_dataloader(list(two_files), context_len=10, batch_size=7, shuffle=False, drop_last=False)
    batches = list(loader)  # finite without drop_last
    starts = [int(r[0]) for b in batches for r in _rows(b)[0]]
    expected = list(range(A0, A0 + 290, 10)) + list(range(B0, B0 + 90, 10))
    assert starts == expected  # file order, stride = context_len, no window across the join
    assert len(_rows(batches[-1])[0]) == (len(expected) % 7 or 7)  # final partial batch


def test_sequential_with_drop_last_wraps_and_keeps_full_batches(two_files):
    loader = build_dataloader(list(two_files), context_len=10, batch_size=7, shuffle=False, drop_last=True)
    n_windows = 29 + 9
    starts = [int(r[0]) for _ in range(12) for r in _rows(next(loader))[0]]  # 84 windows > 2 passes
    one_pass = list(range(A0, A0 + 290, 10)) + list(range(B0, B0 + 90, 10))
    assert len(one_pass) == n_windows
    assert starts == (one_pass * 3)[:84]


def test_errors_name_the_problem_file(tmp_path, two_files):
    a, b = two_files
    with pytest.raises(FileNotFoundError, match="missing.bin"):
        build_dataloader([a, tmp_path / "missing.bin"], 10, 4)
    short = _write(tmp_path / "short.bin", 5, 5)
    with pytest.raises(ValueError, match="short.bin"):
        build_dataloader([a, short], 10, 4)
    odd = tmp_path / "odd.bin"
    odd.write_bytes(b"\x01\x02\x03")
    with pytest.raises(ValueError, match="odd.bin"):
        build_dataloader([a, odd], 10, 4)
    with pytest.raises(ValueError):
        build_dataloader([], 10, 4)
    with pytest.raises(ValueError):
        build_dataloader([a, b], 0, 4)
