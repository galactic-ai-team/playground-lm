"""Tests for data/tokenize_shards.py (Nova). Run: python -m pytest data/

Uses a stub tokenizer (char-code ids, EOT 50256 like the real one) so
the suite needs neither tiktoken nor the GPT-2 assets.
"""

import json
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).parent))

from common import make_record, write_jsonl
from tokenize_shards import tokenize_split


class StubTokenizer:
    eot_token_id = 50256

    def encode(self, text, *, add_eot=False):
        ids = [ord(c) % 100 for c in text]
        if add_eot:
            ids.append(self.eot_token_id)
        return ids


def _docs(tmp: Path, name: str, texts: list[str]) -> Path:
    p = tmp / name
    write_jsonl(p, [make_record(t, source="test") for t in texts])
    return p


def _read_bin(path: Path) -> list[int]:
    return np.fromfile(path, dtype=np.uint16).tolist()


def test_tokenize_writes_docs_with_eot(tmp_path):
    f = _docs(tmp_path, "playground-train.jsonl", ["ab", "c"])
    state = tokenize_split([f], tmp_path / "out", "train", StubTokenizer())
    assert state["finished"] and state["docs_total"] == 2
    # "ab" -> [97 % 100, 98 % 100] = [97, 98]; "c" -> [99]; EOT per doc.
    assert _read_bin(tmp_path / "out" / "train.bin") == [97, 98, 50256, 99, 50256]


def test_resume_matches_uninterrupted(tmp_path):
    f = _docs(tmp_path, "fineweb-train-00000.jsonl", ["aa", "bb", "cc", "dd"])
    out = tmp_path / "out"
    partial = tokenize_split([f], out, "train", StubTokenizer(),
                             flush_tokens=3, limit_docs=2)
    assert not partial["finished"]
    assert not (out / "train.bin").exists()
    done = tokenize_split([f], out, "train", StubTokenizer(), flush_tokens=3)
    assert done["finished"] and done["docs_total"] == 4
    expected: list[int] = []
    for t in ["aa", "bb", "cc", "dd"]:
        expected += [ord(c) % 100 for c in t] + [50256]
    assert _read_bin(out / "train.bin") == expected


def test_crash_mid_flush_truncates_to_manifest(tmp_path):
    f = _docs(tmp_path, "fineweb-train-00000.jsonl", ["aa", "bb"])
    out = tmp_path / "out"
    tokenize_split([f], out, "train", StubTokenizer(), flush_tokens=3, limit_docs=1)
    # Simulate a crash after a flush hit disk but before the manifest
    # was updated: append garbage tokens to the .part file.
    with (out / "train.bin.part").open("ab") as fh:
        np.asarray([1, 2, 3], dtype=np.uint16).tofile(fh)
    state = tokenize_split([f], out, "train", StubTokenizer(), flush_tokens=3)
    assert state["finished"]
    assert _read_bin(out / "train.bin") == [97, 97, 50256, 98, 98, 50256]


def test_finished_split_is_noop(tmp_path):
    f = _docs(tmp_path, "playground-val.jsonl", ["zz"])
    out = tmp_path / "out"
    tokenize_split([f], out, "val", StubTokenizer())
    before = (out / "val.bin").read_bytes()
    again = tokenize_split([f], out, "val", StubTokenizer())
    assert again["finished"]
    assert (out / "val.bin").read_bytes() == before


def test_manifest_records_progress(tmp_path):
    f = _docs(tmp_path, "playground-train.jsonl", ["a", "b", "c"])
    out = tmp_path / "out"
    tokenize_split([f], out, "train", StubTokenizer(), limit_docs=1)
    state = json.loads((out / "train.tokenize-state.json").read_text())
    assert state["docs_done_in_file"] == 1
    assert state["tokens_written"] == 2  # one char id + EOT
    assert state["finished"] is False
