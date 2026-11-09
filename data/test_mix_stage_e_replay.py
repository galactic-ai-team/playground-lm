"""Tests for data/mix_stage_e_replay.py (Nova, Round 3). Offline:
tiny stub bins + a fake word tokenizer. No downloads, no HF.
Run: python -m pytest data/
"""

import json
import sys
from pathlib import Path

import numpy as np
import pytest

sys.path.insert(0, str(Path(__file__).parent))

from build_stage_e_sft import _g5_ngrams, g5_norm
from mix_stage_e_replay import (
    EOT_ID,
    audit_mixed,
    doc_spans,
    mix,
    replay_budget,
    sample_replay,
)

DOLLY = "databricks/databricks-dolly-15k"
VAL_USER = ("please tell me everything about the history of the roman "
            "empire in great detail today")
VAL_ANSWER = "the roman empire fell in the west in 476 ce"


class Vocab:
    def __init__(self):
        self.word2id: dict[str, int] = {}
        self.id2word: dict[int, str] = {}

    def ids(self, text: str) -> list[int]:
        out = []
        for w in text.lower().split():
            if w not in self.word2id:
                i = len(self.word2id) + 1
                self.word2id[w] = i
                self.id2word[i] = w
            out.append(self.word2id[w])
        return out

    def decode(self, ids) -> str:
        return " ".join(self.id2word[i] for i in ids if i != EOT_ID and i in self.id2word)


def _write_sft(tmp_path: Path, vocab: Vocab, *, leaking: bool = False) -> Path:
    sft = tmp_path / "sft"
    sft.mkdir()
    train_rows = [
        {"id": "t1", "source": DOLLY, "license": "CC BY-SA 3.0",
         "user": "explain photosynthesis simply",
         "assistant": "plants convert sunlight into food"},
        {"id": "t2", "source": DOLLY, "license": "CC BY-SA 3.0",
         "user": "name a primary color", "assistant": "red"},
    ]
    val_rows = [
        {"id": "v1", "source": DOLLY, "license": "CC BY-SA 3.0",
         "user": VAL_USER, "assistant": VAL_ANSWER},
    ]
    if leaking:
        val_rows[0]["user"] = train_rows[0]["user"]
    tokens, masks = [], []
    for row in train_rows:
        p = vocab.ids("user " + row["user"] + " assistant")
        c = vocab.ids(row["assistant"]) + [EOT_ID]
        tokens += p + c
        masks += [0] * len(p) + [1] * len(c)
    np.array(tokens, dtype=np.uint16).tofile(sft / "train.bin")
    np.array(masks, dtype=np.uint8).tofile(sft / "train.masks.bin")
    vtokens = vocab.ids("user " + val_rows[0]["user"] + " assistant") \
        + vocab.ids(val_rows[0]["assistant"]) + [EOT_ID]
    np.array(vtokens, dtype=np.uint16).tofile(sft / "val.bin")
    np.array([1] * len(vtokens), dtype=np.uint8).tofile(sft / "val.masks.bin")
    for name, rows in (("train", train_rows), ("val", val_rows)):
        (sft / f"{name}.jsonl").write_text(
            "".join(json.dumps(r) + "\n" for r in rows))
    (sft / "provenance.json").write_text(json.dumps(
        {"sources": [{"id": DOLLY, "license": "CC BY-SA 3.0"}]}))
    return sft


def _write_replay(tmp_path: Path, vocab: Vocab) -> Path:
    docs = []
    for i in range(12):  # clean docs, 9 tokens each incl. EOT
        docs.append(vocab.ids(
            f"replay document {i} about hills rivers and quiet towns") + [EOT_ID])
    # Exact collision with the val assistant -> val_exact drop.
    docs.append(vocab.ids(VAL_ANSWER) + [EOT_ID])
    # Shares a 13-gram with the val user -> val_ngram drop.
    docs.append(vocab.ids(
        "please tell me everything about the history of the roman empire "
        "in great detail plus extra words here") + [EOT_ID])
    # Shares a 13-gram with the eval text -> eval_ngram drop.
    docs.append(vocab.ids(
        "evalonly alpha bravo charlie delta echo foxtrot golf hotel india "
        "juliet kilo lima mike november") + [EOT_ID])
    flat = np.array([t for d in docs for t in d], dtype=np.uint16)
    path = tmp_path / "replay.bin"
    flat.tofile(path)
    return path


EVAL_TEXTS = ["evalonly alpha bravo charlie delta echo foxtrot golf hotel "
              "india juliet kilo lima mike"]


def test_replay_budget_math():
    assert replay_budget(900, 0.25) == 300
    assert replay_budget(900, 0.5) == 900
    assert replay_budget(1000, 0.1) == 112  # ceil(1000 * 0.1 / 0.9)


def test_doc_spans_keeps_trailing_partial_doc():
    arr = np.array([1, 2, EOT_ID, 3, 4], dtype=np.uint16)
    assert doc_spans(arr) == [(0, 3), (3, 5)]


def test_mix_end_to_end(tmp_path):
    vocab = Vocab()
    sft = _write_sft(tmp_path, vocab)
    replay = _write_replay(tmp_path, vocab)
    out = tmp_path / "mixed"
    prov = mix(sft_dir=sft, replay_bins=[replay], ratio=0.25, seed=5,
               out_dir=out, tokenizer=vocab, eval_texts=EVAL_TEXTS)
    sft_tokens = prov["token_counts"]["sft_train"]
    budget = replay_budget(sft_tokens, 0.25)
    assert prov["token_counts"]["replay"] >= budget
    assert prov["token_counts"]["mixed_train"] == sft_tokens + prov["token_counts"]["replay"]
    assert 0.25 <= prov["ratio_actual_replay_share"] < 0.45
    # Drop *counts* depend on how many poisoned docs the sampler walks
    # past before the budget fills; they are asserted exactly in
    # test_sample_replay_drops_collisions (unbounded budget). Here the
    # safety property is deterministic: no poisoned text is ever kept
    # (see the replay.jsonl assertions below).
    checks = prov["checks"]
    assert checks["pass"] and checks["sft_prefix_masks_identical"]
    assert checks["replay_segment_full_loss"] and checks["val_bin_identical"]
    licenses = {s["id"]: s.get("license") for s in prov["sources"]}
    assert licenses["fineweb-edu"] == "ODC-By v1.0"
    assert licenses[DOLLY] == "CC BY-SA 3.0"
    assert prov["mask_policy"]["replay"].startswith("full-loss")
    kept_texts = [json.loads(l)["text"] for l in (out / "replay.jsonl").read_text().splitlines()]
    assert all("roman" not in t for t in kept_texts)
    assert all("evalonly" not in t for t in kept_texts)


def test_sample_replay_drops_collisions(tmp_path):
    vocab = Vocab()
    replay = _write_replay(tmp_path, vocab)
    val_users = {g5_norm(VAL_USER)}
    val_answers = {g5_norm(VAL_ANSWER)}
    val_grams = _g5_ngrams(VAL_USER)
    eval_grams: set = set()
    for t in EVAL_TEXTS:
        eval_grams |= _g5_ngrams(t)
    tokens, records, report = sample_replay(
        [replay], budget=10**9, seed=5, tokenizer=vocab,
        val_users=val_users, val_answers=val_answers,
        val_user_grams=val_grams, eval_grams=eval_grams)
    assert report["drops"] == {"val_exact": 1, "val_ngram": 1, "eval_ngram": 1}
    assert report["documents_kept"] == 12
    assert tokens.size == 12 * 10  # 9 words + EOT per clean doc
    assert all("roman" not in r["text"] and "evalonly" not in r["text"]
               for r in records)


def test_mix_refuses_leaking_sft(tmp_path):
    vocab = Vocab()
    sft = _write_sft(tmp_path, vocab, leaking=True)
    replay = _write_replay(tmp_path, vocab)
    with pytest.raises(SystemExit):
        mix(sft_dir=sft, replay_bins=[replay], ratio=0.25, seed=5,
            out_dir=tmp_path / "mixed", tokenizer=vocab)


def test_mix_is_deterministic(tmp_path):
    hashes = []
    for name in ("m1", "m2"):
        # Rebuild identical fixtures per run under separate roots.
        root = tmp_path / name
        root.mkdir()
        v = Vocab()
        sft_dir = _write_sft(root, v)
        replay = _write_replay(root, v)
        prov = mix(sft_dir=sft_dir, replay_bins=[replay], ratio=0.5, seed=9,
                   out_dir=root / "mixed", tokenizer=v)
        hashes.append(prov["outputs_sha256"]["train_bin"])
    assert hashes[0] == hashes[1]


def test_mix_without_tokenizer_records_skip(tmp_path):
    vocab = Vocab()
    sft = _write_sft(tmp_path, vocab)
    replay = _write_replay(tmp_path, vocab)
    prov = mix(sft_dir=sft, replay_bins=[replay], ratio=0.1, seed=5,
               out_dir=tmp_path / "mixed", tokenizer=None)
    assert prov["replay_sampling"]["text_checked"] is False
    assert prov["checks"]["pass"]
    with pytest.raises(SystemExit):
        mix(sft_dir=sft, replay_bins=[replay], ratio=0.1, seed=5,
            out_dir=tmp_path / "mixed2", tokenizer=None,
            require_text_check=True)


def test_audit_mixed_detects_mask_tampering(tmp_path):
    vocab = Vocab()
    sft = _write_sft(tmp_path, vocab)
    replay = _write_replay(tmp_path, vocab)
    out = tmp_path / "mixed"
    mix(sft_dir=sft, replay_bins=[replay], ratio=0.25, seed=5,
        out_dir=out, tokenizer=vocab)
    masks = np.fromfile(out / "train.masks.bin", dtype=np.uint8)
    masks[-1] = 0  # break full-loss replay tail
    masks.tofile(out / "train.masks.bin")
    checks = audit_mixed(out, sft)
    assert checks["replay_segment_full_loss"] is False
    assert checks["pass"] is False
