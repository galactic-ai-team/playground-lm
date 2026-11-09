"""Tests for data/audit_stage_e_release.py (Nova). Synthetic fixtures
only — no real corpus, no eval contents. Run: python -m pytest data/
"""

import json
import struct
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent))

from audit_stage_e_release import audit
from build_stage_e_corpus import load_denylist

DOLLY = "databricks/databricks-dolly-15k"
ARITH = "generated-arithmetic"
IDENT = "generated-identity"
MIT = "MIT (this repository)"


def _row(rid, source, license_, user):
    return {"id": rid, "source": source, "license": license_,
            "user": user, "assistant": "ok",
            "text": f"User: {user}\nAssistant: ok"}


def _provenance():
    return {
        "sources": [
            {"id": DOLLY, "license": "CC BY-SA 3.0"},
            {"id": ARITH, "license": "generated exact-label arithmetic"},
            {"id": IDENT, "license": "generated"},
        ],
        "counts": {"by_source_train": {DOLLY: 1, ARITH: 1}},
        "train": {"examples": 2, "tokens": 10, "supervised_tokens": 4},
        "val": {"examples": 1, "tokens": 5, "supervised_tokens": 2},
    }


def _rows():
    train = [
        _row("t1", DOLLY, "CC BY-SA 3.0", "unique train prompt alpha"),
        _row("t2", ARITH, MIT, "What is 2 + 2?"),
    ]
    val = [_row("v1", IDENT, MIT, "Who built PlaygroundLM?")]
    return train, val


def _write_bins(tmp_path: Path, split: str, tokens: int, supervised: int):
    (tmp_path / f"{split}.bin").write_bytes(struct.pack(f"<{tokens}H", *([7] * tokens)))
    masks = bytes([1] * supervised + [0] * (tokens - supervised))
    (tmp_path / f"{split}.masks.bin").write_bytes(masks)


def _denylist(tmp_path: Path) -> list[str]:
    path = tmp_path / "frozen.json"
    path.write_text(json.dumps({"prompts": [
        {"user": "an unrelated frozen prompt with several distinct "
                 "words inside it for safety"}]}))
    texts, schema = load_denylist(path)
    assert schema == "prompts[].user"
    return texts


def test_clean_fixture_has_no_blockers(tmp_path):
    train, val = _rows()
    _write_bins(tmp_path, "train", 10, 4)
    _write_bins(tmp_path, "val", 5, 2)
    report = audit(provenance=_provenance(), train_rows=train, val_rows=val,
                   denylist_texts=_denylist(tmp_path), corpus_dir=tmp_path)
    assert report["release_blockers"] == [], report


def test_train_val_user_overlap_is_a_blocker(tmp_path):
    train, val = _rows()
    # Punctuation/case variant: only the G5 norm (strips non-word chars)
    # catches this — the round-1 whitespace norm missed dolly-4758's class.
    val = val + [_row("v2", IDENT, MIT, "Unique, train prompt ALPHA!")]
    report = audit(provenance=_provenance(), train_rows=train, val_rows=val)
    assert "split_user_disjoint" in report["release_blockers"]
    # The report carries counts/hashes only, never the prompt text.
    assert "unique train prompt alpha" not in json.dumps(report).lower()


def test_denylist_hit_is_a_blocker(tmp_path):
    texts = _denylist(tmp_path)
    train, val = _rows()
    train[0] = _row("t1", DOLLY, "CC BY-SA 3.0", texts[0])
    report = audit(provenance=_provenance(), train_rows=train, val_rows=val,
                   denylist_texts=texts)
    assert "denylist_clean" in report["release_blockers"]


def test_license_mismatch_is_a_blocker():
    train, val = _rows()
    train[0] = _row("t1", DOLLY, MIT, "unique train prompt alpha")
    report = audit(provenance=_provenance(), train_rows=train, val_rows=val)
    assert "jsonl_license_fields" in report["release_blockers"]
    prov = _provenance()
    prov["sources"][0]["license"] = "MIT"
    report2 = audit(provenance=prov, train_rows=train, val_rows=val)
    assert "provenance_sources_license" in report2["release_blockers"]


def test_mask_parity_failures_are_blockers(tmp_path):
    train, val = _rows()
    _write_bins(tmp_path, "train", 10, 4)
    (tmp_path / "val.bin").write_bytes(struct.pack("<5H", *([7] * 5)))
    (tmp_path / "val.masks.bin").write_bytes(bytes([1, 0, 1]))  # too short
    report = audit(provenance=_provenance(), train_rows=train, val_rows=val,
                   corpus_dir=tmp_path)
    assert "bins_masks_parity[val]" in report["release_blockers"]
    assert "bins_masks_parity[train]" not in report["release_blockers"]


def test_cleanup_reconciliation_arithmetic():
    prov = _provenance()
    prov["train"]["examples"] = 100
    prov["val"]["examples"] = 20
    expect = {
        "cleanup": {"denylist_exact_train": 3, "denylist_exact_val": 1,
                    "user_overlaps_removed": 5},
        "final": {"train_examples": 92, "val_examples": 19},
    }
    report = audit(provenance=prov, expect=expect)
    assert report["release_blockers"] == [], report
    expect["final"]["train_examples"] = 91
    report = audit(provenance=prov, expect=expect)
    assert "cleanup_reconciliation" in report["release_blockers"]
