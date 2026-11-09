"""Stage E eval-review gates on synthetic receipts (Claude). No weights, no eval items."""

import hashlib
import json
import math

import numpy as np
import pytest

from evals import stage_e_review_gates as g


# ---- helpers ----------------------------------------------------------------------

def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def pack(*examples, eot=50256):
    toks, masks = [], []
    for i, (p, c) in enumerate(examples):
        toks += [100 + (7 * i + j) % 300 for j in range(p)] + [500 + (7 * i + j) % 400 for j in range(c)] + [eot]
        masks += [0] * p + [1] * (c + 1)
    return np.array(toks, dtype=np.uint16), np.array(masks, dtype=np.uint8)


def corpus(tmp_path, train_rows, val_rows, n_examples=30):
    d = tmp_path / "corpus"
    d.mkdir()
    for split, rows in (("train", train_rows), ("val", val_rows)):
        t, m = pack(*[(3 + i % 4, 2 + i % 5) for i in range(n_examples)])
        t.tofile(d / f"{split}.bin")
        m.tofile(d / f"{split}.masks.bin")
        (d / f"{split}.jsonl").write_text("".join(json.dumps(r) + "\n" for r in rows))
    return d


def rows(prefix, n, source="dolly"):
    return [{"user": f"{prefix} question number {i}", "source": source,
             "assistant": f"{prefix} answer {i} " + " ".join(f"w{prefix}{i}{k}" for k in range(20))}
            for i in range(n)]


# ---- G1 ---------------------------------------------------------------------------

def test_checksums_pass_and_detect_tampering(tmp_path):
    (tmp_path / "meta").mkdir()
    (tmp_path / "candidate_31128").mkdir()
    w = tmp_path / "candidate_31128" / "weights.safetensors"
    w.write_bytes(b"weights")
    (tmp_path / "meta" / "SHA256SUMS.txt").write_text(f"{sha(w)}  candidate_31128/weights.safetensors\n")
    assert g.gate_checksums(tmp_path).status == g.PASS
    w.write_bytes(b"tampered")
    gate = g.gate_checksums(tmp_path)
    assert gate.status == g.FAIL and gate.detail["mismatched"] == ["candidate_31128/weights.safetensors"]


# ---- G2 / G7 ----------------------------------------------------------------------

def test_same_corpus_requires_identical_val_and_windows(tmp_path):
    val = tmp_path / "val.bin"
    val.write_bytes(b"\x01\x00" * 10)
    r = {"val_sha256": sha(val), "windows": 12, "supervised_tokens": 300}
    assert g.gate_same_corpus({"C": r, "E": dict(r)}, val).status == g.PASS
    assert g.gate_same_corpus({"C": r, "E": {**r, "val_sha256": "0" * 64}}, val).status == g.FAIL
    assert g.gate_same_corpus({"C": r, "E": {**r, "supervised_tokens": 299}}, val).status == g.FAIL
    assert g.gate_same_corpus({"C": r}, val).status == g.FAIL


def test_val_claim_flags_batch_weighted_and_repeated_windows():
    smoke = {"mean_masked_val_loss": 3.0857, "batches": 40, "batch_size": 6, "windows": 50}
    gate = g.gate_val_claim({"C": smoke, "E": {**smoke, "mean_masked_val_loss": 2.7745}}, "C", "E")
    assert gate.status == g.WARN
    assert gate.detail["delta"] == pytest.approx(-0.3112)
    assert any("batch-weighted" in i for i in gate.detail["issues"])
    assert any("more windows than one pass" in i for i in gate.detail["issues"])
    assert "corpus fit" in gate.detail["allowed_claim"] and "not generalization" in gate.detail["allowed_claim"]
    good = {"token_weighted_loss": 3.0, "single_pass": True}
    assert g.gate_val_claim({"C": good, "E": {**good, "token_weighted_loss": 2.8}}, "C", "E").status == g.PASS


# ---- G3 / G4 (sft_data audit) --------------------------------------------------------

def test_masks_and_shift_on_well_formed_corpus(tmp_path):
    d = corpus(tmp_path, rows("t", 5), rows("v", 5))
    masks = g.gate_masks(d, context_len=16, expected_examples={"train": 30, "val": 30})
    assert masks.status == g.PASS, masks.detail
    assert masks.detail["val"]["malformed"] == 0
    shift = g.gate_shift(d, context_len=16, n_windows=20)
    assert shift.status == g.PASS and shift.detail["supervised_checked"] > 0


def test_masks_fail_on_count_mismatch_and_malformed_data(tmp_path):
    d = corpus(tmp_path, rows("t", 5), rows("v", 5))
    assert g.gate_masks(d, 16, {"train": 8371, "val": 439}).status == g.FAIL
    m = np.fromfile(d / "val.masks.bin", dtype=np.uint8)
    m[0] = 2
    m.tofile(d / "val.masks.bin")
    assert g.gate_masks(d, 16, None).status == g.FAIL


# ---- G5 ---------------------------------------------------------------------------

def test_leakage_clean_split_passes(tmp_path):
    d = corpus(tmp_path, rows("t", 20), rows("v", 10))
    gate = g.gate_leakage(d)
    assert gate.status == g.PASS
    assert gate.detail["val_exact_user_in_train"] == 0 and gate.detail["ngram_overlap_rate"] == 0


def test_leakage_exact_user_overlap_fails_and_ngram_overlap_warns(tmp_path):
    train, val = rows("t", 20), rows("v", 10)
    val[0]["user"] = train[3]["user"].upper()  # case/spacing changes don't hide a duplicate
    d = corpus(tmp_path, train, val)
    assert g.gate_leakage(d).status == g.FAIL
    tmp2 = tmp_path / "b"
    tmp2.mkdir()
    train, val = rows("t", 20), rows("v", 10)
    val[1]["assistant"] = "prefix " + train[5]["assistant"]  # long shared answer text
    d2 = corpus(tmp2, train, val)
    gate = g.gate_leakage(d2)
    assert gate.status == g.WARN and gate.detail["val_completions_with_13gram_in_train"] == 1


# ---- G6 ---------------------------------------------------------------------------

def stage_e_metrics(path, steps=range(31102, 31129), shift=0.0):
    lines = [json.dumps({"step": s, "lr": g.scheduled_lr(s, 3e-5, 6, 31101, 31128) * (1 + shift)}) for s in steps]
    path.write_text("\n".join(lines) + "\n")
    return path


def test_lr_schedule_matches_packet_figures(tmp_path):
    m = stage_e_metrics(tmp_path / "metrics.jsonl")
    gate = g.gate_lr_schedule(m, 3e-5, 6, 31101, 31128)
    assert gate.status == g.PASS
    # The packet's first / peak / last LR are exactly this schedule-local curve.
    assert gate.detail["first_lr"] == pytest.approx(5e-6)
    assert gate.detail["peak_lr_logged"] == pytest.approx(3e-5)
    assert gate.detail["last_lr"] == pytest.approx(3e-6)
    assert gate.detail["rows"] == 27


def test_lr_schedule_detects_absolute_step_schedule_or_missing_steps(tmp_path):
    bad = tmp_path / "bad.jsonl"
    # What a non-schedule-local run would log: warmup counted from step 0.
    bad.write_text("\n".join(json.dumps({"step": s, "lr": 3e-5 * (0.1 + 0.45 * (1 + math.cos(math.pi * 0.99)))})
                             for s in range(31102, 31129)))
    assert g.gate_lr_schedule(bad, 3e-5, 6, 31101, 31128).status == g.FAIL
    short = stage_e_metrics(tmp_path / "short.jsonl", steps=range(31102, 31120))
    assert g.gate_lr_schedule(short, 3e-5, 6, 31101, 31128).status == g.FAIL


# ---- G8 ---------------------------------------------------------------------------

def behavior(passes, suite="public_frozen_development_only"):
    return {"suite_type": suite, "prompts_sha256": "p", "expect_sha256": "e",
            "rows": [{"id": f"item{i}", "pass": bool(x)} for i, x in enumerate(passes)]}


def test_public_dev_is_descriptive_and_paired():
    c = behavior([1, 1, 1] + [0] * 11)        # 3/14
    e = behavior([1, 1, 1, 1, 1, 1] + [0] * 8)  # 6/14, gaining 3 and losing none
    gate = g.gate_public_dev({"C": c, "E": e}, "C", "E")
    assert gate.status == g.WARN  # never PASS: not promotion evidence
    assert (gate.detail["gained"], gate.detail["lost"]) == (3, 0)
    assert gate.detail["mcnemar_exact_p"] == pytest.approx(0.25)  # not significant at n=14


def test_public_dev_rejects_mismatched_suites():
    c, e = behavior([1, 0]), behavior([1, 1])
    e["expect_sha256"] = "other"
    assert g.gate_public_dev({"C": c, "E": e}, "C", "E").status == g.FAIL


def test_mcnemar_exact():
    assert g.mcnemar_exact(0, 0) == 1.0
    assert g.mcnemar_exact(3, 0) == pytest.approx(0.25)
    assert g.mcnemar_exact(10, 0) == pytest.approx(2 / 1024)
    assert g.mcnemar_exact(4, 4) == 1.0


# ---- G9 / CLI ---------------------------------------------------------------------

def test_promotion_blocked_without_all_receipts():
    gate = g.gate_promotion(None, None, None)
    assert gate.status == g.BLOCK and len(gate.detail["missing"]) == 3
    # Round 2: a regression "receipt" that isn't a valid one no longer counts
    # (full G9 coverage, including the PASS case, is in test_regression_gate.py).
    gate = g.gate_promotion({"suite_sha256_frozen_before_scoring": "abc"}, {"fineweb_val": 1}, "Stephon")
    assert gate.status == g.BLOCK and any("valid regression receipt" in m for m in gate.detail["missing"])


def test_cli_end_to_end_reports_and_blocks_promotion(tmp_path, capsys):
    d = corpus(tmp_path, rows("t", 8), rows("v", 4))
    val_sha = sha(d / "val.bin")
    rc = {"val_sha256": val_sha, "token_weighted_loss": 3.08, "single_pass": True, "windows": 9, "supervised_tokens": 120}
    (tmp_path / "c.json").write_text(json.dumps(rc))
    (tmp_path / "e.json").write_text(json.dumps({**rc, "token_weighted_loss": 2.77}))
    (tmp_path / "cb.json").write_text(json.dumps(behavior([1, 0, 0])))
    (tmp_path / "eb.json").write_text(json.dumps(behavior([1, 1, 0])))
    m = stage_e_metrics(tmp_path / "metrics.jsonl")
    out = tmp_path / "review.json"
    code = g.main(["--corpus", str(d), "--metrics", str(m), "--context-len", "16",
                   "--val-receipt", f"C={tmp_path / 'c.json'}", "--val-receipt", f"E={tmp_path / 'e.json'}",
                   "--behavior", f"C={tmp_path / 'cb.json'}", "--behavior", f"E={tmp_path / 'eb.json'}",
                   "--out", str(out)])
    report = json.loads(out.read_text())
    statuses = {x["gate"]: x["status"] for x in report["gates"]}
    assert code == 0
    assert statuses["G2 same-corpus"] == g.PASS and statuses["G6 lr-schedule"] == g.PASS
    assert statuses["G8 public-dev"] == g.WARN and report["promotion"] == g.BLOCK
