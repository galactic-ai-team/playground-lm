"""Stage E eval-review gates over saved receipts (Claude, eval-review lane).

Runs against files the team already has; trains nothing, loads no weights.
Each gate returns PASS / WARN / FAIL / BLOCK with counts only (no corpus
text, no eval items). Promotion stays BLOCKED until the blind suite is
frozen and scored, a general-ability regression check exists, and Stephon
has approved, whatever the other gates say.

Gates
  G1 checksums     every line of meta/SHA256SUMS.txt matches the file on disk
  G2 same-corpus   masked-val receipts for C and E name the same val sha256,
                   which matches the val.bin on disk
  G3 masks         train/val token+mask pairs pass the SFT audit (0/1 masks,
                   in-vocab ids, every completion ends in EOT after a prompt)
  G4 shift         for sampled windows, every supervised y[t] equals x[t+1]
  G5 leakage       train/val overlap: exact user, exact assistant, and the
                   share of val completions with a long n-gram found in train
  G6 lr-schedule   logged LR matches the schedule-local warmup + cosine
  G7 val-claim     masked-val delta is labelled corpus-fit only; flags
                   batch-weighted means and repeated windows
  G8 public-dev    paired exact McNemar on per-item pass/fail; WARN-ONLY by
                   design: the public development probes are not a blind
                   holdout, so G8 never returns PASS and never counts toward
                   promotion, however large the gain
  G9 promotion     BLOCK unless (a) a valid regression receipt
                   (evals/regression_receipt.py) gives the candidate an
                   overall PASS, (b) the blind suite was hash-frozen before
                   scoring, (c) G5 leakage did not FAIL in this run, and
                   (d) Stephon approved. Round 2 (Claude): (a) now checks the
                   receipt's contents, not just that a file exists. Stage F:
                   per-checkpoint receipts (run_regression_receipt.py) can be
                   passed with --regression-checkpoint-receipt NAME=path and
                   are combined after checking they used identical settings.
  G10 private-win  Stage F official bar (Stephon): on a FRESH, hash-frozen
                   private suite, the candidate beats the incumbent (E31140,
                   live v0.5 Chat) only if exact paired McNemar p < alpha
                   (default 0.05) AND net cases (gained - lost) >= +5.
                   A 1-item bump is not a win; net +5 with p >= 0.05 is
                   not a win either. Reads an owner-only receipt; counts only.

Usage
  python evals/stage_e_review_gates.py --freeze-root /workspace/plm_stageE/candidate_ep1_frozen \\
      --corpus /workspace/plm_stageE/sft_candidate --metrics <metrics.jsonl> \\
      --val-receipt C=<c.json> --val-receipt E=<e.json> \\
      --behavior C=<c_behavior.json> --behavior E=<e_behavior.json> --out review.json
Exit code: 0 if no gate FAILs (BLOCK on promotion is expected), 1 otherwise.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import re
import sys
from collections import Counter
from dataclasses import dataclass, field
from pathlib import Path
from typing import Iterable

PASS, WARN, FAIL, BLOCK = "PASS", "WARN", "FAIL", "BLOCK"


@dataclass
class Gate:
    name: str
    status: str
    detail: dict = field(default_factory=dict)

    def to_dict(self) -> dict:
        return {"gate": self.name, "status": self.status, **self.detail}


def sha256_file(path: Path, chunk: int = 1 << 20) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for block in iter(lambda: f.read(chunk), b""):
            h.update(block)
    return h.hexdigest()


# ---- G1 ---------------------------------------------------------------------------

def gate_checksums(freeze_root: Path) -> Gate:
    sums = freeze_root / "meta" / "SHA256SUMS.txt"
    if not sums.is_file():
        return Gate("G1 checksums", FAIL, {"error": "meta/SHA256SUMS.txt missing"})
    checked, bad, missing = 0, [], []
    for line in sums.read_text().splitlines():
        if not line.strip():
            continue
        digest, name = line.split(maxsplit=1)
        path = freeze_root / name.lstrip("*").strip()
        if not path.is_file():
            missing.append(str(path.relative_to(freeze_root)))
            continue
        checked += 1
        if sha256_file(path) != digest.lower():
            bad.append(str(path.relative_to(freeze_root)))
    status = PASS if checked and not bad and not missing else FAIL
    return Gate("G1 checksums", status, {"checked": checked, "mismatched": bad, "missing": missing})


# ---- G2 ---------------------------------------------------------------------------

def gate_same_corpus(val_receipts: dict[str, dict], val_bin: Path | None) -> Gate:
    if len(val_receipts) < 2:
        return Gate("G2 same-corpus", FAIL, {"error": "need masked-val receipts for both checkpoints"})
    shas = {k: r.get("val_sha256") for k, r in val_receipts.items()}
    detail: dict = {"val_sha256_by_checkpoint": shas}
    if any(v is None for v in shas.values()):
        return Gate("G2 same-corpus", FAIL, {**detail, "error": "receipt lacks val_sha256"})
    if len(set(shas.values())) != 1:
        return Gate("G2 same-corpus", FAIL, {**detail, "error": "checkpoints scored on different val files"})
    if val_bin is not None:
        on_disk = sha256_file(val_bin)
        detail["val_bin_sha256"] = on_disk
        if on_disk != next(iter(shas.values())):
            return Gate("G2 same-corpus", FAIL, {**detail, "error": "receipt val sha != val.bin on disk"})
    windows = {k: (r.get("windows"), r.get("supervised_tokens")) for k, r in val_receipts.items()}
    if all(v != (None, None) for v in windows.values()) and len(set(windows.values())) != 1:
        return Gate("G2 same-corpus", FAIL, {**detail, "error": "different window sets", "windows": windows})
    return Gate("G2 same-corpus", PASS, detail)


# ---- G3 / G4 (need numpy + playground_lm.train.sft_data) ---------------------------

def gate_masks(corpus: Path, context_len: int, expected_examples: dict[str, int] | None) -> Gate:
    from playground_lm.train.sft_data import SFTDataError, audit_sft_files

    detail, status = {}, PASS
    for split in ("train", "val"):
        try:
            audit = audit_sft_files(corpus / f"{split}.bin", corpus / f"{split}.masks.bin", context_len)
        except (SFTDataError, FileNotFoundError) as e:
            return Gate("G3 masks", FAIL, {split: f"{type(e).__name__}: {e}"})
        d = audit.to_dict()
        detail[split] = {k: d[k] for k in ("examples", "tokens", "supervised_tokens", "oversized_examples",
                                           "unsupervisable_tokens", "malformed", "supervisable_fraction")}
        want = (expected_examples or {}).get(split)
        if want is not None and want != audit.examples:
            status = FAIL
            detail[split]["expected_examples"] = want
        if audit.unsupervisable_tokens:
            status = WARN if status == PASS else status
    return Gate("G3 masks", status, detail)


def gate_shift(corpus: Path, context_len: int, n_windows: int = 64, seed: int = 0) -> Gate:
    import numpy as np

    from playground_lm.model import IGNORE_INDEX
    from playground_lm.train.sft_data import audit_sft_arrays, window_targets

    tokens = np.memmap(corpus / "val.bin", dtype=np.uint16, mode="r")
    masks = np.memmap(corpus / "val.masks.bin", dtype=np.uint8, mode="r")
    _, layout = audit_sft_arrays(tokens, masks, context_len)
    rng = np.random.default_rng(seed)
    starts = rng.choice(np.minimum(layout.starts, tokens.size - context_len - 1), size=n_windows)
    checked = bad = 0
    for s in starts:
        x, y = window_targets(tokens, masks, layout.starts, int(s), context_len)
        sup = np.flatnonzero(y != IGNORE_INDEX)
        sup = sup[sup < context_len - 1]  # x[t+1] exists inside the window
        checked += sup.size
        bad += int(np.count_nonzero(y[sup] != x[sup + 1]))
        bad += int(np.count_nonzero(masks[int(s) + 1 + np.flatnonzero(y != IGNORE_INDEX)] != 1))
    status = PASS if checked and not bad else FAIL
    return Gate("G4 shift", status, {"windows": int(n_windows), "supervised_checked": int(checked), "violations": bad})


# ---- G5 -------------------------------------------------------------------------------

_WORD = re.compile(r"\w+", re.UNICODE)


def _words(text: str) -> list[str]:
    return _WORD.findall(text.lower())


def _ngrams(words: list[str], n: int) -> set[tuple[str, ...]]:
    return {tuple(words[i : i + n]) for i in range(len(words) - n + 1)}


def _load_jsonl(path: Path) -> list[dict]:
    with open(path, encoding="utf-8") as f:
        return [json.loads(line) for line in f if line.strip()]


def gate_leakage(corpus: Path, ngram: int = 13, warn_rate: float = 0.02) -> Gate:
    train, val = _load_jsonl(corpus / "train.jsonl"), _load_jsonl(corpus / "val.jsonl")
    norm = lambda s: " ".join(_words(s or ""))
    train_users = {norm(r.get("user")) for r in train}
    train_answers = {norm(r.get("assistant")) for r in train}
    exact_user = sum(norm(r.get("user")) in train_users for r in val)
    exact_answer = sum(norm(r.get("assistant")) in train_answers for r in val)
    train_grams: set = set()
    for r in train:
        train_grams |= _ngrams(_words(r.get("assistant") or ""), ngram)
    by_source: Counter = Counter()
    hits_by_source: Counter = Counter()
    long_enough = hits = 0
    for r in val:
        grams = _ngrams(_words(r.get("assistant") or ""), ngram)
        src = r.get("source", "unknown")
        by_source[src] += 1
        if grams:
            long_enough += 1
            if grams & train_grams:
                hits += 1
                hits_by_source[src] += 1
    rate = hits / long_enough if long_enough else 0.0
    status = FAIL if exact_user else (WARN if exact_answer or rate > warn_rate else PASS)
    return Gate("G5 leakage", status, {
        "train_examples": len(train), "val_examples": len(val),
        "val_exact_user_in_train": exact_user,
        "val_exact_assistant_in_train": exact_answer,
        f"val_completions_with_{ngram}gram_in_train": hits,
        "val_completions_long_enough": long_enough,
        "ngram_overlap_rate": round(rate, 4),
        "overlap_by_source": {s: [hits_by_source[s], by_source[s]] for s in sorted(by_source)},
        "note": "short templated completions (generated arithmetic/identity) are below the n-gram "
                "length; check them by source, they can share templates across train and val",
    })


# ---- G6 -------------------------------------------------------------------------------

def scheduled_lr(step: int, peak: float, warmup: int, schedule_start: int, max_steps: int) -> float:
    """loop._learning_rate on schedule-local steps (mirrors train/loop.py)."""
    local, total = step - schedule_start, max_steps - schedule_start
    if local <= warmup:
        return peak * local / warmup
    progress = (local - warmup) / (total - warmup)
    return peak * (0.1 + 0.45 * (1.0 + math.cos(math.pi * progress)))


def gate_lr_schedule(metrics: Path, peak: float, warmup: int, schedule_start: int,
                     max_steps: int, rel_tol: float = 1e-6) -> Gate:
    rows = [json.loads(line) for line in metrics.read_text().splitlines() if line.strip()]
    rows = [r for r in rows if "lr" in r and r.get("step", 0) > schedule_start]
    if not rows:
        return Gate("G6 lr-schedule", FAIL, {"error": "no metrics rows after schedule_start"})
    worst = max(abs(r["lr"] - scheduled_lr(r["step"], peak, warmup, schedule_start, max_steps))
                / max(scheduled_lr(r["step"], peak, warmup, schedule_start, max_steps), 1e-30)
                for r in rows)
    steps = [r["step"] for r in rows]
    detail = {"rows": len(rows), "first_step": steps[0], "last_step": steps[-1],
              "first_lr": rows[0]["lr"], "peak_lr_logged": max(r["lr"] for r in rows),
              "last_lr": rows[-1]["lr"], "max_rel_error": worst}
    ok = worst <= rel_tol and steps[0] == schedule_start + 1 and steps[-1] == max_steps
    return Gate("G6 lr-schedule", PASS if ok else FAIL, detail)


# ---- G7 -------------------------------------------------------------------------------

def gate_val_claim(val_receipts: dict[str, dict], baseline: str, candidate: str) -> Gate:
    c, e = val_receipts[baseline], val_receipts[candidate]
    loss = lambda r: r.get("token_weighted_loss", r.get("mean_masked_val_loss"))
    detail = {"baseline": loss(c), "candidate": loss(e),
              "delta": None if loss(c) is None or loss(e) is None else round(loss(e) - loss(c), 4),
              "allowed_claim": "lower masked loss on the Stage E SFT val split (corpus fit); "
                               "not generalization, not a benchmark result"}
    issues = []
    for k, r in ((baseline, c), (candidate, e)):
        if "token_weighted_loss" not in r:
            issues.append(f"{k}: batch-weighted mean only (each batch counts equally regardless of "
                          f"how many completion tokens it supervises)")
        if r.get("single_pass") is False or (r.get("windows") and r.get("batches") and r.get("batch_size")
                                             and r["batches"] * r["batch_size"] > r["windows"]):
            issues.append(f"{k}: scored more windows than one pass (val windows repeated)")
    detail["issues"] = issues
    return Gate("G7 val-claim", WARN if issues else PASS, detail)


# ---- G8 -------------------------------------------------------------------------------

def mcnemar_exact(b: int, c: int) -> float:
    """Two-sided exact McNemar p-value from discordant counts b, c."""
    n = b + c
    if n == 0:
        return 1.0
    k = min(b, c)
    p = sum(math.comb(n, i) for i in range(k + 1)) / 2 ** n
    return min(1.0, 2 * p)


def gate_public_dev(behavior: dict[str, dict], baseline: str, candidate: str) -> Gate:
    rc, re_ = behavior[baseline], behavior[candidate]
    issues = [k for k, r in behavior.items() if r.get("suite_type") != "public_frozen_development_only"]
    if rc.get("prompts_sha256") != re_.get("prompts_sha256") or rc.get("expect_sha256") != re_.get("expect_sha256"):
        return Gate("G8 public-dev", FAIL, {"error": "the two runs used different prompt/expect files"})
    pc = {r["id"]: bool(r["pass"]) for r in rc["rows"]}
    pe = {r["id"]: bool(r["pass"]) for r in re_["rows"]}
    if set(pc) != set(pe):
        return Gate("G8 public-dev", FAIL, {"error": "different item ids"})
    gained = sum(pe[i] and not pc[i] for i in pc)
    lost = sum(pc[i] and not pe[i] for i in pc)
    return Gate("G8 public-dev", WARN, {
        "n": len(pc), "baseline_passed": sum(pc.values()), "candidate_passed": sum(pe.values()),
        "gained": gained, "lost": lost, "mcnemar_exact_p": round(mcnemar_exact(gained, lost), 4),
        "suite_type_issues": issues,
        "status_meaning": "always WARN: public development probes are descriptive only, never promotion evidence",
    })


# ---- G9 -------------------------------------------------------------------------------

def gate_promotion(blind_receipt: dict | None, regression_receipt: dict | None, approved_by: str | None,
                   candidate: str = "E", leakage_status: str | None = None) -> Gate:
    from evals.regression_gate import validate_receipt, verdict

    missing, detail = [], {}
    if not blind_receipt or not blind_receipt.get("suite_sha256_frozen_before_scoring"):
        missing.append("private blind suite: hash frozen before scoring, then C vs candidate scored")
    if not regression_receipt:
        missing.append("general-ability regression receipt (FineWeb-Edu val loss, HellaSwag, LAMBADA)")
    else:
        problems = validate_receipt(regression_receipt)
        if problems:
            missing.append("valid regression receipt: " + "; ".join(problems))
        elif candidate not in regression_receipt["models"]:
            missing.append(f"regression receipt has no candidate {candidate!r}")
        else:
            v = verdict(regression_receipt, candidate)
            detail["regression"] = {"status": v["status"],
                                    "metrics": {k: m["status"] for k, m in v["metrics"].items()}}
            if v["status"] != PASS:
                missing.append(f"regression verdict PASS for {candidate} (got {v['status']})")
    if leakage_status == FAIL:
        missing.append("G5 leakage must not FAIL (re-run on the current corpus)")
    if approved_by != "Stephon":
        missing.append("Stephon's approval")
    return Gate("G9 promotion", BLOCK if missing else PASS, {"candidate": candidate, "missing": missing, **detail})


# ---- G10 ------------------------------------------------------------------------------

def min_clean_gains(alpha: float = 0.05) -> int:
    """Fewest gains (with no losses) that give exact McNemar p < alpha."""
    b = 1
    while mcnemar_exact(b, 0) >= alpha:
        b += 1
    return b


def gate_private_win(private: dict | None, candidate: str, incumbent: str = "E31140",
                     alpha: float = 0.05, used_suites: Iterable[str] = (),
                     min_net: int = 5) -> Gate:
    name = "G10 private-win"
    if not private:
        return Gate(name, BLOCK, {"missing": ["fresh frozen private suite scored for candidate and incumbent"]})
    suite = private.get("suite_sha256_frozen_before_scoring")
    if not suite:
        return Gate(name, BLOCK, {"missing": ["suite hash frozen before scoring"]})
    if suite in set(used_suites):
        return Gate(name, BLOCK, {"missing": ["a FRESH suite (this hash was already used)"]})
    results = private.get("results") or {}
    if candidate not in results or incumbent not in results:
        return Gate(name, BLOCK, {"missing": [f"results for both {candidate} and {incumbent}"]})
    pc = {r["id"]: bool(r["pass"]) for r in results[candidate]["rows"]}
    pi = {r["id"]: bool(r["pass"]) for r in results[incumbent]["rows"]}
    if set(pc) != set(pi):
        return Gate(name, FAIL, {"error": "candidate and incumbent scored on different items"})
    gained = sum(pc[i] and not pi[i] for i in pc)
    lost = sum(pi[i] and not pc[i] for i in pc)
    net = gained - lost
    p = mcnemar_exact(gained, lost)
    # Official Stage F bar: McNemar p < alpha AND net >= +5 (Stephon, 2026-11-09).
    win = gained > lost and p < alpha and net >= min_net
    return Gate(name, PASS if win else BLOCK, {
        "candidate": candidate, "incumbent": incumbent, "n": len(pc),
        "candidate_passed": sum(pc.values()), "incumbent_passed": sum(pi.values()),
        "gained": gained, "lost": lost, "net": net, "mcnemar_exact_p": round(p, 4),
        "alpha": alpha, "min_net": min_net,
        "min_clean_gains_for_p": min_clean_gains(alpha),
    })


# ---- CLI --------------------------------------------------------------------------------

def _pairs(values: Iterable[str]) -> dict[str, dict]:
    out = {}
    for v in values or []:
        k, _, p = v.partition("=")
        out[k] = json.loads(Path(p).read_text(encoding="utf-8"))
    return out


def run(args) -> list[Gate]:
    gates: list[Gate] = []
    if args.freeze_root:
        gates.append(gate_checksums(Path(args.freeze_root)))
    val = _pairs(args.val_receipt)
    corpus = Path(args.corpus) if args.corpus else None
    if val:
        gates.append(gate_same_corpus(val, corpus / "val.bin" if corpus else None))
    if corpus:
        expected = {"train": args.expect_train_examples, "val": args.expect_val_examples}
        gates.append(gate_masks(corpus, args.context_len, expected))
        gates.append(gate_shift(corpus, args.context_len))
        gates.append(gate_leakage(corpus))
    if args.metrics:
        peak, warm, start, top = args.peak_lr, args.warmup, args.schedule_start, args.max_steps
        if args.sweep_run:  # round 2: the run's own config (sweep_config.SweepRun.to_dict())
            run = json.loads(Path(args.sweep_run).read_text())
            peak, warm, start, top = run["peak_lr"], run["warmup"], run["schedule_start"], run["max_steps"]
        gates.append(gate_lr_schedule(Path(args.metrics), peak, warm, start, top))
    if len(val) >= 2:
        gates.append(gate_val_claim(val, args.baseline, args.candidate))
    beh = _pairs(args.behavior)
    if len(beh) >= 2:
        gates.append(gate_public_dev(beh, args.baseline, args.candidate))
    blind = json.loads(Path(args.blind_receipt).read_text()) if args.blind_receipt else None
    regr = json.loads(Path(args.regression_receipt).read_text()) if args.regression_receipt else None
    if args.regression_checkpoint_receipt:
        from evals.regression_gate import combine_checkpoint_receipts

        per = _pairs(args.regression_checkpoint_receipt)
        base = per.pop(args.baseline, None)
        if base is None:
            raise SystemExit(f"--regression-checkpoint-receipt needs {args.baseline}=<C31101 receipt>")
        regr = combine_checkpoint_receipts(base, per, args.baseline)
    leak = next((x.status for x in gates if x.name == "G5 leakage"), None)
    gates.append(gate_promotion(blind, regr, args.approved_by, args.candidate, leak))
    private = json.loads(Path(args.private_receipt).read_text()) if args.private_receipt else None
    gates.append(gate_private_win(private, args.candidate, args.incumbent, args.private_alpha,
                                  args.used_suite_sha or (), args.private_min_net))
    return gates


def main(argv=None) -> int:
    p = argparse.ArgumentParser(description="Stage E eval-review gates over saved receipts")
    p.add_argument("--freeze-root")
    p.add_argument("--corpus", help="dir with train/val .bin, .masks.bin, .jsonl")
    p.add_argument("--metrics", help="training metrics.jsonl")
    p.add_argument("--val-receipt", action="append", help="NAME=path (masked-val receipt)")
    p.add_argument("--behavior", action="append", help="NAME=path (run_behavior_eval report)")
    p.add_argument("--blind-receipt")
    p.add_argument("--regression-receipt")
    p.add_argument("--approved-by")
    p.add_argument("--baseline", default="C")
    p.add_argument("--candidate", default="E")
    p.add_argument("--context-len", type=int, default=1024)
    p.add_argument("--expect-train-examples", type=int)
    p.add_argument("--expect-val-examples", type=int)
    p.add_argument("--peak-lr", type=float, default=3e-5)
    p.add_argument("--warmup", type=int, default=6)
    p.add_argument("--schedule-start", type=int, default=31101)
    p.add_argument("--max-steps", type=int, default=31128)
    p.add_argument("--sweep-run", help="run config JSON (SweepRun.to_dict()); overrides the four LR args")
    p.add_argument("--regression-checkpoint-receipt", action="append",
                   help="NAME=path per-checkpoint receipt (run_regression_receipt.py); include the baseline")
    p.add_argument("--private-receipt", help="owner-only private-suite results (G10); never commit it")
    p.add_argument("--incumbent", default="E31140")
    p.add_argument("--private-alpha", type=float, default=0.05)
    p.add_argument("--private-min-net", type=int, default=5,
                   help="official Stage F bar: require net (gained-lost) >= this many cases")
    p.add_argument("--used-suite-sha", action="append", help="hash of an already-used private suite")
    p.add_argument("--out")
    args = p.parse_args(argv)
    gates = run(args)
    g9 = next(g.status for g in gates if g.name == "G9 promotion")
    g10 = next(g.status for g in gates if g.name == "G10 private-win")
    report = {"gates": [g.to_dict() for g in gates],
              "promotion": PASS if g9 == PASS and g10 == PASS else BLOCK}
    text = json.dumps(report, indent=2)
    if args.out:
        Path(args.out).write_text(text + "\n")
    print(text)
    return 1 if any(g.status == FAIL for g in gates) else 0


if __name__ == "__main__":
    sys.exit(main())
