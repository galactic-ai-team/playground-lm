"""Stage E release audit — provenance / denylist / split / masks (Nova).

Verifies, at hash-and-count level only, that a Stage E SFT corpus and
its provenance record describe the same artifacts that were frozen and
trained. It never prints prompt, response, or eval text: details are
counts, record ids, and sha256 digests, so the report is safe to share
across lanes (including about corpora built from held-out-adjacent
material).

What it checks (each failure is a named release blocker):
  provenance   every source carries a license; the Dolly source is
               exactly "CC BY-SA 3.0" (share-alike attribution chain).
  jsonl        every record has source+license; license matches the
               canonical string for its source; per-source train
               counts match provenance counts.by_source_train.
  split        no record id appears in both splits; no normalized
               user text appears in both splits (the train/val user
               overlap class removed during candidate prep).
  denylist     re-runs the fail-closed denylist (Nova's loader) on the
               user field of both splits: zero exact or 13-gram hits.
  bins         if token/mask bins are present: sha256 vs expected
               values when provided; token count = bin bytes / 2
               (uint16 LE); masks length == token count (uint8);
               masks values in {0,1}; masks sum == the split's
               supervised_tokens in provenance.
  reconcile    if an expectations file gives post-cleanup counts and
               the cleanup tallies (denylist exact drops per split,
               train-side user overlaps removed), the pre-cleanup
               provenance numbers must reconcile exactly:
               final = provenance - exact_drops - overlaps_removed.

Usage:
  python data/audit_stage_e_release.py \
      --corpus-dir /path/to/sft_candidate \
      --denylist evals/behavior/behavior_eval.prompts.frozen.json \
      --expect /path/to/freeze_expectations.json
Exit code 0 = no blockers, 1 = at least one blocker.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))

from build_stage_e_corpus import load_denylist
from build_stage_e_sft import _g5_ngrams, g5_norm

CANONICAL_LICENSE = {
    "databricks/databricks-dolly-15k": "CC BY-SA 3.0",
    "dolly-15k": "CC BY-SA 3.0",
    "generated-arithmetic": "MIT (this repository)",
    "arithmetic-generated": "MIT (this repository)",
    "generated-identity": "MIT (this repository)",
    "team-identity": "MIT (this repository)",
}
DOLLY_SOURCES = {"databricks/databricks-dolly-15k", "dolly-15k"}


def _sha256_file(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def _read_jsonl(path: Path) -> list[dict]:
    return [json.loads(line) for line in path.read_text().splitlines() if line.strip()]


class Report:
    def __init__(self) -> None:
        self.checks: list[dict] = []

    def add(self, name: str, ok: bool, detail: dict) -> None:
        self.checks.append(
            {"name": name, "status": "pass" if ok else "BLOCKER", "detail": detail}
        )

    @property
    def blockers(self) -> list[str]:
        return [c["name"] for c in self.checks if c["status"] != "pass"]

    def to_dict(self) -> dict:
        return {"checks": self.checks, "release_blockers": self.blockers}


def audit(
    *,
    provenance: dict | None,
    train_rows: list[dict] | None = None,
    val_rows: list[dict] | None = None,
    denylist_texts: list[str] | None = None,
    corpus_dir: Path | None = None,
    expect: dict | None = None,
) -> dict:
    rep = Report()
    expect = expect or {}

    # ---------------------------------------------------------- provenance
    if provenance is not None:
        sources = provenance.get("sources") or []
        by_id = {s.get("id"): s for s in sources}
        dolly_ok = any(
            by_id.get(src, {}).get("license") == "CC BY-SA 3.0"
            for src in DOLLY_SOURCES & set(by_id)
        ) if (DOLLY_SOURCES & set(by_id)) else False
        missing_license = [s.get("id") for s in sources if not s.get("license")]
        rep.add(
            "provenance_sources_license",
            bool(sources) and not missing_license and dolly_ok,
            {
                "sources": sorted(by_id),
                "missing_license": missing_license,
                "dolly_cc_by_sa_3_0": dolly_ok,
            },
        )

    # --------------------------------------------------------------- jsonl
    if train_rows is not None and val_rows is not None:
        all_rows = train_rows + val_rows
        bad_license = [
            r.get("id") for r in all_rows
            if r.get("source") not in CANONICAL_LICENSE
            or r.get("license") != CANONICAL_LICENSE[r["source"]]
        ]
        rep.add(
            "jsonl_license_fields",
            not bad_license,
            {"records": len(all_rows), "bad_license_record_ids": bad_license[:10]},
        )
        if provenance is not None:
            expected_counts = (provenance.get("counts") or {}).get("by_source_train")
            if expected_counts is not None:
                actual: dict[str, int] = {}
                for r in train_rows:
                    actual[r["source"]] = actual.get(r["source"], 0) + 1
                rep.add(
                    "jsonl_source_counts_match_provenance",
                    actual == expected_counts,
                    {"provenance": expected_counts, "actual": actual},
                )
        train_ids = {r["id"] for r in train_rows}
        val_ids = {r["id"] for r in val_rows}
        rep.add(
            "split_id_disjoint",
            not (train_ids & val_ids),
            {"shared_ids": sorted(train_ids & val_ids)[:10]},
        )
        train_users = {g5_norm(r.get("user", "")) for r in train_rows}
        val_users = {g5_norm(r.get("user", "")) for r in val_rows}
        shared_users = train_users & val_users
        rep.add(
            "split_user_disjoint",
            not shared_users,
            {
                "norm": "g5",
                "shared_user_count": len(shared_users),
                "shared_user_sha256": sorted(
                    hashlib.sha256(u.encode()).hexdigest()[:16]
                    for u in shared_users
                )[:10],
            },
        )

    # ------------------------------------------------------------ denylist
    if denylist_texts is not None and train_rows is not None and val_rows is not None:
        deny_exact = {g5_norm(t) for t in denylist_texts}
        deny_grams: set = set()
        for t in denylist_texts:
            deny_grams |= _g5_ngrams(t)
        hits = {"exact": 0, "ngram": 0}
        for r in train_rows + val_rows:
            user = g5_norm(r.get("user", ""))
            if user in deny_exact:
                hits["exact"] += 1
            elif _g5_ngrams(user) & deny_grams:
                hits["ngram"] += 1
        rep.add(
            "denylist_clean",
            hits == {"exact": 0, "ngram": 0},
            {"denylist_texts": len(denylist_texts), "hits": hits},
        )

    # ---------------------------------------------------------------- bins
    if corpus_dir is not None and provenance is not None:
        expected_sha = expect.get("sha256") or {}
        for split, rows_key in (("train", "train"), ("val", "val")):
            prov = provenance.get(split) or {}
            bin_path = corpus_dir / f"{split}.bin"
            mask_path = corpus_dir / f"{split}.masks.bin"
            if not bin_path.exists() and not mask_path.exists():
                continue
            detail: dict = {}
            ok = True
            if bin_path.exists():
                digest = _sha256_file(bin_path)
                detail["bin_sha256"] = digest
                tokens = bin_path.stat().st_size // 2
                detail["tokens_from_bin"] = tokens
                if prov.get("tokens") is not None and tokens != prov["tokens"]:
                    ok = False
                    detail["provenance_tokens"] = prov["tokens"]
                want = expected_sha.get(f"{split}.bin")
                if want is not None and digest != want:
                    ok = False
                    detail["expected_bin_sha256"] = want
            else:
                ok = False
                detail["bin_missing"] = True
                tokens = 0
            if mask_path.exists():
                raw = mask_path.read_bytes()
                detail["mask_len"] = len(raw)
                detail["mask_supervised"] = sum(raw)
                if len(raw) != tokens:
                    ok = False
                if not set(raw) <= {0, 1}:
                    ok = False
                    detail["mask_values_not_binary"] = True
                if prov.get("supervised_tokens") is not None and sum(raw) != prov["supervised_tokens"]:
                    ok = False
                    detail["provenance_supervised_tokens"] = prov["supervised_tokens"]
                want_m = expected_sha.get(f"{split}.masks.bin")
                if want_m is not None:
                    digest_m = _sha256_file(mask_path)
                    detail["masks_sha256"] = digest_m
                    if digest_m != want_m:
                        ok = False
                        detail["expected_masks_sha256"] = want_m
            else:
                ok = False
                detail["masks_missing"] = True
            rep.add(f"bins_masks_parity[{split}]", ok, detail)

    # ----------------------------------------------------------- reconcile
    cleanup = expect.get("cleanup") or {}
    final = expect.get("final") or {}
    if provenance is not None and cleanup and final:
        detail = {}
        ok = True
        for split in ("train", "val"):
            prov_n = (provenance.get(split) or {}).get("examples")
            want_n = final.get(f"{split}_examples")
            if prov_n is None or want_n is None:
                continue
            drops = int(cleanup.get(f"denylist_exact_{split}", 0))
            overlaps = int(cleanup.get("user_overlaps_removed", 0)) if split == "train" else 0
            reconciled = prov_n - drops - overlaps
            detail[split] = {
                "provenance_examples": prov_n,
                "denylist_exact_dropped": drops,
                "user_overlaps_removed": overlaps,
                "reconciled": reconciled,
                "expected_final": want_n,
            }
            if reconciled != want_n:
                ok = False
        rep.add("cleanup_reconciliation", ok, detail)

    return rep.to_dict()


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--corpus-dir", type=Path, default=None,
                    help="dir with train/val JSONL + bins + provenance.json")
    ap.add_argument("--provenance", type=Path, default=None)
    ap.add_argument("--denylist", type=Path, default=None,
                    help="frozen prompts JSON (fail-closed loader)")
    ap.add_argument("--allow-empty-denylist", action="store_true")
    ap.add_argument("--expect", type=Path, default=None,
                    help="JSON: {sha256: {...}, cleanup: {...}, final: {...}}")
    args = ap.parse_args()

    provenance = None
    prov_path = args.provenance or (
        args.corpus_dir / "provenance.json" if args.corpus_dir else None
    )
    if prov_path is not None and prov_path.exists():
        provenance = json.loads(prov_path.read_text())

    train_rows = val_rows = None
    if args.corpus_dir is not None:
        if (args.corpus_dir / "train.jsonl").exists():
            train_rows = _read_jsonl(args.corpus_dir / "train.jsonl")
        if (args.corpus_dir / "val.jsonl").exists():
            val_rows = _read_jsonl(args.corpus_dir / "val.jsonl")

    denylist_texts = None
    if args.denylist is not None:
        denylist_texts, _schema = load_denylist(
            args.denylist, required=not args.allow_empty_denylist
        )

    expect = json.loads(args.expect.read_text()) if args.expect else None
    report = audit(
        provenance=provenance,
        train_rows=train_rows,
        val_rows=val_rows,
        denylist_texts=denylist_texts,
        corpus_dir=args.corpus_dir,
        expect=expect,
    )
    print(json.dumps(report, indent=2))
    sys.exit(1 if report["release_blockers"] else 0)


if __name__ == "__main__":
    main()
