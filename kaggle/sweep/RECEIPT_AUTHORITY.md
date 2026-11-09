# Receipt authority

**Origin:** Kaggle kernel version <N> of `galacticaiteam/<slug>`
(`kaggle kernels output galacticaiteam/<slug> -p <local>`).

**Derived copy:** this directory. It is a byte-identical copy of the origin
at the time of writing, and has been verified by
`tools/verify_freeze_receipt.py`.

**If they disagree:**

1. The Kaggle origin wins. Re-download from Kaggle to a fresh directory.
2. Re-run `verify_freeze_receipt.py` on the fresh download.
3. If it passes, replace this directory (rename the old one to
   `<dirname>.stale-<timestamp>` — do not delete).
4. If it fails, the origin itself may be corrupted. Notify the team; do not
   overwrite anything.

**Do not** edit `meta/FREEZE_MANIFEST.json` or `meta/SHA256SUMS.txt` to match
drifted local bytes. The manifest describes the origin; if local bytes
drifted, the local copy is wrong.

**Optimizer state:** `*_state.pt` files are not part of the receipt contract.
Their absence is fine; their presence is not hashed and not verified.