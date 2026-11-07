#!/usr/bin/env bash
# kaggle/stage_c_train/smoke_s4_env.sh
#
# Preflight checks for Stage C Session 4. Run this before pushing s4.
# Read-only; no mutations, no uploads.
#
# Usage:
#   ./smoke_s4_env.sh                  # uses default INPUT=/kaggle/input/notebooks/galacticaiteam
#   INPUT=/path/to/mount ./smoke_s4_env.sh
#
# Exit code: 0 if all checks pass, 1 if any check fails.

set -uo pipefail

INPUT="${INPUT:-/kaggle/input/notebooks/galacticaiteam}"
TRAIN_SLUG="plm-stage-c-train"
DATA_SLUGS=(plm-stage-c-data-prep-s0 plm-stage-c-data-prep-s1 plm-stage-c-data-prep-s2 plm-stage-c-data-prep-s3 plm-stage-c-data-prep-s4 plm-stage-c-data-prep-s5 plm-stage-c-data-prep-s6 plm-stage-c-data-prep-s7)
EXPECTED_FINAL_STEP=31101
EXPECTED_TRAIN_TOKENS=2857393718
S3_MIN_STEP=28000
S3_MAX_STEP=29500
QUOTA_LIMIT_H=30

FAIL=0
PASS="✓"
BAD="✗"

ok()   { printf "  %s %s\n" "$PASS" "$1"; }
fail() { printf "  %s %s\n" "$BAD" "$1"; FAIL=1; }
hdr()  { printf "\n%s\n" "$1"; }

hdr "Stage C S4 preflight"
printf "INPUT=%s\n" "$INPUT"

# 1. Training kernel mount exists (self-reference for resume).
hdr "1. Training kernel mount"
TRAIN_MOUNT="$INPUT/$TRAIN_SLUG"
if [ ! -d "$TRAIN_MOUNT" ]; then
  fail "train kernel mount not found at $TRAIN_MOUNT"
  echo "     push the train kernel at least once with kernel-metadata-resume.json"
else
  ok "train mount: $TRAIN_MOUNT"
  MANIFEST="$TRAIN_MOUNT/runs/manifest.json"
  if [ ! -f "$MANIFEST" ]; then
    fail "manifest.json not found at $MANIFEST"
  else
    ok "manifest: $MANIFEST"
    LATEST_STEP=$(python3 -c "import json,sys; print(json.load(open('$MANIFEST'))['latest_step'])" 2>/dev/null || echo "")
    if [ -z "$LATEST_STEP" ]; then
      fail "could not parse latest_step from manifest"
    elif [ "$LATEST_STEP" -lt "$S3_MIN_STEP" ] || [ "$LATEST_STEP" -gt "$S3_MAX_STEP" ]; then
      fail "latest_step=$LATEST_STEP outside expected range [$S3_MIN_STEP, $S3_MAX_STEP]"
      echo "     verify s3 exited cleanly before pushing s4"
    else
      ok "latest_step=$LATEST_STEP in [$S3_MIN_STEP, $S3_MAX_STEP]"
    fi

    LATEST_TOKENS=$(python3 -c "import json; print(json.load(open('$MANIFEST')).get('latest_tokens_seen','?'))" 2>/dev/null || echo "?")
    ok "latest_tokens_seen=$LATEST_TOKENS"
  fi

  FP="$TRAIN_MOUNT/runs/train_bin_fingerprint.txt"
  if [ -f "$FP" ]; then
    ok "fingerprint: $FP"
    head -c 200 "$FP" | sed 's/^/     /'
    echo
  else
    fail "fingerprint not found at $FP (Lumi's integration writes this after size check)"
  fi
fi

# 2. Data-prep kernel mounts.
hdr "2. Data-prep mounts (8 sessions)"
for slug in "${DATA_SLUGS[@]}"; do
  d="$INPUT/$slug"
  if [ ! -d "$d" ]; then
    fail "$slug not mounted"
    continue
  fi
  n=$(find "$d" -name 'train.bin' 2>/dev/null | head -1)
  if [ -z "$n" ]; then
    fail "$slug: no train.bin found"
  else
    sz=$(stat -c%s "$n" 2>/dev/null || echo 0)
    tok=$((sz / 2))
    ok "$slug: $n ($tok tokens)"
  fi
done

# 3. Val.bin (session 0 only).
hdr "3. Validation set"
VAL="$INPUT/plm-stage-c-data-prep-s0/stage-c/session-0/final/val.bin"
if [ -f "$VAL" ]; then
  vsz=$(stat -c%s "$VAL" 2>/dev/null || echo 0)
  vtok=$((vsz / 2))
  ok "val.bin: $vtok tokens"
else
  fail "val.bin not found at $VAL"
fi

# 4. Total train tokens (sum across sessions).
hdr "4. Total train tokens (expected $EXPECTED_TRAIN_TOKENS)"
TOTAL=0
for slug in "${DATA_SLUGS[@]}"; do
  n=$(find "$INPUT/$slug" -name 'train.bin' 2>/dev/null | head -1)
  if [ -n "$n" ]; then
    sz=$(stat -c%s "$n" 2>/dev/null || echo 0)
    TOTAL=$((TOTAL + sz / 2))
  fi
done
if [ "$TOTAL" -eq 0 ]; then
  fail "could not total train tokens (no mounts found)"
elif [ "$TOTAL" -ne "$EXPECTED_TRAIN_TOKENS" ]; then
  fail "total=$TOTAL ≠ expected=$EXPECTED_TRAIN_TOKENS"
  echo "     if a data-prep session was rerun, update EXPECTED_TRAIN_TOKENS in this script"
else
  ok "total=$TOTAL matches expected"
fi

# 5. Final step target (from main's stage_c_train.py default).
hdr "5. Final step target"
DEFAULT_STEP=$(python3 -c "
import re
p = 'kaggle/stage_c_train/stage_c_train.py'
try:
    s = open(p).read()
    m = re.search(r'PLM_STAGE_C_MAX_STEPS\", \"(\d+)\"', s)
    print(m.group(1) if m else '')
except Exception:
    print('')
" 2>/dev/null || echo "")
if [ -z "$DEFAULT_STEP" ]; then
  echo "  (skipped: stage_c_train.py not readable from cwd — run from repo root to check)"
elif [ "$DEFAULT_STEP" -eq "$EXPECTED_FINAL_STEP" ]; then
  ok "default PLM_STAGE_C_MAX_STEPS=$DEFAULT_STEP"
else
  fail "default PLM_STAGE_C_MAX_STEPS=$DEFAULT_STEP ≠ expected $EXPECTED_FINAL_STEP"
  echo "     check kaggle/stage_c_train/stage_c_train.py on main"
fi
echo "  reminder: do NOT set PLM_STAGE_C_MAX_STEPS in the s4 env"

# 6. GPU quota (informational; needs kaggle CLI login to query).
hdr "6. GPU quota"
if command -v kaggle >/dev/null 2>&1; then
  echo "  (informational) current quota: check https://www.kaggle.com/settings"
  echo "  s4 expected to use ~2–4 GPU-hours; weekly limit 30h, resets Nov 14"
else
  echo "  (skipped: kaggle CLI not installed)"
fi

# Summary.
hdr "Result"
if [ "$FAIL" -eq 0 ]; then
  echo "  $PASS all checks passed — safe to push s4"
  exit 0
else
  echo "  $BAD one or more checks failed — resolve before pushing s4"
  exit 1
fi