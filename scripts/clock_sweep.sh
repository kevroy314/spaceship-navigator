#!/bin/bash
# Does a longer episode clock make two-stop tours flyable?  (ADR 0025's open question)
# One process per clock, because the clock is a global constant patched at import.
set -u
OUT="${1:-data/curriculum/clock}"
PY="${SPACENAV_PY:-python3}"
ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
mkdir -p "$ROOT/$OUT"
for CK in 150 300 600 900; do
  echo "########## clock ${CK}s ##########"
  "$PY" -u "$ROOT/scripts/twostop_clock.py" --clock "$CK" --per-spec 8 \
        --out "$ROOT/$OUT/clock_${CK}.json" 2>&1 | grep -viE "cuda_exec|does not match"
  echo "########## clock ${CK}s rc=$? ##########"
done
echo "SWEEP COMPLETE"
