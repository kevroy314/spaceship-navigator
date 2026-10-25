#!/bin/bash
# One worker: how wide can the 3090 run, then the ghost fleets.
set -u
ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
export SPACENAV_PY="$ROOT/.venv-gpu/bin/python"
PY="$SPACENAV_PY"
echo "######## concurrency: 2 processes ########"
"$ROOT/scripts/gpu_concurrency.sh" 2 120
echo "######## concurrency: 3 processes ########"
"$ROOT/scripts/gpu_concurrency.sh" 3 120
echo "######## ghost fleets ########"
# abl_info_s0 was trained with eta,stretch -- it must be ROLLED OUT that way or
# the policy reads zeros where it expects its two instruments (CLAUDE.md trap).
for spec in "medium 0" "medium 1" "hard 0" "easy 0"; do
  set -- $spec
  echo "---- $1 task $2 ----"
  SPACENAV_INSTRUMENTS=eta,stretch "$PY" -u "$ROOT/scripts/ghost_fleet.py" \
    --run abl_info_s0 --suite "$1" --task "$2" --rollouts 16 --stride 15 \
    --out "data/ghosts/abl_info_s0_$1_$2.json" 2>&1 | grep -viE "cuda_exec|does not match" | tail -6
done
echo "######## ALL DONE ########"
ls -la "$ROOT/data/ghosts/" 2>/dev/null
