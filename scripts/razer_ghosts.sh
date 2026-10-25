#!/bin/bash
# Ghost fleets only -- the concurrency question is settled (one job saturates
# the card; two is 25% slower in aggregate), so nothing runs wide here.
set -u
ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
PY="$ROOT/.venv-gpu/bin/python"
for spec in "medium 0" "medium 1" "hard 0" "easy 0"; do
  set -- $spec
  echo "######## $1 task $2 ########"
  # abl_info_s0 was trained with eta,stretch; roll it out the same way or the
  # policy reads zeros where it expects its instruments (CLAUDE.md trap).
  SPACENAV_INSTRUMENTS=eta,stretch "$PY" -u "$ROOT/scripts/ghost_fleet.py" \
    --run abl_info_s0 --suite "$1" --task "$2" --rollouts 16 --stride 15 \
    --out "data/ghosts/$1_$2.json" 2>&1 | grep -viE "cuda_exec|does not match" | tail -5
done
echo "######## DONE ########"
ls -la "$ROOT/data/ghosts/"
