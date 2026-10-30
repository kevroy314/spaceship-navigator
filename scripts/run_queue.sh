#!/bin/bash
# Sequential training queue for one GPU.
#
# One long-lived process per queue rather than one per task: the cluster's 3090
# has historically failed during process startup, so the spare-compute skill
# asks for a single worker that loops.  Each line of the queue file is
#
#     <run-name> <SPACENAV_INSTRUMENTS spec> <seed> [extra ...]
#
# Optional trailing tokens, any order:
#     pool=NAME        train.py --pool NAME            (default train)
#     updates=N        this run's length                (default: the queue's)
#     dmodel=N         train.py --d-model N           (default 128)
#     VAR=VALUE        exported for this run only, e.g. SPACENAV_EPISODE_S=600
#                      SPACENAV_SUITES=data/eval/suites_long.pkl  (key must be
#                      upper-case to be treated as an environment variable)
#
# Blank lines and #-comments are skipped, and a run whose eval.jsonl already
# reaches its update count is skipped too, so the queue is resumable after a drop.
#
#   scripts/run_queue.sh queues/ablation.txt 3000 /path/to/logs
set -u
QUEUE="$1"; UPDATES="${2:-3000}"; LOGDIR="${3:-/tmp/spacenav-queue}"
ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
PY="${SPACENAV_PY:-$ROOT/.venv-gpu/bin/python}"
mkdir -p "$LOGDIR"

done_already () {   # run $1 reached update $2?
  local f="$ROOT/data/rl/$1/eval.jsonl"
  [ -f "$f" ] || return 1
  "$PY" - "$f" "$2" <<'EOF' 2>/dev/null
import json,sys
try:
    r=[json.loads(l) for l in open(sys.argv[1]) if l.strip()]
    sys.exit(0 if r and r[-1]["update"] >= int(sys.argv[2]) else 1)
except Exception:
    sys.exit(1)
EOF
}

echo "=== queue $QUEUE, $UPDATES updates per run, python $PY ==="
while read -r RUN SPEC SEED EXTRA; do
  case "${RUN:-}" in ""|\#*) continue;; esac
  POOL=train; N=$UPDATES; DMODEL=128; ENVS=()
  for tok in ${EXTRA:-}; do
    case "$tok" in
      pool=*)    POOL="${tok#pool=}";;
      updates=*) N="${tok#updates=}";;
      dmodel=*)  DMODEL="${tok#dmodel=}";;
      [A-Z]*=*)  ENVS+=("$tok");;
      *) echo "=== $RUN: unknown token '$tok', skipping run ==="; continue 2;;
    esac
  done
  if done_already "$RUN" "$N"; then
    echo "=== skip $RUN (already at $N updates) ==="; continue
  fi
  echo "=== $(date -Is) start $RUN  instruments=$SPEC seed=$SEED pool=$POOL updates=$N ${ENVS[*]:-} ==="
  env ${ENVS[@]+"${ENVS[@]}"} SPACENAV_INSTRUMENTS="$SPEC" "$PY" -u "$ROOT/scripts/train.py" \
      --run "$RUN" --updates "$N" --ckpt-every 100 --seed "$SEED" --pool "$POOL" --d-model "$DMODEL" \
      > "$LOGDIR/$RUN.log" 2>&1
  RC=$?
  echo "=== $(date -Is) end $RUN rc=$RC ==="
  [ $RC -ne 0 ] && tail -5 "$LOGDIR/$RUN.log"
  sleep 20          # let the card settle between runs
done < "$QUEUE"
echo "=== queue complete ==="
