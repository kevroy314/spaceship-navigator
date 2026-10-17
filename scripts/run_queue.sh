#!/bin/bash
# Sequential training queue for one GPU.
#
# One long-lived process per queue rather than one per task: the cluster's 3090
# has historically failed during process startup, so the spare-compute skill
# asks for a single worker that loops.  Each line of the queue file is
#
#     <run-name> <SPACENAV_INSTRUMENTS spec> <seed>
#
# Blank lines and #-comments are skipped, and a run whose eval.jsonl already
# reaches --updates is skipped too, so the queue is resumable after a drop.
#
#   scripts/run_queue.sh queues/ablation.txt 3000 /path/to/logs
set -u
QUEUE="$1"; UPDATES="${2:-3000}"; LOGDIR="${3:-/tmp/spacenav-queue}"
ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
PY="${SPACENAV_PY:-$ROOT/.venv-gpu/bin/python}"
mkdir -p "$LOGDIR"

done_already () {   # run reached the target update?
  local f="$ROOT/data/rl/$1/eval.jsonl"
  [ -f "$f" ] || return 1
  "$PY" - "$f" "$UPDATES" <<'EOF' 2>/dev/null
import json,sys
try:
    r=[json.loads(l) for l in open(sys.argv[1]) if l.strip()]
    sys.exit(0 if r and r[-1]["update"] >= int(sys.argv[2]) else 1)
except Exception:
    sys.exit(1)
EOF
}

echo "=== queue $QUEUE, $UPDATES updates per run, python $PY ==="
while read -r RUN SPEC SEED; do
  case "${RUN:-}" in ""|\#*) continue;; esac
  if done_already "$RUN"; then
    echo "=== skip $RUN (already at $UPDATES updates) ==="; continue
  fi
  echo "=== $(date -Is) start $RUN  instruments=$SPEC seed=$SEED ==="
  SPACENAV_INSTRUMENTS="$SPEC" "$PY" -u "$ROOT/scripts/train.py" \
      --run "$RUN" --updates "$UPDATES" --ckpt-every 100 --seed "$SEED" \
      > "$LOGDIR/$RUN.log" 2>&1
  RC=$?
  echo "=== $(date -Is) end $RUN rc=$RC ==="
  [ $RC -ne 0 ] && tail -5 "$LOGDIR/$RUN.log"
  sleep 20          # let the card settle between runs
done < "$QUEUE"
echo "=== queue complete ==="
