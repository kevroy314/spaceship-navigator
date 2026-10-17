#!/bin/bash
# One long-lived worker for the 3090: the clock sweep, then the ablation queue.
# The spare-compute skill asks for one worker that loops rather than many short
# process starts, because the card has historically failed during startup.
set -u
ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
export SPACENAV_PY="$ROOT/.venv-gpu/bin/python"
LOGS="$ROOT/data/queuelogs"; mkdir -p "$LOGS"
echo "=== $(date -Is) clock sweep ==="
"$ROOT/scripts/clock_sweep.sh" data/curriculum/clock > "$LOGS/clock_sweep.log" 2>&1
echo "=== $(date -Is) clock sweep rc=$? ==="
grep -E "^clock|ARRIVED|outcomes" "$LOGS/clock_sweep.log" || true
echo "=== $(date -Is) ablation queue ==="
"$ROOT/scripts/run_queue.sh" "$ROOT/queues/ablation.txt" 3000 "$LOGS"
echo "=== $(date -Is) ALL DONE ==="
