#!/bin/bash
# One long-lived worker for the 3090: the long-clock runs, then the big-budget
# runs (queues/longclock.txt).  One process that loops, as the spare-compute
# skill asks, because the card has historically failed during process startup.
#
# data/ is not synced to the node, so the long-clock suites are staged on the
# cluster share (/mnt/x/cluster/data/spaceship-navigator/ from WSL) and copied
# in here before anything starts.  Results stay on the node's local disk under
# data/rl/<run>/; the small files (config, train/eval jsonl) plus the final
# checkpoint are tarred back to the share at the end.
set -u
ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
export SPACENAV_PY="$ROOT/.venv-gpu/bin/python"
SHARE=/mnt/cluster/data/spaceship-navigator
LOGS="$ROOT/data/queuelogs"; mkdir -p "$LOGS" "$ROOT/data/eval" "$ROOT/data/pools"

echo "=== $(date -Is) stage suites_long.pkl and long_val.npz from $SHARE ==="
cp -f "$SHARE/suites_long.pkl" "$ROOT/data/eval/suites_long.pkl"
cp -f "$SHARE/long_val.npz" "$ROOT/data/pools/long_val.npz"
for f in data/eval/suites_long.pkl data/pools/long_val.npz data/pools/long.npz data/eval/suites.pkl; do
  if [ ! -s "$ROOT/$f" ]; then echo "=== missing $f, refusing to start ==="; exit 2; fi
done
ls -la "$ROOT/data/eval" "$ROOT/data/pools"
"$SPACENAV_PY" -c "import jax; print('jax backend', jax.default_backend(), jax.devices())"

echo "=== $(date -Is) long-clock queue ==="
"$ROOT/scripts/run_queue.sh" "$ROOT/queues/longclock.txt" 3000 "$LOGS"

echo "=== $(date -Is) tar results to the share ==="
cd "$ROOT" && tar czf "$SHARE/results/evals_long.tgz" \
    $(for r in long600_s0 long600_s1 long600_s2 budget10k_s0 budget10k_s1; do
        for f in config.json train.jsonl eval.jsonl; do [ -f "data/rl/$r/$f" ] && echo "data/rl/$r/$f"; done
        ls data/rl/$r/ckpt_*.msgpack 2>/dev/null | tail -1
      done)
ls -la "$SHARE/results/evals_long.tgz"
echo "=== $(date -Is) ALL DONE ==="
