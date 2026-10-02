---
name: training-run
description: Use when launching, watching, resuming or reporting a spacenav RL training run — background launch, the live dashboard, ETA checks, resuming after a crash or reboot, and publishing the report and ghost-fleet replay. Covers the traps that silently waste a multi-hour run (Pascal kernel failure, self-killing process cleanup, GPU contention, curriculum too hard to bootstrap).
---

# Running a training job

A run is 3–5 hours on the 1080 Ti. Most of the cost of getting one wrong is the
wall clock, so the order below front-loads the checks that fail fast.

## 1. Before launching

| Check | Command | Why |
|---|---|---|
| Nothing else on the GPU | `nvidia-smi --query-compute-apps=pid,used_memory --format=csv` | A second JAX process halves throughput (seen: 34k → 6k steps/s) |
| Pools match the families | `python scripts/build_pools.py` | Adding a family in `spacenav/levels/families.py` does nothing until pools are rebuilt |
| Suites match the task shape | `python -c "from spacenav.rl.evaluate import build_suites; build_suites()"` | Suites cache tasks **and** the pilot baseline; a changed `TaskConfig` makes old suites meaningless |
| Bands still hold | `python scripts/build_curriculum.py --out v<n> --n 8` | The probe decides which levels are human-flyable and which are agent-only. A changed `TaskConfig`, fuel band or family set moves the bands, and the game's curriculum browser reads the saved JSON |
| Difficulty is sane | run the pilot on each tier | Aim for pilot success ~90 / 50 / 15 / 40% across easy/medium/hard/blind — enough headroom to show learning, not so much the reward never fires |

## 2. Launch, in the background

```bash
conda run --no-capture-output -n spaceship python scripts/train.py \
  --run ppo_v4 --updates 3000 --ckpt-every 100 --log-every 10
```

Use the harness's background mode, never `&`. Output: `data/rl/<run>/` holds
`config.json`, `train.jsonl`, `eval.jsonl`, `ckpt_XXXXX.msgpack`,
`state_latest.msgpack` (params + optimiser state, for resuming) and
`traj_XXXXX.pkl` (showcase trajectories + per-task costs).

## 3. Watch it

```bash
conda run --no-capture-output -n spaceship python -m spacenav.trainviz --port 8767
```

`http://<lan-ip>:8767` — progress, ETA, eval suites against the pilot, live
charts, updating over SSE. It reads only the JSONL files and imports no JAX, so
it stays responsive while the GPU is saturated. Check `ss -ltn` before picking a
port: **8765** is the game server, **8766** is the local nginx, 8767 was free.

To check from the shell instead: `curl -s localhost:8767/api/runs | python3 -m json.tool`.

**Judge a run early.** By ~15M steps a healthy run is well clear of zero on the
easy suite (v2: 99% at 13M). If it is still near zero, stop and fix the task
rather than paying for the full run.

## 4. When it stops unexpectedly

```bash
python scripts/train.py --run <same-run> --updates <same> --resume
```

Resume restores `state_latest.msgpack` (params + Adam state) and continues the
update counter, so the curriculum and LR schedule stay aligned. Runs from before
that file existed fall back to the newest `ckpt_*.msgpack` with a fresh Adam
state — survivable, but it shows as a dip for a few checkpoints.

## 5. Publish the result

```bash
python scripts/references.py --per-suite 40 --samples 512 --iters 20   # ~3 min/suite, GPU
python scripts/make_report.py --run <run> --note "anything unusual about this run"
python scripts/make_replay.py --run <run>
```

Then publish with the Artifact tool. **Pass the existing artifact URL** to update
the same page instead of creating a second one; the report and the replay are
separate artifacts with their own URLs (see the project memory).

## Traps

| Symptom | Cause |
|---|---|
| A long job prints nothing for ten minutes and looks hung | `conda run` buffers **all** stdout until the process exits. Call the env's python directly — `/home/kevin/anaconda3/envs/spaceship/bin/python script.py > log 2>&1 &` — and poll the log |
| `probe`/`build_curriculum` sits at 100% CPU and ~5% GPU for minutes | That is XLA compiling, not running: the probe holds a dozen nested scans. It compiles once per batch shape, so keep `--batch` fixed across the run and let the first batch pay the cost |
| A reflex baseline suddenly fails most tasks | Check `deltav_budget` before anything else. The default `TaskConfig` carries the *training* band (down to 0.55x the direct-flight estimate) and a policy that does not ration fuel cannot fly that. Human missions use (0.9, 1.45) |
| The session itself dies mid-command, or a task exits 137/144 | `pkill -f <pat>` / `pgrep -f <pat>` matches the shell running it, because `-f` sees its own argv. Kill by port (`fuser -k 8767/tcp`) or by a PID captured earlier |
| `INTERNAL: CUDA error: Failed to load in-memory CUBIN` during the first update | A scatter kernel (the gradient of `take_along_axis`) has no Pascal build in this jaxlib. The PPO loss uses a one-hot product instead — keep it that way |
| Dashboard returns HTTP 500 `Out of range float values are not JSON compliant` | A rate with no episodes behind it logs as NaN. `trainviz.clean()` and `train.to_py()` replace non-finite values with null |
| Training success sits at ~2% for tens of millions of steps | The task distribution is too hard from step one. Ramp it: `PPOConfig.curriculum()` widens tour length, rendezvous share, fuel budget and thrust over the first half. v3 went from 2% at 13M to 62% at 17M purely from ramping |
| Throughput collapses to a third | Another JAX process (game server, planner, report build) is sharing the GPU. The game server pins itself to CPU via `JAX_PLATFORMS=cpu`; do the same for report builds |
| Gradients come out all-NaN in the planner | Differentiating `sqrt` at r = 0 (a body against itself) or `sqrt(m/r)` with m = 0 (a massless debris anchor). `physics.py` guards the second; `opt/planner.py` stops the gradient through body motion |
| New run's numbers look incomparable to the last | Rebuilding pools or suites changes the tasks. Re-score the old checkpoint with `scripts/eval_ckpt.py <ckpt> --tag <name>` so the report can show a fair previous-run line |

## Interpreting the eval line

`[ckpt 1100] easy=92%/win54% medium=47%/win31% hard=14%/win10% blind=34%/win19%`

`win` is against the scripted pilot on the same tasks: the agent arrives where
the pilot fails, or both arrive and the agent's weighted cost is lower. A rising
success rate with a flat win rate means the agent is getting *there*, not getting
*better*. A drop on `blind` while the seen suites keep climbing is the
overfitting signal — that is what the regression pills in the report track.
