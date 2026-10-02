# spacenav

A 2D n-body space-navigation game and RL testbed on JAX/GPU.  A small ship must
get from A to B through gravitationally bound systems — planetary systems,
Jovian and Saturnian moon systems, binaries, star clusters around a black hole —
while minimising some mix of **time, path length, fuel and exposure** to a
designated region, and surviving radiation, atmospheres and debris.

The old realistic-scale prototype is archived in `legacy/`.

## Quick start

```bash
conda activate spaceship
pip install -e .                      # jax[cuda12]: CUDA 13 dropped Pascal (1080 Ti)
python scripts/build_pools.py         # ~90 s: 12k train levels + validation pools
python -m spacenav.server             # http://<host>:8765 (binds 0.0.0.0)
pytest                                # physics, levels, JS<->JAX parity
python scripts/diagnose.py            # tuning report -> data/reports/
```

**Controls:** ←/→ (A/D) turn · ↑ (W) thrust, Shift for gentle · Space start/pause ·
hold Z to rewind (then fly on from there) · [ ] game speed ¼×–8× · R restart · F follow
ship · P ballistic prediction · wheel zoom · drag pan · H hide panels.  On phones the
same controls are on-screen pads, a time bar (⏪ − speed +), pinch zoom and a ☰ drawer.
Rewound runs stay valid action sequences, so they save and re-simulate like any other.  Episodes are deep-linkable: `#val_seen-230-3-f`
(`pool-level-taskseed-{f|r}`).

## Design

**Units.** G = 1, masses are GM, distances in game units (u), time in seconds.
Levels are ~1000–3000 u across; a typical trip takes 15–40 s.  All tunables are
in `spacenav/constants.py`.

**Physics** (`spacenav/physics.py`). Full O(N²) gravity between up to 32
bodies (uniform-sphere softening inside a radius), kick-drift-kick leapfrog at
60 Hz, 4 substeps per 15 Hz control tick.  The ship is a test particle, so a
level's body motion is fixed by its initial snapshot.  An episode download is
~1.5 KB of JSON: the browser integrates the bodies from that snapshot with the
same leapfrog (float64; within ~0.01–0.2 u of JAX's float32 over 120 s) and
then the ship.  `tests/test_js_parity.py` checks both the browser-integrated and
the JAX-ephemeris paths against JAX.  The server runs JAX on the CPU so it never
queues behind a training run.

**Ship.** Arcade rotate-and-thrust: turn rate 3.5 rad/s, 4 u/s² main engine,
30 s of burn (120 u/s Δv), 100 hull.

**Hazards.** Crashing into a surface is instant death.  Hull damage comes from
radiation (∝ luminosity / r²), atmospheric heating (∝ ρv³, with matching drag,
so aerobraking is a trade), debris (∝ speed relative to the local debris flow —
rings circulate, so matching orbit is cheap), and radiation belts (flat rate).
Sensor zones are harmless but count for the exposure objective.

**Levels** (`spacenav/levels/`). Systems are orbit trees placed in Jacobi
coordinates; stations and clouds are placed Hill-aware.  Every level is
simulated for 300 s on the GPU and rejected on collisions, orbit escapes or
energy drift (> 2e-3); ~98% pass.  Episodes start from one of 10 snapshots.

| split | families | purpose |
|---|---|---|
| `train` | sol_like, jovian, saturnian, binary_star, bh_cluster, asteroid_field, compact_system, debris_maze | training (2000 each) |
| `val_seen` | same families, unseen seeds | generalisation within families |
| `val_holdout` | trinary, pulsar, rogue_flyby | **blind** families never trained on |

**Tasks** (`spacenav/env.py`). A starts in a parking orbit around the locally
dominant subsystem.  B is either a fixed point to fly through or a station to
rendezvous with (≤ 3 u/s relative).  `sample_start` / `sample_target` are
separate so training can hold A fixed for several rounds and then reroll it.
Objective weights are a one-hot or Dirichlet mix and are part of the observation,
so one policy serves every objective.  Reward = −Δ(weighted normalised cost) −
damage + potential shaping ± terminal bonus.

**Observation.** Ego-centric (ship's nose = +x): a self vector (velocity,
local gravity, target geometry, fuel, hull, objective weights) plus one token
per body and per zone with masks, for a set/attention encoder.

## Status (2026-09-21)

* Simulator, level pools, env, scripted pilot baseline, browser game with replay
  overlays, JS↔JAX parity (identical outcomes; ≤0.05 u over 20 s).
* ~2M env-ticks/s at 8k parallel envs on a 1080 Ti (observations included).
* Pilot baseline (`scripts/diagnose.py`): 88–98% fly-through success per family,
  42–93% rendezvous; coasting succeeds ≤5%, so every task needs piloting.
* Known limit: very close passes (< ~30 u) of a black hole get only ~50 steps
  per orbit.

## RL

```bash
python scripts/train.py --run ppo_v1 --updates 2000 --ckpt-every 100
python scripts/make_report.py --run ppo_v1      # -> data/reports/ppo_v1_report.html
```

```bash
python scripts/make_replay.py --run <run>       # animated ghost-fleet replay page
python -m spacenav.trainviz --port 8767         # live training dashboard (SSE, mobile-friendly)
python scripts/references.py                    # best-known flights -> data/eval/references.pkl
python scripts/eval_ckpt.py data/rl/<run>/ckpt_XXXXX.msgpack --tag <name>
```

PPO (PureJaxRL-style, one jitted update) over 1024 parallel ships on the `train`
pool; each ship keeps its level and start A for 4 episodes, then rerolls.  The
action set is the human one (turn ×{left, none, right} × throttle {off, ½, full}).
The network embeds bodies and zones as tokens and pools them with cross-attention
from the ship's own features, so it runs on any body count.  Every checkpoint is
scored on four frozen suites (`data/eval/suites.pkl`: easy / medium / hard
rendezvous / blind held-out families, 600 tasks each) against the scripted pilot;
a drop below an earlier best by more than two binomial standard errors is logged
as a regression.  γ = 0.999 (trips take 20-40 s, so a shorter horizon hides the
arrival bonus); the learning rate and entropy bonus anneal over the run.

**Reference flights** (`spacenav/opt/planner.py`). "Minimum time" needs a
yardstick, so a cross-entropy planner searches piecewise-constant control
sequences, seeded from the scripted pilot and scored on the true objective by
rolling out in the real environment — a reference is always a flight that works.
Gradient-based shooting through the differentiable sim was tried first and
abandoned: over hundreds of ticks of n-body flight the loss surface is dominated
by flyby chaos (250 Adam iterations moved the closest approach by ~5%).

Pascal note: the gradient of `take_along_axis` is a scatter whose kernel will not
load on a 1080 Ti in this jaxlib ("Failed to load in-memory CUBIN"); the loss uses
a one-hot product instead.

**Next:** trajectory-optimiser references per task (so "minimum time" has a
yardstick), PPO with a set encoder, frozen validation task sets, per-checkpoint
evaluation on `val_seen` + `val_holdout`, metrics dashboard.
