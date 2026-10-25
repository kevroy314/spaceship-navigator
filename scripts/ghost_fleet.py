"""Every checkpoint of a run flying the same mission, as a fleet of ghost tracks.

The RL analogue of a TrackMania ghost replay: one fixed course, one line per
attempt, coloured by how far through training the policy was when it flew.

Each checkpoint is rolled out `--rollouts` times with the policy *sampling* its
actions rather than taking the argmax, because the spread within one checkpoint
is the thing worth seeing -- an early policy sprays, a trained one bundles.  The
frozen-suite evaluation in `rl/evaluate.py` is greedy and reports one number per
checkpoint; this says what the distribution behind that number looks like.

    python scripts/ghost_fleet.py --run ppo_v5_noinstr --suite medium \
        --task 0 --rollouts 16 --out data/ghosts/v5_medium_0.json
"""

import argparse
import json
import pickle
from pathlib import Path

import jax
import jax.numpy as jnp
import numpy as np
from flax import serialization

from spacenav import constants as C
from spacenav import env as E
from spacenav import physics as P
from spacenav.levels.build import load_pool
from spacenav.rl.evaluate import SUITES_PATH
from spacenav.rl.network import ACTIONS
from spacenav.rl.ppo import PPOConfig, make
from spacenav.types import REASONS

ROOT = Path(__file__).resolve().parents[1]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--run", required=True)
    ap.add_argument("--suite", default="medium")
    ap.add_argument("--task", type=int, default=0, help="index into the frozen suite")
    ap.add_argument("--rollouts", type=int, default=16)
    ap.add_argument("--stride", type=int, default=15, help="keep every Nth tick")
    ap.add_argument("--every", type=int, default=1, help="use every Nth checkpoint")
    ap.add_argument("--out", required=True)
    args = ap.parse_args()

    suites = pickle.load(open(SUITES_PATH, "rb"))
    s = suites[args.suite]
    # s["pool"] is a pool *name*; s["level_idx"] indexes into it, the way
    # rl/evaluate.py's Evaluator does it.
    suite_pool, _ = load_pool(ROOT / "data" / "pools" / f"{s['pool']}.npz")
    lvl_i = int(np.asarray(s["level_idx"])[args.task])
    level = jax.tree.map(lambda x: x[lvl_i], suite_pool)
    task = jax.tree.map(lambda x: jnp.asarray(x)[args.task], s["task"])
    print(f"suite {args.suite}: pool {s['pool']}, level {lvl_i}, "
          f"{int(np.asarray(task.wp_active).sum())} waypoints", flush=True)

    pool, _ = load_pool(ROOT / "data" / "pools" / "train.npz")
    cfg = PPOConfig(total_updates=3000)
    net, runner, _ = make(cfg, pool, jax.random.PRNGKey(0))
    template = runner.params

    def one(params, key):
        def policy(obs, state):
            logits, _ = net.apply(params, obs)
            return ACTIONS[jax.random.categorical(jax.random.fold_in(key, state.tick), logits)]
        return E.rollout(level, task, policy)

    fleet = jax.jit(jax.vmap(one, in_axes=(None, 0)))

    run_dir = ROOT / "data" / "rl" / args.run
    ckpts = sorted(run_dir.glob("ckpt_*.msgpack"))[:: args.every]
    if not ckpts:
        raise SystemExit(f"no checkpoints in {run_dir}")
    print(f"{args.run} / {args.suite} / task {args.task}: {len(ckpts)} checkpoints "
          f"x {args.rollouts} rollouts = {len(ckpts) * args.rollouts} ghosts", flush=True)

    keys = jax.random.split(jax.random.PRNGKey(7), args.rollouts)
    ghosts = []
    for ci, cp in enumerate(ckpts):
        upd = int(cp.stem.split("_")[1])
        params = serialization.from_bytes(template, cp.read_bytes())
        final, tr = fleet(params, keys)
        pos = np.asarray(tr["pos"])            # (R, T, 2)
        st = np.asarray(tr["status"])          # (R, T)
        vis = np.asarray(tr["visited"])
        for r in range(pos.shape[0]):
            nz = np.nonzero(st[r] != 0)[0]
            end = int(nz[0]) + 1 if len(nz) else pos.shape[1]
            ghosts.append(dict(update=upd, roll=r,
                               status=REASONS[int(st[r][end - 1])],
                               visited=int(vis[r][end - 1]),
                               ticks=end,
                               pos=pos[r][:end:args.stride]))
        print(f"  [{ci+1:3d}/{len(ckpts)}] update {upd:5d}  "
              f"arrived {sum(1 for g in ghosts[-pos.shape[0]:] if g['status']=='arrived')}"
              f"/{pos.shape[0]}", flush=True)

    # backdrop: where the bodies and the waypoints actually are over the episode
    # roll_bodies steps at PHYS_DT, so a control tick is SUBSTEPS of them; take
    # every SUBSTEPS-th sample to land back on the ship's tick grid.
    bp0, bv0 = E.snapshot_state(level, task.snapshot)
    bps, bvs = P.roll_bodies(level, bp0, bv0, C.MAX_EPISODE_TICKS * C.SUBSTEPS)
    bps = np.asarray(bps)[:: C.SUBSTEPS]        # (T+1, N, 2) on control ticks
    bvs = np.asarray(bvs)[:: C.SUBSTEPS]
    active = np.asarray(level.active)
    radius = np.asarray(level.radius)
    mass = np.asarray(level.mass)
    kind = np.asarray(level.kind)
    wpos, _ = jax.vmap(lambda p, v: E.waypoint_states(task, p, v))(
        jnp.asarray(bps), jnp.asarray(bvs))
    wpos = np.asarray(wpos)                     # (T+1, K, 2)
    wact = np.asarray(task.wp_active)

    # one common scale: fit everything the page draws into a square box
    allpts = [g["pos"] for g in ghosts if len(g["pos"])]
    pts = np.concatenate(allpts + [bps[:, active, :].reshape(-1, 2)], axis=0)
    lo = pts.min(axis=0)
    hi = pts.max(axis=0)
    span = float(max(hi[0] - lo[0], hi[1] - lo[1])) or 1.0
    cx, cy = (lo + hi) / 2.0

    def q(a):
        """To a 0..1000 integer box, y already screen-down."""
        a = np.asarray(a, np.float64).reshape(-1, 2)
        x = (a[:, 0] - cx) / span * 1000.0 + 500.0
        y = (cy - a[:, 1]) / span * 1000.0 + 500.0
        return np.stack([np.round(x), np.round(y)], -1).astype(int).tolist()

    out = dict(
        run=args.run, suite=args.suite, task=args.task,
        rollouts=args.rollouts, stride=args.stride,
        ctrl_dt=C.CTRL_DT, episode_s=C.MAX_EPISODE_TIME,
        updates=sorted({g["update"] for g in ghosts}),
        scale=dict(span=span, cx=float(cx), cy=float(cy)),
        bodies=[dict(r=float(radius[i] / span * 1000.0),
                     kind=int(kind[i]), mass=float(mass[i]),
                     path=q(bps[:: args.stride * 4, i, :]))
                for i in range(len(active)) if active[i]],
        waypoints=[dict(r=float(np.asarray(task.wp_radius)[k] / span * 1000.0),
                        path=q(wpos[:: args.stride * 4, k, :]))
                   for k in range(len(wact)) if wact[k]],
        ghosts=[dict(u=g["update"], s=g["status"], v=g["visited"],
                     t=g["ticks"], p=q(g["pos"]))
                for g in ghosts],
    )
    p = Path(args.out)
    if not p.is_absolute():
        p = ROOT / p
    p.parent.mkdir(parents=True, exist_ok=True)
    p.write_text(json.dumps(out, separators=(",", ":")))
    arr = sum(1 for g in ghosts if g["status"] == "arrived")
    print(f"\n{len(ghosts)} ghosts, {arr} arrived ({arr/len(ghosts):.0%}), "
          f"{p.stat().st_size/1024:.0f} KB -> {p}")


if __name__ == "__main__":
    main()
