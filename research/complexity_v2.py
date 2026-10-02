"""Which solution descriptors actually separate 'interesting' missions from boring ones?

Interesting is operationalised two ways, both policy-independent enough to be useful:
  A) the naive point-and-burn pilot fails but the trained agent succeeds
     (Deceptive-Games style: a greedy trap the cleverer plan escapes), and
  B) both succeed but the agent's weighted cost is much lower.

Descriptors are computed on a SMOOTHED control signal and an arc-length resampled
path, because the raw policy chatters its throttle every tick and any unsmoothed
"number of burns" measures the dithering rather than the flight plan.

Writes research/out/complexity_v2.json
"""

import json
from pathlib import Path

import jax
import jax.numpy as jnp
import numpy as np

from spacenav import baselines as B
from spacenav import env as E
from spacenav.rl.evaluate import _gather
from spacenav.rl.network import ACTIONS
from research.solution_complexity import load_policy

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "research" / "out"
N = 240
SMOOTH_S = 1.0          # seconds of moving average on the throttle


def smooth(x, w):
    if w <= 1:
        return x
    k = np.ones(w) / w
    return np.convolve(x, k, mode="same")


def descriptors(task, fin, tr, n_bodies_pos=None):
    status = np.asarray(tr["status"])
    end = int(np.argmax(status != 0)) + 1 if (status != 0).any() else len(status)
    pos, vel = np.asarray(tr["pos"])[:end], np.asarray(tr["vel"])[:end]
    thr = np.asarray(tr["thrust"])[:end]
    start = np.asarray(task.start_pos)
    wp = np.asarray(task.wp_pos)[np.asarray(task.wp_active)]
    chain = np.concatenate([start[None], wp])
    legs = np.linalg.norm(np.diff(chain, axis=0), axis=-1)
    tour = float(legs.sum())

    w = max(1, int(SMOOTH_S / 0.0667))
    thr_s = smooth(thr, w)
    burning = thr_s > 0.15
    burns = int(np.sum(np.diff(burning.astype(int)) == 1) + int(burning[:1].any()))

    path = float(np.linalg.norm(np.diff(np.concatenate([start[None], pos]), axis=0), axis=-1).sum())
    ang = np.unwrap(np.arctan2(vel[:, 1], vel[:, 0]))
    turning = float(np.abs(np.diff(smooth(ang, w))).sum() / (2 * np.pi))

    accel = float(task.accel)
    dv_used = float(fin.costs.fuel) * accel
    dv_direct = float(E.deltav_estimate(jnp.asarray(legs), jnp.asarray(np.asarray(task.wp_v_tol)[:len(legs)]), accel))
    # speed profile: how much of the flight's speed came from gravity rather than the engine
    speed = np.linalg.norm(vel, axis=-1)
    return dict(
        solved=int(fin.status) == 1, time=float(fin.costs.time),
        detour=path / max(tour, 1.0), burns=burns, turning=turning,
        ballistic=float(1 - burning.mean()),
        dv_ratio=dv_used / max(dv_direct, 1e-6),
        dv_margin=1 - float(fin.costs.fuel) / float(task.fuel),
        speed_gain=float(speed.max() / max(dv_used, 1e-6)),      # >1 means gravity gave speed
        objective=float(E.objective_cost(task, fin.costs)),
        waypoints=int(np.sum(np.asarray(task.wp_active))),
        budget=float(task.fuel) * accel / max(dv_direct, 1e-6),
    )


def main():
    net, params, ev = load_policy(ROOT / "data/rl/ppo_v3/ckpt_03000.msgpack")
    run_agent = jax.jit(jax.vmap(lambda lv, t: E.rollout(
        lv, t, lambda obs, st: ACTIONS[jnp.argmax(net.apply(params, obs)[0])])))
    run_pilot = jax.jit(jax.vmap(lambda lv, t: E.rollout(lv, t, B.pilot(lv, t))))

    rows = []
    for name, s in ev.suites.items():
        n = min(N, len(s["level_idx"]))
        lv = _gather(ev.pools[s["pool"]], s["level_idx"][:n])
        task = jax.tree.map(lambda x: jnp.asarray(x[:n]), s["task"])
        fa, ta = run_agent(lv, task)
        fp, tp = run_pilot(lv, task)
        for i in range(n):
            one = lambda t: jax.tree.map(lambda x: x[i], t)
            a = descriptors(one(task), one(fa), one(ta))
            p = descriptors(one(task), one(fp), one(tp))
            rows.append(dict(suite=name, family=str(s["family"][i]), agent=a, pilot=p,
                             agent_only=bool(a["solved"] and not p["solved"]),
                             both=bool(a["solved"] and p["solved"]),
                             gain=(p["objective"] - a["objective"]) if (a["solved"] and p["solved"]) else None))
    (OUT / "complexity_v2.json").write_text(json.dumps(rows))

    # Which descriptors separate "agent-only" (a greedy trap) from "both solved" (boring)?
    keys = ["detour", "burns", "turning", "ballistic", "dv_ratio", "dv_margin", "speed_gain", "time"]
    ao = [r["agent"] for r in rows if r["agent_only"]]
    bo = [r["agent"] for r in rows if r["both"]]
    print(f"agent-only n={len(ao)}   both-solved n={len(bo)}\n")
    print(f"{'descriptor':12s} {'agent-only':>12s} {'both':>10s} {'ratio':>7s}  {'AUC':>5s}")
    for k in keys:
        x, y = np.array([r[k] for r in ao]), np.array([r[k] for r in bo])
        if not len(x) or not len(y):
            continue
        # AUC = P(descriptor higher on an agent-only level) via Mann-Whitney
        gt = (x[:, None] > y[None, :]).mean() + 0.5 * (x[:, None] == y[None, :]).mean()
        print(f"{k:12s} {np.median(x):12.2f} {np.median(y):10.2f} {np.median(x)/max(np.median(y),1e-9):7.2f}  {gt:5.2f}")
    print("\n(AUC 0.5 = no separation, 1.0 = descriptor perfectly flags the missions the pilot cannot do)")


if __name__ == "__main__":
    main()
