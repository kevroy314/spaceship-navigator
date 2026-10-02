"""Does solution complexity actually distinguish interesting missions from boring ones?

For a sample of missions from each frozen suite we fly the trained policy and the
scripted pilot, and measure properties of the *solution* rather than the level:
how far the path deviates from a straight line, how many separate burns it takes,
how much of the flight is ballistic, how much delta-v the geometry gave away for
free (Tisserand-style bookkeeping at each flyby), and how much better the agent
does than the naive point-and-burn pilot.

Writes research/out/solution_complexity.json
"""

import json
from pathlib import Path

import jax
import jax.numpy as jnp
import numpy as np
from flax import serialization

from spacenav import baselines as B
from spacenav import constants as C
from spacenav import env as E
from spacenav.rl.evaluate import Evaluator, _gather
from spacenav.rl.network import ACTIONS, ActorCritic

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "research" / "out"
N_PER_SUITE = 200


def load_policy(ckpt):
    net = ActorCritic(d=128)
    ev = Evaluator(net)
    s0 = next(iter(ev.suites.values()))
    lv0 = jax.tree.map(lambda x: x[0], ev.pools[s0["pool"]])
    t0 = jax.tree.map(lambda x: jnp.asarray(x[0]), s0["task"])
    template = net.init(jax.random.PRNGKey(0), E.observe(lv0, t0, E.reset(lv0, t0)))
    params = serialization.from_bytes(template, Path(ckpt).read_bytes())
    return net, params, ev


def metrics(level, task, fin, tr):
    """Per-mission solution descriptors, computed from a rollout trace."""
    status = np.asarray(tr["status"])
    end = int(np.argmax(status != 0)) + 1 if (status != 0).any() else len(status)
    pos = np.asarray(tr["pos"])[:end]
    vel = np.asarray(tr["vel"])[:end]
    thr = np.asarray(tr["thrust"])[:end]
    start = np.asarray(task.start_pos)
    wp = np.asarray(task.wp_pos)[np.asarray(task.wp_active)]
    chain = np.concatenate([start[None], wp])
    tour = float(np.linalg.norm(np.diff(chain, axis=0), axis=-1).sum())

    path = float(np.linalg.norm(np.diff(np.concatenate([start[None], pos]), axis=0), axis=-1).sum())
    burning = thr > 0.05
    burns = int(np.sum(np.diff(burning.astype(int)) == 1) + int(burning[:1].any()))
    # turning of the velocity vector, in units of full turns
    ang = np.arctan2(vel[:, 1], vel[:, 0])
    turning = float(np.abs(np.diff(np.unwrap(ang))).sum() / (2 * np.pi))

    accel = float(task.accel)
    dv_used = float(fin.costs.fuel) * accel
    legs = np.linalg.norm(np.diff(chain, axis=0), axis=-1)
    dv_direct = float(E.deltav_estimate(jnp.asarray(legs), jnp.asarray(np.asarray(task.wp_v_tol)[:len(legs)]),
                                        accel))

    return dict(
        ticks=end, time=float(fin.costs.time), status=int(fin.status),
        tour=tour, path=path, detour=path / max(tour, 1.0),
        burns=burns, duty=float(burning.mean()), ballistic=float(1 - burning.mean()),
        turning=turning, dv_used=dv_used, dv_direct=dv_direct,
        dv_ratio=dv_used / max(dv_direct, 1e-6),
        fuel_frac=float(fin.costs.fuel) / float(task.fuel),
        damage=float(fin.costs.damage), objective=float(E.objective_cost(task, fin.costs)),
        waypoints=int(np.sum(np.asarray(task.wp_active))),
        accel=accel, dv_budget=float(task.fuel) * accel / max(dv_direct, 1e-6),
    )


def main():
    OUT.mkdir(parents=True, exist_ok=True)
    net, params, ev = load_policy(ROOT / "data/rl/ppo_v3/ckpt_03000.msgpack")

    def agent(lv, task):
        def pol(obs, state):
            logits, _ = net.apply(params, obs)
            return ACTIONS[jnp.argmax(logits)]
        return E.rollout(lv, task, pol)
    run_agent = jax.jit(jax.vmap(agent))
    run_pilot = jax.jit(jax.vmap(lambda lv, t: E.rollout(lv, t, B.pilot(lv, t))))

    out = {}
    for name, s in ev.suites.items():
        n = min(N_PER_SUITE, len(s["level_idx"]))
        idx = s["level_idx"][:n]
        lv = _gather(ev.pools[s["pool"]], idx)
        task = jax.tree.map(lambda x: jnp.asarray(x[:n]), s["task"])
        fa, ta = run_agent(lv, task)
        fp, tp = run_pilot(lv, task)
        rows = []
        for i in range(n):
            one = lambda t: jax.tree.map(lambda x: x[i], t)
            ma = metrics(one(lv), one(task), one(fa), one(ta))
            mp = metrics(one(lv), one(task), one(fp), one(tp))
            rows.append(dict(agent=ma, pilot=mp, family=str(s["family"][i])))
        out[name] = rows
        arr = lambda k, who="agent", ok=True: np.array(
            [r[who][k] for r in rows if (r[who]["status"] == 1 or not ok)])
        a_ok = np.array([r["agent"]["status"] == 1 for r in rows])
        p_ok = np.array([r["pilot"]["status"] == 1 for r in rows])
        print(f"{name:7s} agent {a_ok.mean():.0%} pilot {p_ok.mean():.0%} | "
              f"detour {np.median(arr('detour')):.2f} burns {np.median(arr('burns')):.0f} "
              f"ballistic {np.median(arr('ballistic')):.2f} dv/direct {np.median(arr('dv_ratio')):.2f} "
              f"| agent-only levels: {int((a_ok & ~p_ok).sum())}", flush=True)
    (OUT / "solution_complexity.json").write_text(json.dumps(out))
    print("wrote", OUT / "solution_complexity.json")


if __name__ == "__main__":
    main()
