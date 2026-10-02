"""Compute near-optimal reference flights for a subset of every frozen suite.

    python scripts/references.py [--per-suite 64] [--samples 384] [--iters 18]

For each sampled task the cross-entropy planner searches for the cheapest flight
it can find (seeded from the scripted pilot), and the reference is the better of
that plan and the pilot itself — i.e. the best flight we know of.  An agent's
cost divided by this reference is the honest "how far off optimal" number, and
the reference time is a real answer to "what is the minimum time here".

Writes data/eval/references.pkl.
"""

import argparse
import pickle
import time

import jax
import jax.numpy as jnp
import numpy as np

from spacenav.levels.build import load_pool
from spacenav.opt.planner import PlanConfig, plan_batch
from spacenav.rl.evaluate import DATA, _gather, load_suites
from spacenav.types import ARRIVED


def subset_indices(suite, per_suite):
    n = len(suite["level_idx"])
    stride = max(n // per_suite, 1)
    idx = set(range(0, n, stride))
    idx |= {int(i) for i in suite["showcase"]}      # the report's showcase tasks
    return np.array(sorted(idx))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--per-suite", type=int, default=64)
    ap.add_argument("--samples", type=int, default=384)
    ap.add_argument("--iters", type=int, default=18)
    ap.add_argument("--segments", type=int, default=24)
    ap.add_argument("--batch", type=int, default=16)
    ap.add_argument("--ticks", type=int, default=450)   # 30 s of flight
    args = ap.parse_args()
    cfg = PlanConfig(samples=args.samples, iters=args.iters, segments=args.segments,
                     ticks=args.ticks, elites=max(8, args.samples // 12))
    suites = load_suites()
    pools = {}
    out = {}
    run = jax.jit(lambda l, t, k: plan_batch(l, t, k, cfg))
    for name, s in suites.items():
        pools.setdefault(s["pool"], load_pool(DATA / "pools" / f"{s['pool']}.npz")[0])
        idx = subset_indices(s, args.per_suite)
        task_all = jax.tree.map(jnp.asarray, s["task"])
        res = []
        t0 = time.time()
        for b in range(0, len(idx), args.batch):
            chunk = idx[b: b + args.batch]
            lv = _gather(pools[s["pool"]], s["level_idx"][chunk])
            task = jax.tree.map(lambda x: x[chunk], task_all)
            res.append(jax.device_get(run(lv, task, jax.random.PRNGKey(b))))
            print(f"  {name}: {b + len(chunk)}/{len(idx)} ({time.time() - t0:.0f}s)", flush=True)
        r = {k: np.concatenate([x[k] for x in res]) for k in res[0]}

        pilot_cost = s["pilot"]["objective"][idx]
        pilot_ok = s["pilot"]["status"][idx] == ARRIVED
        plan_ok, plan_cost = r["arrived"], r["cost"]
        use_plan = plan_ok & (~pilot_ok | (plan_cost < pilot_cost))
        cost = np.where(use_plan, plan_cost, np.where(pilot_ok, pilot_cost, np.inf))
        out[name] = dict(
            idx=idx, cost=cost, have=np.isfinite(cost), source=np.where(use_plan, "planner", "pilot"),
            plan_cost=plan_cost, plan_arrived=plan_ok, plan_time=r["time"], plan_damage=r["damage"],
            pilot_cost=pilot_cost, pilot_arrived=pilot_ok,
            pos={int(i): r["pos"][j][: int(r["ticks"][j])].astype(np.float32)
                 for j, i in enumerate(idx) if i in set(int(x) for x in s["showcase"]) and plan_ok[j]},
        )
        beat = np.mean(plan_cost[plan_ok & pilot_ok] < pilot_cost[plan_ok & pilot_ok]) if (plan_ok & pilot_ok).any() else 0
        print(f"{name}: planner arrives {plan_ok.mean():.0%} of {len(idx)}, beats pilot on {beat:.0%}, "
              f"reference available for {np.isfinite(cost).mean():.0%} ({time.time() - t0:.0f}s)", flush=True)
    with open(DATA / "eval" / "references.pkl", "wb") as f:
        pickle.dump(out, f)
    print("wrote", DATA / "eval" / "references.pkl")


if __name__ == "__main__":
    main()
