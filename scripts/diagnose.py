"""Simulation tuning report.

    python scripts/diagnose.py [--pool val_seen] [--n 128]

For every family in the pool:
  * integrator convergence: pilot action sequences replayed at PHYS_DT/4
  * pilot / coast outcomes for fly-through and rendezvous tasks
  * hazard breakdown (which damage sources dominate)
Writes data/reports/sim_report.json and prints a table.
"""

import argparse
import collections
import json
import time
from pathlib import Path

import jax
import jax.numpy as jnp
import numpy as np

from spacenav import baselines as B
from spacenav import constants as C
from spacenav import env as E
from spacenav import physics as P
from spacenav.levels.build import load_pool
from spacenav.types import REASONS

ROOT = Path(__file__).resolve().parents[1]


def fine_replay(level, task, actions, refine=4):
    """Open-loop replay of per-tick actions with the physics step divided by `refine`."""
    h = C.PHYS_DT / refine
    bp, bv = E.snapshot_state(level, task.snapshot)
    x, v, ang = task.start_pos, task.start_vel, task.start_angle

    def tick(carry, a):
        bp, bv, x, v, ang = carry
        turn, thr = jnp.clip(a[0], -1, 1), jnp.clip(a[1], 0, 1)

        def sub(c, _):
            bp, bv, x, v, ang = c
            ang = ang + turn * C.SHIP_TURN_RATE * h
            acc = thr * task.accel * jnp.stack([jnp.cos(ang), jnp.sin(ang)])
            bp, bv, x, v = P.substep(level, bp, bv, x, v, acc, h)
            return (bp, bv, x, v, ang), None
        carry, _ = jax.lax.scan(sub, (bp, bv, x, v, ang), None, length=C.SUBSTEPS * refine)
        return carry, carry[2]
    _, xs = jax.lax.scan(tick, (bp, bv, x, v, ang), actions)
    return xs


def family_report(levels, idx, cfg, key):
    lv = jax.tree.map(lambda a: a[idx], levels)
    n = len(idx)
    tasks, ok = jax.vmap(lambda k, l: E.sample_task(k, l, cfg))(jax.random.split(key, n), lv)
    out = {"task_ok": float(np.mean(np.asarray(ok)))}
    for name in ("pilot", "coast"):
        pol = getattr(B, name)
        fin, tr = jax.jit(jax.vmap(lambda l, t: E.rollout(l, t, pol(l, t))))(lv, tasks)
        st = np.asarray(fin.status)
        arrived = st == 1
        c = jax.tree.map(np.asarray, fin.costs)
        res = {"outcomes": {REASONS[k]: int(v) for k, v in sorted(collections.Counter(st.tolist()).items())},
               "success": float(arrived.mean())}
        if arrived.any():
            res.update(time=float(c.time[arrived].mean()), fuel=float(c.fuel[arrived].mean()),
                       damage=float(c.damage[arrived].mean()),
                       objective=float(np.asarray(jax.vmap(E.objective_cost)(tasks, fin.costs))[arrived].mean()))
        out[name] = res
        if name == "pilot":
            # convergence: replay the first 40 s of the pilot's actions at 4x finer steps
            ticks = min(600, C.MAX_EPISODE_TICKS)
            acts = jnp.stack([tr["turn"], tr["thrust"]], -1)[:, :ticks]
            fine = jax.jit(jax.vmap(fine_replay))(lv, tasks, acts)
            coarse = tr["pos"][:, :ticks]
            alive = np.asarray(tr["status"][:, :ticks]) == 0
            err = np.linalg.norm(np.asarray(fine - coarse), axis=-1)
            err = np.where(alive, err, 0.0).max(axis=1)
            out["convergence_err"] = {"median": float(np.median(err)), "p90": float(np.percentile(err, 90)),
                                      "max": float(err.max())}
            # damage sources along the pilot's path, while it was flying
            running = (np.asarray(tr["status"]) == 0) | (np.arange(C.MAX_EPISODE_TICKS)[None] == 0)
            parts = np.asarray(tr["damage_parts"])[running].sum(0)
            names = ("radiation", "atmosphere", "debris", "belts")
            out["pilot_damage_share"] = {k: round(float(v / max(parts.sum(), 1e-9)), 2) for k, v in zip(names, parts)}
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--pool", default="val_seen")
    ap.add_argument("--n", type=int, default=128)
    args = ap.parse_args()
    levels, meta = load_pool(ROOT / "data" / "pools" / f"{args.pool}.npz")
    fams = collections.defaultdict(list)
    for i, m in enumerate(meta):
        fams[m["family"]].append(i)
    report = {"constants": {k: getattr(C, k) for k in dir(C) if k.isupper() and not k.startswith("KIND")
                            and not k.startswith("ZONE_") and isinstance(getattr(C, k), (int, float))}}
    t0 = time.time()
    for fam, idx in fams.items():
        idx = np.asarray(idx[: args.n])
        drift = [m["drift"] for m in meta if m["family"] == fam]
        report[fam] = {"energy_drift_p99": float(np.percentile(drift, 99))}
        for mode, cfg in (("flythrough", E.TaskConfig(p_rendezvous=0.0)),
                          ("rendezvous", E.TaskConfig(p_rendezvous=1.0))):
            report[fam][mode] = family_report(levels, idx[: args.n], cfg, jax.random.PRNGKey(123))
        print(f"{fam}: done ({time.time() - t0:.0f}s)", flush=True)
    out = ROOT / "data" / "reports"
    out.mkdir(parents=True, exist_ok=True)
    (out / f"sim_report_{args.pool}.json").write_text(json.dumps(report, indent=2))

    print(f"\n{'family':15s} {'mode':11s} {'pilot':>6s} {'coast':>6s} {'T':>5s} {'fuel':>5s} {'dmg':>5s} "
          f"{'conv p90':>9s}  damage share / failures")
    for fam in fams:
        for mode in ("flythrough", "rendezvous"):
            r = report[fam][mode]
            p = r["pilot"]
            fails = {k: v for k, v in p["outcomes"].items() if k != "arrived"}
            print(f"{fam:15s} {mode:11s} {p['success']:6.0%} {r['coast']['success']:6.0%} "
                  f"{p.get('time', 0):5.1f} {p.get('fuel', 0):5.1f} {p.get('damage', 0):5.1f} "
                  f"{r['convergence_err']['p90']:9.4f}  {r['pilot_damage_share']} {fails}")


if __name__ == "__main__":
    main()
