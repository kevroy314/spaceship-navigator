#!/usr/bin/env python
"""Build a difficulty curve by inverting the dual, then fly both routes on each rung.

The point of a closed-form demand (`spacenav.demand`) is that you do not have to
simulate a level to find out whether it is interesting. So this:

  1. samples candidate episodes cheaply and scores D = eta / A on each, no rollouts;
  2. for each rung of the curve, picks a candidate whose geometry can reach it and
     **solves** for the tank that puts it exactly there (`demand.retune`);
  3. only then plans twice per chosen level — once with patched conics, once with
     the full field — and flies both in the true world.

Step 3 is the expensive part and it runs on a handful of levels instead of
thousands, which is the whole return on having a predictor.

    python scripts/make_curve.py --out curve_v1

Writes data/curriculum/<out>.json: geometry, both routes, and the structure
descriptors (coast_ok, control MDL, action switches) for each rung.
"""

import argparse
import json
import time
from pathlib import Path

import jax
import numpy as np

from spacenav import constants as C
from spacenav import demand as DM
from spacenav import episodes as EP
from spacenav import reporting
from spacenav.levels.families import FAMILY_NAMES
from spacenav.opt.routes import RouteConfig, mdl_steps, pair_batch
from spacenav.types import REASONS

OUT = Path(__file__).resolve().parents[1] / "data" / "curriculum"

# Where to draw candidates from, as (mission spec, families or None for any).
# Contested placement raises eta and a weak engine lowers A, but the top of the
# ladder needs eta near 1, and no single waypoint can deliver that: `terms`
# averages over the leg midpoints too, which sit in open space.  Comparable-mass
# systems are contested *everywhere* (measured: binary_star 0.41, star_cluster
# 0.54 median eta against sol_like's 0.001), so that is where the upper rungs
# come from — not from removing fuel until nothing can fly.
WIDE = ("binary_star", "star_cluster", "bh_cluster")
SPECS = (
    ("1w0m", None), ("2w50m", None),
    ("1w100mc", None), ("2w100mc", None),
    ("1w100mcl", None), ("2w100mcl", None), ("2w50mcl", None),
    ("1w100mc", WIDE), ("1w100mcl", WIDE), ("2w100mcl", WIDE), ("2w50mcl", WIDE),
)


def candidates(pool, specs, n_each, rng):
    """Cheap pass: sample episodes and score the closed-form demand on each."""
    levels, meta = EP.pool(pool)
    n_levels = int(levels.active.shape[0])
    by_family = {}
    for i, m in enumerate(meta):
        by_family.setdefault(m["family"], []).append(i)
    out = []
    for spec, families in specs:
        pick = [i for f in (families or by_family) for i in by_family.get(f, [])]
        got, tries = 0, 0
        while got < n_each and tries < n_each * 20:
            tries += 1
            idx = int(pick[rng.integers(len(pick))]) if pick else int(rng.integers(n_levels))
            seed = int(rng.integers(1, 10 ** 6))
            eid = EP.episode_id(pool, idx, seed, spec)
            try:
                level, task, ok = EP._level_task(eid)
            except Exception:
                continue
            if not ok:
                continue
            s = DM.survey(level, task)
            if not s["feasible"]:
                continue        # the tour overruns the episode clock: infeasible, not hard
            out.append(dict(eid=eid, spec=spec, level=level, task=task,
                            base_fuel=float(task.fuel),
                            family=FAMILY_NAMES[int(level.family)], **s))
            got += 1
    return out


def pick_rungs(cands, rungs, per_rung=1, span=1.7, fuel_floor=1.0):
    """`per_rung` episodes for each rung, each retuned so its demand lands there.

    A level can only be retuned within the ship's tank limits, so the achieved D
    is reported rather than assumed, and a rung with too few candidates close
    enough is left short instead of padded with levels that do not belong there.
    The first pick on each rung is the one the page animates; the rest exist so
    the outcome can be quoted as a rate.
    """
    chosen = []
    used = set()
    for target in rungs:
        scored = []
        for c in cands:
            if c["eid"] in used:
                continue
            tuned = DM.retune(c["level"], c["task"], target)
            got = DM.survey(c["level"], tuned)
            if not got["feasible"]:
                continue
            err = abs(np.log(max(got["D"], 1e-9)) - np.log(target))
            # Never take fuel away.  A rung reached by starving the tank is an
            # infeasible level dressed up as a demanding one, and the candidate
            # statistics show it is unnecessary: contested missions in
            # comparable-mass systems already sit at D ~ 1-3 on the tank the game
            # sized for them.  So demand is reached through geometry, and fuel is
            # only ever *added*, to pull a level down to an easier rung.
            if float(tuned.fuel) < fuel_floor * c["base_fuel"]:
                continue
            if err <= np.log(span):
                scored.append((err, dict(c, task=tuned, **got)))
        # among the levels that can sit on this rung, take the most contested
        # geometry: same D, but reached through eta rather than through hunger,
        # which also leaves the largest tank and so the best chance of flying
        scored.sort(key=lambda p: -p[1]["eta"])
        take = scored[:per_rung]
        if not take:
            print(f"  rung D={target:<8g} empty: nothing within {span}x that keeps "
                  f"{fuel_floor:.0%} of its tank")
            continue
        for n, (_, best) in enumerate(take):
            used.add(best["eid"])
            best["target_D"] = target
            best["rep"] = (n == 0)
            chosen.append(best)
        rep = take[0][1]
        print(f"  rung D={target:<8g} {len(take)} level(s), e.g. {rep['eid']:26s} "
              f"D={rep['D']:.4g} eta={rep['eta']:.3f} A={rep['A']:.2f} {rep['family']}")
    return chosen


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default="curve_v1")
    ap.add_argument("--pool", default="train")
    ap.add_argument("--per-spec", type=int, default=14)
    ap.add_argument("--per-rung", type=int, default=10)
    ap.add_argument("--seed", type=int, default=3)
    args = ap.parse_args()

    rng = np.random.default_rng(args.seed)
    t0 = time.time()
    cands = candidates(args.pool, SPECS, args.per_spec, rng)
    Ds = np.array([c["D"] for c in cands])
    print(f"{len(cands)} feasible candidates scored in {time.time()-t0:.0f}s; "
          f"D from {Ds.min():.2g} to {Ds.max():.2g} (median {np.median(Ds):.2g})", flush=True)
    for spec, families in SPECS:
        sub = [c for c in cands if c["spec"] == spec
               and (families is None or c["family"] in families)]
        if sub:
            tag = spec + ("" if families is None else " [wide]")
            print(f"  {tag:18s} n={len(sub):3d} D median {np.median([c['D'] for c in sub]):.4g}"
                  f"  eta median {np.median([c['eta'] for c in sub]):.3f}")

    print("\npicking one episode per rung:", flush=True)
    chosen = pick_rungs(cands, DM.RUNGS, args.per_rung)
    if not chosen:
        raise SystemExit("no rung could be filled")

    print(f"\nflying {len(chosen)} levels, two routes each...", flush=True)
    cfg = RouteConfig()
    levels = jax.tree.map(lambda *a: np.stack(a), *[c["level"] for c in chosen])
    tasks = jax.tree.map(lambda *a: np.stack(a), *[c["task"] for c in chosen])
    t1 = time.time()
    r = jax.tree.map(np.asarray, jax.jit(lambda l, t, k: pair_batch(l, t, k, cfg))(
        levels, tasks, jax.random.PRNGKey(args.seed)))
    print(f"flown in {time.time()-t1:.0f}s", flush=True)

    rows = []
    for i, c in enumerate(chosen):
        geom = reporting.episode_geometry(c["level"], c["task"], cfg.ticks, body_stride=8)
        routes = {}
        for name in ("approx", "ideal"):
            keep = int(np.sum(r[f"{name}_alive"][i])) + 1
            routes[name] = dict(
                arrived=bool(r[f"{name}_arrived"][i]),
                status=REASONS[int(r[f"{name}_status"][i])],
                cost=None if not np.isfinite(r[f"{name}_cost"][i]) else round(float(r[f"{name}_cost"][i]), 3),
                time=round(float(r[f"{name}_time"][i]), 1),
                fuel=round(float(r[f"{name}_fuel"][i]), 2),
                gap=round(float(min(r[f"{name}_gap"][i], 99)), 2),
                switches=int(r[f"{name}_switches"][i]),
                path=reporting.r1(r[f"{name}_pos"][i][:keep]),
                actions=[int(a) for a in r[f"{name}_actions"][i][:keep]],
            )
        rows.append(dict(
            id=c["eid"], spec=c["spec"], family=c["family"],
            target_D=c["target_D"], D=round(float(c["D"]), 5), eta=round(float(c["eta"]), 4),
            A=round(float(c["A"]), 3), impulse=round(float(c["impulse"]), 3),
            dv=round(float(c["dv"]), 2), T=round(float(c["T"]), 1),
            rung=c["rung"], rung_name=c["rung_name"], rep=bool(c.get("rep")),
            coast_ok=bool(r["coast_ok"][i]), mdl=int(r["mdl"][i]),
            mdl_ok=[bool(x) for x in r["mdl_ok"][i]],
            geometry=geom, routes=routes,
        ))
        a, b = routes["approx"], routes["ideal"]
        print(f"  D={c['D']:<9.4g} approx={a['status']:<9s} ideal={b['status']:<9s} "
              f"mdl={int(r['mdl'][i])} switches={a['switches']}/{b['switches']} "
              f"coast={bool(r['coast_ok'][i])}")

    # the headline: does a coarse plan stop working as the demand rises?
    print("\nby rung:")
    for target in DM.RUNGS:
        grp = [x for x in rows if x["target_D"] == target]
        if not grp:
            continue
        ar = np.mean([x["routes"]["approx"]["arrived"] for x in grp])
        ir = np.mean([x["routes"]["ideal"]["arrived"] for x in grp])
        print(f"  D~{target:<8g} n={len(grp)}  approx arrives {ar:.0%}  ideal arrives {ir:.0%}  "
              f"mdl median {np.median([x['mdl'] for x in grp]):.0f}  "
              f"free rides {sum(x['coast_ok'] for x in grp)}")

    OUT.mkdir(parents=True, exist_ok=True)
    path = OUT / f"{args.out}.json"
    path.write_text(json.dumps(dict(
        rungs=list(DM.RUNGS), rung_names=list(DM.RUNG_NAMES),
        config=cfg._asdict(), ctrl_dt=C.CTRL_DT, stride=cfg.stride,
        mdl_steps=list(mdl_steps(cfg.segments)),
        levels=rows), indent=1))
    print(f"\nwrote {path} ({path.stat().st_size//1024} KB)")


if __name__ == "__main__":
    main()
