"""Does a longer episode clock make two-stop tours flyable?

ADR 0025 measured 0/12 arrivals for two-stop tours at 150 s across three search
budgets, with TIMEOUT dominating, and concluded the family is infeasible at that
clock rather than hard.  It left open whether a longer clock separates "needs
more time" from "needs a better plan".  This runs the same candidate set at
several clocks.

The episode clock is a global constant (spacenav/env.py:446 compares the tick
against C.MAX_EPISODE_TICKS), so it is patched here before anything is traced,
and one clock is run per process.  The candidate set is built with a fixed 150 s
feasibility filter in every run so the set is identical across clocks.

Control resolution is held constant, not segment count: 48 segments per 150 s,
so a 900 s plan gets 288 segments of the same 3.1 s each.  Holding the count
instead would confound "more time" with "coarser control".
"""
import argparse, json, sys

ap = argparse.ArgumentParser()
ap.add_argument("--clock", type=float, required=True)   # seconds
ap.add_argument("--per-spec", type=int, default=8)
ap.add_argument("--out", required=True)
args = ap.parse_args()

# Patch the clock BEFORE importing anything that traces against it.
from spacenav import constants as C
C.MAX_EPISODE_TIME = float(args.clock)
C.MAX_EPISODE_TICKS = int(round(args.clock / C.CTRL_DT))

import jax, numpy as np
from spacenav import demand as DM
from spacenav import episodes as EP
from spacenav.levels.families import FAMILY_NAMES
from spacenav.opt.probe import ProbeConfig, cem, fly_from
from spacenav.opt.planner import expand
from spacenav.types import ARRIVED, REASONS

FILTER_S = 150.0            # the set is fixed; only the flown clock varies
WIDE = ("binary_star", "star_cluster", "bh_cluster")
TWO_STOP = (("2w50m", None), ("2w100mc", None), ("2w50ml", None), ("2w100mcl", WIDE))

TICKS = C.MAX_EPISODE_TICKS
SEGMENTS = max(8, int(round(48 * args.clock / 150.0)))
cfg = ProbeConfig(ticks=TICKS, segments=SEGMENTS, samples=128, elites=16, iters=10)
print(f"clock {args.clock:.0f}s = {TICKS} ticks, {SEGMENTS} segments "
      f"({args.clock/SEGMENTS:.2f}s each), samples {cfg.samples} x iters {cfg.iters}",
      flush=True)

levels, meta = EP.pool("long")
by_family = {}
for i, m in enumerate(meta):
    by_family.setdefault(m["family"], []).append(i)

rng = np.random.default_rng(20251016)      # fixed: identical set in every process
cands = []
for spec, families in TWO_STOP:
    pick = [i for f in (families or by_family) for i in by_family.get(f, [])]
    got, tries = 0, 0
    while got < args.per_spec and tries < args.per_spec * 40:
        tries += 1
        idx = int(pick[rng.integers(len(pick))])
        seed = int(rng.integers(1, 10 ** 6))
        eid = EP.episode_id("long", idx, seed, spec)
        try:
            level, task, ok = EP._level_task(eid)
        except Exception:
            continue
        if not bool(ok):
            continue
        if not DM.survey(level, task, FILTER_S)["feasible"]:
            continue
        cands.append(dict(eid=eid, spec=spec, level=level, task=task,
                          family=FAMILY_NAMES[int(level.family)]))
        got += 1
print(f"{len(cands)} two-stop candidates", flush=True)

plan = jax.jit(lambda l, t, s, k: cem(l, t, s, k, cfg, TICKS))
flyit = jax.jit(lambda l, t, s, seg: fly_from(l, t, s, expand(seg, TICKS)))

import spacenav.env as E
rows = []
for i, c in enumerate(cands):
    st = E.reset(c["level"], c["task"])
    seg = plan(c["level"], c["task"], st, jax.random.PRNGKey(1000 + i))
    final, gap, _, _ = flyit(c["level"], c["task"], st, seg)
    status = int(final.status)
    rows.append(dict(eid=c["eid"], spec=c["spec"], family=c["family"],
                     status=REASONS[status], arrived=status == ARRIVED,
                     gap=float(gap), t=float(final.t),
                     visited=int(np.asarray(final.visited).sum())))
    print(f"  [{i+1:2d}/{len(cands)}] {c['spec']:9s} {REASONS[status]:8s} "
          f"gap {float(gap):6.2f}  visited {rows[-1]['visited']}  t={float(final.t):6.1f}s",
          flush=True)

arr = sum(r["arrived"] for r in rows)
print(f"\nclock {args.clock:.0f}s: ARRIVED {arr}/{len(rows)}")
from collections import Counter
print("  outcomes:", dict(Counter(r["status"] for r in rows)))
print("  mean waypoints visited:", float(np.mean([r['visited'] for r in rows])))
json.dump(dict(clock=args.clock, ticks=TICKS, segments=SEGMENTS,
               arrived=arr, n=len(rows), rows=rows), open(args.out, "w"), indent=1)
