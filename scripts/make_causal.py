#!/usr/bin/env python
"""Measure what each level forces, then name the skill it teaches.

Every number is measured before any label is applied; labelling a level by which
generator produced it would make the taxonomy unfalsifiable.

Three corrections from the v2 run are built in (docs/decisions/0019, 0020):

* **The measurement clock is explicit** (`MEASURE_S`), not inherited from
  `MAX_EPISODE_TIME`. A game clock and a measurement clock need not agree, and
  `demand.survey` silently used whichever was global.
* **Sufficiency is measured over a two-burn family**, sampled over its six
  parameters. The one-burn family won in under 0.3% of its own box on most
  levels, so three of five causal maps were reporting the limits of the probe
  rather than the shape of the level. The one-burn fraction is kept beside it:
  if the gap is large, the family was the constraint.
* **Timing comes from `delay_window`**, which shifts the best plan's start by
  single control ticks (0.07-0.8 s). The previous `window_s` came from a volume
  grid cell of 22.5 s, compared against a 0.2 s human threshold -- a hundredfold
  mismatch that made the `one-exact-boost` branch unreachable at any value.

The candidate mix samples the A axis deliberately. In v2 four of seven sources
were low-thrust specs, where A < 1 occurs 45-80% of the time, so the mix selected
for `gravity-assist` and it took half the candidates.

    python scripts/make_causal.py --out causal_v3 --pool long
"""

import argparse
import json
import time
from pathlib import Path

import jax
import numpy as np

import spacenav.lessons as LS
from spacenav import constants as C
from spacenav import demand as DM
from spacenav import env as E
from spacenav import episodes as EP
from spacenav import reporting
from spacenav.levels.build import index_level
from spacenav.levels.coastpath import CoastConfig, best_coast_task
from spacenav.levels.families import FAMILY_NAMES
from spacenav.opt.necessity import (NeedConfig, ace_profile, adaptivity, boundary_exponent,
                                    necessity_scalars, strategy_volume, two_burn_sufficiency,
                                    volume_scalars)
from spacenav.opt.planner import expand
from spacenav.opt.probe import HUMAN, ProbeConfig, cem, delay_window, fly_from
from spacenav.types import ARRIVED, REASONS

OUT = Path(__file__).resolve().parents[1] / "data" / "curriculum"

# The measurement clock. Independent of MAX_EPISODE_TIME, which is the game's.
MEASURE_S = 150.0
MEASURE_TICKS = int(round(MEASURE_S / C.CTRL_DT))

WIDE = ("binary_star", "star_cluster", "bh_cluster")
# Spread across the A axis on purpose: strong-thrust ordinary specs for the
# high-A bands, contested specs for high eta, and only two low-thrust entries
# for the low-A end (docs/decisions/0020).
SOURCES = (
    ("coast", None, 3),         # Level 0, built by inverting a ballistic arc
    ("1w0m", None, 3),          # A ~ 6: the simplest thing that works
    ("2w50m", None, 3),         # A ~ 2.4
    ("1w100m", None, 3),        # moving target, A ~ 1.5: leading
    ("1w100mc", None, 3),       # contested, strong thrust: high eta, A still > 1
    ("2w100mc", None, 3),       # contested, two stops
    ("1w100mc", WIDE, 2),       # comparable-mass families: eta highest
    ("2w50ml", None, 2),        # low thrust, ordinary placement: fuel-bound
    ("2w100mcl", WIDE, 2),      # the extreme corner
)


def moving_frac(task):
    act = np.asarray(task.wp_active)
    return 0.0 if not act.any() else float((np.asarray(task.wp_anchor)[act] >= 0).mean())


def build_candidates(pool, rng, coast_cfg):
    levels, meta = EP.pool(pool)
    by_family = {}
    for i, m in enumerate(meta):
        by_family.setdefault(m["family"], []).append(i)
    coast = jax.jit(lambda l, k: best_coast_task(k, l, coast_cfg))

    out = []
    for spec, families, n in SOURCES:
        pick = [i for f in (families or by_family) for i in by_family.get(f, [])]
        got, tries = 0, 0
        while got < n and tries < n * 30:
            tries += 1
            idx = int(pick[rng.integers(len(pick))])
            seed = int(rng.integers(1, 10 ** 6))
            if spec == "coast":
                level = index_level(levels, idx)
                task, ok, score = coast(level, jax.random.PRNGKey(seed))
                eid = f"{pool}-{idx}-{seed}-coast"
            else:
                eid = EP.episode_id(pool, idx, seed, spec)
                try:
                    level, task, ok = EP._level_task(eid)
                except Exception:
                    continue
                score = 0.0
            if not bool(ok):
                continue
            s = DM.survey(level, task, MEASURE_S)
            if not s["feasible"]:
                continue
            out.append(dict(eid=eid, spec=spec, level=level, task=task, turns=float(score),
                            wide=families is not None,
                            family=FAMILY_NAMES[int(level.family)],
                            moving_frac=moving_frac(task), **s))
            got += 1
    return out


def fin(x):
    if x is None:
        return None
    x = float(x)
    return None if not np.isfinite(x) else round(x, 4)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default="causal_v3")
    ap.add_argument("--pool", default="long")
    ap.add_argument("--seed", type=int, default=17)
    ap.add_argument("--mc", type=int, default=4096, help="two-burn Monte Carlo draws")
    args = ap.parse_args()

    rng = np.random.default_rng(args.seed)
    ncfg = NeedConfig(ticks=MEASURE_TICKS, n_time=20, n_dir=24, n_dv=6, cell_samples=3)
    fine = ncfg._replace(cell_samples=5)
    pcfg = ProbeConfig(ticks=MEASURE_TICKS, segments=48)
    ccfg = CoastConfig(targets=6, candidates=48, ticks=MEASURE_TICKS)

    print(f"measurement clock {MEASURE_S:.0f}s ({MEASURE_TICKS} ticks); "
          f"game clock {C.MAX_EPISODE_TIME:.0f}s; pool {args.pool}", flush=True)
    cands = build_candidates(args.pool, rng, ccfg)
    print(f"{len(cands)} candidates", flush=True)

    vol_fn = jax.jit(lambda l, t: strategy_volume(l, t, ncfg))
    fine_fn = jax.jit(lambda l, t, k: strategy_volume(l, t, fine, k))
    alpha_fn = jax.jit(lambda l, t, k: boundary_exponent(l, t, k, ncfg))
    tb_fn = jax.jit(lambda l, t, k: two_burn_sufficiency(l, t, k, ncfg, args.mc))
    adapt_fn = jax.jit(lambda l, t, k: adaptivity(l, t, k, ncfg))

    @jax.jit
    def plan_features(level, task, key):
        """One cross-entropy plan, and everything that can be read off it."""
        st = E.reset(level, task)
        seg = cem(level, task, st, key, pcfg, pcfg.ticks, 0)
        acts = expand(seg, pcfg.ticks)
        final, _, pos, alive = fly_from(level, task, st, acts, 0)
        arrived = final.status == ARRIVED
        delays = delay_window(level, task, acts, pcfg.delays)
        ace = ace_profile(level, task, seg, jax.random.fold_in(key, 1), ncfg)
        return dict(arrived=arrived, status=final.status, delays=delays, ace=ace,
                    pos=pos[::8], **necessity_scalars(ace))

    t0 = time.time()
    hdr = (f"{'episode':28s} {'D':>8s} {'A':>6s} {'eta':>6s} {'suf1':>6s} {'suf2':>6s} "
           f"{'win_s':>6s} {'conc':>5s} {'alpha':>6s}  lesson")
    print("\n" + hdr, flush=True)
    for c in cands:
        lv, tk = c["level"], c["task"]
        vol = vol_fn(lv, tk)
        vs = volume_scalars(vol)
        al = alpha_fn(lv, tk, jax.random.PRNGKey(7))
        suf2 = float(tb_fn(lv, tk, jax.random.PRNGKey(11)))
        pf = jax.tree.map(np.asarray, plan_features(lv, tk, jax.random.PRNGKey(5)))
        ad = adapt_fn(lv, tk, jax.random.PRNGKey(9))

        # the human-timing window comes from shifting the plan by single ticks,
        # never from a grid cell
        window = 0.0
        for ok, d in sorted(zip(pf["delays"].tolist(), pcfg.delays), key=lambda p: p[1]):
            if not ok:
                break
            window = d * C.CTRL_DT
        arrived = bool(pf["arrived"])

        c["vol"] = vol
        c["feats"] = dict(
            coast_ok=c["spec"] == "coast", D=c["D"], A=c["A"], eta=c["eta"],
            moving_frac=c["moving_frac"],
            sufficiency=suf2, sufficiency_1burn=float(vs["sufficiency"]),
            leading=float(vs["leading"]), min_dv=float(vs["min_dv"]),
            window_s=window if arrived else 0.0,
            concentration=float(pf["concentration"]) if arrived else None,
            n_critical=float(pf["n_critical"]) if arrived else None,
            mdl=1 if bool(vs["win_any"]) else 99,
            alpha=float(al["alpha"]) if bool(al["measurable"]) else None,
            open_loop=bool(ad["open_loop"]), replanned=bool(ad["replanned"]))
        c["alpha_curve"] = dict(eps=[fin(e) for e in np.asarray(al["eps"])],
                                flip=[fin(f) for f in np.asarray(al["flip"])],
                                win_rate=fin(al["win_rate"]),
                                measurable=bool(al["measurable"]))
        c["plan"] = dict(path=reporting.r1(pf["pos"]), status=REASONS[int(pf["status"])])
        c["ace"] = [fin(x) for x in pf["ace"]]
        c["lesson"], c["tags"] = LS.classify(c["feats"]), LS.tags(c["feats"])
        a = c["feats"]["alpha"]
        cc = c["feats"]["concentration"]
        print(f"{c['eid']:28s} {c['D']:8.4f} {c['A']:6.2f} {c['eta']:6.3f} "
              f"{c['feats']['sufficiency_1burn']:6.3f} {suf2:6.3f} "
              f"{c['feats']['window_s']:6.2f} {('  n/a' if cc is None else f'{cc:5.2f}')} "
              f"{('   n/a' if a is None else f'{a:6.2f}')}  {c['lesson']}", flush=True)
    print(f"\nmeasured in {time.time()-t0:.0f}s", flush=True)

    best = {"free-ride": lambda c: c["turns"],
            "dead-heading": lambda c: c["feats"]["sufficiency"],
            "leading": lambda c: abs(c["feats"]["leading"]),
            "conics": lambda c: c["feats"]["D"],
            "gravity-assist": lambda c: -c["feats"]["A"],
            "rolling-with-it": lambda c: -(c["feats"]["alpha"] or 1.0),
            "one-exact-boost": lambda c: -c["feats"]["window_s"],
            "lottery": lambda c: -(c["feats"]["alpha"] or 1.0)}
    reps = {}
    for c in cands:
        k = c["lesson"]
        if k is None:
            continue
        sc = best[k](c)
        if k not in reps or sc > reps[k][0]:
            reps[k] = (sc, c)
    print("lessons with an example:", ", ".join(sorted(reps)) or "none")
    print("lessons with none:", ", ".join(k for k in LS.ORDER if k not in reps) or "-")

    rows = []
    for key, (_, c) in reps.items():
        v = jax.tree.map(np.asarray, fine_fn(c["level"], c["task"], jax.random.PRNGKey(13)))
        rows.append(dict(
            lesson=key, **LS.describe(key), id=c["eid"], spec=c["spec"], family=c["family"],
            D=fin(c["D"]), A=fin(c["A"]), eta=fin(c["eta"]), turns=fin(c["turns"]),
            features={k2: (fin(v2) if isinstance(v2, float) else v2)
                      for k2, v2 in c["feats"].items()},
            necessity={k2: c["feats"][k2] for k2 in ("concentration", "n_critical")},
            ace=c["ace"], plan=c["plan"], alpha_curve=c["alpha_curve"],
            human_viable=LS.human_viable(c["feats"]),
            volume=dict(win=np.round(np.asarray(v["win"]), 3).tolist(),
                        times=[fin(float(t) * C.CTRL_DT) for t in v["times"]],
                        dirs=[fin(float(np.degrees(d))) for d in v["dirs"]],
                        dvs=[fin(float(x)) for x in v["dvs"]]),
            geometry=reporting.episode_geometry(c["level"], c["task"], MEASURE_TICKS,
                                                body_stride=8)))

    OUT.mkdir(parents=True, exist_ok=True)
    path = OUT / f"{args.out}.json"
    path.write_text(json.dumps(dict(
        syllabus=LS.syllabus(),
        human=dict(window_min=HUMAN.window_min, jitter_pass=HUMAN.jitter_pass,
                   timing_sd=HUMAN.timing_sd, reaction=HUMAN.reaction),
        need_config=ncfg._asdict(), ctrl_dt=C.CTRL_DT, episode_seconds=MEASURE_S,
        game_seconds=C.MAX_EPISODE_TIME, pool=args.pool, two_burn_draws=args.mc,
        lessons=rows,
        survey=[dict(id=c["eid"], spec=c["spec"], family=c["family"], wide=c["wide"],
                     lesson=c["lesson"], tags=list(c["tags"]),
                     alpha_curve=c["alpha_curve"],
                     **{k2: (fin(v2) if isinstance(v2, float) else v2)
                        for k2, v2 in c["feats"].items()}) for c in cands]), indent=1))
    print(f"\nwrote {path} ({path.stat().st_size // 1024} KB, {len(rows)} lessons)")


if __name__ == "__main__":
    main()
