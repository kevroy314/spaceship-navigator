#!/usr/bin/env python
"""Measure what each level forces, then name the skill it teaches.

Order matters: every number is measured before any label is applied.  Naming a
level by which generator produced it would make the taxonomy unfalsifiable.

Three stages, cheapest first:

  1. **demand** (free, closed form) -- D = eta / A.
  2. **the strategy volume and the uncertainty exponent.**  The volume sweeps the
     single-burn family (when x which way x how much delta-v) and holds a win
     *fraction* per cell, not a yes/no: an n-body basin boundary can be fractal,
     and a binary grid would report its own spacing as structure.  `alpha` says
     how resolvable the boundary is at all.
  3. **adaptivity** (expensive) -- does re-planning rescue what one committed
     plan loses?  Run only where it decides something: on the chaotic levels,
     where it separates "chaos you can roll with" from a lottery, and on the
     representative of each lesson.

    python scripts/make_causal.py --out causal_v1

Writes data/curriculum/<out>.json: the syllabus, a survey row per candidate, and
for each lesson one representative with its causal volume, necessity profile and
geometry.
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
                                    necessity_scalars, strategy_volume, volume_scalars)
from spacenav.opt.planner import expand
from spacenav.opt.probe import HUMAN, ProbeConfig, cem, fly_from
from spacenav.types import ARRIVED, REASONS

OUT = Path(__file__).resolve().parents[1] / "data" / "curriculum"

# Candidate sources, spread deliberately.  A lesson no generator can reach should
# show up as an empty slot, not be quietly filled by a near miss.
SOURCES = (
    ("coast", None, 4),          # Level 0, built by inverting a ballistic arc
    ("1w0m", None, 4),           # one stop, strong engine: the simplest thing
    ("1w100m", None, 4),         # one moving stop: leading
    ("1w100mc", None, 3),        # contested placement
    ("1w100mcl", ("binary_star", "star_cluster", "bh_cluster"), 3),
    ("2w100mcl", ("binary_star", "star_cluster", "bh_cluster"), 3),
    ("2w50mcl", None, 3),
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
        while got < n and tries < n * 25:
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
            s = DM.survey(level, task)
            if not s["feasible"]:
                continue
            out.append(dict(eid=eid, spec=spec, level=level, task=task, turns=float(score),
                            family=FAMILY_NAMES[int(level.family)],
                            moving_frac=moving_frac(task), **s))
            got += 1
    return out


def fin(x):
    """JSON-safe float."""
    if x is None:
        return None
    x = float(x)
    return None if not np.isfinite(x) else round(x, 4)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default="causal_v1")
    ap.add_argument("--pool", default="train")
    ap.add_argument("--seed", type=int, default=11)
    args = ap.parse_args()

    rng = np.random.default_rng(args.seed)
    ncfg = NeedConfig(n_time=24, n_dir=24, n_dv=6, cell_samples=3)
    fine = ncfg._replace(cell_samples=5)          # for the representatives
    pcfg = ProbeConfig()
    ccfg = CoastConfig(targets=6, candidates=48)

    cands = build_candidates(args.pool, rng, ccfg)
    print(f"{len(cands)} candidates", flush=True)

    vol_fn = jax.jit(lambda l, t: strategy_volume(l, t, ncfg))
    fine_fn = jax.jit(lambda l, t, k: strategy_volume(l, t, fine, k))
    alpha_fn = jax.jit(lambda l, t, k: boundary_exponent(l, t, k, ncfg))
    adapt_fn = jax.jit(lambda l, t, k: adaptivity(l, t, k, ncfg))

    t0 = time.time()
    print(f"\n{'episode':30s} {'D':>8s} {'A':>7s} {'suff':>6s} {'lead':>6s} {'win_s':>6s} "
          f"{'alpha':>6s}  lesson", flush=True)
    for c in cands:
        vol = vol_fn(c["level"], c["task"])
        sc = {k: v for k, v in volume_scalars(vol).items()}
        al = alpha_fn(c["level"], c["task"], jax.random.PRNGKey(7))
        c["vol_np"] = jax.tree.map(np.asarray, vol)
        c["feats"] = dict(
            coast_ok=c["spec"] == "coast", D=c["D"], A=c["A"], eta=c["eta"],
            moving_frac=c["moving_frac"], sufficiency=float(sc["sufficiency"]),
            leading=float(sc["leading"]), window_s=float(sc["window_s"]),
            min_dv=float(sc["min_dv"]),
            # an exponent fitted to an empty winning set measures nothing, so it
            # is reported as unknown rather than as a fractal boundary
            alpha=float(al["alpha"]) if bool(al["measurable"]) else None,
            mdl=1 if bool(sc["win_any"]) else 99)
        c["alpha_curve"] = dict(eps=[fin(e) for e in np.asarray(al["eps"])],
                                flip=[fin(f) for f in np.asarray(al["flip"])],
                                win_rate=fin(al["win_rate"]),
                                measurable=bool(al["measurable"]),
                                alpha_raw=fin(al["alpha"]))
        c["lesson"], c["tags"] = LS.classify(c["feats"]), LS.tags(c["feats"])
        a = c["feats"]["alpha"]
        print(f"{c['eid']:30s} {c['D']:8.4f} {c['A']:7.2f} "
              f"{c['feats']['sufficiency']:6.3f} {c['feats']['leading']:6.2f} "
              f"{c['feats']['window_s']:6.1f} {'   n/a' if a is None else f'{a:6.2f}'}"
              f"  {c['lesson']}", flush=True)
    print(f"volumes + alpha in {time.time()-t0:.0f}s", flush=True)

    # stage 2b: the intervention profile, only where it can decide a lesson.
    # `one-exact-boost` needs concentration and n_critical, and a window under a
    # human reaction time is a necessary condition for it -- so measure necessity
    # on exactly those candidates rather than all of them. Without this the
    # lesson can never match, which is why it never appeared.
    tight = [c for c in cands
             if 0 < c["feats"]["window_s"] < HUMAN.window_min and c["feats"]["mdl"] < 99]
    print(f"\n{len(tight)} candidates with a window under {HUMAN.window_min}s; "
          f"measuring per-decision necessity", flush=True)
    for c in tight:
        lv, tk = c["level"], c["task"]
        st = E.reset(lv, tk)
        seg = cem(lv, tk, st, jax.random.PRNGKey(5), pcfg, pcfg.ticks, 0)
        final, *_ = fly_from(lv, tk, st, expand(seg, pcfg.ticks), 0)
        if int(final.status) != ARRIVED:
            print(f"  {c['eid']:30s} no arriving plan; necessity undefined", flush=True)
            continue
        a = np.asarray(ace_profile(lv, tk, seg, jax.random.PRNGKey(6), ncfg))
        nsc = {k2: float(v) for k2, v in necessity_scalars(jax.numpy.asarray(a)).items()}
        c["feats"].update(concentration=nsc["concentration"], n_critical=nsc["n_critical"])
        c["ace_pre"] = [fin(x) for x in a]
        c["lesson"], c["tags"] = LS.classify(c["feats"]), LS.tags(c["feats"])
        print(f"  {c['eid']:30s} concentration {nsc['concentration']:.2f} "
              f"n_critical {int(nsc['n_critical'])}  -> {c['lesson']}", flush=True)

    # stage 3: re-planning, where it changes the answer
    chaotic = [c for c in cands if not LS.resolvable(c["feats"])]
    print(f"\n{len(chaotic)} chaotic candidates (alpha < {LS.ALPHA_RESOLVABLE}); "
          f"testing whether re-planning rescues them", flush=True)
    for c in chaotic:
        ad = jax.tree.map(lambda x: bool(x) if x.dtype == bool else float(x),
                          adapt_fn(c["level"], c["task"], jax.random.PRNGKey(9)))
        c["feats"].update(open_loop=ad["open_loop"], replanned=ad["replanned"])
        c["lesson"], c["tags"] = LS.classify(c["feats"]), LS.tags(c["feats"])
        print(f"  {c['eid']:30s} open-loop {str(ad['open_loop'])[0]} "
              f"replanned {str(ad['replanned'])[0]}  -> {c['lesson']}", flush=True)

    # one representative per lesson: the clearest instance of its own signature
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
    print("\nlessons with an example:", ", ".join(sorted(reps)) or "none")
    print("lessons with none:", ", ".join(k for k in LS.ORDER if k not in reps) or "-")

    rows = []
    for key, (_, c) in reps.items():
        lv, tk = c["level"], c["task"]
        st = E.reset(lv, tk)
        ace, nsc, plan = None, {}, None
        seg = cem(lv, tk, st, jax.random.PRNGKey(5), pcfg, pcfg.ticks, 0)
        final, _, pos, alive = fly_from(lv, tk, st, expand(seg, pcfg.ticks), 0)
        plan = dict(path=reporting.r1(np.asarray(pos)[::8]), status=REASONS[int(final.status)])
        if int(final.status) == ARRIVED:
            a = np.asarray(ace_profile(lv, tk, seg, jax.random.PRNGKey(6), ncfg))
            ace = [fin(x) for x in a]
            nsc = {k2: fin(v) for k2, v in necessity_scalars(jax.numpy.asarray(a)).items()}
        if "open_loop" not in c["feats"]:
            ad = jax.tree.map(lambda x: bool(x) if x.dtype == bool else float(x),
                              adapt_fn(lv, tk, jax.random.PRNGKey(9)))
            c["feats"].update(open_loop=ad["open_loop"], replanned=ad["replanned"])
        v = jax.tree.map(np.asarray, fine_fn(lv, tk, jax.random.PRNGKey(13)))
        a_txt = "n/a" if c["feats"]["alpha"] is None else f"{c['feats']['alpha']:.2f}"
        print(f"  {key:16s} plan {plan['status']:9s} concentration "
              f"{nsc.get('concentration')}  alpha {a_txt}", flush=True)
        rows.append(dict(
            lesson=key, **LS.describe(key), id=c["eid"], spec=c["spec"], family=c["family"],
            D=fin(c["D"]), A=fin(c["A"]), eta=fin(c["eta"]), turns=fin(c["turns"]),
            features={k2: (fin(v2) if isinstance(v2, float) else v2)
                      for k2, v2 in c["feats"].items()},
            necessity=nsc, ace=ace, plan=plan, alpha_curve=c["alpha_curve"],
            human_viable=LS.human_viable({**c["feats"], **nsc}),
            volume=dict(win=np.round(np.asarray(v["win"]), 3).tolist(),
                        times=[fin(float(t) * C.CTRL_DT) for t in v["times"]],
                        dirs=[fin(float(np.degrees(d))) for d in v["dirs"]],
                        dvs=[fin(float(x)) for x in v["dvs"]]),
            geometry=reporting.episode_geometry(lv, tk, pcfg.ticks, body_stride=8)))

    OUT.mkdir(parents=True, exist_ok=True)
    path = OUT / f"{args.out}.json"
    path.write_text(json.dumps(dict(
        syllabus=LS.syllabus(),
        human=dict(window_min=HUMAN.window_min, jitter_pass=HUMAN.jitter_pass,
                   timing_sd=HUMAN.timing_sd, reaction=HUMAN.reaction),
        need_config=ncfg._asdict(), ctrl_dt=C.CTRL_DT, lessons=rows,
        survey=[dict(id=c["eid"], spec=c["spec"], family=c["family"], lesson=c["lesson"],
                     tags=list(c["tags"]),
                     **{k2: (fin(v2) if isinstance(v2, float) else v2)
                        for k2, v2 in c["feats"].items()}) for c in cands]), indent=1))
    print(f"\nwrote {path} ({path.stat().st_size // 1024} KB, {len(rows)} lessons)")


if __name__ == "__main__":
    main()
