"""How much does patience alone buy?  A zero-thrust craft, left to coast.

    python scripts/noop_clock.py measure --clock 900                  # the four long suites
    python scripts/noop_clock.py build10x --per-family 16             # a pool validated to 9,000 s
    python scripts/noop_clock.py measure10x --clock 9000              # tasks from that pool
    python scripts/noop_clock.py figure                               # paper/figures/noop.{pdf,png}

The optimizer (docs/decisions/0032) and the reflex pilot (0034) both say what a
longer clock buys a navigator.  This asks the degenerate question underneath:
with no action at all, when does a craft first touch each of its targets, and
how often does it die first?  A task that no-op can finish is a task the clock
alone is scoring.

Per task the rollout records, at physics-step resolution (1/60 s):
  * first contact with each target: the first step the craft is inside the
    target's arrival radius, ignoring the velocity tolerance and the tour order;
  * the environment's own ending (arrived / crashed / destroyed / lost / timeout)
    and the control tick it happened on; "arrived" is the tour as specified,
    in order, velocity tolerance honoured;
  * the closest approach to each target over the whole episode.

The clock is read once, at import of spacenav.constants, so `--clock` is written
to SPACENAV_EPISODE_S before anything from spacenav is imported (the same rule
scripts/build_long_suites.py follows).  Levels in `long_val` are validated to
LEVEL_HORIZON = 900 s from their own t = 0, and a suite task may start at any of
16 snapshots up to 300 s, so a 900 s episode from a late snapshot runs up to
300 s past its validated window; each task's validated remaining time is stored
with the data (`valid_s`) and the report quotes the censored number next to the
raw one.  The 10x pool is validated to 9,000 s with a single snapshot, so every
episode on it is inside the validated window.
"""

import argparse
import json
import os
import pickle
import sys
import time
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
FIG = ROOT / "paper" / "figures"
DATA = ROOT / "data"
SUITE_NAMES = ("easy", "medium", "hard", "blind")
BIG = np.int32(2 ** 30)


def _set_clock(clock):
    env_clock = os.environ.get("SPACENAV_EPISODE_S")
    if env_clock is not None and abs(float(env_clock) - clock) > 1e-6:
        sys.exit(f"SPACENAV_EPISODE_S={env_clock} disagrees with --clock {clock}; set one")
    os.environ["SPACENAV_EPISODE_S"] = repr(float(clock))


# --------------------------------------------------------------------- rollout
def _coast_fn():
    """A jitted, task-batched zero-thrust rollout returning per-task summaries."""
    import jax
    import jax.numpy as jnp
    from spacenav import constants as C
    from spacenav import env as E
    from spacenav.types import RUNNING

    K = C.MAX_WAYPOINTS
    zero = jnp.float32(0.0)

    def coast(level, task):
        s0 = E.reset(level, task)
        wp0, _ = E.waypoint_states(task, s0.body_pos, s0.body_vel)
        d0 = jnp.sqrt(jnp.sum((s0.ship.pos[None] - wp0) ** 2, -1))
        c0 = (s0.body_pos, s0.body_vel, s0.ship, s0.costs, s0.status, s0.visited)
        first = jnp.where(d0 < task.wp_radius, jnp.int32(0), jnp.int32(BIG))
        carry0 = (c0, first, d0, jnp.int32(BIG), jnp.int32(0))

        def sub(i, cc):
            c, first, dmin, tick = cc
            alive = c[4] == RUNNING
            c = E._substep(level, task, zero, zero, c)
            bp, bv, ship = c[0], c[1], c[2]
            wpos, _ = E.waypoint_states(task, bp, bv)
            d = jnp.sqrt(jnp.sum((ship.pos[None] - wpos) ** 2, -1))
            step_no = tick * C.SUBSTEPS + i + 1
            near = alive & (d < task.wp_radius)
            first = jnp.where(near, jnp.minimum(first, step_no), first)
            dmin = jnp.where(alive, jnp.minimum(dmin, d), dmin)
            return c, first, dmin, tick

        def tick_fn(carry, _):
            c, first, dmin, end_tick, tick = carry
            c, first, dmin, _ = jax.lax.fori_loop(0, C.SUBSTEPS, sub, (c, first, dmin, tick))
            bp, bv, ship, costs, status, visited = c
            tick1 = tick + 1
            status = jnp.where((status == RUNNING) & (tick1 >= C.MAX_EPISODE_TICKS),
                               jnp.int32(5), status)
            end_tick = jnp.where((status != RUNNING) & (end_tick == BIG), tick1, end_tick)
            return ((bp, bv, ship, costs, status, visited), first, dmin, end_tick, tick1), None

        carry, _ = jax.lax.scan(tick_fn, carry0, None, length=C.MAX_EPISODE_TICKS)
        (bp, bv, ship, costs, status, visited), first, dmin, end_tick, _ = carry
        return dict(first_step=first, min_dist=dmin, status=status, end_tick=end_tick,
                    visited=visited, health=ship.health, path=costs.path)

    return jax.jit(jax.vmap(coast))


def _run_batches(levels, tasks, chunk, label):
    """Run the coasting rollout over all tasks in chunks of `chunk`; numpy results."""
    import jax
    coast = _coast_fn()
    n = int(np.asarray(tasks.snapshot).shape[0])
    outs = []
    t0 = time.time()
    for a in range(0, n, chunk):
        b = min(n, a + chunk)
        lv = jax.tree.map(lambda x: x[a:b], levels)
        tk = jax.tree.map(lambda x: x[a:b], tasks)
        out = jax.device_get(coast(lv, tk))
        outs.append(out)
        print(f"  {label}: {b}/{n} tasks, {time.time() - t0:.0f} s", flush=True)
    return {k: np.concatenate([o[k] for o in outs]) for k in outs[0]}


def _pack(name, out, tasks, families, valid_s, clock):
    from spacenav import constants as C
    wp_active = np.asarray(tasks.wp_active)
    first_s = np.where(out["first_step"] < BIG, out["first_step"] * C.PHYS_DT, np.inf)
    first_s = np.where(wp_active, first_s, np.nan)
    end_s = out["end_tick"] * C.CTRL_DT
    return {
        f"{name}/first_contact_s": first_s.astype(np.float32),     # (n, K), nan = inactive
        f"{name}/min_dist": np.where(wp_active, out["min_dist"], np.nan).astype(np.float32),
        f"{name}/status": out["status"].astype(np.int8),
        f"{name}/end_s": end_s.astype(np.float32),
        f"{name}/visited": (out["visited"] & wp_active),
        f"{name}/n_targets": wp_active.sum(1).astype(np.int8),
        f"{name}/rendezvous": ((np.asarray(tasks.wp_v_tol) < C.FLYTHROUGH_V_TOL / 2)
                               & wp_active).any(1),
        f"{name}/order_free": np.asarray(tasks.order_free).astype(bool),
        f"{name}/snapshot": np.asarray(tasks.snapshot).astype(np.int16),
        f"{name}/valid_s": valid_s.astype(np.float32),
        f"{name}/family": np.asarray(families).astype(str),
        f"{name}/health": out["health"].astype(np.float32),
        f"{name}/clock_s": np.float32(clock),
    }


def _merge_npz(path, new):
    old = {}
    if path.exists():
        z = np.load(path, allow_pickle=False)
        old = {k: z[k] for k in z.files}
    prefix = {k.split("/")[0] for k in new}
    old = {k: v for k, v in old.items() if k.split("/")[0] not in prefix}
    np.savez_compressed(path, **old, **new)


# -------------------------------------------------------------------- measure
def cmd_measure(args):
    _set_clock(args.clock)
    import jax
    from spacenav import constants as C
    from spacenav.levels.build import load_pool
    suites = pickle.load(open(ROOT / args.suites, "rb"))
    print(f"clock {C.MAX_EPISODE_TIME:.0f} s = {C.MAX_EPISODE_TICKS} ticks; suites {args.suites}",
          flush=True)
    pools = {}
    new = {}
    for name in SUITE_NAMES:
        s = suites[name]
        if s["pool"] not in pools:
            pools[s["pool"]] = load_pool(DATA / "pools" / f"{s['pool']}.npz")[0]
        pool = pools[s["pool"]]
        levels = jax.tree.map(lambda x: x[np.asarray(s["level_idx"])], pool)
        tasks = jax.tree.map(lambda x: np.asarray(x), s["task"])
        # a level is validated to LEVEL_HORIZON from its own t = 0; the task starts
        # at snapshot * SNAPSHOT_INTERVAL, so this much of the episode is validated
        valid_s = C.LEVEL_HORIZON - np.asarray(tasks.snapshot) * C.SNAPSHOT_INTERVAL
        out = _run_batches(levels, tasks, args.chunk, name)
        new.update(_pack(f"suite_{name}", out, tasks, s["family"], valid_s, args.clock))
        _summary(f"suite_{name}", new)
    FIG.mkdir(parents=True, exist_ok=True)
    _merge_npz(FIG / "noop_data.npz", new)
    print(f"wrote {FIG / 'noop_data.npz'}")


def _summary(name, d):
    from spacenav.types import REASONS
    st = d[f"{name}/status"]
    fc = d[f"{name}/first_contact_s"][:, 0]
    n = len(st)
    counts = {REASONS[int(k)]: int(v) for k, v in zip(*np.unique(st, return_counts=True))}
    print(f"  {name}: n={n} endings {counts}; first-target contact by 150 s "
          f"{np.mean(fc <= 150):.3f}, 600 s {np.mean(fc <= 600):.3f}, ever {np.mean(np.isfinite(fc)):.3f}",
          flush=True)


# -------------------------------------------------------------- 10x pool build
HORIZON_10X = 9000.0
LONG10X_SEED0 = 50_000_000    # train 0, val_seen 10M, val_holdout 20M, long 30M, long_val 40M


def cmd_build10x(args):
    # N_SNAPSHOTS and N_CHUNKS in levels/build.py derive from these at import, so
    # both are patched first: a single snapshot at t = 0, validated for 9,000 s.
    _set_clock(HORIZON_10X)
    from spacenav import constants as C
    C.LEVEL_HORIZON = HORIZON_10X
    import jax
    import jax.numpy as jnp
    from spacenav.levels import build as B
    from spacenav.levels.families import TRAIN_FAMILIES
    assert B.N_SNAPSHOTS == 1 and B.N_CHUNKS == 450, (B.N_SNAPSHOTS, B.N_CHUNKS)
    print(f"validating to {C.LEVEL_HORIZON:.0f} s: {B.N_CHUNKS} chunks x {B.STEPS_PER_SNAPSHOT} "
          f"steps, {B.N_SNAPSHOTS} snapshot, batch {args.batch}, {args.per_family}/family "
          f"over {len(TRAIN_FAMILIES)} training families", flush=True)
    accepted, meta, report = [], [], {}
    seed = LONG10X_SEED0
    t_all = time.time()
    for fam in TRAIN_FAMILIES:
        band = (0.3, 3.0) if fam == "rogue_flyby" else B.PARENT_BAND
        reasons = dict(tried=0, collision=0, orbit=0, energy=0, nonfinite=0)
        got = []
        t0 = time.time()
        for _ in range(args.max_tries):
            if len(got) >= args.per_family:
                break
            raws = []
            for _ in range(args.batch):
                raws.append(B.sample_raw(fam, seed)); seed += 1
            stacked = {k: jnp.asarray(np.stack([r[k] for r in raws])) for k in B._ARR_KEYS}
            out = jax.device_get(B._validate_batch(stacked))
            for i, raw in enumerate(raws):
                reasons["tried"] += 1
                if not out["finite"][i]:
                    reasons["nonfinite"] += 1; continue
                if out["margin"][i] <= 0:
                    reasons["collision"] += 1; continue
                if not (band[0] < out["parent_min"][i] and out["parent_max"][i] < band[1]):
                    reasons["orbit"] += 1; continue
                if out["drift"][i] > B.ENERGY_TOL:
                    reasons["energy"] += 1; continue
                if len(got) < args.per_family:
                    raw = dict(raw, snap_pos=out["snap_pos"][i], snap_vel=out["snap_vel"][i])
                    raw["level_radius"] = np.float32(B.level_radius(raw, out["max_r"][i]))
                    got.append(raw)
                    meta.append(dict(family=fam, seed=int(raw["seed"]), n_bodies=int(raw["n_bodies"]),
                                     drift=float(out["drift"][i]), margin=float(out["margin"][i]),
                                     level_radius=float(raw["level_radius"])))
        rej = sum(v for k, v in reasons.items() if k != "tried")
        report[fam] = dict(reasons, accepted=len(got),
                           accept_rate=round(1 - rej / reasons["tried"], 3),
                           seconds=round(time.time() - t0, 1))
        print(f"  {fam}: {len(got)}/{args.per_family} accepted, rate {report[fam]['accept_rate']:.1%} "
              f"over {reasons['tried']} tried {reasons}, {report[fam]['seconds']} s", flush=True)
        accepted += got
    levels = B.stack_levels(accepted)
    B.save_pool(DATA / "pools" / "long10x.npz", levels, meta)
    report["_total"] = dict(levels=len(accepted), seconds=round(time.time() - t_all, 1),
                            horizon_s=HORIZON_10X, seed0=LONG10X_SEED0)
    (DATA / "pools" / "report_long10x.json").write_text(json.dumps(report, indent=2))
    print(f"wrote {DATA / 'pools' / 'long10x.npz'}: {len(accepted)} levels in "
          f"{report['_total']['seconds']} s", flush=True)


def cmd_measure10x(args):
    _set_clock(args.clock)
    import jax
    from spacenav import constants as C
    from spacenav import env as E
    from spacenav.levels.build import load_pool
    from spacenav.rl.evaluate import SUITE_SPECS
    pool, meta = load_pool(DATA / "pools" / "long10x.npz")
    fam_of = np.array([m["family"] for m in meta])
    print(f"clock {C.MAX_EPISODE_TIME:.0f} s = {C.MAX_EPISODE_TICKS} ticks; pool long10x "
          f"({len(meta)} levels, {pool.snap_pos.shape[1]} snapshot)", flush=True)
    new = {}
    for name in SUITE_NAMES:
        spec = SUITE_SPECS[name]
        use = np.nonzero(np.isin(fam_of, spec["families"]))[0] if spec["families"] \
            else np.arange(len(meta))
        sample = jax.jit(jax.vmap(lambda k, l: E.sample_task(k, l, spec["cfg"])))
        key = jax.random.PRNGKey(args.seed * 1000 + sum(map(ord, name)))
        idx, tasks = [], []
        for rep in range(args.tasks_per_level):
            key, sub = jax.random.split(key)
            lv = jax.tree.map(lambda x: x[use], pool)
            t, ok = sample(jax.random.split(sub, len(use)), lv)
            ok = np.asarray(ok)
            for j in np.nonzero(ok)[0]:
                idx.append(use[j]); tasks.append(jax.tree.map(lambda x: np.asarray(x[j]), t))
        idx = np.asarray(idx)
        tasks = jax.tree.map(lambda *xs: np.stack(xs), *tasks)
        levels = jax.tree.map(lambda x: x[idx], pool)
        print(f"  {name}: {len(idx)} tasks on {len(use)} levels", flush=True)
        out = _run_batches(levels, tasks, args.chunk, name)
        valid_s = np.full(len(idx), HORIZON_10X, np.float32)
        new.update(_pack(f"long10x_{name}", out, tasks, fam_of[idx], valid_s, args.clock))
        _summary(f"long10x_{name}", new)
    _merge_npz(FIG / "noop_data.npz", new)
    print(f"wrote {FIG / 'noop_data.npz'}")


# --------------------------------------------------------------------- figure
def _cum(times, grid):
    """Fraction of tasks whose event time is <= each grid point (inf = never)."""
    t = np.sort(times[np.isfinite(times)])
    return np.searchsorted(t, grid, side="right") / len(times)


def cmd_figure(args):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    sys.path.insert(0, str(ROOT / "scripts"))
    import paper_figures as PF           # house style: rcParams and palette

    z = np.load(FIG / "noop_data.npz")
    d = {k: z[k] for k in z.files}
    have10x = any(k.startswith("long10x_") for k in d)
    LBL = {"easy": "Easy", "medium": "Medium", "hard": "Hard", "blind": "Blind"}
    DEAD = (2, 3, 4)   # crashed, destroyed, lost

    def curves(prefix, grid):
        fc = d[f"{prefix}/first_contact_s"]
        st = d[f"{prefix}/status"]
        end = d[f"{prefix}/end_s"]
        any_contact = np.nanmin(np.where(np.isnan(fc), np.inf, fc), axis=1)
        tour = np.where(st == 1, end, np.inf)
        dead = np.where(np.isin(st, DEAD), end, np.inf)
        return _cum(any_contact, grid), _cum(tour, grid), _cum(dead, grid)

    nrow = 2 if have10x else 1
    fig = plt.figure(figsize=(7.0, 2.45 * nrow + 0.5))
    gs = fig.add_gridspec(1, 4, wspace=0.42, top=0.80 if have10x else 0.72,
                          bottom=0.57 if have10x else 0.12)
    grid = np.linspace(0, 900, 901)
    axes = [fig.add_subplot(gs[0, i]) for i in range(4)]
    YMAX = 32

    def spread(items, gap=2.4):
        """Nudge (y, label, color) triples apart so right-edge labels do not overlap."""
        items = sorted(items, key=lambda it: it[0])
        ys = [it[0] for it in items]
        for i in range(1, len(ys)):
            ys[i] = max(ys[i], ys[i - 1] + gap)
        for i in range(len(ys) - 2, -1, -1):          # push back down if the top ran off
            ys[i] = min(ys[i], ys[i + 1] - gap)
        return [(y, it[1], it[2]) for y, it in zip(ys, items)]

    for ax, name in zip(axes, SUITE_NAMES):
        p = f"suite_{name}"
        contact, tour, dead = curves(p, grid)
        n = len(d[f"{p}/status"])
        for x in (150, 600):
            ax.axvline(x, color=PF.GREY, lw=0.5, ls=(0, (2, 2)), zorder=1)
        ax.plot(grid, 100 * dead, color=PF.RED, lw=1.0, zorder=2)
        ax.plot(grid, 100 * contact, color=PF.BLUE, lw=1.4, zorder=3)
        ax.plot(grid, 100 * tour, color=PF.GREEN, lw=1.2, zorder=3)
        # values at the clock (stacked beside the 150 s rule) and at 900 s (right edge)
        for y, (v, c) in zip((YMAX - 2, YMAX - 4.6, YMAX - 7.2),
                             ((f"{100 * contact[150]:.0f}", PF.BLUE),
                              (f"{100 * tour[150]:.0f}", PF.GREEN),
                              (f"{100 * dead[150]:.0f}", PF.RED))):
            ax.text(165, y, v, color=c, fontsize=7, ha="left", va="center")
        for y, v, c in spread([(100 * contact[900], f"{100 * contact[900]:.0f}", PF.BLUE),
                               (100 * tour[900], f"{100 * tour[900]:.0f}", PF.GREEN),
                               (100 * dead[900], f"{100 * dead[900]:.0f}", PF.RED)]):
            ax.text(915, y, v, color=c, fontsize=7, ha="left", va="center", clip_on=False)
        ax.set_xlim(0, 900); ax.set_ylim(0, YMAX)
        ax.set_xticks([0, 150, 600, 900]); ax.set_yticks([0, 10, 20, 30])
        ax.set_title(f"{LBL[name]} (n = {n})", fontsize=8.5)
        ax.set_xlabel("episode time (s)")
        if ax is axes[0]:
            ax.set_ylabel("tasks (%)")
        else:
            ax.set_yticklabels([])
        ax.tick_params(length=2.5)
    # legend as one coloured row under the title, off every panel
    ytop = axes[0].get_position().y1
    for x, txt, c in ((0.13, "any target contacted", PF.BLUE), (0.40, "tour completed", PF.GREEN),
                      (0.62, "crashed, destroyed or lost", PF.RED)):
        fig.text(x, ytop + 0.085, txt, color=c, fontsize=7.5, va="bottom", ha="left")
    fig.suptitle("Zero thrust on the long-clock suites, run to 900 s: contact keeps rising "
                 "past the 150 s clock, and death rises faster", fontsize=9, y=0.985)

    if have10x:
        grid10 = np.concatenate([np.linspace(0, 100, 401)[:-1], np.geomspace(100, 9000, 1500)])
        shades = {"easy": "#9ec5f2", "medium": "#5a9be6", "hard": PF.BLUE, "blind": "#17407a"}
        reds = {"easy": "#f2b0b0", "medium": "#e67a7a", "hard": PF.RED, "blind": "#8a1f1f"}
        axc = fig.add_axes([0.125, 0.09, 0.33, 0.27])
        axd = fig.add_axes([0.60, 0.09, 0.30, 0.27])
        labels_c, labels_d, ns = [], [], []
        for name in SUITE_NAMES:
            p = f"long10x_{name}"
            contact, tour, dead = curves(p, grid10)
            n = len(d[f"{p}/status"])
            axc.plot(grid10, 100 * contact, color=shades[name], lw=1.3, zorder=3)
            axd.plot(grid10, 100 * dead, color=reds[name], lw=1.1, zorder=3)
            labels_c.append((100 * contact[-1], f"{LBL[name]} {100 * contact[-1]:.0f}", shades[name]))
            ns.append(f"{LBL[name].lower()} {n}")
            labels_d.append((100 * dead[-1], f"{LBL[name]} {100 * dead[-1]:.0f}", reds[name]))
        for ax, labels, ymax, gap, title in (
                (axc, labels_c, 16, 1.7, "any target contacted, by suite specification"),
                (axd, labels_d, 100, 7.5, "crashed, destroyed or lost")):
            for x in (150, 600, 900):
                ax.axvline(x, color=PF.GREY, lw=0.5, ls=(0, (2, 2)), zorder=1)
            for y, v, c in spread(labels, gap):
                ax.text(9000 * 1.04, y, v, color=c, fontsize=7, ha="left", va="center", clip_on=False)
            ax.set_xscale("log"); ax.set_xlim(10, 9000); ax.set_ylim(0, ymax)
            ax.set_xticks([10, 30, 100, 300, 900, 3000, 9000])
            ax.set_xticklabels(["10", "30", "100", "300", "900", "3000", "9000"])
            ax.set_xlabel("episode time (s), log scale")
            ax.set_title(title, fontsize=8.5)
            ax.tick_params(length=2.5)
        axc.set_ylabel("tasks (%)")
        axc.set_yticks([0, 5, 10, 15]); axd.set_yticks([0, 25, 50, 75, 100])
        fig.text(0.5, axc.get_position().y1 + 0.075,
                 "Pool validated to 9,000 s, no action for 9,000 s; tasks: " + ", ".join(ns),
                 fontsize=9, ha="center", va="bottom")
    fig.savefig(FIG / "noop.pdf")
    fig.savefig(FIG / "noop.png")
    plt.close(fig)
    print(f"wrote {FIG / 'noop.pdf'} and .png")


def main():
    ap = argparse.ArgumentParser()
    sub = ap.add_subparsers(dest="cmd", required=True)
    m = sub.add_parser("measure", help="zero-thrust rollout on the long suites")
    m.add_argument("--clock", type=float, default=900.0)
    m.add_argument("--suites", default="data/eval/suites_long.pkl")
    m.add_argument("--chunk", type=int, default=150)
    b = sub.add_parser("build10x", help="build data/pools/long10x.npz validated to 9,000 s")
    b.add_argument("--per-family", type=int, default=16)
    b.add_argument("--batch", type=int, default=128)
    b.add_argument("--max-tries", type=int, default=4)
    m10 = sub.add_parser("measure10x", help="zero-thrust rollout on long10x tasks")
    m10.add_argument("--clock", type=float, default=HORIZON_10X)
    m10.add_argument("--tasks-per-level", type=int, default=1)
    m10.add_argument("--chunk", type=int, default=150)
    m10.add_argument("--seed", type=int, default=0)
    sub.add_parser("figure", help="draw paper/figures/noop.{pdf,png} from noop_data.npz")
    args = ap.parse_args()
    {"measure": cmd_measure, "build10x": cmd_build10x,
     "measure10x": cmd_measure10x, "figure": cmd_figure}[args.cmd](args)


if __name__ == "__main__":
    main()
