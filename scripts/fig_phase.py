"""Phase-space portraits of four level families, computed on real levels.

One column per family (sol_like, jovian, binary_star, star_cluster), one
representative level each from `data/pools/long.npz`.  Three rows, each a
computed field or a computed sweep, never a schematic:

  1. eta       the share of the local field a one-body model discards, on a
               dense grid over the playable disc at the episode-start snapshot,
               with the equal-pull shells (d = a*sqrt(m/M)) drawn on top.
  2. FTLE      a 20 s finite-time Lyapunov exponent on the same grid: a test
               particle on the local circular parking orbit is displaced by
               eps in x and in y, the three copies are integrated against the
               true ephemeris, and the exponent is ln(sigma_max(F))/T with F
               the deformation gradient.  Measured, not the tidal closed form
               (which is also stored, as `stretch`).
  3. basin     the single-burn win fraction over (burn time x burn heading)
               from one fixed task on the level, via opt.necessity's jittered
               strategy volume, marginalised over burn size.

    python scripts/fig_phase.py compute --pool data/pools/long.npz --out paper/figures
    python scripts/fig_phase.py render  --data paper/figures/phase_data.npz

`compute` writes phase_data.npz (and nothing else); `render` draws phase.pdf
and phase.png from it without touching the environment, so the figure can be
restyled without the GPU.  The basin sweep is the only expensive part
(n_time x n_dir x n_dv x cell_samples rollouts of a full episode per level);
`--basin-res` scales it down for a local smoke test.
"""
import argparse
import json
import os
import pathlib
import time

import numpy as np

ROOT = pathlib.Path(__file__).resolve().parents[1]
FAMILIES = ["sol_like", "jovian", "binary_star", "star_cluster"]
FTLE_T = 20.0          # s, the horizon of the finite-time exponent
FTLE_EPS = 0.5         # u, the displacement used for the deformation gradient


# ======================================================================= compute
def eta_at(P, level, bp, pts):
    import jax.numpy as jnp
    full = P.gravity_at(pts, bp, level, 0)
    dom = P.gravity_at(pts, bp, level, 1)
    num = jnp.sqrt(jnp.sum((full - dom) ** 2, -1))
    den = jnp.sqrt(jnp.sum(full * full, -1)) + 1e-12
    return num / den, den


def dominant_at(level, bp, pts):
    """Index of the strongest attractor at each point: the one-body model's choice.
    Its boundaries on the grid are the equal-pull surface, measured rather than
    taken from the parent/child closed form, so a binary's own dividing curve
    and the Apollonius circles around planets both appear."""
    import jax.numpy as jnp
    d = bp[None] - pts[:, None]
    r = jnp.sqrt(jnp.sum(d * d, -1))
    soft = jnp.maximum(jnp.maximum(r, level.radius[None]), 1e-6)
    m = jnp.where(level.active, level.mass, 0.0)
    return jnp.argmax(m[None] / soft ** 2 * (r / soft), axis=-1)


def stretch_at(level, bp, pts):
    """sqrt(sum 2 GM / r^3): the tidal closed form `instrument_panel` reports."""
    import jax.numpy as jnp
    d = bp[None] - pts[:, None]
    r = jnp.sqrt(jnp.sum(d * d, -1))
    r = jnp.maximum(r, jnp.maximum(level.radius, 1.0)[None])
    m = jnp.where(level.active, level.mass, 0.0)
    return jnp.sqrt(jnp.sum(2.0 * m[None] / r ** 3, -1))


def ftle_at(level, bp, bv, pts, T=FTLE_T, eps=FTLE_EPS):
    """Finite-time Lyapunov exponent over T seconds from the parking orbit at pts.

    Three copies per point (base, +eps x, +eps y) share one body ephemeris and
    are integrated with the same kick-drift-kick leapfrog the ship uses, no
    thrust and no drag.  F = [dX/dx0] by finite difference; exponent is
    ln(sigma_max(F)) / T.  Positive everywhere in a gravitational field; the
    *contrast* across the disc is the portrait.
    """
    import jax
    import jax.numpy as jnp
    from spacenav import constants as C
    from spacenav import env as E
    from spacenav import physics as P
    h = C.PHYS_DT
    n_steps = int(round(T / h))
    M = pts.shape[0]
    v0 = jax.vmap(lambda x: E.parking_velocity(level, bp, bv, x, 1.0))(pts)
    X0 = jnp.concatenate([pts, pts + jnp.array([eps, 0.0]), pts + jnp.array([0.0, eps])], 0)
    V0 = jnp.concatenate([v0, v0, v0], 0)

    def f(carry, _):
        b_p, b_v, X, V = carry
        a0 = P.gravity_at(X, b_p, level, 0)
        Vh = V + 0.5 * h * a0
        X1 = X + h * Vh
        b_p1, b_v1 = P.bodies_step(b_p, b_v, level, h)
        a1 = P.gravity_at(X1, b_p1, level, 0)
        return (b_p1, b_v1, X1, Vh + 0.5 * h * a1), None

    (_, _, X, _), _ = jax.lax.scan(f, (bp, bv, X0, V0), None, length=n_steps)
    Fx = (X[M:2 * M] - X[:M]) / eps
    Fy = (X[2 * M:] - X[:M]) / eps
    # Cauchy-Green tensor C = F^T F, 2x2 symmetric; largest eigenvalue closed form
    a = Fx[:, 0] ** 2 + Fx[:, 1] ** 2
    d = Fy[:, 0] ** 2 + Fy[:, 1] ** 2
    b = Fx[:, 0] * Fy[:, 0] + Fx[:, 1] * Fy[:, 1]
    lam_max = 0.5 * (a + d) + jnp.sqrt(jnp.maximum(0.25 * (a - d) ** 2 + b * b, 0.0))
    return jnp.log(jnp.sqrt(jnp.maximum(lam_max, 1e-12))) / T


def pick_levels(pool, meta, families, n_pts=512):
    """One level per family: the one whose disc-median eta is nearest the
    family's median of that quantity, so the column is typical, not extreme.
    Done in float64 numpy on the host: the ranking has near-ties, and a float32
    device median flipped the star_cluster pick between the 1080 Ti and the 3090."""
    rs = np.random.RandomState(0)
    rr = np.sqrt(rs.uniform(0.02, 1.0, n_pts)); th = rs.uniform(0, 2 * np.pi, n_pts)
    unit = np.stack([rr * np.cos(th), rr * np.sin(th)], -1)               # (P, 2)
    bp = np.asarray(pool.snap_pos[:, 0], np.float64)                        # (L, N, 2)
    m = np.where(np.asarray(pool.active), np.asarray(pool.mass, np.float64), 0.0)
    rad = np.asarray(pool.radius, np.float64)
    R = np.asarray(pool.level_radius, np.float64)
    pts = unit[None] * R[:, None, None]                                     # (L, P, 2)
    d = bp[:, None] - pts[:, :, None]                                       # (L, P, N, 2)
    r = np.sqrt(np.sum(d * d, -1))
    soft = np.maximum(np.maximum(r, rad[:, None]), 1e-6)
    w = m[:, None] / soft ** 3                                              # (L, P, N)
    full = np.sum(w[..., None] * d, 2)
    pull = w * r
    top = np.argmax(pull, -1)
    dom = np.take_along_axis(w[..., None] * d, top[..., None, None], 2)[:, :, 0]
    e = np.linalg.norm(full - dom, axis=-1) / (np.linalg.norm(full, axis=-1) + 1e-12)
    e = np.median(e, 1)                                                     # (L,)
    picks, fam_med = {}, {}
    for fam in families:
        idx = np.array([i for i, mt in enumerate(meta) if mt["family"] == fam])
        med = float(np.median(e[idx]))
        picks[fam] = int(idx[np.argmin(np.abs(e[idx] - med))])
        fam_med[fam] = med
    return picks, fam_med, e


def pick_task(level, base_seed, seeds=64):
    """First sampled single-target task (default TaskConfig) that is valid and
    that the scripted pilot flies to arrival: the solvability gate, so the
    basin map is of a level somebody can fly.  `base_seed` differs per family;
    the loadout draws do not depend on the level, so one shared seed would hand
    every column the same engine and the same start snapshot."""
    import jax
    from spacenav import baselines
    from spacenav import env as E
    from spacenav.types import ARRIVED
    cfg = E.TaskConfig()
    sample = jax.jit(lambda k: E.sample_task(k, level, cfg))
    fly = jax.jit(lambda t: E.rollout(level, t, baselines.pilot(level, t))[0].status)
    for s in range(seeds):
        task, ok = sample(jax.random.PRNGKey(base_seed + s))
        if not bool(ok):
            continue
        if int(fly(task)) == ARRIVED:
            return task, s
    raise RuntimeError("no pilot-solvable task found in %d seeds" % seeds)


def compute(args):
    os.environ.setdefault("XLA_PYTHON_CLIENT_PREALLOCATE", "false")
    import jax
    import jax.numpy as jnp
    from spacenav import constants as C
    from spacenav import demand
    from spacenav import env as E
    from spacenav import physics as P
    from spacenav.levels.build import load_pool
    from spacenav.opt.necessity import NeedConfig, strategy_volume, volume_scalars
    from spacenav.types import ARRIVED

    print("backend", jax.default_backend(), flush=True)
    pool, meta = load_pool(args.pool)
    t0 = time.time()
    picks, fam_med, all_eta = pick_levels(pool, meta, FAMILIES)
    print("picked", picks, "family medians", {k: round(v, 4) for k, v in fam_med.items()},
          f"({time.time() - t0:.0f} s)", flush=True)

    res, fres = args.res, args.ftle_res
    out = {"families": np.array(FAMILIES), "res": res, "ftle_res": fres, "ftle_T": FTLE_T, "ftle_eps": FTLE_EPS,
           "node": os.environ.get("CLUSTER_NODE", "local"), "backend": jax.default_backend()}
    # the sweep's heading axis: a full circle in equal cells, the last cell one
    # step short of wrapping onto the first.  The parking-orbit velocity at the
    # start offsets the winning heading from the bearing, so the +-60 degree
    # window necessity.py uses for its own purposes clips the winning set here.
    span = np.radians(args.dir_span_deg) * (1.0 - 1.0 / args.n_dir if args.dir_span_deg >= 360 else 1.0)
    cfg = NeedConfig(n_time=args.n_time, n_dir=args.n_dir, n_dv=args.n_dv,
                     cell_samples=args.cell_samples, dir_span=float(span))
    vol_j = jax.jit(strategy_volume, static_argnames=("cfg",))
    eta_j = jax.jit(lambda lv, bp, pts: eta_at(P, lv, bp, pts))
    str_j = jax.jit(stretch_at)
    ftle_j = jax.jit(ftle_at, static_argnames=("T",))
    dom_j = jax.jit(dominant_at)

    for fi, fam in enumerate(FAMILIES):
        i = picks[fam]
        level = jax.tree.map(lambda x: x[i], pool)
        task, seed = pick_task(level, 1000 + 100 * fi)
        snap = int(task.snapshot)
        bp, bv = E.snapshot_state(level, snap)
        R = float(level.level_radius)
        # closed-form fields on a fine grid (free), the integrated FTLE on a coarser one
        xs = np.linspace(-R, R, res)
        gx, gy = np.meshgrid(xs, xs)
        pts = jnp.asarray(np.stack([gx.ravel(), gy.ravel()], -1), jnp.float32)
        fxs = np.linspace(-R, R, fres)
        fgx, fgy = np.meshgrid(fxs, fxs)
        fpts = jnp.asarray(np.stack([fgx.ravel(), fgy.ravel()], -1), jnp.float32)

        t0 = time.time()
        eta, gmag = eta_j(level, bp, pts)
        stretch = str_j(level, bp, pts)
        dom = dom_j(level, bp, pts)
        ftle = ftle_j(level, bp, bv, fpts, T=FTLE_T)
        ftle60 = ftle_j(level, bp, bv, fpts, T=3 * FTLE_T)
        stretch_f = str_j(level, bp, fpts)
        eta, gmag, stretch, dom = (np.asarray(a).reshape(res, res) for a in (eta, gmag, stretch, dom))
        ftle, ftle60, stretch_f = (np.asarray(a).reshape(fres, fres) for a in (ftle, ftle60, stretch_f))
        t_fields = time.time() - t0

        # closed-form demand of the chosen task, and the task geometry
        tm = demand.survey(level, task)
        st = E.reset(level, task)
        tpos, _ = E.target_state(task, st)
        has, r_eq, hill, _ = E.contested_zone(level, bp)

        t0 = time.time()
        vol = vol_j(level, task, cfg=cfg)
        vol = jax.tree.map(np.asarray, vol)
        sc = jax.tree.map(float, volume_scalars(jax.tree.map(jnp.asarray, vol)))
        t_basin = time.time() - t0
        bearing = float(np.arctan2(*(np.asarray(tpos) - np.asarray(task.start_pos))[::-1]))

        # statistics the records can be checked against
        rr = np.sqrt(gx ** 2 + gy ** 2)
        inside = rr <= R
        act = np.asarray(level.active)
        bpn = np.asarray(bp); rad = np.asarray(level.radius)
        in_body = np.zeros_like(inside)
        for j in np.where(act & (rad > 0))[0]:
            in_body |= np.hypot(gx - bpn[j, 0], gy - bpn[j, 1]) < rad[j]
        ok = inside & ~in_body & np.isfinite(eta)
        frr = np.sqrt(fgx ** 2 + fgy ** 2)
        f_in_body = np.zeros(fgx.shape, bool)
        for j in np.where(act & (rad > 0))[0]:
            f_in_body |= np.hypot(fgx - bpn[j, 0], fgy - bpn[j, 1]) < rad[j]
        fok = (frr <= R) & ~f_in_body & np.isfinite(ftle) & np.isfinite(ftle60)
        shell = []
        for j in np.where(np.asarray(has))[0]:
            dist = np.hypot(gx - bpn[j, 0], gy - bpn[j, 1]) / float(r_eq[j])
            shell.append(eta[ok & (np.abs(dist - 1.0) < 0.15)])
        shell = np.concatenate(shell) if shell else np.array([np.nan])
        edge = np.zeros_like(ok)
        edge[1:, :] |= dom[1:, :] != dom[:-1, :]
        edge[:, 1:] |= dom[:, 1:] != dom[:, :-1]
        edge &= ok
        stats = dict(level=i, seed=int(meta[i]["seed"]), task_seed=seed, snapshot=snap,
                     eta_disc_median=float(np.median(eta[ok])), eta_shell_median=float(np.nanmedian(shell)),
                     n_shells=int(np.asarray(has).sum()),
                     eta_boundary_median=float(np.median(eta[edge])) if edge.any() else float("nan"),
                     ftle_disc_median=float(np.median(ftle[fok])), ftle_p90=float(np.percentile(ftle[fok], 90)),
                     ftle60_disc_median=float(np.median(ftle60[fok])),
                     ftle60_p90=float(np.percentile(ftle60[fok], 90)),
                     stretch_disc_median=float(np.median(stretch[ok])),
                     corr_log_ftle_log_stretch=float(np.corrcoef(np.log(np.maximum(ftle[fok], 1e-6)),
                                                                 np.log(stretch_f[fok]))[0, 1]),
                     corr_log_ftle60_log_stretch=float(np.corrcoef(np.log(np.maximum(ftle60[fok], 1e-6)),
                                                                   np.log(stretch_f[fok]))[0, 1]),
                     D=tm["D"], A=tm["A"], eta_task=tm["eta"], T=tm["T"], accel=float(task.accel),
                     fuel=float(task.fuel), sufficiency=sc["sufficiency"], window_s=sc["window_s"],
                     leading=sc["leading"], min_dv=sc["min_dv"], t_fields_s=t_fields, t_basin_s=t_basin,
                     rollouts=int(cfg.n_time * cfg.n_dir * cfg.n_dv * cfg.cell_samples))
        print(fam, json.dumps({k: (round(v, 4) if isinstance(v, float) else v) for k, v in stats.items()}),
              flush=True)

        out[f"{fam}/R"] = R
        out[f"{fam}/eta"] = eta.astype(np.float32)
        out[f"{fam}/gmag"] = gmag.astype(np.float32)
        out[f"{fam}/stretch"] = stretch.astype(np.float32)
        out[f"{fam}/ftle"] = ftle.astype(np.float32)
        out[f"{fam}/ftle60"] = ftle60.astype(np.float32)
        out[f"{fam}/dom"] = dom.astype(np.int16)
        out[f"{fam}/body_pos"] = bpn[act]
        out[f"{fam}/body_radius"] = rad[act]
        out[f"{fam}/body_mass"] = np.asarray(level.mass)[act]
        out[f"{fam}/body_kind"] = np.asarray(level.kind)[act]
        out[f"{fam}/shell_pos"] = bpn[np.asarray(has)]
        out[f"{fam}/shell_r"] = np.asarray(r_eq)[np.asarray(has)]
        out[f"{fam}/start"] = np.asarray(task.start_pos)
        out[f"{fam}/start_vel"] = np.asarray(task.start_vel)
        out[f"{fam}/target"] = np.asarray(tpos)
        out[f"{fam}/target_r"] = float(task.wp_radius[0])
        out[f"{fam}/bearing"] = bearing
        out[f"{fam}/win"] = vol["win"].astype(np.float32)         # (n_time, n_dir, n_dv)
        out[f"{fam}/times_s"] = vol["times"].astype(np.float32) * C.CTRL_DT
        out[f"{fam}/dirs"] = vol["dirs"].astype(np.float32)
        out[f"{fam}/dvs"] = vol["dvs"].astype(np.float32)
        out[f"{fam}/stats"] = json.dumps(stats)

    outdir = pathlib.Path(args.out); outdir.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(outdir / "phase_data.npz", **out)
    print("wrote", outdir / "phase_data.npz", flush=True)


# ======================================================================== render
def render(args):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from matplotlib.colors import LogNorm, Normalize
    from matplotlib.cm import ScalarMappable
    from matplotlib.patches import Circle

    BLUE, ORANGE, GREEN, RED, GREY = "#2a78d6", "#eb6834", "#0ca30c", "#d03b3b", "#6b7280"
    plt.rcParams.update({
        "figure.dpi": 150, "savefig.dpi": 200, "savefig.bbox": "tight",
        "font.family": "serif", "font.serif": ["DejaVu Serif"], "font.size": 8.5,
        "axes.labelsize": 8.5, "axes.titlesize": 9, "legend.fontsize": 8,
        "xtick.labelsize": 7.5, "ytick.labelsize": 7.5,
        "axes.spines.top": False, "axes.spines.right": False,
        "axes.linewidth": 0.6, "xtick.major.width": 0.6, "ytick.major.width": 0.6,
        "lines.solid_capstyle": "round", "pdf.fonttype": 42,
    })
    z = np.load(args.data, allow_pickle=True)
    fams = [str(f) for f in z["families"]]
    stats = {f: json.loads(str(z[f"{f}/stats"])) for f in fams}
    LABEL = {"sol_like": "sol-like", "jovian": "jovian", "saturnian": "saturnian",
             "binary_star": "binary star", "star_cluster": "star cluster"}

    # one scale per row, shared across columns.  eta spans three decades by
    # construction; the FTLE scale is set from the pooled data so the four
    # columns share it; the win fraction is a probability.
    eta_norm = LogNorm(1e-3, 1.0)
    ftle_all = np.concatenate([z[f"{f}/ftle"][np.isfinite(z[f"{f}/ftle"])] for f in fams])
    ftle_norm = LogNorm(max(np.percentile(ftle_all, 1), 1e-3), np.percentile(ftle_all, 99.7))
    win_norm = Normalize(0, 1)
    cm_eta, cm_ftle, cm_win = plt.get_cmap("viridis"), plt.get_cmap("magma"), plt.get_cmap("Blues")

    ncol = len(fams)
    fig = plt.figure(figsize=(7.0, 6.5))
    gs = fig.add_gridspec(3, ncol + 1, width_ratios=[1] * ncol + [0.045],
                          wspace=0.10, hspace=0.32, left=0.10, right=0.925, top=0.905, bottom=0.075)
    axes = np.array([[fig.add_subplot(gs[r, c]) for c in range(ncol)] for r in range(3)])
    cax = [fig.add_subplot(gs[r, ncol]) for r in range(3)]

    def spatial(ax, f, field, norm, cmap, shells):
        R = float(z[f"{f}/R"])
        res = field.shape[0]
        disc = Circle((0, 0), R, facecolor="none", edgecolor="#c9ced9", lw=0.5, zorder=2)
        ax.add_patch(disc)
        im = ax.imshow(field, origin="lower", extent=(-R, R, -R, R), norm=norm, cmap=cmap,
                       interpolation="nearest", zorder=1)
        im.set_clip_path(disc)
        if shells:
            # the equal-pull surface, measured: where the strongest attractor changes
            dom = z[f"{f}/dom"]
            xs = np.linspace(-R, R, dom.shape[0])
            for j in np.unique(dom):
                ind = (dom == j).astype(float)
                if 0 < ind.sum() < ind.size:
                    cs = ax.contour(xs, xs, ind, levels=[0.5], colors="white", linewidths=0.8,
                                    linestyles="dashed", zorder=4)
                    for coll in getattr(cs, "collections", [cs]):
                        coll.set_clip_path(disc)
                        coll.set_dashes([(0, (2.2, 1.6))])
        for c, r in zip(z[f"{f}/body_pos"], z[f"{f}/body_radius"]):
            # true radius, filled; a thin ring at a visible size so a 2 u moon is findable
            ax.add_patch(Circle(c, max(float(r), 1e-3), facecolor="white", edgecolor="#555",
                                lw=0.3, zorder=5))
            if r < 0.012 * R:
                ax.add_patch(Circle(c, 0.012 * R, facecolor="none", edgecolor="white", lw=0.5, zorder=5))
        s, t = z[f"{f}/start"], z[f"{f}/target"]
        ax.plot([s[0]], [s[1]], marker="^", ms=5, color=ORANGE, mec="white", mew=0.5, zorder=6, ls="none")
        ax.add_patch(Circle(t, max(float(z[f"{f}/target_r"]), 0.02 * R), facecolor="none",
                            edgecolor=GREEN, lw=1.3, zorder=6))
        ax.set_xlim(-R * 1.03, R * 1.03); ax.set_ylim(-R * 1.03, R * 1.03)
        ax.set_aspect("equal"); ax.set_xticks([]); ax.set_yticks([])
        for sp in ax.spines.values():
            sp.set_visible(False)

    for c, f in enumerate(fams):
        st = stats[f]
        spatial(axes[0, c], f, z[f"{f}/eta"], eta_norm, cm_eta, shells=True)
        spatial(axes[1, c], f, z[f"{f}/ftle"], ftle_norm, cm_ftle, shells=False)
        axes[0, c].set_title(LABEL.get(f, f), fontsize=9.5, pad=14)
        axes[0, c].text(0.5, 1.035, rf"median $\eta$ = {st['eta_disc_median']:.2g}",
                        transform=axes[0, c].transAxes, ha="center", va="bottom", fontsize=7.5, color="#444")
        axes[1, c].text(0.5, 1.02, f"median {st['ftle_disc_median'] * 1e3:.1f} " + r"$\times 10^{-3}$ s$^{-1}$",
                        transform=axes[1, c].transAxes, ha="center", va="bottom", fontsize=7.5, color="#444")

        # basin map: burn time (s) on y, heading relative to the bearing (deg) on x.
        # Each burn size wins along its own thin stripe at its own lead angle, so
        # the mean over burn size interleaves four sparse stripes; the question the
        # row asks is which (time, heading) reaches the target at all, which is the
        # best burn size.  Headings are pooled in pairs (1 deg cells -> 2 deg) so a
        # one-cell stripe survives rasterisation; the pooled cell takes the max.
        win = z[f"{f}/win"].max(axis=2)
        t_s = z[f"{f}/times_s"]
        d = np.degrees(z[f"{f}/dirs"] - z[f"{f}/bearing"])
        d = (d + 180.0) % 360.0 - 180.0                   # centre the axis on the bearing
        order = np.argsort(d); d, win = d[order], win[:, order]
        if win.shape[1] % 2 == 0 and win.shape[1] >= 180:
            win = win.reshape(win.shape[0], -1, 2).max(axis=2)
            d = d.reshape(-1, 2).mean(axis=1)
        plane_frac = float(win.mean())
        ax = axes[2, c]
        ax.imshow(win, origin="lower", aspect="auto", norm=win_norm, cmap=cm_win, interpolation="nearest",
                  extent=(d[0] - (d[1] - d[0]) / 2, d[-1] + (d[1] - d[0]) / 2,
                          t_s[0] - (t_s[1] - t_s[0]) / 2, t_s[-1] + (t_s[1] - t_s[0]) / 2))
        ax.axvline(0, color=GREY, lw=0.5, ls=":", zorder=3)
        # the heading the ship is already moving along at t = 0 (its parking orbit)
        v = z[f"{f}/start_vel"]
        vh = np.degrees(np.arctan2(v[1], v[0]) - z[f"{f}/bearing"])
        vh = (vh + 180.0) % 360.0 - 180.0
        ax.plot([vh], [t_s[-1] + (t_s[1] - t_s[0]) * 2.6], marker="v", ms=4, color=ORANGE,
                mec="white", mew=0.4, clip_on=False, zorder=6, ls="none")
        ax.text(0.5, 1.02, f"wins on {100 * plane_frac:.1f}% of the plane",
                transform=ax.transAxes, ha="center", va="bottom", fontsize=7.5, color="#444")
        full = d[-1] - d[0] > 300
        ax.set_xticks([-90, 0, 90] if full else [-60, -30, 0, 30, 60])
        if full:
            ax.set_xlim(-180, 180)
        if c == 0:
            ax.set_ylabel("burn start (s)")
        else:
            ax.set_yticklabels([])
        ax.tick_params(length=2.5)
        for sp in ax.spines.values():
            sp.set_color("#c9ced9")
    fig.text(0.5, 0.012, "burn heading relative to the bearing to the target (deg)", ha="center",
             va="bottom", fontsize=8.5)

    # row labels on the left, colorbars on the right
    rows = [(r"(a) $\eta$, one-body model error", r"$\eta$"),
            (f"(b) {FTLE_T:.0f} s Lyapunov exponent", r"FTLE (s$^{-1}$)"),
            ("(c) single-burn wins", "win fraction, best burn size")]
    for r, ((lab, clab), norm, cmap) in enumerate(zip(rows, [eta_norm, ftle_norm, win_norm],
                                                      [cm_eta, cm_ftle, cm_win])):
        cb = fig.colorbar(ScalarMappable(norm=norm, cmap=cmap), cax=cax[r])
        cb.set_label(clab, fontsize=8); cb.outline.set_linewidth(0.4); cb.ax.tick_params(labelsize=7, length=2.5)
        bb = axes[r, 0].get_position()
        fig.text(0.012, 0.5 * (bb.y0 + bb.y1), lab, rotation=90, ha="center", va="center", fontsize=8)
    fig.text(0.5, 0.975, "bodies at true radius (white)      – – equal-pull surface      "
             "▲ start      ○ target      ▼ heading of the start velocity", ha="center", va="center", fontsize=7.5, color="#444")

    out = pathlib.Path(args.data).parent
    fig.savefig(out / "phase.pdf"); fig.savefig(out / "phase.png", dpi=300)
    plt.close(fig)
    print("phase.pdf  " + "  ".join(
        f"{f}: level {stats[f]['level']} eta {stats[f]['eta_disc_median']:.4f} "
        f"shell {stats[f]['eta_shell_median']:.3f} D {stats[f]['D']:.3g} "
        f"win {stats[f]['sufficiency']:.4f}" for f in fams))


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    sub = ap.add_subparsers(dest="cmd", required=True)
    c = sub.add_parser("compute")
    c.add_argument("--pool", default=str(ROOT / "data" / "pools" / "long.npz"))
    c.add_argument("--out", default=str(ROOT / "paper" / "figures"))
    c.add_argument("--res", type=int, default=512, help="grid side for the closed-form fields")
    c.add_argument("--ftle-res", type=int, default=256, help="grid side for the integrated FTLE")
    c.add_argument("--n-time", type=int, default=64)
    c.add_argument("--n-dir", type=int, default=360)
    c.add_argument("--n-dv", type=int, default=4)
    c.add_argument("--cell-samples", type=int, default=4)
    c.add_argument("--dir-span-deg", type=float, default=360.0)
    r = sub.add_parser("render")
    r.add_argument("--data", default=str(ROOT / "paper" / "figures" / "phase_data.npz"))
    a = ap.parse_args()
    {"compute": compute, "render": render}[a.cmd](a)
