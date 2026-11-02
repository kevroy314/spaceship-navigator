r"""The state-machine picture of the task, and one real flight labelled by it.

    JAX_PLATFORMS=cpu XLA_PYTHON_CLIENT_PREALLOCATE=false \
        python scripts/fig_regimes.py [--task hard:556 --roll 0 --chaos ballistic] [--all]

Everything runs on the CPU: 16 sampled rollouts of a trained policy per task
and, along each, ~150 twenty-second coasting integrations (about 30 s per task
once compiled).  Writes paper/figures/regimes.pdf and regimes.png, and prints
the state fractions of every task measured (`--all` adds the four ghost-fleet
tasks, whose asteroid_field and sol_like numbers the paper quotes alongside).
Panels:

  (a) the state machine of Section 2's first picture, drawn as a diagram
      whose states are regimes of predictability rather than positions:
      well-behaved (plan), contested (re-model), chaotic encounter (bound
      the exit; the region as a noise filter mapping an entry set onto a
      bounded exit set) and hazard (avoid; constrains the route
      independently of gravity);
  (b) the same states measured along a real flight on a real level: the
      flown path coloured by state, a timeline bar of the state sequence
      over the 150 s episode, and beneath it the two exponents the
      chaotic label can be read from, against the critical value.

State membership is measured, not drawn, with precedence
hazard > chaotic > contested > well-behaved:

  contested   eta = |a_full - a_dom| / |a_full| at the craft exceeds ETA_THR
              (physics.gravity_at with topk 0 and 1, as scripts/fig_phase.py);
  hazard      the craft is inside a debris or radiation zone (zone_weights
              > 0 for a harmful kind), inside an atmosphere, or taking damage
              above RAD_THR = a tenth of its health per full episode;
  chaotic     an exponent measured from the craft's own (x, v) exceeds
              LAM_CRIT = 0.0150 /s, the paper's critical value.  Two readings
              are computed and `--chaos` picks the one that labels the flight
              (default ballistic, the paper's own; both are drawn in the strip
              beneath the timeline, since the disagreement is the lesson):
                ftle20     the 20 s deformation-gradient FTLE of fig_phase.py
                           (coasting copies displaced eps in x and y);
                ballistic  the paper's own measurement: one tick of thrust as
                           the perturbation and sep = a t exp(lam t) fitted
                           over 2-20 s, so the ballistic factor is divided
                           out before the exponent is compared with LAM_CRIT.
              The two disagree in kind.  On the four ghost tasks (16 rollouts
              each, after hazard precedence) the raw 20 s FTLE labels 17-25%
              of ticks chaotic, the ballistic-corrected exponent 0% on three
              tasks and 6% on easy:0.  The raw reading is tidal shear beside
              the dominant body: along every flight log(lam20) tracks the
              tidal closed form sqrt(sum 2GM/r^3) at r = 0.86-1.00, and every
              sample above LAM_CRIT lies within 290-434 u of the star.

Trajectories: hard-suite tasks from data/eval/suites.pkl flown with the last
checkpoint of a local run (default ppo_v5_noinstr, panel off), 16 sampled
rollouts with the keys scripts/ghost_fleet.py uses; the first rollout is drawn.
The flight shown is hard:556 (star_cluster, level 2428), chosen by screening
all 132 binary_star and star_cluster tasks of the suite for the first rollout
that spends the most time contested while touching a hazard and still entering
a third state.  On those two families eta > 0.1 holds on every tick the craft
is not in a hazard (the companion stars always supply more than a tenth of the
field), so no such flight has a well-behaved segment; the ghost-fleet tasks
(easy:0, medium:0, medium:1, hard:0; `--all`) are the converse, single-dominant-
body levels on which the contested state never appears.  The ghost JSON files
themselves (data/ghosts/*.json) hold positions quantised to a 1000-step box
every 15 ticks and no velocity, which cannot seed an exponent, and the
checkpoints behind them (abl_info_s*) are not on this machine, so the same
suite tasks are re-flown here instead.

Draft caption (the coordinator owns the final wording):

  Figure N.  Regimes of predictability as a state machine, and one flight
  read through it.  (a) The first picture of Section 2: the states are
  regimes, not positions.  In a well-behaved field a one- or few-body model
  holds and the navigator plans ahead; in a contested field no single body
  dominates, the model must be richer, but the dynamics remain predictable;
  in a chaotic encounter nearby trajectories diverge, so the exit cannot be
  chosen, only bounded, and the region acts as a noise filter mapping the
  set of entry states onto a bounded set of exits for which the plan on the
  far side is made; a hazard (radiation, debris, atmosphere) constrains the
  route independently of gravity and can be entered from any regime.
  (b) The states measured along one flight of a trained policy
  (ppo_v5_noinstr, sampled rollout 1 of 16) on hard-suite task 556
  (star_cluster family, level 2428, R = 1574 u), with precedence hazard >
  chaotic > contested > well-behaved.  Contested: eta > 0.1 at the craft;
  hazard: inside a debris or radiation zone or an atmosphere, or taking
  damage faster than a tenth of the craft's health per episode; chaotic: the
  exponent measured from the craft's own state, one tick of thrust as the
  perturbation and the ballistic factor divided out as in Section 3.4,
  exceeds the critical 0.0150 /s.  The map crops the flown region: three
  luminous stars of the cluster, bodies at true radius, grey curves their
  paths over the episode, green rings the three targets where reached (64 s
  and 88 s; the third rides a station beside a star and was not reached
  before the clock).  The bar beneath is the state sequence over the 150 s
  episode: contested throughout, a 2 s chaotic sliver at 131 s on the dive
  towards the cluster core, and a hazard from 133 s in the stars' near
  field; the well-behaved state is never entered, since in a cluster the
  other stars always supply more than a tenth of the field.  The lowest
  strip shows the two exponents along the flight against the critical value
  (dashed): the raw 20 s finite-time exponent of Figure~\ref{fig:phase},
  which is tidal shear and crosses the line on 7% of the ticks, and the
  paper's own reading, which crosses on 1%.  On the four ghost-fleet tasks
  (asteroid_field and sol_like levels, 16 rollouts each) the contested state
  never appears and the chaotic state appears on 0% of ticks on three and 6%
  on the fourth, all of that within 430 u of a star.  Drawn by
  \code{scripts/fig\_regimes.py}.
"""
import argparse
import os
import pathlib
import pickle
import sys
import time

os.environ.setdefault("JAX_PLATFORMS", "cpu")
os.environ.setdefault("XLA_PYTHON_CLIENT_PREALLOCATE", "false")

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.collections import LineCollection
from matplotlib.colors import ListedColormap
from matplotlib.patches import Annulus, Circle, Ellipse, FancyArrowPatch, FancyBboxPatch, Polygon

ROOT = pathlib.Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
OUT = ROOT / "paper" / "figures"
OUT.mkdir(parents=True, exist_ok=True)

# thresholds
ETA_THR = 0.1            # contested when the one-body model discards more than this
LAM_CRIT = 0.0150        # /s, log(1/gamma)/dt at gamma = 0.999, dt = 1/15 s (Section 3.4)
FTLE_T = 20.0            # s, the window of the finite-time exponent
FTLE_EPS = 0.5           # u, the displacement for the deformation gradient
FIT_FROM = 2.0           # s, the ballistic fit uses 2..FTLE_T s
SAMPLE = 15              # ticks between exponent samples (1 s)
TASKS = [("easy", 0), ("medium", 0), ("medium", 1), ("hard", 0)]   # the ghost-fleet tasks

# house style (scripts/paper_figures.py)
BLUE, ORANGE, GREEN, RED, GREY = "#2a78d6", "#eb6834", "#0ca30c", "#d03b3b", "#6b7280"
BODY_FACE, BODY_EDGE = "#d6dae2", "#6b7280"
STAR_FACE = "#f2b134"
ATMO = "#8fb8e8"
INK = "#1f2430"
ACCENT = "#8e44ad"       # the chaotic state
STATE_NAMES = ["well-behaved", "contested", "chaotic", "hazard"]
STATE_JOBS = ["plan", "re-model", "bound the exit", "avoid"]
STATE_COLS = [BLUE, ORANGE, ACCENT, RED]
plt.rcParams.update({
    "figure.dpi": 150, "savefig.dpi": 200, "savefig.bbox": "tight",
    "font.family": "serif", "font.serif": ["DejaVu Serif"], "font.size": 8.0,
    "axes.labelsize": 8.0, "axes.titlesize": 8.5, "legend.fontsize": 7.5,
    "xtick.labelsize": 7.0, "ytick.labelsize": 7.0,
    "axes.spines.top": False, "axes.spines.right": False,
    "axes.linewidth": 0.6, "xtick.major.width": 0.6, "ytick.major.width": 0.6,
    "lines.solid_capstyle": "round", "pdf.fonttype": 42,
})
SMALL = 6.6


# ------------------------------------------------------------------ measure
def measure(suite, ti, run, n_roll, seed):
    """Fly a suite task n_roll times with the run's last checkpoint and read
    eta, the hazard terms and two exponents along every rollout."""
    import jax
    import jax.numpy as jnp
    from flax import serialization
    from spacenav import constants as C
    from spacenav import env as E
    from spacenav import physics as P
    from spacenav.levels.build import load_pool
    from spacenav.levels.families import FAMILY_NAMES
    from spacenav.rl.network import ACTIONS, ActorCritic
    from spacenav.rl.ppo import PPOConfig

    E.set_instruments("off")          # ppo_v5_noinstr was trained with the panel zeroed
    t0 = time.time()
    suites = pickle.load(open(ROOT / "data" / "eval" / "suites.pkl", "rb"))
    s = suites[suite]
    pool, meta = load_pool(ROOT / "data" / "pools" / f"{s['pool']}.npz")
    lvl_i = int(np.asarray(s["level_idx"])[ti])
    level = jax.tree.map(lambda x: x[lvl_i], pool)
    task = jax.tree.map(lambda x: jnp.asarray(x)[ti], s["task"])
    family = FAMILY_NAMES[int(level.family)]

    net = ActorCritic(d=PPOConfig().d_model)
    template = net.init(jax.random.PRNGKey(0), E.observe(level, task, E.reset(level, task)))
    ckpt = sorted((ROOT / "data" / "rl" / run).glob("ckpt_*.msgpack"))[-1]
    params = serialization.from_bytes(template, ckpt.read_bytes())

    def one(key):
        def policy(obs, state):
            logits, _ = net.apply(params, obs)
            return ACTIONS[jax.random.categorical(jax.random.fold_in(key, state.tick), logits)]
        return E.rollout(level, task, policy)

    keys = jax.random.split(jax.random.PRNGKey(seed), n_roll)
    _, tr = jax.jit(jax.vmap(one))(keys)
    tr = {k: np.asarray(v) for k, v in tr.items()}
    st = tr["status"]
    T = C.MAX_EPISODE_TICKS
    ends = np.array([int(np.nonzero(r != 0)[0][0]) + 1 if np.any(r != 0) else T for r in st])
    status = np.array([int(st[r][ends[r] - 1]) for r in range(n_roll)])

    # the body ephemeris on substeps, long enough for a window opened on the last tick
    n_sub = (T + int(round(FTLE_T / C.CTRL_DT)) + 1) * C.SUBSTEPS
    bp0, bv0 = E.snapshot_state(level, task.snapshot)
    bps_s, bvs_s = P.roll_bodies(level, bp0, bv0, n_sub)
    bps = bps_s[::C.SUBSTEPS]                       # (k) = bodies after control tick k
    bvs = bvs_s[::C.SUBSTEPS]
    wpos, _ = jax.vmap(lambda p, v: E.waypoint_states(task, p, v))(bps[:T + 1], bvs[:T + 1])

    # eta and the hazard geometry on every tick of every rollout
    def per_tick(x, bp):
        full = P.gravity_at(x[None], bp, level, 0)[0]
        dom = P.gravity_at(x[None], bp, level, 1)[0]
        eta = jnp.sqrt(jnp.sum((full - dom) ** 2)) / (jnp.sqrt(jnp.sum(full ** 2)) + 1e-12)
        w, _, _ = P.zone_weights(level, bp, x)
        harmful = (level.zone_kind == C.ZONE_DEBRIS) | (level.zone_kind == C.ZONE_RADIATION)
        in_zone = jnp.any((w > 0.0) & harmful & level.zone_active)
        d = x[None] - bp
        r = jnp.sqrt(jnp.sum(d * d, -1))
        in_atmo = jnp.any(level.active & (level.atmo_h > 0) & (r - level.radius < level.atmo_h))
        return eta, in_zone, in_atmo
    eta, in_zone, in_atmo = jax.jit(jax.vmap(jax.vmap(per_tick, in_axes=(0, 0)), in_axes=(0, None)))(
        jnp.asarray(tr["pos"]), bps[1:T + 1])
    eta, in_zone, in_atmo = np.asarray(eta), np.asarray(in_zone), np.asarray(in_atmo)

    # exponents every SAMPLE ticks, from the craft's own state against the true ephemeris
    h = C.PHYS_DT
    n_steps = int(round(FTLE_T / h))

    def coast(bp, bv, X0, V0):
        """Integrate copies of the craft (no thrust, no drag) with the bodies, the
        leapfrog of physics.substep; returns final positions and the separation of
        each copy from the first at every control tick."""
        def f(carry, _):
            b_p, b_v, X, V = carry
            a0 = P.gravity_at(X, b_p, level, 0)
            Vh = V + 0.5 * h * a0
            X1 = X + h * Vh
            b_p1, b_v1 = P.bodies_step(b_p, b_v, level, h)
            a1 = P.gravity_at(X1, b_p1, level, 0)
            sep = jnp.sqrt(jnp.sum((X1[1:] - X1[0]) ** 2, -1))
            return (b_p1, b_v1, X1, Vh + 0.5 * h * a1), sep
        (_, _, X, _), seps = jax.lax.scan(f, (bp, bv, X0, V0), None, length=n_steps)
        return X, seps[C.SUBSTEPS - 1::C.SUBSTEPS]

    def ftle(bp, bv, x, v):
        X0 = jnp.stack([x, x + jnp.array([FTLE_EPS, 0.0]), x + jnp.array([0.0, FTLE_EPS])])
        X, _ = coast(bp, bv, X0, jnp.stack([v, v, v]))
        Fx, Fy = (X[1] - X[0]) / FTLE_EPS, (X[2] - X[0]) / FTLE_EPS
        a, d = Fx[0] ** 2 + Fx[1] ** 2, Fy[0] ** 2 + Fy[1] ** 2
        b = Fx[0] * Fy[0] + Fx[1] * Fy[1]
        lam = 0.5 * (a + d) + jnp.sqrt(jnp.maximum(0.25 * (a - d) ** 2 + b * b, 0.0))
        return jnp.log(jnp.sqrt(jnp.maximum(lam, 1e-12))) / FTLE_T

    def kick(bp, bv, x, v, angle):
        dv = task.accel * C.CTRL_DT                 # one tick of full thrust
        u = jnp.array([jnp.cos(angle), jnp.sin(angle)])
        w = jnp.array([-u[1], u[0]])
        _, seps = coast(bp, bv, jnp.stack([x, x, x]), jnp.stack([v, v + dv * u, v + dv * w]))
        return seps                                 # (ticks, 2)

    idx = np.arange(0, T, SAMPLE)
    sub = (idx + 1) * C.SUBSTEPS
    bpk, bvk = bps_s[sub], bvs_s[sub]
    xk, vk, ak = (jnp.asarray(tr[k][:, idx]) for k in ("pos", "vel", "angle"))
    over = lambda f, n: jax.jit(jax.vmap(jax.vmap(f, in_axes=(0,) * n), in_axes=(None, None) + (0,) * (n - 2)))
    lam20 = np.asarray(over(ftle, 4)(bpk, bvk, xk, vk))                        # (R, K)
    seps = np.asarray(over(kick, 5)(bpk, bvk, xk, vk, ak))                      # (R, K, ticks, 2)
    tt = (np.arange(seps.shape[2]) + 1) * C.CTRL_DT
    m = tt >= FIT_FROM
    y = np.log(np.maximum(seps[..., m, :], 1e-9) / tt[m][:, None])
    A = np.stack([np.ones(m.sum()), tt[m]], 1)
    coef = np.linalg.lstsq(A, y.reshape(-1, m.sum()).T, rcond=None)[0]
    lam_bc = coef[1].reshape(seps.shape[0], seps.shape[1], 2).max(-1)           # (R, K)

    L = {k: np.asarray(getattr(level, k)) for k in level._fields}
    Tk = {k: np.asarray(getattr(task, k)) for k in task._fields}
    print(f"{suite}:{ti}  level {lvl_i} ({family}, R = {float(level.level_radius):.0f} u)  "
          f"{run}/{ckpt.name}  {n_roll} rollouts  ({time.time() - t0:.0f} s)", flush=True)
    return dict(suite=suite, ti=ti, level_idx=lvl_i, family=family, run=run, ckpt=ckpt.name,
                level=L, task=Tk, bps=np.asarray(bps[:T + 1]), wpos=np.asarray(wpos),
                pos=tr["pos"], vel=tr["vel"], visited=tr["visited"], damage=tr["damage_parts"],
                health=tr["health"], ends=ends, status=status, eta=eta, in_zone=in_zone,
                in_atmo=in_atmo, idx=idx, lam20=lam20, lam_bc=lam_bc,
                C=dict(R=float(level.level_radius), T=T, CTRL_DT=C.CTRL_DT,
                       T_MAX=C.MAX_EPISODE_TIME, HEALTH=C.SHIP_HEALTH,
                       RAD_THR=0.1 * C.SHIP_HEALTH / C.MAX_EPISODE_TIME,
                       TARGET_RADIUS=C.TARGET_RADIUS, KIND_STATION=C.KIND_STATION,
                       KIND_TRACER=C.KIND_TRACER, ZONE_DEBRIS=C.ZONE_DEBRIS,
                       ZONE_RADIATION=C.ZONE_RADIATION, ZONE_SENSOR=C.ZONE_SENSOR))


def states(D, r, chaos):
    """Per-tick state of rollout r: 0 well-behaved, 1 contested, 2 chaotic, 3 hazard."""
    n = int(D["ends"][r])
    lam = D["lam20"] if chaos == "ftle20" else D["lam_bc"]
    chaotic = np.repeat(lam[r] > LAM_CRIT, SAMPLE)[:n]
    hazard = D["in_zone"][r, :n] | D["in_atmo"][r, :n] | (D["damage"][r, :n].sum(-1) > D["C"]["RAD_THR"])
    contested = D["eta"][r, :n] > ETA_THR
    return np.where(hazard, 3, np.where(chaotic, 2, np.where(contested, 1, 0)))


def fractions(D, chaos):
    """Fraction of ticks in each state: (per rollout, mean over rollouts)."""
    per = np.array([np.bincount(states(D, r, chaos), minlength=4) / D["ends"][r]
                    for r in range(len(D["ends"]))])
    return per, per.mean(0)


# ------------------------------------------------------------------ helpers
def letter(fig, x, y, s):
    fig.text(x, y, s, fontsize=9.5, fontweight="bold", va="top", ha="left", color=INK)


def frame(ax):
    ax.set_xticks([]); ax.set_yticks([])
    for sp in ax.spines.values():
        sp.set_visible(True); sp.set_color("#c9ced9"); sp.set_linewidth(0.5)


def ship_glyph(ax, x, y, angle, size, color=BLUE, zorder=10):
    c, s = np.cos(angle), np.sin(angle)
    pts = np.array([[1.0, 0.0], [-0.7, 0.6], [-0.35, 0.0], [-0.7, -0.6]]) * size
    pts = pts @ np.array([[c, s], [-s, c]]) + np.array([x, y])
    ax.add_patch(Polygon(pts, closed=True, facecolor=color, edgecolor="white", lw=0.5, zorder=zorder))


def arrow(ax, p, q, color=GREY, lw=0.8, ms=7, rad=0.0, zorder=5, ls="-", style="-|>"):
    ax.add_patch(FancyArrowPatch(p, q, arrowstyle=style, mutation_scale=ms, color=color, lw=lw,
                                 connectionstyle=f"arc3,rad={rad}", zorder=zorder, linestyle=ls,
                                 shrinkA=0, shrinkB=0))


# ------------------------------------------------------------------- panels
def panel_machine(ax):
    """The state machine: boxes are regimes; arrows are what the field and the
    craft's path admit; the italic line under each title is the navigator's job."""
    ax.set_xlim(0, 100); ax.set_ylim(0, 100); ax.axis("off")
    # (centre x, centre y, width, height, title, job, what it means, colour, fill, dashed)
    boxes = [
        (20, 84, 36, 20, "well-behaved", "plan ahead", "a one- or few-body\nmodel holds", BLUE, "white", False),
        (80, 84, 36, 20, "contested", "re-model", "no body dominates;\nstill predictable", ORANGE, "white", False),
        (50, 46, 46, 34, "chaotic encounter", "bound the exit",
         "nearby paths diverge; the exit\ncannot be chosen, only bounded", ACCENT, "#f3eaf8", False),
        (50, 8, 90, 15.5, "hazard", "avoid",
         "radiation, debris, atmosphere: the route\nis constrained independently of gravity", RED, "white", True),
    ]
    for cx, cy, w, h, title, job, what, col, fill, dashed in boxes:
        ax.add_patch(FancyBboxPatch((cx - w / 2, cy - h / 2), w, h, boxstyle="round,pad=0,rounding_size=2.5",
                                    facecolor=fill, edgecolor=col, lw=1.0 if fill != "white" else 0.8,
                                    ls=(0, (3, 2)) if dashed else "-", zorder=3))
        top = cy + h / 2
        if dashed:      # the hazard bar is short: title and job on one line
            ax.text(cx - w / 2 + 3, top - 2.0, title, fontsize=7.4, fontweight="bold", color=col,
                    ha="left", va="top", zorder=5)
            ax.text(cx + w / 2 - 3, top - 2.0, job, fontsize=SMALL, color=col, ha="right", va="top",
                    style="italic", zorder=5)
            ax.text(cx, top - 7.0, what, fontsize=SMALL - 0.4, color=INK, ha="center", va="top",
                    zorder=5, linespacing=1.15)
            continue
        ax.text(cx, top - 2.0, title, fontsize=7.4, fontweight="bold", color=col, ha="center",
                va="top", zorder=5)
        ax.text(cx, top - 6.6, job, fontsize=SMALL, color=col, ha="center", va="top",
                style="italic", zorder=5)
        ax.text(cx, top - 10.8, what, fontsize=SMALL - 0.4, color=INK, ha="center", va="top",
                zorder=5, linespacing=1.15)
        if title.startswith("chaotic"):
            # the noise filter: a small entry set mapped onto a larger, bounded exit set
            ex, ey = cx - 16, cy - 8
            ox, oy = cx + 8, cy - 8
            ax.add_patch(Circle((ex, ey), 1.5, facecolor=ACCENT, edgecolor="none", zorder=6))
            ax.add_patch(Ellipse((ox, oy), 15, 9, facecolor="none", edgecolor=ACCENT, lw=0.8,
                                 ls=(0, (2, 1.5)), zorder=6))
            for k in range(7):
                a = (k - 3) / 3.0
                tx, ty = ox + 4.5 * a * np.cos(0.4 * a) - 1.0, oy + 3.0 * np.sin(1.4 * a + 0.3)
                ax.annotate("", (tx, ty), (ex + 1.7, ey), arrowprops=dict(
                    arrowstyle="-", color=ACCENT, lw=0.45, alpha=0.7,
                    connectionstyle=f"arc3,rad={0.25 * a}"), zorder=6)
                ax.plot([tx], [ty], marker="o", ms=1.6, color=ACCENT, zorder=7, ls="none")
            ax.text(ex, ey - 3.0, "entry set", fontsize=SMALL - 0.8, color=ACCENT, ha="center",
                    va="top", zorder=6)
            ax.text(ox, oy - 5.4, "exit set, bounded", fontsize=SMALL - 0.8, color=ACCENT,
                    ha="center", va="top", zorder=6)

    # transitions: what the field and the craft's path admit
    def lab(x, y, s, **kw):
        props = dict(fontsize=SMALL - 0.8, color=GREY, ha="center", va="center", zorder=6,
                     linespacing=1.1)
        props.update(kw)
        ax.text(x, y, s, **props)
    arrow(ax, (38, 88.5), (62, 88.5))
    lab(50, 92.0, "a second\npull grows")
    arrow(ax, (62, 79.5), (38, 79.5))
    lab(50, 76.0, "one body\ndominates again")
    arrow(ax, (27, 74), (39, 63))
    arrow(ax, (73, 74), (61, 63))
    lab(50, 67.5, "close encounter")
    arrow(ax, (27, 52), (11, 74), rad=0.35, color=ACCENT)
    lab(2, 54, "exit: bounded,\nnot chosen", ha="left", color=ACCENT)
    arrow(ax, (50, 29), (50, 15.75), style="<|-|>", color=RED)
    lab(53, 22.5, "entered from, and\nleft to, any regime", ha="left", color=RED)


def panel_flight(ax_map, ax_bar, ax_lam, D, r, chaos):
    L, T, K = D["level"], D["task"], D["C"]
    n = int(D["ends"][r])
    st = states(D, r, chaos)
    pos = D["pos"][r, :n]
    bps, wpos = D["bps"], D["wpos"]
    R = K["R"]
    act = L["active"]
    dt = K["CTRL_DT"]

    # the map: the flown region and the luminous body it circles, not the whole disc
    lum = [bps[0, i] for i in np.nonzero(act)[0] if L["lum"][i] > 0]
    ext = np.concatenate([pos] + [np.asarray(l)[None] for l in lum], 0)
    lo, hi = ext.min(0), ext.max(0)
    span = float(max(hi[0] - lo[0], hi[1] - lo[1])) * 1.12 + 60
    cx, cy = (lo + hi) / 2
    half = span / 2
    inside = lambda q: abs(q[0] - cx) < 0.95 * half and abs(q[1] - cy) < 0.95 * half
    ax_map.set_xlim(cx - half, cx + half); ax_map.set_ylim(cy - half, cy + half)
    ax_map.set_aspect("equal")
    ax_map.add_patch(Circle((0, 0), R, facecolor="none", edgecolor=GREY, lw=0.6, ls=(0, (1, 2)), zorder=1))
    for z in np.nonzero(L["zone_active"])[0]:
        a = int(L["zone_anchor"][z])
        zx, zy = (bps[0, a] if a >= 0 else (0.0, 0.0))
        r_in, r_out = float(L["zone_r_in"][z]), float(L["zone_r_out"][z])
        kind = int(L["zone_kind"][z])
        if kind == K["ZONE_DEBRIS"]:
            ax_map.add_patch(Annulus((zx, zy), r_out, r_out - max(r_in, 1e-3), facecolor=ORANGE,
                                     alpha=0.13, edgecolor=ORANGE, lw=0.4, zorder=1))
        elif kind == K["ZONE_RADIATION"]:
            ax_map.add_patch(Annulus((zx, zy), r_out, r_out - max(r_in, 1e-3), facecolor=RED,
                                     alpha=0.15, edgecolor=RED, lw=0.4, zorder=1))
        elif kind == K["ZONE_SENSOR"]:
            ax_map.add_patch(Circle((zx, zy), r_out, facecolor="none", edgecolor=GREY, lw=0.5,
                                    ls=(0, (3, 2)), zorder=2, alpha=0.8))
    anchors = set(int(a) for a in T["wp_anchor"][T["wp_active"]] if a >= 0)
    for i in np.nonzero(act)[0]:
        kind = int(L["kind"][i])
        if kind == K["KIND_TRACER"] or (kind == K["KIND_STATION"] and i in anchors):
            continue
        ax_map.plot(bps[:, i, 0], bps[:, i, 1], color=GREY, lw=0.45, alpha=0.55, zorder=2)
    for i in np.nonzero(act)[0]:
        x, y = bps[0, i]
        kind = int(L["kind"][i])
        rad = float(L["radius"][i])
        if kind == K["KIND_TRACER"]:
            continue
        if kind == K["KIND_STATION"]:
            ax_map.plot([x], [y], marker="s", ms=3.0, color=GREY, mfc="white", mew=0.7, zorder=7)
            continue
        if L["lum"][i] > 0:
            for k, a in ((4.0, 0.06), (2.8, 0.10), (1.8, 0.18)):
                ax_map.add_patch(Circle((x, y), rad * k, facecolor=STAR_FACE, edgecolor="none", alpha=a, zorder=3))
            ax_map.add_patch(Circle((x, y), rad, facecolor=STAR_FACE, edgecolor=ORANGE, lw=0.5, zorder=5))
        else:
            if L["atmo_h"][i] > 0:
                ax_map.add_patch(Circle((x, y), rad + float(L["atmo_h"][i]), facecolor=ATMO,
                                        edgecolor="none", alpha=0.55, zorder=4))
            ax_map.add_patch(Circle((x, y), rad, facecolor=BODY_FACE, edgecolor=BODY_EDGE, lw=0.5, zorder=5))
            if rad < 8:
                ax_map.plot([x], [y], marker="o", ms=5, mfc="none", mec=GREY, mew=0.5, zorder=6)

    # targets, numbered, drawn where they were when reached
    hits = []
    for k in range(int(np.sum(T["wp_active"]))):
        nz = np.nonzero(D["visited"][r, :n] > k)[0]
        hits.append(int(nz[0]) + 1 if len(nz) else None)
    for k, hit in enumerate(hits):
        pa = wpos[:, k]
        if T["wp_anchor"][k] >= 0:
            ax_map.plot(pa[:, 0], pa[:, 1], color=GREEN, lw=0.7, alpha=0.5, zorder=6)
        at = pa[hit] if hit is not None else pa[0]
        ax_map.add_patch(Circle(at, K["TARGET_RADIUS"], facecolor="none", edgecolor=GREEN,
                                lw=1.0, zorder=7))
        if inside(at):
            ax_map.text(at[0] + K["TARGET_RADIUS"] * 1.3, at[1] + K["TARGET_RADIUS"] * 1.0, str(k + 1),
                        fontsize=SMALL, color=GREEN, fontweight="bold", ha="left", va="bottom",
                        zorder=20, clip_on=True)

    # the flight, coloured by state
    seg = np.stack([pos[:-1], pos[1:]], 1)
    lc = LineCollection(seg, colors=[STATE_COLS[s] for s in st[1:]], linewidths=1.1, zorder=9)
    ax_map.add_collection(lc)
    sp = T["start_pos"]
    ship_glyph(ax_map, sp[0], sp[1], float(T["start_angle"]), 0.022 * span, zorder=11)
    ax_map.text(sp[0], sp[1] - 0.03 * span, "start", fontsize=SMALL, color=BLUE, ha="center", va="top",
                zorder=20, clip_on=True)
    ep = pos[-1]
    ax_map.plot([ep[0]], [ep[1]], marker="x", ms=4, color=INK, mew=0.9, zorder=12)
    # scale bar and the level note
    x0, y0 = cx - 0.93 * half, cy - 0.93 * half
    ax_map.plot([x0, x0 + 200], [y0, y0], color=INK, lw=1.2, zorder=20, solid_capstyle="butt")
    ax_map.text(x0 + 100, y0 + 0.015 * span, "200 u", fontsize=SMALL, ha="center", va="bottom", color=INK, zorder=20)
    frame(ax_map)

    # the timeline bar: the state sequence
    Tm = K["T_MAX"]
    cmap = ListedColormap(STATE_COLS)
    ax_bar.imshow(st[None, :], aspect="auto", cmap=cmap, vmin=-0.5, vmax=3.5,
                  extent=(0, n * dt, 0, 1), interpolation="nearest", zorder=2)
    if n * dt < Tm:
        ax_bar.axvspan(n * dt, Tm, facecolor="#eceef2", edgecolor="none", zorder=1)
    for k, hit in enumerate(hits):
        if hit is not None:
            ax_bar.plot([hit * dt], [1.0], marker="v", ms=4, color=GREEN, mec="white", mew=0.4,
                        clip_on=False, zorder=6, ls="none")
            ax_bar.text(hit * dt, 1.25, str(k + 1), fontsize=SMALL - 0.4, color=GREEN, fontweight="bold",
                        ha="center", va="bottom", clip_on=False)
    ax_bar.set_xlim(0, Tm); ax_bar.set_ylim(0, 1)
    ax_bar.set_xticks([]); ax_bar.set_yticks([])
    for sp_ in ax_bar.spines.values():
        sp_.set_visible(False)
    ax_bar.set_ylabel("state", fontsize=SMALL, rotation=0, ha="right", va="center", labelpad=4)
    ax_bar.text(0, 1.25, "targets reached:", fontsize=SMALL - 0.8, color=GREEN, ha="left", va="bottom",
                clip_on=False)

    # the exponents along the flight, against the critical value
    t_s = (D["idx"] + 1) * dt
    ok = D["idx"] < n
    floor = 3e-6
    ax_lam.axhline(LAM_CRIT, color=INK, lw=0.6, ls=(0, (3, 2)), zorder=3)
    ax_lam.plot(t_s[ok], np.maximum(D["lam20"][r, ok], floor), color=ACCENT, lw=0.8, zorder=4)
    ax_lam.plot(t_s[ok], np.maximum(D["lam_bc"][r, ok], floor), color=GREY, lw=0.8, zorder=4)
    ax_lam.set_yscale("log"); ax_lam.set_ylim(floor, 0.5)
    ax_lam.set_yticks([1e-4, 1e-2]); ax_lam.set_yticklabels([r"$10^{-4}$", r"$10^{-2}$"])
    ax_lam.set_yticks([], minor=True)
    ax_lam.tick_params(labelsize=SMALL - 0.6, length=2, width=0.5, pad=1.5)
    ax_lam.set_xlim(0, Tm)
    ax_lam.set_xticks([0, 50, 100, 150])
    ax_lam.set_xlabel("episode time (s)", labelpad=1.5)
    ax_lam.set_ylabel(r"$\lambda$ (s$^{-1}$)", fontsize=SMALL, labelpad=2)
    ax_lam.spines["left"].set_linewidth(0.5); ax_lam.spines["bottom"].set_linewidth(0.5)
    ax_lam.text(Tm / 2, LAM_CRIT * 1.3, r"$\lambda_{\mathrm{crit}}$ = 0.015", fontsize=SMALL - 0.6,
                color=INK, ha="center", va="bottom")
    ax_lam.text(1.5, 0.5 / 1.25, f"{FTLE_T:.0f} s FTLE (raw)", fontsize=SMALL - 0.6, color=ACCENT,
                ha="left", va="top")
    ax_lam.text(Tm - 1.5, floor * 1.35, "ballistic factor divided out", fontsize=SMALL - 0.6,
                color=GREY, ha="right", va="bottom")
    return hits


# --------------------------------------------------------------------- main
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--task", default="hard:556", help="suite:index of the flight drawn in (b)")
    ap.add_argument("--roll", type=int, default=0, help="which of the sampled rollouts to draw")
    ap.add_argument("--chaos", default="ballistic", choices=["ftle20", "ballistic"],
                    help="which exponent labels the chaotic state")
    ap.add_argument("--run", default="ppo_v5_noinstr")
    ap.add_argument("--rollouts", type=int, default=16)
    ap.add_argument("--seed", type=int, default=7, help="the ghost fleet's sampling seed")
    ap.add_argument("--all", action="store_true", help="also measure the other three ghost tasks")
    ap.add_argument("--cache", default=None, help="pickle the measurements here and reuse them")
    ap.add_argument("--out", default=str(OUT))
    args = ap.parse_args()

    want = [tuple((args.task.split(":")[0], int(args.task.split(":")[1])))]
    if args.all:
        want += [t for t in TASKS if t not in want]
    cache = {}
    if args.cache and pathlib.Path(args.cache).exists():
        cache = pickle.load(open(args.cache, "rb"))
    data = {}
    for suite, ti in want:
        key = f"{suite}:{ti}"
        if key not in cache:
            cache[key] = measure(suite, ti, args.run, args.rollouts, args.seed)
            if args.cache:
                pickle.dump(cache, open(args.cache, "wb"))
        data[key] = cache[key]

    # the fractions the paper will quote
    print(f"\nstate fractions (ticks), precedence hazard > chaotic > contested > well-behaved;\n"
          f"eta > {ETA_THR}, damage > {0.1 * 100 / 150:.3f} /s or inside a harmful zone or atmosphere, "
          f"lambda > {LAM_CRIT} /s\n")
    print(f"{'task':<10} {'level':>6} {'family':<15} {'chaos reading':<10} {'which':<9} "
          + " ".join(f"{s:>12}" for s in STATE_NAMES) + "   outcomes")
    for key, D in data.items():
        outcomes = {}
        for s in D["status"]:
            outcomes[s] = outcomes.get(s, 0) + 1
        from spacenav.types import REASONS
        oc = ", ".join(f"{REASONS[k]} {v}" for k, v in sorted(outcomes.items()))
        for chaos in ("ftle20", "ballistic"):
            per, mean = fractions(D, chaos)
            rr = args.roll
            print(f"{key:<10} {D['level_idx']:>6} {D['family']:<15} {chaos:<10} {'roll ' + str(rr + 1):<9} "
                  + " ".join(f"{100 * v:11.1f}%" for v in per[rr])
                  + f"   {REASONS[int(D['status'][rr])]} at {D['ends'][rr] * D['C']['CTRL_DT']:.0f} s")
            print(f"{'':<10} {'':>6} {'':<15} {chaos:<10} {'mean 16':<9} "
                  + " ".join(f"{100 * v:11.1f}%" for v in mean) + f"   {oc}")
        emax = max(float(D["eta"][r, :D["ends"][r]].max()) for r in range(len(D["ends"])))
        print(f"{'':<10} eta max over rollouts {emax:.4f}; lam20 median {np.median(D['lam20']):.4f}, "
              f"lam_bc median {np.median(D['lam_bc']):.4f}\n")

    # the figure
    D = data[f"{want[0][0]}:{want[0][1]}"]
    W, H = 6.4, 3.2
    fig = plt.figure(figsize=(W, H))
    axes = lambda x, y, w, h: fig.add_axes([x / W, y / H, w / W, h / H])
    ax_a = axes(0.05, 0.08, 2.95, 2.9)
    ax_map = axes(3.35, 1.0, 2.0, 2.0)
    ax_bar = axes(3.35, 0.72, 2.0, 0.16)
    ax_lam = axes(3.35, 0.27, 2.0, 0.42)
    panel_machine(ax_a)
    panel_flight(ax_map, ax_bar, ax_lam, D, args.roll, args.chaos)
    letter(fig, 0.02 / W, 3.17 / H, "(a)  regimes as a state machine")
    letter(fig, 3.2 / W, 3.17 / H, "(b)  one flight, read through it")
    # legend for (b), to the right of the map
    x0 = 5.45 / W
    for k, (nm, job, col) in enumerate(zip(STATE_NAMES, STATE_JOBS, STATE_COLS)):
        y = (2.95 - 0.19 * k) / H
        fig.patches.append(plt.Rectangle((x0, y - 0.008), 0.025, 0.028, transform=fig.transFigure,
                                         facecolor=col, edgecolor="none"))
        fig.text(x0 + 0.032, y + 0.006, nm, fontsize=SMALL - 0.3, color=col, va="center", ha="left")
    fig.text(x0, (2.95 - 0.19 * 4) / H + 0.006,
             "precedence\nhazard > chaotic\n> contested\n> well-behaved", fontsize=SMALL - 0.8,
             color=GREY, va="top", ha="left", linespacing=1.15)
    L = D["level"]
    zk = set(int(k) for k in L["zone_kind"][L["zone_active"]])
    zones = []
    if D["C"]["ZONE_DEBRIS"] in zk:
        zones.append("orange: debris")
    if D["C"]["ZONE_RADIATION"] in zk:
        zones.append("red: radiation")
    if D["C"]["ZONE_SENSOR"] in zk:
        zones.append("dashed: sensor")
    zones = ("shaded " if zones and zones[0] != "dashed: sensor" else "") + ", ".join(zones) + "; " if zones else ""
    fig.text(x0, 1.72 / H, f"{D['family'].replace('_', ' ')}\nfamily, level\n{D['level_idx']}, R = {D['C']['R']:.0f} u;\n"
             f"{D['suite']} suite\ntask {D['ti']}; {D['run']},\nrollout {args.roll + 1} of {len(D['ends'])}\n\n"
             "bodies at true\nradius; grey:\ntheir paths;\ngreen: targets\nwhere reached;\n"
             + zones.replace(", ", ",\n").replace("; ", ";\n") + "dotted: the\nplay disc",
             fontsize=SMALL - 0.8, color=GREY, va="top", ha="left", linespacing=1.15)
    out = pathlib.Path(args.out)
    fig.savefig(out / "regimes.pdf"); fig.savefig(out / "regimes.png", dpi=300)
    plt.close(fig)
    print("wrote", out / "regimes.pdf", out / "regimes.png", flush=True)


if __name__ == "__main__":
    main()
