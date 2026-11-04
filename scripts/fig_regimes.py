r"""The state-machine picture of the task, and the states along a real flight.

    JAX_PLATFORMS=cpu XLA_PYTHON_CLIENT_PREALLOCATE=false \
        python scripts/fig_regimes.py [--task TASK --roll 0 --chaos ballistic] [--all]

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
      flown path coloured by state, the hazard the flight crosses labelled,
      a timeline bar of the state sequence over the 150 s episode with the
      time of each state change, and beneath it the two exponents the
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
              The two disagree in kind (ADR 0039): the raw reading is tidal
              shear beside the dominant body; along every flight log(lam20)
              tracks sqrt(sum 2GM/r^3) at r = 0.86-1.00.

Trajectories: suite tasks from data/eval/suites.pkl flown with the last
checkpoint of a local run (default ppo_v5_noinstr, panel off), 16 sampled
rollouts with the keys scripts/ghost_fleet.py uses; `--roll` picks the one
drawn.  The flight shown, hard:339 (jovian, level 309), was chosen by a
screen (session scratchpad, screen.py: eta and hazard only, no exponents) of
all 528 jovian, saturnian, sol_like and bh_cluster tasks of the hard and
medium suites, the hierarchical families on which a route can cross an
equal-pull shell.  Tasks were ranked by the number of states (well-behaved,
contested, hazard) the first rollout spends more than 5% of its ticks in,
then by the smallest of those fractions.  Tasks with all three above 5%:
jovian 4 of 132, saturnian 0, sol_like 9, bh_cluster 37; the top of the
ranking is bh_cluster flights that reach no target and spend 23-57% of the
episode in a star's radiation.  Seven tasks were measured in full (the top
three, medium:116, hard:122, hard:87, and the four best that reach a target,
hard:339, hard:373, medium:518, medium:91); hard:339 is the one whose map
reads best: 55 / 29 / 1 / 15% well-behaved / contested / chaotic / hazard on
the first rollout, two of three targets reached, and each state attached to
a visible cause (a moon pass, the giant's belts).  The contested families
(binary_star, star_cluster; ADR 0039) are contested on every tick outside a
hazard and were not re-screened.  On the jovian flights the raw 20 s FTLE
labels ~40% of ticks chaotic and the ballistic reading 1-5%: 0039's tidal
reading again, now beside a giant.

Draft caption (the coordinator owns the final wording; `--caption` prints a
version with the numbers filled from the measurement):

  Figure N.  Regimes of predictability as a state machine, and the states
  along a flight.  (a) The first picture of Section 2: the states are
  regimes, not positions.  In a well-behaved field a one- or few-body model
  holds and the navigator plans ahead; in a contested field no single body
  dominates, the model must be richer, but the dynamics remain predictable;
  in a chaotic encounter nearby trajectories diverge, so only the exit set
  is predictable, and the region acts as a noise filter mapping the set of
  entry states onto a bounded set of exits for which the plan on the far
  side is made; a hazard (radiation, debris, atmosphere) constrains the
  route independently of gravity and can be entered from any regime.
  (b) The states measured along a flight of a trained policy
  (ppo_v5_noinstr, sampled rollout 1 of 16) on hard-suite task 339 (jovian
  family, level 309, R = 919 u), with precedence hazard > chaotic >
  contested > well-behaved.  Contested: eta > 0.1 at the craft; hazard:
  inside a debris or radiation zone or an atmosphere, or taking damage
  faster than a tenth of the craft's health per episode; chaotic: the
  exponent measured from the craft's own state, one tick of thrust as the
  perturbation and the ballistic factor divided out as in Section 3.4,
  exceeds the critical 0.0150 /s.  The map crops the flown region: the giant
  with its atmosphere and two radiation belts (grey hatch), three moons at
  true radius with their paths over the episode, two sensor zones (dashed,
  harmless), and ink rings the targets where reached (1 at 66 s, a station
  riding beside a moon; 2 at 128 s, inside the inner belt; 3 not reached).
  The flight crosses the radiation belts three times from 102 s, 23 s inside
  in all.  The bar is the state sequence over the 150 s episode with the
  time of each change beneath it: well-behaved on the approach; contested
  from 37 s to 79 s while a moon's pull rivals the giant's (eta reaches 2.0
  at 26 u from the moon); well-behaved again on the transfer; hazard on each
  belt crossing; and two one-second chaotic samples at close passes of a
  moon (66 s at 23 u, 147 s at 60 u).  Fractions of ticks well-behaved /
  contested / chaotic / hazard: 55 / 29 / 1 / 15% on this rollout and 53 /
  27 / 2 / 18% averaged over the 16 rollouts; with the raw 20 s finite-time
  exponent of Figure~\ref{fig:phase} labelling the chaotic state instead,
  32 / 12 / 42 / 15% and 30 / 10 / 41 / 18%.  The lowest strip shows both
  exponents along the flight against the critical value (dashed): the raw
  exponent sits above it for most of the time spent near the giant, which
  is tidal shear (ADR 0039); the paper's own reading, which colours the bar,
  crosses it on 2 of the 150 one-second samples.  Drawn by
  \code{scripts/fig\_regimes.py}.

Colours: the four states are Okabe-Ito blue, bluish green, vermilion and a
violet (#9B5DE5) that passes the dataviz palette validator against the other
three under protan/deutan simulation (the house purple #8E44AD does not:
Delta E 1.2 from the blue under deutan).  Everything that is not a state is
grey or black: bodies, their paths, the hazard zones (grey hatch), the
atmosphere (pale grey), the targets (ink rings), so no shading on the map can
be mistaken for a state.  The star keeps its yellow.
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
from matplotlib.legend_handler import HandlerPatch
from matplotlib.lines import Line2D
from matplotlib.patches import Annulus, Circle, Ellipse, FancyArrowPatch, FancyBboxPatch, Patch, Polygon

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

# house style (scripts/paper_figures.py) for everything that is not a state
GREY = "#6b7280"
BODY_FACE, BODY_EDGE = "#d6dae2", "#6b7280"
STAR_FACE, STAR_EDGE = "#f2b134", "#c98a1a"
ATMO = "#dfe3ea"                          # pale grey, so it cannot read as a state
ZONE_FACE, ZONE_EDGE = "#eceef2", "#9aa0ad"
INK = "#1f2430"
# the four states: Okabe-Ito blue, bluish green, vermilion, and a violet validated against them
S_WELL, S_CONT, S_CHAOS, S_HAZ = "#0072B2", "#009E73", "#9B5DE5", "#D55E00"
STATE_NAMES = ["well-behaved", "contested", "chaotic", "hazard"]
STATE_JOBS = ["plan", "re-model", "bound the exit", "avoid"]
STATE_COLS = [S_WELL, S_CONT, S_CHAOS, S_HAZ]
HAZ_LABEL = {0: "star's radiation", 1: "atmosphere", 2: "debris belt", 3: "radiation belt"}
plt.rcParams.update({
    "figure.dpi": 150, "savefig.dpi": 200, "savefig.bbox": "tight",
    "font.family": "serif", "font.serif": ["DejaVu Serif"], "font.size": 8.0,
    "axes.labelsize": 8.0, "axes.titlesize": 8.5, "legend.fontsize": 7.5,
    "xtick.labelsize": 7.0, "ytick.labelsize": 7.0,
    "axes.spines.top": False, "axes.spines.right": False,
    "axes.linewidth": 0.6, "xtick.major.width": 0.6, "ytick.major.width": 0.6,
    "lines.solid_capstyle": "round", "pdf.fonttype": 42, "hatch.linewidth": 0.4,
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

    # eta and the hazard geometry on every tick of every rollout, per zone and per body
    def per_tick(x, bp):
        full = P.gravity_at(x[None], bp, level, 0)[0]
        dom = P.gravity_at(x[None], bp, level, 1)[0]
        eta = jnp.sqrt(jnp.sum((full - dom) ** 2)) / (jnp.sqrt(jnp.sum(full ** 2)) + 1e-12)
        w, _, _ = P.zone_weights(level, bp, x)
        harmful = (level.zone_kind == C.ZONE_DEBRIS) | (level.zone_kind == C.ZONE_RADIATION)
        zone_hit = (w > 0.0) & harmful & level.zone_active
        d = x[None] - bp
        r = jnp.sqrt(jnp.sum(d * d, -1))
        atmo_hit = level.active & (level.atmo_h > 0) & (r - level.radius < level.atmo_h)
        return eta, zone_hit, atmo_hit
    eta, zone_hit, atmo_hit = jax.jit(jax.vmap(jax.vmap(per_tick, in_axes=(0, 0)), in_axes=(0, None)))(
        jnp.asarray(tr["pos"]), bps[1:T + 1])
    eta, zone_hit, atmo_hit = np.asarray(eta), np.asarray(zone_hit), np.asarray(atmo_hit)

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
                health=tr["health"], ends=ends, status=status, eta=eta,
                zone_hit=zone_hit, atmo_hit=atmo_hit,
                in_zone=zone_hit.any(-1), in_atmo=atmo_hit.any(-1), idx=idx, lam20=lam20, lam_bc=lam_bc,
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


def hazards_met(D, r):
    """Which hazards rollout r is in, tick by tick: a list of (label, kind index,
    anchor body or zone index, first tick, tick count), ordered by first entry.
    Kinds follow physics.hazard_terms: 0 star radiation, 1 atmosphere, 2 debris
    zone, 3 radiation zone."""
    n = int(D["ends"][r])
    L, K = D["level"], D["C"]
    out = []
    for z in np.nonzero(L["zone_active"])[0]:
        hit = D["zone_hit"][r, :n, z]
        if hit.any():
            kind = 2 if int(L["zone_kind"][z]) == K["ZONE_DEBRIS"] else 3
            out.append((HAZ_LABEL[kind], kind, int(z), int(np.nonzero(hit)[0][0]), int(hit.sum())))
    for b in np.nonzero(L["active"])[0]:
        hit = D["atmo_hit"][r, :n, b]
        if hit.any():
            out.append((HAZ_LABEL[1], 1, int(b), int(np.nonzero(hit)[0][0]), int(hit.sum())))
    covered = D["in_zone"][r, :n] | D["in_atmo"][r, :n]
    rad = (D["damage"][r, :n, 0] > K["RAD_THR"]) & ~covered
    if rad.any():
        out.append((HAZ_LABEL[0], 0, -1, int(np.nonzero(rad)[0][0]), int(rad.sum())))
    return sorted(out, key=lambda t: t[3])


# ------------------------------------------------------------------ helpers
def letter(fig, x, y, s):
    fig.text(x, y, s, fontsize=9.5, fontweight="bold", va="top", ha="left", color=INK)


def frame(ax):
    ax.set_xticks([]); ax.set_yticks([])
    for sp in ax.spines.values():
        sp.set_visible(True); sp.set_color("#c9ced9"); sp.set_linewidth(0.5)


def ship_glyph(ax, x, y, angle, size, color=INK, zorder=10):
    c, s = np.cos(angle), np.sin(angle)
    pts = np.array([[1.0, 0.0], [-0.7, 0.6], [-0.35, 0.0], [-0.7, -0.6]]) * size
    pts = pts @ np.array([[c, s], [-s, c]]) + np.array([x, y])
    ax.add_patch(Polygon(pts, closed=True, facecolor=color, edgecolor="white", lw=0.5, zorder=zorder))


def arrow(ax, p, q, color=GREY, lw=0.8, ms=7, rad=0.0, zorder=5, ls="-", style="-|>"):
    ax.add_patch(FancyArrowPatch(p, q, arrowstyle=style, mutation_scale=ms, color=color, lw=lw,
                                 connectionstyle=f"arc3,rad={rad}", zorder=zorder, linestyle=ls,
                                 shrinkA=0, shrinkB=0))


class HandlerRing(HandlerPatch):
    """A legend handle drawn as a ring (for targets, sensor zones, the play disc)."""
    def __init__(self, **kw):
        super().__init__(patch_func=self._ring, **kw)

    @staticmethod
    def _ring(legend, orig_handle, xdescent, ydescent, width, height, fontsize):
        p = Circle((width / 2 - xdescent, height / 2 - ydescent), 0.42 * height)
        p.update_from(orig_handle)
        return p


class HandlerDisc(HandlerPatch):
    """A legend handle drawn as a filled disc (bodies, the star)."""
    def __init__(self, scale=0.45, **kw):
        self.scale = scale
        super().__init__(patch_func=self._disc, **kw)

    def _disc(self, legend, orig_handle, xdescent, ydescent, width, height, fontsize):
        p = Circle((width / 2 - xdescent, height / 2 - ydescent), self.scale * height)
        p.update_from(orig_handle)
        return p


# ------------------------------------------------------------------- panels
def panel_machine(ax):
    """The state machine: boxes are regimes; arrows are what the field and the
    craft's path admit; the italic line under each title is the navigator's job."""
    ax.set_xlim(0, 100); ax.set_ylim(0, 100); ax.axis("off")
    # (centre x, centre y, width, height, title, job, what it means, colour, fill, dashed)
    boxes = [
        (20, 84, 36, 20, "well-behaved", "plan ahead", "a one- or few-body\nmodel holds", S_WELL, "white", False),
        (80, 84, 36, 20, "contested", "re-model", "no body dominates;\nstill predictable", S_CONT, "white", False),
        (50, 46, 46, 34, "chaotic encounter", "bound the exit",
         "nearby paths diverge; only the\nexit set is predictable", S_CHAOS, "#f3edfb", False),
        (50, 8, 90, 15.5, "hazard", "avoid",
         "radiation, debris, atmosphere: the route\nis constrained independently of gravity", S_HAZ, "white", True),
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
            ax.add_patch(Circle((ex, ey), 1.5, facecolor=S_CHAOS, edgecolor="none", zorder=6))
            ax.add_patch(Ellipse((ox, oy), 15, 9, facecolor="none", edgecolor=S_CHAOS, lw=0.8,
                                 ls=(0, (2, 1.5)), zorder=6))
            for k in range(7):
                a = (k - 3) / 3.0
                tx, ty = ox + 4.5 * a * np.cos(0.4 * a) - 1.0, oy + 3.0 * np.sin(1.4 * a + 0.3)
                ax.annotate("", (tx, ty), (ex + 1.7, ey), arrowprops=dict(
                    arrowstyle="-", color=S_CHAOS, lw=0.45, alpha=0.7,
                    connectionstyle=f"arc3,rad={0.25 * a}"), zorder=6)
                ax.plot([tx], [ty], marker="o", ms=1.6, color=S_CHAOS, zorder=7, ls="none")
            ax.text(ex, ey - 3.0, "entry set", fontsize=SMALL - 0.8, color=S_CHAOS, ha="center",
                    va="top", zorder=6)
            ax.text(ox, oy - 5.4, "exit set, bounded", fontsize=SMALL - 0.8, color=S_CHAOS,
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
    arrow(ax, (27, 52), (11, 74), rad=0.35, color=S_CHAOS)
    lab(2, 54, "exits within\na known set", ha="left", color=S_CHAOS)
    arrow(ax, (50, 29), (50, 15.75), style="<|-|>", color=S_HAZ)
    lab(53, 22.5, "entered from, and\nleft to, any regime", ha="left", color=S_HAZ)


def place_label(ax, anchor, avoid, half, cx, cy, span):
    """Choose a text position for a leader label: among sixteen directions at two
    radii from the anchor, the one inside the map farthest from the points to avoid."""
    best, best_d = None, -1.0
    for rr in (0.22, 0.30):
        for k in range(16):
            a = 2 * np.pi * k / 16
            p = anchor + rr * span * np.array([np.cos(a), np.sin(a)])
            if abs(p[0] - cx) > 0.78 * half or abs(p[1] - cy) > 0.80 * half:
                continue
            d = np.min(np.sqrt(np.sum((avoid - p) ** 2, -1)))
            if d > best_d:
                best, best_d = p, d
    return best if best is not None else anchor + np.array([0.0, 0.2 * span])


def panel_flight(ax_map, ax_bar, ax_lam, D, r, chaos):
    """Returns (hits, hazards met, legend entries drawn)."""
    L, T, K = D["level"], D["task"], D["C"]
    n = int(D["ends"][r])
    st = states(D, r, chaos)
    pos = D["pos"][r, :n]
    bps, wpos = D["bps"], D["wpos"]
    R = K["R"]
    act = L["active"]
    dt = K["CTRL_DT"]
    legend = {}       # label -> handle, in drawing order

    # the map: the flown region, its targets and the bodies within reach of it
    hits = []
    for k in range(int(np.sum(T["wp_active"]))):
        nz = np.nonzero(D["visited"][r, :n] > k)[0]
        hits.append(int(nz[0]) + 1 if len(nz) else None)
    tpos = [wpos[hit if hit is not None else 0, k] for k, hit in enumerate(hits)]
    ext = np.concatenate([pos, np.array(tpos).reshape(-1, 2)], 0)
    lo, hi = ext.min(0), ext.max(0)
    near = [bps[0, i] for i in np.nonzero(act)[0]
            if int(L["kind"][i]) != K["KIND_TRACER"] and L["mass"][i] > 0
            and np.all(bps[0, i] > lo - 0.25 * (hi - lo)) and np.all(bps[0, i] < hi + 0.25 * (hi - lo))]
    ext = np.concatenate([ext] + [np.asarray(b)[None] for b in near], 0)
    lo, hi = ext.min(0), ext.max(0)
    span = float(max(hi[0] - lo[0], hi[1] - lo[1])) * 1.15 + 60
    cx, cy = (lo + hi) / 2
    half = span / 2
    inside = lambda q: abs(q[0] - cx) < 0.95 * half and abs(q[1] - cy) < 0.95 * half
    in_view = lambda q, rad: abs(q[0] - cx) < half + rad and abs(q[1] - cy) < half + rad   # drawn in the crop
    ax_map.set_xlim(cx - half, cx + half); ax_map.set_ylim(cy - half, cy + half)
    ax_map.set_aspect("equal")
    if R - np.sqrt(cx ** 2 + cy ** 2) < 1.5 * half:
        ax_map.add_patch(Circle((0, 0), R, facecolor="none", edgecolor=GREY, lw=0.6, ls=(0, (1, 2)), zorder=1))
        legend["play disc"] = Circle((0, 0), 1, facecolor="none", edgecolor=GREY, lw=0.6, ls=(0, (1, 2)))
    zone_patch = {}
    for z in np.nonzero(L["zone_active"])[0]:
        a = int(L["zone_anchor"][z])
        zx, zy = (bps[0, a] if a >= 0 else (0.0, 0.0))
        r_in, r_out = float(L["zone_r_in"][z]), float(L["zone_r_out"][z])
        kind = int(L["zone_kind"][z])
        if not in_view((zx, zy), r_out):
            continue
        if kind == K["ZONE_DEBRIS"]:
            p = Annulus((zx, zy), r_out, r_out - max(r_in, 1e-3), facecolor=ZONE_FACE, hatch="////",
                        edgecolor=ZONE_EDGE, lw=0.4, zorder=1)
            legend["debris belt (hazard)"] = Patch(facecolor=ZONE_FACE, hatch="////", edgecolor=ZONE_EDGE, lw=0.4)
        elif kind == K["ZONE_RADIATION"]:
            p = Annulus((zx, zy), r_out, r_out - max(r_in, 1e-3), facecolor=ZONE_FACE, hatch="xxxx",
                        edgecolor=ZONE_EDGE, lw=0.4, zorder=1)
            legend["radiation belt (hazard)"] = Patch(facecolor=ZONE_FACE, hatch="xxxx", edgecolor=ZONE_EDGE, lw=0.4)
        elif kind == K["ZONE_SENSOR"]:
            p = Circle((zx, zy), r_out, facecolor="none", edgecolor=GREY, lw=0.5, ls=(0, (3, 2)), zorder=2, alpha=0.8)
            legend["sensor zone (harmless)"] = Circle((0, 0), 1, facecolor="none", edgecolor=GREY, lw=0.5, ls=(0, (3, 2)))
        else:
            continue
        ax_map.add_patch(p)
        zone_patch[int(z)] = p
    anchors = set(int(a) for a in T["wp_anchor"][T["wp_active"]] if a >= 0)
    for i in np.nonzero(act)[0]:
        kind = int(L["kind"][i])
        if kind == K["KIND_TRACER"] or (kind == K["KIND_STATION"] and i in anchors):
            continue
        ax_map.plot(bps[:, i, 0], bps[:, i, 1], color=GREY, lw=0.45, alpha=0.55, zorder=2)
        legend.setdefault("body paths over the episode", Line2D([], [], color=GREY, lw=0.6, alpha=0.7))
    for i in np.nonzero(act)[0]:
        x, y = bps[0, i]
        kind = int(L["kind"][i])
        rad = float(L["radius"][i])
        if kind == K["KIND_TRACER"] or not in_view((x, y), rad + float(L["atmo_h"][i])):
            continue
        if kind == K["KIND_STATION"]:
            ax_map.plot([x], [y], marker="s", ms=3.0, color=GREY, mfc="white", mew=0.7, zorder=7)
            legend.setdefault("station", Line2D([], [], marker="s", ms=3.0, color=GREY, mfc="white", mew=0.7, ls="none"))
            continue
        if L["lum"][i] > 0:
            for k, a in ((4.0, 0.06), (2.8, 0.10), (1.8, 0.18)):
                ax_map.add_patch(Circle((x, y), rad * k, facecolor=STAR_FACE, edgecolor="none", alpha=a, zorder=3))
            ax_map.add_patch(Circle((x, y), rad, facecolor=STAR_FACE, edgecolor=STAR_EDGE, lw=0.5, zorder=5))
            legend.setdefault("luminous star", Circle((0, 0), 1, facecolor=STAR_FACE, edgecolor=STAR_EDGE, lw=0.5))
        else:
            if L["atmo_h"][i] > 0:
                ax_map.add_patch(Circle((x, y), rad + float(L["atmo_h"][i]), facecolor=ATMO,
                                        edgecolor=ZONE_EDGE, lw=0.3, zorder=4))
                legend.setdefault("atmosphere (hazard)", Circle((0, 0), 1, facecolor=ATMO, edgecolor=ZONE_EDGE, lw=0.3))
            ax_map.add_patch(Circle((x, y), rad, facecolor=BODY_FACE, edgecolor=BODY_EDGE, lw=0.5, zorder=5))
            legend.setdefault("body, true radius", Circle((0, 0), 1, facecolor=BODY_FACE, edgecolor=BODY_EDGE, lw=0.5))
            if rad < 8:
                ax_map.plot([x], [y], marker="o", ms=5, mfc="none", mec=GREY, mew=0.5, zorder=6)

    # targets, numbered, drawn where they were when reached
    for k, hit in enumerate(hits):
        pa = wpos[:, k]
        if T["wp_anchor"][k] >= 0:
            ax_map.plot(pa[:, 0], pa[:, 1], color=INK, lw=0.5, alpha=0.35, ls=(0, (1, 1.5)), zorder=6)
        at = pa[hit] if hit is not None else pa[0]
        if hit is not None:
            ax_map.add_patch(Circle(at, K["TARGET_RADIUS"], facecolor="none", edgecolor=INK, lw=0.9, zorder=7))
            legend["target, reached"] = Circle((0, 0), 1, facecolor="none", edgecolor=INK, lw=0.9)
        else:
            ax_map.add_patch(Circle(at, K["TARGET_RADIUS"], facecolor="none", edgecolor=INK, lw=0.7,
                                    ls=(0, (2, 1.5)), zorder=7))
            legend["target, not reached"] = Circle((0, 0), 1, facecolor="none", edgecolor=INK, lw=0.7, ls=(0, (2, 1.5)))
        if inside(at):
            ax_map.text(at[0] + K["TARGET_RADIUS"] * 1.3, at[1] + K["TARGET_RADIUS"] * 1.0, str(k + 1),
                        fontsize=SMALL, color=INK, fontweight="bold", ha="left", va="bottom",
                        zorder=20, clip_on=True)

    # the flight, coloured by state, on a white underlay so it reads over the hatch
    seg = np.stack([pos[:-1], pos[1:]], 1)
    ax_map.add_collection(LineCollection(seg, colors="white", linewidths=2.4, zorder=8))
    ax_map.add_collection(LineCollection(seg, colors=[STATE_COLS[s] for s in st[1:]], linewidths=1.2, zorder=9))
    sp = T["start_pos"]
    ship_glyph(ax_map, sp[0], sp[1], float(T["start_angle"]), 0.022 * span, zorder=11)
    ax_map.text(sp[0], sp[1] - 0.03 * span, "start", fontsize=SMALL, color=INK, ha="center", va="top",
                zorder=20, clip_on=True)
    ep = pos[-1]
    ax_map.plot([ep[0]], [ep[1]], marker="x", ms=4, color=INK, mew=0.9, zorder=12)
    legend["start (craft's heading)"] = Line2D([], [], marker=(3, 0, -90), ms=5, color=INK, ls="none")
    legend["end of episode"] = Line2D([], [], marker="x", ms=4, color=INK, mew=0.9, ls="none")

    # the hazards the flight crosses, labelled with a leader
    met = hazards_met(D, r)
    avoid = np.concatenate([pos[::5]] + [np.asarray(b)[None] for b in near] + [np.array(tpos).reshape(-1, 2)], 0)
    labelled = set()
    for lab_, kind, which, t0, cnt in met:
        if lab_ in labelled:          # one label per hazard kind (two belts read as one)
            continue
        labelled.add(lab_)
        if sum(m[0] == lab_ for m in met) > 1 and lab_.endswith("belt"):
            lab_ = lab_ + "s"
        anchor = pos[min(t0 + min(cnt, 30) // 2, n - 1)]
        p = place_label(ax_map, anchor, avoid, half, cx, cy, span)
        ax_map.annotate(lab_, xy=anchor, xytext=p, fontsize=SMALL - 0.4, color=INK, ha="center", va="center",
                        zorder=21, arrowprops=dict(arrowstyle="-", color=INK, lw=0.5, shrinkA=0, shrinkB=2),
                        bbox=dict(boxstyle="round,pad=0.15", facecolor="white", edgecolor="none", alpha=0.85))
        avoid = np.concatenate([avoid, p[None]], 0)

    # scale bar
    x0, y0 = cx - 0.93 * half, cy - 0.93 * half
    ax_map.plot([x0, x0 + 200], [y0, y0], color=INK, lw=1.2, zorder=20, solid_capstyle="butt")
    ax_map.text(x0 + 100, y0 + 0.015 * span, "200 u", fontsize=SMALL, ha="center", va="bottom", color=INK, zorder=20)
    frame(ax_map)

    # the timeline bar: the state sequence, with the time of each state change
    Tm = K["T_MAX"]
    cmap = ListedColormap(STATE_COLS)
    ax_bar.imshow(st[None, :], aspect="auto", cmap=cmap, vmin=-0.5, vmax=3.5,
                  extent=(0, n * dt, 0, 1), interpolation="nearest", zorder=2)
    if n * dt < Tm:
        ax_bar.axvspan(n * dt, Tm, facecolor="#eceef2", edgecolor="none", zorder=1)
    changes = np.nonzero(st[1:] != st[:-1])[0] + 1
    last = -1e9
    for c in changes:
        t = c * dt
        ax_bar.plot([t, t], [0, 1], color="white", lw=0.4, zorder=3)
        if t - last >= 12.0:
            ax_bar.text(t, -0.2, f"{t:.0f}", fontsize=SMALL - 1.0, color=INK, ha="center", va="top", clip_on=False)
            last = t
    for k, hit in enumerate(hits):
        if hit is not None:
            ax_bar.plot([hit * dt], [1.0], marker="v", ms=4, color=INK, mec="white", mew=0.4,
                        clip_on=False, zorder=6, ls="none")
            ax_bar.text(hit * dt, 1.25, str(k + 1), fontsize=SMALL - 0.4, color=INK, fontweight="bold",
                        ha="center", va="bottom", clip_on=False)
    ax_bar.set_xlim(0, Tm); ax_bar.set_ylim(0, 1)
    ax_bar.set_xticks([]); ax_bar.set_yticks([])
    for sp_ in ax_bar.spines.values():
        sp_.set_visible(False)
    ax_bar.set_ylabel("state", fontsize=SMALL, rotation=0, ha="right", va="center", labelpad=4)
    ax_bar.text(0, 1.25, "targets reached:" if any(h is not None for h in hits) else "no target reached",
                fontsize=SMALL - 0.8, color=INK, ha="left", va="bottom", clip_on=False)

    # the exponents along the flight, against the critical value
    t_s = (D["idx"] + 1) * dt
    ok = D["idx"] < n
    floor = 3e-6
    ax_lam.axhline(LAM_CRIT, color=INK, lw=0.6, ls=(0, (3, 2)), zorder=3)
    ax_lam.plot(t_s[ok], np.maximum(D["lam20"][r, ok], floor), color=GREY, lw=0.8, zorder=4)
    ax_lam.plot(t_s[ok], np.maximum(D["lam_bc"][r, ok], floor), color=INK, lw=0.8, zorder=4)
    ax_lam.set_yscale("log"); ax_lam.set_ylim(floor, 0.5)
    ax_lam.set_yticks([1e-4, 1e-2]); ax_lam.set_yticklabels([r"$10^{-4}$", r"$10^{-2}$"])
    ax_lam.set_yticks([], minor=True)
    ax_lam.tick_params(labelsize=SMALL - 0.6, length=2, width=0.5, pad=1.5)
    ax_lam.set_xlim(0, Tm)
    ax_lam.set_xticks([0, 50, 100, 150])
    ax_lam.set_xlabel("episode time (s)", labelpad=1.5)
    ax_lam.set_ylabel(r"$\lambda$ (s$^{-1}$)", fontsize=SMALL, labelpad=2)
    ax_lam.spines["left"].set_linewidth(0.5); ax_lam.spines["bottom"].set_linewidth(0.5)
    legend[f"raw {FTLE_T:.0f} s FTLE"] = Line2D([], [], color=GREY, lw=0.8)
    legend["ballistic factor divided out"] = Line2D([], [], color=INK, lw=0.8)
    legend[r"$\lambda_{\mathrm{crit}}$ = " + f"{LAM_CRIT} /s"] = Line2D([], [], color=INK, lw=0.6, ls=(0, (3, 2)))
    return hits, met, legend


def caption(D, r, chaos, hits, met):
    """The draft caption, filled with the measured numbers."""
    from spacenav.types import REASONS
    per_b, mean_b = fractions(D, "ballistic")
    per_f, mean_f = fractions(D, "ftle20")
    dt = D["C"]["CTRL_DT"]
    n = int(D["ends"][r])
    st = states(D, r, chaos)
    changes = np.nonzero(st[1:] != st[:-1])[0] + 1
    seq = ", ".join(f"{STATE_NAMES[st[c]]} at {c * dt:.0f} s" for c in changes)
    pct = lambda v: " / ".join(f"{100 * x:.0f}%" for x in v)
    reached = [f"{k + 1} at {h * dt:.0f} s" for k, h in enumerate(hits) if h is not None]
    missed = [str(k + 1) for k, h in enumerate(hits) if h is None]
    # hazard crossings grouped by kind: number of entries into the hazard state, time inside
    hz = (st == 3)
    entries = int(np.sum(hz[1:] & ~hz[:-1]) + hz[0])
    kinds = {}
    for lab_, _, _, t0, cnt in met:
        kinds.setdefault(lab_, [0, 10 ** 9])
        kinds[lab_][0] += cnt
        kinds[lab_][1] = min(kinds[lab_][1], t0)
    haz = "; ".join(f"the {lab_}{'s' if sum(m[0] == lab_ for m in met) > 1 and lab_.endswith('belt') else ''} "
                    f"from {t0 * dt:.0f} s, {cnt * dt:.0f} s inside" for lab_, (cnt, t0) in kinds.items())
    haz = f"{haz} ({entries} crossing{'s' if entries != 1 else ''} in all)" if haz else ""
    lam = D["lam20"] if chaos == "ftle20" else D["lam_bc"]
    cross = [f"{(D['idx'][k] + 1) * dt:.0f} s" for k in np.nonzero(lam[r] > LAM_CRIT)[0] if D["idx"][k] < n]
    crossing = (f"crosses it on {len(cross)} of the {int(np.sum(D['idx'] < n))} one-second samples "
                f"({', '.join(cross)})" if cross else "never crosses it")
    return (
        f"Figure N.  Regimes of predictability as a state machine, and the states along a flight.  "
        f"(a) The first picture of Section 2: the states are regimes, not positions.  In a well-behaved "
        f"field a one- or few-body model holds and the navigator plans ahead; in a contested field no "
        f"single body dominates, the model must be richer, but the dynamics remain predictable; in a "
        f"chaotic encounter nearby trajectories diverge, so only the exit set is predictable, and "
        f"the region acts as a noise filter mapping the set of entry states onto a bounded set of exits "
        f"for which the plan on the far side is made; a hazard (radiation, debris, atmosphere) "
        f"constrains the route independently of gravity and can be entered from any regime.  "
        f"(b) The states measured along a flight of a trained policy ({D['run']}, sampled rollout "
        f"{r + 1} of {len(D['ends'])}) on {D['suite']}-suite task {D['ti']} ({D['family']} family, level "
        f"{D['level_idx']}, R = {D['C']['R']:.0f} u), with precedence hazard > chaotic > contested > "
        f"well-behaved.  Contested: eta > {ETA_THR} at the craft; hazard: inside a debris or radiation "
        f"zone or an atmosphere, or taking damage faster than a tenth of the craft's health per episode; "
        f"chaotic: the exponent measured from the craft's own state, one tick of thrust as the "
        f"perturbation and the ballistic factor divided out as in Section 3.4, exceeds the critical "
        f"{LAM_CRIT} /s.  The map crops the flown region; bodies at true radius, grey curves their paths "
        f"over the episode, grey hatching the hazard zones, ink rings the targets where reached "
        f"({', '.join(reached) if reached else 'none reached'}"
        f"{'; ' + ', '.join(missed) + ' not reached' if missed else ''}).  The flight crosses "
        f"{haz if haz else 'no hazard'}.  The bar is the state sequence over the {D['C']['T_MAX']:.0f} s "
        f"episode ({REASONS[int(D['status'][r])]} at {n * dt:.0f} s), with the time of each change "
        f"beneath it: {seq if seq else 'no change'}.  Fractions of ticks well-behaved / contested / "
        f"chaotic / hazard: {pct(per_b[r])} on this rollout and {pct(mean_b)} averaged over the 16 "
        f"rollouts; with the raw 20 s finite-time exponent of Figure~\\ref{{fig:phase}} labelling the "
        f"chaotic state instead, {pct(per_f[r])} and {pct(mean_f)}.  The lowest strip shows both "
        f"exponents along the flight against the critical value (dashed): the raw exponent is tidal "
        f"shear beside the dominant body (ADR 0039); the paper's own reading, which colours the bar, "
        f"{crossing}.  Drawn by \\code{{scripts/fig\\_regimes.py}}."
    )


# --------------------------------------------------------------------- main
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--task", default="hard:339", help="suite:index of the flight drawn in (b)")
    ap.add_argument("--roll", type=int, default=0, help="which of the sampled rollouts to draw")
    ap.add_argument("--chaos", default="ballistic", choices=["ftle20", "ballistic"],
                    help="which exponent labels the chaotic state")
    ap.add_argument("--run", default="ppo_v5_noinstr")
    ap.add_argument("--rollouts", type=int, default=16)
    ap.add_argument("--seed", type=int, default=7, help="the ghost fleet's sampling seed")
    ap.add_argument("--all", action="store_true", help="also measure the other three ghost tasks")
    ap.add_argument("--cache", default=None, help="pickle the measurements here and reuse them")
    ap.add_argument("--out", default=str(OUT))
    ap.add_argument("--caption", action="store_true", help="print the draft caption with the numbers filled in")
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
    from spacenav.types import REASONS
    for key, D in data.items():
        outcomes = {}
        for s in D["status"]:
            outcomes[s] = outcomes.get(s, 0) + 1
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
    W, H = 6.6, 3.4
    fig = plt.figure(figsize=(W, H))
    axes = lambda x, y, w, h: fig.add_axes([x / W, y / H, w / W, h / H])
    ax_a = axes(0.05, 0.08, 2.95, 3.05)
    ax_map = axes(3.32, 1.17, 2.0, 2.0)
    ax_bar = axes(3.32, 0.90, 2.0, 0.15)
    ax_lam = axes(3.32, 0.30, 2.0, 0.40)
    panel_machine(ax_a)
    hits, met, legend = panel_flight(ax_map, ax_bar, ax_lam, D, args.roll, args.chaos)
    letter(fig, 0.02 / W, 3.37 / H, "(a)  the state machine")
    letter(fig, 3.2 / W, 3.37 / H, "(b)  the states along a flight")

    # legend for (b): every drawn element, states first
    handles = [Line2D([], [], color=c, lw=2.2) for c in STATE_COLS]
    labels = [f"{nm} ({job})" for nm, job in zip(STATE_NAMES, STATE_JOBS)]
    order = ["luminous star", "body, true radius", "station", "body paths over the episode",
             "atmosphere (hazard)", "debris belt (hazard)", "radiation belt (hazard)",
             "sensor zone (harmless)", "target, reached", "target, not reached",
             "start (craft's heading)", "end of episode", "play disc",
             f"raw {FTLE_T:.0f} s FTLE", "ballistic factor divided out",
             r"$\lambda_{\mathrm{crit}}$ = " + f"{LAM_CRIT} /s"]
    for k in order:
        if k in legend:
            handles.append(legend[k]); labels.append(k)
    hmap = {Circle: HandlerRing()}
    for h in handles:
        if isinstance(h, Circle) and h.get_facecolor()[-1] > 0 and h.get_facecolor()[:3] != (0, 0, 0):
            hmap[h] = HandlerDisc()
    leg = fig.legend(handles, labels, loc="upper left", bbox_to_anchor=(5.38 / W, 3.19 / H), frameon=False,
                     fontsize=SMALL - 0.8, handlelength=1.5, handleheight=0.9, labelspacing=0.32,
                     borderaxespad=0, handletextpad=0.6, handler_map=hmap)
    for t in leg.get_texts():
        t.set_color(INK)
    fig.text(5.44 / W, 0.06 / H,
             f"{D['family'].replace('_', ' ')} family, level {D['level_idx']},\nR = {D['C']['R']:.0f} u; "
             f"{D['suite']} suite, task {D['ti']};\n{D['run']}, rollout {args.roll + 1} of {len(D['ends'])};\n"
             "precedence hazard > chaotic\n> contested > well-behaved",
             fontsize=SMALL - 1.0, color=GREY, va="bottom", ha="left", linespacing=1.15)
    out = pathlib.Path(args.out)
    fig.savefig(out / "regimes.pdf"); fig.savefig(out / "regimes.png", dpi=300)
    plt.close(fig)
    print("wrote", out / "regimes.pdf", out / "regimes.png", flush=True)
    if args.caption:
        print("\n" + caption(D, args.roll, args.chaos, hits, met))


if __name__ == "__main__":
    main()
