"""The task figure: one page that shows a reader everything the task is.

    XLA_PYTHON_CLIENT_PREALLOCATE=false XLA_PYTHON_CLIENT_MEM_FRACTION=0.15 \\
        python scripts/fig_task.py [--suite hard --task 63 --run ppo_v5_noinstr]

(JAX_PLATFORMS=cpu also works: the only simulation is one 150 s body roll and
32 policy rollouts, under a minute on a CPU.)

Writes paper/figures/task.pdf and task.png.  Panels:

  (a) a real level (the frozen hard suite, task 63, level 1563 of val_seen,
      family asteroid_field): bodies at true radius, their paths over one
      150 s episode, the luminous star, atmospheres, a circulating debris
      belt, a sensor zone, the three targets of the tour and the path a
      trained policy actually flew;
  (b) the observation at one instant of that flight, in the craft's own frame;
  (c) the nine actions;
  (d) the goal kinds;
  (e) the loadout;
  (f) the episode clock, with the flown episode's fuel and health;
  (g) how an episode ends, and units.

The label positions in (a) are hand-placed for this task; another task will
need its own (see LABELS).

Draft caption (the coordinator owns the final wording):

  Figure 1.  The task.  (a) One level from the hard evaluation suite
  (asteroid_field family, play disc R = 625 u).  Discs are bodies at true
  radius; the star is luminous (radiation damage falls as L/r^2) and the two
  giants carry atmospheres (drag and heating, pale rings).  Bodies smaller
  than 8 u carry an open locator ring.  Grey curves are the bodies' paths over
  the 150 s episode; the shaded annulus is a debris belt that circulates with
  the star's field; the dashed circle is a sensor zone, which does no harm.
  Green rings are the three targets of the tour, numbered in order: a fixed
  point to fly through, a station to fly through, and a rendezvous hung 56 u
  off the outer giant, to be met at a relative speed under 3 u/s.  Moving
  targets drag their own paths; a ring is drawn where the target was when the
  craft reached it.  The blue curve is the path a trained policy flew: it
  reached the first two targets at 43 s and 83 s and then circled the star
  until the clock ran out without matching the giant's speed.  The dot marks
  the instant shown in (b).  (b) What the policy is given at that instant:
  everything is relative to the craft and rotated so its nose points along
  +x.  Arrows are drawn in observation units (velocity over 20 u/s, local
  gravity over 4 u/s^2); the three bodies that pull hardest are shown with
  their distance and pull.  Each body is one token; the self block carries
  the craft, the loadout and the tour.  (c) The action set; one action is
  held for one control tick.  (d) The four goal kinds.  (e) The loadout is
  drawn per task: engine acceleration and tank size, with the tank set to a
  multiple of a direct-flight estimate.  (f) The episode is 2250 control
  ticks at 15 Hz, each four physics substeps at 60 Hz; fuel and health of the
  flight in (a), with the times its targets were reached.  (g) The five ways
  an episode ends; an empty tank is not one of them.  Units.
"""
import argparse
import pathlib
import pickle
import sys

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.patches import Annulus, Circle, FancyArrowPatch, Polygon, Rectangle

ROOT = pathlib.Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
OUT = ROOT / "paper" / "figures"
OUT.mkdir(parents=True, exist_ok=True)

# house style (scripts/paper_figures.py)
BLUE, ORANGE, GREEN, RED, GREY = "#2a78d6", "#eb6834", "#0ca30c", "#d03b3b", "#6b7280"
BODY_FACE, BODY_EDGE = "#d6dae2", "#6b7280"
STAR_FACE = "#f2b134"
ATMO = "#8fb8e8"
INK = "#1f2430"
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
BOX = dict(boxstyle="round,pad=0.15", fc="white", ec="none", alpha=0.8)

# label anchors for panel (a), in units of the play radius R, for hard/63
LABELS = dict(
    # (x, y, ha, va, side of the ring the leader meets)
    wp=[(-1.08, 0.64, "left", "bottom", "top"), (0.34, -0.88, "left", "top", "right"),
        (0.40, -0.56, "left", "top", "bottom")],
    star=(-0.30, 0.90, "left", "center"),
    giant=(0.33, 0.66, "left", "bottom"),
    rock=(-1.08, -0.78, "left", "top"),
    sensor=(0.57, 0.16, "left", "center"),
    belt=(0.72, -0.30),
    start=(0.05, -0.69, "center", "top"),
    mid=(-1.08, -0.44, "left", "top"),
    end=(-1.08, -0.62, "left", "top"),
)


# --------------------------------------------------------------------- data
def load(suite, task_i, run, n_roll, seed):
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
    from spacenav.types import REASONS

    # the checkpoint was trained with the panel zeroed; what it sees must match
    E.set_instruments("off")

    suites = pickle.load(open(ROOT / "data" / "eval" / "suites.pkl", "rb"))
    s = suites[suite]
    pool, meta = load_pool(ROOT / "data" / "pools" / f"{s['pool']}.npz")
    lvl_i = int(np.asarray(s["level_idx"])[task_i])
    level = jax.tree.map(lambda x: x[lvl_i], pool)
    task = jax.tree.map(lambda x: jnp.asarray(x)[task_i], s["task"])
    print(f"suite {suite} task {task_i}: pool {s['pool']} level {lvl_i} "
          f"family {FAMILY_NAMES[int(level.family)]} meta {meta[lvl_i]}", flush=True)

    # bodies and targets over the episode, on the control-tick grid
    bp0, bv0 = E.snapshot_state(level, task.snapshot)
    bps, bvs = P.roll_bodies(level, bp0, bv0, C.MAX_EPISODE_TICKS * C.SUBSTEPS)
    bps = np.asarray(bps)[:: C.SUBSTEPS]
    bvs = np.asarray(bvs)[:: C.SUBSTEPS]
    wpos, wvel = jax.vmap(lambda p, v: E.waypoint_states(task, p, v))(jnp.asarray(bps),
                                                                      jnp.asarray(bvs))
    wpos, wvel = np.asarray(wpos), np.asarray(wvel)

    # fly it with the run's last checkpoint, best of n sampled rollouts
    s0 = E.reset(level, task)
    obs0 = E.observe(level, task, s0)
    sizes = {k: tuple(np.asarray(v).shape) for k, v in obs0.items()}
    print("observation shapes:", sizes, flush=True)
    net = ActorCritic(d=PPOConfig().d_model)
    template = net.init(jax.random.PRNGKey(0), obs0)
    ckpt = sorted((ROOT / "data" / "rl" / run).glob("ckpt_*.msgpack"))[-1]
    params = serialization.from_bytes(template, ckpt.read_bytes())
    print(f"policy: {ckpt.relative_to(ROOT)}", flush=True)

    def one(key):
        def policy(obs, state):
            logits, _ = net.apply(params, obs)
            return ACTIONS[jax.random.categorical(jax.random.fold_in(key, state.tick), logits)]
        return E.rollout(level, task, policy)

    keys = jax.random.split(jax.random.PRNGKey(seed), n_roll)
    _, tr = jax.jit(jax.vmap(one))(keys)
    tr = {k: np.asarray(v) for k, v in tr.items()}
    st = tr["status"]
    ends = np.array([int(np.nonzero(r != 0)[0][0]) + 1 if np.any(r != 0) else len(r) for r in st])
    vis = np.array([tr["visited"][r][ends[r] - 1] for r in range(n_roll)])
    arrived = np.array([st[r][ends[r] - 1] == 1 for r in range(n_roll)])
    # most targets visited, then arrived, then the longest flight
    order = np.lexsort((ends, arrived, vis))
    best = int(order[-1])
    end = int(ends[best])
    print(f"{n_roll} rollouts: visited {np.bincount(vis, minlength=4)}, "
          f"arrived {int(arrived.sum())}; using rollout {best}: "
          f"{REASONS[int(st[best][end - 1])]} after {end} ticks, "
          f"{int(vis[best])} targets", flush=True)
    flight = {k: v[best][:end] for k, v in tr.items()}
    flight["end"] = end
    flight["status"] = int(st[best][end - 1])
    # the tick at which each target was reached
    hits = []
    for k in range(int(np.sum(np.asarray(task.wp_active)))):
        nz = np.nonzero(flight["visited"] > k)[0]
        hits.append(int(nz[0]) + 1 if len(nz) else None)
    flight["hits"] = hits
    print("targets reached at ticks", hits, flush=True)

    # the instant shown in (b): seven tenths of the way along the first leg
    mid = int(0.7 * hits[0]) if hits[0] else max(2, int(0.4 * end))
    pos, vel, ang = flight["pos"][mid], flight["vel"][mid], float(flight["angle"][mid])
    g = np.asarray(P.gravity_at(jnp.asarray(pos)[None], jnp.asarray(bps[mid + 1]), level)[0])
    # legs of the tour as sampled (start -> targets at the snapshot), for the loadout
    chain = np.concatenate([np.asarray(task.start_pos)[None], wpos[0]], 0)
    legs = np.sqrt(np.sum((chain[1:] - chain[:-1]) ** 2, -1))
    act = np.asarray(task.wp_active)
    legs = np.where(act, legs, 0.0)
    dv_needed = float(E.deltav_estimate(jnp.asarray(legs), task.wp_v_tol, task.accel))

    L = {k: np.asarray(getattr(level, k)) for k in level._fields}
    T = {k: np.asarray(getattr(task, k)) for k in task._fields}
    return dict(level=L, task=T, bps=bps, bvs=bvs, wpos=wpos, wvel=wvel, flight=flight,
                mid=mid, mid_pos=pos, mid_vel=vel, mid_ang=ang, mid_g=g, dv_needed=dv_needed,
                family=FAMILY_NAMES[int(level.family)], level_idx=lvl_i, sizes=sizes,
                C=dict(R=float(level.level_radius), TARGET_RADIUS=C.TARGET_RADIUS,
                       RDV=C.RENDEZVOUS_V_TOL, OOB=C.OUT_OF_BOUNDS_FACTOR,
                       SHIP_ACCEL=C.SHIP_ACCEL, TURN=C.SHIP_TURN_RATE,
                       ACCEL_RANGE=C.SHIP_ACCEL_RANGE, FUEL_LIMITS=C.SHIP_FUEL_LIMITS,
                       BUDGET=C.DELTAV_BUDGET_RANGE, T_MAX=C.MAX_EPISODE_TIME,
                       TICKS=C.MAX_EPISODE_TICKS, SUBSTEPS=C.SUBSTEPS, PHYS_DT=C.PHYS_DT,
                       CTRL_DT=C.CTRL_DT, HEALTH=C.SHIP_HEALTH, MAX_WP=C.MAX_WAYPOINTS,
                       KIND_NAMES=C.KIND_NAMES, KIND_STATION=C.KIND_STATION,
                       KIND_TRACER=C.KIND_TRACER, ZONE_DEBRIS=C.ZONE_DEBRIS,
                       ZONE_RADIATION=C.ZONE_RADIATION, ZONE_SENSOR=C.ZONE_SENSOR),
                reasons=REASONS)


# ------------------------------------------------------------------ helpers
def letter(fig, x, y, s):
    fig.text(x, y, s, fontsize=9.5, fontweight="bold", va="top", ha="left", color=INK)


def frame(ax):
    ax.set_xticks([]); ax.set_yticks([])
    for sp in ax.spines.values():
        sp.set_visible(True); sp.set_color("#c9ced9"); sp.set_linewidth(0.5)


def ship_glyph(ax, x, y, angle, size, color=BLUE, zorder=10, lw=0.5):
    """A small arrowhead-shaped craft pointing along `angle` (radians)."""
    c, s = np.cos(angle), np.sin(angle)
    pts = np.array([[1.0, 0.0], [-0.7, 0.6], [-0.35, 0.0], [-0.7, -0.6]]) * size
    pts = pts @ np.array([[c, s], [-s, c]]) + np.array([x, y])
    ax.add_patch(Polygon(pts, closed=True, facecolor=color, edgecolor="white", lw=lw,
                         zorder=zorder))


def arrow(ax, p, q, color, lw=1.0, ms=6, ls="-", zorder=8, alpha=1.0):
    ax.add_patch(FancyArrowPatch(p, q, arrowstyle="-|>", mutation_scale=ms, color=color,
                                 lw=lw, linestyle=ls, zorder=zorder, alpha=alpha,
                                 shrinkA=0, shrinkB=0))


def ring(ax, x, y, r, color=GREEN, lw=1.0, ls="-", zorder=6, alpha=1.0):
    ax.add_patch(Circle((x, y), r, facecolor="none", edgecolor=color, lw=lw, ls=ls,
                        zorder=zorder, alpha=alpha))


def note(ax, text, at, where, R, color=INK, leader=GREY, bold=False, box=True, relpos=None):
    """Annotation with its text at a hand-placed spot (in units of R).  `relpos`
    is where on the text box the leader starts, (0, 1) being its top-left."""
    x, y, ha, va = where[:4]
    props = dict(arrowstyle="-", color=leader, lw=0.45, shrinkA=0, shrinkB=1)
    if relpos is not None:
        props["relpos"] = relpos
    ax.annotate(text, at, xytext=(x * R, y * R), fontsize=SMALL, color=color, ha=ha, va=va,
                fontweight="bold" if bold else "normal", arrowprops=props,
                bbox=BOX if box else None, zorder=20)


# ------------------------------------------------------------------- panels
def panel_level(ax, D):
    L, T, K = D["level"], D["task"], D["C"]
    bps, wpos, fl = D["bps"], D["wpos"], D["flight"]
    R = K["R"]
    half = 1.15 * R
    act = L["active"]

    # play disc, belt, sensor zones
    ax.add_patch(Circle((0, 0), R, facecolor="none", edgecolor=GREY, lw=0.6, ls=(0, (1, 2)),
                        zorder=1))
    ax.text(-0.985 * half, 0.985 * half, f"play disc, R = {R:.0f} u (dotted)\n"
            f"lost beyond {K['OOB']} R", fontsize=SMALL, color=GREY, va="top", ha="left",
            zorder=20)
    for z in np.nonzero(L["zone_active"])[0]:
        a = int(L["zone_anchor"][z])
        cx, cy = (bps[0, a] if a >= 0 else (0.0, 0.0))
        r_in, r_out = float(L["zone_r_in"][z]), float(L["zone_r_out"][z])
        kind = int(L["zone_kind"][z])
        if kind == K["ZONE_DEBRIS"]:
            ax.add_patch(Annulus((cx, cy), r_out, r_out - max(r_in, 1e-3), facecolor=ORANGE,
                                 alpha=0.13, edgecolor=ORANGE, lw=0.4, zorder=1))
            bx, by = LABELS["belt"]
            ax.text(bx * R, by * R, f"debris belt\n{r_in:.0f}–{r_out:.0f} u, circulating\n"
                    "damage ∝ speed through it", fontsize=SMALL, color=ORANGE, ha="center",
                    va="center", zorder=20)
        elif kind == K["ZONE_RADIATION"]:
            ax.add_patch(Annulus((cx, cy), r_out, r_out - max(r_in, 1e-3), facecolor=RED,
                                 alpha=0.15, edgecolor=RED, lw=0.4, zorder=1))
        elif kind == K["ZONE_SENSOR"]:
            ring(ax, cx, cy, r_out, color=GREY, lw=0.5, ls=(0, (3, 2)), zorder=2, alpha=0.8)
            note(ax, "sensor zone\n(harmless)", (cx + r_out * 0.95, cy + r_out * 0.3),
                 LABELS["sensor"], R, color=GREY)

    # body paths over the episode (a station's path is its target's path, drawn below)
    anchors = set(int(a) for a in T["wp_anchor"][T["wp_active"]] if a >= 0)
    for i in np.nonzero(act)[0]:
        kind = int(L["kind"][i])
        if kind == K["KIND_TRACER"] or (kind == K["KIND_STATION"] and i in anchors):
            continue
        pa = bps[:, i]
        ax.plot(pa[:, 0], pa[:, 1], color=GREY, lw=0.45, alpha=0.55, zorder=2)

    # bodies at true radius
    for i in np.nonzero(act)[0]:
        x, y = bps[0, i]
        kind = int(L["kind"][i])
        r = float(L["radius"][i])
        if kind == K["KIND_TRACER"]:
            continue
        if kind == K["KIND_STATION"]:
            ax.plot([x], [y], marker="s", ms=3.2, color=GREY, mfc="white", mew=0.7, zorder=7)
            continue
        if L["lum"][i] > 0:
            for k, a in ((4.0, 0.06), (2.8, 0.10), (1.8, 0.18)):
                ax.add_patch(Circle((x, y), r * k, facecolor=STAR_FACE, edgecolor="none",
                                    alpha=a, zorder=3))
            ax.add_patch(Circle((x, y), r, facecolor=STAR_FACE, edgecolor=ORANGE, lw=0.5,
                                zorder=5))
        else:
            if L["atmo_h"][i] > 0:
                ax.add_patch(Circle((x, y), r + float(L["atmo_h"][i]), facecolor=ATMO,
                                    edgecolor="none", alpha=0.55, zorder=4))
            ax.add_patch(Circle((x, y), r, facecolor=BODY_FACE, edgecolor=BODY_EDGE, lw=0.5,
                                zorder=5))
            if r < 8:
                ax.plot([x], [y], marker="o", ms=6, mfc="none", mec=GREY, mew=0.5, zorder=6)

    # labels for the bodies that matter
    star = [i for i in np.nonzero(act)[0] if L["lum"][i] > 0]
    if star:
        i = star[0]
        x, y = bps[0, i]
        note(ax, f"star, GM = {L['mass'][i]:.0f}, luminous:\ndamage ∝ L / r²",
             (x - 0.3 * L["radius"][i], y + 0.95 * L["radius"][i]), LABELS["star"], R)
    giants = [i for i in np.nonzero(act)[0] if L["atmo_h"][i] > 0]
    if giants:
        i = giants[0]
        x, y = bps[0, i]
        note(ax, "giant with an atmosphere\n(pale ring: drag and heating)",
             (x, y + L["radius"][i] + L["atmo_h"][i]), LABELS["giant"], R)
    rocks = [i for i in np.nonzero(act)[0] if L["radius"][i] < 8 and L["mass"][i] > 0]
    if rocks:
        i = rocks[np.argmin(bps[0, rocks, 1])]
        x, y = bps[0, i]
        note(ax, "asteroid, r < 8 u\n(open ring = locator)", (x - 7, y), LABELS["rock"], R)

    # the tour: targets, their paths, numbered in order
    for n, k in enumerate(np.nonzero(T["wp_active"])[0]):
        anchored = T["wp_anchor"][k] >= 0
        rdv = T["wp_v_tol"][k] < 10
        pa = wpos[:, k]
        if anchored:
            ax.plot(pa[:, 0], pa[:, 1], color=GREEN, lw=0.8, alpha=0.55, zorder=6)
        hit = fl["hits"][k]
        at = pa[hit] if hit is not None else pa[0]
        ring(ax, pa[0, 0], pa[0, 1], K["TARGET_RADIUS"], lw=0.7, alpha=0.45 if hit else 1.0)
        if hit is not None:
            ring(ax, at[0], at[1], K["TARGET_RADIUS"], lw=1.1)
        if anchored:
            kind = K["KIND_NAMES"][int(L["kind"][T["wp_anchor"][k]])]
            off = float(np.hypot(*T["wp_offset"][k]))
            what = "station" if kind == "station" else f"hung {off:.0f} u off a {kind}"
        else:
            what = "fixed point"
        what += (f",\nrendezvous (< {K['RDV']:g} u/s)" if rdv else ", fly through")
        if hit is None:
            what += "\nnot reached"
        x, y, ha, va, side = LABELS["wp"][n]
        rr = K["TARGET_RADIUS"]
        anchor = {"top": (at[0], at[1] + rr), "bottom": (at[0], at[1] - rr),
                  "right": (at[0] + rr, at[1]), "left": (at[0] - rr, at[1])}[side]
        note(ax, f"{n + 1}  {what}", anchor, (x, y, ha, va), R, color=GREEN, leader=GREEN,
             bold=True, relpos=(0, 1) if va == "top" else None)

    # the flight
    p = fl["pos"]
    ax.plot(p[:, 0], p[:, 1], color=BLUE, lw=0.9, zorder=8)
    sp = T["start_pos"]
    ship_glyph(ax, sp[0], sp[1], float(T["start_angle"]), 14, zorder=11)
    note(ax, "start", (sp[0], sp[1] - 8), LABELS["start"], R, color=BLUE, leader=BLUE)
    mp = D["mid_pos"]
    ax.plot([mp[0]], [mp[1]], marker="o", ms=4.5, color=BLUE, mec="white", mew=0.6, zorder=12)
    note(ax, f"t = {D['mid'] * K['CTRL_DT']:.0f} s, shown in (b)", (mp[0], mp[1]),
         LABELS["mid"], R, color=BLUE, leader=BLUE)
    ep = p[-1]
    ax.plot([ep[0]], [ep[1]], marker="x", ms=4, color=BLUE, mew=0.9, zorder=12)
    reason = D["reasons"][fl["status"]]
    note(ax, f"end: {reason} at {fl['end'] * K['CTRL_DT']:.0f} s", (ep[0], ep[1]),
         LABELS["end"], R, color=BLUE, leader=BLUE)

    # scale bar, bottom left
    x0, y0 = -0.93 * half, -0.93 * half
    ax.plot([x0, x0 + 200], [y0, y0], color=INK, lw=1.2, zorder=20, solid_capstyle="butt")
    ax.text(x0 + 100, y0 + 0.02 * half, "200 u", fontsize=SMALL, ha="center", va="bottom",
            color=INK, zorder=20)
    ax.text(0.985 * half, 0.985 * half,
            f"{D['family'].replace('_', ' ')} family, level {D['level_idx']}\n"
            f"grey: the bodies' paths over 150 s", fontsize=SMALL, color=GREY, ha="right",
            va="top", zorder=20)

    ax.set_xlim(-half, half); ax.set_ylim(-half, half); ax.set_aspect("equal")
    frame(ax)


def panel_observation(ax, D):
    L, T, K = D["level"], D["task"], D["C"]
    mid = D["mid"]
    pos, vel, ang, g = D["mid_pos"], D["mid_vel"], D["mid_ang"], D["mid_g"]
    bps = D["bps"][mid + 1]
    wpos, wvel = D["wpos"][mid + 1], D["wvel"][mid + 1]
    c, s = np.cos(-ang), np.sin(-ang)
    Rm = np.array([[c, -s], [s, c]])
    rot = lambda v: v @ Rm.T

    act = L["active"] & (L["kind"] != K["KIND_TRACER"])
    rel = rot(bps - pos[None])
    dist = np.hypot(rel[:, 0], rel[:, 1])
    pull = np.where(act, L["mass"], 0.0) / np.maximum(dist, L["radius"]) ** 2
    near = list(np.argsort(pull)[::-1][:3])
    leg = int(D["flight"]["leg"][mid])
    t_rel = rot(wpos[leg] - pos)
    t_vrel = rot(wvel[leg] - vel)
    half = 1.32 * max(max(dist[near]), np.hypot(*t_rel))

    # observation units: v / 20 u/s, g / 4 u/s^2, drawn at a common display scale
    vscale = 0.42 * half
    v_arrow = rot(vel) / 20.0 * vscale
    g_arrow = rot(g) / K["SHIP_ACCEL"] * vscale

    labels = []     # text boxes placed after a repulsion pass so none overlap
    # (x, y, half-width[, half-height]) of things labels must not cover
    obstacles = [(0.0, 0.0, 0.10 * half),
                 (-0.10 * half, 0.84 * half, 0.90 * half, 0.16 * half)]   # the corner text

    def add(x, y, text, color, anchor=None, rank=2):
        labels.append(dict(x=float(x), y=float(y), text=text, color=color, anchor=anchor,
                           rank=rank))

    # nose axis
    ax.plot([0, half], [0, 0], color="#c9ced9", lw=0.6, zorder=1)
    add(0.86 * half, 0.04 * half, "nose, +x", GREY)
    labels[-1]["fixed"] = True
    labels[-1]["rank"] = 0

    # relative position vectors to the bodies that pull hardest
    for i in near:
        ax.plot([0, rel[i, 0]], [0, rel[i, 1]], color=GREY, lw=0.5, ls=(0, (2, 2)), zorder=3)
        r = max(float(L["radius"][i]), 0.012 * half)
        face = STAR_FACE if L["lum"][i] > 0 else BODY_FACE
        edge = ORANGE if L["lum"][i] > 0 else BODY_EDGE
        if L["atmo_h"][i] > 0:
            ax.add_patch(Circle((rel[i, 0], rel[i, 1]), r + L["atmo_h"][i], facecolor=ATMO,
                                edgecolor="none", alpha=0.5, zorder=4))
        ax.add_patch(Circle((rel[i, 0], rel[i, 1]), r, facecolor=face, edgecolor=edge, lw=0.5,
                            zorder=5))
        obstacles.append((rel[i, 0], rel[i, 1], r + L["atmo_h"][i] + 0.03 * half))
        kind = K["KIND_NAMES"][int(L["kind"][i])].replace("_", " ")
        u = rel[i] / dist[i]
        lx, ly = rel[i] + u * (r + 0.10 * half)
        add(lx, ly, f"{kind}, d = {dist[i]:.0f} u\npull {pull[i]:.2g} u/s²", INK,
            anchor=(rel[i, 0] + u[0] * r, rel[i, 1] + u[1] * r))

    # target
    ax.plot([0, t_rel[0]], [0, t_rel[1]], color=GREEN, lw=0.6, ls=(0, (2, 2)), zorder=3)
    ring(ax, t_rel[0], t_rel[1], K["TARGET_RADIUS"], lw=1.0, zorder=6)
    tv = t_vrel / 20.0 * vscale
    if np.hypot(*tv) > 0.02 * half:
        arrow(ax, (t_rel[0], t_rel[1]), (t_rel[0] + tv[0], t_rel[1] + tv[1]), GREEN, lw=0.9,
              ms=6)
    rdv = T["wp_v_tol"][leg] < 10
    obstacles.append((t_rel[0], t_rel[1], K["TARGET_RADIUS"] + 0.03 * half))
    obstacles.append((t_rel[0] + 0.5 * tv[0], t_rel[1] + 0.5 * tv[1], 0.5 * np.hypot(*tv)))
    u = t_rel / max(np.hypot(*t_rel), 1e-6)
    lx, ly = t_rel + u * 0.14 * half
    add(lx, ly, f"target {leg + 1}, d = {np.hypot(*t_rel):.0f} u\n"
        f"relative speed {np.hypot(*t_vrel):.1f} u/s"
        + (f"\nmust be < {K['RDV']:g} u/s to count" if rdv else "\nany speed counts"), GREEN,
        anchor=(t_rel[0] + u[0] * K["TARGET_RADIUS"], t_rel[1] + u[1] * K["TARGET_RADIUS"]),
        rank=1)

    # the craft, its velocity and the local gravity
    ship_glyph(ax, 0, 0, 0.0, 0.05 * half, zorder=10)
    arrow(ax, (0, 0), (v_arrow[0], v_arrow[1]), BLUE, lw=1.2, ms=8, zorder=9)
    arrow(ax, (0, 0), (g_arrow[0], g_arrow[1]), RED, lw=1.2, ms=8, zorder=9)
    # arrow labels sit beside the arrow, on the side away from the other arrow
    for vec, other, txt, col in ((v_arrow, g_arrow, f"v = {np.hypot(*vel):.1f} u/s", BLUE),
                                 (g_arrow, v_arrow, f"g = {np.hypot(*g):.2f} u/s²", RED)):
        n = np.hypot(*vec)
        u = vec / max(n, 1e-6)
        perp = np.array([-u[1], u[0]])
        if np.dot(perp, other) > 0:
            perp = -perp
        add(*(0.65 * vec + perp * 0.09 * half), txt, col, anchor=tuple(0.65 * vec), rank=3)
        obstacles.append((0.5 * vec[0], 0.5 * vec[1], 0.5 * n))

    # place each label at the best of several candidate spots around what it
    # names: least overlap with labels already placed, the bodies, the arrows
    # and the frame edge
    ch, lh = 0.044 * half, 0.100 * half       # character width, line height (padded)

    def box(l):
        lines = l["text"].split("\n")
        return 0.5 * max(len(x) for x in lines) * ch, 0.5 * len(lines) * lh

    def overlap(x, y, w, h, ox, oy, ow, oh):
        return max(0.0, w + ow - abs(x - ox)) * max(0.0, h + oh - abs(y - oy))

    placed = []
    labels.sort(key=lambda l: l["rank"])      # the target first, then bodies, then arrows
    dirs = [np.array([np.cos(a), np.sin(a)]) for a in np.linspace(0, 2 * np.pi, 12, endpoint=False)]
    for l in labels:
        w, h = box(l)
        if l["anchor"] is None and l.get("fixed"):
            placed.append((l["x"], l["y"], w, h))
            continue
        base = np.array(l["anchor"]) if l["anchor"] is not None else np.array([l["x"], l["y"]])
        cands = [np.array([l["x"], l["y"]])]
        if l["anchor"] is not None:
            for k in (0.12, 0.22, 0.36, 0.52, 0.70):
                cands += [base + d * (k * half + 0.6 * (w * abs(d[0]) + h * abs(d[1]))) for d in dirs]
        best, best_cost = None, None
        for c in cands:
            x, y = c
            cost = 0.0
            cost += 4.0 * (max(0.0, abs(x) + w - 0.98 * half) + max(0.0, abs(y) + h - 0.98 * half)) * half
            for (px, py, pw, ph) in placed:
                cost += overlap(x, y, w, h, px, py, pw, ph)
            for o in obstacles:
                ox, oy, ow, oh = o if len(o) == 4 else (o[0], o[1], o[2], o[2])
                cost += overlap(x, y, w, h, ox, oy, ow, oh)
            cost += 0.5 * np.hypot(*(c - base))         # prefer staying close
            if best_cost is None or cost < best_cost:
                best, best_cost = c, cost
        l["x"], l["y"] = float(best[0]), float(best[1])
        placed.append((l["x"], l["y"], w, h))
    for l in labels:
        if l["anchor"] is not None:
            ax.plot([l["anchor"][0], l["x"]], [l["anchor"][1], l["y"]], color=GREY, lw=0.4,
                    zorder=19)
        ax.text(l["x"], l["y"], l["text"], fontsize=SMALL, color=l["color"], ha="center",
                va="center", zorder=21, bbox=BOX)
    ax.text(-0.985 * half, 0.985 * half,
            f"craft frame at t = {mid * K['CTRL_DT']:.0f} s, nose along +x\n"
            "arrows in observation units: v / 20 u/s, g / 4 u/s²\n"
            "the three bodies that pull hardest are shown",
            fontsize=SMALL, color=GREY, ha="left", va="top", zorder=20)
    ax.set_xlim(-half, half); ax.set_ylim(-half, half); ax.set_aspect("equal")
    frame(ax)


def panel_tokens(ax, D):
    """What one body token and the self block contain (no axes, text only)."""
    ax.axis("off")
    sz = D["sizes"]
    nb, fb = sz["bodies"]
    nz, fz = sz["zones"]
    fs = sz["self"][0]
    K = D["C"]
    lines = [
        (f"one body token ({fb} numbers) per body, up to {nb}:", True),
        ("relative position, direction, distance, relative velocity, its pull ÷ the", False),
        ("engine, GM, radius, atmosphere, luminosity, clearance, is-a-target, kind", False),
        (f"one zone token ({fz}) per zone, up to {nz}:", True),
        ("where the band is, its radii, signed distance to it, kind, strength, spin", False),
        (f"self block ({fs}):", True),
        ("v, g, the current target's offset, distance and relative velocity; fuel,", False),
        ("Δv left, Δv left ÷ Δv the tour still needs, engine, health, time left,", False),
        ("targets pending, order-free, damage rate, exposure, distance ÷ R, the", False),
        (f"objective weights, 4 field-readout slots (Sec. 3); 13 per target slot (×{K['MAX_WP']})", False),
        ("tokens are pooled by attention with the self block as the query, so", None),
        ("nothing depends on how many bodies there are or their order", None),
    ]
    y = 1.0
    for text, head in lines:
        ax.text(0.0, y, text, fontsize=SMALL, color=GREY if head is None else INK,
                fontweight="bold" if head else "normal", ha="left", va="top",
                transform=ax.transAxes)
        y -= 0.083


def panel_actions(ax, D):
    K = D["C"]
    turns = [("left", +1), ("none", 0), ("right", -1)]
    thr = [("full", 1.0), ("gentle", 0.3), ("off", 0.0)]
    for cx, (tname, t) in enumerate(turns):
        for cy, (gname, gth) in enumerate(thr):
            x, y = cx, 2 - cy
            ax.add_patch(Rectangle((x - 0.46, y - 0.46), 0.92, 0.92, facecolor="#f5f6f8",
                                   edgecolor="#c9ced9", lw=0.5, zorder=1))
            if gth > 0:
                fl = 0.16 + 0.3 * gth
                ax.add_patch(Polygon([[x - 0.09, y - 0.1], [x + 0.09, y - 0.1],
                                      [x, y - 0.1 - fl]], closed=True, facecolor=ORANGE,
                                     edgecolor="none", alpha=0.9, zorder=3))
            ship_glyph(ax, x, y + 0.04, np.pi / 2, 0.26, zorder=5)
            if t != 0:
                sgn = 1 if t > 0 else -1
                a0, a1 = (np.pi / 2 + 0.25 * sgn, np.pi / 2 + 1.05 * sgn)
                th = np.linspace(a0, a1, 12)
                r = 0.36
                ax.plot(x + r * np.cos(th), y + 0.04 + r * np.sin(th), color=BLUE, lw=0.9,
                        zorder=4)
                arrow(ax, (x + r * np.cos(th[-2]), y + 0.04 + r * np.sin(th[-2])),
                      (x + r * np.cos(th[-1]), y + 0.04 + r * np.sin(th[-1])), BLUE, lw=0.9,
                      ms=5, zorder=4)
    ax.text(-0.55, 2.95, "turn", fontsize=SMALL, ha="right", va="bottom", color=GREY)
    for cx, (tname, t) in enumerate(turns):
        ax.text(cx, 2.56, f"{tname}\n{t:+d}" if t else "none\n0", fontsize=SMALL,
                ha="center", va="bottom", color=INK)
    ax.text(-0.55, 2.62, "throttle", fontsize=SMALL, ha="right", va="bottom", color=GREY)
    for cy, (gname, gth) in enumerate(thr):
        ax.text(-0.55, 2 - cy, f"{gname}\n{gth:g}", fontsize=SMALL, ha="right", va="center",
                color=INK)
    ax.text(1.0, -0.6, f"turn rate {K['TURN']} rad/s, no angular\ninertia; thrust = throttle × "
            f"engine\nacceleration; an action is held\nfor one tick ({K['CTRL_DT'] * 1000:.0f} ms)",
            fontsize=SMALL, ha="center", va="top", color=GREY, linespacing=1.3)
    ax.set_xlim(-1.3, 2.55); ax.set_ylim(-1.75, 3.35); ax.set_aspect("equal"); ax.axis("off")


def panel_goals(ax, D):
    K = D["C"]
    r = K["TARGET_RADIUS"]
    W, Hc = 200.0, 150.0     # cell size in u
    cells = ["fly-through", "rendezvous", "station", "attached to a body"]
    for n, name in enumerate(cells):
        col, row = n % 2, n // 2
        cx, cy = col * W + W / 2, -row * Hc
        ax.text(cx, cy + 46, name, fontsize=SMALL + 0.4, fontweight="bold", ha="center",
                va="bottom", color=INK)
        if n == 0:
            ring(ax, cx, cy, r, lw=1.0)
            p = np.array([cx - 42, cy - 26]); q = np.array([cx + 38, cy + 24])
            ax.plot([p[0], q[0]], [p[1], q[1]], color=BLUE, lw=0.9, zorder=4)
            arrow(ax, (q[0] - 1, q[1] - 0.7), (q[0], q[1]), BLUE, lw=0.9, ms=7)
            ax.text(cx, cy - 36, f"pass within {r:.0f} u,\nat any speed", fontsize=SMALL,
                    ha="center", va="top", color=GREY)
        elif n == 1:
            ring(ax, cx - 10, cy, r, lw=1.0)
            arrow(ax, (cx - 10, cy), (cx + 28, cy + 10), GREEN, lw=1.0, ms=7)
            ax.text(cx + 31, cy + 13, "v target", fontsize=SMALL, ha="left", va="bottom",
                    color=GREEN)
            ax.add_patch(Circle((cx + 28, cy + 10), 9, facecolor=GREEN, alpha=0.15,
                                edgecolor=GREEN, lw=0.4, ls=(0, (2, 1.5))))
            arrow(ax, (cx - 14, cy - 3), (cx + 23, cy + 14), BLUE, lw=1.0, ms=7)
            ax.text(cx - 16, cy - 7, "v craft", fontsize=SMALL, ha="right", va="top",
                    color=BLUE)
            ax.text(cx, cy - 36, f"pass within {r:.0f} u, and\n|v craft − v target| < {K['RDV']:g} u/s",
                    fontsize=SMALL, ha="center", va="top", color=GREY)
        elif n == 2:
            bx, by = cx - 20, cy - 10
            ax.add_patch(Circle((bx, by), 12, facecolor=BODY_FACE, edgecolor=BODY_EDGE, lw=0.5,
                                zorder=3))
            th = np.linspace(0, 2 * np.pi, 80)
            ax.plot(bx + 32 * np.cos(th), by + 32 * np.sin(th), color=GREY, lw=0.5,
                    ls=(0, (2, 1.5)), zorder=2)
            sx, sy = bx + 32 * np.cos(0.6), by + 32 * np.sin(0.6)
            ax.plot([sx], [sy], marker="s", ms=3.2, color=GREY, mfc="white", mew=0.7, zorder=5)
            ring(ax, sx, sy, r, lw=1.0, zorder=4)
            arrow(ax, (bx + 32 * np.cos(1.5), by + 32 * np.sin(1.5)),
                  (bx + 32 * np.cos(1.7), by + 32 * np.sin(1.7)), GREY, lw=0.5, ms=5)
            ax.text(cx, cy - 46, "a massless marker\non its own orbit;\nfly through or\nrendezvous",
                    fontsize=SMALL, ha="center", va="top", color=GREY)
        else:
            th = np.linspace(-0.5, 0.5, 40)
            Rb = 170.0
            ox, oy = cx, cy - Rb - 6
            ax.plot(ox + Rb * np.sin(th), oy + Rb * np.cos(th), color=GREY, lw=0.5, zorder=2)
            bx, by = ox + Rb * np.sin(-0.1), oy + Rb * np.cos(-0.1)
            ax.add_patch(Circle((bx, by), 15, facecolor=ATMO, edgecolor="none", alpha=0.5,
                                zorder=2))
            ax.add_patch(Circle((bx, by), 11, facecolor=BODY_FACE, edgecolor=BODY_EDGE, lw=0.5,
                                zorder=3))
            Ro = Rb + 30
            ax.plot(ox + Ro * np.sin(th), oy + Ro * np.cos(th), color=GREEN, lw=0.7,
                    alpha=0.6, zorder=2)
            tx, ty = ox + Ro * np.sin(-0.1), oy + Ro * np.cos(-0.1)
            ring(ax, tx, ty, r, lw=1.0, zorder=4)
            ax.plot([bx, tx], [by, ty], color=GREY, lw=0.4, zorder=2)
            ax.text(cx, cy - 46, "rides a planet or\nmoon at a fixed\noffset; fly through\nor rendezvous",
                    fontsize=SMALL, ha="center", va="top", color=GREY)
    ax.text(W, -Hc - 135, f"a task is a tour of 1 to {K['MAX_WP']} such targets, each a ring of "
            f"radius {r:.0f} u,\nvisited in sequence or, when the task says so, in any order",
            fontsize=SMALL, ha="center", va="top", color=INK)
    ax.set_xlim(-2, 2 * W + 2); ax.set_ylim(-Hc - 165, 62); ax.set_aspect("equal"); ax.axis("off")


def panel_loadout(ax, D):
    K, T = D["C"], D["task"]
    a, f = float(T["accel"]), float(T["fuel"])
    dv = a * f
    need = D["dv_needed"]
    b = dv / need
    lo, hi = K["BUDGET"]
    tr = ax.transAxes
    ax.text(0.0, 1.0, f"engine  a ∈ [{K['ACCEL_RANGE'][0]:g}, {K['ACCEL_RANGE'][1]:g}] u/s²,\n"
            "log-uniform", fontsize=SMALL, ha="left", va="top", color=INK, transform=tr)
    ax.text(0.0, 0.87, f"tank  f ∈ [{K['FUEL_LIMITS'][0]:g}, {K['FUEL_LIMITS'][1]:g}] s of "
            "full burn", fontsize=SMALL, ha="left", va="top", color=INK, transform=tr)
    ax.text(0.0, 0.78, "Δv = a · f, with the tank sized so\nthat Δv ÷ a direct-flight estimate\n"
            f"is uniform in [{lo:g}, {hi:g}], then clipped\nto the tank limits",
            fontsize=SMALL, ha="left", va="top", color=INK, transform=tr, linespacing=1.3)
    # the bar
    y0, h = 0.36, 0.08
    xmax = 1.6
    ax.add_patch(Rectangle((0.0, y0), 1.0, h, facecolor="#eceef2", edgecolor="none",
                           transform=tr))
    ax.add_patch(Rectangle((lo / xmax, y0), (hi - lo) / xmax, h, facecolor=BLUE, alpha=0.25,
                           edgecolor="none", transform=tr))
    for v in (0.0, 0.5, 1.0, 1.5):
        ax.plot([v / xmax, v / xmax], [y0 - 0.02, y0], color=GREY, lw=0.5, transform=tr)
        ax.text(v / xmax, y0 - 0.03, f"{v:g}", fontsize=SMALL, ha="center", va="top",
                color=GREY, transform=tr)
    ax.plot([1.0 / xmax] * 2, [y0, y0 + h], color=GREY, lw=0.6, ls=(0, (2, 1.5)), transform=tr)
    ax.text(0.5, y0 - 0.11, "Δv ÷ direct-flight estimate", fontsize=SMALL, ha="center",
            va="top", color=INK, transform=tr)
    ax.text((lo + hi) / 2 / xmax, y0 + h + 0.015, "sampled range", fontsize=SMALL, ha="center",
            va="bottom", color=BLUE, transform=tr)
    bx = min(b, xmax * 0.98) / xmax
    ax.plot([bx], [y0 + h / 2], marker="v", ms=5, color=ORANGE, mec="white", mew=0.4,
            transform=tr, clip_on=False, zorder=10)
    ax.text(0.0, 0.0, f"this task (▾): a = {a:.2f} u/s², f = {f:.0f} s,\nΔv = {dv:.0f} u/s "
            f"against ≈ {need:.0f} u/s for a direct\nflight, a ratio of {b:.2f}",
            fontsize=SMALL, ha="left", va="bottom", color=ORANGE, transform=tr,
            linespacing=1.3)
    ax.axis("off")


def panel_timeline(ax, D):
    K, fl = D["C"], D["flight"]
    Tm, dt = K["T_MAX"], K["CTRL_DT"]
    t = (np.arange(fl["end"]) + 1) * dt
    ax.plot(t, fl["fuel"] / float(D["task"]["fuel"]), color=BLUE, lw=1.0, label="fuel ÷ tank")
    ax.plot(t, fl["health"] / K["HEALTH"], color=RED, lw=1.0, label="health ÷ 100")
    for k, hit in enumerate(fl["hits"]):
        if hit is None:
            continue
        ax.axvline(hit * dt, color=GREEN, lw=0.7, ls=(0, (3, 2)), zorder=2)
        ax.text(hit * dt + 1.2, 0.56 - 0.2 * k, f"target {k + 1}\n{hit * dt:.0f} s",
                fontsize=SMALL, color=GREEN, ha="left", va="center")
    te = fl["end"] * dt
    reason = D["reasons"][fl["status"]]
    ax.axvline(te, color=BLUE if fl["end"] < K["TICKS"] else GREY, lw=0.7, zorder=2)
    if fl["end"] < K["TICKS"]:
        ax.text(te + 1.2, 0.92, f"{reason}\n{te:.0f} s", fontsize=SMALL, color=BLUE,
                ha="left", va="top")
        ax.axvline(Tm, color=GREY, lw=0.7, zorder=2)
    ax.text(Tm - 1.0, 0.92, f"{reason}: tick {K['TICKS']}\n= {Tm:.0f} s", fontsize=SMALL,
            color=GREY, ha="right", va="top")
    ax.set_xlim(0, Tm); ax.set_ylim(0, 1.05)
    ax.set_xlabel("episode time (s)", labelpad=2)
    ax.set_yticks([0, 0.5, 1.0])
    sec = ax.secondary_xaxis("top", functions=(lambda x: x / dt, lambda x: x * dt))
    sec.set_xticks([0, 750, 1500, 2250])
    sec.tick_params(labelsize=SMALL, length=2, width=0.5)
    sec.spines["top"].set_linewidth(0.5)
    sec.set_xlabel(f"control tick ({1 / dt:.0f} Hz; each tick is {K['SUBSTEPS']} physics "
                   f"substeps of {K['PHYS_DT'] * 1000:.1f} ms)", fontsize=SMALL, labelpad=2)
    ax.legend(loc="lower left", frameon=False, ncol=2, handlelength=1.4, columnspacing=1.0,
              borderaxespad=0.2)


def panel_terminations(ax, D):
    ax.axis("off")
    K = D["C"]
    rows = [
        ("arrived", GREEN, "every target reached; the only success"),
        ("crashed", RED, "inside a solid body's radius"),
        ("destroyed", RED, "health ≤ 0: star radiation, atmospheric\nheating, debris, radiation belts"),
        ("lost", GREY, f"farther than {K['OOB']} R from the centre"),
        ("timeout", GREY, f"tick {K['TICKS']} = {K['T_MAX']:.0f} s, tour unfinished"),
        ("tank empty", BLUE, "not an ending: the craft coasts on"),
    ]
    ax.text(0.0, 1.0, "how an episode ends (checked every substep)", fontsize=SMALL + 0.3,
            fontweight="bold", ha="left", va="top", color=INK, transform=ax.transAxes)
    y = 0.86
    for name, col, what in rows:
        ax.text(0.0, y, name, fontsize=SMALL, color=col, fontweight="bold", ha="left", va="top",
                transform=ax.transAxes)
        ax.text(0.30, y, what, fontsize=SMALL, color=INK, ha="left", va="top",
                transform=ax.transAxes)
        y -= 0.14 if "\n" not in what else 0.22


def panel_units(ax, D):
    ax.axis("off")
    ax.text(0.0, 1.0, "units", fontsize=SMALL + 0.3, fontweight="bold", ha="left", va="top",
            color=INK, transform=ax.transAxes)
    ax.text(0.0, 0.80,
            "u  game length unit (a level is 1000–3000 u across)\n"
            "s  game second; physics at 60 Hz, the pilot at 15 Hz\n"
            "G = 1, so a body's mass is its GM, in u³/s²; a massless\n"
            "    body (station, tracer) is carried by gravity, exerts none\n"
            "speeds in u/s, accelerations in u/s²; health 0–100, damage in /s",
            fontsize=SMALL, color=INK, ha="left", va="top", transform=ax.transAxes,
            linespacing=1.35)


# --------------------------------------------------------------------- main
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--suite", default="hard")
    ap.add_argument("--task", type=int, default=63)
    ap.add_argument("--run", default="ppo_v5_noinstr")
    ap.add_argument("--rollouts", type=int, default=32)
    ap.add_argument("--seed", type=int, default=7)
    args = ap.parse_args()
    D = load(args.suite, args.task, args.run, args.rollouts, args.seed)

    W, H = 7.0, 9.75
    fig = plt.figure(figsize=(W, H))

    def axes(x, y, w, h):          # inch-based placement
        return fig.add_axes([x / W, y / H, w / W, h / H])

    top = H - 0.30
    a_s = 3.6
    ax_a = axes(0.12, top - a_s, a_s, a_s)
    ax_b = axes(3.95, top - 2.55, 2.85, 2.55)
    ax_tok = axes(3.95, top - a_s - 0.02, 2.95, 1.0)
    ax_c = axes(0.12, 3.3, 1.75, 2.05)
    ax_d = axes(2.0, 3.22, 3.15, 2.18)
    ax_e = axes(5.3, 3.3, 1.65, 2.05)
    ax_f = axes(0.45, 1.28, 3.3, 1.35)
    ax_g = axes(4.05, 1.0, 2.9, 1.9)
    ax_u = axes(0.12, 0.1, 3.8, 0.72)

    panel_level(ax_a, D)
    panel_observation(ax_b, D)
    panel_tokens(ax_tok, D)
    panel_actions(ax_c, D)
    panel_goals(ax_d, D)
    panel_loadout(ax_e, D)
    panel_timeline(ax_f, D)
    panel_terminations(ax_g, D)
    panel_units(ax_u, D)

    letter(fig, 0.02 / W, (top + 0.16) / H, "(a)  the level, and one flight")
    letter(fig, 3.85 / W, (top + 0.16) / H, "(b)  what the policy observes")
    letter(fig, 0.02 / W, 5.55 / H, "(c)  the nine actions")
    letter(fig, 1.95 / W, 5.55 / H, "(d)  goal kinds")
    letter(fig, 5.2 / W, 5.55 / H, "(e)  loadout")
    letter(fig, 0.02 / W, 3.1 / H, "(f)  the episode clock")
    letter(fig, 3.95 / W, 3.1 / H, "(g)  how an episode ends, and units")

    fig.savefig(OUT / "task.pdf")
    fig.savefig(OUT / "task.png")
    plt.close(fig)
    print("wrote", OUT / "task.pdf", OUT / "task.png", flush=True)


if __name__ == "__main__":
    main()
