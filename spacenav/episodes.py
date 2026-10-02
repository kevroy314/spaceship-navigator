"""Concrete episodes for the web client: level + task + initial body state.

An episode is identified by (pool, level index, seed, rendezvous flag) and is
fully deterministic, so humans and agents can be compared on the same universe.
The client gets a few-KB description and integrates the body ephemeris itself
from `init`; `ephemeris()` (the JAX-computed version, ~1 MB) remains for tests.
"""

import functools
import hashlib
import re
from pathlib import Path

import jax
import jax.numpy as jnp
import numpy as np

from spacenav import baselines
from spacenav import constants as C
from spacenav import env as E
from spacenav.levels.build import index_level, load_pool
from spacenav.levels.families import FAMILY_NAMES
from spacenav.physics import roll_bodies
from spacenav.types import REASONS, Task

DATA = Path(__file__).resolve().parents[1] / "data"
N_FRAMES = C.MAX_EPISODE_TICKS * C.SUBSTEPS + 1


@functools.lru_cache(maxsize=None)
def pool(name):
    levels, meta = load_pool(DATA / "pools" / f"{name}.npz")
    return levels, meta


def pool_names():
    return sorted(p.stem for p in (DATA / "pools").glob("*.npz"))


# A mission spec says what shape the mission has: how many stops, what share of
# them ride an orbit rather than sitting still, and whether any must be docked
# with.  Written "<stops>w<moving%>m" plus "d" for docking, e.g. 4w50m = four
# stops, half of them moving, all fly-through.  The old single letters still work.
SPEC_RE = re.compile(r"^([1-9])w(\d{1,3})m(d?)(c?)(l?)$")
LEGACY = {"f": "1w0m", "r": "1w100md", "t": "3w50m"}


def spec_to_config(spec):
    """Mission spec -> TaskConfig.  Human missions get a comfortable tank: the
    tight fuel budgets the agent trains on are not what makes flying fun."""
    m = SPEC_RE.match(spec)
    if not m:
        raise ValueError(f"bad mission spec {spec!r}")
    stops = min(int(m.group(1)), C.MAX_WAYPOINTS)
    moving = min(int(m.group(2)), 100) / 100.0
    dock = bool(m.group(3))
    chaos = bool(m.group(4))
    low = bool(m.group(5))
    # A "low thrust" mission shrinks the engine until gravity's impulse over the
    # flight is comparable to the whole tank, which is what makes modelling the
    # field worth anything: a ship that can overpower gravity never has to
    # understand it.  The floor is set by the clock, not by taste -- a tour takes
    # about sum(2*sqrt(leg/accel)) seconds, so an engine below ~0.3 cannot finish
    # inside MAX_EPISODE_TIME and the level becomes infeasible rather than hard.
    return E.TaskConfig(
        waypoints=(stops, stops), p_station=moving,
        p_rendezvous=1.0 if dock else 0.0,
        min_dist=110.0, min_leg=110.0,
        max_dist=(280.0 if stops > 1 else 420.0) if low else (700.0 if stops > 1 else 900.0),
        deltav_budget=(0.9, 1.45),
        accel_range=(0.5, 1.0) if low else (1.3, 3.0),
        p_contested=1.0 if chaos else 0.0,
    )


def normalise_spec(spec):
    spec = LEGACY.get(spec, spec)
    m = SPEC_RE.match(spec)
    if not m:
        raise ValueError(f"unknown mission kind {spec!r}")
    return (f"{min(int(m.group(1)), C.MAX_WAYPOINTS)}w{min(int(m.group(2)), 100)}m"
            f"{m.group(3)}{m.group(4)}{m.group(5)}")


def spec_summary(spec):
    m = SPEC_RE.match(spec)
    return dict(stops=int(m.group(1)), moving=int(m.group(2)), docking=bool(m.group(3)),
                chaos=bool(m.group(4)), low_thrust=bool(m.group(5)))


def episode_id(pool_name, index, seed, mode="1w0m"):
    if mode is True or mode is False:          # legacy: rendezvous flag
        mode = "r" if mode else "f"
    return f"{pool_name}-{index}-{seed}-{normalise_spec(mode)}"


def parse_id(eid):
    pool_name, index, seed, mode = eid.rsplit("-", 3)
    return pool_name, int(index), int(seed), normalise_spec(mode)


@functools.lru_cache(maxsize=256)
def _level_task(eid):
    pool_name, index, seed, mode = parse_id(eid)
    levels, _ = pool(pool_name)
    level = index_level(levels, index)
    task, ok = jax.jit(E.sample_task, static_argnums=2)(
        jax.random.PRNGKey(seed), level, spec_to_config(mode))
    return level, task, bool(ok)


_roll = jax.jit(roll_bodies, static_argnums=3)


@functools.lru_cache(maxsize=32)
def ephemeris(eid) -> bytes:
    level, task, _ = _level_task(eid)
    n = int(np.sum(np.asarray(level.active)))
    ps, vs = _roll(level, level.snap_pos[task.snapshot], level.snap_vel[task.snapshot], N_FRAMES - 1)
    arr = np.concatenate([np.asarray(ps)[:, :n], np.asarray(vs)[:, :n]], axis=-1)  # (F, n, 4)
    return arr.astype("<f4").tobytes()


def constants_dict():
    names = ["PHYS_DT", "SUBSTEPS", "CTRL_DT", "MAX_EPISODE_TIME", "MAX_EPISODE_TICKS",
             "SHIP_ACCEL", "SHIP_TURN_RATE", "SHIP_FUEL", "SHIP_HEALTH", "MAX_WAYPOINTS",
             "ATMO_SCALE_FRAC",
             "ATMO_DRAG", "ATMO_HEAT", "RADIATION_K", "DEBRIS_K", "ZONE_EDGE",
             "OUT_OF_BOUNDS_FACTOR", "TARGET_RADIUS", "RENDEZVOUS_V_TOL"]
    out = {k: getattr(C, k) for k in names}
    out["KIND_NAMES"] = C.KIND_NAMES
    out["ZONE_NAMES"] = C.ZONE_NAMES
    out["OBJECTIVES"] = list(C.OBJECTIVES)
    out["REASONS"] = REASONS
    return out


def describe(eid):
    level, task, ok = _level_task(eid)
    pool_name, index, seed, mode = parse_id(eid)
    lv = jax.tree.map(np.asarray, level)
    tk = jax.tree.map(np.asarray, task)
    n = int(lv.active.sum())
    bodies = [dict(kind=int(lv.kind[i]), mass=float(lv.mass[i]), radius=float(lv.radius[i]),
                   atmo_h=float(lv.atmo_h[i]), atmo_rho=float(lv.atmo_rho[i]),
                   lum=float(lv.lum[i]), parent=int(lv.parent[i])) for i in range(n)]
    zones = [dict(anchor=int(lv.zone_anchor[z]), r_in=float(lv.zone_r_in[z]),
                  r_out=float(lv.zone_r_out[z]), kind=int(lv.zone_kind[z]),
                  strength=float(lv.zone_strength[z]), spin=float(lv.zone_spin[z]),
                  exposure=bool(tk.exposure_mask[z]))
             for z in range(len(lv.zone_active)) if lv.zone_active[z]]
    return dict(
        id=eid, pool=pool_name, index=index, seed=seed, mode=mode,
        mission=spec_summary(mode), valid=ok,
        family=FAMILY_NAMES[int(lv.family)], level_radius=float(lv.level_radius),
        snapshot=int(tk.snapshot), bodies=bodies, zones=zones,
        # initial body state: the client integrates the ephemeris itself (no big download)
        init=np.concatenate([lv.snap_pos[int(tk.snapshot)][:n], lv.snap_vel[int(tk.snapshot)][:n]],
                            axis=-1).astype(np.float64).tolist(),
        task=dict(start_pos=tk.start_pos.tolist(), start_vel=tk.start_vel.tolist(),
                  start_angle=float(tk.start_angle),
                  waypoints=[dict(anchor=int(tk.wp_anchor[w]), pos=tk.wp_pos[w].tolist(),
                                  offset=tk.wp_offset[w].tolist(),
                                  radius=float(tk.wp_radius[w]),
                                  v_tol=float(min(tk.wp_v_tol[w], 1e9)),
                                  rendezvous=bool(tk.wp_v_tol[w] < C.FLYTHROUGH_V_TOL / 2))
                             for w in range(len(tk.wp_active)) if tk.wp_active[w]],
                  order_free=bool(tk.order_free), accel=float(tk.accel), fuel=float(tk.fuel),
                  weights=tk.weights.tolist(), ref_scale=tk.ref_scale.tolist()),
        ephemeris=dict(frames=N_FRAMES, bodies=n, dt=C.PHYS_DT, layout="frame,body,[px,py,vx,vy] float32 LE"),
        constants=constants_dict(),
    )


# ---------------------------------------------------------------------------
# simulating runs in JAX (agents, and re-checking human runs)
# ---------------------------------------------------------------------------

def _summary(level, task, final, trace):
    tr = jax.tree.map(np.asarray, trace)
    status = tr["status"]
    done = np.nonzero(status != 0)[0]
    end = int(done[0]) + 1 if len(done) else len(status)
    costs = final.costs
    return dict(
        status=REASONS[int(final.status)],
        ticks=end,
        costs=dict(time=float(costs.time), path=float(costs.path), fuel=float(costs.fuel),
                   exposure=float(costs.exposure), damage=float(costs.damage)),
        objective=float(E.objective_cost(task, costs)),
        ret=float(tr["reward"][:end].sum()),
        trajectory=np.concatenate([
            np.concatenate([np.asarray(task.start_pos)[None], tr["pos"][:end]]),
            np.concatenate([[float(task.start_angle)], tr["angle"][:end]])[:, None],
            np.concatenate([[0.0], tr["thrust"][:end]])[:, None],
            np.concatenate([[C.SHIP_HEALTH], tr["health"][:end]])[:, None],
            np.concatenate([[float(task.fuel)], tr["fuel"][:end]])[:, None],
        ], axis=1).round(3).tolist(),
        columns=["x", "y", "angle", "thrust", "health", "fuel"],
    )


@functools.partial(jax.jit, static_argnums=2)
def _rollout_policy(level, task, name):
    pol = getattr(baselines, name)(level, task)
    return E.rollout(level, task, pol)


@jax.jit
def _rollout_actions(level, task, actions):
    return E.rollout_actions(level, task, actions)


def simulate_baseline(eid, name="pilot"):
    level, task, _ = _level_task(eid)
    final, trace = _rollout_policy(level, task, name)
    return _summary(level, task, final, trace)


def simulate_actions(eid, actions):
    level, task, _ = _level_task(eid)
    a = np.zeros((C.MAX_EPISODE_TICKS, 2), np.float32)
    actions = np.asarray(actions, np.float32).reshape(-1, 2)[: C.MAX_EPISODE_TICKS]
    a[: len(actions)] = actions
    final, trace = _rollout_actions(level, task, jnp.asarray(a))
    return _summary(level, task, final, trace)


def run_hash(payload: bytes) -> str:
    return hashlib.sha1(payload).hexdigest()[:10]
