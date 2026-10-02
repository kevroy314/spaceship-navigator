"""Functional, fully-jittable mission environment.

    task  = sample_task(key, level, cfg)                 # or sample_start + sample_tour
    state = reset(level, task)
    state, reward, info = step(level, task, state, action)
    obs   = observe(level, task, state)

A mission is a tour of up to MAX_WAYPOINTS waypoints (a single target is a tour
of one), flown with a per-mission ship loadout: thrust and fuel are sampled
against the tour's direct-flight cost, and are often too small to fly it
directly — which is what makes gravity worth using rather than fighting.

Actions are (turn in [-1, 1], throttle in [0, 1]) held for one control tick
(SUBSTEPS physics steps).  Everything is batched with jax.vmap by the caller.
"""

from typing import NamedTuple

import jax
import jax.numpy as jnp

from spacenav import constants as C
from spacenav import physics as P
from spacenav.types import (ARRIVED, CRASHED, DESTROYED, LOST, RUNNING, TIMEOUT, Costs, EnvState,
                            Level, Ship, Task)

K = C.MAX_WAYPOINTS

# The "advanced flight computer": the quantities this project uses to reason
# about a level, handed to the pilot as readouts.  A player earns them lesson by
# lesson (see `spacenav.lessons.INSTRUMENTS`); an agent simply gets them as extra
# observation channels.  That makes the whole framing falsifiable -- if these
# really are the right abstractions, an agent that can see them should learn
# faster than one that cannot.  Set False to zero the channels for that ablation,
# which keeps the observation shape identical so the two runs stay comparable.
INSTRUMENTS = True


class TaskConfig(NamedTuple):
    """Distribution over missions (the curriculum knobs)."""
    waypoints: tuple = (1, 1)     # inclusive range of tour length
    p_order_free: float = 0.0     # chance the tour may be flown in any order
    p_station: float = 0.35       # chance a waypoint is a station (rides a body's orbit)
    p_rendezvous: float = 0.0     # chance a station waypoint needs a velocity match
    min_dist: float = 120.0       # leg length bounds for free-space waypoints
    max_dist: float = 750.0
    min_leg: float = 100.0        # waypoints must be at least this far apart
    p_single_objective: float = 0.5
    objective_probs: tuple = (0.4, 0.2, 0.3, 0.1)   # time, path, fuel, exposure
    start_margin: float = 20.0    # clearance from surfaces/atmospheres at waypoints
    max_start_damage: float = 0.3 # damage/s allowed at the start and at waypoints
    deltav_budget: tuple = C.DELTAV_BUDGET_RANGE    # fuel / direct-flight estimate
    accel_range: tuple = C.SHIP_ACCEL_RANGE
    p_contested: float = 0.0      # chance a waypoint sits on an equal-pull surface


class RewardConfig(NamedTuple):
    success: float = 5.0          # completing the whole tour
    waypoint: float = 2.0         # each waypoint reached
    failure: float = -5.0
    damage: float = 1.0           # weight on (damage / max health)
    shaping: float = 1.0          # potential-based, on remaining tour distance
    gamma: float = 0.999


# ---------------------------------------------------------------------------
# helpers
# ---------------------------------------------------------------------------

def snapshot_state(level: Level, snapshot):
    return level.snap_pos[snapshot], level.snap_vel[snapshot]


def waypoint_states(task: Task, body_pos, body_vel):
    """Position and velocity of every waypoint right now (K, 2) each.

    An anchored waypoint rides its body at a fixed offset, so it can hang above a
    planet or moon instead of sitting inside it."""
    anchored = (task.wp_anchor >= 0)[:, None]
    i = jnp.maximum(task.wp_anchor, 0)
    return (jnp.where(anchored, body_pos[i] + task.wp_offset, task.wp_pos),
            jnp.where(anchored, body_vel[i], jnp.zeros_like(task.wp_pos)))


def current_leg(task: Task, visited, ship_pos, wp_pos):
    """Which waypoint the ship is flying to: the next in sequence, or the nearest
    outstanding one when the tour may be flown in any order."""
    pending = task.wp_active & ~visited
    order = jnp.where(pending, jnp.arange(K), K + 10)
    nearest = jnp.where(pending, jnp.sum((wp_pos - ship_pos[None]) ** 2, -1), jnp.inf)
    return jnp.where(task.order_free, jnp.argmin(nearest), jnp.argmin(order))


def target_state(task: Task, state: EnvState):
    """Position and velocity of the waypoint currently being flown to."""
    wpos, wvel = waypoint_states(task, state.body_pos, state.body_vel)
    i = current_leg(task, state.visited, state.ship.pos, wpos)
    return wpos[i], wvel[i]


def deltav_estimate(dists, v_tols, accel):
    """Rough delta-v to fly a chain of legs: accelerate and coast, braking only
    where a velocity match is required."""
    brake = v_tols < C.FLYTHROUGH_V_TOL / 2
    per_leg = jnp.where(brake, 2.0 * jnp.sqrt(accel * dists), jnp.sqrt(2.0 * accel * dists))
    return jnp.sum(jnp.where(dists > 0, per_leg, 0.0))


def _heading(angle):
    return jnp.stack([jnp.cos(angle), jnp.sin(angle)])


def _clearance(level: Level, body_pos, x):
    """Distance from x to the nearest solid surface+atmosphere."""
    d = jnp.sqrt(jnp.sum((body_pos - x[None]) ** 2, axis=-1))
    solid = level.active & (level.radius > 0)
    return jnp.min(jnp.where(solid, d - level.radius - level.atmo_h, jnp.inf))


def _valid_point(level, body_pos, body_vel, x, cfg: TaskConfig):
    dmg, _, _ = P.hazards(level, body_pos, body_vel, x, jnp.zeros(2),
                          jnp.zeros_like(level.zone_active))
    inside = jnp.sqrt(jnp.sum(x * x)) < 0.9 * level.level_radius
    return (_clearance(level, body_pos, x) > cfg.start_margin) & (dmg < cfg.max_start_damage) & inside


def _descendants(level: Level, j, depth=6):
    """Mask of bodies whose ancestry includes body j."""
    desc = jnp.zeros_like(level.active)
    cur = level.parent
    for _ in range(depth):
        desc = desc | (cur == j)
        cur = jnp.where(cur >= 0, level.parent[jnp.maximum(cur, 0)], -1)
    return desc


def parking_velocity(level: Level, body_pos, body_vel, x, sign):
    """Velocity of a circular orbit at x around the locally dominant subsystem.

    The dominant body j is the one whose Hill sphere x sits deepest inside
    (largest m / r^3, with the factor 3 from the Hill criterion for bodies that
    orbit something).  The orbit is around the barycentre of j plus everything
    orbiting j inside x's radius, so circumbinary starts orbit the binary.
    """
    d = x[None] - body_pos
    r = jnp.sqrt(jnp.sum(d * d, axis=-1)) + 1e-6
    m = jnp.where(level.active, level.mass, 0.0)
    hill_k = jnp.where(level.parent >= 0, 1.0 / 3.0, 1.0)
    j = jnp.argmax(hill_k * m / r**3)
    inner = _descendants(level, j) & (jnp.sqrt(jnp.sum((body_pos - body_pos[j]) ** 2, -1)) < r[j])
    member = (jnp.arange(m.shape[0]) == j) | inner
    ms = jnp.where(member, m, 0.0)
    M = jnp.sum(ms) + 1e-9
    c_pos = jnp.sum(ms[:, None] * body_pos, 0) / M
    c_vel = jnp.sum(ms[:, None] * body_vel, 0) / M
    rel = x - c_pos
    rr = jnp.sqrt(jnp.sum(rel * rel)) + 1e-6
    rhat = rel / rr
    tangent = sign * jnp.stack([-rhat[1], rhat[0]])
    return c_vel + jnp.sqrt(M / rr) * tangent


def _sample_point(key, level, body_pos, body_vel, cfg, n=96, center=None, dmin=0.0, dmax=jnp.inf):
    """First valid point among n candidates (uniform over the play disc)."""
    k1, k2 = jax.random.split(key)
    R = level.level_radius
    r = R * 0.9 * jnp.sqrt(jax.random.uniform(k1, (n,), minval=0.003, maxval=1.0))
    th = jax.random.uniform(k2, (n,), maxval=2 * jnp.pi)
    pts = jnp.stack([r * jnp.cos(th), r * jnp.sin(th)], axis=-1)
    ok = jax.vmap(lambda x: _valid_point(level, body_pos, body_vel, x, cfg))(pts)
    if center is not None:
        dist = jnp.sqrt(jnp.sum((pts - center[None]) ** 2, axis=-1))
        ok = ok & (dist >= dmin) & (dist <= dmax)
    i = jnp.argmax(ok)
    return pts[i], ok[i]


# ---------------------------------------------------------------------------
# task sampling
# ---------------------------------------------------------------------------

class Start(NamedTuple):
    snapshot: jnp.ndarray
    pos: jnp.ndarray
    vel: jnp.ndarray
    angle: jnp.ndarray
    ok: jnp.ndarray


def sample_start(key, level: Level, cfg: TaskConfig = TaskConfig()) -> Start:
    k0, k1, k2, k3 = jax.random.split(key, 4)
    snap = jax.random.randint(k0, (), 0, level.snap_pos.shape[0])
    bp, bv = snapshot_state(level, snap)
    x, ok = _sample_point(k1, level, bp, bv, cfg)
    sign = jnp.where(jax.random.uniform(k2) < 0.85, 1.0, -1.0)
    v = parking_velocity(level, bp, bv, x, sign)
    angle = jax.random.uniform(k3, maxval=2 * jnp.pi)
    return Start(snap, x, v, angle, ok)


def contested_zone(level: Level, bp):
    """Where no single body dominates the field: the equal-pull surface.

    A human flies by patched conics — only the body you are orbiting matters —
    and that model is exact wherever one attractor dwarfs the rest.  It is
    *maximally* wrong on the surface where a secondary's pull equals its host's,
    at `d = a*sqrt(m/M)`, which is the real boundary of the two-body story and
    sits at roughly half a Hill radius.

    Measured in this sim, the discarded-force fraction eta = |a_full - a_dom|/|a_full|
    peaks near 0.7 on that surface and falls off as 1/d^2: 0.24 at twice the
    radius, 0.02 at eight times.  So this is the one knob that reliably breaks a
    coarse model, and it is closed-form — no search, no simulation, no new pools.

    Note this is *not* the Chirikov/Wisdom resonance-overlap scale.  Resonance
    overlap governs orbital evolution over many periods; an episode here lasts a
    fraction of one orbit, so asymptotic chaos never gets the time to bite and
    the Wisdom band (a few Hill radii out) is an eta ~ 0.02 desert.  Force
    competition is instantaneous, which is why it is the scale that matters.

    Returns (usable, r_eq, hill, radial unit vector) per body.
    """
    host = jnp.maximum(level.parent, 0)
    has = (level.parent >= 0) & level.active & (level.mass > 0) & (level.kind != C.KIND_TRACER)
    mu = jnp.clip(level.mass / jnp.maximum(level.mass[host], 1e-9), 1e-12, 1.0)
    rvec = bp - bp[host]
    a_p = jnp.sqrt(jnp.sum(rvec * rvec, -1) + 1e-9)
    r_eq = a_p * jnp.sqrt(mu)                       # equal-pull radius
    hill = a_p * (mu / 3) ** (1 / 3)
    # softened gravity inside a body makes the surface meaningless if it falls there
    clear = level.radius + level.atmo_h
    return has & (2.0 * r_eq > 1.3 * clear) & (r_eq > 0), r_eq, hill, rvec / a_p[:, None]


def chaotic_zone(level: Level, bp):
    """Deprecated alias kept for callers that only want Hill radii."""
    has, _, hill, u = contested_zone(level, bp)
    return has, 1.3 * hill, hill, u


def _contested_waypoint(key, level, bp, cfg):
    """A waypoint parked on a secondary's equal-pull surface, where the dominant-body
    model is maximally wrong and the approach is irreducibly three-body."""
    k0, k1, k2 = jax.random.split(key, 3)
    good, r_eq, _, _ = contested_zone(level, bp)
    idx = jax.random.categorical(k0, jnp.where(good, 0.0, -jnp.inf))
    # straddle the surface tightly: eta peaks near 0.7 at 1.0 r_eq and is already
    # down to 0.24 at 2 r_eq, so a wide shell dilutes the one thing we want
    lo = jnp.maximum(0.8 * r_eq[idx], 1.3 * (level.radius[idx] + level.atmo_h[idx]))
    hi = jnp.maximum(1.3 * r_eq[idx], lo * 1.05)
    mag = jax.random.uniform(k1, minval=lo, maxval=hi)
    theta = jax.random.uniform(k2, maxval=2 * jnp.pi)
    offset = mag * jnp.stack([jnp.cos(theta), jnp.sin(theta)])
    return idx, offset, jnp.any(good)


def _sample_waypoint(key, level, bp, bv, cfg, prev_pos, used):
    """One waypoint: riding a station, hanging off a planet or moon, or fixed in space.

    A moving waypoint is either a station (which already orbits something) or a
    solid body with an offset that clears its surface and atmosphere - flying to a
    body's centre would just be a crash."""
    k0, k1, k2, k3, k4, k5 = jax.random.split(key, 6)
    far = jnp.sqrt(jnp.sum((bp - prev_pos[None]) ** 2, axis=-1)) > cfg.min_leg
    stations = level.active & (level.kind == C.KIND_STATION) & ~used & far
    solid = level.active & (level.radius > 0) & (level.kind != C.KIND_TRACER) \
        & (level.kind != C.KIND_BLACK_HOLE) & ~used & far

    st_idx = jax.random.categorical(k0, jnp.where(stations, 0.0, -jnp.inf))
    bo_idx = jax.random.categorical(k1, jnp.where(solid, 0.0, -jnp.inf))
    want_moving = jax.random.uniform(k2) < cfg.p_station
    # prefer a station when one is free, otherwise hang the waypoint off a body
    take_station = want_moving & jnp.any(stations) & (jax.random.uniform(k3) < 0.5)
    take_body = want_moving & jnp.any(solid) & ~take_station
    moving = take_station | take_body
    idx = jnp.where(take_station, st_idx, bo_idx)

    theta = jax.random.uniform(k4, maxval=2 * jnp.pi)
    clear = level.radius[idx] + level.atmo_h[idx] + cfg.start_margin + 15.0
    offset = jnp.where(take_body, clear, 0.0) * jnp.stack([jnp.cos(theta), jnp.sin(theta)])

    # a contested waypoint overrides the choice above: it rides a secondary, but
    # parked on its equal-pull surface rather than just clear of its atmosphere
    ch_idx, ch_off, ch_ok = _contested_waypoint(jax.random.fold_in(k4, 7), level, bp, cfg)
    chaotic = (jax.random.uniform(jax.random.fold_in(k2, 3)) < cfg.p_contested) & ch_ok \
        & (jnp.sqrt(jnp.sum((bp[ch_idx] + ch_off - prev_pos) ** 2)) > cfg.min_leg)
    idx = jnp.where(chaotic, ch_idx, idx)
    offset = jnp.where(chaotic, ch_off, offset)
    take_station = take_station & ~chaotic
    take_body = take_body | chaotic

    free, free_ok = _sample_point(k5, level, bp, bv, cfg, center=prev_pos,
                                  dmin=jnp.maximum(cfg.min_dist, cfg.min_leg), dmax=cfg.max_dist)
    rendez = moving & (jax.random.uniform(jax.random.fold_in(k3, 1)) < cfg.p_rendezvous)
    anchor = jnp.where(moving, idx, -1)
    pos = jnp.where(moving, bp[idx] + offset, free)
    offset = jnp.where(moving, offset, jnp.zeros(2))
    v_tol = jnp.where(rendez, C.RENDEZVOUS_V_TOL, C.FLYTHROUGH_V_TOL)
    ok = moving | free_ok
    return anchor, pos, offset, v_tol, ok, used | ((jnp.arange(bp.shape[0]) == idx) & moving)


def sample_tour(key, level: Level, start: Start, cfg: TaskConfig = TaskConfig()):
    """Waypoints, objective weights and the ship loadout for one mission."""
    keys = jax.random.split(key, K + 5)
    bp, bv = snapshot_state(level, start.snapshot)

    # the upper bound may be a traced float from the training curriculum
    wp_hi = jnp.clip(jnp.round(jnp.asarray(cfg.waypoints[1], jnp.float32)), 1, K).astype(jnp.int32)
    n_wp = jax.random.randint(keys[0], (), jnp.int32(cfg.waypoints[0]), wp_hi + 1)
    order_free = jax.random.uniform(keys[1]) < cfg.p_order_free

    def add(carry, i):
        prev, used, ok = carry
        anchor, pos, offset, v_tol, good, used = _sample_waypoint(
            keys[2 + i], level, bp, bv, cfg, prev, used)
        active = i < n_wp
        # inactive slots keep the previous position so the chain distances stay clean
        prev = jnp.where(active, pos, prev)
        return (prev, used, ok & (good | ~active)), (anchor, pos, offset, v_tol, active)

    (_, _, ok), (anchors, positions, offsets, v_tols, actives) = jax.lax.scan(
        add, (start.pos, jnp.zeros_like(level.active), start.ok), jnp.arange(K))
    anchors = jnp.where(actives, anchors, -1)
    v_tols = jnp.where(actives, v_tols, C.FLYTHROUGH_V_TOL)

    # --- objective ----------------------------------------------------------
    zones = level.zone_active
    has_zone = jnp.any(zones)
    zone_pick = jax.random.categorical(keys[K + 2], jnp.where(
        zones, jnp.where(level.zone_kind == C.ZONE_SENSOR, 1.0, 0.0), -jnp.inf))
    exposure_mask = zones & (jnp.arange(zones.shape[0]) == zone_pick)
    probs = jnp.asarray(cfg.objective_probs)
    probs = probs.at[3].set(jnp.where(has_zone, probs[3], 0.0))
    one = jax.nn.one_hot(jax.random.categorical(keys[K + 3], jnp.log(probs + 1e-12)), C.N_OBJECTIVES)
    mix = jax.random.dirichlet(keys[K + 4], jnp.ones(C.N_OBJECTIVES)) * (probs > 0)
    mix = mix / jnp.sum(mix)
    single = jax.random.uniform(jax.random.fold_in(keys[K + 4], 1)) < cfg.p_single_objective
    weights = jnp.where(single, one, mix)

    # --- legs, loadout and cost references ----------------------------------
    chain = jnp.concatenate([start.pos[None], positions], 0)          # (K+1, 2)
    legs = jnp.sqrt(jnp.sum((chain[1:] - chain[:-1]) ** 2, -1))
    legs = jnp.where(actives, legs, 0.0)
    total = jnp.sum(legs)

    accel = jnp.exp(jax.random.uniform(keys[K], minval=jnp.log(cfg.accel_range[0]),
                                       maxval=jnp.log(cfg.accel_range[1])))
    dv_needed = deltav_estimate(legs, v_tols, accel)
    budget = jax.random.uniform(keys[K + 1], minval=cfg.deltav_budget[0], maxval=cfg.deltav_budget[1])
    fuel = jnp.clip(budget * dv_needed / accel, *C.SHIP_FUEL_LIMITS)

    t_ref = jnp.sum(jnp.where(actives, 2.0 * jnp.sqrt(legs / accel), 0.0))
    ref = jnp.stack([t_ref, jnp.maximum(total, 1.0), jnp.maximum(0.5 * fuel, 1.0), t_ref])

    task = Task(snapshot=start.snapshot, start_pos=start.pos, start_vel=start.vel,
                start_angle=start.angle, wp_anchor=anchors, wp_pos=positions,
                wp_offset=jnp.where(actives[:, None], offsets, 0.0),
                wp_radius=jnp.full(K, C.TARGET_RADIUS), wp_v_tol=v_tols, wp_active=actives,
                order_free=order_free, accel=accel, fuel=fuel, weights=weights,
                exposure_mask=exposure_mask, ref_scale=ref)
    return task, ok & (total > 0)


def sample_task(key, level: Level, cfg: TaskConfig = TaskConfig()):
    k1, k2 = jax.random.split(key)
    return sample_tour(k2, level, sample_start(k1, level, cfg), cfg)


# ---------------------------------------------------------------------------
# dynamics
# ---------------------------------------------------------------------------

def reset(level: Level, task: Task) -> EnvState:
    bp, bv = snapshot_state(level, task.snapshot)
    z = jnp.float32(0)
    visited = jnp.zeros(K, bool)
    wpos, _ = waypoint_states(task, bp, bv)
    return EnvState(t=z, tick=jnp.int32(0), body_pos=bp, body_vel=bv,
                    ship=Ship(task.start_pos, task.start_vel, task.start_angle,
                              task.fuel, jnp.float32(C.SHIP_HEALTH)),
                    costs=Costs(z, z, z, z, z), status=jnp.int32(RUNNING), thrust=z,
                    visited=visited, leg=current_leg(task, visited, task.start_pos, wpos))


def _substep(level: Level, task: Task, turn, throttle, carry, topk: int = 0):
    bp, bv, ship, costs, status, visited = carry
    h = C.PHYS_DT
    running = status == RUNNING
    angle = ship.angle + turn * C.SHIP_TURN_RATE * h
    burn = jnp.minimum(throttle * h, ship.fuel)
    acc = (burn / h) * task.accel * _heading(angle)
    bp1, bv1, x1, v1 = P.substep(level, bp, bv, ship.pos, ship.vel, acc, h, topk)
    dmg, expo, crashed = P.hazards(level, bp1, bv1, x1, v1, task.exposure_mask)
    health = ship.health - dmg * h

    wpos, wvel = waypoint_states(task, bp1, bv1)
    leg = current_leg(task, visited, x1, wpos)
    reach = (jnp.sum((x1[None] - wpos) ** 2, -1) < task.wp_radius ** 2) & \
            (jnp.sum((v1[None] - wvel) ** 2, -1) < task.wp_v_tol ** 2)
    allowed = task.order_free | (jnp.arange(K) == leg)
    hit = reach & allowed & task.wp_active & ~visited & running
    visited1 = visited | hit
    finished = jnp.all(visited1 | ~task.wp_active)

    lost = jnp.sum(x1 * x1) > (level.level_radius * C.OUT_OF_BOUNDS_FACTOR) ** 2
    new_status = jnp.select([crashed, health <= 0, finished, lost],
                            [CRASHED, DESTROYED, ARRIVED, LOST], RUNNING)
    new_ship = Ship(x1, v1, angle, ship.fuel - burn, jnp.maximum(health, 0.0))
    new_costs = Costs(costs.time + h, costs.path + jnp.sqrt(jnp.sum((x1 - ship.pos) ** 2)),
                      costs.fuel + burn, costs.exposure + expo * h, costs.damage + dmg * h)
    keep = lambda new, old: jax.tree.map(lambda a, b: jnp.where(running, a, b), new, old)
    return (bp1, bv1, keep(new_ship, ship), keep(new_costs, costs),
            jnp.where(running, new_status, status), jnp.where(running, visited1, visited))


def step(level: Level, task: Task, state: EnvState, action, rcfg: RewardConfig = RewardConfig(),
         topk: int = 0):
    """One control tick.  `topk` restricts the gravity the *ship* feels to the k
    dominant attractors (see physics.gravity_at): the environment under a coarse
    model of itself, used to ask whether a mesoscale plan would have worked."""
    turn = jnp.clip(action[0], -1.0, 1.0)
    throttle = jnp.clip(action[1], 0.0, 1.0)
    carry = (state.body_pos, state.body_vel, state.ship, state.costs, state.status, state.visited)
    carry = jax.lax.fori_loop(
        0, C.SUBSTEPS, lambda _, c: _substep(level, task, turn, throttle, c, topk), carry)
    bp, bv, ship, costs, status, visited = carry
    tick = state.tick + 1
    status = jnp.where((status == RUNNING) & (tick >= C.MAX_EPISODE_TICKS), TIMEOUT, status)
    wpos, _ = waypoint_states(task, bp, bv)
    new = EnvState(t=state.t + C.CTRL_DT, tick=tick, body_pos=bp, body_vel=bv, ship=ship,
                   costs=costs, status=status,
                   thrust=jnp.where(state.status == RUNNING, throttle, 0.0),
                   visited=visited, leg=current_leg(task, visited, ship.pos, wpos))
    reward = compute_reward(level, task, state, new, rcfg)
    was_running = state.status == RUNNING
    info = dict(done=status != RUNNING, just_done=was_running & (status != RUNNING),
                status=status, costs=costs, visited=jnp.sum(visited & task.wp_active))
    return new, reward, info


def objective_cost(task: Task, costs: Costs):
    """Scalar, weight-normalised cost of a mission so far (lower is better)."""
    return jnp.sum(task.weights * costs.vector() / task.ref_scale)


def _potential(task: Task, state: EnvState):
    """Negative remaining tour length: distance to the current waypoint plus a
    flat allowance for each one still outstanding, so reaching a waypoint does
    not make the potential jump."""
    wpos, _ = waypoint_states(task, state.body_pos, state.body_vel)
    i = current_leg(task, state.visited, state.ship.pos, wpos)
    d = jnp.sqrt(jnp.sum((state.ship.pos - wpos[i]) ** 2))
    pending = jnp.sum(task.wp_active & ~state.visited)
    n_total = jnp.maximum(jnp.sum(task.wp_active), 1)
    rest = jnp.maximum(pending - 1, 0) * task.ref_scale[1] / n_total
    return -(d + rest) / (task.ref_scale[1] + 1.0)


def compute_reward(level, task, prev: EnvState, new: EnvState, rcfg: RewardConfig):
    running = prev.status == RUNNING
    d_cost = objective_cost(task, new.costs) - objective_cost(task, prev.costs)
    d_dmg = (new.costs.damage - prev.costs.damage) / C.SHIP_HEALTH
    shaping = rcfg.shaping * (rcfg.gamma * _potential(task, new) - _potential(task, prev))
    reached = jnp.sum((new.visited & ~prev.visited) & task.wp_active)
    terminal = jnp.where(new.status == ARRIVED, rcfg.success,
                         jnp.where(new.status != RUNNING, rcfg.failure, 0.0))
    r = -d_cost - rcfg.damage * d_dmg + shaping + terminal + rcfg.waypoint * reached
    return jnp.where(running, r, 0.0)


# ---------------------------------------------------------------------------
# observation
# ---------------------------------------------------------------------------

def _symlog(x):
    return jnp.sign(x) * jnp.log1p(jnp.abs(x))


def instrument_panel(level: Level, task: Task, state: EnvState):
    """Four readouts, in the units the lessons are named after.

      eta       share of local force that is *not* the dominant body.  Near zero
                a one-body picture is right; near one it is lying to you.
      budget    the tank in units of the impulse gravity will still deliver
                (log-compressed).  Below 1 you cannot power through the field.
      demand    eta / budget: how much of the tank a wrong model would cost.
      stretch   local divergence rate, sqrt of the tidal gradient.  How fast the
                flow pulls nearby trajectories apart, so how soon prediction
                stops being worth anything.

    Zeroed when `INSTRUMENTS` is False, so an ablation keeps the same shape.
    """
    ship, bp = state.ship, state.body_pos
    g_full = P.gravity_at(ship.pos[None], bp, level, 0)[0]
    g_dom = P.gravity_at(ship.pos[None], bp, level, 1)[0]
    gmag = jnp.sqrt(jnp.sum(g_full ** 2) + 1e-12)
    eta = jnp.sqrt(jnp.sum((g_full - g_dom) ** 2) + 1e-18) / gmag

    t_left = jnp.maximum(C.MAX_EPISODE_TIME - state.t, C.CTRL_DT)
    budget = (ship.fuel * task.accel) / jnp.maximum(gmag * t_left, 1e-6)
    demand = eta / jnp.maximum(budget, 1e-6)

    # tidal gradient: sum of 2*GM/r^3 over the bodies that matter, whose square
    # root is a local stretching rate (1/s)
    d = bp - ship.pos[None]
    r = jnp.maximum(jnp.sqrt(jnp.sum(d * d, -1)), jnp.maximum(level.radius, 1.0))
    m = jnp.where(level.active, level.mass, 0.0)
    stretch = jnp.sqrt(jnp.sum(2.0 * m / r ** 3))

    panel = jnp.array([eta, _symlog(budget), _symlog(demand), stretch * 10.0])
    return panel if INSTRUMENTS else jnp.zeros_like(panel)


def observe(level: Level, task: Task, state: EnvState):
    """Ego-centric observation: everything relative to the ship and rotated so
    the ship's nose points along +x.  Returns a dict of fixed-shape arrays:

      self   (F_s,)      ship, loadout, tour and objective features
      bodies (N, F_b)    one token per body, `body_mask` (N,)
      zones  (Z, F_z)    one token per zone, `zone_mask` (Z,)
    """
    ship = state.ship
    c, s = jnp.cos(-ship.angle), jnp.sin(-ship.angle)
    R = jnp.array([[c, -s], [s, c]])
    rot = lambda v: v @ R.T
    L = 100.0   # length scale
    V = 20.0    # speed scale

    bp, bv = state.body_pos, state.body_vel
    wpos, wvel = waypoint_states(task, bp, bv)
    leg = state.leg
    tpos, tvel = wpos[leg], wvel[leg]
    g = P.gravity_at(ship.pos[None], bp, level)[0]
    t_rel = rot(tpos - ship.pos)
    t_dist = jnp.sqrt(jnp.sum(t_rel**2))
    dmg, expo, _ = P.hazards(level, bp, bv, ship.pos, ship.vel, task.exposure_mask)

    # what the remaining tour still demands, against what is left in the tank
    pending = task.wp_active & ~state.visited
    chain = jnp.concatenate([ship.pos[None], wpos], 0)
    legs = jnp.where(pending, jnp.sqrt(jnp.sum((chain[1:] - chain[:-1]) ** 2, -1)), 0.0)
    dv_needed = deltav_estimate(legs, task.wp_v_tol, task.accel)
    dv_left = ship.fuel * task.accel

    self_feat = jnp.concatenate([
        rot(ship.vel) / V,
        rot(g) / C.SHIP_ACCEL,
        _symlog(t_rel / L),
        t_rel / (t_dist + 1e-6),
        jnp.array([_symlog(t_dist / L)]),
        rot(tvel - ship.vel) / V,
        jnp.array([
            ship.fuel / jnp.maximum(task.fuel, 1e-3),
            dv_left / 100.0,
            dv_left / (dv_needed + 1e-3),            # can the tank still do the job?
            _symlog(dv_needed / 100.0),
            task.accel / C.SHIP_ACCEL,
            ship.health / C.SHIP_HEALTH,
            1.0 - state.t / C.MAX_EPISODE_TIME,
            jnp.sum(pending) / K,
            task.order_free.astype(jnp.float32),
            _symlog(dmg),
            expo,
            jnp.sqrt(jnp.sum(ship.pos**2)) / level.level_radius,
        ]),
        task.weights,
        instrument_panel(level, task, state),
    ])

    # one block per waypoint: where it is, how fast it moves, whether it is done
    wp_rel = rot(wpos - ship.pos[None])
    wp_dist = jnp.sqrt(jnp.sum(wp_rel**2, -1)) + 1e-6
    wp_feat = jnp.concatenate([
        _symlog(wp_rel / L), wp_rel / wp_dist[:, None], _symlog(wp_dist / L)[:, None],
        rot(wvel - ship.vel[None]) / V, task.wp_radius[:, None] / L,
        jnp.minimum(task.wp_v_tol, 50.0)[:, None] / V,
        state.visited[:, None].astype(jnp.float32),
        (jnp.arange(K) == leg)[:, None].astype(jnp.float32),
        task.wp_active[:, None].astype(jnp.float32),
        (task.wp_anchor >= 0)[:, None].astype(jnp.float32),
    ], axis=-1).reshape(-1)
    self_feat = jnp.concatenate([self_feat, wp_feat])

    rel = rot(bp - ship.pos[None])
    dist = jnp.sqrt(jnp.sum(rel**2, axis=-1)) + 1e-6
    vrel = rot(bv - ship.vel[None])
    acc = level.mass / jnp.maximum(dist, level.radius) ** 2
    kind_oh = jax.nn.one_hot(level.kind, 10)
    is_wp = jnp.any((jnp.arange(level.kind.shape[0])[:, None] == task.wp_anchor[None, :])
                    & task.wp_active[None, :], axis=-1)
    body_feat = jnp.concatenate([
        _symlog(rel / L), rel / dist[:, None], _symlog(dist / L)[:, None],
        vrel / V, _symlog(acc / task.accel)[:, None],
        jnp.log1p(level.mass)[:, None] / 10.0, level.radius[:, None] / L,
        level.atmo_h[:, None] / L, jnp.log1p(level.lum)[:, None] / 10.0,
        ((dist - level.radius - level.atmo_h) / L)[:, None],
        is_wp[:, None].astype(jnp.float32),
        kind_oh,
    ], axis=-1)

    anchor = jnp.maximum(level.zone_anchor, 0)
    zc = jnp.where((level.zone_anchor >= 0)[:, None], bp[anchor], 0.0)
    zrel = rot(zc - ship.pos[None])
    zd = jnp.sqrt(jnp.sum(zrel**2, axis=-1)) + 1e-6
    # signed distance to the band: negative inside
    band = jnp.maximum(level.zone_r_in - zd, zd - level.zone_r_out)
    zone_feat = jnp.concatenate([
        _symlog(zrel / L), zrel / zd[:, None], _symlog(zd / L)[:, None],
        level.zone_r_in[:, None] / L, level.zone_r_out[:, None] / L, _symlog(band / L)[:, None],
        jax.nn.one_hot(level.zone_kind, 4), level.zone_strength[:, None], level.zone_spin[:, None],
        task.exposure_mask[:, None].astype(jnp.float32),
    ], axis=-1)
    return dict(self=self_feat, bodies=body_feat, body_mask=level.active,
                zones=zone_feat, zone_mask=level.zone_active)


# ---------------------------------------------------------------------------
# rollouts
# ---------------------------------------------------------------------------

def rollout_actions(level: Level, task: Task, actions, n_ticks=None,
                    rcfg: "RewardConfig" = None):
    """Replay a fixed action sequence.  Skips the observation (which a replay never
    uses), so it compiles and runs much faster than rollout() with a constant policy."""
    rcfg = rcfg or RewardConfig()
    n_ticks = n_ticks or actions.shape[0]

    def f(state, a):
        new, r, info = step(level, task, state, a, rcfg)
        return new, dict(pos=new.ship.pos, vel=new.ship.vel, angle=new.ship.angle,
                         thrust=new.thrust, turn=a[0], health=new.ship.health,
                         fuel=new.ship.fuel, reward=r, status=new.status, leg=new.leg)
    return jax.lax.scan(f, reset(level, task), actions[:n_ticks])


def rollout(level: Level, task: Task, policy, n_ticks=C.MAX_EPISODE_TICKS,
            rcfg: RewardConfig = RewardConfig()):
    """Run `policy(obs, state) -> action` for n_ticks; returns the final state and a
    per-tick trace (ship pos/vel/angle/thrust/health/fuel, reward, status)."""
    def f(state, _):
        action = policy(observe(level, task, state), state)
        new, r, info = step(level, task, state, action, rcfg)
        parts, _, _ = P.hazard_terms(level, new.body_pos, new.body_vel, new.ship.pos, new.ship.vel,
                                     task.exposure_mask)
        rec = dict(pos=new.ship.pos, vel=new.ship.vel, angle=new.ship.angle, thrust=new.thrust,
                   turn=jnp.clip(action[0], -1, 1), health=new.ship.health, fuel=new.ship.fuel,
                   reward=r, status=new.status, damage_parts=parts, leg=new.leg,
                   visited=jnp.sum(new.visited & task.wp_active))
        return new, rec
    s0 = reset(level, task)
    final, trace = jax.lax.scan(f, s0, None, length=n_ticks)
    return final, trace
