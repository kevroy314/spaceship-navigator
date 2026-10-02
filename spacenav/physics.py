"""Pure-JAX physics: n-body gravity, the ship integrator and the hazard field.

The ship is a test particle: it feels every body but moves none of them, so a
level's body trajectories are fully determined by its initial snapshot.  That
lets the web client replay a JAX-computed ephemeris exactly and only integrate
the ship itself.

Integrator: kick-drift-kick leapfrog (symplectic for the bodies; the ship's
thrust and atmospheric drag make it merely second-order accurate).
"""

import jax
import jax.numpy as jnp

from spacenav import constants as C
from spacenav.types import Level

EPS = 1e-6


def gravity_at(points, body_pos, level: Level, topk: int = 0):
    """Acceleration at each of `points` (M, 2) due to all active bodies.

    Inside a body's radius the field falls off linearly (uniform sphere), which
    keeps it finite everywhere and makes the self-interaction exactly zero.

    `topk > 0` keeps only the k strongest attractors at each point and discards
    the rest: that is the "patched conics" family of coarse models a human
    actually reasons with (k=1 is "only the body I am orbiting matters").  It
    must be a Python int, since it changes the traced graph.  Body motion always
    uses the full field — only the test particle's felt force is restricted.

    Note: differentiating this at r = 0 (a body against itself, as in bodies_step)
    gives NaN.  Body motion never depends on the controls, so gradient-based
    trajectory optimisation stops the gradient there instead (see opt/shooting.py).
    """
    d = body_pos[None, :, :] - points[:, None, :]              # (M, N, 2)
    r = jnp.sqrt(jnp.sum(d * d, axis=-1))                       # (M, N)
    soft = jnp.maximum(jnp.maximum(r, level.radius[None, :]), EPS)
    m = jnp.where(level.active, level.mass, 0.0)
    w = m[None, :] / soft**3                                    # (M, N) pull per unit offset
    if topk:
        pull = w * r                                            # |a| from each body
        cut = jnp.min(jax.lax.top_k(pull, min(topk, pull.shape[-1]))[0], axis=-1)
        w = jnp.where(pull >= cut[:, None], w, 0.0)
    return jnp.sum(w[..., None] * d, axis=1)


def bodies_step(body_pos, body_vel, level: Level, h=C.PHYS_DT):
    """Advance bodies one leapfrog step."""
    v = body_vel + 0.5 * h * gravity_at(body_pos, body_pos, level)
    p = body_pos + h * v
    v = v + 0.5 * h * gravity_at(p, p, level)
    return p, v


def _atmosphere(level: Level, body_pos, body_vel, x, v):
    """Drag acceleration on the ship and heating damage rate."""
    d = x[None, :] - body_pos                                   # (N, 2)
    r = jnp.sqrt(jnp.sum(d * d, axis=-1))
    alt = jnp.maximum(r - level.radius, 0.0)
    has = (level.atmo_h > 0) & level.active
    scale = jnp.maximum(level.atmo_h * C.ATMO_SCALE_FRAC, EPS)
    rho = jnp.where(has & (alt < level.atmo_h), level.atmo_rho * jnp.exp(-alt / scale), 0.0)
    v_rel = v[None, :] - body_vel                               # (N, 2)
    speed = jnp.sqrt(jnp.sum(v_rel * v_rel, axis=-1) + EPS)
    drag = -C.ATMO_DRAG * jnp.sum((rho * speed)[:, None] * v_rel, axis=0)
    heat = C.ATMO_HEAT * jnp.sum(rho * speed**3)
    return drag, heat


def zone_weights(level: Level, body_pos, x):
    """Soft membership of point x in every zone, plus geometry for the flow."""
    anchor_pos = jnp.where((level.zone_anchor >= 0)[:, None],
                           body_pos[jnp.maximum(level.zone_anchor, 0)], 0.0)
    d = x[None, :] - anchor_pos                                 # (Z, 2)
    r = jnp.sqrt(jnp.sum(d * d, axis=-1) + EPS)
    edge = jnp.maximum(C.ZONE_EDGE * (level.zone_r_out - level.zone_r_in), EPS)
    inner = jnp.where(level.zone_r_in > 0,
                      jnp.clip((r - level.zone_r_in) / edge, 0.0, 1.0), 1.0)
    outer = jnp.clip((level.zone_r_out - r) / edge, 0.0, 1.0)
    w = jnp.where(level.zone_active, inner * outer, 0.0)
    return w, d, r


def hazards(level: Level, body_pos, body_vel, x, v, exposure_mask):
    """Instantaneous damage rate, exposure rate and crash flag at (x, v)."""
    parts, exposure, crashed = hazard_terms(level, body_pos, body_vel, x, v, exposure_mask)
    return jnp.sum(parts), exposure, crashed


def hazard_terms(level: Level, body_pos, body_vel, x, v, exposure_mask):
    """Damage rate split by source: [radiation, atmosphere heat, debris, radiation belts]."""
    # radiation from luminous bodies
    d = x[None, :] - body_pos
    r = jnp.sqrt(jnp.sum(d * d, axis=-1))
    lum = jnp.where(level.active, level.lum, 0.0)
    rad = C.RADIATION_K * jnp.sum(lum / jnp.maximum(r, jnp.maximum(level.radius, 1.0))**2)

    solid = level.active & (level.radius > 0) & (level.kind != C.KIND_STATION) \
        & (level.kind != C.KIND_TRACER)
    crashed = jnp.any(solid & (r < level.radius))

    _, heat = _atmosphere(level, body_pos, body_vel, x, v)

    # zones
    w, dz, rz = zone_weights(level, body_pos, x)
    anchor = jnp.maximum(level.zone_anchor, 0)
    anchored = (level.zone_anchor >= 0)
    anchor_vel = jnp.where(anchored[:, None], body_vel[anchor], 0.0)
    anchor_mass = jnp.where(anchored, level.mass[anchor], 0.0)
    tangent = jnp.stack([-dz[:, 1], dz[:, 0]], axis=-1) / rz[:, None]
    # double-where: identical forward value, but no 0/0 in the gradient when a zone's
    # anchor is massless (a cloud tracer), which would otherwise poison every gradient
    massive = anchor_mass > 0
    circ = jnp.where(massive, jnp.sqrt(jnp.where(massive, anchor_mass, 1.0) / rz), 0.0)
    flow = anchor_vel + (level.zone_spin * circ)[:, None] * tangent
    rel = jnp.sqrt(jnp.sum((v[None, :] - flow) ** 2, axis=-1) + EPS)
    debris = jnp.sum(jnp.where(level.zone_kind == C.ZONE_DEBRIS,
                               C.DEBRIS_K * level.zone_strength * rel * w, 0.0))
    belts = jnp.sum(jnp.where(level.zone_kind == C.ZONE_RADIATION,
                              level.zone_strength * w, 0.0))
    exposure = jnp.minimum(jnp.sum(jnp.where(exposure_mask, w, 0.0)), 1.0)

    return jnp.stack([rad, heat, debris, belts]), exposure, crashed


def ship_accel(level, body_pos, body_vel, x, v, thrust_acc, topk: int = 0):
    g = gravity_at(x[None, :], body_pos, level, topk)[0]
    drag, _ = _atmosphere(level, body_pos, body_vel, x, v)
    return g + drag + thrust_acc


def substep(level: Level, body_pos, body_vel, x, v, thrust_acc, h=C.PHYS_DT, topk: int = 0):
    """One joint leapfrog step of bodies and ship.

    `thrust_acc` is held constant across the step.  `topk` restricts only the
    ship's felt gravity (see `gravity_at`); the bodies always move truthfully.
    """
    a0 = ship_accel(level, body_pos, body_vel, x, v, thrust_acc, topk)
    v_half = v + 0.5 * h * a0
    x1 = x + h * v_half
    bp1, bv1 = bodies_step(body_pos, body_vel, level, h)
    a1 = ship_accel(level, bp1, bv1, x1, v_half, thrust_acc, topk)
    v1 = v_half + 0.5 * h * a1
    return bp1, bv1, x1, v1


def roll_bodies(level: Level, body_pos, body_vel, n_steps, h=C.PHYS_DT):
    """Integrate bodies for n_steps; returns the trajectory (n_steps+1, N, 2) of positions/velocities."""
    def f(carry, _):
        p, v = bodies_step(*carry, level, h)
        return (p, v), (p, v)
    _, (ps, vs) = jax.lax.scan(f, (body_pos, body_vel), None, length=n_steps)
    ps = jnp.concatenate([body_pos[None], ps], axis=0)
    vs = jnp.concatenate([body_vel[None], vs], axis=0)
    return ps, vs


def energy(level: Level, body_pos, body_vel):
    """Total energy of the massive bodies (for integrator diagnostics)."""
    m = jnp.where(level.active, level.mass, 0.0)
    ke = 0.5 * jnp.sum(m * jnp.sum(body_vel**2, axis=-1))
    d = body_pos[:, None, :] - body_pos[None, :, :]
    r = jnp.sqrt(jnp.sum(d * d, axis=-1))
    soft = jnp.maximum(jnp.maximum(r, level.radius[None, :]), EPS)
    pair = m[:, None] * m[None, :] / soft
    pair = jnp.where(jnp.eye(pair.shape[0], dtype=bool), 0.0, pair)
    return ke - 0.5 * jnp.sum(pair)
