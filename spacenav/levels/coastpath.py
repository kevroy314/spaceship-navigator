"""Level 0: put the targets on a path the ship already flies.

Every other generator here picks targets and then asks whether a flight exists.
This one runs the ship ballistically — engine off, pure gravity — and *places*
the targets on the curve it traces.  A solution is then guaranteed by
construction, and the solution is "touch nothing".

That is not a degenerate level, it is the opening lesson, and it earns its place
the way "hold right" does in a platformer: the player does nothing and watches a
complicated thing happen, and the thing they learn is that the field is doing
real work.  Everything afterwards is a departure from it.

Each target is anchored to the body nearest the ship at the moment it passes, so
the targets *drift* — they are moving objects on their own orbits that the ship
happens to meet.  Nothing is pinned to empty space.

This is also the cheap direction of the generation problem.  Searching forwards
for interesting flights costs a planner per candidate; running one ballistic arc
and reading targets off it costs a single rollout, and the arc's own shape
decides how good the level is.
"""

from typing import NamedTuple

import jax
import jax.numpy as jnp

from spacenav import constants as C
from spacenav import env as E
from spacenav import physics as P
from spacenav.types import Level, Task

K = C.MAX_WAYPOINTS


class CoastConfig(NamedTuple):
    targets: int = 6              # how many the ship should collect on the way
    ticks: int = C.MAX_EPISODE_TICKS
    margin: float = 25.0          # clearance a target keeps from its anchor's surface
    lead_frac: float = 0.06       # skip the first few % of the arc: no target at the start
    tail_frac: float = 0.97       # and stop short of the very end
    candidates: int = 64          # ballistic arcs to try before keeping the best
    fuel: float = 12.0            # a tank it does not need, so the player may still steer


def ballistic(level: Level, start, ticks: int):
    """The ship's path with the engine off, sampled per control tick.

    Returns ship positions, body positions and a liveness mask (the arc stops
    being usable once it hits something or leaves the level).
    """
    bp0, bv0 = E.snapshot_state(level, start.snapshot)

    def tick(carry, _):
        bp, bv, x, v = carry
        for _ in range(C.SUBSTEPS):
            bp, bv, x, v = P.substep(level, bp, bv, x, v, jnp.zeros(2))
        d = jnp.sqrt(jnp.sum((bp - x[None]) ** 2, -1))
        solid = level.active & (level.radius > 0) & (level.kind != C.KIND_STATION) \
            & (level.kind != C.KIND_TRACER)
        hit = jnp.any(solid & (d < level.radius))
        gone = jnp.sum(x * x) > (level.level_radius * C.OUT_OF_BOUNDS_FACTOR) ** 2
        return (bp, bv, x, v), (x, bp, ~(hit | gone))

    _, (xs, bps, ok) = jax.lax.scan(tick, (bp0, bv0, start.pos, start.vel), None, length=ticks)
    # once the arc is spoiled it stays spoiled
    live = jnp.cumprod(ok.astype(jnp.int32)).astype(bool)
    return xs, bps, live


def _turning(xs, live):
    """Total heading change along the arc, in turns: how much the field bends it.

    A straight coast is a dull level however many targets sit on it, so this is
    the score that picks one arc over another.
    """
    d = xs[1:] - xs[:-1]
    ang = jnp.arctan2(d[:, 1], d[:, 0])
    step = jnp.abs((ang[1:] - ang[:-1] + jnp.pi) % (2 * jnp.pi) - jnp.pi)
    return jnp.sum(jnp.where(live[2:], step, 0.0)) / (2 * jnp.pi)


def _attach(level: Level, xs, bps, idx):
    """Anchor a target to the body nearest the ship at tick `idx`.

    The offset is frozen at that moment, so the target rides the body's orbit and
    the ship meets it exactly there.  Bodies the ship would be *inside* are
    rejected in favour of one it can pass cleanly.
    """
    x, bp = xs[idx], bps[idx]
    d = jnp.sqrt(jnp.sum((bp - x[None]) ** 2, -1) + 1e-9)
    clear = level.radius + level.atmo_h + 10.0
    usable = level.active & (level.kind != C.KIND_TRACER) & (d > clear)
    anchor = jnp.argmin(jnp.where(usable, d, jnp.inf))
    return anchor, x - bp[anchor], jnp.any(usable)


def coast_task(key, level: Level, cfg: CoastConfig = CoastConfig()):
    """One Level-0 mission: targets strung along a single ballistic arc."""
    start = E.sample_start(key, level, E.TaskConfig())
    xs, bps, live = ballistic(level, start, cfg.ticks)
    n_live = jnp.sum(live)

    # spread the targets over the usable part of the arc
    fracs = jnp.linspace(cfg.lead_frac, cfg.tail_frac, cfg.targets)
    idx = jnp.clip((fracs * n_live.astype(jnp.float32)).astype(jnp.int32), 1, cfg.ticks - 1)

    anchors, offsets, oks = jax.vmap(lambda i: _attach(level, xs, bps, i))(idx)
    active = jnp.arange(K) < cfg.targets
    pad = lambda a, fill: jnp.concatenate(
        [a, jnp.full((K - cfg.targets,) + a.shape[1:], fill, a.dtype)], axis=0)

    enough = (n_live > cfg.ticks // 3) & jnp.all(oks) & start.ok
    task = Task(
        snapshot=start.snapshot, start_pos=start.pos, start_vel=start.vel,
        start_angle=start.angle,
        wp_anchor=pad(anchors, 0), wp_pos=pad(xs[idx], 0.0), wp_offset=pad(offsets, 0.0),
        wp_radius=jnp.full(K, C.TARGET_RADIUS),
        wp_v_tol=jnp.full(K, C.FLYTHROUGH_V_TOL),
        wp_active=active, order_free=jnp.bool_(False),
        accel=jnp.float32(1.6), fuel=jnp.float32(cfg.fuel),
        weights=jnp.array([1.0, 0.0, 0.0, 0.0]),
        exposure_mask=jnp.zeros_like(level.zone_active),
        # the tour is done when the last target is collected; using the whole
        # episode here makes every Level-0 task look like it overruns the clock
        ref_scale=jnp.array([idx[-1].astype(jnp.float32) * C.CTRL_DT,
                             1000.0, cfg.fuel, 1.0]),
    )
    return task, enough, _turning(xs, live)


def best_coast_task(key, level: Level, cfg: CoastConfig = CoastConfig()):
    """Try many arcs on one level and keep the most convoluted valid one."""
    tasks, oks, scores = jax.vmap(lambda k: coast_task(k, level, cfg))(
        jax.random.split(key, cfg.candidates))
    pick = jnp.argmax(jnp.where(oks, scores, -jnp.inf))
    return jax.tree.map(lambda a: a[pick], tasks), oks[pick], scores[pick]
