"""Hand-written pilots: solvability probes, tuning tools and RL baselines."""

from typing import NamedTuple

import jax.numpy as jnp

from spacenav import constants as C
from spacenav import physics as P
from spacenav.env import target_state
from spacenav.types import EnvState, Level, Task


class PilotConfig(NamedTuple):
    cruise: float = 35.0        # max approach speed relative to the target (u/s)
    brake_frac: float = 0.6     # plan braking with this fraction of max thrust
    tau: float = 0.8            # velocity-error time constant (s)
    avoid_margin: float = 25.0  # start steering away this far above surface+atmosphere
    avoid_gain: float = 30.0
    aim_tol: float = 0.5        # rad: only burn when the nose is within this of the burn direction
    deadband: float = 0.4       # u/s^2: don't burn for smaller corrections


def pilot(level: Level, task: Task, cfg: PilotConfig = PilotConfig()):
    """Proportional navigation towards the target with gravity compensation."""

    def policy(obs, state: EnvState):
        ship = state.ship
        tpos, tvel = target_state(task, state)
        rel = tpos - ship.pos
        d = jnp.sqrt(jnp.sum(rel**2)) + 1e-6
        u = rel / d
        rendezvous = task.wp_v_tol[state.leg] < C.FLYTHROUGH_V_TOL / 2
        a_max = task.accel
        # a weak engine cannot hold a fast cruise, and would only waste the tank trying
        cruise = jnp.minimum(cfg.cruise, 0.7 * jnp.sqrt(2 * a_max * d))
        brake = jnp.sqrt(2 * cfg.brake_frac * a_max * jnp.maximum(d - 0.3 * task.wp_radius[state.leg], 0))
        speed = jnp.where(rendezvous, jnp.minimum(cruise, brake), cruise)
        v_des = tvel + u * speed

        # steer away from surfaces we are closing on
        dp = ship.pos[None] - state.body_pos
        r = jnp.sqrt(jnp.sum(dp**2, axis=-1)) + 1e-6
        alt = r - level.radius - level.atmo_h
        solid = level.active & (level.radius > 0) & (level.kind != C.KIND_STATION) \
            & (level.kind != C.KIND_TRACER)
        push = jnp.clip((cfg.avoid_margin - alt) / cfg.avoid_margin, 0.0, 1.0) * solid
        v_des = v_des + cfg.avoid_gain * jnp.sum(push[:, None] * dp / r[:, None], axis=0)

        g = P.gravity_at(ship.pos[None], state.body_pos, level)[0]
        a_req = (v_des - ship.vel) / cfg.tau - g
        a_mag = jnp.sqrt(jnp.sum(a_req**2)) + 1e-6
        want = jnp.arctan2(a_req[1], a_req[0])
        err = jnp.angle(jnp.exp(1j * (want - ship.angle)))
        turn = jnp.clip(err / (C.SHIP_TURN_RATE * C.CTRL_DT), -1.0, 1.0)
        aligned = jnp.abs(err) < cfg.aim_tol
        throttle = jnp.where(aligned & (a_mag > cfg.deadband * a_max / C.SHIP_ACCEL),
                             jnp.clip(a_mag / a_max, 0.0, 1.0) * jnp.cos(err), 0.0)
        return jnp.stack([turn, throttle])

    return policy


def coast(level: Level, task: Task):
    """Do nothing (for measuring how hostile a start position is)."""
    return lambda obs, state: jnp.zeros(2)
