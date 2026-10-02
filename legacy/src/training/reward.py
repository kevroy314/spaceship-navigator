"""Reward function for spaceship navigation.

Reward components:
1. Large bonus for reaching target (within approach_radius)
2. Proximity reward: scales with closeness to target
3. Velocity matching bonus: reward for low relative velocity near target
4. Time penalty: small per-step cost to encourage efficiency
5. Fuel penalty: cost proportional to fuel consumed this step
"""

import jax
import jax.numpy as jnp
from typing import NamedTuple


class RewardConfig(NamedTuple):
    """Reward function hyperparameters."""
    arrival_bonus: float = 1000.0
    proximity_scale: float = 10.0
    time_penalty: float = -0.01
    fuel_penalty: float = -0.1
    velocity_match_bonus: float = 100.0
    approach_radius: float = 1.496e9    # 0.01 AU in meters
    approach_velocity: float = 1000.0    # m/s threshold for velocity bonus
    max_distance: float = 1.496e13       # 100 AU in meters (domain size)


@jax.jit
def compute_reward(
    ship_pos: jnp.ndarray,
    ship_vel: jnp.ndarray,
    target_pos: jnp.ndarray,
    target_vel: jnp.ndarray,
    fuel_consumed: float,
    prev_distance: float,
    config: RewardConfig,
) -> tuple[float, dict]:
    """Compute reward for current timestep.

    Args:
        ship_pos: Ship position (2,)
        ship_vel: Ship velocity (2,)
        target_pos: Target body position (2,)
        target_vel: Target body velocity (2,)
        fuel_consumed: Fuel mass consumed this step (kg)
        prev_distance: Distance to target at previous step (m)
        config: Reward hyperparameters

    Returns:
        (total_reward, info_dict) where info_dict has component breakdowns
    """
    # Distance to target
    displacement = target_pos - ship_pos
    distance = jnp.linalg.norm(displacement)

    # Relative velocity
    rel_vel = ship_vel - target_vel
    rel_speed = jnp.linalg.norm(rel_vel)

    # 1. Arrival bonus — large reward for getting within approach radius
    reached_target = distance < config.approach_radius
    arrival_reward = jnp.where(reached_target, config.arrival_bonus, 0.0)

    # 2. Proximity reward — reward for getting closer
    # Normalized distance change (positive = got closer)
    distance_improvement = (prev_distance - distance) / config.max_distance
    proximity_reward = config.proximity_scale * distance_improvement

    # 3. Velocity matching bonus — only applies near target
    # Smooth activation: ramps up as ship approaches within 10x approach radius
    near_target = jnp.exp(-distance / (10.0 * config.approach_radius))
    vel_match_quality = jnp.exp(-rel_speed / config.approach_velocity)
    velocity_reward = config.velocity_match_bonus * near_target * vel_match_quality

    # 4. Time penalty — constant per step
    time_reward = config.time_penalty

    # 5. Fuel penalty — proportional to fuel consumed
    fuel_reward = config.fuel_penalty * fuel_consumed

    total = arrival_reward + proximity_reward + velocity_reward + time_reward + fuel_reward

    info = {
        'arrival_reward': arrival_reward,
        'proximity_reward': proximity_reward,
        'velocity_reward': velocity_reward,
        'time_reward': time_reward,
        'fuel_reward': fuel_reward,
        'distance': distance,
        'rel_speed': rel_speed,
        'reached_target': reached_target,
    }

    return total, info


@jax.jit
def compute_done(
    distance: float,
    step: int,
    fuel_mass: float,
    approach_radius: float,
    max_steps: int,
) -> tuple[bool, bool]:
    """Check episode termination conditions.

    Returns:
        (done, truncated) — done if target reached or fuel depleted,
        truncated if max steps reached.
    """
    reached = distance < approach_radius
    out_of_fuel = fuel_mass <= 0.0
    timed_out = step >= max_steps

    terminated = reached | out_of_fuel
    truncated = timed_out & ~terminated

    return terminated | truncated, truncated
