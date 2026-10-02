"""Ship physics module for 2D spaceship navigation.

Implements Newtonian mechanics with Tsiolkovsky rocket equation mass loss.
All functions are pure and compatible with ``jax.jit``.
"""

from typing import NamedTuple

import jax.numpy as jnp


# ---------------------------------------------------------------------------
# Data structures
# ---------------------------------------------------------------------------

class ShipState(NamedTuple):
    """Instantaneous state of the ship."""
    pos: jnp.ndarray          # (2,) position in meters
    vel: jnp.ndarray          # (2,) velocity in m/s
    heading: float             # angle in radians (0 = right, pi/2 = up)
    angular_vel: float         # angular velocity in rad/s
    fuel_mass: float           # remaining fuel in kg
    dry_mass: float            # structural mass in kg (constant)


class ShipConfig(NamedTuple):
    """Static design parameters for a ship."""
    dry_mass: float                    # kg — structural mass
    fuel_mass: float                   # kg — initial fuel load
    max_thrust: float                  # N  — peak main-engine thrust
    exhaust_velocity: float            # m/s — main-engine effective exhaust velocity
    max_torque: float                  # N*m — peak rotation thruster torque
    moment_arm: float                  # m  — distance from CoM to RCS thruster
    rotation_exhaust_velocity: float   # m/s — effective exhaust velocity for RCS
    moment_of_inertia: float           # kg*m^2 — rotational inertia about CoM
    radius: float                      # m  — bounding radius for collision detection


# ---------------------------------------------------------------------------
# Helper functions
# ---------------------------------------------------------------------------

def total_mass(state: ShipState) -> float:
    """Total ship mass (dry structure + remaining fuel)."""
    return state.dry_mass + state.fuel_mass


def heading_vector(heading: float) -> jnp.ndarray:
    """Unit direction vector for a given heading angle.

    Convention: heading=0 -> +x (right), heading=pi/2 -> +y (up).
    """
    return jnp.array([jnp.cos(heading), jnp.sin(heading)])


def fuel_fraction(state: ShipState) -> float:
    """Fraction of fuel remaining relative to total mass (range 0..1)."""
    return state.fuel_mass / (state.dry_mass + state.fuel_mass)


# ---------------------------------------------------------------------------
# Physics step
# ---------------------------------------------------------------------------

def ship_step(
    state: ShipState,
    action: jnp.ndarray,
    gravity_force: jnp.ndarray,
    dt: float,
    *,
    config: ShipConfig,
) -> ShipState:
    """Advance the ship by one time step.

    This is a pure function safe for ``jax.jit``.  To obtain the four-arg
    signature ``(state, action, gravity_force, dt)`` required by a simulation
    loop, use :func:`make_ship_step` which closes over ``config``.

    Parameters
    ----------
    state : ShipState
        Current ship state.
    action : array-like, shape (2,)
        ``(main_thrust_frac, rotation_thrust_frac)``.
        ``main_thrust_frac`` is clamped to [0, 1].
        ``rotation_thrust_frac`` uses the full [-1, 1] range
        (positive = counter-clockwise).
    gravity_force : array-like, shape (2,)
        External gravitational force in Newtons acting on the ship.
    dt : float
        Time step in seconds.
    config : ShipConfig
        Static ship design parameters (keyword-only).

    Returns
    -------
    ShipState
        The updated state after the step.
    """
    # -- Clamp action fractions --------------------------------------------
    main_frac = jnp.clip(action[0], 0.0, 1.0)
    rot_frac = jnp.clip(action[1], -1.0, 1.0)

    # -- Requested thrust & torque -----------------------------------------
    thrust_requested = main_frac * config.max_thrust    # N
    torque_requested = rot_frac * config.max_torque      # N*m

    # -- Fuel consumption (Tsiolkovsky rocket equation) --------------------
    #
    # The Tsiolkovsky rocket equation relates thrust to mass flow:
    #
    #     F = v_e * (dm/dt)   =>   dm/dt = F / v_e
    #
    # Main engine:
    #     dm_main = |thrust| / exhaust_velocity * dt
    #
    # Rotation (RCS) thrusters:
    #     Torque is produced by a thruster pair at distance `moment_arm`
    #     from the centre of mass, so the force is |torque| / moment_arm.
    #     dm_rot = |torque| / (moment_arm * rotation_exhaust_velocity) * dt
    #
    # RCS fuel use is typically much smaller than main-engine use.
    dm_main = jnp.abs(thrust_requested) / config.exhaust_velocity * dt
    dm_rot = (jnp.abs(torque_requested)
              / (config.moment_arm * config.rotation_exhaust_velocity) * dt)
    dm_total = dm_main + dm_rot

    # -- Fuel guard --------------------------------------------------------
    # If fuel is exhausted no thrust or torque can be applied.
    # We use jnp.where (not Python if) so the function stays traceable.
    has_fuel = state.fuel_mass > 0.0

    # Never consume more fuel than remains.
    dm_clamped = jnp.minimum(dm_total, state.fuel_mass)

    # Scale thrust/torque proportionally when fuel is insufficient for the
    # full request, and zero everything when the tank is empty.
    scale = jnp.where(dm_total > 0.0, dm_clamped / dm_total, 0.0)
    scale = jnp.where(has_fuel, scale, 0.0)

    thrust = thrust_requested * scale    # effective thrust  (N)
    torque = torque_requested * scale    # effective torque   (N*m)
    fuel_used = dm_clamped * scale       # actual fuel burned (kg)

    new_fuel = jnp.maximum(state.fuel_mass - fuel_used, 0.0)

    # -- Rotational dynamics -----------------------------------------------
    alpha = torque / config.moment_of_inertia              # rad/s^2
    new_angular_vel = state.angular_vel + alpha * dt       # rad/s
    new_heading = state.heading + new_angular_vel * dt     # rad

    # -- Translational dynamics --------------------------------------------
    mass = state.dry_mass + state.fuel_mass   # mass at start of step
    direction = heading_vector(state.heading)
    thrust_force = thrust * direction          # (2,) N

    accel = (thrust_force + gravity_force) / mass   # (2,) m/s^2
    new_vel = state.vel + accel * dt                # (2,) m/s
    new_pos = state.pos + new_vel * dt              # (2,) m

    return ShipState(
        pos=new_pos,
        vel=new_vel,
        heading=new_heading,
        angular_vel=new_angular_vel,
        fuel_mass=new_fuel,
        dry_mass=state.dry_mass,
    )


def make_ship_step(config: ShipConfig):
    """Return a four-argument step function with ``config`` closed over.

    The returned callable has the signature::

        step_fn(state, action, gravity_force, dt) -> ShipState

    and is pure / jittable.

    Example::

        step = make_ship_step(my_config)
        new_state = jax.jit(step)(state, action, gravity, 0.01)
    """

    def step_fn(
        state: ShipState,
        action: jnp.ndarray,
        gravity_force: jnp.ndarray,
        dt: float,
    ) -> ShipState:
        return ship_step(state, action, gravity_force, dt, config=config)

    return step_fn
