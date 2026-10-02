"""Symplectic integrators for gravitational simulation.

All functions are pure JAX and fully jittable.
"""

import jax.numpy as jnp


def leapfrog_kick(vel, acc, dt_half):
    """Update velocities using accelerations (half-step).

    v_{n+1/2} = v_n + a_n * dt/2

    Args:
        vel: Velocities, shape (..., D).
        acc: Accelerations, shape (..., D).
        dt_half: Half the timestep (dt / 2).

    Returns:
        Updated velocities, same shape as vel.
    """
    return vel + acc * dt_half


def leapfrog_drift(pos, vel, dt):
    """Update positions using velocities (full step).

    x_{n+1} = x_n + v_{n+1/2} * dt

    Args:
        pos: Positions, shape (..., D).
        vel: Velocities, shape (..., D).
        dt: Full timestep.

    Returns:
        Updated positions, same shape as pos.
    """
    return pos + vel * dt


def leapfrog_step(pos, vel, acc, force_fn, dt):
    """Full leapfrog (kick-drift-kick) integration step.

    Sequence:
        1. kick:  v_{n+1/2} = v_n     + a_n   * dt/2
        2. drift: x_{n+1}   = x_n     + v_{n+1/2} * dt
        3. recompute: a_{n+1} = force_fn(x_{n+1})
        4. kick:  v_{n+1}   = v_{n+1/2} + a_{n+1} * dt/2

    Args:
        pos: Positions, shape (..., D).
        vel: Velocities, shape (..., D).
        acc: Current accelerations, shape (..., D).
        force_fn: Callable (pos) -> acc. Must be a pure function suitable
            for use inside jit. Recomputes accelerations from new positions.
        dt: Timestep (scalar).

    Returns:
        Tuple of (new_pos, new_vel, new_acc).
    """
    dt_half = dt * 0.5

    # Kick
    vel_half = leapfrog_kick(vel, acc, dt_half)

    # Drift
    new_pos = leapfrog_drift(pos, vel_half, dt)

    # Recompute forces at new positions
    new_acc = force_fn(new_pos)

    # Kick
    new_vel = leapfrog_kick(vel_half, new_acc, dt_half)

    return new_pos, new_vel, new_acc


def symplectic_euler(pos, vel, force_fn, dt):
    """Symplectic (semi-implicit) Euler integrator — first-order fallback.

    Sequence:
        1. a_{n}   = force_fn(x_n)
        2. v_{n+1} = v_n + a_n * dt
        3. x_{n+1} = x_n + v_{n+1} * dt   (uses *updated* velocity)

    This is symplectic because the position update uses the new velocity,
    making it a valid symplectic map (preserves phase-space volume).

    Args:
        pos: Positions, shape (..., D).
        vel: Velocities, shape (..., D).
        force_fn: Callable (pos) -> acc. Pure, jittable.
        dt: Timestep (scalar).

    Returns:
        Tuple of (new_pos, new_vel, new_acc).
    """
    acc = force_fn(pos)
    new_vel = vel + acc * dt
    new_pos = pos + new_vel * dt
    return new_pos, new_vel, acc
