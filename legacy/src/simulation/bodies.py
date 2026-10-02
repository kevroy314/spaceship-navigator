"""Gravitational body state management for 2D simulation.

Uses padded fixed-size arrays with an `active` mask so the body count can
vary at runtime without triggering JAX recompilation.

All public functions are pure JAX and fully jittable.
"""

from typing import NamedTuple

import jax
import jax.numpy as jnp

from simulation.integrator import leapfrog_kick, leapfrog_drift

# Maximum number of bodies supported without recompilation.
MAX_BODIES: int = 32


class BodyState(NamedTuple):
    """State vector for all gravitational bodies.

    Arrays are padded to ``MAX_BODIES`` along the first axis.  Only rows
    where ``active`` is True represent real bodies.
    """
    pos: jnp.ndarray       # (N, 2) positions in metres
    vel: jnp.ndarray       # (N, 2) velocities in m/s
    mass: jnp.ndarray      # (N,)   masses in kg
    radius: jnp.ndarray    # (N,)   radii in metres (visualisation / collision)
    is_target: jnp.ndarray # (N,)   boolean mask — exactly one True entry
    active: jnp.ndarray    # (N,)   boolean mask — True for real bodies


# ---------------------------------------------------------------------------
# Construction helpers
# ---------------------------------------------------------------------------

def create_body_state(
    pos,
    vel,
    mass,
    radius,
    is_target,
    max_bodies: int = MAX_BODIES,
):
    """Build a ``BodyState`` from variable-length inputs, padded to *max_bodies*.

    Args:
        pos: (n, 2) array of positions.
        vel: (n, 2) array of velocities.
        mass: (n,) array of masses.
        radius: (n,) array of radii.
        is_target: (n,) boolean array (exactly one True).
        max_bodies: Pad size.  Defaults to ``MAX_BODIES``.

    Returns:
        A ``BodyState`` with arrays of shape (max_bodies, ...).
    """
    n = pos.shape[0]

    def _pad1(arr, fill=0.0):
        pad_width = max_bodies - n
        return jnp.concatenate([arr, jnp.full((pad_width,), fill)])

    def _pad2(arr, fill=0.0):
        pad_width = max_bodies - n
        return jnp.concatenate([arr, jnp.full((pad_width, 2), fill)])

    active = _pad1(jnp.ones(n, dtype=jnp.bool_), fill=0).astype(jnp.bool_)

    return BodyState(
        pos=_pad2(jnp.asarray(pos, dtype=jnp.float32)),
        vel=_pad2(jnp.asarray(vel, dtype=jnp.float32)),
        mass=_pad1(jnp.asarray(mass, dtype=jnp.float32)),
        radius=_pad1(jnp.asarray(radius, dtype=jnp.float32)),
        is_target=_pad1(jnp.asarray(is_target, dtype=jnp.bool_), fill=0).astype(jnp.bool_),
        active=active,
    )


# ---------------------------------------------------------------------------
# CIC force interpolation from a potential-field grid
# ---------------------------------------------------------------------------

def _cic_interp_force(pos_single, potential_field, grid_config):
    """Interpolate the gravitational acceleration at a single position using CIC.

    Cloud-In-Cell (CIC) bilinear interpolation of the *negative gradient* of
    the potential field.

    Args:
        pos_single: (2,) world-space position.
        potential_field: (Gx, Gy) scalar potential on a uniform grid.
        grid_config: Dict-like with keys:
            - origin: (2,) world-space position of grid cell (0, 0).
            - cell_size: scalar spacing between grid nodes.

    Returns:
        (2,) gravitational acceleration vector at *pos_single*.
    """
    origin = grid_config["origin"]
    h = grid_config["cell_size"]

    # Continuous grid coordinates
    gc = (pos_single - origin) / h  # (2,)

    # Integer cell index (lower-left corner)
    i0 = jnp.floor(gc).astype(jnp.int32)  # (2,)

    # Fractional offset within the cell [0, 1)
    f = gc - i0.astype(jnp.float32)  # (2,)

    gx, gy = potential_field.shape

    # Clamp indices so we never read out of bounds.
    ix0 = jnp.clip(i0[0], 0, gx - 2)
    iy0 = jnp.clip(i0[1], 0, gy - 2)

    # Four surrounding potential values
    phi00 = potential_field[ix0,     iy0]
    phi10 = potential_field[ix0 + 1, iy0]
    phi01 = potential_field[ix0,     iy0 + 1]
    phi11 = potential_field[ix0 + 1, iy0 + 1]

    # Finite-difference gradient along x at the two y-edges, then interpolate
    # along y to get the gradient at the fractional position.
    dphi_dx_y0 = (phi10 - phi00) / h
    dphi_dx_y1 = (phi11 - phi01) / h
    dphi_dx = dphi_dx_y0 * (1.0 - f[1]) + dphi_dx_y1 * f[1]

    dphi_dy_x0 = (phi01 - phi00) / h
    dphi_dy_x1 = (phi11 - phi10) / h
    dphi_dy = dphi_dy_x0 * (1.0 - f[0]) + dphi_dy_x1 * f[0]

    # Acceleration = -grad(phi)
    return -jnp.array([dphi_dx, dphi_dy])


def _interp_forces(pos, active, potential_field, grid_config):
    """Vectorised CIC force interpolation for all bodies.

    Inactive bodies receive zero acceleration.

    Args:
        pos: (N, 2) positions.
        active: (N,) boolean mask.
        potential_field: (Gx, Gy) potential grid.
        grid_config: Grid metadata dict.

    Returns:
        (N, 2) acceleration vectors.
    """
    acc = jax.vmap(lambda p: _cic_interp_force(p, potential_field, grid_config))(pos)
    # Zero out inactive bodies
    mask = active[:, None].astype(acc.dtype)
    return acc * mask


# ---------------------------------------------------------------------------
# Time-stepping
# ---------------------------------------------------------------------------

def bodies_step(state, potential_field, grid_config, dt):
    """Advance all bodies by one timestep using leapfrog integration.

    The gravitational potential field must already be computed (e.g. by a
    Poisson solver or direct summation on a grid).  Forces are interpolated
    from the grid to body positions via CIC.

    Scheme (kick-drift-kick):
        1. Interpolate acceleration from potential field.
        2. Half-kick velocities.
        3. Drift positions.
        4. Re-interpolate acceleration at new positions.
        5. Half-kick velocities again.

    Args:
        state: Current ``BodyState``.
        potential_field: (Gx, Gy) gravitational potential array.
        grid_config: Dict with ``origin`` (2,) and ``cell_size`` scalar.
        dt: Timestep in seconds (scalar).

    Returns:
        Updated ``BodyState``.
    """
    dt_half = dt * 0.5

    # --- Kick 1 ---
    acc0 = _interp_forces(state.pos, state.active, potential_field, grid_config)
    vel_half = leapfrog_kick(state.vel, acc0, dt_half)

    # --- Drift ---
    new_pos = leapfrog_drift(state.pos, vel_half, dt)

    # --- Kick 2 (forces at new positions) ---
    acc1 = _interp_forces(new_pos, state.active, potential_field, grid_config)
    new_vel = leapfrog_kick(vel_half, acc1, dt_half)

    # Zero out inactive rows to keep padding clean.
    mask2 = state.active[:, None].astype(new_pos.dtype)
    new_pos = new_pos * mask2
    new_vel = new_vel * mask2

    return state._replace(pos=new_pos, vel=new_vel)


# ---------------------------------------------------------------------------
# Target helpers
# ---------------------------------------------------------------------------

def get_target_state(state):
    """Extract position and velocity of the designated target body.

    Args:
        state: A ``BodyState``.

    Returns:
        Tuple (pos, vel) each of shape (2,).  If no target is marked the
        result is zeros (safe for jit — no dynamic shapes).
    """
    # is_target is a boolean (N,) mask with exactly one True entry.
    # Use a dot product to select the row without dynamic indexing.
    weight = state.is_target.astype(jnp.float32)  # (N,)
    target_pos = jnp.einsum("n,nd->d", weight, state.pos)
    target_vel = jnp.einsum("n,nd->d", weight, state.vel)
    return target_pos, target_vel
