"""Light-speed-delayed gravitational field observation for RL spaceship navigation.

The ship observes the gravitational potential field, but information travels at
the speed of light. Distant regions of the field appear as they were in the past,
creating a "stale" observation where nearby values are current and distant ones
are delayed.
"""

from __future__ import annotations

import functools
from typing import NamedTuple

import jax
import jax.numpy as jnp


class LightDelayObserver(NamedTuple):
    """Immutable state for a light-delay observation system.

    Attributes:
        history_buffer: Ring buffer of past potential fields,
            shape (buffer_size, grid_H, grid_W).
        buffer_idx: Current write position in the ring buffer.
        buffer_size: Total history capacity.
        grid_size: Physics grid resolution (e.g. 256).
        obs_grid_size: Downsampled observation size for RL (e.g. 64).
        domain_size: Physical domain extent in meters.
        c: Speed of light in m/s.
        dt: Simulation timestep in seconds.
    """

    history_buffer: jnp.ndarray  # (buffer_size, grid_size, grid_size)
    buffer_idx: int
    buffer_size: int
    grid_size: int
    obs_grid_size: int
    domain_size: float
    c: float
    dt: float


def init_observer(
    buffer_size: int = 128,
    grid_size: int = 256,
    obs_grid_size: int = 64,
    domain_size: float = 1.0e12,
    c: float = 299_792_458.0,
    dt: float = 1.0,
) -> LightDelayObserver:
    """Create an observer with a zeroed history buffer.

    Args:
        buffer_size: Number of past frames to retain.
        grid_size: Side length of the square physics potential grid.
        obs_grid_size: Side length of the downsampled observation grid.
        domain_size: Physical extent of the simulation domain in meters.
        c: Speed of light in m/s.
        dt: Simulation timestep in seconds.

    Returns:
        A freshly initialised ``LightDelayObserver``.
    """
    history_buffer = jnp.zeros((buffer_size, grid_size, grid_size), dtype=jnp.float32)
    return LightDelayObserver(
        history_buffer=history_buffer,
        buffer_idx=jnp.int32(0),
        buffer_size=buffer_size,
        grid_size=grid_size,
        obs_grid_size=obs_grid_size,
        domain_size=domain_size,
        c=c,
        dt=dt,
    )


@jax.jit
def update_observer(
    observer: LightDelayObserver,
    new_potential_field: jnp.ndarray,
) -> LightDelayObserver:
    """Write a new potential field snapshot into the ring buffer.

    Args:
        observer: Current observer state.
        new_potential_field: Potential field of shape (grid_size, grid_size).

    Returns:
        Updated observer with the new frame written and buffer_idx advanced.
    """
    updated_buffer = observer.history_buffer.at[observer.buffer_idx].set(
        new_potential_field
    )
    next_idx = (observer.buffer_idx + 1) % observer.buffer_size
    return observer._replace(
        history_buffer=updated_buffer,
        buffer_idx=next_idx,
    )


@functools.partial(jax.jit, static_argnums=())
def observe(
    observer: LightDelayObserver,
    ship_position: jnp.ndarray,
) -> jnp.ndarray:
    """Produce a light-delay-corrected, downsampled observation.

    For every cell in the full physics grid the light travel time from the
    ship is computed and the corresponding historical potential value is
    looked up from the ring buffer.  The resulting 2-D field is then
    downsampled to ``obs_grid_size x obs_grid_size`` via average pooling
    and normalised to zero-mean, unit-variance.

    The observation is kept in world-frame — the light delay already
    encodes the ship's position implicitly since nearby cells are more
    current.

    Args:
        observer: Current observer state (must contain filled history).
        ship_position: Ship (x, y) position in meters, shape (2,).

    Returns:
        Normalised observation array of shape (obs_grid_size, obs_grid_size).
    """
    grid_size = observer.grid_size
    domain_size = observer.domain_size
    c = observer.c
    dt = observer.dt
    buffer_size = observer.buffer_size

    # --- cell centres in physical coordinates --------------------------------
    cell_size = domain_size / grid_size
    coords_1d = jnp.arange(grid_size, dtype=jnp.float32) * cell_size + 0.5 * cell_size
    cx, cy = jnp.meshgrid(coords_1d, coords_1d, indexing="xy")  # (grid, grid)

    # --- distance from ship to every cell ------------------------------------
    dx = cx - ship_position[0]
    dy = cy - ship_position[1]
    distances = jnp.sqrt(dx * dx + dy * dy)  # metres

    # --- delay in timesteps --------------------------------------------------
    delay_steps = jnp.floor(distances / (c * dt)).astype(jnp.int32)
    delay_steps = jnp.clip(delay_steps, 0, buffer_size - 1)

    # --- look up historical values via gather --------------------------------
    # buffer_idx currently points to the *next* write slot, so the most
    # recent frame is at (buffer_idx - 1).  Going back by ``delay_steps``
    # more gives the frame index for each cell.
    frame_indices = (observer.buffer_idx - 1 - delay_steps) % buffer_size

    # Flatten for advanced indexing then reshape back.
    row_indices = jnp.broadcast_to(
        jnp.arange(grid_size, dtype=jnp.int32)[:, None], (grid_size, grid_size)
    )
    col_indices = jnp.broadcast_to(
        jnp.arange(grid_size, dtype=jnp.int32)[None, :], (grid_size, grid_size)
    )

    delayed_field = observer.history_buffer[
        frame_indices, row_indices, col_indices
    ]  # (grid_size, grid_size)

    # --- downsample via average pooling --------------------------------------
    obs_grid_size = observer.obs_grid_size
    delayed_field = _average_pool(delayed_field, grid_size, obs_grid_size)

    # --- normalise -----------------------------------------------------------
    eps = 1e-8
    mean = jnp.mean(delayed_field)
    std = jnp.std(delayed_field)
    observation = (delayed_field - mean) / (std + eps)

    return observation


def _average_pool(
    field: jnp.ndarray,
    grid_size: int,
    obs_grid_size: int,
) -> jnp.ndarray:
    """Downsample *field* from (grid_size, grid_size) to (obs_grid_size, obs_grid_size).

    Uses reshape-then-mean when the grid divides evenly, otherwise falls
    back to ``jax.image.resize`` with bilinear interpolation (area average
    for downsampling).
    """
    # Fast path: exact integer ratio (256 -> 64 => ratio 4).
    ratio = grid_size // obs_grid_size
    even = ratio * obs_grid_size == grid_size

    def _reshape_pool(_field: jnp.ndarray) -> jnp.ndarray:
        return (
            _field.reshape(obs_grid_size, ratio, obs_grid_size, ratio)
            .mean(axis=(1, 3))
        )

    def _resize_pool(_field: jnp.ndarray) -> jnp.ndarray:
        return jax.image.resize(
            _field,
            shape=(obs_grid_size, obs_grid_size),
            method="bilinear",
        )

    return jax.lax.cond(even, _reshape_pool, _resize_pool, field)
