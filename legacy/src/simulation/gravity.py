"""
2D FFT-based Poisson gravity solver using the Hockney-Eastwood method.

Solves the 2D Poisson equation for gravity:
    nabla^2 Phi = 4 pi G sigma
where sigma is the surface mass density and Phi is the gravitational potential.

Uses zero-padded convolution (grid doubled to 2N x 2N) to enforce open
(isolated) boundary conditions, avoiding the periodic artifacts of a naive
FFT solve.  Mass assignment and force interpolation use the Cloud-in-Cell
(CIC / bilinear) scheme.

All functions are pure JAX and are compatible with jit / vmap / grad.
"""

from __future__ import annotations

import functools
from typing import NamedTuple

import jax
import jax.numpy as jnp


# ---------------------------------------------------------------------------
# Physical constant (SI).  Callers can rescale to game units externally.
# ---------------------------------------------------------------------------
G_GRAV = 6.67430e-11  # m^3 kg^-1 s^-2


# ---------------------------------------------------------------------------
# Solver state — a frozen container of precomputed spectral arrays.
# Using a NamedTuple so it is a valid JAX pytree leaf-container and works
# seamlessly with jit.
# ---------------------------------------------------------------------------
class SolverConfig(NamedTuple):
    """Static (non-traced) solver parameters.

    These values are used to determine array shapes and must be concrete
    Python values — they cannot be JAX tracers.
    """
    n: int       # grid cells per axis
    dx: float    # cell spacing in metres


class SolverState(NamedTuple):
    """Precomputed arrays for the Poisson solve (JAX-traced).

    Only contains arrays that participate in JAX tracing.
    For shape-determining parameters, use the companion ``SolverConfig``.
    """
    green_hat: jnp.ndarray  # (2*n, n+1) — rfft2 of the Green's function


# ---------------------------------------------------------------------------
# Construction
# ---------------------------------------------------------------------------
def create_solver(
    n: int = 256,
    domain_size: float = 100.0 * 1.496e11,  # 100 AU in metres
) -> tuple[SolverConfig, SolverState]:
    """Build the precomputed spectral arrays for an n x n physical grid.

    Parameters
    ----------
    n : int
        Number of grid cells per axis in the *physical* domain.
    domain_size : float
        Side length of the square physical domain in metres.

    Returns
    -------
    (SolverConfig, SolverState)
        Config holds static ints/floats, State holds JAX arrays.
    """
    dx = domain_size / n

    # ------------------------------------------------------------------
    # Green's function for 2D gravity on the doubled (2N x 2N) grid.
    #
    # The free-space Green's function for the 2D Laplacian is:
    #     G(r) = -1 / (2 pi) * ln(r)
    #
    # We sample it on a grid centred at (0, 0) that spans
    # [-N*dx, (N-1)*dx] in each dimension so that the circular
    # convolution on the doubled grid reproduces the *linear*
    # (aperiodic) convolution on the physical grid.
    # ------------------------------------------------------------------
    # Cell-centre coordinates on the doubled grid, wrapped so that
    # index 0 corresponds to displacement 0 (convolution convention).
    ix = jnp.arange(2 * n)
    # Map indices to signed displacements: 0..N-1 -> 0..N-1,  N..2N-1 -> -N..-1
    ix = jnp.where(ix < n, ix, ix - 2 * n)

    # 2D coordinate grids (cell centres, in metres)
    rx = ix.astype(jnp.float64) * dx  # shape (2N,)
    ry = rx  # symmetric grid

    # Meshgrid — axis=0 is y, axis=1 is x (row-major)
    yy, xx = jnp.meshgrid(ry, rx, indexing="ij")
    r2 = xx ** 2 + yy ** 2

    # Avoid log(0) at the origin.  The standard regularisation is to
    # replace r=0 with a softening length of order dx so the self-force
    # is finite and well-behaved.
    softening2 = (0.5 * dx) ** 2
    r2_safe = jnp.where(r2 > 0.0, r2, softening2)

    # The fundamental solution of ∇²G = δ in 2D is:
    #     G(r) = +1/(2π) ln(r)  =  +1/(4π) ln(r²)
    # This yields Φ > 0 that increases with r (well at origin),
    # so F = -∇Φ is attractive (points inward).
    green = 1.0 / (4.0 * jnp.pi) * jnp.log(r2_safe)

    # Pre-multiply by the gravitational coupling so the convolution
    # directly yields the potential:
    #     Phi = 4 pi G  *  (green  *conv*  sigma)
    # Folding the 4 pi G into the kernel saves a multiply later.
    green = green * (4.0 * jnp.pi * G_GRAV)

    # Forward-transform (real FFT on 2N x 2N grid).
    green_hat = jnp.fft.rfft2(green)

    return SolverConfig(n=n, dx=dx), SolverState(green_hat=green_hat)


# ---------------------------------------------------------------------------
# CIC (Cloud-in-Cell) mass assignment
# ---------------------------------------------------------------------------
@functools.partial(jax.jit, static_argnums=(2, 3))
def cic_deposit(
    positions: jnp.ndarray,
    masses: jnp.ndarray,
    n: int,
    dx: float,
) -> jnp.ndarray:
    """Deposit point masses onto an n x n grid using CIC (bilinear) weights.

    The grid origin is at (0, 0); cell centres are at (i+0.5)*dx.

    Parameters
    ----------
    positions : (N_bodies, 2)  — (x, y) in metres.
    masses    : (N_bodies,)    — mass of each body in kg.
    n         : int            — grid cells per axis.
    dx        : float          — cell spacing in metres.

    Returns
    -------
    density : (n, n)  — surface mass density (kg / m^2) on the grid.
    """
    # Normalised coordinates (0-based, cell-centre at i+0.5).
    x = positions[:, 0] / dx
    y = positions[:, 1] / dx

    # Nearest grid point to the *left* of the particle.
    # CIC stencil spans cells (ix, ix+1) x (iy, iy+1).
    ix = jnp.floor(x - 0.5).astype(jnp.int32)
    iy = jnp.floor(y - 0.5).astype(jnp.int32)

    # Fractional distance from left cell centre to particle.
    fx = (x - 0.5) - ix.astype(jnp.float64)
    fy = (y - 0.5) - iy.astype(jnp.float64)

    # Bilinear weights (4 contributions per particle).
    w00 = (1.0 - fx) * (1.0 - fy)
    w10 = fx * (1.0 - fy)
    w01 = (1.0 - fx) * fy
    w11 = fx * fy

    # Clamp indices to [0, n-1] so out-of-domain particles are silently
    # folded to the boundary.  For a game this avoids hard crashes;
    # physically it is the caller's job to keep bodies inside the domain.
    def _clamp(i):
        return jnp.clip(i, 0, n - 1)

    ix0 = _clamp(ix)
    ix1 = _clamp(ix + 1)
    iy0 = _clamp(iy)
    iy1 = _clamp(iy + 1)

    # Scatter-add onto the density grid.
    # JAX has no in-place mutation; we use `.at[].add()` on a zero array.
    cell_area = dx * dx
    density = jnp.zeros((n, n), dtype=jnp.float64)
    density = density.at[iy0, ix0].add(masses * w00 / cell_area)
    density = density.at[iy0, ix1].add(masses * w10 / cell_area)
    density = density.at[iy1, ix0].add(masses * w01 / cell_area)
    density = density.at[iy1, ix1].add(masses * w11 / cell_area)

    return density


# ---------------------------------------------------------------------------
# FFT Poisson solve (Hockney-Eastwood)
# ---------------------------------------------------------------------------
@functools.partial(jax.jit, static_argnums=(1,))
def solve_potential(
    density: jnp.ndarray,
    cfg: SolverConfig,
    state: SolverState,
) -> jnp.ndarray:
    """Solve for the gravitational potential on the physical grid.

    Implements the Hockney-Eastwood isolated-boundary convolution:
      1. Zero-pad density from N x N  ->  2N x 2N.
      2. FFT, multiply by precomputed Green's function, inverse FFT.
      3. Extract the N x N physical region.

    Parameters
    ----------
    density : (n, n)
        Surface mass density in kg / m^2.
    cfg : SolverConfig
        Static grid parameters (n, dx).
    state : SolverState
        Precomputed solver arrays (green_hat).

    Returns
    -------
    potential : (n, n)
        Gravitational potential Phi in J / kg (= m^2 / s^2).
    """
    n = cfg.n
    dx = cfg.dx

    # 1. Zero-pad to 2N x 2N.
    padded = jnp.zeros((2 * n, 2 * n), dtype=density.dtype)
    padded = padded.at[:n, :n].set(density)

    # 2. Forward transform.
    rho_hat = jnp.fft.rfft2(padded)

    # 3. Convolution in Fourier space: multiply by Green's function.
    #    The factor dx^2 converts the discrete sum into an integral
    #    (convolution theorem for sampled signals).
    phi_hat = rho_hat * state.green_hat * (dx * dx)

    # 4. Inverse transform (result is real-valued).
    phi_padded = jnp.fft.irfft2(phi_hat, s=(2 * n, 2 * n))

    # 5. Extract the physical N x N patch.
    potential = phi_padded[:n, :n]

    return potential


# ---------------------------------------------------------------------------
# Force computation via finite differences
# ---------------------------------------------------------------------------
@jax.jit
def gradient_potential(
    potential: jnp.ndarray,
    dx: float,
) -> tuple[jnp.ndarray, jnp.ndarray]:
    """Compute the gravitational acceleration field from the potential.

    Uses second-order central differences in the interior and one-sided
    differences at the boundaries.

        a_x = -dPhi/dx,   a_y = -dPhi/dy

    Parameters
    ----------
    potential : (n, n)
    dx : float — grid spacing (metres).

    Returns
    -------
    (ax, ay) : tuple of (n, n) arrays — acceleration components (m / s^2).
        ax corresponds to the x-direction (axis=1),
        ay corresponds to the y-direction (axis=0).
    """
    # jnp.gradient returns derivatives along each axis in order (axis 0, 1).
    # axis 0 = y, axis 1 = x.
    dphi_dy, dphi_dx = jnp.gradient(potential, dx)

    ax = -dphi_dx
    ay = -dphi_dy

    return ax, ay


# ---------------------------------------------------------------------------
# CIC force interpolation back to particles
# ---------------------------------------------------------------------------
@functools.partial(jax.jit, static_argnums=(3, 4))
def cic_interpolate(
    positions: jnp.ndarray,
    field_x: jnp.ndarray,
    field_y: jnp.ndarray,
    n: int,
    dx: float,
) -> jnp.ndarray:
    """Interpolate a 2D vector field to particle positions using CIC weights.

    Uses the *same* bilinear weights as `cic_deposit` so the scheme is
    momentum-conserving (Newton III) on the grid level.

    Parameters
    ----------
    positions : (N_bodies, 2)
    field_x   : (n, n) — x-component of the field.
    field_y   : (n, n) — y-component of the field.
    n, dx     : grid parameters.

    Returns
    -------
    values : (N_bodies, 2) — interpolated (vx, vy) at each particle.
    """
    x = positions[:, 0] / dx
    y = positions[:, 1] / dx

    ix = jnp.floor(x - 0.5).astype(jnp.int32)
    iy = jnp.floor(y - 0.5).astype(jnp.int32)

    fx = (x - 0.5) - ix.astype(jnp.float64)
    fy = (y - 0.5) - iy.astype(jnp.float64)

    w00 = (1.0 - fx) * (1.0 - fy)
    w10 = fx * (1.0 - fy)
    w01 = (1.0 - fx) * fy
    w11 = fx * fy

    def _clamp(i):
        return jnp.clip(i, 0, n - 1)

    ix0 = _clamp(ix)
    ix1 = _clamp(ix + 1)
    iy0 = _clamp(iy)
    iy1 = _clamp(iy + 1)

    # Gather from the field grids (same weight pattern as deposit).
    def _interp(field):
        return (
            w00 * field[iy0, ix0]
            + w10 * field[iy0, ix1]
            + w01 * field[iy1, ix0]
            + w11 * field[iy1, ix1]
        )

    vx = _interp(field_x)
    vy = _interp(field_y)

    return jnp.stack([vx, vy], axis=-1)


# ---------------------------------------------------------------------------
# Full pipeline: positions + masses  -->  per-particle accelerations
# ---------------------------------------------------------------------------
@functools.partial(jax.jit, static_argnums=(2,))
def compute_accelerations(
    positions: jnp.ndarray,
    masses: jnp.ndarray,
    cfg: SolverConfig,
    state: SolverState,
) -> jnp.ndarray:
    """End-to-end gravity solve: deposit, FFT solve, differentiate, interpolate.

    Parameters
    ----------
    positions : (N_bodies, 2) — (x, y) in metres.
    masses    : (N_bodies,)   — body masses in kg.
    cfg       : SolverConfig  — static grid parameters.
    state     : SolverState   — precomputed solver data.

    Returns
    -------
    accelerations : (N_bodies, 2) — (ax, ay) in m / s^2.
    """
    n = cfg.n
    dx = cfg.dx

    # 1. Mass assignment (CIC).
    density = cic_deposit(positions, masses, n, dx)

    # 2. Poisson solve for potential.
    potential = solve_potential(density, cfg, state)

    # 3. Gradient  ->  acceleration field on the grid.
    ax_field, ay_field = gradient_potential(potential, dx)

    # 4. Interpolate accelerations to particle positions (CIC).
    accelerations = cic_interpolate(positions, ax_field, ay_field, n, dx)

    return accelerations


# ---------------------------------------------------------------------------
# Direct N-body forces (3D Newtonian 1/r² law, projected onto 2D plane)
#
# For small body counts (< 100) this is faster and more physically accurate
# than the Poisson solver.  Use the Poisson solver for the smooth potential
# field (observation / visualization) and this for actual dynamics.
# ---------------------------------------------------------------------------

@jax.jit
def direct_nbody_accelerations(
    positions: jnp.ndarray,
    masses: jnp.ndarray,
    active: jnp.ndarray,
    softening: float = 1.0e9,
) -> jnp.ndarray:
    """Compute gravitational accelerations via direct pairwise summation.

    Uses the 3D Newtonian force law F = -G*M*m/r^2 projected onto the 2D
    plane.  This gives physically intuitive orbital dynamics (Kepler orbits)
    even though the simulation is 2D.

    Parameters
    ----------
    positions : (N, 2) — body positions in metres.
    masses    : (N,)   — body masses in kg.
    active    : (N,)   — boolean mask (inactive bodies ignored).
    softening : float  — softening length in metres to avoid singularities.

    Returns
    -------
    accelerations : (N, 2) — gravitational acceleration per body (m/s²).
    """
    # Pairwise displacement: disp[i, j] = pos[j] - pos[i]
    disp = positions[None, :, :] - positions[:, None, :]   # (N, N, 2)
    r2 = jnp.sum(disp ** 2, axis=-1)                       # (N, N)

    # Softened distance: avoids division by zero at r=0
    r2_soft = r2 + softening ** 2
    r = jnp.sqrt(r2_soft)                                  # (N, N)

    # 3D gravitational acceleration magnitude: |a_ij| = G * m_j / r²
    # Direction: unit vector from i toward j = disp / r
    # Full vector: a_ij = G * m_j / r² * (disp / r) = G * m_j * disp / r³
    inv_r3 = 1.0 / (r2_soft * r)                           # 1 / r³

    # Mask: only active bodies contribute, and no self-interaction
    mask = active[None, :] * active[:, None]                # (N, N)
    eye_mask = 1.0 - jnp.eye(positions.shape[0])            # (N, N)
    mask = mask * eye_mask

    # Acceleration on body i from body j:
    #   a_i = sum_j  G * m_j * (pos_j - pos_i) / |r_ij|^3
    weighted = G_GRAV * masses[None, :] * inv_r3 * mask     # (N, N)
    acc = jnp.sum(weighted[:, :, None] * disp, axis=1)      # (N, 2)

    # Zero out inactive bodies
    acc = acc * active[:, None]

    return acc


@jax.jit
def direct_acceleration_at(
    point: jnp.ndarray,
    body_positions: jnp.ndarray,
    body_masses: jnp.ndarray,
    body_active: jnp.ndarray,
    softening: float = 1.0e9,
) -> jnp.ndarray:
    """Compute gravitational acceleration at a single point from all bodies.

    Parameters
    ----------
    point          : (2,) — position to evaluate at (metres).
    body_positions : (N, 2)
    body_masses    : (N,)
    body_active    : (N,)
    softening      : float

    Returns
    -------
    acceleration : (2,) — m/s².
    """
    disp = body_positions - point[None, :]       # (N, 2)
    r2 = jnp.sum(disp ** 2, axis=-1)            # (N,)
    r2_soft = r2 + softening ** 2
    r = jnp.sqrt(r2_soft)
    inv_r3 = 1.0 / (r2_soft * r)

    # a = sum_j G * m_j * disp_j / |r_j|^3
    weighted = G_GRAV * body_masses * inv_r3 * body_active  # (N,)
    acc = jnp.sum(weighted[:, None] * disp, axis=0)         # (2,)

    return acc
