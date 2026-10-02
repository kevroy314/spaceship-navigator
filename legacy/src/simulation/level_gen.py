"""Procedural level generation for 2D solar-system-scale gravitational simulation.

Generates randomized levels with a central star, orbiting planets, rogue bodies,
and a ship start/target configuration.  All functions use ``jax.random`` for
reproducibility and are designed to be jittable (fixed-size padded arrays with
an ``active`` boolean mask).

Constants are in SI (meters, kilograms, seconds).
"""

from typing import NamedTuple, Tuple

import jax
import jax.numpy as jnp

from simulation.bodies import BodyState, MAX_BODIES

# ---------------------------------------------------------------------------
# Physical constants
# ---------------------------------------------------------------------------

G: float = 6.674e-11           # gravitational constant  [m^3 kg^-1 s^-2]
AU: float = 1.496e11           # astronomical unit        [m]

# ---------------------------------------------------------------------------
# Data containers
# ---------------------------------------------------------------------------


class LevelConfig(NamedTuple):
    """All tuneable knobs for procedural level generation.

    Distributions are based on real solar system data:
    - Geometric orbit spacing (Titius-Bode-like, ratio ~1.4-2.0x)
    - Bimodal mass distribution (terrestrial + gas giant)
    - Rayleigh eccentricity distribution (most planets near-circular)
    """

    seed: int = 42
    num_planets_range: Tuple[int, int] = (4, 10)
    primary_mass_range: Tuple[float, float] = (1.0e30, 2.5e30)
    terrestrial_mass_range: Tuple[float, float] = (1.0e23, 1.0e25)
    giant_mass_range: Tuple[float, float] = (1.0e26, 1.0e28)
    giant_fraction: float = 0.25
    rogue_mass_range: Tuple[float, float] = (1.0e15, 1.0e20)
    num_rogues_range: Tuple[int, int] = (1, 3)
    inner_orbit_au: float = 0.3
    orbit_spacing_range: Tuple[float, float] = (1.4, 2.0)
    eccentricity_sigma: float = 0.06
    max_eccentricity: float = 0.3
    min_start_target_distance: float = 5.0    # AU
    domain_size: float = 100.0                 # AU
    softening: float = 1.0e9                   # meters


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _uniform(key, low, high, shape=()):
    """Sample uniformly from [low, high)."""
    return jax.random.uniform(key, shape=shape, minval=low, maxval=high)


def _log_uniform(key, low, high, shape=()):
    """Sample log-uniformly from [low, high) (useful for masses/radii)."""
    log_low = jnp.log(low)
    log_high = jnp.log(high)
    return jnp.exp(jax.random.uniform(key, shape=shape, minval=log_low, maxval=log_high))


def _mass_to_radius(mass):
    """Rough radius estimate: R ~ M^(1/3) scaled so Earth-mass ~ Earth-radius.

    This is *not* physically accurate — it just gives a plausible radius for
    collision / proximity checks.  Returns radius in meters.
    """
    earth_mass = 5.972e24
    earth_radius = 6.371e6
    return earth_radius * (mass / earth_mass) ** (1.0 / 3.0)


def _circular_orbit_speed(primary_mass, orbital_radius):
    """Circular-orbit speed: v = sqrt(G * M / r)."""
    return jnp.sqrt(G * primary_mass / orbital_radius)


# ---------------------------------------------------------------------------
# Main generation
# ---------------------------------------------------------------------------


def generate_level(
    config: LevelConfig,
    key: jnp.ndarray,
) -> Tuple[BodyState, jnp.ndarray, jnp.ndarray]:
    """Procedurally generate one simulation level.

    Uses realistic solar-system-inspired distributions:
    - Geometric orbit spacing (Titius-Bode-like)
    - Bimodal mass distribution (terrestrial + gas giant)
    - Rayleigh eccentricity distribution
    - 3D Newtonian circular orbit velocities: v = sqrt(GM/r)

    Args:
        config: Generation parameters.
        key: A ``jax.random.PRNGKey``.

    Returns:
        (body_state, ship_start_pos, ship_start_vel)
    """
    domain_m = config.domain_size * AU
    half_domain = domain_m / 2.0

    (
        k_primary_offset, k_primary_mass,
        k_num_planets, k_planets,
        k_num_rogues, k_rogues,
        k_target, k_ship,
    ) = jax.random.split(key, 8)

    # ------------------------------------------------------------------
    # 1. Primary body (star) at domain centre
    # ------------------------------------------------------------------
    center = jnp.array([half_domain, half_domain])
    primary_offset = _uniform(k_primary_offset, -0.3 * AU, 0.3 * AU, shape=(2,))
    primary_pos = center + primary_offset
    primary_mass = _log_uniform(
        k_primary_mass,
        config.primary_mass_range[0],
        config.primary_mass_range[1],
    )
    primary_vel = jnp.zeros(2)

    # ------------------------------------------------------------------
    # 2. Planets with geometric orbit spacing + bimodal masses
    # ------------------------------------------------------------------
    num_pl_min, num_pl_max = config.num_planets_range
    num_planets = jax.random.randint(
        k_num_planets, shape=(), minval=num_pl_min, maxval=num_pl_max + 1,
    )

    planet_keys = jax.random.split(k_planets, MAX_BODIES)

    def _make_planet(idx, subkey):
        k_spacing, k_ecc, k_phase, k_mass_type, k_mass = jax.random.split(subkey, 5)

        # Geometric orbit spacing: r_i = inner_orbit * prod(spacing_j, j<i)
        # For vmapped code, compute as: r = inner_orbit * spacing^idx
        spacing = _uniform(k_spacing,
                           config.orbit_spacing_range[0],
                           config.orbit_spacing_range[1])
        orbital_radius_au = config.inner_orbit_au * (spacing ** idx.astype(float))
        orbital_radius = orbital_radius_au * AU

        # Rayleigh eccentricity: |N(0,σ)| clamped
        ecc_raw = jnp.abs(jax.random.normal(k_ecc) * config.eccentricity_sigma)
        ecc = jnp.minimum(ecc_raw, config.max_eccentricity)

        # Random orbital phase
        phase = _uniform(k_phase, 0.0, 2.0 * jnp.pi)

        # Bimodal mass: terrestrial or giant
        is_giant = jax.random.uniform(k_mass_type) < config.giant_fraction
        mass_terr = _log_uniform(k_mass,
                                 config.terrestrial_mass_range[0],
                                 config.terrestrial_mass_range[1])
        mass_giant = _log_uniform(k_mass,
                                  config.giant_mass_range[0],
                                  config.giant_mass_range[1])
        mass = jnp.where(is_giant, mass_giant, mass_terr)

        # Position at current orbital phase (start at periapsis-ish)
        r = orbital_radius * (1.0 - ecc)
        rel_pos = r * jnp.array([jnp.cos(phase), jnp.sin(phase)])
        pos = primary_pos + rel_pos

        # Velocity: circular orbit speed (3D Kepler: v = sqrt(GM/r))
        # At periapsis of elliptical orbit: v_p = v_circ * sqrt((1+e)/(1-e))
        v_circ = _circular_orbit_speed(primary_mass, r)
        v_peri = v_circ * jnp.sqrt((1.0 + ecc) / jnp.maximum(1.0 - ecc, 0.01))
        tangent = jnp.array([-jnp.sin(phase), jnp.cos(phase)])
        vel = v_peri * tangent

        return pos, vel, mass

    planet_pos_all, planet_vel_all, planet_mass_all = jax.vmap(
        _make_planet,
    )(jnp.arange(MAX_BODIES), planet_keys)

    # ------------------------------------------------------------------
    # 3. Rogue bodies (asteroids/comets on hyperbolic trajectories)
    # ------------------------------------------------------------------
    num_rogues_min, num_rogues_max = config.num_rogues_range
    num_rogues = jax.random.randint(
        k_num_rogues, shape=(), minval=num_rogues_min, maxval=num_rogues_max + 1,
    )

    rogue_keys = jax.random.split(k_rogues, MAX_BODIES)

    def _make_rogue(idx, subkey):
        k_angle, k_dist, k_speed, k_mass = jax.random.split(subkey, 4)

        angle = _uniform(k_angle, 0.0, 2.0 * jnp.pi)
        dist = _uniform(k_dist, 30.0 * AU, 45.0 * AU)
        pos = primary_pos + dist * jnp.array([jnp.cos(angle), jnp.sin(angle)])

        v_esc = _circular_orbit_speed(primary_mass, dist) * jnp.sqrt(2.0)
        speed = _uniform(k_speed, v_esc * 1.05, v_esc * 1.5)
        inward = -jnp.array([jnp.cos(angle), jnp.sin(angle)])
        deflection = _uniform(k_dist, -jnp.pi / 6.0, jnp.pi / 6.0)
        cos_d, sin_d = jnp.cos(deflection), jnp.sin(deflection)
        vel_dir = jnp.array([
            inward[0] * cos_d - inward[1] * sin_d,
            inward[0] * sin_d + inward[1] * cos_d,
        ])
        vel = speed * vel_dir

        mass = _log_uniform(k_mass,
                            config.rogue_mass_range[0],
                            config.rogue_mass_range[1])
        return pos, vel, mass

    rogue_pos_all, rogue_vel_all, rogue_mass_all = jax.vmap(
        _make_rogue,
    )(jnp.arange(MAX_BODIES), rogue_keys)

    # ------------------------------------------------------------------
    # 4. Assemble padded arrays
    # ------------------------------------------------------------------
    total_active = 1 + num_planets + num_rogues
    total_active = jnp.minimum(total_active, MAX_BODIES)

    all_pos = jnp.zeros((MAX_BODIES, 2))
    all_vel = jnp.zeros((MAX_BODIES, 2))
    all_mass = jnp.zeros((MAX_BODIES,))

    all_pos = all_pos.at[0].set(primary_pos)
    all_vel = all_vel.at[0].set(primary_vel)
    all_mass = all_mass.at[0].set(primary_mass)

    all_pos = all_pos.at[1:].set(planet_pos_all[:MAX_BODIES - 1])
    all_vel = all_vel.at[1:].set(planet_vel_all[:MAX_BODIES - 1])
    all_mass = all_mass.at[1:].set(planet_mass_all[:MAX_BODIES - 1])

    # Place rogues after planets
    def _place_rogue(carry, idx):
        pos_arr, vel_arr, mass_arr = carry
        slot = 1 + num_planets + idx
        in_bounds = (slot < MAX_BODIES) & (idx < num_rogues)
        pos_arr = jnp.where(in_bounds,
                            pos_arr.at[slot].set(rogue_pos_all[idx]), pos_arr)
        vel_arr = jnp.where(in_bounds,
                            vel_arr.at[slot].set(rogue_vel_all[idx]), vel_arr)
        mass_arr = jnp.where(in_bounds,
                             mass_arr.at[slot].set(rogue_mass_all[idx]), mass_arr)
        return (pos_arr, vel_arr, mass_arr), None

    max_rogues = config.num_rogues_range[1]
    (all_pos, all_vel, all_mass), _ = jax.lax.scan(
        _place_rogue, (all_pos, all_vel, all_mass), jnp.arange(max_rogues),
    )

    body_indices = jnp.arange(MAX_BODIES)
    active = body_indices < total_active
    radii = _mass_to_radius(all_mass)

    # ------------------------------------------------------------------
    # 5. Target: random planet (not the star)
    # ------------------------------------------------------------------
    target_idx = jax.random.randint(k_target, shape=(), minval=1, maxval=num_pl_max + 1)
    target_idx = jnp.clip(target_idx, 1, num_planets)
    is_target = body_indices == target_idx

    # ------------------------------------------------------------------
    # 6. Ship start: in orbit, some distance from target
    # ------------------------------------------------------------------
    k_ship_angle, k_ship_dist = jax.random.split(k_ship, 2)
    target_pos = all_pos[target_idx]

    ship_angle = _uniform(k_ship_angle, 0.0, 2.0 * jnp.pi)
    ship_dist = _uniform(
        k_ship_dist,
        config.min_start_target_distance * AU,
        config.min_start_target_distance * AU * 2.0,
    )
    ship_start_pos = target_pos + ship_dist * jnp.array([
        jnp.cos(ship_angle), jnp.sin(ship_angle),
    ])
    # Clamp to domain
    ship_start_pos = jnp.clip(ship_start_pos, AU, domain_m - AU)

    # Ship velocity: match local circular orbit around primary
    r_ship = jnp.linalg.norm(ship_start_pos - primary_pos)
    r_ship = jnp.maximum(r_ship, AU * 0.1)
    v_orbit = _circular_orbit_speed(primary_mass, r_ship)
    radial = (ship_start_pos - primary_pos) / r_ship
    ship_tangent = jnp.array([-radial[1], radial[0]])
    ship_start_vel = v_orbit * ship_tangent

    body_state = BodyState(
        pos=all_pos, vel=all_vel, mass=all_mass,
        radius=radii, active=active, is_target=is_target,
    )

    return body_state, ship_start_pos, ship_start_vel


# ---------------------------------------------------------------------------
# Validation
# ---------------------------------------------------------------------------


def validate_level(
    body_state: BodyState,
    ship_start_pos: jnp.ndarray,
    corridor_width_factor: float = 3.0,
) -> jnp.ndarray:
    """Run basic sanity checks on a generated level.

    All checks are computed with JAX and return a single boolean (True = valid).
    The function is jittable.

    Checks:
        1. No two active bodies overlap (centres closer than sum of radii).
        2. Ship start is not inside any active body's radius.
        3. At least one active body (other than the target) lies within a
           broadened corridor between the ship start and the target.

    Args:
        body_state: Padded body state from :func:`generate_level`.
        ship_start_pos: Ship starting position, shape ``(2,)``.
        corridor_width_factor: Half-width of the corridor is this factor
            times the body's radius (default 3.0).

    Returns:
        Scalar boolean ``jnp.ndarray``.  True if the level passes.
    """
    pos = body_state.pos          # (N, 2)
    radii = body_state.radius     # (N,)
    active = body_state.active    # (N,)
    is_target = body_state.is_target  # (N,)

    n = MAX_BODIES

    # ------------------------------------------------------------------
    # Check 1: no overlapping active bodies
    # ------------------------------------------------------------------
    # Pairwise distances
    diff = pos[:, None, :] - pos[None, :, :]              # (N, N, 2)
    dist = jnp.sqrt(jnp.sum(diff ** 2, axis=-1) + 1e-30)  # (N, N)
    sum_radii = radii[:, None] + radii[None, :]            # (N, N)

    # Mask: only check pairs where both are active and i != j
    both_active = active[:, None] & active[None, :]
    not_self = ~jnp.eye(n, dtype=bool)
    overlap_check = both_active & not_self

    overlaps = overlap_check & (dist < sum_radii)
    no_overlaps = ~jnp.any(overlaps)

    # ------------------------------------------------------------------
    # Check 2: ship not inside any body
    # ------------------------------------------------------------------
    ship_body_dist = jnp.sqrt(jnp.sum((pos - ship_start_pos[None, :]) ** 2, axis=-1) + 1e-30)
    ship_inside = active & (ship_body_dist < radii)
    ship_safe = ~jnp.any(ship_inside)

    # ------------------------------------------------------------------
    # Check 3: path from start to target crosses >= 1 gravitational well
    # ------------------------------------------------------------------
    target_pos = jnp.sum(pos * is_target[:, None], axis=0)  # (2,)

    # Direction vector along the corridor
    corridor_vec = target_pos - ship_start_pos                   # (2,)
    corridor_len = jnp.sqrt(jnp.sum(corridor_vec ** 2) + 1e-30)
    corridor_dir = corridor_vec / corridor_len

    # Project each body centre onto the corridor line
    to_body = pos - ship_start_pos[None, :]                      # (N, 2)
    proj = jnp.sum(to_body * corridor_dir[None, :], axis=-1)     # (N,)
    perp_vec = to_body - proj[:, None] * corridor_dir[None, :]   # (N, 2)
    perp_dist = jnp.sqrt(jnp.sum(perp_vec ** 2, axis=-1) + 1e-30)  # (N,)

    # Body is "in corridor" if its projection falls between 0 and corridor_len
    # and its perpendicular distance is within corridor_width_factor * radius.
    in_corridor = (
        active
        & ~is_target
        & (proj > 0.0)
        & (proj < corridor_len)
        & (perp_dist < corridor_width_factor * radii)
    )
    has_obstacle = jnp.any(in_corridor)

    return no_overlaps & ship_safe & has_obstacle


# ---------------------------------------------------------------------------
# Deterministic hashing
# ---------------------------------------------------------------------------


def level_to_config_hash(config: LevelConfig) -> int:
    """Return a deterministic 64-bit hash of level-generation parameters.

    Useful for identifying levels reproducibly.  Two configs with identical
    fields will always produce the same hash.

    This function is *not* jittable (it uses Python-level hashing) and is
    intended for bookkeeping / logging.

    Args:
        config: A :class:`LevelConfig` instance.

    Returns:
        A Python ``int`` in the range ``[0, 2**64)``.
    """
    import hashlib
    import struct

    h = hashlib.sha256()
    for field_name in config._fields:
        value = getattr(config, field_name)
        if isinstance(value, tuple):
            for v in value:
                h.update(struct.pack("<d", float(v)))
        elif isinstance(value, (int,)):
            h.update(struct.pack("<q", value))
        elif isinstance(value, float):
            h.update(struct.pack("<d", value))
        else:
            h.update(str(value).encode("utf-8"))

    digest = h.digest()
    # Take first 8 bytes as unsigned 64-bit int (little-endian)
    return struct.unpack("<Q", digest[:8])[0]
