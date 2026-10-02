"""Gymnasium-compatible environment for spaceship navigation.

Wraps the JAX-based simulation into a standard RL interface.
Designed for use with PureJaxRL (fully JIT-compiled training).
"""

from typing import NamedTuple, Optional
import jax
import jax.numpy as jnp
import functools


class EnvConfig(NamedTuple):
    """Full environment configuration."""
    # Grid / physics
    grid_size: int = 256
    domain_size: float = 1.496e13      # 100 AU in meters
    dt: float = 600.0                   # 10 minutes
    max_steps: int = 4320              # 30 days
    G: float = 6.674e-11
    c: float = 299792458.0
    AU: float = 1.496e11

    # Ship
    ship_dry_mass: float = 1000.0
    ship_fuel_mass: float = 500.0
    ship_max_thrust: float = 10.0
    ship_exhaust_velocity: float = 30000.0
    ship_max_torque: float = 0.1
    ship_moment_arm: float = 2.0
    ship_rotation_exhaust_velocity: float = 3000000.0
    ship_moment_of_inertia: float = 5000.0

    # Target
    approach_radius: float = 1.496e9   # 0.01 AU
    approach_velocity: float = 1000.0

    # Observation
    obs_grid_size: int = 64
    history_buffer_size: int = 128

    # Reward
    arrival_bonus: float = 1000.0
    proximity_scale: float = 10.0
    time_penalty: float = -0.01
    fuel_penalty: float = -0.1
    velocity_match_bonus: float = 100.0

    # Level gen
    max_bodies: int = 32
    num_bodies_min: int = 5
    num_bodies_max: int = 20


class EnvState(NamedTuple):
    """Full environment state (carried through steps)."""
    # Ship
    ship_pos: jnp.ndarray           # (2,)
    ship_vel: jnp.ndarray           # (2,)
    ship_heading: float
    ship_angular_vel: float
    ship_fuel_mass: float

    # Bodies
    body_pos: jnp.ndarray           # (max_bodies, 2)
    body_vel: jnp.ndarray           # (max_bodies, 2)
    body_mass: jnp.ndarray          # (max_bodies,)
    body_active: jnp.ndarray        # (max_bodies,)
    body_is_target: jnp.ndarray     # (max_bodies,)

    # Observation buffer
    potential_history: jnp.ndarray  # (buffer_size, grid_size, grid_size)
    buffer_idx: int

    # Episode state
    step: int
    prev_distance: float
    cumulative_reward: float
    done: bool


class SpaceshipEnv:
    """JAX-native spaceship navigation environment.

    Compatible with PureJaxRL's environment interface:
    - reset(key) -> (obs, state)
    - step(key, state, action) -> (obs, state, reward, done, info)
    """

    def __init__(self, config: EnvConfig):
        self.config = config
        # Precompute solver arrays
        self._init_solver()

    def _init_solver(self):
        """Precompute FFT solver arrays (Green's function, wavenumbers)."""
        cfg = self.config
        N = cfg.grid_size
        h = cfg.domain_size / N

        # Padded grid for open boundaries (Hockney-Eastwood)
        N2 = 2 * N

        # Green's function for 2D gravity: G(r) = -1/(2π) * ln(r)
        x = jnp.arange(N2) * h - (N2 // 2) * h
        y = jnp.arange(N2) * h - (N2 // 2) * h
        X, Y = jnp.meshgrid(x, y, indexing='ij')
        R = jnp.sqrt(X**2 + Y**2)
        R = jnp.where(R < h * 0.5, h * 0.5, R)  # Softening

        green = -1.0 / (2.0 * jnp.pi) * jnp.log(R)
        green = jnp.fft.ifftshift(green)

        self._green_hat = jnp.fft.rfft2(green)
        self._h = h
        self._N2 = N2

    @functools.partial(jax.jit, static_argnums=(0,))
    def _solve_potential(self, body_pos, body_mass, body_active):
        """Compute gravitational potential field from body positions."""
        cfg = self.config
        N = cfg.grid_size
        h = self._h
        N2 = self._N2

        # CIC mass assignment onto padded grid
        density = jnp.zeros((N2, N2))

        # Convert positions to grid coordinates
        grid_pos = body_pos / h  # (max_bodies, 2)

        def assign_one_body(density, args):
            pos, mass, active = args
            ix = jnp.floor(pos[0]).astype(int)
            iy = jnp.floor(pos[1]).astype(int)
            dx = pos[0] - ix
            dy = pos[1] - iy

            # CIC weights
            w00 = (1.0 - dx) * (1.0 - dy) * mass * active
            w10 = dx * (1.0 - dy) * mass * active
            w01 = (1.0 - dx) * dy * mass * active
            w11 = dx * dy * mass * active

            # Offset to center of padded grid
            ox, oy = N // 2, N // 2
            ix, iy = ix + ox, iy + oy

            # Clamp indices
            ix = jnp.clip(ix, 0, N2 - 2)
            iy = jnp.clip(iy, 0, N2 - 2)

            density = density.at[ix, iy].add(w00)
            density = density.at[ix + 1, iy].add(w10)
            density = density.at[ix, iy + 1].add(w01)
            density = density.at[ix + 1, iy + 1].add(w11)
            return density, None

        density, _ = jax.lax.scan(
            assign_one_body, density,
            (grid_pos, body_mass, body_active)
        )

        # Density to field: ρ / h² (2D density per cell area)
        density = density / (h * h)

        # FFT convolution with Green's function
        rho_hat = jnp.fft.rfft2(density)
        phi_hat = 4.0 * jnp.pi * cfg.G * rho_hat * self._green_hat
        potential_full = jnp.fft.irfft2(phi_hat, s=(N2, N2))

        # Extract physical domain
        ox, oy = N // 2, N // 2
        potential = potential_full[ox:ox + N, oy:oy + N]

        return potential

    @functools.partial(jax.jit, static_argnums=(0,))
    def _interpolate_force(self, potential, pos):
        """Interpolate gravitational force at a position using CIC."""
        cfg = self.config
        N = cfg.grid_size
        h = self._h

        # Force = -gradient(potential) via central differences
        Fx = -(jnp.roll(potential, -1, axis=0) - jnp.roll(potential, 1, axis=0)) / (2.0 * h)
        Fy = -(jnp.roll(potential, -1, axis=1) - jnp.roll(potential, 1, axis=1)) / (2.0 * h)

        # CIC interpolation at pos
        grid_pos = pos / h
        ix = jnp.floor(grid_pos[0]).astype(int)
        iy = jnp.floor(grid_pos[1]).astype(int)
        dx = grid_pos[0] - ix
        dy = grid_pos[1] - iy

        ix = jnp.clip(ix, 0, N - 2)
        iy = jnp.clip(iy, 0, N - 2)

        fx = (Fx[ix, iy] * (1 - dx) * (1 - dy) +
              Fx[ix + 1, iy] * dx * (1 - dy) +
              Fx[ix, iy + 1] * (1 - dx) * dy +
              Fx[ix + 1, iy + 1] * dx * dy)

        fy = (Fy[ix, iy] * (1 - dx) * (1 - dy) +
              Fy[ix + 1, iy] * dx * (1 - dy) +
              Fy[ix, iy + 1] * (1 - dx) * dy +
              Fy[ix + 1, iy + 1] * dx * dy)

        return jnp.array([fx, fy])

    @functools.partial(jax.jit, static_argnums=(0,))
    def _compute_observation(self, state: EnvState):
        """Compute light-delayed potential field observation."""
        cfg = self.config
        N = cfg.grid_size
        obs_N = cfg.obs_grid_size
        h = self._h

        # Distance from ship to each grid cell
        cell_x = jnp.arange(N) * h + h / 2
        cell_y = jnp.arange(N) * h + h / 2
        CX, CY = jnp.meshgrid(cell_x, cell_y, indexing='ij')

        dist = jnp.sqrt((CX - state.ship_pos[0])**2 + (CY - state.ship_pos[1])**2)
        delay_steps = jnp.floor(dist / (cfg.c * cfg.dt)).astype(int)
        delay_steps = jnp.clip(delay_steps, 0, cfg.history_buffer_size - 1)

        # Gather from ring buffer
        history_idx = (state.buffer_idx - delay_steps) % cfg.history_buffer_size
        # Advanced indexing: for each (i,j), get potential_history[history_idx[i,j], i, j]
        i_idx = jnp.arange(N)[:, None] * jnp.ones(N, dtype=int)[None, :]
        j_idx = jnp.ones(N, dtype=int)[:, None] * jnp.arange(N)[None, :]
        delayed_field = state.potential_history[history_idx, i_idx.astype(int), j_idx.astype(int)]

        # Downsample via average pooling
        factor = N // obs_N
        obs = delayed_field.reshape(obs_N, factor, obs_N, factor).mean(axis=(1, 3))

        # Normalize
        obs_mean = jnp.mean(obs)
        obs_std = jnp.std(obs) + 1e-8
        obs = (obs - obs_mean) / obs_std

        # Add ship info as extra channels (stack as multi-channel observation)
        # Channel 0: delayed potential field
        # Channel 1: ship position encoded as gaussian blob at ship location
        ship_grid_x = state.ship_pos[0] / (cfg.domain_size) * obs_N
        ship_grid_y = state.ship_pos[1] / (cfg.domain_size) * obs_N
        obs_x = jnp.arange(obs_N)
        obs_y = jnp.arange(obs_N)
        OX, OY = jnp.meshgrid(obs_x, obs_y, indexing='ij')
        ship_channel = jnp.exp(-((OX - ship_grid_x)**2 + (OY - ship_grid_y)**2) / (2.0 * 4.0))

        # Channel 2: target position (from body_is_target)
        target_idx = jnp.argmax(state.body_is_target)
        target_pos = state.body_pos[target_idx]
        target_gx = target_pos[0] / cfg.domain_size * obs_N
        target_gy = target_pos[1] / cfg.domain_size * obs_N
        target_channel = jnp.exp(-((OX - target_gx)**2 + (OY - target_gy)**2) / (2.0 * 4.0))

        # Stack: (3, obs_N, obs_N)
        observation = jnp.stack([obs, ship_channel, target_channel], axis=0)

        # Also include scalar observations
        speed = jnp.linalg.norm(state.ship_vel)
        fuel_frac = state.ship_fuel_mass / self.config.ship_fuel_mass
        scalar_obs = jnp.array([
            state.ship_heading / jnp.pi,          # normalized heading
            state.ship_angular_vel / 0.1,          # normalized angular velocity
            speed / 30000.0,                        # normalized speed
            fuel_frac,                              # fuel fraction
            state.prev_distance / cfg.domain_size,  # normalized distance to target
        ])

        return observation, scalar_obs

    @functools.partial(jax.jit, static_argnums=(0,))
    def reset(self, key: jnp.ndarray) -> tuple:
        """Reset environment with a new random level.

        Returns: (observation, state)
        """
        cfg = self.config

        # Generate level
        key, subkey = jax.random.split(key)
        body_pos, body_vel, body_mass, body_active, body_is_target, ship_pos, ship_vel = \
            self._generate_level(subkey)

        # Initial potential
        potential = self._solve_potential(body_pos, body_mass, body_active)

        # Initialize history buffer
        history = jnp.broadcast_to(
            potential[None, :, :],
            (cfg.history_buffer_size, cfg.grid_size, cfg.grid_size)
        ).copy()

        # Initial distance to target
        target_idx = jnp.argmax(body_is_target)
        init_distance = jnp.linalg.norm(ship_pos - body_pos[target_idx])

        state = EnvState(
            ship_pos=ship_pos,
            ship_vel=ship_vel,
            ship_heading=0.0,
            ship_angular_vel=0.0,
            ship_fuel_mass=cfg.ship_fuel_mass,
            body_pos=body_pos,
            body_vel=body_vel,
            body_mass=body_mass,
            body_active=body_active,
            body_is_target=body_is_target,
            potential_history=history,
            buffer_idx=0,
            step=0,
            prev_distance=init_distance,
            cumulative_reward=0.0,
            done=False,
        )

        obs_grid, obs_scalar = self._compute_observation(state)
        return (obs_grid, obs_scalar), state

    @functools.partial(jax.jit, static_argnums=(0,))
    def step(self, key: jnp.ndarray, state: EnvState, action: jnp.ndarray) -> tuple:
        """Execute one environment step.

        Args:
            key: PRNG key (unused currently but required by interface)
            state: Current environment state
            action: (2,) array [main_thrust (0-1), rotation_thrust (-1 to 1)]

        Returns:
            (observation, next_state, reward, done, info)
        """
        cfg = self.config

        # Compute potential and forces
        potential = self._solve_potential(state.body_pos, state.body_mass, state.body_active)
        grav_force = self._interpolate_force(potential, state.ship_pos)

        # Ship physics step
        main_thrust = jnp.clip(action[0], 0.0, 1.0)
        rot_thrust = jnp.clip(action[1], -1.0, 1.0)

        heading_vec = jnp.array([jnp.cos(state.ship_heading), jnp.sin(state.ship_heading)])
        total_mass = cfg.ship_dry_mass + state.ship_fuel_mass

        # Thrust force
        thrust_magnitude = main_thrust * cfg.ship_max_thrust
        thrust_force = thrust_magnitude * heading_vec

        # Fuel consumption (Tsiolkovsky)
        main_fuel_rate = thrust_magnitude / cfg.ship_exhaust_velocity
        rot_fuel_rate = jnp.abs(rot_thrust) * cfg.ship_max_torque / \
            (cfg.ship_moment_arm * cfg.ship_rotation_exhaust_velocity)
        total_fuel_rate = main_fuel_rate + rot_fuel_rate
        fuel_consumed = jnp.minimum(total_fuel_rate * cfg.dt, state.ship_fuel_mass)
        has_fuel = state.ship_fuel_mass > 0

        # Apply thrust only if fuel available
        effective_thrust = jnp.where(has_fuel, thrust_force, jnp.zeros(2))
        effective_torque = jnp.where(has_fuel, rot_thrust * cfg.ship_max_torque, 0.0)

        # Update ship state (symplectic Euler for simplicity in env step)
        new_angular_vel = state.ship_angular_vel + effective_torque / cfg.ship_moment_of_inertia * cfg.dt
        new_heading = state.ship_heading + new_angular_vel * cfg.dt
        new_heading = new_heading % (2.0 * jnp.pi)

        acceleration = (effective_thrust + grav_force * total_mass) / total_mass
        new_vel = state.ship_vel + acceleration * cfg.dt
        new_pos = state.ship_pos + new_vel * cfg.dt
        new_fuel = state.ship_fuel_mass - jnp.where(has_fuel, fuel_consumed, 0.0)
        new_fuel = jnp.maximum(new_fuel, 0.0)

        # Update bodies (leapfrog on potential field)
        def step_one_body(args):
            pos, vel, mass, active = args
            force = self._interpolate_force(potential, pos)
            new_v = vel + force * cfg.dt
            new_p = pos + new_v * cfg.dt
            # Wrap or clamp to domain
            new_p = jnp.clip(new_p, 0, cfg.domain_size)
            return jnp.where(active, new_p, pos), jnp.where(active, new_v, vel)

        new_body_pos, new_body_vel = jax.vmap(step_one_body)(
            (state.body_pos, state.body_vel, state.body_mass, state.body_active)
        )

        # Update potential history buffer
        new_history = state.potential_history.at[state.buffer_idx % cfg.history_buffer_size].set(potential)
        new_buffer_idx = (state.buffer_idx + 1) % cfg.history_buffer_size

        # Reward computation
        target_idx = jnp.argmax(state.body_is_target)
        target_pos = new_body_pos[target_idx]
        target_vel = new_body_vel[target_idx]
        distance = jnp.linalg.norm(new_pos - target_pos)
        rel_speed = jnp.linalg.norm(new_vel - target_vel)

        # Arrival
        reached = distance < cfg.approach_radius
        arrival_reward = jnp.where(reached, cfg.arrival_bonus, 0.0)

        # Proximity improvement
        dist_improvement = (state.prev_distance - distance) / cfg.domain_size
        proximity_reward = cfg.proximity_scale * dist_improvement

        # Velocity matching near target
        near_target = jnp.exp(-distance / (10.0 * cfg.approach_radius))
        vel_match = jnp.exp(-rel_speed / cfg.approach_velocity)
        velocity_reward = cfg.velocity_match_bonus * near_target * vel_match

        # Penalties
        time_reward = cfg.time_penalty
        fuel_reward = cfg.fuel_penalty * fuel_consumed

        reward = arrival_reward + proximity_reward + velocity_reward + time_reward + fuel_reward

        # Done conditions
        done = reached | (state.step + 1 >= cfg.max_steps) | (new_fuel <= 0.0)

        new_state = EnvState(
            ship_pos=new_pos,
            ship_vel=new_vel,
            ship_heading=new_heading,
            ship_angular_vel=new_angular_vel,
            ship_fuel_mass=new_fuel,
            body_pos=new_body_pos,
            body_vel=new_body_vel,
            body_mass=state.body_mass,
            body_active=state.body_active,
            body_is_target=state.body_is_target,
            potential_history=new_history,
            buffer_idx=new_buffer_idx,
            step=state.step + 1,
            prev_distance=distance,
            cumulative_reward=state.cumulative_reward + reward,
            done=done,
        )

        obs_grid, obs_scalar = self._compute_observation(new_state)

        info = {
            'distance': distance,
            'rel_speed': rel_speed,
            'reached_target': reached,
            'fuel_consumed': fuel_consumed,
            'reward_components': jnp.array([
                arrival_reward, proximity_reward, velocity_reward,
                time_reward, fuel_reward,
            ]),
        }

        return (obs_grid, obs_scalar), new_state, reward, done, info

    @functools.partial(jax.jit, static_argnums=(0,))
    def _generate_level(self, key):
        """Generate a random level. Returns body arrays and ship start."""
        cfg = self.config
        max_b = cfg.max_bodies

        key, k1, k2, k3, k4, k5, k6, k7 = jax.random.split(key, 8)

        # Number of bodies (sample, then mask)
        num_bodies = jax.random.randint(k1, (), cfg.num_bodies_min, cfg.num_bodies_max + 1)

        # Primary body (star) near center
        center = cfg.domain_size / 2.0
        primary_pos = jnp.array([center, center]) + jax.random.normal(k2, (2,)) * cfg.domain_size * 0.02
        primary_mass = jax.random.uniform(k3, ()) * 1.0e30 + 1.0e30  # 1-2 solar masses

        # Secondary bodies in orbits
        angles = jax.random.uniform(k4, (max_b,)) * 2 * jnp.pi
        radii = jax.random.uniform(k5, (max_b,)) * cfg.domain_size * 0.4 + cfg.domain_size * 0.05

        body_pos = jnp.stack([
            primary_pos[0] + radii * jnp.cos(angles),
            primary_pos[1] + radii * jnp.sin(angles),
        ], axis=-1)

        # Set first body as primary
        body_pos = body_pos.at[0].set(primary_pos)

        # Masses: primary, then log-uniform for secondaries
        log_masses = jax.random.uniform(k6, (max_b,)) * (28 - 24) + 24  # 1e24 to 1e28
        body_mass = jnp.power(10.0, log_masses)
        body_mass = body_mass.at[0].set(primary_mass)

        # Orbital velocities for secondaries
        r_from_primary = jnp.linalg.norm(body_pos - primary_pos, axis=-1)
        r_from_primary = jnp.maximum(r_from_primary, 1.0)  # avoid division by zero
        v_orbital = jnp.sqrt(cfg.G * primary_mass / r_from_primary)

        # Perpendicular to radial direction
        radial = body_pos - primary_pos
        perp = jnp.stack([-radial[:, 1], radial[:, 0]], axis=-1)
        perp_norm = jnp.linalg.norm(perp, axis=-1, keepdims=True)
        perp_norm = jnp.maximum(perp_norm, 1.0)
        perp = perp / perp_norm

        body_vel = perp * v_orbital[:, None]
        body_vel = body_vel.at[0].set(jnp.zeros(2))  # Primary stationary

        # Active mask
        body_active = jnp.arange(max_b) < num_bodies

        # Target: pick a secondary body (not the primary)
        target_idx = jax.random.randint(k7, (), 1, jnp.maximum(num_bodies, 2))
        body_is_target = jnp.arange(max_b) == target_idx

        # Body radii (proportional to log mass, for visualization)
        body_radius = jnp.log10(jnp.maximum(body_mass, 1.0)) * 1e8

        # Ship start: opposite side of domain from target
        target_pos = body_pos[target_idx]
        key, k8, k9 = jax.random.split(key, 3)
        ship_angle = jax.random.uniform(k8, ()) * 2 * jnp.pi
        ship_dist = cfg.domain_size * 0.3 + jax.random.uniform(k9, ()) * cfg.domain_size * 0.15
        ship_pos = target_pos + ship_dist * jnp.array([jnp.cos(ship_angle), jnp.sin(ship_angle)])
        ship_pos = jnp.clip(ship_pos, cfg.domain_size * 0.05, cfg.domain_size * 0.95)

        # Ship initial velocity: local orbital velocity around nearest massive body
        r_ship_primary = jnp.linalg.norm(ship_pos - primary_pos)
        v_ship = jnp.sqrt(cfg.G * primary_mass / jnp.maximum(r_ship_primary, 1.0))
        ship_radial = ship_pos - primary_pos
        ship_perp = jnp.array([-ship_radial[1], ship_radial[0]])
        ship_perp = ship_perp / jnp.maximum(jnp.linalg.norm(ship_perp), 1.0)
        ship_vel = ship_perp * v_ship

        return body_pos, body_vel, body_mass, body_active, body_is_target, ship_pos, ship_vel
