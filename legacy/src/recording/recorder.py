"""Episode recording and replay serialization.

Records simulation state at each timestep for replay visualization
and potential record-based (offline) RL training.
"""

from typing import NamedTuple
import jax
import jax.numpy as jnp
import numpy as np
import msgpack
import hashlib
import json
from pathlib import Path


class EpisodeFrame(NamedTuple):
    """Single frame of recorded episode data."""
    # Ship state
    ship_pos: jnp.ndarray        # (2,)
    ship_vel: jnp.ndarray        # (2,)
    ship_heading: float
    ship_angular_vel: float
    ship_fuel_mass: float
    ship_total_mass: float

    # Action taken
    action: jnp.ndarray          # (2,) [main_thrust, rotation_thrust]

    # Bodies state
    body_positions: jnp.ndarray  # (max_bodies, 2)
    body_masses: jnp.ndarray     # (max_bodies,)
    body_active: jnp.ndarray     # (max_bodies,)

    # Potential field (downsampled for storage)
    potential_field: jnp.ndarray  # (obs_grid_size, obs_grid_size)

    # Metrics
    reward: float
    cumulative_reward: float
    distance_to_target: float
    relative_velocity: float
    step: int


class EpisodeRecording(NamedTuple):
    """Full episode recording."""
    config_hash: str
    training_iteration: int
    episode_id: int
    seed: int

    # Stacked frames: each field is batched over time axis
    ship_pos: np.ndarray          # (T, 2)
    ship_vel: np.ndarray          # (T, 2)
    ship_heading: np.ndarray      # (T,)
    ship_angular_vel: np.ndarray  # (T,)
    ship_fuel_mass: np.ndarray    # (T,)
    ship_total_mass: np.ndarray   # (T,)
    actions: np.ndarray           # (T, 2)
    body_positions: np.ndarray    # (T, max_bodies, 2)
    body_masses: np.ndarray       # (max_bodies,) — constant
    body_active: np.ndarray       # (max_bodies,) — constant
    body_is_target: np.ndarray    # (max_bodies,) — constant
    potential_fields: np.ndarray  # (T, obs_grid_size, obs_grid_size)
    rewards: np.ndarray           # (T,)
    cumulative_rewards: np.ndarray  # (T,)
    distance_to_target: np.ndarray  # (T,)
    relative_velocity: np.ndarray   # (T,)
    total_reward: float
    num_steps: int
    reached_target: bool


def config_hash(config: dict) -> str:
    """Deterministic hash of simulation configuration."""
    serialized = json.dumps(config, sort_keys=True, default=str)
    return hashlib.sha256(serialized.encode()).hexdigest()[:12]


class EpisodeRecorder:
    """Accumulates frames during an episode and produces a recording."""

    def __init__(self, config_hash: str, training_iteration: int,
                 episode_id: int, seed: int, obs_grid_size: int = 64):
        self.config_hash = config_hash
        self.training_iteration = training_iteration
        self.episode_id = episode_id
        self.seed = seed
        self.obs_grid_size = obs_grid_size
        self.frames = []
        self._body_masses = None
        self._body_active = None
        self._body_is_target = None

    def record_frame(self, ship_state, action, body_state,
                     potential_field, reward, cumulative_reward,
                     distance_to_target, relative_velocity, step):
        """Record a single simulation frame."""
        # Downsample potential field for storage
        field = np.array(potential_field)
        if field.shape[0] != self.obs_grid_size:
            # Average pooling downsample
            factor = field.shape[0] // self.obs_grid_size
            field = field.reshape(
                self.obs_grid_size, factor,
                self.obs_grid_size, factor
            ).mean(axis=(1, 3))

        self.frames.append({
            'ship_pos': np.array(ship_state.pos),
            'ship_vel': np.array(ship_state.vel),
            'ship_heading': float(ship_state.heading),
            'ship_angular_vel': float(ship_state.angular_vel),
            'ship_fuel_mass': float(ship_state.fuel_mass),
            'ship_total_mass': float(ship_state.dry_mass + ship_state.fuel_mass),
            'action': np.array(action),
            'body_positions': np.array(body_state.pos),
            'reward': float(reward),
            'cumulative_reward': float(cumulative_reward),
            'distance_to_target': float(distance_to_target),
            'relative_velocity': float(relative_velocity),
            'potential_field': field,
            'step': int(step),
        })

        if self._body_masses is None:
            self._body_masses = np.array(body_state.mass)
            self._body_active = np.array(body_state.active)
            self._body_is_target = np.array(body_state.is_target)

    def finalize(self, reached_target: bool) -> EpisodeRecording:
        """Produce a finalized recording from accumulated frames."""
        T = len(self.frames)
        return EpisodeRecording(
            config_hash=self.config_hash,
            training_iteration=self.training_iteration,
            episode_id=self.episode_id,
            seed=self.seed,
            ship_pos=np.stack([f['ship_pos'] for f in self.frames]),
            ship_vel=np.stack([f['ship_vel'] for f in self.frames]),
            ship_heading=np.array([f['ship_heading'] for f in self.frames]),
            ship_angular_vel=np.array([f['ship_angular_vel'] for f in self.frames]),
            ship_fuel_mass=np.array([f['ship_fuel_mass'] for f in self.frames]),
            ship_total_mass=np.array([f['ship_total_mass'] for f in self.frames]),
            actions=np.stack([f['action'] for f in self.frames]),
            body_positions=np.stack([f['body_positions'] for f in self.frames]),
            body_masses=self._body_masses,
            body_active=self._body_active,
            body_is_target=self._body_is_target,
            potential_fields=np.stack([f['potential_field'] for f in self.frames]),
            rewards=np.array([f['reward'] for f in self.frames]),
            cumulative_rewards=np.array([f['cumulative_reward'] for f in self.frames]),
            distance_to_target=np.array([f['distance_to_target'] for f in self.frames]),
            relative_velocity=np.array([f['relative_velocity'] for f in self.frames]),
            total_reward=float(sum(f['reward'] for f in self.frames)),
            num_steps=T,
            reached_target=reached_target,
        )


def save_recording(recording: EpisodeRecording, base_dir: str = "data"):
    """Save episode recording to disk as compressed numpy archive."""
    path = Path(base_dir) / recording.config_hash / f"iter_{recording.training_iteration:06d}"
    path.mkdir(parents=True, exist_ok=True)
    filepath = path / f"episode_{recording.episode_id:06d}.npz"

    np.savez_compressed(
        filepath,
        # Metadata
        config_hash=recording.config_hash,
        training_iteration=recording.training_iteration,
        episode_id=recording.episode_id,
        seed=recording.seed,
        total_reward=recording.total_reward,
        num_steps=recording.num_steps,
        reached_target=recording.reached_target,
        # Trajectory
        ship_pos=recording.ship_pos,
        ship_vel=recording.ship_vel,
        ship_heading=recording.ship_heading,
        ship_angular_vel=recording.ship_angular_vel,
        ship_fuel_mass=recording.ship_fuel_mass,
        ship_total_mass=recording.ship_total_mass,
        actions=recording.actions,
        body_positions=recording.body_positions,
        body_masses=recording.body_masses,
        body_active=recording.body_active,
        body_is_target=recording.body_is_target,
        potential_fields=recording.potential_fields,
        rewards=recording.rewards,
        cumulative_rewards=recording.cumulative_rewards,
        distance_to_target=recording.distance_to_target,
        relative_velocity=recording.relative_velocity,
    )
    return str(filepath)


def load_recording(filepath: str) -> dict:
    """Load episode recording from disk. Returns dict for JSON/msgpack serialization."""
    data = np.load(filepath, allow_pickle=False)
    return {key: data[key].tolist() if data[key].ndim == 0 else data[key].tolist()
            for key in data.files}


def list_configs(base_dir: str = "data") -> list[dict]:
    """List all configuration hashes with their training run info."""
    base = Path(base_dir)
    if not base.exists():
        return []

    configs = []
    for config_dir in sorted(base.iterdir()):
        if not config_dir.is_dir():
            continue
        iterations = []
        for iter_dir in sorted(config_dir.iterdir()):
            if not iter_dir.is_dir() or not iter_dir.name.startswith("iter_"):
                continue
            episodes = list(iter_dir.glob("episode_*.npz"))
            iterations.append({
                "iteration": int(iter_dir.name.split("_")[1]),
                "num_episodes": len(episodes),
            })
        configs.append({
            "config_hash": config_dir.name,
            "iterations": iterations,
            "total_episodes": sum(i["num_episodes"] for i in iterations),
        })
    return configs
