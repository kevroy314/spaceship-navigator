"""FastAPI backend for spaceship navigator.

Provides:
- REST API for config/replay data browsing
- WebSocket for live play mode
"""

import asyncio
import json
import time
from pathlib import Path
from typing import Optional

import numpy as np
import uvicorn
import yaml
from fastapi import FastAPI, WebSocket, WebSocketDisconnect, Query
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles
from fastapi.responses import JSONResponse

from recording.recorder import list_configs, load_recording, save_recording, config_hash

app = FastAPI(title="Spaceship Navigator", version="0.1.0")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

DATA_DIR = Path("data")
CONFIG_DIR = Path("configs")


# ---------------------------------------------------------------------------
# REST API — Config & Replay browsing
# ---------------------------------------------------------------------------

@app.get("/api/configs")
async def get_configs():
    """List all training configurations."""
    configs = list_configs(str(DATA_DIR))
    # Also load config YAML metadata if available
    for cfg in configs:
        meta_path = DATA_DIR / cfg["config_hash"] / "config.yaml"
        if meta_path.exists():
            with open(meta_path) as f:
                cfg["metadata"] = yaml.safe_load(f)
    return configs


@app.get("/api/configs/{config_id}/iterations")
async def get_iterations(config_id: str):
    """List training iterations for a config."""
    config_dir = DATA_DIR / config_id
    if not config_dir.exists():
        return JSONResponse({"error": "Config not found"}, status_code=404)

    iterations = []
    for iter_dir in sorted(config_dir.iterdir()):
        if not iter_dir.is_dir() or not iter_dir.name.startswith("iter_"):
            continue
        episodes = list(iter_dir.glob("episode_*.npz"))
        # Load summary stats from first/last episodes
        rewards = []
        for ep in episodes:
            data = np.load(ep, allow_pickle=False)
            rewards.append(float(data["total_reward"]))
        iterations.append({
            "iteration": int(iter_dir.name.split("_")[1]),
            "num_episodes": len(episodes),
            "mean_reward": float(np.mean(rewards)) if rewards else 0,
            "max_reward": float(np.max(rewards)) if rewards else 0,
        })
    return iterations


@app.get("/api/configs/{config_id}/iterations/{iteration}/episodes")
async def get_episodes(config_id: str, iteration: int):
    """List episodes for a specific training iteration."""
    iter_dir = DATA_DIR / config_id / f"iter_{iteration:06d}"
    if not iter_dir.exists():
        return JSONResponse({"error": "Iteration not found"}, status_code=404)

    episodes = []
    for ep_file in sorted(iter_dir.glob("episode_*.npz")):
        data = np.load(ep_file, allow_pickle=False)
        episodes.append({
            "episode_id": int(data["episode_id"]),
            "total_reward": float(data["total_reward"]),
            "num_steps": int(data["num_steps"]),
            "reached_target": bool(data["reached_target"]),
        })
    return episodes


@app.get("/api/configs/{config_id}/iterations/{iteration}/episodes/{episode_id}")
async def get_episode(config_id: str, iteration: int, episode_id: int):
    """Get full episode replay data."""
    filepath = DATA_DIR / config_id / f"iter_{iteration:06d}" / f"episode_{episode_id:06d}.npz"
    if not filepath.exists():
        return JSONResponse({"error": "Episode not found"}, status_code=404)

    data = load_recording(str(filepath))
    return data


@app.get("/api/configs/{config_id}/learning_curve")
async def get_learning_curve(config_id: str):
    """Get reward over training iterations for plotting."""
    config_dir = DATA_DIR / config_id
    if not config_dir.exists():
        return JSONResponse({"error": "Config not found"}, status_code=404)

    curve = []
    for iter_dir in sorted(config_dir.iterdir()):
        if not iter_dir.is_dir() or not iter_dir.name.startswith("iter_"):
            continue
        rewards = []
        for ep_file in iter_dir.glob("episode_*.npz"):
            data = np.load(ep_file, allow_pickle=False)
            rewards.append(float(data["total_reward"]))
        if rewards:
            curve.append({
                "iteration": int(iter_dir.name.split("_")[1]),
                "mean_reward": float(np.mean(rewards)),
                "std_reward": float(np.std(rewards)),
                "max_reward": float(np.max(rewards)),
                "min_reward": float(np.min(rewards)),
            })
    return curve


# ---------------------------------------------------------------------------
# WebSocket — Live Play Mode
# ---------------------------------------------------------------------------

class GameSession:
    """Manages a live play session."""

    def __init__(self, config_path: str = "configs/default.yaml"):
        self.config = self._load_config(config_path)
        self.sim = None
        self.recorder = None
        self.step_count = 0
        self.running = False

    def _load_config(self, path):
        with open(path) as f:
            return yaml.safe_load(f)

    async def initialize(self):
        """Initialize simulation — lazy import to avoid loading JAX for REST-only."""
        from simulation.gravity import (
            create_solver, cic_deposit, solve_potential,
            gradient_potential, cic_interpolate,
            direct_nbody_accelerations, direct_acceleration_at,
        )
        from simulation.ship import ShipState, ShipConfig, make_ship_step
        from simulation.bodies import bodies_step, get_target_state
        from simulation.level_gen import generate_level, LevelConfig
        from observation.light_delay import init_observer, update_observer, observe
        from recording.recorder import EpisodeRecorder, config_hash
        import jax
        import jax.numpy as jnp

        self._jax = jax
        self._jnp = jnp
        self._bodies_step = bodies_step
        self._get_target_state = get_target_state
        self._update_observer = update_observer
        self._observe = observe
        self._cic_deposit = cic_deposit
        self._solve_potential = solve_potential
        self._gradient_potential = gradient_potential
        self._cic_interpolate = cic_interpolate
        self._direct_nbody_acc = direct_nbody_accelerations
        self._direct_acc_at = direct_acceleration_at

        sim_cfg = self.config["simulation"]
        level_cfg = self.config["level_gen"]
        obs_cfg = self.config["observation"]
        domain_m = sim_cfg["domain_size"] * sim_cfg["AU"]

        # Initialize solver (precompute spectral arrays)
        self.solver_cfg, self.solver_state = create_solver(
            n=sim_cfg["grid_size"],
            domain_size=domain_m,
        )
        self.grid_config = {
            "origin": jnp.zeros(2),
            "cell_size": domain_m / sim_cfg["grid_size"],
        }

        # Generate level
        key = jax.random.PRNGKey(level_cfg["seed"])
        lc = LevelConfig(
            seed=level_cfg["seed"],
            num_planets_range=tuple(level_cfg["num_planets_range"]),
            primary_mass_range=tuple(level_cfg["primary_mass_range"]),
            terrestrial_mass_range=tuple(level_cfg["terrestrial_mass_range"]),
            giant_mass_range=tuple(level_cfg["giant_mass_range"]),
            giant_fraction=level_cfg["giant_fraction"],
            rogue_mass_range=tuple(level_cfg["rogue_mass_range"]),
            num_rogues_range=tuple(level_cfg["num_rogues_range"]),
            inner_orbit_au=level_cfg["inner_orbit_au"],
            orbit_spacing_range=tuple(level_cfg["orbit_spacing_range"]),
            eccentricity_sigma=level_cfg["eccentricity_sigma"],
            max_eccentricity=level_cfg["max_eccentricity"],
            min_start_target_distance=level_cfg["min_start_target_distance"],
            domain_size=sim_cfg["domain_size"],
            softening=level_cfg.get("softening", 1.0e9),
        )
        self.bodies, ship_pos, ship_vel = generate_level(lc, key)

        # Initialize ship
        sc = sim_cfg["ship"]
        self.ship_config = ShipConfig(
            dry_mass=sc["dry_mass"],
            fuel_mass=sc["fuel_mass"],
            max_thrust=sc["max_thrust"],
            exhaust_velocity=sc["exhaust_velocity"],
            max_torque=sc["max_torque"],
            moment_arm=sc["moment_arm"],
            rotation_exhaust_velocity=sc["rotation_exhaust_velocity"],
            moment_of_inertia=sc["moment_of_inertia"],
            radius=sc["radius"],
        )
        self._ship_step = make_ship_step(self.ship_config)
        self.ship = ShipState(
            pos=jnp.array(ship_pos),
            vel=jnp.array(ship_vel),
            heading=0.0,
            angular_vel=0.0,
            fuel_mass=sc["fuel_mass"],
            dry_mass=sc["dry_mass"],
        )

        # Initialize observer
        self.observer = init_observer(
            buffer_size=obs_cfg["history_buffer_size"],
            grid_size=sim_cfg["grid_size"],
            obs_grid_size=obs_cfg["obs_grid_size"],
            domain_size=domain_m,
            c=sim_cfg["c"],
            dt=sim_cfg["dt"],
        )

        # Config hash for recording
        self.cfg_hash = config_hash(self.config)
        self.recorder = EpisodeRecorder(
            config_hash=self.cfg_hash,
            training_iteration=0,  # play mode = iteration 0
            episode_id=int(time.time()),
            seed=level_cfg["seed"],
            obs_grid_size=obs_cfg["obs_grid_size"],
        )

        self.dt = sim_cfg["dt"]
        self.domain_size = domain_m
        self.step_count = 0
        self.cumulative_reward = 0.0
        self.running = True

    def get_state_dict(self):
        """Serialize current state for WebSocket transmission."""
        jnp = self._jnp
        target_pos, target_vel = self._get_target_state(self.bodies)
        distance = float(jnp.linalg.norm(self.ship.pos - target_pos))
        rel_vel = float(jnp.linalg.norm(self.ship.vel - target_vel))

        n = self.solver_cfg.n
        dx = self.solver_cfg.dx

        # Compute potential field for visualization
        density = self._cic_deposit(self.bodies.pos, self.bodies.mass, n, dx)
        potential = self._solve_potential(density, self.solver_cfg, self.solver_state)

        # Downsample potential for transmission
        vis_size = 128
        factor = n // vis_size
        potential_ds = potential.reshape(vis_size, factor, vis_size, factor).mean(axis=(1, 3))

        return {
            "step": self.step_count,
            "ship": {
                "pos": [float(self.ship.pos[0]), float(self.ship.pos[1])],
                "vel": [float(self.ship.vel[0]), float(self.ship.vel[1])],
                "heading": float(self.ship.heading),
                "angular_vel": float(self.ship.angular_vel),
                "fuel_mass": float(self.ship.fuel_mass),
                "total_mass": float(self.ship.dry_mass + self.ship.fuel_mass),
                "dry_mass": float(self.ship.dry_mass),
            },
            "bodies": {
                "positions": [[float(p[0]), float(p[1])]
                              for p, a in zip(self.bodies.pos, self.bodies.active) if a],
                "masses": [float(m) for m, a in zip(self.bodies.mass, self.bodies.active) if a],
                "is_target": [bool(t) for t, a in zip(self.bodies.is_target, self.bodies.active) if a],
            },
            "metrics": {
                "distance_to_target": distance,
                "relative_velocity": rel_vel,
                "cumulative_reward": self.cumulative_reward,
                "fuel_fraction": float(self.ship.fuel_mass / self.ship_config.fuel_mass),
            },
            "potential_field": potential_ds.tolist(),
            "domain_size": float(self.domain_size),
        }

    async def step(self, action):
        """Execute one simulation step with player action."""
        jnp = self._jnp
        from training.reward import compute_reward, RewardConfig

        n = self.solver_cfg.n
        dx = self.solver_cfg.dx

        # Compute potential field for observation/visualization (Poisson solver)
        density = self._cic_deposit(self.bodies.pos, self.bodies.mass, n, dx)
        potential = self._solve_potential(density, self.solver_cfg, self.solver_state)

        # Update observer with potential field
        self.observer = self._update_observer(self.observer, potential)

        # Compute actual forces via direct N-body (3D 1/r² law)
        # This gives physically correct Keplerian orbits in the 2D plane
        body_acc = self._direct_nbody_acc(
            self.bodies.pos, self.bodies.mass, self.bodies.active
        )
        ship_acc = self._direct_acc_at(
            self.ship.pos, self.bodies.pos, self.bodies.mass, self.bodies.active
        )

        prev_fuel = self.ship.fuel_mass

        # Step ship — ship_step expects force (N), not acceleration
        action_array = jnp.array([action.get("main_thrust", 0.0),
                                   action.get("rotation_thrust", 0.0)])
        total_mass = self.ship.dry_mass + self.ship.fuel_mass
        gravity_force = ship_acc * total_mass
        self.ship = self._ship_step(self.ship, action_array, gravity_force, self.dt)

        # Step bodies using direct N-body accelerations (leapfrog)
        new_vel = self.bodies.vel + body_acc * self.dt
        new_pos = self.bodies.pos + new_vel * self.dt
        # Zero out inactive
        mask = self.bodies.active[:, None].astype(new_pos.dtype)
        self.bodies = self.bodies._replace(
            pos=new_pos * mask,
            vel=new_vel * mask,
        )

        # Compute reward
        target_pos, target_vel = self._get_target_state(self.bodies)
        prev_distance = float(jnp.linalg.norm(self.ship.pos - target_pos))
        fuel_consumed = prev_fuel - self.ship.fuel_mass

        reward_cfg = RewardConfig(
            approach_radius=self.config["simulation"]["target"]["approach_radius"]
                            * self.config["simulation"]["AU"],
            approach_velocity=self.config["simulation"]["target"]["approach_velocity"],
        )
        reward, info = compute_reward(
            self.ship.pos, self.ship.vel, target_pos, target_vel,
            fuel_consumed, prev_distance, reward_cfg,
        )

        self.cumulative_reward += float(reward)
        self.step_count += 1

        # Record frame
        if self.recorder:
            self.recorder.record_frame(
                ship_state=self.ship,
                action=action_array,
                body_state=self.bodies,
                potential_field=potential,
                reward=float(reward),
                cumulative_reward=self.cumulative_reward,
                distance_to_target=float(info['distance']),
                relative_velocity=float(info['rel_speed']),
                step=self.step_count,
            )

        return float(reward), info


@app.websocket("/ws/play")
async def play_websocket(websocket: WebSocket):
    """WebSocket endpoint for live play mode."""
    await websocket.accept()
    session = GameSession()

    try:
        # Initialize simulation (may take a moment for JAX compilation)
        await websocket.send_json({"type": "status", "message": "Initializing simulation..."})
        await session.initialize()
        await websocket.send_json({"type": "status", "message": "Ready"})

        # Send initial state
        state = session.get_state_dict()
        await websocket.send_json({"type": "state", "data": state})

        while session.running:
            try:
                # Wait for player action (with timeout for periodic state updates)
                data = await asyncio.wait_for(websocket.receive_json(), timeout=0.1)

                if data.get("type") == "action":
                    reward, info = await session.step(data.get("action", {}))
                    state = session.get_state_dict()
                    state["reward"] = reward
                    state["reward_info"] = {k: float(v) for k, v in info.items()}
                    await websocket.send_json({"type": "state", "data": state})

                elif data.get("type") == "reset":
                    await session.initialize()
                    state = session.get_state_dict()
                    await websocket.send_json({"type": "state", "data": state})

                elif data.get("type") == "save":
                    if session.recorder:
                        recording = session.recorder.finalize(reached_target=False)
                        filepath = save_recording(recording)
                        await websocket.send_json({
                            "type": "saved",
                            "filepath": filepath,
                            "config_hash": session.cfg_hash,
                        })

            except asyncio.TimeoutError:
                # No action received — still send state update for body movement
                pass

    except WebSocketDisconnect:
        # Save recording on disconnect
        if session.recorder and session.step_count > 0:
            recording = session.recorder.finalize(reached_target=False)
            save_recording(recording)


# ---------------------------------------------------------------------------
# Static file serving (frontend)
# ---------------------------------------------------------------------------

frontend_path = Path(__file__).parent.parent.parent / "frontend" / "dist"
if frontend_path.exists():
    app.mount("/", StaticFiles(directory=str(frontend_path), html=True))


def main():
    """Entry point for `serve` command."""
    import argparse
    parser = argparse.ArgumentParser(description="Spaceship Navigator server")
    parser.add_argument("--port", type=int, default=8000, help="Port to listen on")
    parser.add_argument("--host", type=str, default="0.0.0.0", help="Host to bind to")
    args = parser.parse_args()

    uvicorn.run(
        "server.app:app",
        host=args.host,
        port=args.port,
        reload=True,
        reload_dirs=["src"],
    )


if __name__ == "__main__":
    main()
