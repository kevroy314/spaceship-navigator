"""Training entry point.

Usage:
    python -m training.train --config configs/default.yaml
"""

import argparse
import time
from pathlib import Path

import jax
import jax.numpy as jnp
import yaml
import numpy as np

from simulation.environment import SpaceshipEnv, EnvConfig
from training.ppo import PPOConfig, make_train, create_train_state, ActorCritic
from recording.recorder import config_hash


def load_env_config(config_path: str) -> EnvConfig:
    """Load environment config from YAML."""
    with open(config_path) as f:
        cfg = yaml.safe_load(f)

    sim = cfg["simulation"]
    ship = sim["ship"]
    target = sim["target"]
    obs = cfg["observation"]
    reward = cfg["reward"]
    level = cfg["level_gen"]

    return EnvConfig(
        grid_size=sim["grid_size"],
        domain_size=sim["domain_size"] * sim["AU"],
        dt=sim["dt"],
        max_steps=sim["max_steps"],
        G=sim["G"],
        c=sim["c"],
        AU=sim["AU"],
        ship_dry_mass=ship["dry_mass"],
        ship_fuel_mass=ship["fuel_mass"],
        ship_max_thrust=ship["max_thrust"],
        ship_exhaust_velocity=ship["exhaust_velocity"],
        ship_max_torque=ship["max_torque"],
        ship_moment_arm=ship["moment_arm"],
        ship_rotation_exhaust_velocity=ship["rotation_exhaust_velocity"],
        ship_moment_of_inertia=ship["moment_of_inertia"],
        approach_radius=target["approach_radius"] * sim["AU"],
        approach_velocity=target["approach_velocity"],
        obs_grid_size=obs["obs_grid_size"],
        history_buffer_size=obs["history_buffer_size"],
        arrival_bonus=reward["arrival_bonus"],
        proximity_scale=reward["proximity_scale"],
        time_penalty=reward["time_penalty"],
        fuel_penalty=reward["fuel_penalty"],
        velocity_match_bonus=reward["velocity_match_bonus"],
        num_bodies_min=level["num_bodies_range"][0],
        num_bodies_max=level["num_bodies_range"][1],
    )


def load_ppo_config(config_path: str) -> PPOConfig:
    """Load PPO config from YAML."""
    with open(config_path) as f:
        cfg = yaml.safe_load(f)["training"]

    return PPOConfig(
        num_envs=cfg["num_envs"],
        num_steps=cfg["num_steps"],
        num_epochs=cfg["num_epochs"],
        num_minibatches=cfg["num_minibatches"],
        learning_rate=cfg["learning_rate"],
        gamma=cfg["gamma"],
        gae_lambda=cfg["gae_lambda"],
        clip_eps=cfg["clip_eps"],
        entropy_coeff=cfg["entropy_coeff"],
        value_coeff=cfg["value_coeff"],
        max_grad_norm=cfg["max_grad_norm"],
        total_timesteps=cfg["total_timesteps"],
    )


def main():
    parser = argparse.ArgumentParser(description="Train spaceship navigator agent")
    parser.add_argument("--config", type=str, default="configs/default.yaml")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--checkpoint-dir", type=str, default="data")
    args = parser.parse_args()

    print(f"Loading config from {args.config}")
    env_config = load_env_config(args.config)
    ppo_config = load_ppo_config(args.config)

    with open(args.config) as f:
        full_config = yaml.safe_load(f)
    cfg_hash = config_hash(full_config)
    print(f"Config hash: {cfg_hash}")

    # Create output directory
    save_dir = Path(args.checkpoint_dir) / cfg_hash
    save_dir.mkdir(parents=True, exist_ok=True)

    # Save config alongside data
    with open(save_dir / "config.yaml", "w") as f:
        yaml.dump(full_config, f)

    # Initialize
    print(f"Initializing environment (grid={env_config.grid_size}x{env_config.grid_size}, "
          f"envs={ppo_config.num_envs})")
    env = SpaceshipEnv(env_config)

    key = jax.random.PRNGKey(args.seed)
    train_state, tx, network = create_train_state(key, env, ppo_config)

    # Create training function
    train_step = make_train(env, ppo_config)

    # Training loop
    num_updates = ppo_config.total_timesteps // (ppo_config.num_steps * ppo_config.num_envs)
    print(f"Training for {num_updates} updates "
          f"({ppo_config.total_timesteps:,} total timesteps)")
    print(f"  Batch size: {ppo_config.num_steps * ppo_config.num_envs}")
    print(f"  Minibatch size: {ppo_config.num_steps * ppo_config.num_envs // ppo_config.num_minibatches}")

    # Compile
    print("JIT compiling training step (this may take a minute)...")
    t0 = time.time()
    train_state, metrics = train_step(train_state)
    jax.block_until_ready(metrics)
    compile_time = time.time() - t0
    print(f"Compilation done in {compile_time:.1f}s")
    print(f"  Initial metrics: reward={metrics['mean_reward']:.4f}, "
          f"policy_loss={metrics['policy_loss']:.4f}")

    # Main training loop
    best_reward = -float('inf')
    for update in range(1, num_updates):
        t0 = time.time()
        train_state, metrics = train_step(train_state)
        jax.block_until_ready(metrics)
        step_time = time.time() - t0

        mean_reward = float(metrics['mean_reward'])
        sps = ppo_config.num_steps * ppo_config.num_envs / step_time

        if update % 10 == 0:
            print(f"Update {update}/{num_updates} | "
                  f"reward={mean_reward:.4f} | "
                  f"return={float(metrics['mean_return']):.4f} | "
                  f"policy_loss={float(metrics['policy_loss']):.4f} | "
                  f"entropy={float(metrics['entropy']):.4f} | "
                  f"SPS={sps:.0f} | "
                  f"time={step_time:.3f}s")

        # Save checkpoint
        save_interval = full_config["training"].get("save_interval", 100)
        if update % save_interval == 0 or mean_reward > best_reward:
            if mean_reward > best_reward:
                best_reward = mean_reward
                tag = "best"
            else:
                tag = f"step_{update}"

            ckpt_path = save_dir / f"checkpoint_{tag}.npz"
            flat_params = jax.tree.leaves(train_state.params)
            np.savez(
                ckpt_path,
                *[np.array(p) for p in flat_params],
                update_step=update,
                mean_reward=mean_reward,
            )
            print(f"  Saved checkpoint: {ckpt_path}")

    print(f"\nTraining complete. Best reward: {best_reward:.4f}")
    print(f"Checkpoints saved to: {save_dir}")


if __name__ == "__main__":
    main()
