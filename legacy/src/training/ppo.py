"""PPO training loop — PureJaxRL style.

Fully JIT-compiled PPO with vmap over parallel environments.
Everything stays on GPU: env stepping, rollout collection, gradient updates.
"""

import jax
import jax.numpy as jnp
import flax.linen as nn
import optax
from typing import NamedTuple
import functools


# ---------------------------------------------------------------------------
# Actor-Critic Network
# ---------------------------------------------------------------------------

class ActorCritic(nn.Module):
    """CNN + MLP actor-critic for spaceship navigation.

    Processes:
    - Grid observation (3, obs_N, obs_N): delayed potential + ship/target channels
    - Scalar observation (5,): heading, angular_vel, speed, fuel_frac, distance
    """
    action_dim: int = 2  # main_thrust, rotation_thrust

    @nn.compact
    def __call__(self, obs_grid, obs_scalar):
        # CNN for grid observation
        # Input: (3, H, W) -> transpose to (H, W, 3) for Conv
        x = jnp.transpose(obs_grid, (1, 2, 0))
        x = nn.Conv(features=32, kernel_size=(5, 5), strides=(2, 2))(x)
        x = nn.relu(x)
        x = nn.Conv(features=64, kernel_size=(3, 3), strides=(2, 2))(x)
        x = nn.relu(x)
        x = nn.Conv(features=64, kernel_size=(3, 3), strides=(2, 2))(x)
        x = nn.relu(x)
        x = x.reshape(-1)  # Flatten

        # Combine with scalar observations
        x = jnp.concatenate([x, obs_scalar])
        x = nn.Dense(256)(x)
        x = nn.relu(x)
        x = nn.Dense(128)(x)
        x = nn.relu(x)

        # Actor head: mean and log_std for Gaussian policy
        action_mean = nn.Dense(self.action_dim)(x)
        action_log_std = self.param(
            'action_log_std',
            nn.initializers.zeros,
            (self.action_dim,)
        )

        # Critic head
        value = nn.Dense(1)(x)

        return action_mean, action_log_std, value.squeeze(-1)


# ---------------------------------------------------------------------------
# PPO Data Structures
# ---------------------------------------------------------------------------

class Transition(NamedTuple):
    """Single transition for PPO rollout."""
    obs_grid: jnp.ndarray
    obs_scalar: jnp.ndarray
    action: jnp.ndarray
    log_prob: jnp.ndarray
    value: jnp.ndarray
    reward: jnp.ndarray
    done: jnp.ndarray


class PPOConfig(NamedTuple):
    """PPO hyperparameters."""
    num_envs: int = 32
    num_steps: int = 256
    num_epochs: int = 4
    num_minibatches: int = 8
    learning_rate: float = 3e-4
    gamma: float = 0.999
    gae_lambda: float = 0.95
    clip_eps: float = 0.2
    entropy_coeff: float = 0.01
    value_coeff: float = 0.5
    max_grad_norm: float = 0.5
    total_timesteps: int = 10_000_000
    anneal_lr: bool = True


# ---------------------------------------------------------------------------
# PPO Functions
# ---------------------------------------------------------------------------

def sample_action(key, action_mean, action_log_std):
    """Sample from Gaussian policy."""
    std = jnp.exp(action_log_std)
    noise = jax.random.normal(key, action_mean.shape)
    action = action_mean + noise * std

    # Log probability of Gaussian
    log_prob = -0.5 * (
        ((action - action_mean) / std) ** 2
        + 2 * action_log_std
        + jnp.log(2 * jnp.pi)
    )
    log_prob = jnp.sum(log_prob, axis=-1)

    return action, log_prob


def compute_log_prob(action, action_mean, action_log_std):
    """Compute log probability of action under Gaussian policy."""
    std = jnp.exp(action_log_std)
    log_prob = -0.5 * (
        ((action - action_mean) / std) ** 2
        + 2 * action_log_std
        + jnp.log(2 * jnp.pi)
    )
    return jnp.sum(log_prob, axis=-1)


def compute_gae(rewards, values, dones, gamma, gae_lambda):
    """Compute Generalized Advantage Estimation.

    Args:
        rewards: (T, N) rewards
        values: (T+1, N) value estimates (includes bootstrap)
        dones: (T, N) done flags
        gamma: discount factor
        gae_lambda: GAE lambda

    Returns:
        advantages: (T, N)
        returns: (T, N)
    """
    T = rewards.shape[0]

    def _scan_fn(carry, t):
        gae = carry
        delta = rewards[t] + gamma * values[t + 1] * (1.0 - dones[t]) - values[t]
        gae = delta + gamma * gae_lambda * (1.0 - dones[t]) * gae
        return gae, gae

    _, advantages = jax.lax.scan(
        _scan_fn,
        jnp.zeros_like(rewards[0]),
        jnp.arange(T - 1, -1, -1),  # reverse scan
    )
    advantages = jnp.flip(advantages, axis=0)
    returns = advantages + values[:-1]

    return advantages, returns


def ppo_loss(params, apply_fn, batch, clip_eps, entropy_coeff, value_coeff):
    """Compute PPO clipped objective + value loss + entropy bonus."""
    obs_grid, obs_scalar, actions, old_log_probs, advantages, returns = batch

    # Forward pass
    action_mean, action_log_std, values = apply_fn(params, obs_grid, obs_scalar)
    log_probs = compute_log_prob(actions, action_mean, action_log_std)

    # Policy loss (clipped)
    ratio = jnp.exp(log_probs - old_log_probs)
    advantages_norm = (advantages - jnp.mean(advantages)) / (jnp.std(advantages) + 1e-8)
    pg_loss1 = ratio * advantages_norm
    pg_loss2 = jnp.clip(ratio, 1.0 - clip_eps, 1.0 + clip_eps) * advantages_norm
    policy_loss = -jnp.mean(jnp.minimum(pg_loss1, pg_loss2))

    # Value loss
    value_loss = jnp.mean((values - returns) ** 2)

    # Entropy bonus
    std = jnp.exp(action_log_std)
    entropy = 0.5 * jnp.sum(jnp.log(2 * jnp.pi * jnp.e * std ** 2))

    total_loss = policy_loss + value_coeff * value_loss - entropy_coeff * entropy

    return total_loss, {
        'policy_loss': policy_loss,
        'value_loss': value_loss,
        'entropy': entropy,
        'approx_kl': jnp.mean((ratio - 1) - jnp.log(ratio)),
    }


# ---------------------------------------------------------------------------
# Training Loop
# ---------------------------------------------------------------------------

class TrainState(NamedTuple):
    """Mutable training state."""
    params: dict
    opt_state: optax.OptState
    key: jnp.ndarray
    env_states: object  # vmapped EnvState
    update_step: int


def create_train_state(key, env, ppo_config):
    """Initialize training state."""
    key, init_key, env_key = jax.random.split(key, 3)

    # Initialize network
    network = ActorCritic()
    dummy_grid = jnp.zeros((3, env.config.obs_grid_size, env.config.obs_grid_size))
    dummy_scalar = jnp.zeros(5)
    params = network.init(init_key, dummy_grid, dummy_scalar)

    # Optimizer
    tx = optax.chain(
        optax.clip_by_global_norm(ppo_config.max_grad_norm),
        optax.adam(ppo_config.learning_rate),
    )
    opt_state = tx.init(params)

    # Initialize environments (vmapped)
    env_keys = jax.random.split(env_key, ppo_config.num_envs)
    _, env_states = jax.vmap(env.reset)(env_keys)

    return TrainState(
        params=params,
        opt_state=opt_state,
        key=key,
        env_states=env_states,
        update_step=0,
    ), tx, network


def make_train(env, ppo_config):
    """Create the full training function (JIT-compiled)."""
    network = ActorCritic()
    tx = optax.chain(
        optax.clip_by_global_norm(ppo_config.max_grad_norm),
        optax.adam(ppo_config.learning_rate),
    )

    @jax.jit
    def _train_step(train_state):
        """Single PPO update: collect rollout -> compute GAE -> update params."""
        params, opt_state, key, env_states, update_step = train_state

        # --- Rollout collection ---
        def _env_step(carry, _):
            env_states, key = carry
            key, action_key, step_key = jax.random.split(key, 3)

            # Get observations from states
            obs = jax.vmap(env._compute_observation)(env_states)
            obs_grid, obs_scalar = obs

            # Forward pass (vmapped over envs)
            action_mean, action_log_std, values = jax.vmap(
                functools.partial(network.apply, params)
            )(obs_grid, obs_scalar)

            # Sample actions
            action_keys = jax.random.split(action_key, ppo_config.num_envs)
            actions, log_probs = jax.vmap(sample_action)(action_keys, action_mean, action_log_std)

            # Clamp actions
            actions_clamped = jnp.stack([
                jnp.clip(actions[:, 0], 0.0, 1.0),   # main thrust
                jnp.clip(actions[:, 1], -1.0, 1.0),   # rotation
            ], axis=-1)

            # Step environments
            step_keys = jax.random.split(step_key, ppo_config.num_envs)
            _, next_states, rewards, dones, infos = jax.vmap(env.step)(
                step_keys, env_states, actions_clamped
            )

            # Auto-reset done environments
            reset_keys = jax.random.split(key, ppo_config.num_envs)
            _, reset_states = jax.vmap(env.reset)(reset_keys)

            # Select reset or continued state
            next_states = jax.tree.map(
                lambda r, c, d: jnp.where(d[:, None] if r.ndim > 1 else d, r, c)
                if hasattr(r, 'ndim') else jnp.where(d, r, c),
                reset_states, next_states, dones,
            )

            transition = Transition(
                obs_grid=obs_grid,
                obs_scalar=obs_scalar,
                action=actions,
                log_prob=log_probs,
                value=values,
                reward=rewards,
                done=dones,
            )

            return (next_states, key), transition

        (env_states, key), rollout = jax.lax.scan(
            _env_step,
            (env_states, key),
            None,
            length=ppo_config.num_steps,
        )

        # Bootstrap value
        final_obs = jax.vmap(env._compute_observation)(env_states)
        _, _, bootstrap_values = jax.vmap(
            functools.partial(network.apply, params)
        )(final_obs[0], final_obs[1])

        values_with_bootstrap = jnp.concatenate([rollout.value, bootstrap_values[None]], axis=0)
        advantages, returns = compute_gae(
            rollout.reward, values_with_bootstrap, rollout.done,
            ppo_config.gamma, ppo_config.gae_lambda,
        )

        # --- PPO Update ---
        # Flatten rollout: (T, N, ...) -> (T*N, ...)
        batch_size = ppo_config.num_steps * ppo_config.num_envs
        minibatch_size = batch_size // ppo_config.num_minibatches

        def _flatten(x):
            return x.reshape(batch_size, *x.shape[2:])

        flat_rollout = jax.tree.map(_flatten, rollout)
        flat_advantages = _flatten(advantages)
        flat_returns = _flatten(returns)

        def _update_epoch(carry, _):
            params, opt_state, key = carry
            key, perm_key = jax.random.split(key)
            permutation = jax.random.permutation(perm_key, batch_size)

            def _update_minibatch(carry, batch_indices):
                params, opt_state = carry
                batch = (
                    flat_rollout.obs_grid[batch_indices],
                    flat_rollout.obs_scalar[batch_indices],
                    flat_rollout.action[batch_indices],
                    flat_rollout.log_prob[batch_indices],
                    flat_advantages[batch_indices],
                    flat_returns[batch_indices],
                )

                grad_fn = jax.value_and_grad(ppo_loss, has_aux=True)
                (loss, metrics), grads = grad_fn(
                    params, network.apply, batch,
                    ppo_config.clip_eps, ppo_config.entropy_coeff, ppo_config.value_coeff,
                )
                updates, opt_state_new = tx.update(grads, opt_state, params)
                params_new = optax.apply_updates(params, updates)
                return (params_new, opt_state_new), metrics

            # Create minibatch indices
            mb_indices = permutation.reshape(ppo_config.num_minibatches, minibatch_size)
            (params, opt_state), metrics = jax.lax.scan(
                _update_minibatch, (params, opt_state), mb_indices
            )
            return (params, opt_state, key), metrics

        (params, opt_state, key), epoch_metrics = jax.lax.scan(
            _update_epoch,
            (params, opt_state, key),
            None,
            length=ppo_config.num_epochs,
        )

        # Aggregate metrics
        mean_reward = jnp.mean(rollout.reward)
        mean_return = jnp.mean(returns)

        new_train_state = TrainState(
            params=params,
            opt_state=opt_state,
            key=key,
            env_states=env_states,
            update_step=update_step + 1,
        )

        metrics = {
            'mean_reward': mean_reward,
            'mean_return': mean_return,
            'policy_loss': jnp.mean(epoch_metrics['policy_loss']),
            'value_loss': jnp.mean(epoch_metrics['value_loss']),
            'entropy': jnp.mean(epoch_metrics['entropy']),
            'approx_kl': jnp.mean(epoch_metrics['approx_kl']),
        }

        return new_train_state, metrics

    return _train_step
