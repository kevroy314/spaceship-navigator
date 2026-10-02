"""Batched, auto-resetting training environment over a level pool.

Each env slot keeps its level and start point A for `episodes_per_start`
episodes (only B and the objective are resampled), then draws a new level and a
new A.  Everything lives on the GPU and is stepped with one jitted call.
"""

from typing import NamedTuple

import jax
import jax.numpy as jnp

from spacenav import env as E
from spacenav.rl.network import ACTIONS
from spacenav.types import ARRIVED, EnvState, Level, Task


class VecState(NamedTuple):
    level_idx: jnp.ndarray   # (B,)
    start: E.Start           # batched
    remaining: jnp.ndarray   # (B,) episodes left on this start
    task: Task               # batched
    state: EnvState          # batched
    ep_return: jnp.ndarray   # (B,) running return of the current episode


def gather(pool: Level, idx) -> Level:
    return jax.tree.map(lambda x: x[idx], pool)


def _new_start(key, pool, cfg):
    k1, k2 = jax.random.split(key)
    idx = jax.random.randint(k1, (), 0, pool.mass.shape[0])
    lv = gather(pool, idx)
    return idx, E.sample_start(k2, lv, cfg)


def _reset_one(key, pool, cfg, idx, start, remaining, force_new, episodes_per_start):
    """Next episode for one slot: same level/start unless it is used up."""
    k1, k2 = jax.random.split(key)
    new_idx, new_start = _new_start(k1, pool, cfg)
    fresh = force_new | (remaining <= 0) | ~start.ok
    idx = jnp.where(fresh, new_idx, idx)
    start = jax.tree.map(lambda a, b: jnp.where(fresh, a, b), new_start, start)
    remaining = jnp.where(fresh, episodes_per_start, remaining)
    lv = gather(pool, idx)
    task, _ = E.sample_tour(k2, lv, start, cfg)
    return idx, start, remaining - 1, task, E.reset(lv, task)


def vec_reset(key, pool, cfg: E.TaskConfig, n_envs, episodes_per_start):
    keys = jax.random.split(key, n_envs)
    dummy_idx = jnp.zeros(n_envs, jnp.int32)
    lv0 = gather(pool, dummy_idx)
    dummy_start = jax.vmap(lambda k, l: E.sample_start(k, l, cfg))(keys, lv0)
    idx, start, rem, task, state = jax.vmap(
        lambda k, i, s: _reset_one(k, pool, cfg, i, s, jnp.int32(0), True, episodes_per_start)
    )(jax.random.split(key, n_envs), dummy_idx, dummy_start)
    return VecState(idx, start, rem, task, state, jnp.zeros(n_envs))


def observe(pool, vs: VecState):
    return jax.vmap(E.observe)(gather(pool, vs.level_idx), vs.task, vs.state)


def vec_step(key, pool, vs: VecState, action_ids, cfg: E.TaskConfig, rcfg: E.RewardConfig,
             episodes_per_start):
    """Step every env; finished ones are reset.  Returns (vs, obs, reward, done, info)."""
    lv = gather(pool, vs.level_idx)
    actions = ACTIONS[action_ids]
    state, reward, info = jax.vmap(lambda l, t, s, a: E.step(l, t, s, a, rcfg))(
        lv, vs.task, vs.state, actions)
    done = info["done"]
    ep_return = vs.ep_return + reward

    # episode summaries for logging (valid where done)
    fin = dict(done=done, success=done & (state.status == ARRIVED), ret=ep_return,
               status=state.status, length=state.tick,
               objective=jax.vmap(E.objective_cost)(vs.task, state.costs),
               damage=state.costs.damage, fuel=state.costs.fuel,
               rendezvous=jnp.any(vs.task.wp_v_tol < 1e8, axis=-1),
               waypoints=jnp.sum(vs.task.wp_active, axis=-1),
               reached=jnp.sum(state.visited & vs.task.wp_active, axis=-1))

    n = done.shape[0]
    idx, start, rem, task, fresh = jax.vmap(
        lambda k, i, s, r: _reset_one(k, pool, cfg, i, s, r, False, episodes_per_start)
    )(jax.random.split(key, n), vs.level_idx, vs.start, vs.remaining)
    sel = lambda a, b: jax.tree.map(lambda x, y: jnp.where(
        done.reshape(done.shape + (1,) * (x.ndim - 1)), x, y), a, b)
    vs = VecState(level_idx=jnp.where(done, idx, vs.level_idx), start=sel(start, vs.start),
                  remaining=jnp.where(done, rem, vs.remaining), task=sel(task, vs.task),
                  state=sel(fresh, state), ep_return=jnp.where(done, 0.0, ep_return))
    return vs, observe(pool, vs), reward, done, fin
