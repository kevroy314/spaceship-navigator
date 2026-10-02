"""PPO, PureJaxRL style: rollout, GAE and all minibatch epochs are one jitted call."""

from dataclasses import dataclass
from typing import NamedTuple

import jax
import jax.numpy as jnp
import optax

from spacenav import constants as C
from spacenav import env as E
from spacenav.rl.network import ActorCritic
from spacenav.rl.vecenv import VecState, observe, vec_reset, vec_step


@dataclass(frozen=True)
class PPOConfig:
    n_envs: int = 1024
    n_steps: int = 128
    epochs: int = 4
    minibatches: int = 8
    lr: float = 3e-4
    lr_final_frac: float = 0.1       # linear anneal to this fraction of lr by the end
    ent_coef_final: float = 0.002    # entropy bonus anneals from ent_coef to this
    total_updates: int = 2000        # horizon the schedules are written against
    gamma: float = 0.999             # ~65 s of lookahead at 15 Hz; trips take 20-40 s
    lam: float = 0.95
    clip: float = 0.2
    ent_coef: float = 0.01
    vf_coef: float = 0.5
    max_grad_norm: float = 0.5
    episodes_per_start: int = 4       # keep A for this many episodes, then reroll
    d_model: int = 128
    # Curriculum, ramped over the first ramp_frac of training: more rendezvous legs,
    # longer tours, and eventually tours that may be flown in any order.
    rdv_start: float = 0.1
    rdv_end: float = 0.45
    rdv_ramp_frac: float = 0.5
    wp_max_start: float = 1.0     # longest tour allowed at the start
    wp_max_end: float = 3.0
    order_free_end: float = 0.3
    # Scale bootstrapping: waypoints parked where two bodies pull equally, so a
    # patched-conic policy ("only the body I orbit matters") stops being enough.
    # Held at zero until the ship can fly at all, then ramped over the back half.
    contested_end: float = 0.35
    # The tank starts generous and tightens: a ship that has never learnt to fly
    # cannot learn it while also being short of fuel (v3 sat at 2% for 13M steps).
    budget_start: tuple = (1.1, 1.6)
    budget_end: tuple = C.DELTAV_BUDGET_RANGE
    accel_start: tuple = (1.5, 3.0)
    accel_end: tuple = C.SHIP_ACCEL_RANGE

    @property
    def batch(self):
        return self.n_envs * self.n_steps

    def entropy_coef(self, frac):
        f = min(max(frac, 0.0), 1.0)
        return self.ent_coef + (self.ent_coef_final - self.ent_coef) * f

    def _ramp(self, frac):
        return min(frac / self.rdv_ramp_frac, 1.0) if self.rdv_ramp_frac > 0 else 1.0

    def p_rendezvous(self, frac):
        return self.rdv_start + (self.rdv_end - self.rdv_start) * self._ramp(frac)

    def p_contested(self, frac):
        """Zero until half-way, then ramped in over the next 40% of training.

        Not keyed to `_ramp`, which has already saturated by then: contested
        waypoints are the last thing to arrive, after the ship can fly, tour and
        ration fuel, because they are the ones that punish a coarse policy.
        """
        return self.contested_end * min(max((frac - 0.5) / 0.4, 0.0), 1.0)

    def curriculum(self, frac):
        """The knobs that widen the mission distribution as training goes on."""
        r = self._ramp(frac)
        lerp = lambda a, b: tuple(float(x + (y - x) * r) for x, y in zip(a, b))
        return dict(p_rdv=self.p_rendezvous(frac),
                    wp_max=self.wp_max_start + (self.wp_max_end - self.wp_max_start) * r,
                    p_order_free=self.order_free_end * max(0.0, (r - 0.5) * 2),
                    budget=lerp(self.budget_start, self.budget_end),
                    accel=lerp(self.accel_start, self.accel_end),
                    p_contested=self.p_contested(frac))


class Transition(NamedTuple):
    obs: dict
    action: jnp.ndarray
    logp: jnp.ndarray
    value: jnp.ndarray
    reward: jnp.ndarray
    done: jnp.ndarray


class Runner(NamedTuple):
    params: dict
    opt_state: tuple
    vs: VecState
    obs: dict
    key: jnp.ndarray


def make(cfg: PPOConfig, pool, key):
    net = ActorCritic(d=cfg.d_model)
    steps_per_update = cfg.epochs * cfg.minibatches
    lr_sched = optax.linear_schedule(cfg.lr, cfg.lr * cfg.lr_final_frac,
                                     transition_steps=max(cfg.total_updates * steps_per_update, 1))
    tx = optax.chain(optax.clip_by_global_norm(cfg.max_grad_norm), optax.adam(lr_sched, eps=1e-5))
    k_env, k_init, k_run = jax.random.split(key, 3)
    base_cfg = E.TaskConfig(p_rendezvous=cfg.rdv_start, deltav_budget=cfg.budget_start,
                            accel_range=cfg.accel_start)
    vs = jax.jit(vec_reset, static_argnums=(3, 4))(k_env, pool, base_cfg, cfg.n_envs,
                                                  cfg.episodes_per_start)
    obs = jax.jit(observe)(pool, vs)
    params = net.init(k_init, jax.tree.map(lambda x: x[0], obs))
    runner = Runner(params, tx.init(params), vs, obs, k_run)
    rcfg = E.RewardConfig(gamma=cfg.gamma)

    def loss_fn(params, mb, ent_coef):
        tr, adv, target = mb
        logits, v = net.apply(params, tr.obs)
        logp_all = jax.nn.log_softmax(logits)
        # one-hot product instead of take_along_axis: its gradient is a scatter, which has
        # no working kernel on Pascal GPUs in this jaxlib ("Failed to load in-memory CUBIN")
        logp = jnp.sum(logp_all * jax.nn.one_hot(tr.action, logp_all.shape[-1]), axis=-1)
        ratio = jnp.exp(logp - tr.logp)
        adv = (adv - adv.mean()) / (adv.std() + 1e-8)
        pg = -jnp.minimum(ratio * adv, jnp.clip(ratio, 1 - cfg.clip, 1 + cfg.clip) * adv).mean()
        v_clip = tr.value + jnp.clip(v - tr.value, -cfg.clip, cfg.clip)
        vf = 0.5 * jnp.maximum((v - target) ** 2, (v_clip - target) ** 2).mean()
        ent = -(jnp.exp(logp_all) * logp_all).sum(-1).mean()
        loss = pg + cfg.vf_coef * vf - ent_coef * ent
        info = dict(pg_loss=pg, vf_loss=vf, entropy=ent,
                    approx_kl=((ratio - 1) - jnp.log(ratio)).mean(),
                    clip_frac=(jnp.abs(ratio - 1) > cfg.clip).mean())
        return loss, info

    @jax.jit
    def update(runner: Runner, p_rdv, ent_coef, wp_max, p_order_free, budget, accel,
               p_contested):
        params, opt_state, vs, obs, key = runner
        tcfg = base_cfg._replace(p_rendezvous=p_rdv, p_order_free=p_order_free,
                                 waypoints=(1, wp_max), deltav_budget=budget, accel_range=accel,
                                 p_contested=p_contested)

        def env_step(carry, _):
            vs, obs, key = carry
            key, k_act, k_env = jax.random.split(key, 3)
            logits, value = net.apply(params, obs)
            a = jax.random.categorical(k_act, logits)
            logp = jnp.take_along_axis(jax.nn.log_softmax(logits), a[:, None], -1)[:, 0]
            vs2, obs2, r, done, fin = vec_step(k_env, pool, vs, a, tcfg, rcfg, cfg.episodes_per_start)
            return (vs2, obs2, key), (Transition(obs, a, logp, value, r, done), fin)

        (vs, obs, key), (traj, fins) = jax.lax.scan(env_step, (vs, obs, key), None, cfg.n_steps)
        _, last_v = net.apply(params, obs)

        def gae(carry, t):
            adv_next, v_next = carry
            nd = 1.0 - t.done
            delta = t.reward + cfg.gamma * v_next * nd - t.value
            adv = delta + cfg.gamma * cfg.lam * nd * adv_next
            return (adv, t.value), adv
        _, advs = jax.lax.scan(gae, (jnp.zeros_like(last_v), last_v), traj, reverse=True)
        targets = advs + traj.value

        flat = jax.tree.map(lambda x: x.reshape((cfg.batch,) + x.shape[2:]), (traj, advs, targets))

        def epoch(carry, _):
            params, opt_state, key = carry
            key, k = jax.random.split(key)
            perm = jax.random.permutation(k, cfg.batch)
            mbs = jax.tree.map(lambda x: x[perm].reshape((cfg.minibatches, -1) + x.shape[1:]), flat)

            def mb_step(c, mb):
                params, opt_state = c
                (_, info), grads = jax.value_and_grad(loss_fn, has_aux=True)(params, mb, ent_coef)
                upd, opt_state = tx.update(grads, opt_state, params)
                return (optax.apply_updates(params, upd), opt_state), info
            (params, opt_state), infos = jax.lax.scan(mb_step, (params, opt_state), mbs)
            return (params, opt_state, key), infos

        key, k_ep = jax.random.split(key)
        (params, opt_state, _), infos = jax.lax.scan(epoch, (params, opt_state, k_ep), None, cfg.epochs)

        d = fins["done"]
        n = jnp.maximum(d.sum(), 1)
        mean_done = lambda x: jnp.sum(jnp.where(d, x, 0.0)) / n
        rdv = d & fins["rendezvous"]
        metrics = dict(
            episodes=d.sum(),
            ep_return=mean_done(fins["ret"]),
            success=mean_done(fins["success"].astype(jnp.float32)),
            success_flythrough=jnp.sum(fins["success"] & ~fins["rendezvous"]) / jnp.maximum((d & ~fins["rendezvous"]).sum(), 1),
            success_rendezvous=jnp.sum(fins["success"] & fins["rendezvous"]) / jnp.maximum(rdv.sum(), 1),
            ep_length=mean_done(fins["length"].astype(jnp.float32)),
            waypoints=mean_done(fins["waypoints"].astype(jnp.float32)),
            reached_frac=mean_done(fins["reached"] / jnp.maximum(fins["waypoints"], 1)),
            damage=mean_done(fins["damage"]),
            fuel=mean_done(fins["fuel"]),
            crash=mean_done((fins["status"] == 2).astype(jnp.float32)),
            destroyed=mean_done((fins["status"] == 3).astype(jnp.float32)),
            lost=mean_done((fins["status"] == 4).astype(jnp.float32)),
            timeout=mean_done((fins["status"] == 5).astype(jnp.float32)),
            value_mean=traj.value.mean(),
            ent_coef=ent_coef,
            **jax.tree.map(jnp.mean, infos),
        )
        return Runner(params, opt_state, vs, obs, key), metrics

    return net, runner, update
