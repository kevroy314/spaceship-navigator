"""Near-optimal reference flights: the yardstick for "how good is this cost?".

The environment is differentiable, but gradients through hundreds of control
ticks of n-body flight are badly conditioned — a long flight is chaotic, so the
loss surface is dominated by flyby sensitivity.  Direct shooting with Adam moved
the closest approach by only a few percent in 250 iterations, so this plans with
the cross-entropy method on the **true** objective instead: hundreds of candidate
control sequences per iteration, each rolled out in the real environment (hard
terminations included), refitting a Gaussian to the elite set.

Every reference is therefore a flight the agent could have flown, scored by
exactly the metric the agent is scored by.  A plan is a few piecewise-constant
segments — a small search space, and enough for coast-and-burn flights.
"""

from typing import NamedTuple

import jax
import jax.numpy as jnp

from spacenav import baselines
from spacenav import constants as C
from spacenav import env as E
from spacenav.types import ARRIVED, Level, Task


class PlanConfig(NamedTuple):
    ticks: int = 600           # 40 s of flight at 15 Hz
    segments: int = 16         # piecewise-constant control segments
    samples: int = 256
    elites: int = 24
    iters: int = 14
    init_sigma: float = 1.2
    sigma_floor: float = 0.06
    miss_penalty: float = 10.0  # charged for not arriving, plus the closest approach


def expand(seg, ticks):
    """Segments -> per-tick actions (turn in [-1, 1], throttle in [0, 1])."""
    per = ticks // seg.shape[-2] + 1
    a = jnp.repeat(seg, per, axis=-2)[..., :ticks, :]
    return jnp.stack([jnp.tanh(a[..., 0]), jax.nn.sigmoid(a[..., 1])], axis=-1)


def fly(level: Level, task: Task, actions):
    """Roll actions out in the real environment, tracking the best "arrival gap".

    gap = max(distance / target radius, relative speed / tolerance): below 1 is an
    arrival.  Ranking misses by distance alone lets a rendezvous plan fly straight
    through the station at speed and never learn to brake."""
    def f(carry, a):
        state, best_gap, closest = carry
        new, _, _ = E.step(level, task, state, a)
        tpos, tvel = E.target_state(task, new)
        d = jnp.sqrt(jnp.sum((new.ship.pos - tpos) ** 2) + 1e-9)
        dv = jnp.sqrt(jnp.sum((new.ship.vel - tvel) ** 2) + 1e-9)
        # how far from reaching the current waypoint, counting waypoints already done
        left = jnp.sum(task.wp_active & ~new.visited)
        gap = left - 1 + jnp.maximum(d / task.wp_radius[new.leg], dv / task.wp_v_tol[new.leg])
        running = state.status == 0
        return (new, jnp.where(running, jnp.minimum(best_gap, gap), best_gap),
                jnp.where(running, jnp.minimum(closest, d), closest)), new.ship.pos

    init = (E.reset(level, task), jnp.float32(jnp.inf), jnp.float32(jnp.inf))
    (final, gap, closest), pos = jax.lax.scan(f, init, actions)
    return final, gap, closest, pos


def score(task: Task, final, gap, cfg: PlanConfig):
    """Weighted cost if it arrived, else a penalty that still rewards closing the gap."""
    return jnp.where(final.status == ARRIVED, E.objective_cost(task, final.costs),
                     cfg.miss_penalty + gap)


def pilot_seed(level: Level, task: Task, cfg: PlanConfig):
    """Segment parameters that reproduce the scripted pilot's flight.

    The pilot arrives on most tasks, so seeding there means the search starts from a
    working flight and spends its budget on making it cheaper."""
    _, tr = E.rollout(level, task, baselines.pilot(level, task), n_ticks=cfg.ticks)
    per = cfg.ticks // cfg.segments
    turn = tr["turn"][: per * cfg.segments].reshape(cfg.segments, per).mean(1)
    thr = tr["thrust"][: per * cfg.segments].reshape(cfg.segments, per).mean(1)
    raw_turn = jnp.arctanh(jnp.clip(turn, -0.95, 0.95))
    raw_thr = jnp.log(jnp.clip(thr, 0.02, 0.98) / (1 - jnp.clip(thr, 0.02, 0.98)))
    return jnp.stack([raw_turn, raw_thr], -1)


def plan(level: Level, task: Task, key, cfg: PlanConfig = PlanConfig()):
    """Cross-entropy search over control segments; returns the best flight found."""
    shape = (cfg.segments, 2)
    mu0 = pilot_seed(level, task, cfg)

    def run(seg):
        final, gap, _, _ = fly(level, task, expand(seg, cfg.ticks))
        return score(task, final, gap, cfg)

    def step(carry, key):
        mu, sigma, best_seg, best_score = carry
        samples = mu[None] + sigma[None] * jax.random.normal(key, (cfg.samples,) + shape)
        samples = samples.at[0].set(mu).at[1].set(best_seg)    # keep mean and incumbent best
        scores = jax.vmap(run)(samples)
        order = jnp.argsort(scores)
        elite = samples[order[: cfg.elites]]
        top, top_score = samples[order[0]], scores[order[0]]
        better = top_score < best_score
        best_seg = jnp.where(better, top, best_seg)
        best_score = jnp.where(better, top_score, best_score)
        return (elite.mean(0), jnp.maximum(elite.std(0), cfg.sigma_floor), best_seg,
                best_score), best_score

    init = (mu0, jnp.full(shape, cfg.init_sigma), mu0, run(mu0))
    (_, _, best_seg, _), curve = jax.lax.scan(step, init, jax.random.split(key, cfg.iters))
    final, gap, closest, pos = fly(level, task, expand(best_seg, cfg.ticks))
    arrived = final.status == ARRIVED
    ticks = jnp.where(arrived, jnp.round(final.costs.time / C.CTRL_DT).astype(jnp.int32), cfg.ticks)
    return dict(cost=jnp.where(arrived, E.objective_cost(task, final.costs), jnp.inf),
                arrived=arrived, status=final.status, time=final.costs.time,
                fuel=final.costs.fuel, damage=final.costs.damage, closest=closest, gap=gap,
                ticks=ticks, pos=pos, curve=curve)


def plan_batch(levels, tasks, key, cfg: PlanConfig = PlanConfig()):
    n = jax.tree.leaves(tasks)[0].shape[0]
    return jax.vmap(lambda l, t, k: plan(l, t, k, cfg))(levels, tasks, jax.random.split(key, n))
