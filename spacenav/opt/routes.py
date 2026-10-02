"""Two routes per level: the one a coarse model plans, and the one the parameters allow.

The comparison is controlled — same search, same budget, same number of control
segments, flown in the same true world. The *only* difference is which model the
planner was allowed to see while planning:

* **approximate**: `topk=1`, patched conics. Only the dominant attractor exists.
  This is the route a well-trained intuition produces.
* **idealised**: the full field. This is the route the parameters actually allow,
  and the one you can only find by computing with them.

Alongside them this measures how much *structure* each answer has, which is a
different question from whether a coarse model suffices (see `spacenav.demand`):

* `coast_ok` — does the level solve itself if you do nothing? A free ride is
  pretty to watch and empty to play, and it is a generator bug rather than a
  difficulty tier, so it gets detected and excluded rather than ranked.
* `mdl` — the fewest piecewise-constant segments that still arrive. Control
  description length in a fixed, meaningful basis (burns at 15 Hz), which is the
  action-space analogue of smoothing the path before measuring its complexity.
  Raw per-tick action counts measure throttle chatter instead of the plan.
* `switches` — action changes after quantising to the nine controls a human
  actually has. "How many decisions did the pilot make."

A level worth building is one where `mdl` is more than a burn or two *and* the
approximate route misses: the answer has structure, and a coarse model cannot
find that structure.
"""

from typing import NamedTuple

import jax
import jax.numpy as jnp

from spacenav import constants as C
from spacenav import env as E
from spacenav.opt.planner import expand
from spacenav.opt.probe import ProbeConfig, cem, fly_from
from spacenav.types import ARRIVED, Level, Task

# the control set a person has: turn left/straight/right x off/gentle/full
TURNS = jnp.array([-1.0, 0.0, 1.0])
THROTTLES = jnp.array([0.0, 0.3, 1.0])

# segment counts to test for control description length.  Only the ones that
# divide the plan's segment count are usable, since coarsening groups segments
# evenly; `mdl_steps()` filters them so a non-default RouteConfig cannot reshape
# an empty axis.
MDL_STEPS = (1, 2, 3, 4, 6, 8, 12, 16, 24, 48)


def mdl_steps(segments: int):
    return tuple(m for m in MDL_STEPS if m <= segments and segments % m == 0)


class RouteConfig(NamedTuple):
    # The horizon must cover the whole mission, not a convenient slice of it: a
    # weak-engine tour takes 40-130 s, so a 600-tick (40 s) search reports every
    # route as still running and measures nothing.  Give it the episode's own clock.
    ticks: int = C.MAX_EPISODE_TICKS
    # 48 segments is ~3 s each.  Measured on easy ordinary missions, where
    # patched conics is accurate and both routes therefore *should* tie: 12
    # segments gives 83/83, 24 gives 83/100, 48 gives 100/100.  Anything coarser
    # penalises the approximate route for the search being weak rather than for
    # the model being wrong, which would fake the very result we are testing.
    segments: int = 48
    samples: int = 256
    elites: int = 32
    iters: int = 12
    init_sigma: float = 1.2
    sigma_floor: float = 0.06
    miss_penalty: float = 10.0
    stride: int = 8            # trajectory samples kept for the page

    def probe_cfg(self):
        """ProbeConfig view, so the cross-entropy search is shared code."""
        return ProbeConfig(ticks=self.ticks, segments=self.segments, samples=self.samples,
                           elites=self.elites, iters=self.iters, init_sigma=self.init_sigma,
                           sigma_floor=self.sigma_floor, miss_penalty=self.miss_penalty)


def quantise(actions):
    """Snap a continuous plan onto the nine controls a human has."""
    ti = jnp.argmin(jnp.abs(actions[:, :1] - TURNS[None, :]), axis=1)
    hi = jnp.argmin(jnp.abs(actions[:, 1:2] - THROTTLES[None, :]), axis=1)
    return ti * 3 + hi, jnp.stack([TURNS[ti], THROTTLES[hi]], axis=-1)


def switches(actions):
    """How many times the quantised control changes: decisions, not ticks."""
    ids, _ = quantise(actions)
    return jnp.sum(ids[1:] != ids[:-1])


def coarsen(seg, m):
    """Average `seg` down to `m` segments and back up: the plan at resolution m."""
    n = seg.shape[0]
    g = n // m
    return jnp.repeat(seg[: g * m].reshape(m, g, 2).mean(1), g, axis=0)


def control_mdl(level: Level, task: Task, seg, ticks, steps=None):
    """Fewest segments that still arrive, and arrival at each resolution tested.

    The plan is coarsened, not re-searched, so this measures how compressible the
    found solution is rather than how hard a coarser search would be.
    """
    st = E.reset(level, task)
    steps = steps or mdl_steps(seg.shape[0])

    def at(m):
        final, *_ = fly_from(level, task, st, expand(coarsen(seg, m), ticks), 0)
        return final.status == ARRIVED

    ok = jnp.stack([at(m) for m in steps])
    # smallest resolution that still arrives; 0 means none of them did
    best = jnp.min(jnp.where(ok, jnp.asarray(steps), jnp.iinfo(jnp.int32).max))
    return jnp.where(jnp.any(ok), best, 0), ok


def pair(level: Level, task: Task, key, cfg: RouteConfig = RouteConfig()):
    """Plan with patched conics and with the full field; fly both for real."""
    pcfg = cfg.probe_cfg()
    st = E.reset(level, task)
    keys = jax.random.split(key, 3)

    out = {}
    segs = {}
    for name, topk, k in (("approx", 1, keys[0]), ("ideal", 0, keys[1])):
        seg = cem(level, task, st, k, pcfg, cfg.ticks, topk)
        acts = expand(seg, cfg.ticks)
        final, gap, pos, alive = fly_from(level, task, st, acts, 0)
        arrived = final.status == ARRIVED
        segs[name] = seg
        out[name] = dict(
            arrived=arrived, status=final.status, gap=gap,
            cost=jnp.where(arrived, E.objective_cost(task, final.costs), jnp.inf),
            time=final.costs.time, fuel=final.costs.fuel, damage=final.costs.damage,
            ticks=jnp.sum(alive), switches=switches(acts),
            pos=pos[:: cfg.stride], alive=alive[:: cfg.stride],
            actions=quantise(acts)[0][:: cfg.stride],
        )

    # does the level fly itself?
    coast, _, _, _ = fly_from(level, task, st, jnp.zeros((cfg.ticks, 2)), 0)
    mdl, mdl_ok = control_mdl(level, task, segs["ideal"], cfg.ticks,
                              mdl_steps(cfg.segments))

    flat = dict(coast_ok=coast.status == ARRIVED, mdl=mdl, mdl_ok=mdl_ok)
    for name, d in out.items():
        for k, v in d.items():
            flat[f"{name}_{k}"] = v
    return flat


def pair_batch(levels, tasks, key, cfg: RouteConfig = RouteConfig()):
    n = jax.tree.leaves(tasks)[0].shape[0]
    return jax.vmap(lambda l, t, k: pair(l, t, k, cfg))(levels, tasks, jax.random.split(key, n))
