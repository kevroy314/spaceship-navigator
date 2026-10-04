"""How much model, and how much precision, does a level actually demand?

Two independent questions, deliberately not collapsed into one number:

**Model scale.**  A human flies by patched conics: only the body you are
orbiting matters, splice two-body arcs together.  `physics.gravity_at(topk=k)`
is that model exactly, and it comes in a ladder (k=1 dominant attractor, k=2,
..., full field).  So we can *plan inside a coarse model and fly the plan in the
true world*, replanning as a human would.  The smallest k that still arrives is
the level's model scale: k=1 means mesoscale intuition is sufficient, and "no k
short of the full field" means the level cannot be flown by anyone reasoning
with a simplified picture of the forces, however well they execute.

**Execution precision.**  Given the right plan, how much timing slack is there?
Two probes: a uniform delay (how wide is the launch window) and per-burn onset
jitter (can a hand hold this).  Human numbers to compare against are in
`HUMAN`: ~20 ms motor timing jitter, ~200 ms reaction time, ~3 Hz closed-loop
correction bandwidth.  A level can be perfectly compressible and still
unflyable by a person, and the two failure modes need different level design,
which is why they stay separate axes.

Cheap companions: the Lyapunov rate along the reference flight (how fast is
precision eaten), and the "incoherence budget" — the fraction of the force a
patched-conic model throws away, integrated along the flight — which is the
closed-form predictor of the expensive model-scale measurement.
"""

from typing import NamedTuple

import jax
import jax.numpy as jnp

from spacenav import baselines
from spacenav import constants as C
from spacenav import env as E
from spacenav import physics as P
from spacenav.opt.planner import PlanConfig, expand, score
from spacenav.types import ARRIVED, Level, Task


class Human(NamedTuple):
    """Measured limits of a human pilot, for calling a level flyable or not.

    Timing jitter is the SD of onset asynchrony for a well-practised discrete
    action (Repp 2005: 3-5% of the inter-onset interval, ~15-25 ms at 500 ms).
    `window_min` is simple visual reaction time: a window narrower than this
    cannot be aimed at, only hit by luck.  `jitter_pass` is the success rate we
    insist on at that jitter before calling a level humanly executable.
    """
    timing_sd: float = 0.020        # s, motor onset jitter
    reaction: float = 0.200         # s, visual simple RT
    window_min: float = 0.200       # s, narrowest aimable window
    bandwidth: float = 3.0          # Hz, closed-loop correction ceiling
    jitter_pass: float = 0.75       # required success rate under own jitter


HUMAN = Human()


class ProbeConfig(NamedTuple):
    # The horizon is the episode's own clock.  Anything shorter silently reports
    # every route as still running: a weak-engine tour needs 40-130 s, so a 40 s
    # search measures nothing at all (it cost a whole build to learn that).
    ticks: int = C.MAX_EPISODE_TICKS
    # Segment duration is what matters, not the count: 12 segments over a 600 s
    # episode is 50 s of constant thrust each, and the capacity sweep at 150 s
    # showed ~3 s segments were needed for a plan to arrive at all. 48 keeps each
    # segment at 12.5 s, which is also about the granularity at which a burn is a
    # recognisable manoeuvre rather than a tick.
    segments: int = 48
    samples: int = 96
    elites: int = 12
    iters: int = 7
    init_sigma: float = 1.2
    sigma_floor: float = 0.06
    miss_penalty: float = 10.0
    ladder: tuple = (1, 2, 0)              # coarse models to try; 0 = full field
    replans: int = 1                       # mid-course corrections allowed per model
    delays: tuple = (1, 2, 3, 5, 8, 12)    # uniform delay probe, in control ticks
    # (one tick is 67 ms at 15 Hz, so this spans 0.07-0.8 s of lateness)
    jitters: tuple = (0.5, 1.0, 2.0, 4.0)  # onset jitter SD, in control ticks
    jitter_samples: int = 24

    def plan_cfg(self):
        return PlanConfig(ticks=self.ticks, segments=self.segments, samples=self.samples,
                          elites=self.elites, iters=self.iters, init_sigma=self.init_sigma,
                          sigma_floor=self.sigma_floor, miss_penalty=self.miss_penalty)


# ---------------------------------------------------------------------------
# flying and planning from an arbitrary state, in an arbitrary model
# ---------------------------------------------------------------------------

def fly_from(level: Level, task: Task, state, actions, topk: int = 0):
    """Roll `actions` out from `state` under the `topk` model, tracking arrival gap.

    gap = (waypoints outstanding - 1) + max(distance/radius, speed error/tolerance):
    below 1 means the current waypoint is reached.  Ranking misses by distance
    alone would let a rendezvous plan sail through the station at speed.
    """
    def f(carry, a):
        st, best_gap = carry
        new, _, _ = E.step(level, task, st, a, topk=topk)
        tpos, tvel = E.target_state(task, new)
        d = jnp.sqrt(jnp.sum((new.ship.pos - tpos) ** 2) + 1e-9)
        dv = jnp.sqrt(jnp.sum((new.ship.vel - tvel) ** 2) + 1e-9)
        left = jnp.sum(task.wp_active & ~new.visited)
        gap = left - 1 + jnp.maximum(d / task.wp_radius[new.leg], dv / task.wp_v_tol[new.leg])
        running = st.status == 0
        return ((new, jnp.where(running, jnp.minimum(best_gap, gap), best_gap)),
                (new.ship.pos, running))

    (final, gap), (pos, alive) = jax.lax.scan(f, (state, jnp.float32(jnp.inf)), actions)
    return final, gap, pos, alive


def pilot_seed_from(level: Level, task: Task, state, ticks: int, segments: int, topk: int = 0):
    """Segment parameters reproducing the scripted pilot's flight from `state`.

    The pilot is proportional navigation with gravity compensation — a reflex, not
    a plan — and it arrives on most easy tasks, so seeding the search there spends
    the budget on making a working flight cheaper rather than on finding one.  Run
    inside the coarse model too, so a coarse search starts from a coarse-plausible
    flight.
    """
    pol = baselines.pilot(level, task)

    def f(st, _):
        a = pol(E.observe(level, task, st), st)
        new, _, _ = E.step(level, task, st, a, topk=topk)
        return new, (jnp.clip(a[0], -1, 1), new.thrust)

    _, (turn, thr) = jax.lax.scan(f, state, None, length=ticks)
    per = ticks // segments
    turn = turn[: per * segments].reshape(segments, per).mean(1)
    thr = thr[: per * segments].reshape(segments, per).mean(1)
    raw_turn = jnp.arctanh(jnp.clip(turn, -0.95, 0.95))
    thr = jnp.clip(thr, 0.02, 0.98)
    return jnp.stack([raw_turn, jnp.log(thr / (1 - thr))], -1)


def cem(level: Level, task: Task, state, key, cfg: ProbeConfig, ticks: int, topk: int = 0):
    """Cross-entropy search for control segments, scored inside the `topk` model."""
    shape = (cfg.segments, 2)
    pcfg = cfg.plan_cfg()
    mu0 = pilot_seed_from(level, task, state, ticks, cfg.segments, topk)

    def run(seg):
        final, gap, *_ = fly_from(level, task, state, expand(seg, ticks), topk)
        return score(task, final, gap, pcfg)

    def it(carry, k):
        mu, sigma, best, best_score = carry
        s = mu[None] + sigma[None] * jax.random.normal(k, (cfg.samples,) + shape)
        s = s.at[0].set(mu).at[1].set(best)
        sc = jax.vmap(run)(s)
        order = jnp.argsort(sc)
        elite = s[order[: cfg.elites]]
        better = sc[order[0]] < best_score
        return (elite.mean(0), jnp.maximum(elite.std(0), cfg.sigma_floor),
                jnp.where(better, s[order[0]], best),
                jnp.where(better, sc[order[0]], best_score)), None

    init = (mu0, jnp.full(shape, cfg.init_sigma), mu0, run(mu0))
    (_, _, best, _), _ = jax.lax.scan(it, init, jax.random.split(key, cfg.iters))
    return best


def fly_model(level: Level, task: Task, key, cfg: ProbeConfig, topk: int = 0):
    """Plan under the `topk` model, fly in the **true** world, replan mid-course.

    This is the honest test of a coarse model: the pilot may re-plan from where
    they actually are (`cfg.replans` times, receding horizon), so a failure is a
    failure of the model, not of open-loop luck.  Returns the true-world outcome
    and the actions actually flown.
    """
    chunk = cfg.ticks // (cfg.replans + 1)
    state = E.reset(level, task)
    keys = jax.random.split(key, cfg.replans + 1)
    flown, first = [], None
    for r in range(cfg.replans + 1):
        left = cfg.ticks - r * chunk
        seg = cem(level, task, state, keys[r], cfg, left, topk)
        first = seg if first is None else first
        acts = expand(seg, left)
        take = acts[:chunk] if r < cfg.replans else acts
        state, *_ = fly_from(level, task, state, take, 0)        # fly the TRUE world
        flown.append(take)
    actions = jnp.concatenate(flown, axis=0)
    arrived = state.status == ARRIVED
    return dict(arrived=arrived, status=state.status,
                cost=jnp.where(arrived, E.objective_cost(task, state.costs), jnp.inf),
                time=state.costs.time, fuel=state.costs.fuel, damage=state.costs.damage,
                actions=actions, seg0=first)


# ---------------------------------------------------------------------------
# precision probes: given the right plan, how much slack is there?
# ---------------------------------------------------------------------------

def delay_window(level: Level, task: Task, actions, delays):
    """Arrival after starting the same plan `d` ticks late, for each d.

    A level whose only transfer sits in a narrow synodic window fails here while
    being perfectly smooth otherwise: the plan is right and the date is wrong.
    """
    n = actions.shape[0]
    state = E.reset(level, task)

    def one(d):
        shifted = jnp.where((jnp.arange(n) >= d)[:, None],
                            jnp.roll(actions, d, axis=0), jnp.zeros_like(actions))
        final, *_ = fly_from(level, task, state, shifted, 0)
        return final.status == ARRIVED

    # one vmapped sweep, not one graph per delay: the shapes are identical and the
    # whole probe compiles for several minutes otherwise
    return jax.vmap(one)(jnp.asarray(delays, jnp.int32))


def jitter_rate(level: Level, task: Task, seg, key, cfg: ProbeConfig):
    """Success rate per jitter level when each burn's onset slips, in control ticks.

    Per-segment onset jitter is the standard model of human motor timing: the
    pilot knows the plan and starts each burn a little early or late.  Swept over
    `cfg.jitters` in one vmap, which keeps the compile to a single graph.
    """
    per = cfg.ticks // cfg.segments
    edges = jnp.arange(cfg.segments) * per
    t = jnp.arange(cfg.ticks)
    state = E.reset(level, task)

    def one(k, sd_ticks):
        eps = sd_ticks * jax.random.normal(k, (cfg.segments,))
        b = jnp.sort(jnp.clip(edges + eps, 0.0, cfg.ticks - 1.0))
        idx = jnp.clip(jnp.sum(b[None, :] <= t[:, None], axis=1) - 1, 0, cfg.segments - 1)
        a = seg[idx]
        acts = jnp.stack([jnp.tanh(a[:, 0]), jax.nn.sigmoid(a[:, 1])], axis=-1)
        final, *_ = fly_from(level, task, state, acts, 0)
        return final.status == ARRIVED

    keys = jax.random.split(key, cfg.jitter_samples)
    sds = jnp.asarray(cfg.jitters, jnp.float32)
    grid = jax.vmap(lambda sd: jax.vmap(lambda k: one(k, sd))(keys))(sds)
    return jnp.mean(grid, axis=1)


def lyapunov(level: Level, task: Task, actions, eps=1e-4):
    """Divergence rate of the flown trajectory: ln(|dx_end|/|dx_0|) / flight time.

    How fast the flow eats precision.  A horizon H = lambda * T much above 1 means
    no amount of up-front planning survives to the end without correction.
    """
    state = E.reset(level, task)
    nudged = state._replace(ship=state.ship._replace(pos=state.ship.pos + jnp.array([eps, 0.0])))
    a, *_ = fly_from(level, task, state, actions, 0)
    b, *_ = fly_from(level, task, nudged, actions, 0)
    sep = jnp.sqrt(jnp.sum((a.ship.pos - b.ship.pos) ** 2) + 1e-18)
    t = jnp.maximum(a.costs.time, C.CTRL_DT)
    return jnp.log(sep / eps) / t


def incoherence(level: Level, task: Task, pos, alive):
    """Fraction of the gravitational force a dominant-body model discards.

    eta = |a_full - a_dominant| / |a_full| along the flight.  Closed-form, one
    pass, and the cheap predictor of whether a patched-conic plan can work: a
    two-body arc of duration T accumulates a position error ~ eta*|a|*T^2/2.
    """
    steps = pos.shape[0] * C.SUBSTEPS
    bp, _ = P.roll_bodies(level, *E.snapshot_state(level, task.snapshot), steps)
    bp = bp[:: C.SUBSTEPS][: pos.shape[0]]                        # (T, N, 2)
    full = jax.vmap(lambda x, b: P.gravity_at(x[None], b, level, 0)[0])(pos, bp)
    dom = jax.vmap(lambda x, b: P.gravity_at(x[None], b, level, 1)[0])(pos, bp)
    eta = jnp.sqrt(jnp.sum((full - dom) ** 2, -1) + 1e-18) / \
        jnp.sqrt(jnp.sum(full ** 2, -1) + 1e-12)
    # only the ticks actually flown: after a flight ends the ship is parked *at the
    # waypoint*, which on a contested level is the highest-eta point in the level,
    # so averaging the frozen tail in would inflate exactly the cases under test
    w = alive.astype(jnp.float32)
    n = jnp.maximum(jnp.sum(w), 1.0)
    return (jnp.sum(eta * w) / n, jnp.max(jnp.where(alive, eta, -jnp.inf)),
            jnp.sum(eta * w) * C.CTRL_DT)


# ---------------------------------------------------------------------------
# the whole probe
# ---------------------------------------------------------------------------

def thrust_authority(level: Level, task: Task, pos, alive):
    """Engine acceleration over local gravity, median along the flight.

    The quantity that decides whether modelling the field accurately matters at
    all.  A ship with 30x the local gravity can overpower any mistake its model
    makes, so a coarse model costs it nothing; near 1 the field shapes the
    trajectory and a wrong model sends it somewhere else.  Chasing eta without
    checking this measures a force the ship does not care about.
    """
    steps = pos.shape[0] * C.SUBSTEPS
    bp, _ = P.roll_bodies(level, *E.snapshot_state(level, task.snapshot), steps)
    bp = bp[:: C.SUBSTEPS][: pos.shape[0]]
    g = jax.vmap(lambda x, b: P.gravity_at(x[None], b, level, 0)[0])(pos, bp)
    mag = jnp.sqrt(jnp.sum(g * g, -1) + 1e-12)
    # nanmedian, not median: the dead ticks after the flight ends are masked out
    # with NaN, and jnp.median *propagates* NaN, so a plain median returned NaN
    # for every flight that arrived or died before the horizon -- which is nearly
    # all of them.  Only the live ticks were ever meant to count.
    return jnp.nanmedian(jnp.where(alive, task.accel / mag, jnp.nan))


def scale_crossings(level: Level, task: Task, pos, alive):
    """How many bodies' Hill spheres the flight enters, and the deepest entry.

    A flight that stays in one body's domain is single-scale; one that leaves the
    star's domain, enters a planet's and then a moon's is three-scale, and that
    count is what a curriculum meant to bootstrap scales should ramp.
    """
    steps = pos.shape[0] * C.SUBSTEPS
    bp, _ = P.roll_bodies(level, *E.snapshot_state(level, task.snapshot), steps)
    bp = bp[:: C.SUBSTEPS][: pos.shape[0]]                        # (T, N, 2)
    has, _, hill, _ = E.contested_zone(level, bp[0])
    d = jnp.sqrt(jnp.sum((pos[:, None, :] - bp) ** 2, -1) + 1e-9)  # (T, N)
    ratio = d / jnp.maximum(hill[None, :], 1e-6)
    inside = has[None, :] & (ratio < 1.0) & alive[:, None]
    seen = has[None, :] & alive[:, None]
    return jnp.sum(jnp.any(inside, axis=0)), jnp.min(jnp.where(seen, ratio, jnp.inf))


def probe(level: Level, task: Task, key, cfg: ProbeConfig = ProbeConfig()):
    """Place one (level, task) on both axes.  Everything a band decision needs."""
    keys = jax.random.split(key, 8)
    runs = {k: fly_model(level, task, keys[i], cfg, k) for i, k in enumerate(cfg.ladder)}
    fine = runs[0]

    # the reference plan for the precision probes: full knowledge, no mid-course
    # corrections — reuse the full-model run's opening plan rather than searching again
    seg = fine["seg0"]
    acts = expand(seg, cfg.ticks)
    ref, gap, pos, alive = fly_from(level, task, E.reset(level, task), acts, 0)
    ref_ok = ref.status == ARRIVED

    delays = delay_window(level, task, acts, cfg.delays)
    jit = jitter_rate(level, task, seg, keys[6], cfg)
    lam = lyapunov(level, task, acts)
    eta_mean, eta_max, eta_int = incoherence(level, task, pos, alive)
    n_scales, hill_depth = scale_crossings(level, task, pos, alive)
    authority = thrust_authority(level, task, pos, alive)

    # the reflex policy, and the reflex policy through a human channel
    pilot_final, _ = E.rollout(level, task, baselines.pilot(level, task), n_ticks=cfg.ticks)

    out = dict(
        ref_ok=ref_ok, ref_gap=gap, ref_cost=jnp.where(ref_ok, E.objective_cost(task, ref.costs),
                                                       jnp.inf),
        pilot_ok=pilot_final.status == ARRIVED,
        delays=delays, jitter=jit, lam=lam, horizon=lam * jnp.maximum(ref.costs.time, 1.0),
        eta_mean=eta_mean, eta_max=eta_max, eta_int=eta_int,
        n_scales=n_scales, hill_depth=hill_depth, authority=authority,
    )
    for k, r in runs.items():
        tag = "full" if k == 0 else f"k{k}"
        out[f"ok_{tag}"] = r["arrived"]
        out[f"cost_{tag}"] = r["cost"]
    return out


def probe_batch(levels, tasks, key, cfg: ProbeConfig = ProbeConfig()):
    n = jax.tree.leaves(tasks)[0].shape[0]
    return jax.vmap(lambda l, t, k: probe(l, t, k, cfg))(levels, tasks, jax.random.split(key, n))
