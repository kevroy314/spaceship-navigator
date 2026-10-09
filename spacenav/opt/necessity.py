"""Which decisions a level actually forces, and which it merely permits.

Two different questions, and level design lives in the gap between them:

* **Necessity** — is there a decision that *must* be made a particular way?
  Measured by intervention: take a plan that works, randomise the control on one
  segment while holding every other segment fixed, and see how often it still
  arrives.  That is a `do()` on a single decision, and the drop in success is
  that decision's causal effect.  A profile with one tall spike is a level with
  one irreplaceable commitment; a flat profile is a level that forgives
  everything individually.

* **Sufficiency** — how many different ways are there to win?  Measured by
  sweeping a whole family of strategies rather than perturbing one plan.

For the sweep the family is deliberately the simplest one a player can describe:
**coast, turn to a heading, burn once.**  Three numbers — when, which way, how
much delta-v — so the entire decision space is a 3-D volume that can be
enumerated instead of searched.  The shape of the winning set inside that volume
is the level's causal map:

* a broad blob      -> many sufficient options, forgiving
* a thin sliver     -> one necessary action, nothing else works
* several blobs     -> genuinely distinct strategies, and the gaps between them
                       are the decisions that commit you to one
* a diagonal stripe -> *leading*: burn later and you must aim further ahead.
                       The correlation between burn time and heading inside the
                       winning set measures it directly.

Position and velocity are not separate axes because they are not free: the burn
time fixes where the ship is and how fast, so "when" carries them.  Sliced along
delta-v, the volume shows how regions appear, widen and merge as the tank grows.

**A warning about the sweep.**  An n-body flight is chaotic scattering, and the
basins of a chaotic scattering problem have *fractal* boundaries.  Where that
holds, a grid does not sample the map, it aliases it: between two winning cells
can lie a dense set of losing ones, and in the riddled case every neighbourhood
of a winner contains losers.  Crisp "decision regions" would then be an artefact
of the grid spacing and nothing else.

So the volume is not reported as a set of regions.  Each cell holds the *win
fraction* over a small jittered neighbourhood, which is well defined whatever the
boundary looks like and converges with sampling instead of with grid refinement;
and `boundary_exponent` measures how resolvable the boundary is at all, via the
uncertainty exponent of Grebogi, McDonald, Ott and Yorke: perturb a parameter
point by eps and the chance the outcome flips scales as f(eps) ~ eps^alpha.
alpha near 1 is a smooth, codimension-one boundary, where knowing the parameters
better pays off proportionally.  alpha near 0 is a fractal boundary, where it
buys almost nothing and only ensemble statements are meaningful.

That exponent is *not* by itself a reason to throw a level away.  A fractal
region is a resource: it noises the flight and forces the pilot to adapt
downstream, which is precisely what defeats a memorised open-loop plan.  What
matters is **where** the chaos sits relative to the slack:

* chaos upstream of spare fuel and time -> the flight is knocked off course and
  re-planning recovers it.  That is a skill, and a closed-loop one, which is the
  only kind that transfers to a moving opponent.
* chaos at the moment of commitment, with nothing left to correct with -> the
  pilot chose correctly and lost anyway.  That is a lottery.

`adaptivity` separates them by measuring whether re-planning rescues what an
open-loop plan loses.  So "learnable" versus "entropic" is not alpha alone; it is
alpha together with how much adaptation buys.
"""

from typing import NamedTuple

import jax
import jax.numpy as jnp

from spacenav import constants as C
from spacenav import env as E
from spacenav.opt.planner import expand
from spacenav.opt.probe import ProbeConfig, cem, fly_from
from spacenav.types import ARRIVED, Level, Task


class NeedConfig(NamedTuple):
    ticks: int = C.MAX_EPISODE_TICKS
    segments: int = 48
    resamples: int = 8            # random replacements per segment, for the intervention
    sigma: float = 1.2            # prior width the replacement is drawn from
    # the strategy volume: when to burn x which way x how much
    n_time: int = 28
    n_dir: int = 24
    n_dv: int = 6
    dv_lo: float = 0.15           # fraction of the tank spent on the single burn
    dv_hi: float = 1.0
    start_frac: float = 0.0       # earliest burn, as a fraction of the episode
    end_frac: float = 0.75        # latest burn: leave time to arrive afterwards
    cell_samples: int = 3         # jittered draws per cell; >1 turns the volume
                                  # from a binary map into a probability field, which
                                  # is what lets a cell wider than the arrival
                                  # tolerance still report the fraction of itself
                                  # that wins instead of rounding to zero
    dir_span: float = 2.1         # radians swept, centred on the bearing to the
                                  # target.  A full 360 degree sweep at 24 steps is
                                  # 15 degrees a cell against a ~3 degree arrival
                                  # tolerance, so it misses solutions rather than
                                  # finding them; the burn is going to be roughly
                                  # toward the target, so spend the resolution there
    # uncertainty exponent: how far apart two parameter points are, as a fraction
    # of the whole parameter box, when asking whether their outcomes differ
    eps: tuple = (0.002, 0.005, 0.012, 0.03, 0.07)
    # 96 gave flip rates that were tiny integer counts (0/1/3/5/5), so every
    # fitted alpha had a bootstrap CI straddling the resolvable threshold and none
    # was trustworthy (docs/decisions/0009). This is a parallelism problem, not a
    # speed one.
    eps_samples: int = 768


# ---------------------------------------------------------------------------
# necessity: intervene on one decision at a time
# ---------------------------------------------------------------------------

def ace_profile(level: Level, task: Task, seg, key, cfg: NeedConfig = NeedConfig()):
    """Per-segment causal effect: 1 - P(arrive | that one decision randomised).

    `seg` must be a plan that arrives, otherwise the baseline is meaningless and
    every effect reads as zero.
    """
    def one(k, i):
        draw = cfg.sigma * jax.random.normal(k, (2,))
        trial = seg.at[i].set(draw)
        final, *_ = fly_from(level, task, E.reset(level, task), expand(trial, cfg.ticks), 0)
        return (final.status == ARRIVED).astype(jnp.float32)

    keys = jax.random.split(key, cfg.segments * cfg.resamples).reshape(
        cfg.segments, cfg.resamples, 2)
    held = jax.vmap(lambda i, ks: jnp.mean(jax.vmap(lambda k: one(k, i))(ks)))(
        jnp.arange(cfg.segments), keys)
    return 1.0 - held                      # (segments,) causal effect of each decision


def necessity_scalars(ace):
    """Summarise a profile: how much it matters, and how concentrated it is."""
    total = jnp.sum(ace)
    p = ace / jnp.maximum(total, 1e-6)
    n = ace.shape[0]
    # normalised entropy: 0 = one decision carries everything, 1 = perfectly diffuse
    ent = -jnp.sum(jnp.where(p > 0, p * jnp.log(jnp.maximum(p, 1e-12)), 0.0)) / jnp.log(n)
    return dict(need_max=jnp.max(ace), need_mean=jnp.mean(ace), need_total=total,
                concentration=1.0 - ent, n_critical=jnp.sum(ace > 0.5))


# ---------------------------------------------------------------------------
# sufficiency: enumerate the single-burn family
# ---------------------------------------------------------------------------

def _bearing(level: Level, task: Task):
    """Direction from the start to the first target, as the sweep's centre."""
    st = E.reset(level, task)
    tpos, _ = E.target_state(task, st)
    d = tpos - task.start_pos
    return jnp.arctan2(d[1], d[0])


def _dir_axis(level: Level, task: Task, cfg):
    c = _bearing(level, task)
    return c + jnp.linspace(-cfg.dir_span / 2, cfg.dir_span / 2, cfg.n_dir)


def one_burn_actions(task: Task, t_start: int, heading, dv_frac, ticks: int):
    """Coast, turn to `heading`, burn once, coast again.

    Turning costs time but no fuel, so the ship rotates during the coast before
    the burn.  If there is not enough time to come round, it turns as far as it
    can -- which is itself a real constraint the volume should show.
    """
    t = jnp.arange(ticks)
    err = (heading - task.start_angle + jnp.pi) % (2 * jnp.pi) - jnp.pi
    turn_ticks = jnp.abs(err) / (C.SHIP_TURN_RATE * C.CTRL_DT)
    turning = t < jnp.minimum(turn_ticks, t_start)
    burn_ticks = dv_frac * task.fuel / C.CTRL_DT
    burning = (t >= t_start) & (t < t_start + burn_ticks)
    return jnp.stack([jnp.where(turning, jnp.sign(err), 0.0),
                      jnp.where(burning, 1.0, 0.0)], axis=-1)


def two_burn_actions(task: Task, t1, h1, dv1, t2, h2, dv2, ticks: int):
    """Coast, turn, burn, coast, turn again, burn again, coast.

    The second heading is reachable analytically because the ship is already
    holding `h1` when the first burn ends, so the turn between burns is a known
    arc.  Turning costs time but no fuel.
    """
    t = jnp.arange(ticks)
    rate = C.SHIP_TURN_RATE * C.CTRL_DT

    e1 = (h1 - task.start_angle + jnp.pi) % (2 * jnp.pi) - jnp.pi
    turn1 = t < jnp.minimum(jnp.abs(e1) / rate, t1)
    n1 = dv1 * task.fuel / C.CTRL_DT
    burn1 = (t >= t1) & (t < t1 + n1)

    e2 = (h2 - h1 + jnp.pi) % (2 * jnp.pi) - jnp.pi
    s2 = t1 + n1                                   # the ship is holding h1 from here
    turn2 = (t >= s2) & (t < s2 + jnp.minimum(jnp.abs(e2) / rate,
                                              jnp.maximum(t2 - s2, 0.0)))
    n2 = dv2 * task.fuel / C.CTRL_DT
    burn2 = (t >= t2) & (t < t2 + n2)

    turn = jnp.where(turn1, jnp.sign(e1), jnp.where(turn2, jnp.sign(e2), 0.0))
    return jnp.stack([turn, jnp.where(burn1 | burn2, 1.0, 0.0)], axis=-1)


def two_burn_sufficiency(level: Level, task: Task, key, cfg: NeedConfig = NeedConfig(),
                         samples: int = 4096):
    """How often a *two*-burn plan wins, by Monte Carlo over its six parameters.

    The one-burn family wins in under 0.3% of its own parameter box on most
    levels, which left three of five causal maps effectively blank — they were
    reporting the limits of the probe, not the shape of the level
    (docs/decisions/0019). Two burns is six parameters, so a grid is out; a
    sampled win fraction is the honest estimate and costs one rollout per draw.

    Returned alongside the one-burn fraction so the two are comparable: if the
    gap is large, the family was the constraint.
    """
    lo = _bearing(level, task) - cfg.dir_span / 2
    st = E.reset(level, task)

    def one(k):
        u = jax.random.uniform(k, (6,))
        t1 = u[0] * 0.5 * cfg.ticks
        dv1 = cfg.dv_lo + u[2] * (cfg.dv_hi - cfg.dv_lo) * 0.6
        # the second burn starts after the first ends, and leaves room to arrive
        s2 = t1 + dv1 * task.fuel / C.CTRL_DT
        t2 = s2 + u[3] * jnp.maximum(cfg.end_frac * cfg.ticks - s2, 1.0)
        dv2 = cfg.dv_lo + u[5] * jnp.maximum(cfg.dv_hi - dv1 - cfg.dv_lo, 0.0)
        acts = two_burn_actions(task, t1.astype(jnp.int32), lo + u[1] * cfg.dir_span, dv1,
                                t2.astype(jnp.int32), lo + u[4] * cfg.dir_span, dv2,
                                cfg.ticks)
        final, *_ = fly_from(level, task, st, acts, 0)
        return final.status == ARRIVED

    wins = jax.lax.map(lambda k: one(k), jax.random.split(key, samples))
    return jnp.mean(wins.astype(jnp.float32))


def strategy_volume(level: Level, task: Task, cfg: NeedConfig = NeedConfig(), key=None):
    """Win fraction for single burns over (when, which way, how much delta-v).

    With `cell_samples` above 1 each cell is averaged over jittered draws inside
    its own footprint, so the result is a probability field rather than a binary
    map.  That is the honest object when the boundary may be fractal: a single
    sample per cell would report whichever side of a fine-grained boundary it
    happened to land on.

    Returned with its axes, so a page can slice it and label the slices in the
    units a player thinks in: seconds, degrees, and fraction of the tank.
    """
    times = jnp.linspace(cfg.start_frac * cfg.ticks, cfg.end_frac * cfg.ticks,
                         cfg.n_time).astype(jnp.int32)
    dirs = _dir_axis(level, task, cfg)
    dvs = jnp.linspace(cfg.dv_lo, cfg.dv_hi, cfg.n_dv)
    st = E.reset(level, task)

    def win(t_start, heading, dv):
        acts = one_burn_actions(task, t_start, heading, dv, cfg.ticks)
        final, gap, *_ = fly_from(level, task, st, acts, 0)
        return (final.status == ARRIVED), gap

    # cell footprints, so a jittered draw stays inside the cell it reports
    dt = (times[1] - times[0]).astype(jnp.float32) if cfg.n_time > 1 else 1.0
    dd = (dirs[1] - dirs[0]) if cfg.n_dir > 1 else 0.0
    dvv = (dvs[1] - dvs[0]) if cfg.n_dv > 1 else 0.0
    key = jax.random.PRNGKey(0) if key is None else key

    def cell(t_start, heading, dv, k):
        """Mean outcome over `cell_samples` draws inside this cell."""
        def draw(kk):
            u = jax.random.uniform(kk, (3,), minval=-0.5, maxval=0.5)
            jt = jnp.where(cfg.cell_samples > 1, u[0] * dt, 0.0)
            jd = jnp.where(cfg.cell_samples > 1, u[1] * dd, 0.0)
            jv = jnp.where(cfg.cell_samples > 1, u[2] * dvv, 0.0)
            t = jnp.clip(t_start.astype(jnp.float32) + jt, 0.0,
                         cfg.ticks - 1.0).astype(jnp.int32)
            ok, gap = win(t, heading + jd, jnp.clip(dv + jv, 0.01, 1.0))
            return ok.astype(jnp.float32), gap
        oks, gaps = jax.vmap(draw)(jax.random.split(k, cfg.cell_samples))
        return jnp.mean(oks), jnp.min(gaps)

    # vmap over directions and dv inside, scan over time outside: keeps the live
    # working set small enough for a 11 GB card at 2250 ticks
    def row(k, t_start):
        k1, k2 = jax.random.split(k)
        ks = jax.random.split(k1, cfg.n_dir * cfg.n_dv).reshape(cfg.n_dir, cfg.n_dv, 2)
        p, gap = jax.vmap(lambda h, kr: jax.vmap(lambda d, kk: cell(t_start, h, d, kk))(dvs, kr))(
            dirs, ks)
        return k2, (p, gap)

    _, (p, gap) = jax.lax.scan(row, key, times)
    return dict(win=p, gap=gap, times=times, dirs=dirs, dvs=dvs)   # (n_time, n_dir, n_dv)


def adaptivity(level: Level, task: Task, key, cfg: NeedConfig = NeedConfig(), replans: int = 2):
    """How much re-planning buys over committing to one plan.

    Both pilots see the full field and get the same search budget.  One commits
    at t=0; the other flies a third of the way, looks at where it actually is,
    and plans again.  On a level whose chaos sits upstream of its slack the
    second arrives and the first does not, and the difference is the value of
    adapting -- the thing a fractal region is *for*.

    Returns both outcomes plus the gap, so a level can be called forgiving
    (adaptation recovers it) or a lottery (nothing recovers it).
    """
    pcfg = ProbeConfig(ticks=cfg.ticks, segments=cfg.segments)
    st = E.reset(level, task)
    keys = jax.random.split(key, replans + 2)

    seg = cem(level, task, st, keys[0], pcfg, cfg.ticks, 0)
    open_final, *_ = fly_from(level, task, st, expand(seg, cfg.ticks), 0)

    chunk = cfg.ticks // (replans + 1)
    state, cur = st, seg
    for r in range(replans + 1):
        left = cfg.ticks - r * chunk
        if r:
            cur = cem(level, task, state, keys[r], pcfg, left, 0)
        acts = expand(cur, left)
        state, *_ = fly_from(level, task, state, acts[:chunk] if r < replans else acts, 0)

    open_ok = open_final.status == ARRIVED
    mpc_ok = state.status == ARRIVED
    return dict(open_loop=open_ok, replanned=mpc_ok,
                recovers=mpc_ok & ~open_ok, adaptation_gain=
                mpc_ok.astype(jnp.float32) - open_ok.astype(jnp.float32))


def boundary_exponent(level: Level, task: Task, key, cfg: NeedConfig = NeedConfig()):
    """Uncertainty exponent of the win/lose boundary: f(eps) ~ eps^alpha.

    Draw parameter points uniformly from the single-burn box, perturb each by a
    fraction `eps` of the box in a random direction, and record how often the
    outcome flips.  Fitting log f against log eps gives alpha.

    alpha near 1: a smooth boundary of codimension one.  Knowing the parameters
    twice as well halves the chance of being wrong, so precision is a skill and
    an exhaustive map at fine enough spacing is meaningful.

    alpha near 0: a fractal boundary.  Precision buys almost nothing, forward
    prediction is probabilistic however good the model, and only statements about
    ensembles survive.  Reported alongside the volume so no one reads crisp
    regions into a map that cannot have them.
    """
    t_lo, t_hi = cfg.start_frac * cfg.ticks, cfg.end_frac * cfg.ticks
    d_lo = _bearing(level, task) - cfg.dir_span / 2
    st = E.reset(level, task)

    def outcome(u):
        """u in [0,1]^3 -> did this single burn arrive?"""
        t = jnp.clip(t_lo + u[0] * (t_hi - t_lo), 0.0, cfg.ticks - 1.0).astype(jnp.int32)
        acts = one_burn_actions(task, t, d_lo + u[1] * cfg.dir_span,
                                cfg.dv_lo + u[2] * (cfg.dv_hi - cfg.dv_lo), cfg.ticks)
        final, *_ = fly_from(level, task, st, acts, 0)
        return final.status == ARRIVED

    def flip_rate(eps, k):
        k1, k2 = jax.random.split(k)
        base = jax.random.uniform(k1, (cfg.eps_samples, 3))
        step = jax.random.normal(k2, (cfg.eps_samples, 3))
        step = step / jnp.maximum(jnp.linalg.norm(step, axis=-1, keepdims=True), 1e-9)
        other = jnp.clip(base + eps * step, 0.0, 1.0)
        a = jax.vmap(outcome)(base)
        b = jax.vmap(outcome)(other)
        return jnp.mean(a != b), jnp.mean(a)

    keys = jax.random.split(key, len(cfg.eps))
    rates, wins = zip(*[flip_rate(e, k) for e, k in zip(cfg.eps, keys)])
    f = jnp.stack(rates)
    e = jnp.asarray(cfg.eps)
    # An exponent fitted to an empty winning set is not a measurement of
    # fractality, it is a measurement of nothing: with almost no wins there are
    # almost no flips, and the slope comes out near zero whatever the boundary
    # is really like.  `measurable` says whether to believe it.
    win_rate = jnp.mean(jnp.stack(wins))
    measurable = (win_rate > 0.02) & (jnp.sum(f > 0) >= 2)
    # least squares on the log-log slope, over the eps where anything flipped
    good = f > 0
    lx, ly = jnp.log(e), jnp.log(jnp.maximum(f, 1e-6))
    w = good.astype(jnp.float32)
    n = jnp.maximum(jnp.sum(w), 1.0)
    mx, my = jnp.sum(w * lx) / n, jnp.sum(w * ly) / n
    alpha = jnp.sum(w * (lx - mx) * (ly - my)) / jnp.maximum(jnp.sum(w * (lx - mx) ** 2), 1e-9)
    return dict(alpha=alpha, flip=f, eps=e, win_rate=win_rate, measurable=measurable)


def volume_scalars(vol):
    """Sufficiency, and the signature of leading.

    `leading` is the correlation between burn time and burn heading across the
    winning set: a diagonal stripe means a later burn must be aimed further
    ahead, which is exactly what leading a moving target is.
    """
    win = vol["win"]
    n_t, n_d, n_v = win.shape
    frac = jnp.mean(win.astype(jnp.float32))

    ti = jnp.arange(n_t)[:, None, None] * jnp.ones_like(win, jnp.float32)
    # headings are circular: correlate against the unwrapped angle nearest the
    # winning set's mean direction, so a stripe crossing 0 degrees still reads
    ang = jnp.arange(n_d)[None, :, None] * jnp.ones_like(win, jnp.float32)
    w = win.astype(jnp.float32)
    tot = jnp.maximum(jnp.sum(w), 1.0)
    mt, ma = jnp.sum(w * ti) / tot, jnp.sum(w * ang) / tot
    dt, da = ti - mt, ((ang - ma + n_d / 2) % n_d) - n_d / 2
    cov = jnp.sum(w * dt * da) / tot
    sd = jnp.sqrt(jnp.sum(w * dt ** 2) / tot) * jnp.sqrt(jnp.sum(w * da ** 2) / tot)
    return dict(sufficiency=frac, leading=cov / jnp.maximum(sd, 1e-6),
                win_any=jnp.any(win),
                # how much of the tank the cheapest winning burn needs
                min_dv=jnp.min(jnp.where(jnp.any(win, axis=(0, 1)), vol["dvs"], jnp.inf)),
                # widest contiguous run of winning burn times, in seconds: the
                # launch window a player would actually have to hit
                window_s=jnp.max(jnp.sum(jnp.any(win, axis=2).astype(jnp.int32), axis=0))
                * (vol["times"][1] - vol["times"][0]) * C.CTRL_DT)
