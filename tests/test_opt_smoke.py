"""Smoke tests for the measurement modules: `opt.probe`, `opt.routes`, `opt.necessity`.

These are the expensive parts of the pipeline (cross-entropy search over the whole
episode clock), so every config here is shrunk to the smallest shape that still
exercises the code path, and the pieces are compiled **separately**: wrapping all
of `probe()` in one jit costs many minutes of XLA time on a single graph, which
is not a test, it is a build.

They assert *structure* -- the keys a report consumer reads, finiteness, and the
invariants that hold regardless of outcome -- not which route wins, which depends
on the search budget they have been denied.
"""

import jax
import jax.numpy as jnp
import numpy as np
import pytest

from spacenav import constants as C
from spacenav import env as E
from spacenav.opt import necessity as N
from spacenav.opt import probe as PR
from spacenav.opt import routes as R
from spacenav.opt.planner import expand
from helpers import hierarchy_level

TICKS = 60          # 4 s: enough to fly somewhere, cheap enough to compile
SEG = 6
TINY = PR.ProbeConfig(ticks=TICKS, segments=SEG, samples=16, elites=4, iters=2)


@pytest.fixture(scope="module")
def world():
    lvl = hierarchy_level()
    cfg = E.TaskConfig(deltav_budget=(1.0, 1.4))
    sample = jax.jit(E.sample_task, static_argnums=2)
    for seed in range(12):          # task sampling can reject; don't bet on one seed
        task, ok = sample(jax.random.PRNGKey(seed), lvl, cfg)
        if bool(ok):
            return lvl, task
    pytest.fail("no valid task sampled on the hierarchy level in 12 tries")


@pytest.fixture(scope="module")
def flight(world):
    """One coasting flight, reused by the cheap probes so each compiles once."""
    lvl, task = world
    st = E.reset(lvl, task)
    acts = jnp.zeros((TICKS, 2))
    return (lvl, task, acts) + tuple(jax.jit(lambda: PR.fly_from(lvl, task, st, acts, 0))())


# ---------------------------------------------------------------------------
# opt.probe
# ---------------------------------------------------------------------------

def test_fly_from_reports_a_finite_gap(flight):
    lvl, task, acts, final, gap, pos, alive = flight
    assert pos.shape == (TICKS, 2)
    assert alive.shape == (TICKS,)
    assert np.isfinite(float(gap))
    assert bool(alive[0])
    # topk=1 is a different world for the ship but not for the bodies
    st = E.reset(lvl, task)
    k1, *_ = jax.jit(lambda: PR.fly_from(lvl, task, st, acts, 1))()
    np.testing.assert_allclose(np.asarray(k1.body_pos), np.asarray(final.body_pos), atol=1e-4)


def test_cem_returns_usable_segments(world):
    lvl, task = world
    seg = jax.jit(lambda k: PR.cem(lvl, task, E.reset(lvl, task), k, TINY, TICKS, 0))(
        jax.random.PRNGKey(0))
    assert seg.shape == (SEG, 2)
    assert np.all(np.isfinite(np.asarray(seg)))
    acts = np.asarray(expand(seg, TICKS))
    assert acts.shape == (TICKS, 2)
    assert np.all((acts[:, 0] >= -1) & (acts[:, 0] <= 1))
    assert np.all((acts[:, 1] >= 0) & (acts[:, 1] <= 1))


def test_incoherence_is_positive_and_ignores_the_frozen_tail(world):
    """eta = |discarded| / |total| is positive and finite but deliberately NOT
    capped at 1: where the full field nearly cancels, a one-body model is wrong
    by more than the whole resultant, and that is a real reading.

    Averaging the parked ship in would inflate eta on exactly the levels under
    test, so only live ticks count: a flight alive for one tick must report that
    tick's eta, not the 59 it spent frozen.
    """
    lvl, task = world
    st = E.reset(lvl, task)
    pos = jnp.tile(st.ship.pos, (TICKS, 1))
    alive = jnp.arange(TICKS) < 1
    mean, mx, integral = jax.jit(lambda: PR.incoherence(lvl, task, pos, alive))()
    assert float(mean) > 0.0 and np.isfinite(float(mean))
    assert float(mean) == pytest.approx(float(mx), rel=1e-5)
    assert float(integral) == pytest.approx(float(mean) * C.CTRL_DT, rel=1e-5)


def test_thrust_authority_ignores_the_dead_ticks(world):
    """Regression: `thrust_authority` masks the post-flight ticks with NaN, so it
    must use nanmedian.  A plain median propagates the NaN and reports NaN for
    every flight that ended before the horizon, which is nearly all of them."""
    lvl, task = world
    st = E.reset(lvl, task)
    pos = jnp.tile(st.ship.pos, (TICKS, 1))
    f = jax.jit(lambda a: PR.thrust_authority(lvl, task, pos, a))
    full = f(jnp.ones(TICKS, bool))
    one = f(jnp.arange(TICKS) < 1)
    assert np.isfinite(float(one)), "masked-out ticks must not poison the median"
    assert float(one) > 0.0
    # the ship is parked and 4 s is a blink on these orbits, so the field it sees
    # barely moves and the two medians land in the same place
    assert float(one) == pytest.approx(float(full), rel=0.1)


def test_lyapunov_and_scale_crossings(flight):
    lvl, task, acts, final, gap, pos, alive = flight
    lam = jax.jit(lambda: PR.lyapunov(lvl, task, acts))()
    assert np.isfinite(float(lam))
    n_scales, depth = jax.jit(lambda: PR.scale_crossings(lvl, task, pos, alive))()
    assert 0 <= int(n_scales) <= int(np.sum(np.asarray(lvl.active)))
    assert float(depth) > 0.0          # a ratio of distance to Hill radius


def test_delay_window_and_jitter_rate(world):
    lvl, task = world
    cfg = TINY._replace(delays=(1, 3), jitters=(0.5, 1.0), jitter_samples=4)
    acts = jnp.zeros((TICKS, 2))
    ok = jax.jit(lambda: PR.delay_window(lvl, task, acts, cfg.delays))()
    assert np.asarray(ok).shape == (2,)
    assert np.asarray(ok).dtype == bool
    rate = jax.jit(lambda k: PR.jitter_rate(lvl, task, jnp.zeros((SEG, 2)), k, cfg))(
        jax.random.PRNGKey(1))
    r = np.asarray(rate)
    assert r.shape == (2,)
    assert np.all((r >= 0.0) & (r <= 1.0))


def test_probe_reports_both_axes(world):
    """The integration check, at the smallest shape that still runs: every key a
    `curriculum.band` decision reads must be present and finite."""
    lvl, task = world
    cfg = PR.ProbeConfig(ticks=30, segments=3, samples=8, elites=2, iters=1,
                         ladder=(1, 0), replans=0, delays=(2,),
                         jitters=(1.0,), jitter_samples=2)
    out = jax.jit(lambda k: PR.probe(lvl, task, k, cfg))(jax.random.PRNGKey(1))
    for k in ("ref_ok", "ref_gap", "ref_cost", "pilot_ok", "delays", "jitter", "lam",
              "horizon", "eta_mean", "eta_max", "eta_int", "n_scales", "hill_depth",
              "authority", "ok_full", "ok_k1", "cost_full", "cost_k1"):
        assert k in out, k
    assert np.asarray(out["delays"]).shape == (1,)
    assert np.asarray(out["jitter"]).shape == (1,)
    for k in ("eta_mean", "eta_max", "eta_int", "lam", "horizon", "authority"):
        assert np.isfinite(float(out[k])), (k, out[k])
    assert float(out["eta_max"]) >= float(out["eta_mean"]) >= 0.0
    assert float(out["authority"]) > 0.0


# ---------------------------------------------------------------------------
# opt.routes
# ---------------------------------------------------------------------------

def test_routes_pair_compares_two_models(world):
    lvl, task = world
    cfg = R.RouteConfig(ticks=TICKS, segments=SEG, samples=16, elites=4, iters=1, stride=4)
    out = jax.jit(lambda k: R.pair(lvl, task, k, cfg))(jax.random.PRNGKey(2))
    for name in ("approx", "ideal"):
        for f in ("arrived", "status", "gap", "cost", "time", "fuel", "ticks", "switches",
                  "pos", "alive", "actions"):
            assert f"{name}_{f}" in out, f"{name}_{f}"
    assert "coast_ok" in out and "mdl" in out
    assert np.asarray(out["mdl_ok"]).shape == (len(R.mdl_steps(SEG)),)
    # mdl is either 0 (nothing arrived at any resolution) or one of the resolutions
    assert int(out["mdl"]) in (0,) + R.mdl_steps(SEG)
    assert int(out["approx_switches"]) >= 0
    assert np.asarray(out["ideal_pos"]).shape == (TICKS // 4 + (TICKS % 4 > 0), 2)


def test_mdl_steps_only_divides_evenly():
    assert R.mdl_steps(48) == (1, 2, 3, 4, 6, 8, 12, 16, 24, 48)
    assert R.mdl_steps(6) == (1, 2, 3, 6)
    assert all(6 % m == 0 for m in R.mdl_steps(6))
    assert R.mdl_steps(1) == (1,)
    # the filter exists so a non-default segment count cannot reshape an empty
    # axis in `coarsen`: 5 shares no divisor with 2, 3, 4...
    assert R.mdl_steps(5) == (1,)


def test_quantise_and_switches_snap_to_the_human_control_set():
    acts = jnp.asarray([[-0.9, 0.95], [-0.9, 0.95], [0.1, 0.2], [1.0, 0.0]])
    ids, snapped = R.quantise(acts)
    # allclose, not array_equal: THROTTLES[1] is float32 0.3, which is not
    # bit-equal to Python's 0.3
    np.testing.assert_allclose(np.asarray(snapped),
                               [[-1.0, 1.0], [-1.0, 1.0], [0.0, 0.3], [1.0, 0.0]], atol=1e-6)
    assert int(R.switches(acts)) == 2
    assert np.all(np.asarray(ids) < 9)


def test_coarsen_averages_and_restores_the_shape():
    seg = jnp.arange(12, dtype=jnp.float32).reshape(6, 2)
    out = R.coarsen(seg, 3)
    assert out.shape == seg.shape
    # pairs of segments are replaced by their mean, twice over
    np.testing.assert_allclose(np.asarray(out[0]), np.asarray(out[1]))
    np.testing.assert_allclose(np.asarray(out[0]), [1.0, 2.0])
    np.testing.assert_allclose(np.asarray(R.coarsen(seg, 6)), np.asarray(seg))


# ---------------------------------------------------------------------------
# opt.necessity
# ---------------------------------------------------------------------------

def test_one_burn_actions_is_coast_turn_burn_coast(world):
    lvl, task = world
    acts = np.asarray(N.one_burn_actions(task, 10, float(task.start_angle) + 1.0, 0.5, TICKS))
    assert acts.shape == (TICKS, 2)
    assert np.all(acts[:10, 1] == 0.0), "no burn before the start tick"
    burn_ticks = 0.5 * float(task.fuel) / C.CTRL_DT
    assert np.sum(acts[:, 1] > 0) == pytest.approx(min(burn_ticks, TICKS - 10), abs=1.0)
    # the turn happens during the coast, never during the burn
    assert np.all(acts[:, 0][acts[:, 1] > 0] == 0.0)
    # already pointing the right way means no turn at all
    straight = np.asarray(N.one_burn_actions(task, 10, float(task.start_angle), 0.5, TICKS))
    assert np.all(straight[:, 0] == 0.0)


def test_ace_profile_and_scalars(world):
    lvl, task = world
    cfg = N.NeedConfig(ticks=TICKS, segments=SEG, resamples=2)
    seg = jnp.zeros((SEG, 2))
    ace = jax.jit(lambda k: N.ace_profile(lvl, task, seg, k, cfg))(jax.random.PRNGKey(4))
    assert ace.shape == (SEG,)
    a = np.asarray(ace)
    assert np.all((a >= 0.0) & (a <= 1.0))
    sc = jax.tree.map(float, N.necessity_scalars(ace))
    # concentration = 1 - normalised entropy, so a perfectly flat profile lands
    # on 0 only up to float32 error in the log-sum
    assert -1e-5 <= sc["concentration"] <= 1.0 + 1e-5, sc["concentration"]
    assert sc["need_max"] >= sc["need_mean"]
    assert 0 <= sc["n_critical"] <= SEG


def test_necessity_scalars_separate_a_spike_from_a_flat_profile():
    """The whole point of the profile: one tall spike is one irreplaceable
    decision, a flat profile is a level that forgives everything individually."""
    spike = N.necessity_scalars(jnp.array([1.0, 0.0, 0.0, 0.0, 0.0, 0.0]))
    flat = N.necessity_scalars(jnp.full(6, 0.5))
    assert float(spike["concentration"]) == pytest.approx(1.0, abs=1e-5)
    assert float(flat["concentration"]) == pytest.approx(0.0, abs=1e-5)
    assert int(spike["n_critical"]) == 1
    assert int(flat["n_critical"]) == 0          # 0.5 is not > 0.5


def test_strategy_volume_and_scalars(world):
    lvl, task = world
    cfg = N.NeedConfig(ticks=TICKS, segments=SEG, n_time=4, n_dir=4, n_dv=2, cell_samples=2)
    vol = jax.jit(lambda k: N.strategy_volume(lvl, task, cfg, k))(jax.random.PRNGKey(5))
    assert vol["win"].shape == (4, 4, 2)
    w = np.asarray(vol["win"])
    # a probability field, not a binary map
    assert np.all((w >= 0.0) & (w <= 1.0))
    assert np.asarray(vol["times"]).shape == (4,)
    assert np.asarray(vol["dirs"]).shape == (4,)
    sc = N.volume_scalars(vol)
    assert 0.0 <= float(sc["sufficiency"]) <= 1.0
    assert -1.0001 <= float(sc["leading"]) <= 1.0001
    assert float(sc["window_s"]) >= 0.0


def test_volume_scalars_read_a_diagonal_stripe_as_leading():
    """`leading` is the correlation between burn time and heading over the winning
    set, so a diagonal stripe must come out near +1 and a blob near 0."""
    n_t = n_d = 8
    stripe = np.zeros((n_t, n_d, 1), np.float32)
    for i in range(n_t):
        stripe[i, i, 0] = 1.0
    blob = np.zeros((n_t, n_d, 1), np.float32)
    blob[2:6, 2:6, 0] = 1.0
    axes = dict(times=jnp.arange(n_t) * 3, dirs=jnp.linspace(0.0, 1.0, n_d), dvs=jnp.ones(1))
    assert float(N.volume_scalars(dict(win=jnp.asarray(stripe), **axes))["leading"]) > 0.9
    assert abs(float(N.volume_scalars(dict(win=jnp.asarray(blob), **axes))["leading"])) < 0.2


def test_boundary_exponent_flags_an_unmeasurable_fit(world):
    """alpha fitted to an empty winning set measures nothing; `measurable` is the
    flag `lessons` relies on to avoid condemning a level for being hard to probe."""
    lvl, task = world
    cfg = N.NeedConfig(ticks=TICKS, segments=SEG, eps=(0.005, 0.02, 0.07), eps_samples=8)
    out = jax.jit(lambda k: N.boundary_exponent(lvl, task, k, cfg))(jax.random.PRNGKey(6))
    assert np.asarray(out["flip"]).shape == (3,)
    assert np.isfinite(float(out["alpha"]))
    assert 0.0 <= float(out["win_rate"]) <= 1.0
    # 60 ticks is far too short for a single burn to arrive here, so this probe
    # must report itself unmeasurable rather than handing back a bogus exponent
    if float(out["win_rate"]) <= 0.02:
        assert not bool(out["measurable"])


def test_adaptivity_reports_both_pilots(world):
    lvl, task = world
    cfg = N.NeedConfig(ticks=TICKS, segments=SEG)
    out = jax.jit(lambda k: N.adaptivity(lvl, task, k, cfg, 1))(jax.random.PRNGKey(7))
    assert set(out) == {"open_loop", "replanned", "recovers", "adaptation_gain"}
    # `recovers` is exactly "re-planning rescued what one plan lost"
    assert bool(out["recovers"]) == (bool(out["replanned"]) and not bool(out["open_loop"]))
    assert float(out["adaptation_gain"]) in (-1.0, 0.0, 1.0)
