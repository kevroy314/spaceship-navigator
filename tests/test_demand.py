"""`spacenav.demand`: the closed-form model-demand number and its inversion.

D = eta / A is the fraction of the whole tank that flying on a one-body model
would cost.  The design claim that matters is that it *inverts*: given a
placement you can solve for the tank that puts the level on a chosen rung.
`retune` is that solve, so these tests check the algebra actually closes rather
than just that the function returns a number.
"""

import jax.numpy as jnp
import numpy as np
import pytest

from spacenav import constants as C
from spacenav import demand as D
from spacenav.types import Task
from helpers import body_index, hierarchy_level


def contested_task(level, accel=1.0, fuel=10.0, t_ref=60.0, offset=40.0):
    """One waypoint parked on the giant's equal-pull surface (eta is large there).

    The start sits out in the star's domain, so the leg midpoint that `terms`
    also samples is a low-eta point: eta_bar ends up a blend, which is exactly
    what the real sampler produces.
    """
    gi = body_index(level, C.KIND_GIANT)
    K = C.MAX_WAYPOINTS
    anchor = np.full(K, -1, np.int32)
    anchor[0] = gi
    off = np.zeros((K, 2), np.float32)
    off[0] = (offset, 0.0)
    active = np.arange(K) < 1
    return Task(
        snapshot=jnp.int32(0), start_pos=jnp.array([700.0, 0.0]),
        start_vel=jnp.array([0.0, 8.0]), start_angle=jnp.float32(0.0),
        wp_anchor=jnp.asarray(anchor), wp_pos=jnp.zeros((K, 2)), wp_offset=jnp.asarray(off),
        wp_radius=jnp.full(K, C.TARGET_RADIUS),
        wp_v_tol=jnp.full(K, C.FLYTHROUGH_V_TOL),
        wp_active=jnp.asarray(active), order_free=jnp.bool_(False),
        accel=jnp.float32(accel), fuel=jnp.float32(fuel),
        weights=jnp.array([1.0, 0.0, 0.0, 0.0]),
        exposure_mask=jnp.zeros_like(level.zone_active),
        ref_scale=jnp.array([t_ref, 1000.0, fuel, 1.0]),
    )


@pytest.fixture(scope="module")
def lvl():
    return hierarchy_level()


def test_D_is_eta_over_A_and_positive(lvl):
    t = D.terms(lvl, contested_task(lvl))
    # eta is |discarded| / |total|, so it is positive but NOT bounded by 1: near a
    # saddle the full field nearly cancels and the ratio blows up.  Here the
    # waypoint sits on the anti-star side of the giant, where the two pulls add,
    # so eta is the honest "share of the force one body misses", below 1.
    assert 0.0 < float(t.eta) < 1.0
    assert float(t.g) > 0.0 and float(t.T) > 0.0 and float(t.dv) > 0.0
    assert float(t.A) > 0.0
    assert float(t.D) > 0.0
    assert float(t.D) == pytest.approx(float(t.eta) / float(t.A), rel=1e-5)
    assert float(t.impulse) == pytest.approx(float(t.g) * float(t.T), rel=1e-5)
    assert np.isfinite([float(t.D), float(t.A), float(t.eta)]).all()


def test_D_depends_only_on_total_deltav(lvl):
    """Dimensionless in the ship's terms: D is a function of fuel*accel, not of
    either alone.  Doubling the tank while halving the engine leaves it put."""
    a = D.terms(lvl, contested_task(lvl, accel=1.0, fuel=10.0))
    b = D.terms(lvl, contested_task(lvl, accel=0.5, fuel=20.0))
    assert float(b.D) == pytest.approx(float(a.D), rel=1e-4)
    # and it is inversely proportional to the tank
    c = D.terms(lvl, contested_task(lvl, accel=1.0, fuel=20.0))
    assert float(c.D) == pytest.approx(0.5 * float(a.D), rel=1e-4)


def test_retune_hits_the_requested_demand(lvl):
    """The inversion claim of the whole design: solve for the tank, get the rung.

    Only meaningful where the solved fuel is not clipped to SHIP_FUEL_LIMITS, so
    the targets are derived from the level's own geometry to land mid-tank, and
    the test asserts the fuel really did come out unclipped.
    """
    task = contested_task(lvl)
    base = D.terms(lvl, task)
    lo, hi = C.SHIP_FUEL_LIMITS
    # fuel = eta * impulse / (D * accel); pick targets that put fuel well inside
    # the tank limits for this level, so nothing is clipped
    unclipped = 0
    for fuel_want in (0.2 * hi, 0.4 * hi, 0.6 * hi, 0.8 * hi):
        target = float(base.eta * base.impulse) / (fuel_want * float(task.accel))
        tuned = D.retune(lvl, task, target)
        fuel = float(tuned.fuel)
        assert lo < fuel < hi, f"target_D={target} clipped the tank to {fuel}"
        unclipped += 1
        assert fuel == pytest.approx(fuel_want, rel=1e-4)
        got = float(D.terms(lvl, tuned).D)
        assert got == pytest.approx(target, rel=1e-4), (target, got, fuel)
    assert unclipped == 4


def test_retune_is_monotone_and_leaves_geometry_alone(lvl):
    task = contested_task(lvl)
    fuels = [float(D.retune(lvl, task, d).fuel) for d in (0.05, 0.1, 0.2, 0.4)]
    assert fuels == sorted(fuels, reverse=True), fuels
    tuned = D.retune(lvl, task, 0.2)
    # fuel is the one knob with no effect on the route, which is why D inverts in it
    for field in ("start_pos", "wp_anchor", "wp_offset", "wp_active", "accel", "ref_scale"):
        np.testing.assert_array_equal(np.asarray(getattr(tuned, field)),
                                      np.asarray(getattr(task, field)))


def test_retune_clips_rather_than_lying(lvl):
    """A rung the geometry cannot reach returns the clipped tank, and `survey`
    then reports the *achieved* D, not the requested one."""
    task = contested_task(lvl)
    tuned = D.retune(lvl, task, 1e6)          # absurdly hard: demands a thimble of fuel
    assert float(tuned.fuel) == pytest.approx(C.SHIP_FUEL_LIMITS[0])
    assert float(D.terms(lvl, tuned).D) < 1e6


def test_rung_and_survey(lvl):
    task = contested_task(lvl)
    assert D.rung(0.0) == 0
    assert D.rung(1e9) == len(D.RUNGS)
    for i, r in enumerate(D.RUNGS):
        assert D.rung(r * 1.001) == i + 1
    s = D.survey(lvl, task)
    assert set(s) >= {"D", "eta", "g", "T", "dv", "A", "impulse", "feasible",
                      "authority", "rung", "rung_name"}
    assert s["rung_name"] in D.RUNG_NAMES
    assert isinstance(s["feasible"], bool)
    assert s["feasible"] is True        # t_ref = 60 s, well inside the 150 s clock


def test_infeasible_when_the_estimate_overruns_the_clock(lvl):
    t = D.terms(lvl, contested_task(lvl, t_ref=0.9 * C.MAX_EPISODE_TIME))
    assert not bool(t.feasible)
    # T itself is still clamped to the episode, so impulse stays finite
    assert float(t.T) <= C.MAX_EPISODE_TIME
