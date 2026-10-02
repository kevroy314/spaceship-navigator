import jax
import jax.numpy as jnp
import numpy as np
import pytest

from spacenav import constants as C
from spacenav import env as E
from spacenav import physics as P
from spacenav.levels.build import custom_level
from spacenav.levels.tree import Node, Orbit
from spacenav.types import ARRIVED, CRASHED, RUNNING, Task
from helpers import body_index, hierarchy_level


def two_body(a=200.0, gm=5e4, m=10.0):
    star = Node(C.KIND_STAR, gm, 20.0)
    star.add(Node(C.KIND_ROCKY, m, 5.0), Orbit(a, phase=0.0, omega=0.0))
    return custom_level(star)


def empty_level():
    # a lone massless tracer: no gravity anywhere
    return custom_level(Node(C.KIND_TRACER, 0.0, 0.0), level_radius=5000.0)


def make_task(level, start_pos=(0.0, 0.0), start_vel=(0.0, 0.0), angle=0.0, target=(1e4, 0.0),
              accel=C.SHIP_ACCEL, fuel=C.SHIP_FUEL, extra=()):
    """A one-waypoint mission by default; `extra` adds further waypoints in order."""
    z = jnp.zeros(C.MAX_ZONES, bool)
    K = C.MAX_WAYPOINTS
    pts = [target, *extra]
    pos = jnp.array([pts[i] if i < len(pts) else (0.0, 0.0) for i in range(K)], jnp.float32)
    active = jnp.array([i < len(pts) for i in range(K)])
    return Task(snapshot=jnp.int32(0), start_pos=jnp.array(start_pos, jnp.float32),
                start_vel=jnp.array(start_vel, jnp.float32), start_angle=jnp.float32(angle),
                wp_anchor=jnp.full(K, -1, jnp.int32), wp_pos=pos, wp_offset=jnp.zeros((K, 2)),
                wp_radius=jnp.full(K, C.TARGET_RADIUS),
                # fly-through, not a rendezvous: the sentinel, since
                # deltav_estimate branches on v_tol < FLYTHROUGH_V_TOL / 2
                wp_v_tol=jnp.full(K, C.FLYTHROUGH_V_TOL),
                wp_active=active, order_free=jnp.bool_(False),
                accel=jnp.float32(accel), fuel=jnp.float32(fuel),
                # these 4s are N_OBJECTIVES, not a waypoint count: spelled out so
                # they are not mistaken for MAX_WAYPOINTS padding (which is K)
                weights=jnp.eye(C.N_OBJECTIVES)[0],
                exposure_mask=z, ref_scale=jnp.ones(C.N_OBJECTIVES))


def test_kepler_period():
    a, gm, m = 200.0, 5e4, 10.0
    lv = two_body(a, gm, m)
    period = 2 * np.pi * np.sqrt(a**3 / (gm + m))
    n = int(round(period / C.PHYS_DT))
    ps, _ = jax.jit(P.roll_bodies, static_argnums=3)(lv, lv.snap_pos[0], lv.snap_vel[0], n)
    rel0 = np.asarray(ps[0, 1] - ps[0, 0])
    rel1 = np.asarray(ps[-1, 1] - ps[-1, 0])
    assert np.linalg.norm(rel1 - rel0) < 0.01 * a
    dist = np.linalg.norm(np.asarray(ps[:, 1] - ps[:, 0]), axis=-1)
    assert dist.min() > 0.999 * a and dist.max() < 1.001 * a


def test_energy_conserved():
    lv = two_body()
    ps, vs = P.roll_bodies(lv, lv.snap_pos[0], lv.snap_vel[0], 6000)
    e0 = P.energy(lv, ps[0], vs[0])
    e1 = P.energy(lv, ps[-1], vs[-1])
    assert abs((e1 - e0) / e0) < 1e-4


def test_gravity_finite_inside_and_no_self_force():
    lv = two_body()
    pts = lv.snap_pos[0]                       # exactly at the body centres
    g = P.gravity_at(pts, pts, lv)
    assert np.all(np.isfinite(np.asarray(g)))
    g_centre = P.gravity_at(lv.snap_pos[0][:1], lv.snap_pos[0][:1] * 0 + lv.snap_pos[0][:1], lv)
    assert np.all(np.isfinite(np.asarray(g_centre)))


def test_thrust_delta_v_in_empty_space():
    lv = empty_level()
    task = make_task(lv, angle=0.0)
    s = E.reset(lv, task)
    for _ in range(15):                          # 1 s of full throttle, nose along +x
        s, _, _ = E.step(lv, task, s, jnp.array([0.0, 1.0]))
    np.testing.assert_allclose(np.asarray(s.ship.vel), [C.SHIP_ACCEL, 0.0], atol=1e-3)
    np.testing.assert_allclose(float(s.ship.fuel), C.SHIP_FUEL - 1.0, atol=1e-3)
    np.testing.assert_allclose(float(s.ship.pos[0]), 0.5 * C.SHIP_ACCEL, atol=1e-2)


def test_turning_rate():
    lv = empty_level()
    task = make_task(lv)
    s = E.reset(lv, task)
    s, _, _ = E.step(lv, task, s, jnp.array([1.0, 0.0]))
    np.testing.assert_allclose(float(s.ship.angle), C.SHIP_TURN_RATE * C.CTRL_DT, rtol=1e-5)


def test_loadout_sets_thrust_and_tank():
    """Thrust and fuel come from the mission, not from a global constant."""
    lv = empty_level()
    task = make_task(lv, accel=1.0, fuel=5.0)
    s = E.reset(lv, task)
    assert float(s.ship.fuel) == pytest.approx(5.0)
    step = jax.jit(lambda s: E.step(lv, task, s, jnp.array([0.0, 1.0])))
    for _ in range(15):
        s = step(s)[0]
    np.testing.assert_allclose(np.asarray(s.ship.vel), [1.0, 0.0], atol=1e-3)


def test_fuel_runs_out():
    lv = empty_level()
    task = make_task(lv)
    s = E.reset(lv, task)
    step = jax.jit(lambda s: E.step(lv, task, s, jnp.array([0.0, 1.0]))[0])
    for _ in range(int(C.SHIP_FUEL / C.CTRL_DT) + 30):
        s = step(s)
    assert float(s.ship.fuel) == pytest.approx(0.0, abs=1e-5)
    assert float(s.ship.vel[0]) == pytest.approx(C.SHIP_ACCEL * C.SHIP_FUEL, rel=1e-3)


def test_tour_visits_waypoints_in_order():
    """A two-waypoint tour only finishes once both are reached, in sequence."""
    lv = empty_level()
    task = make_task(lv, start_vel=(30.0, 0.0), target=(60.0, 0.0), extra=[(300.0, 0.0)])
    s = E.reset(lv, task)
    step = jax.jit(lambda s: E.step(lv, task, s, jnp.array([0.0, 0.0])))
    for _ in range(40):
        s, _, _ = step(s)
    assert bool(s.visited[0]) and not bool(s.visited[1])     # first passed, second not yet
    assert int(s.status) == RUNNING and int(s.leg) == 1
    for _ in range(130):
        s, _, _ = step(s)
    assert bool(s.visited[1]) and int(s.status) == ARRIVED


def test_arrival_and_freeze():
    lv = empty_level()
    task = make_task(lv, start_vel=(30.0, 0.0), target=(40.0, 0.0))
    s = E.reset(lv, task)
    step = jax.jit(lambda s: E.step(lv, task, s, jnp.array([0.0, 0.0])))
    for _ in range(30):
        s, r, info = step(s)
    assert int(s.status) == ARRIVED
    frozen = s
    s, r, _ = step(s)
    assert float(r) == 0.0
    np.testing.assert_array_equal(np.asarray(s.ship.pos), np.asarray(frozen.ship.pos))
    assert float(s.costs.time) == float(frozen.costs.time)


def test_crash_into_star():
    lv = two_body()
    task = make_task(lv, start_pos=(float(lv.snap_pos[0, 0, 0]) + 40.0, float(lv.snap_pos[0, 0, 1])))
    s = E.reset(lv, task)
    step = jax.jit(lambda s: E.step(lv, task, s, jnp.array([0.0, 0.0]))[0])
    for _ in range(60):
        s = step(s)
    assert int(s.status) == CRASHED


def test_radiation_falls_off():
    lv = two_body()
    star = lv.snap_pos[0, 0]
    z = jnp.zeros(C.MAX_ZONES, bool)
    lv = lv._replace(lum=lv.lum.at[0].set(1000.0))
    d1, _, _ = P.hazards(lv, lv.snap_pos[0], lv.snap_vel[0], star + jnp.array([50.0, 0]), jnp.zeros(2), z)
    d2, _, _ = P.hazards(lv, lv.snap_pos[0], lv.snap_vel[0], star + jnp.array([100.0, 0]), jnp.zeros(2), z)
    assert float(d1) == pytest.approx(4 * float(d2), rel=0.05)


# ---------------------------------------------------------------------------
# patched-conic models: gravity_at(topk=k)
# ---------------------------------------------------------------------------

def _probe_points(lv, n=24):
    """A ring of sample points across the level, avoiding body interiors."""
    th = np.linspace(0.0, 2 * np.pi, n, endpoint=False)
    r = np.linspace(80.0, 0.8 * float(lv.level_radius), n)
    return jnp.asarray(np.stack([r * np.cos(th), r * np.sin(th)], -1), jnp.float32)


def test_topk_zero_is_the_full_field():
    """topk=0 means "no restriction", and must be bit-identical to the default."""
    lv = hierarchy_level()
    pts = _probe_points(lv)
    bp = lv.snap_pos[0]
    np.testing.assert_array_equal(np.asarray(P.gravity_at(pts, bp, lv, 0)),
                                  np.asarray(P.gravity_at(pts, bp, lv)))


def test_topk_truncation_error_is_bounded_by_what_it_discards():
    """A coarse model is a truncated *vector* sum, so the quantity that falls
    monotonically is the discarded force **budget**, not the residual.

    The residual |g_k - g_full| is not pointwise monotone in k, and that is real
    rather than numerical: the bodies a k=2 model throws away can partly cancel
    each other, so a k=2 model is occasionally *more* accurate than k=3.
    Measured on this level, exactly one of 24 probe points does it -- at r = 226
    the residual rises from 3.12e-5 to 3.36e-5 going from k=2 to k=3.  Asserting
    pointwise monotonicity would therefore be asserting something false.

    What does hold, and is what `opt.probe`'s model ladder relies on:
      * sum of |a_j| over the discarded bodies falls strictly with k, to zero;
      * the residual is bounded by that budget (triangle inequality);
      * k = n_active reproduces the full field exactly.
    """
    lv = hierarchy_level()
    pts = _probe_points(lv)
    bp = lv.snap_pos[0]
    full = np.asarray(P.gravity_at(pts, bp, lv, 0))
    n_active = int(np.sum(np.asarray(lv.active)))

    errs = np.stack([np.linalg.norm(np.asarray(P.gravity_at(pts, bp, lv, k)) - full, axis=-1)
                     for k in range(1, n_active + 1)])

    # per-body acceleration magnitudes, so the discarded budget is exact
    d = np.asarray(bp)[None, :, :] - np.asarray(pts)[:, None, :]
    r = np.linalg.norm(d, axis=-1)
    soft = np.maximum(np.maximum(r, np.asarray(lv.radius)[None, :]), P.EPS)
    m = np.where(np.asarray(lv.active), np.asarray(lv.mass), 0.0)
    desc = -np.sort(-(m[None, :] / soft ** 2), axis=1)          # per point, descending
    budget = np.stack([desc[:, k:].sum(1) for k in range(1, n_active + 1)])

    assert np.all(np.diff(budget, axis=0) <= 1e-9), np.max(np.diff(budget, axis=0))
    assert np.all(errs <= budget + 1e-5), np.max(errs - budget)
    assert np.all(errs[-1] == 0.0), errs[-1].max()
    # it does converge, just in the mean rather than pointwise: ~60x per rung here
    assert errs.mean(1)[0] > 20 * errs.mean(1)[1]
    assert list(errs.mean(1)) == sorted(errs.mean(1), reverse=True), errs.mean(1)
    # and k=1 is genuinely coarse somewhere: eta is not uniformly negligible
    assert np.max(errs[0] / np.linalg.norm(full, axis=-1)) > 0.01
    # the one non-monotone point is small, because the budget bounds it
    rise = np.max(np.diff(errs, axis=0))
    assert rise < 0.01 * budget[0].max(), rise


def test_topk_never_exceeds_the_available_bodies():
    """k larger than the body array must clamp, not index out of range."""
    lv = hierarchy_level()
    pts = _probe_points(lv)
    bp = lv.snap_pos[0]
    big = np.asarray(P.gravity_at(pts, bp, lv, 10 * C.MAX_BODIES))
    np.testing.assert_allclose(big, np.asarray(P.gravity_at(pts, bp, lv, 0)), atol=1e-6)


def test_topk_one_picks_the_dominant_attractor():
    """Right next to the giant, k=1 must be the giant alone."""
    lv = hierarchy_level()
    bp = lv.snap_pos[0]
    gi = body_index(lv, C.KIND_GIANT)
    x = bp[gi] + jnp.array([25.0, 0.0])
    dom = np.asarray(P.gravity_at(x[None], bp, lv, 1)[0])
    d = np.asarray(bp[gi] - x)
    r = np.linalg.norm(d)
    expect = float(lv.mass[gi]) / r**3 * d
    np.testing.assert_allclose(dom, expect, rtol=1e-4)


def test_step_topk_restricts_only_the_ship():
    """The bodies always move truthfully; only the test particle's felt force is
    coarsened.  So a coarse step moves the ship elsewhere but the bodies alike."""
    lv = hierarchy_level()
    gi = body_index(lv, C.KIND_GIANT)
    bp = np.asarray(lv.snap_pos[0])
    task = make_task(lv, start_pos=tuple(bp[gi] + np.array([45.0, 0.0])),
                     start_vel=(0.0, 3.0), target=(900.0, 0.0), accel=1.0, fuel=10.0)
    s0 = E.reset(lv, task)
    act = jnp.array([0.0, 0.0])
    # jit each ladder rung: calling E.step eagerly in a loop rebuilds the substep
    # fori_loop body every iteration and recompiles it, which turned 180 steps
    # into four minutes of XLA time
    run = {k: jax.jit(lambda s, k=k: E.step(lv, task, s, act, topk=k)[0]) for k in (0, 1)}
    dflt = jax.jit(lambda s: E.step(lv, task, s, act)[0])
    full, coarse, again = s0, s0, s0
    for _ in range(60):
        full = run[0](full)
        coarse = run[1](coarse)
        again = dflt(again)
    np.testing.assert_allclose(np.asarray(full.body_pos), np.asarray(coarse.body_pos), atol=1e-4)
    # on the giant's equal-pull surface the star is as strong as the giant, so
    # dropping it must visibly move the ship
    sep = np.linalg.norm(np.asarray(full.ship.pos - coarse.ship.pos))
    assert sep > 1.0, sep
    # topk=0 is the default path
    np.testing.assert_array_equal(np.asarray(again.ship.pos), np.asarray(full.ship.pos))


def test_ship_accel_and_substep_pass_topk_through():
    """`topk` has to reach the integrator, not just `gravity_at`: a plumbing test,
    because a silently ignored argument would make every coarse-model
    measurement in opt/ a measurement of the full field."""
    lv = hierarchy_level()
    gi = body_index(lv, C.KIND_GIANT)
    bp, bv = lv.snap_pos[0], lv.snap_vel[0]
    x = bp[gi] + jnp.array([40.0, 0.0])      # on the giant's equal-pull surface
    v = jnp.array([0.0, 3.0])
    thrust = jnp.zeros(2)

    a_full = P.ship_accel(lv, bp, bv, x, v, thrust, 0)
    a_dom = P.ship_accel(lv, bp, bv, x, v, thrust, 1)
    np.testing.assert_allclose(np.asarray(a_full),
                               np.asarray(P.ship_accel(lv, bp, bv, x, v, thrust)), atol=0)
    # Dropping the star costs ~45% of the felt force here, not 50%: the
    # equal-pull radius d = a*sqrt(m/M) is derived with the star at distance a,
    # but a point d *outside* the giant is at a + d from the star, so the star's
    # pull there is (a/(a+d))^2 of the giant's.  With a = 400, d = 39.9 that is
    # 0.827, giving a discarded share of 0.827/1.827 = 0.453 (the moon trims it
    # to 0.449).  Asserting > 0.5 would be asserting the wrong geometry.
    share = (np.linalg.norm(np.asarray(a_full - a_dom))
             / np.linalg.norm(np.asarray(a_full)))
    assert 0.40 < share < 0.50, share

    bp_full, _, x_full, v_full = P.substep(lv, bp, bv, x, v, thrust, topk=0)
    bp_dom, _, x_dom, v_dom = P.substep(lv, bp, bv, x, v, thrust, topk=1)
    # One substep is h = 1/60 s, so the *position* difference is second order,
    # 0.5*da*h^2 = 3.5e-5 u -- below np.allclose's default tolerance at x ~ 435,
    # which is why the 60-tick accumulation test above is the one that sees it.
    # The plumbing shows up at first order in the velocity: dv = da*h.
    da = float(np.linalg.norm(np.asarray(a_full - a_dom)))
    dv = float(np.linalg.norm(np.asarray(v_full - v_dom)))
    assert dv == pytest.approx(da * C.PHYS_DT, rel=0.05), (dv, da * C.PHYS_DT)
    assert np.linalg.norm(np.asarray(x_full - x_dom)) < 1e-3

    # the bodies are integrated with the full field whatever the ship is told
    np.testing.assert_array_equal(np.asarray(bp_full), np.asarray(bp_dom))
    np.testing.assert_array_equal(np.asarray(P.bodies_step(bp, bv, lv)[0]),
                                  np.asarray(bp_dom))
