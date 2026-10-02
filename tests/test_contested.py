"""`env.contested_zone` and contested waypoint placement: the one knob that
reliably breaks a one-body model.

The claim under test is geometric and closed-form: the equal-pull surface sits at
d = a*sqrt(m/M), roughly two thirds of a Hill radius, and eta -- the share of the
local force a dominant-body model throws away -- is large there and falls off
quickly outside it.  `p_contested` then has to actually put waypoints in that
shell, otherwise the flag does nothing.
"""

import jax
import jax.numpy as jnp
import numpy as np
import pytest

from spacenav import constants as C
from spacenav import env as E
from spacenav import physics as P
from helpers import body_index, hierarchy_level


def eta_at(level, bp, x):
    """Discarded-force fraction of the dominant-body model at a point."""
    full = P.gravity_at(x[None], bp, level, 0)[0]
    dom = P.gravity_at(x[None], bp, level, 1)[0]
    return float(jnp.linalg.norm(full - dom) / (jnp.linalg.norm(full) + 1e-12))


@pytest.fixture(scope="module")
def world():
    lvl = hierarchy_level()
    return lvl, lvl.snap_pos[0], body_index(lvl, C.KIND_GIANT)


def test_equal_pull_radius_matches_the_closed_form(world):
    lvl, bp, gi = world
    good, r_eq, hill, u = E.contested_zone(lvl, bp)
    assert bool(good[gi])
    m, M = float(lvl.mass[gi]), float(lvl.mass[int(lvl.parent[gi])])
    a = float(jnp.linalg.norm(bp[gi] - bp[int(lvl.parent[gi])]))
    assert float(r_eq[gi]) == pytest.approx(a * np.sqrt(m / M), rel=1e-4)
    assert float(hill[gi]) == pytest.approx(a * (m / (3 * M)) ** (1 / 3), rel=1e-4)
    # "roughly half a Hill radius": 0.67 for mu = 0.01, and the ratio only moves
    # as mu^(1/6), so this band holds across any sane mass ratio
    assert 0.4 < float(r_eq[gi]) / float(hill[gi]) < 0.95
    # the radial unit vector points from the host to the body
    np.testing.assert_allclose(np.asarray(u[gi]),
                               np.asarray((bp[gi] - bp[int(lvl.parent[gi])]) / a), atol=1e-5)
    # the root star has no host, so there is no surface to speak of
    root = int(np.flatnonzero(np.asarray(lvl.parent) < 0)[0])
    assert not bool(good[root])


def test_the_root_and_massless_bodies_are_excluded(world):
    lvl, bp, _ = world
    good, _, _, _ = E.contested_zone(lvl, bp)
    g = np.asarray(good)
    assert not g[~np.asarray(lvl.active)].any()
    assert not g[np.asarray(lvl.mass) <= 0].any()
    assert not g[np.asarray(lvl.kind) == C.KIND_TRACER].any()


def test_eta_peaks_on_the_surface_and_falls_off_outside(world):
    lvl, bp, gi = world
    _, r_eq, _, u = E.contested_zone(lvl, bp)
    d = float(r_eq[gi])
    # outward along the host-to-body line, so the two pulls add and eta is the
    # clean "fraction of the force the one-body model misses"
    etas = [eta_at(lvl, bp, bp[gi] + k * d * u[gi]) for k in (1.0, 2.0, 4.0, 8.0)]
    assert etas[0] > 0.35, etas
    assert etas[-1] < 0.1, etas
    assert etas == sorted(etas, reverse=True), etas
    # and far enough out it is negligible: patched conics is right almost
    # everywhere, which is why breaking it takes deliberate placement
    far = eta_at(lvl, bp, bp[gi] + 30.0 * d * u[gi])
    assert far < 0.03, far


def test_eta_exceeds_one_where_the_field_cancels(world):
    """Not a bug and not a bound to assert against: between the star and the
    giant the pulls oppose, the resultant nearly vanishes, and a one-body model
    is wrong by more than the whole remaining force."""
    lvl, bp, gi = world
    _, r_eq, _, u = E.contested_zone(lvl, bp)
    inward = eta_at(lvl, bp, bp[gi] - float(r_eq[gi]) * u[gi])
    assert inward > 1.0, inward
    assert np.isfinite(inward)


def test_contested_placement_raises_eta_at_the_waypoint():
    """`p_contested=1.0` has to move waypoints into the shell, not just set a flag."""
    lvl = hierarchy_level()
    n = 16
    keys = jax.random.split(jax.random.PRNGKey(11), n)

    def sample(cfg):
        tasks, oks = jax.vmap(lambda k: E.sample_task(k, lvl, cfg))(keys)
        out = []
        for i in range(n):
            t = jax.tree.map(lambda a: a[i], tasks)
            if not bool(oks[i]):
                continue
            wpos, _ = E.waypoint_states(t, lvl.snap_pos[int(t.snapshot)],
                                        lvl.snap_vel[int(t.snapshot)])
            out.append(eta_at(lvl, lvl.snap_pos[int(t.snapshot)], wpos[0]))
        return np.array(out)

    # p_station=0 for the baseline on purpose: an ordinary *body-hugging*
    # waypoint sits at radius+atmo+35 u, which on this giant is accidentally
    # close to its 40 u equal-pull surface.  That near-miss is exactly why the
    # placement has to be an explicit knob instead of a side effect of p_station.
    base = E.TaskConfig(min_dist=60.0, min_leg=60.0, p_station=0.0)
    plain = sample(base)
    chaos = sample(base._replace(p_contested=1.0))
    assert len(plain) >= 8 and len(chaos) >= 8, (len(plain), len(chaos))
    # the contested sampler parks waypoints at 0.8-1.3 r_eq, where eta is ~0.5;
    # a free-space waypoint is somewhere in the star's domain, where it is tiny
    assert np.median(chaos) > 0.1, np.median(chaos)
    assert np.median(chaos) > 10 * np.median(plain), (np.median(plain), np.median(chaos))
    # every contested waypoint rides a secondary, never a fixed point in space
    tasks, oks = jax.vmap(lambda k: E.sample_task(k, lvl, base._replace(p_contested=1.0)))(keys)
    anchors = np.asarray(tasks.wp_anchor)[:, 0][np.asarray(oks)]
    assert np.mean(anchors >= 0) > 0.8, anchors
    # and the one it rides always has a host: the root star has no equal-pull surface
    parents = np.asarray(lvl.parent)[anchors[anchors >= 0]]
    assert np.all(parents >= 0), parents


def test_chaotic_zone_alias_still_reports_hill_radii(world):
    lvl, bp, gi = world
    has, outer, hill, u = E.chaotic_zone(lvl, bp)
    good, _, hill2, u2 = E.contested_zone(lvl, bp)
    np.testing.assert_array_equal(np.asarray(has), np.asarray(good))
    np.testing.assert_allclose(np.asarray(hill), np.asarray(hill2))
    np.testing.assert_allclose(np.asarray(outer), 1.3 * np.asarray(hill2))
    np.testing.assert_allclose(np.asarray(u), np.asarray(u2))
