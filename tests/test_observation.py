"""The observation contract, and the instrument-panel ablation.

`env.INSTRUMENTS = False` is the ablation that makes the "these are the right
abstractions" claim falsifiable: it must zero the four readouts and touch
*nothing else*, because otherwise the two training runs are not comparable.
"""

import jax.numpy as jnp
import numpy as np
import pytest

from spacenav import constants as C
from spacenav import env as E
from helpers import body_index, hierarchy_level

# self-feature layout, as `observe` builds it: a fixed ship/tour block, then one
# 13-wide block per padded waypoint slot.
SELF_HEAD = 31          # 2 vel + 2 g + 2 + 2 + 1 + 2 tgt + 12 scalars + 4 weights + 4 panel
WP_BLOCK = 13
PANEL = 4               # the instrument_panel channels, last in the head block


@pytest.fixture(scope="module")
def setup():
    lvl = hierarchy_level()
    gi = body_index(lvl, C.KIND_GIANT)
    K = C.MAX_WAYPOINTS
    anchor = np.full(K, -1, np.int32)
    anchor[0] = gi
    off = np.zeros((K, 2), np.float32)
    off[0] = (45.0, 0.0)
    from spacenav.types import Task
    task = Task(
        snapshot=jnp.int32(0), start_pos=jnp.array([620.0, 0.0]),
        start_vel=jnp.array([0.0, 8.0]), start_angle=jnp.float32(0.3),
        wp_anchor=jnp.asarray(anchor), wp_pos=jnp.zeros((K, 2)), wp_offset=jnp.asarray(off),
        wp_radius=jnp.full(K, C.TARGET_RADIUS), wp_v_tol=jnp.full(K, C.FLYTHROUGH_V_TOL),
        wp_active=jnp.asarray(np.arange(K) < 2), order_free=jnp.bool_(False),
        accel=jnp.float32(1.5), fuel=jnp.float32(12.0),
        weights=jnp.array([1.0, 0.0, 0.0, 0.0]),
        exposure_mask=jnp.zeros_like(lvl.zone_active),
        ref_scale=jnp.array([60.0, 1000.0, 12.0, 1.0]),
    )
    # waypoint 1 is a fixed point out in the star's domain
    task = task._replace(wp_pos=task.wp_pos.at[1].set(jnp.array([900.0, 150.0])))
    return lvl, task, E.reset(lvl, task)


def test_self_width_tracks_max_waypoints(setup):
    """The observation widens with MAX_WAYPOINTS, so this is written as the
    layout rather than as the number it currently comes to (109 at K=6)."""
    lvl, task, st = setup
    obs = E.observe(lvl, task, st)
    assert obs["self"].shape == (SELF_HEAD + WP_BLOCK * C.MAX_WAYPOINTS,)
    assert obs["bodies"].shape[0] == C.MAX_BODIES
    assert obs["zones"].shape[0] == C.MAX_ZONES
    assert obs["body_mask"].shape == (C.MAX_BODIES,)
    assert np.all(np.isfinite(np.asarray(obs["self"])))
    assert np.all(np.isfinite(np.asarray(obs["bodies"])))


def test_instrument_panel_is_live_on_a_contested_level(setup):
    lvl, task, st = setup
    panel = np.asarray(E.instrument_panel(lvl, task, st))
    assert panel.shape == (PANEL,)
    # eta is positive, and below 1 at this point because the star's and the
    # giant's pulls add rather than cancel here (near a saddle eta exceeds 1 --
    # see test_contested).  The other three are strictly positive readouts, so a
    # zeroed panel is unambiguously the ablation and not a coincidence.
    assert 0.0 < panel[0] < 1.0
    assert np.all(panel[1:] > 0.0)


def test_instruments_off_zeroes_exactly_four_channels(setup):
    lvl, task, st = setup
    on = {k: np.asarray(v) for k, v in E.observe(lvl, task, st).items()}
    try:
        E.INSTRUMENTS = False
        off = {k: np.asarray(v) for k, v in E.observe(lvl, task, st).items()}
        panel_off = np.asarray(E.instrument_panel(lvl, task, st))
    finally:
        E.INSTRUMENTS = True

    assert np.all(panel_off == 0.0)
    # identical shape, so an ablated run stays comparable with a full one
    for k in on:
        assert on[k].shape == off[k].shape, k
    # everything except `self` is untouched
    for k in ("bodies", "zones", "body_mask", "zone_mask"):
        np.testing.assert_array_equal(on[k], off[k])

    changed = np.nonzero(on["self"] != off["self"])[0]
    assert changed.tolist() == list(range(SELF_HEAD - PANEL, SELF_HEAD)), changed
    assert np.all(off["self"][changed] == 0.0)
    # and the panel really was the only difference
    keep = np.setdiff1d(np.arange(on["self"].size), changed)
    np.testing.assert_array_equal(on["self"][keep], off["self"][keep])


def test_instruments_flag_does_not_change_the_dynamics(setup):
    """The panel is a readout, not a force: flipping it must not move the ship."""
    lvl, task, st = setup
    act = jnp.array([0.4, 0.7])
    a, _, _ = E.step(lvl, task, st, act)
    try:
        E.INSTRUMENTS = False
        b, _, _ = E.step(lvl, task, st, act)
    finally:
        E.INSTRUMENTS = True
    np.testing.assert_array_equal(np.asarray(a.ship.pos), np.asarray(b.ship.pos))
    np.testing.assert_array_equal(np.asarray(a.ship.vel), np.asarray(b.ship.vel))
