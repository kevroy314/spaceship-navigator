"""Level 0: `best_coast_task` must produce a tour a zero-action rollout completes.

The generator runs the ship ballistically and *places* the targets on the curve
it traces, so arrival is guaranteed by construction rather than by search.  If
this ever fails the generator is wrong, not the test — the whole point of the
module is that the free ride is not a lucky outcome.
"""

import jax
import jax.numpy as jnp
import numpy as np
import pytest

from spacenav import baselines as B
from spacenav import constants as C
from spacenav import env as E
from spacenav.levels import coastpath as CP
from spacenav.types import ARRIVED, REASONS
from helpers import hierarchy_level

# Small but still a real tour: enough ticks that the six targets land at least
# ~100 ticks apart (`coast_task` only accepts an arc that stays live for a third
# of the horizon), few enough candidates that the test runs in seconds.
CFG = CP.CoastConfig(targets=C.MAX_WAYPOINTS, candidates=8)


@pytest.fixture(scope="module")
def built():
    lvl = hierarchy_level()
    task, ok, score = jax.jit(lambda k: CP.best_coast_task(k, lvl, CFG))(jax.random.PRNGKey(0))
    return lvl, task, bool(ok), float(score)


def test_best_coast_task_is_valid(built):
    lvl, task, ok, score = built
    assert ok, "no ballistic arc on this level survived long enough to hang targets on"
    assert int(np.sum(np.asarray(task.wp_active))) == CFG.targets
    assert score > 0.0, "the kept arc should bend: a straight coast scores ~0"
    # every target rides a body, so nothing is pinned to empty space
    anchors = np.asarray(task.wp_anchor)[np.asarray(task.wp_active)]
    assert np.all(anchors >= 0)


def test_zero_action_rollout_completes_the_tour(built):
    """The defining property of Level 0."""
    lvl, task, ok, _ = built
    assert ok
    final, trace = jax.jit(lambda: E.rollout(lvl, task, B.coast(lvl, task)))()
    status = int(final.status)
    assert status == ARRIVED, REASONS[status]
    visited = np.asarray(final.visited) | ~np.asarray(task.wp_active)
    assert visited.all(), np.asarray(final.visited)
    # the ship never touched the engine, so the tank is untouched
    assert float(final.ship.fuel) == pytest.approx(float(task.fuel))
    assert float(final.costs.fuel) == pytest.approx(0.0)
    # and it finished inside the clock
    assert float(final.costs.time) < C.MAX_EPISODE_TIME


def test_targets_are_collected_in_order(built):
    """The targets are strung along one arc, so the sequential tour is the arc."""
    lvl, task, ok, _ = built
    assert ok
    _, trace = jax.jit(lambda: E.rollout(lvl, task, B.coast(lvl, task)))()
    reached = np.asarray(trace["visited"])
    assert np.all(np.diff(reached) >= 0), "the visited count must never go backwards"
    assert reached[-1] == CFG.targets


def test_ballistic_arc_matches_a_zero_throttle_rollout(built):
    """`ballistic` is the generator's model of the engine-off flight; it has to be
    the same integrator the environment runs, or the placement is on a curve the
    ship does not fly."""
    lvl, task, ok, _ = built
    assert ok
    start = E.Start(task.snapshot, task.start_pos, task.start_vel, task.start_angle,
                    jnp.bool_(True))
    n = 400
    xs, _, live = jax.jit(lambda: CP.ballistic(lvl, start, n))()
    _, trace = jax.jit(lambda: E.rollout_actions(lvl, task, jnp.zeros((n, 2))))()
    np.testing.assert_allclose(np.asarray(xs), np.asarray(trace["pos"]), atol=1e-3)
    assert bool(live[0])


def test_a_coast_task_is_level_zero_for_the_lesson_matcher(built):
    """`lessons.classify` must call this a free ride, not a defective level."""
    from spacenav import lessons as L
    lvl, task, ok, _ = built
    assert ok
    final, _ = jax.jit(lambda: E.rollout(lvl, task, B.coast(lvl, task)))()
    assert L.classify(dict(coast_ok=bool(final.status == ARRIVED))) == "free-ride"
