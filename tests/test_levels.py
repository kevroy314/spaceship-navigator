import collections

import jax
import numpy as np
import pytest

from spacenav import baselines as B
from spacenav import env as E
from spacenav.levels.build import build_levels
from spacenav.levels.families import ALL_FAMILIES
from spacenav.types import ARRIVED, RUNNING


@pytest.mark.parametrize("family", list(ALL_FAMILIES))
def test_family_builds_and_is_mostly_valid(family):
    levels, meta, reasons = build_levels(family, 8, seed0=1000, batch=16, max_tries=3)
    assert levels.mass.shape[0] == 8
    assert reasons["tried"] <= 32
    assert all(m["drift"] < 2e-3 for m in meta)


def _pilot_success(family, cfg, n=32):
    lv, _, _ = build_levels(family, n, seed0=5000, batch=n)
    keys = jax.random.split(jax.random.PRNGKey(1), n)
    tasks, ok = jax.vmap(lambda k, l: E.sample_task(k, l, cfg))(keys, lv)
    fin, _ = jax.jit(jax.vmap(lambda l, t: E.rollout(l, t, B.pilot(l, t))))(lv, tasks)
    st = np.asarray(fin.status)[np.asarray(ok)]
    return np.mean(st == ARRIVED), collections.Counter(st.tolist())


def test_tasks_are_solvable_by_pilot():
    """The reflex pilot should fly most missions when the tank is comfortable.

    Not with the default TaskConfig: its deltav_budget is the *training* band,
    which reaches 0.55x the direct-flight estimate on purpose, and a policy that
    does not ration fuel cannot fly that — it burns the tank dry and is lost.
    That is a designed property of the curriculum, so assert it explicitly here
    rather than leaving it to break this test later.
    """
    rate, counts = _pilot_success("sol_like", E.TaskConfig(deltav_budget=(0.9, 1.45)))
    assert rate > 0.7, counts
    tight, tight_counts = _pilot_success("sol_like", E.TaskConfig(deltav_budget=(0.55, 0.8)))
    assert tight < rate, (counts, tight_counts)


def test_parking_orbits_hold():
    lv, _, _ = build_levels("binary_star", 32, seed0=7000, batch=32)
    keys = jax.random.split(jax.random.PRNGKey(2), 32)
    tasks, ok = jax.vmap(lambda k, l: E.sample_task(k, l, E.TaskConfig()))(keys, lv)
    fin, _ = jax.jit(jax.vmap(lambda l, t: E.rollout(l, t, B.coast(l, t))))(lv, tasks)
    st = np.asarray(fin.status)
    # coasting in a parking orbit should rarely end in disaster within 2 minutes
    assert np.mean(np.isin(st, [RUNNING, ARRIVED, 5])) > 0.8, collections.Counter(st.tolist())
