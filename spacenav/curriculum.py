"""Bands, and the curriculum built from them.

A level is placed on two axes, measured by `spacenav.opt.probe`, and deliberately
never collapsed into one difficulty number:

* **model scale** — the coarsest model of the forces under which a planner can
  still fly the level successfully in the true world.  `k=1` is patched conics,
  "only the body I am orbiting matters", which is what human orbital intuition
  is.  A level needing the full field is one where no simplified picture of the
  forces will do, however well you fly.
* **execution margin** — how much timing slack the right plan leaves, against
  measured human limits (`probe.HUMAN`): 200 ms to react, ~20 ms of motor
  timing jitter, 67 ms of control quantisation at 15 Hz.

Crossing them gives five bands, and the two failure modes on the right-hand
side are *different kinds of impossible*:

    band        model scale   execution     who can fly it
    ----------  ------------  ------------  -----------------------------------
    trivial     patched       wide          a reflex: point and burn
    human       patched       wide          a person who plans
    calculated  full field    wide          a person who computes the parameters
                                            first; intuition alone never gets
                                            there, but hands are not the limit
    precision   any           narrow        an agent only: the plan is findable
                                            and no human hand can hold it
    opaque      none works    -             nobody; rejected as unfair

`calculated` is the interesting branch: the level is humanly *flyable* but not
humanly *intuitable*.  `precision` is the genuinely non-human one.  Keeping them
apart matters because they need opposite level design — widen the window of a
`precision` level and it becomes `calculated`; simplify the force field of a
`calculated` level and it becomes `human`.
"""

from typing import NamedTuple

import numpy as np

from spacenav import constants as C
from spacenav.opt.probe import HUMAN, ProbeConfig

TRIVIAL, HUMAN_BAND, CALCULATED, PRECISION, OPAQUE = \
    "trivial", "human", "calculated", "precision", "opaque"
BANDS = (TRIVIAL, HUMAN_BAND, CALCULATED, PRECISION, OPAQUE)
HUMAN_PATH = (TRIVIAL, HUMAN_BAND)
BRANCHES = (CALCULATED, PRECISION)


def window_seconds(delays, delay_ticks):
    """Widest uniform start delay the plan tolerates, in seconds.

    Counted as consecutive successes from the shortest delay up: a plan that
    survives 1 and 2 ticks late but not 3 has a window of 2 ticks, and one that
    fails at a single tick late has no window at all.
    """
    w = 0.0
    for ok, d in sorted(zip(np.asarray(delays).tolist(), delay_ticks), key=lambda p: p[1]):
        if not ok:
            break
        w = d * C.CTRL_DT
    return w


def executable(row, cfg: ProbeConfig, human=HUMAN):
    """Can a human hand hold the right plan on this level?

    Two requirements, both measured: the launch window must be wide enough to
    aim at within a reaction time, and the flight must survive onset jitter of
    one control tick (67 ms at 15 Hz — already coarser than the ~20 ms a
    practised hand actually achieves, so this is the generous reading).
    """
    window = window_seconds(row["delays"], cfg.delays)
    jitter = np.asarray(row["jitter"])
    i = int(np.argmin([abs(j - 1.0) for j in cfg.jitters]))
    return (window >= human.window_min) and (jitter[i] >= human.jitter_pass), window, jitter[i]


def model_scale(row, cfg: ProbeConfig):
    """Coarsest model that still flies it: 1 = patched conics, len(ladder) = full field."""
    for k in cfg.ladder:
        if k != 0 and bool(row[f"ok_k{k}"]):
            return k
    return 0          # only the full field works


def band(row, cfg: ProbeConfig = ProbeConfig(), human=HUMAN):
    """Which band this (level, task) falls in."""
    solvable = bool(row["ok_full"]) or bool(row["ref_ok"])
    if not solvable:
        return OPAQUE
    ok, _, _ = executable(row, cfg, human)
    if not ok:
        return PRECISION
    if model_scale(row, cfg) != 1:
        return CALCULATED
    return TRIVIAL if bool(row["pilot_ok"]) else HUMAN_BAND


class Stage(NamedTuple):
    name: str
    band: str
    scales: tuple            # inclusive range of Hill-sphere crossings
    note: str


# The human path ramps the number of scales the flight has to reason across,
# which is the thing a person can bootstrap: one domain, then a handoff between
# two, then a hierarchy.  The branches are not ordered after it — they hang off
# the end of the path and are meant to be unreachable by extrapolating from it.
PATH = (
    Stage("reflex", TRIVIAL, (0, 1), "point and burn: one gravity well, wide margins"),
    Stage("transfer", HUMAN_BAND, (0, 1), "one well, but the tank forces a real transfer"),
    Stage("handoff", HUMAN_BAND, (1, 2), "leave one body's domain and arrive in another's"),
    Stage("hierarchy", HUMAN_BAND, (2, 9), "star to planet to moon: three scales in one flight"),
)
BRANCH_STAGES = (
    Stage("calculated", CALCULATED, (0, 9),
          "patched conics cannot fly it; compute the field and it is flyable"),
    Stage("precision", PRECISION, (0, 9),
          "the plan exists and no human hand can hold its timing"),
)


def assign(rows, ids, cfg: ProbeConfig = ProbeConfig()):
    """Band and stage every probed episode.  `rows` is a dict of arrays."""
    n = len(ids)
    out = []
    for i in range(n):
        row = {k: v[i] for k, v in rows.items()}
        b = band(row, cfg)
        ok, window, jit = executable(row, cfg)
        scales = int(row["n_scales"])
        stage = None
        for st in PATH + BRANCH_STAGES:
            if st.band == b and st.scales[0] <= scales <= st.scales[1]:
                stage = st.name
                break
        out.append(dict(
            id=ids[i], band=b, stage=stage, scales=scales,
            model_scale=model_scale(row, cfg), window_s=round(window, 3),
            jitter_1tick=round(float(jit), 3), executable=bool(ok),
            eta_mean=round(float(row["eta_mean"]), 4), eta_max=round(float(row["eta_max"]), 4),
            lam=round(float(row["lam"]), 4), horizon=round(float(row["horizon"]), 3),
            hill_depth=round(float(min(row["hill_depth"], 1e3)), 3),
            pilot_ok=bool(row["pilot_ok"]), ref_ok=bool(row["ref_ok"]),
            ok_full=bool(row["ok_full"]),
            cost=None if not np.isfinite(row["cost_full"]) else round(float(row["cost_full"]), 3),
        ))
    return out


def summary(assigned):
    """Counts per band and per stage, for a quick read on a probe run."""
    bands, stages = {}, {}
    for a in assigned:
        bands[a["band"]] = bands.get(a["band"], 0) + 1
        stages[a["stage"]] = stages.get(a["stage"], 0) + 1
    return dict(n=len(assigned), bands=bands, stages=stages)
