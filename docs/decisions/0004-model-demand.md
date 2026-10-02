# 0004 — Model demand D = eta / A as the generator's dial

**Date** 2026-10-01 · **Status** accepted

## Context
We wanted levels where a human's coarse mental model fails but computation
succeeds, and we wanted to *construct* them rather than generate-and-test. The
mesoscale model in this domain has a name: patched conics — only the body you
orbit matters. `physics.gravity_at(topk=k)` instantiates it exactly.

## Decision
Score a level by one closed-form, simulation-free quantity:

    eta = |a_full - a_dominant| / |a_full|     the force a one-body model discards
    A   = dv_tank / (|g| * T)                  the tank in units of gravity's impulse
    D   = eta / A                              share of the tank a wrong model costs

Low D: a coarse plan is recoverable. High D: correcting the model error would
burn the budget. `demand.retune` **inverts** it — fuel = eta*|g|*T/(D*accel) —
so a level can be placed on a chosen rung by solving, not searching.

## Consequences
- Thousands of candidates can be scored for free; expensive planning runs only
  on levels already known to sit where we want them.
- Measured: an ordinary mission sits at D ~ 0.0014, with eta = 0.0036 and
  A = 3.8. So gravity supplies about a quarter of the tank, but a *single body*
  explains 99.6% of it. That is why patched conics works here, and why breaking
  it took deliberate placement.
- Validated: pooled over 24 levels per band, a patched-conic planner and a
  full-field planner tie below D = 0.1 (32% vs 42%, Fisher p = 0.74) and
  separate above it (13% vs 54%, p = 0.005), with the coarse planner at 0/8 on
  the top rung.
- **Per-rung numbers (n = 8) are not significant.** Only the pooled test is.
  See [0012](0012-evaluation-statistics.md).
- Fuel is added-only when retuning. Reaching a rung by starving the tank
  produces an infeasible level dressed up as a demanding one.

## Evidence
`spacenav/demand.py`, `scripts/make_curve.py`. Artifact: *Two Routes Per Rung*.
