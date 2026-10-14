# 0025 — Two-stop tours time out at every search budget: the hard tier is infeasible, not hard

**Date** 2026-10-14 · **Status** accepted

## Context
ADR 0023 recorded that the CEM planner flew 1 of 10 two-stop tours and read
that as a planner-capacity limit. If it were capacity, more search would buy
arrivals. Three configurations were run over 12 two-stop candidates at 2250
ticks (150 s, the measurement clock of ADR 0019):

| config | search | arrived | wall | outcomes |
|---|---|---|---|---|
| (a) current | 96 samples x 7 iters | **0/12** | 22 s | 10 timeout, 1 lost, 1 crashed |
| (b) 4x budget | 384 x 14 | **0/12** | 39 s | 8 timeout, 3 lost, 1 crashed |
| (c) receding horizon | 96 x 7, replan x3 | **0/12** | 44 s | 8 timeout, 2 lost, 2 crashed |

Sixteen times the trajectory evaluations between (a) and (b) bought zero
arrivals. The dominant failure is **timeout**, not `lost` and not `crashed`:
the plans fly sensibly and do not get there in time. That is the signature of a
clock that is too short for the geometry, not of a search that cannot find the
plan.

## Decision
Treat two-stop tours as **infeasible within a 150 s episode**, not as hard.
Stop reading the planner's failure on them as a capacity result, and stop
reading the agent's ~8% on the `hard` suite as a learning failure: the suite's
2-3 waypoint tours are drawn from the same family. Before any further work on
multi-stop capability, the episode clock or the waypoint spacing has to change;
ADR 0023's reading is superseded on this point.

## Consequences
- Explains part of what looked like an architecture problem. No model capacity
  or representation fixes a task that a 3,000-trajectory-per-decision optimiser
  cannot complete in the available time.
- Corroborates ADR 0023's other finding — the hard tier is miscalibrated
  (pilot 85/35/8/32 against design targets 90/50/15/40) — with an independent
  measurement that does not involve the agent at all.
- Costs the `hard` suite its meaning as a difficulty rung until it is rebuilt.
  Agent numbers on it to date measure the tier, not the agent.
- Does **not** establish that two-stop tours are infeasible at the 900 s
  `LEVEL_HORIZON`. They were only tested at 150 s. The obvious next measurement
  is the same 12 candidates at a longer clock, which would separate "needs more
  time" from "needs a better plan".
- n=12 candidates is small. The result is a clean 0/36 across configs, so the
  direction is not in doubt, but the per-config rates have wide intervals.

## Evidence
- `scratchpad/twostop.py`, 12 two-stop candidates x 3 configs = 36 plans,
  2250 ticks each. Logged in `scratchpad/twostop.log`.
- ADR 0023 (`0023-planner-fails-multi-stop.md`) — the 1/10 result this revises.
- ADR 0019 (`0019-revert-the-clock-for-measurement.md`) — why 150 s is the
  measurement clock.
