# 0003 — The task was rebuilt because thrust dwarfed gravity

**Date** 2026-09-25 · **Status** accepted

## Context
Run v2 reached 100/100/81/91% across tiers and looked like a success. Inspecting
the flights showed near-straight lines: path/straight-line ratio 1.14, half the
tank unused. The agent had learned one strategy — point at the target and burn —
because it could. Ship delta-v was ~120 u/s against orbital speeds of 10–40 u/s,
so gravity was a rounding error on the ship's own authority.

## Decision
Rebuild the task rather than the agent: waypoint tours (up to 3, later 6, order
free late in training), per-mission thrust and fuel sized against the
direct-flight cost and often *below* it, nine level families.

## Consequences
- v3 scores are not comparable to v1/v2 (suites and the observation shape
  changed). Old checkpoints cannot be re-scored.
- v3: easy 95 / medium 55 / hard 21 / blind 43% against pilot 92/45/12/40.
- The deeper lesson, which took until 2026-10-01 to state properly: sizing the
  tank against the *direct-flight* estimate makes the budget a multiple of the
  very strategy we wanted to make insufficient. See [0004](0004-model-demand.md).
