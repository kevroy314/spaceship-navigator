# 0032 — Two-stop tours are flyable, the clock does help, and 0025's "infeasible" was an artifact of n=12

**Date** 2026-10-25 · **Status** accepted · **Supersedes 0025**

## Context
ADR 0025 flew 12 two-stop candidates at a 150 s clock under three search
budgets, got 0/12 every time with TIMEOUT dominating, and concluded the family
was **infeasible at 150 s rather than hard**. It also set the obvious next
measurement: the same family at a longer clock.

Re-running with 32 candidates per clock instead of 12, and a modestly larger
search (128 samples x 10 iterations against 0025's 96 x 7), control resolution
held at 3.1 s per segment so a longer clock buys more segments rather than
coarser ones:

| clock | ticks | segments | arrived | timeout | crashed / destroyed / out of fuel |
|---|---|---|---|---|---|
| 150 s | 2250 | 48 | **5/32** (16%) | 18 | 4 / 1 / 4 |
| 300 s | 4500 | 96 | **8/32** (25%) | 12 | 9 / 0 / 3 |
| 600 s | 9000 | 192 | **13/32** (41%) | 9 | 2 / 3 / 5 |
| 900 s | 13500 | 288 | **11/32** (34%) | 5 | 5 / 6 / 5 |

Two corrections fall out.

**0025's headline was wrong.** The rate at 150 s is not zero, it is 5/32 = 16%.
At that rate, drawing 0 of 12 has probability 0.84^12 = **0.14** — an ordinary
outcome, not evidence of infeasibility. The three budgets in 0025 flew the
*same* 12 candidates, so they were near-perfectly correlated and did not give
three independent chances to notice.

**The clock does help, and then stops helping.** Timeouts fall monotonically,
18 -> 12 -> 9 -> 5. But the episodes a longer clock frees up do not all become
arrivals: hazard endings rise from 9 to 16 over the same range, and arrivals
peak at 600 s and fall back at 900 s. More time means more exposure.

## Decision
Retract "infeasible". Two-stop tours are **hard and flyable**: ~16% at the
measurement clock, ~41% at 600 s. Treat **600 s** as the useful ceiling for
multi-stop work, not 900 s, and treat hazard exposure rather than the clock as
the binding constraint past that point.

The hard-tier recalibration is now a real choice rather than a forced one: the
tier is not unflyable, it is calibrated against a planner that only lands 16%
of these tours in the time allowed.

## Consequences
- Partially rehabilitates ADR 0023's original reading. Its 1/10 was a small
  sample of a ~16% process, which is unremarkable; its *miscalibration*
  conclusion stands and is now better supported.
- Agent numbers on the `hard` suite become interpretable again, with the caveat
  that the reference planner itself lands only 16% at this clock. 10.7% for the
  agent against 16% for a 3,000-trajectory optimiser is a different and much
  less alarming statement than 10.7% against an impossible task.
- **Costs the confident claim in two shipped records.** 0025 was written with a
  dramatic result and n=12, which is precisely the case the project's third
  mandate says to attack before reporting. The attack was recorded as "the
  obvious next measurement" and then not run for eight days.
- The episode clock is a **global constant** (`spacenav/env.py:446` compares the
  tick against `C.MAX_EPISODE_TICKS`), so this sweep patched it per process. A
  per-pool or per-task clock, which would let gameplay run long while
  measurement stays at 150 s, is still unbuilt.
- n=32 per clock gives roughly +-9 pp at these rates, so 600 s vs 900 s
  (13 vs 11) is well inside noise; the peak's *location* is not established,
  only that the curve flattens.
- Hazard endings were pooled as one column here. Which hazard grows with the
  clock — collision, radiation, fuel — is not separated.

## Evidence
- `scripts/twostop_clock.py` and `scripts/clock_sweep.sh`, 32 candidates x 4
  clocks = 128 plans, specs `2w50m` / `2w100mc` / `2w50ml` / `2w100mcl` from
  pool `long`, run on the cluster RTX 3090. Output in
  `data/curriculum/clock/clock_*.json`.
- ADR 0025 — superseded. ADR 0023 — partially rehabilitated.
- ADR 0019 — why 150 s is the measurement clock.
