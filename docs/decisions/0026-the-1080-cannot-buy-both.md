# 0026 — Scale and steps compete for the same card: 285k params is a budget choice, not a design one

**Date** 2026-10-14 · **Status** accepted

## Context
The standing question was whether the plateau on `medium`/`hard` is a
representation failure or an under-training failure. Our network is 285,322
parameters against SimbaV2's ~5M over 57 tasks, and our 393M env steps against
Kinetix's 376B — roughly 18x under-parameterised and 1000x under-trained
relative to the published results we are reasoning from. That framing implies
"make it bigger *and* train it longer", so the cost was measured on the actual
card before believing it was available.

`scripts/` scaling probe, GTX 1080 Ti, batch 131,072:

| d_model | params | s/update | steps/s | 3000 updates |
|---|---|---|---|---|
| 128 | 285,322 | 4.11 | 31,854 | 3.4 h |
| 256 | 698,634 | 9.00 | 14,559 | 7.5 h |
| 384 | 1,308,554 | 27.45 | 4,775 | **22.9 h** |

Throughput falls faster than parameters rise: 4.6x the parameters costs 6.7x
the wall clock, so steps/s drops 6.7x. The two hypotheses are not independently
affordable on this GPU — at d_model 384 a single seed is a day, and ADR 0012
requires >=5 training seeds to separate seed variance from episode variance.

## Decision
Keep d_model 128 for anything that needs multiple seeds on the 1080 Ti, and
treat "bigger network" as an experiment that requires different hardware rather
than one we have declined. Record the price so the choice is not mistaken for a
belief that 285k parameters is sufficient — it is what fits.

## Consequences
- Makes the architecture question answerable only on a bigger card. An RTX 3090
  (24 GB) is reported available on the spare-compute cluster; at ~3-4x the
  1080 Ti it brings 5 seeds at d_model 256 into a weekend rather than a week.
- Rules out, for now, the SimbaV2-scale comparison that would actually test the
  representation hypothesis.
- Does **not** show that d_model 128 is the cause of the plateau, and does not
  show it is not. It prices the test; it does not run it.
- The probe measures s/update at a fixed batch. A larger model with a smaller
  batch was not measured and may sit at a better point.
- Confounded with ADR 0025: part of the plateau is a tier that is infeasible at
  any capacity, so the expected return on scale is lower than the raw
  SimbaV2/Kinetix gap suggests.

## Evidence
- `scratchpad/scaleprep.py`, 3 configurations, logged in
  `scratchpad/scaleprep.log`. One timing run per d_model.
- SimbaV2: 0.892 normalised over 57 tasks at ~5M params (arXiv:2502.15280).
- Kinetix: 376B env steps, 1M parallel envs (arXiv:2410.23208).
- ADR 0012 (`0012-evaluation-statistics.md`) — the >=5-seed requirement.
- ADR 0025 — the infeasible-tier confound.
