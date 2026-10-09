# 0023 — The planner flies one-stop missions and not two-stop ones

**Date** 2026-10-09 · **Status** accepted

## Measurement
Atlas v4, 23 candidates, 150 s clock, pool `long`. A level counts as flown if a
cross-entropy plan arrived **or** any member of the one- or two-burn strategy
family won. Level-0 coast missions are excluded: they carry six targets but are
solvable by construction, so they say nothing about the planner.

| spec | stops | flown |
|---|---|---|
| 1w0m | 1 | 3/3 |
| 1w100m | 1 | 3/3 |
| 1w100mc | 1 | 5/5 |
| **one stop, total** | | **11/11** |
| 2w100mc | 2 | 1/3 |
| 2w50m | 2 | 0/3 |
| 2w50ml | 2 | 0/2 |
| 2w100mcl | 2 | 0/2 |
| **two stops, total** | | **1/10** |

Planner configuration: 48 segments over 2250 ticks (3.1 s each), 96 samples,
7 iterations, seeded from the scripted pilot and now also from an explicit
zero-thrust candidate.

## Why this matters past the atlas
The CEM planner is two things at once in this project:

1. **The curriculum's yardstick.** Reference flights score what "good" means, and
   a level is called unfair when the planner cannot fly it
   ([0006](0006-two-axis-bands.md)'s `opaque` band).
2. **The intended teacher** in the search-guided architecture
   ([0010](0010-planner-as-teacher.md), [0015](0015-architecture-pick.md)): a
   learned policy seeds a search over the true simulator and the search's output
   is distilled back into the policy.

Both break on multi-stop tours. A teacher that cannot fly the mission has nothing
to teach on exactly the missions the curriculum is built around — and worse, it
means **the hard-tier difficulty may be miscalibrated rather than hard**:
`ppo_v3` scored 21% on the three-stop tier, and if the planner is near 10% on
two stops then 21% may be at or above the reference, not far below it.

That inverts how that result was read in
[0003](0003-task-rebuilt-around-gravity.md), which treated 21% as the agent
falling short.

## Candidate causes, untested
- **Search budget.** 48 segments is 96 parameters against 96 samples per
  iteration and 7 iterations. Thin, and it degrades as the parameter count rises.
- **Joint optimisation of a sequential task.** CEM optimises all 48 segments at
  once. A two-stop tour is two arrivals in sequence, and the score only
  distinguishes them through `gap = (waypoints left - 1) + arrival gap`, so until
  the first waypoint is reached every candidate sits on a plateau above 1.
- **Receding horizon not used.** `opt.necessity.adaptivity` already shows
  re-planning from the true mid-flight state recovers flights an open-loop plan
  loses. The planner used for references does not do this; the architecture in
  [0010](0010-planner-as-teacher.md) assumes it.

## Next experiment, before any teacher work
Fly the 10 two-stop candidates three ways and compare: (a) current CEM,
(b) CEM with 4x the samples and 2x the iterations, (c) receding-horizon CEM that
re-plans at each waypoint. If (c) wins, the reference planner should be
receding-horizon everywhere and [0010](0010-planner-as-teacher.md) is unaffected.
If only (b) wins, the budget was the problem and the cost of a teacher rises. If
neither flies them, the two-stop tours are genuinely infeasible at these fuel
budgets and the *task* needs recalibrating — which would be the most important
of the three answers.

## Corroboration from the eval suites (added during the ppo_v4 launch)
Rebuilding the frozen suites at the current task shape gives scripted-pilot
success of **85 / 35 / 8 / 32 %** across easy / medium / hard / blind, against the
design targets of ~90 / 50 / 15 / 40 % recorded in the `training-run` skill. Every
tier is harder than intended and `hard` is at 8%.

That is a second, independent route to the same conclusion. `ppo_v3` scored 21%
on `hard` — above the reflex pilot's 8%, and plausibly above a planner that flies
one two-stop tour in ten. So the agent was **beating both baselines** on that
tier, which inverts the reading in
[0003](0003-task-rebuilt-around-gravity.md) that treated 21% as falling short.

Consequence for the curriculum: the tier thresholds need resetting against
measured baselines rather than intentions, and "the agent is weak on hard tours"
should not be asserted again without a baseline beside it.

## Evidence
`data/curriculum/causal_v4.json`; artifact *What Each Level Forces* (v3).
Alpha was measurable on 3 of 23 and read 0.79, 0.86, 1.02, with flip counts in
the tens per epsilon and f(eps_min) at most 3% of the saturation bound — a fourth
run agreeing the boundary is smooth.
