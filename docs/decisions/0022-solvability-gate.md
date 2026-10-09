# 0022 — A level nobody can fly gets no lesson

**Date** 2026-10-09 · **Status** accepted

## Context
The v3 atlas run (23 candidates, 150 s measurement clock, pool `long`) assigned
five lessons, up from three in v2. But `conics` took 12 of 23, and inspecting
those rows showed why: **10 of 23 candidates had `sufficiency` 0.000 on both the
one- and two-burn families and no arriving cross-entropy plan.** Nothing flew
them. Eight of those ten were labelled `conics`, because `conics` fires on
D >= 0.1 alone.

So a third of the dominant band was unsolvable levels wearing a lesson label.
Published as-is the page would have been actively misleading.

## Decision
`lessons.classify` now gates on solvability: a level with no arriving plan and no
winning member of the strategy family is **unclassified**, not assigned.
`free-ride` is exempt, because a coast that arrives is its own proof.

    solvable(f) = plan_arrived or sufficiency > 0 or sufficiency_1burn > 0

## Consequences
- The band counts become honest. An unsolvable level is a generation outcome to
  report, not a difficulty tier — the same argument that put `opaque` in the
  band scheme ([0006](0006-two-axis-bands.md)), which the *lesson* scheme had
  simply never applied.
- Three tests added: that an unflown level gets no lesson, that a winning burn
  counts as flown even when the plan fails, and that `free-ride` is exempt.
- The v3 numbers in this record predate the gate. The band counts will change on
  the next run and these are the before figures.

## Two other defects found in the same run

**The planner could not do nothing.** `cem` seeds its samples from the pilot mean
and the incumbent best; throttle is `sigmoid(raw)` with raw ~ N(mu, sigma), so an
all-zero-throttle plan is essentially never drawn. On a Level-0 coast level that
plan *is* the optimum, so the search failed on the easiest levels in the game —
observed on `long-1664-121212-coast`, whose plan did not arrive and whose
necessity profile came back undefined as a result. Fixed by carrying an explicit
zero-thrust candidate alongside the mean and the incumbent. The strategy volume
has the same blind spot by construction: `dv_lo = 0.15`, so the smallest burn it
tests is 15% of the tank and the family cannot express a null action.

**A threshold calibrated against a win rate that does not exist.**
`dead-heading` required `sufficiency > 0.04`. Per
[0021](0021-sparsity-is-the-answer.md) the honest win rate is ~1%, and over the
v3 survey non-coast sufficiency has a median near 0.001 with a maximum of 0.044 —
so the threshold fired on one level in 23, and ordinary point-and-burn levels
classified as `None`. Now 0.01, from a stated rule rather than a tuned number:
*at least one guess in a hundred arrives*. The distribution it came from is in
this record so the next person can see what it was fitted against.

`one-exact-boost`'s `concentration > 0.55` survives inspection: measured
concentration over v3 runs 0.15-0.64 with a median near 0.35, so 0.55 selects the
top couple of levels, which is what "one irreplaceable decision" should mean.
`leading`'s `|leading| > 0.35` is still unvalidated against a distribution.

## Evidence
`data/curriculum/causal_v3.json`, 23 candidates, 1484 s. alpha was measurable on
3 of 23 and read **0.79, 0.86, 1.02** — all above the 0.45 resolvable threshold,
with 1.02 at the smooth codimension-one limit. Three independent runs now agree
the boundary is smooth and the game quasi-integrable
([0018](0018-chaos-needs-close-encounters.md),
[0019](0019-revert-the-clock-for-measurement.md)).
