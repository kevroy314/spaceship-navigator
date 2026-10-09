# 0019 — The 600 s clock hurt the atlas; fix the instruments, not the clock

**Date** 2026-10-04 · **Status** accepted

## Context
The causal atlas was rerun at 600 s on a long-horizon pool, with the
`p_contested` no-op, the PRNG collision, the `one-exact-boost` plumbing gap and
alpha's sample starvation all fixed ([0014](0014-sampler-bugs.md),
[0018](0018-chaos-needs-close-encounters.md)). Lesson coverage went **down**:
3 lessons with an example, against v1's 5.

## What the run measured

| | v1 (150 s) | v2 (600 s) |
|---|---|---|
| lessons with an example | 5 of 8 | **3 of 8** |
| median sufficiency | 0.0029 | **0.0024** |
| levels with sufficiency 0 | — | 6 of 24 |
| alpha measurable | 6 of 24 | 3 of 24 |
| alpha trustworthy | **0 of 5** | **3 of 3** |
| A below 1 | — | 16 of 24 |
| lesson distribution | spread | gravity-assist 12, none 6, free-ride 4, conics 2 |

Four distinct failures, each with its own cause:

1. **Chaos: impossible from a longer clock.** Bound orbits, so a perturbation
   oscillates rather than diverging — [0018](0018-chaos-needs-close-encounters.md).
2. **Sparse causal maps: the one-burn family, not the clock.** Four times the
   flight time moved median sufficiency from 0.0029 to 0.0024, i.e. not at all,
   and six levels now win *nowhere* in their own parameter box. A two-burn family
   (five parameters) is the fix.
3. **The timing axis cannot ask its own question.** The burn-time axis spans
   450 s in 20 cells, so one cell is **22.5 s** against a human reaction
   threshold of **0.2 s**. The smallest expressible window is ~100x the limit it
   is tested against, so `one-exact-boost` can never match — measured, 0 of 24
   candidates had a window under 0.2 s. That question belongs to
   `probe.delay_window`, which perturbs the best plan's start by single control
   ticks (0.07–0.8 s). Putting `window_s` in the volume scalars conflated "how
   many strategies exist" with "how tight is the timing".
4. **The taxonomy collapsed into one band.**
   > **Wrong — corrected by [0020](0020-the-clock-was-not-the-cause.md).** A is
   > *identical* at both clocks for all eight specs; T sits below both clips. The
   > cause is the candidate mix: four of seven sources were low-thrust specs
   > where A < 1 occurs 45–80% of the time.
 The longer clock admits slow tours
   that `feasible` previously rejected, and a larger flight time T lowers
   A = dv/(g*T): median A is now 0.30, with 16 of 24 below 1, which is the
   `gravity-assist` signature. The clock shifted the candidate *population*, not
   only the measurement.

## The one real gain
Where alpha is measurable it is now trustworthy — tens of flips per epsilon
instead of nought to five, f(eps_min)/saturation of 0.01–0.14, so comfortably
sub-saturation — and all three values are **0.53, 0.61, 0.74**, above the 0.45
resolvable threshold. The win/lose boundary is **smooth, not fractal**.

With lambda ~ 0 that is a consistent picture: **this game is quasi-integrable.**
The `lottery` band is empty because the physics does not produce it. That is a
result, not a measurement failure, and it supersedes the open question left by
[0009](0009-probability-field-and-alpha.md).

## Decision
- **Measure the atlas at 150 s.** The instruments were calibrated for it and
  three of the four regressions above are artefacts of the longer clock.
- **Keep 600 s available for gameplay**, which is what it was asked for — more
  reachable waypoints and genuinely branching routes. It is a game setting, not a
  measurement setting, and the two need not match.
- **Fix the instruments before rerunning:** a two-burn strategy family, and
  `delay_window` rather than a grid cell for anything compared against a human
  timing limit.
- Episode length must be a **per-pool** property rather than a global constant,
  or gameplay and measurement cannot use different clocks. Currently
  `MAX_EPISODE_TIME` is global and `LEVEL_HORIZON` must contain it.

## Evidence
`data/curriculum/causal_v2.json`, 24 candidates, pool `long`, 2291 s for volumes
and alpha. Artifact: *What Each Level Forces*.
