# 0020 — The clock did not collapse the taxonomy; the candidate mix did

**Date** 2026-10-08 · **Status** accepted · **Corrects** [0019](0019-revert-the-clock-for-measurement.md)

## Context
[0019](0019-revert-the-clock-for-measurement.md) listed four causes for the v2
atlas regression. Its fourth claim was that the 600 s clock admitted slow tours
whose larger flight time T lowered A = dv/(g*T) below 1, which is why
`gravity-assist` took 12 of 24 candidates. That claim is wrong, and checking it
was cheap.

## Measurement
A per spec at both clocks, 20 episodes each, `scratchpad/abyspec.py`:

| spec | accel | A@600 | A@150 | A<1 | feasible @150 |
|---|---|---|---|---|---|
| 1w0m | 2.24 | 6.29 | **6.29** | 5% | 100% |
| 2w50m | 2.24 | 2.40 | **2.40** | 30% | 100% |
| 1w100mc | 2.24 | 3.51 | **3.51** | 15% | 100% |
| 2w100mc | 2.24 | 1.85 | **1.85** | 40% | 100% |
| 1w100m | 2.24 | 1.47 | **1.47** | 45% | 100% |
| 1w100mcl | 0.79 | 1.17 | **1.17** | 45% | 100% |
| 2w50mcl | 0.79 | 0.95 | **0.95** | 55% | 85% |
| 2w100mcl | 0.79 | 0.52 | **0.52** | 80% | 75% |

**A is identical at both clocks, to the digit, for every spec.** T = t_est sits
below both clips, so the clock cannot move A. Its only real effect is admitting
15–25% more low-thrust multi-stop tours (feasibility 75–85% at 150 s).

## The actual cause
The candidate sources. Four of seven `SOURCES` entries used low-thrust specs
(accel 0.79, where A < 1 occurs 45–80% of the time), and three of those were
additionally restricted to the `wide` comparable-mass families, whose larger |g|
lowers A further. `gravity-assist` is `A < 1`, so the mix selected for it.

## Consequences
- **Rebalance the candidate mix**, not the clock: sample the A axis deliberately,
  with strong-thrust ordinary specs for the high-A bands and only a couple of
  low-thrust entries for the low-A end.
- **Make the lesson tests mutually exclusive** where they currently overlap.
  `gravity-assist` (A < 1) was pre-empting `conics` (D >= 0.1) and `leading` by
  match order alone, so those lessons could not appear on any level that was also
  fuel-tight. Now `gravity-assist` additionally requires low eta: the lesson is
  "fuel is the binding constraint", so a fuel-tight level whose *model* error is
  the harder problem belongs to `conics`. This is the overlap weakness recorded
  in [0007](0007-lessons-and-instruments.md), now addressed rather than noted.
- The three other causes in [0019](0019-revert-the-clock-for-measurement.md)
  stand: bound-orbit physics for chaos, the one-burn family for sparse maps, and
  a 22.5 s grid cell against a 0.2 s threshold for the timing branch.
- The clock decoupling done for [0019](0019-revert-the-clock-for-measurement.md)
  is still worth keeping — `demand.terms` and `survey` now take `episode_s`
  explicitly, so a measurement is never silently inheriting the game clock. It
  just was not the fix it was billed as.

## Pattern worth naming
Three times now an effect has been attributed to the wrong factor — resonance
overlap at the wrong timescale ([0005](0005-equal-pull-not-resonance.md)), full
observability against a coordinating message
([0016](0016-communication-is-equilibrium-selection.md)), and the clock here.
Each was locally plausible and each took one cheap measurement to kill. The
standing mandate in CLAUDE.md asks for exactly that check; it is working, and the
lesson is that a plausible mechanism deserves a measurement *before* it reaches a
decision record, not after.
