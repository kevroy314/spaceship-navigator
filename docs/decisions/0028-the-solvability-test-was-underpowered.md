# 0028 — The pilot-solvability test failed on a 32-episode draw, not on a regression

**Date** 2026-10-14 · **Status** proposed

## Context
`tests/test_levels.py::test_tasks_are_solvable_by_pilot` fails on `main` at
commit `8e24c5d`: the reflex pilot arrives on **20 of 32** missions (0.625)
against an `assert rate > 0.7`. 122 other tests pass. The obvious reading is
that something made the levels harder or the pilot worse.

Both readings are unsupported, because the test cannot resolve the gap it
asserts:

| | value |
|---|---|
| observed | 20/32 = 0.625 |
| standard error | 0.0856 |
| 95% CI | **[0.457, 0.793]** |
| threshold | 0.70 — *inside* the CI |
| n for +-5 pp at p=0.625 (ADR 0012) | 361 |

The assertion is a 7.5 pp claim at n=32, where ADR 0012 puts +-5 pp at n ~ 384.
The test is underpowered by roughly an order of magnitude, and the failure is
two episodes wide.

What moved is identifiable. The test draws a *fixed* sample
(`build_levels(..., seed0=5000)`, `PRNGKey(1)`), and `653127f` changed which
split keys feed `accel` and `budget`. ADR 0027 measured that this did **not**
change the task distribution (KS on D, p = 0.997 over 4,096 tours per arm), so
the draw moved while the distribution stayed put. A deterministic test on one
underpowered sample landed on an unlucky 32.

## Decision
Raise the sample to n=384 in both calls, keeping the 0.7 threshold. Do not
relax the threshold to make the test pass — if the true rate is near 0.625 then
the pilot or the level family genuinely misses the design intent, and that is a
finding in the same family as ADR 0025, not something to hide behind a wider
bound.

`Status: proposed` until the n=384 rate is measured; the run is queued behind
`ppo_v5_noinstr` because only one JAX process may hold the GPU.

## Consequences
- Buys a test that means what it asserts, at a cost: n=384 is 12x the rollouts
  of n=32 in a suite that already takes 8 minutes. The rollout is vmapped, so
  the wall-clock cost is expected to be well under 12x, but it is not free and
  was not measured before this record was written.
- A fixed-seed test over a near-binary outcome needs its n chosen from ADR 0012,
  not from what is quick. Other small-n assertions in the suite have not been
  audited for the same defect and probably should be.
- Does **not** yet establish the pilot's true solvability rate. If it comes
  back near 0.625 the threshold is the thing that is wrong, and this record
  must be superseded rather than quietly amended.
- Does not explain the ppo_v4 regression. The two share a cause only in the
  sense that both were read as regressions before their n was checked.

## Evidence
- `tests/test_levels.py::test_tasks_are_solvable_by_pilot` at `8e24c5d`:
  `assert np.float64(0.625) > 0.7`, 1 failed / 122 passed in 473 s.
- `tests/test_levels.py:22` — `_pilot_success(..., n=32)`, the default now
  overridden.
- ADR 0012 (`0012-evaluation-statistics.md`) — n ~ 384 for +-5 pp.
- ADR 0027 — the distribution is unchanged (KS p = 0.997, n = 4,096 per arm),
  which is what makes "the draw moved" the available explanation.
- `653127f` — the commit that moved the draw.
