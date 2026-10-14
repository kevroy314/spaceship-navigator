# 0027 — The PRNG collision was real hygiene and a null result: it did not shift the training distribution

**Date** 2026-10-14 · **Status** accepted

## Context
`ppo_v4` regressed against `ppo_v3` (final success 89/34/8/25 against
95/55/21/42 on easy/medium/hard/blind). Diffing the two run configs showed they
differ by **exactly one key**: `contested_end = 0.35`, absent in v3. Same seed,
same lr, same gamma, same d_model, same 3000 updates — 28 of 29 keys identical.

`p_contested` ramps only after `frac = 0.5` (`(frac - 0.5) / 0.4`, so update
1500 of 3000), and the gap is visible from update 350. So the contested ramp
cannot explain it. Two env changes act from update 1:

1. the four instrument channels (ADR 0013);
2. the PRNG-collision fix in `653127f` — `accel` moved from `keys[K]` to
   `keys[K+5]` and `budget` from `keys[K+1]` to `keys[K+6]` under
   `split(key, K+7)` instead of `split(key, K+5)`.

The waypoint loop consumes `keys[2..K+1]`, so before the fix the engine and
tank were a deterministic function of where the last two waypoints landed. The
natural inference — and the one the fix's own commit message implies — is that
v3 trained on a distribution where fuel was correlated with geometry, i.e. an
accidentally solvable curriculum, and v4 lost it.

That inference is wrong. Measured over 4,096 tours per arm on matched levels
and matched base keys:

| | old (v3) | new (v4) |
|---|---|---|
| accel mean / sd | 1.4905 / 0.6801 | 1.4806 / 0.6843 |
| budget mean / sd | 27.0912 / 11.3435 | 27.0414 / 11.5495 |
| corr(accel, budget) | **-0.5120** | **-0.5152** |
| median A = dv/(g T) | 2.2905 | 2.2481 |
| median D = eta/A | 0.0050 | 0.0052 |
| feasible fraction | 100.0% | 100.0% |

KS on D: statistic 0.0088, **p = 0.997**. Fisher on feasibility: p = 1.0. The
-0.51 accel/budget correlation is present in *both* arms, so it is structural
in the loadout, not an artifact of the shared key.

## Decision
Record the fix as correct hygiene with no behavioural consequence. Remove the
distribution shift from the list of candidate causes for the v4 regression.

## Consequences
- Narrows the regression's cause space to the **instrument channels** and
  **seed variance** — and ADR 0024 voided the arm that was supposed to test the
  former, so the ablation must be rerun at a matched 3000-update horizon.
- Another instance of the mandate that a factor must be shown to bear on the
  mechanism, not just to exist. Reusing one `split` key across two different
  distributions makes the draws deterministically linked but not
  *statistically* related, which is why the bug was invisible in the data.
- Does not vindicate the old code. The collision would bite the moment anything
  conditioned on the shared key, so the fix stays.
- The two samplers were compared on the task distribution, not on learned
  policies. A null distribution shift makes a learning difference unlikely but
  does not strictly exclude one.

## Evidence
- `scratchpad/distshift.py`, 4,096 tours per arm, matched levels (18,000-level
  `data/pools/train.npz`) and matched base keys; old sampler extracted from
  `653127f^:spacenav/env.py`.
- `data/rl/ppo_v3/config.json` against `data/rl/ppo_v4/config.json` — the
  one-key diff.
- `spacenav/rl/ppo.py:74` — the `(frac - 0.5) / 0.4` contested ramp.
