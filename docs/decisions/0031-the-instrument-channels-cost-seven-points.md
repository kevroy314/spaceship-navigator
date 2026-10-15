# 0031 — The instrument channels cost ~7 pp on medium and blind, and two of the four are redundant

**Date** 2026-10-15 · **Status** accepted · **Resolves the open caveat in 0029**

## Context
ADR 0029 measured a 9.4 pp instruments cost on medium at n=1 training seed and
declined to call it a result, because ADR 0012 requires separating seed variance
from episode variance. Seed 1 of both arms completes the 2x2, all four runs at
3000 updates / 393.2M env steps:

| | easy | medium | hard | blind |
|---|---|---|---|---|
| instruments ON, seed 0 | 87.6% | 33.5% | 8.2% | 23.6% |
| instruments ON, seed 1 | 87.7% | 36.4% | 8.2% | 24.9% |
| instruments OFF, seed 0 | 89.5% | 42.9% | 10.7% | 31.4% |
| instruments OFF, seed 1 | 89.9% | 41.5% | 10.9% | 29.9% |
| **effect (OFF - ON)** | +2.1 | **+7.2** | +2.6 | **+6.4** |
| seed spread, ON | 0.1 | 2.9 | 0.0 | 1.4 |
| seed spread, OFF | 0.4 | 1.4 | 0.2 | 1.6 |

Training-seed variance is small — at most 2.9 pp, and under 2 pp on seven of the
eight cells — so it cannot manufacture a 7 pp effect. Pooling both seeds'
final checkpoints (~1,200 episodes per arm):

| suite | ON | OFF | diff | Fisher p |
|---|---|---|---|---|
| easy | 1050/1200 | 1082/1200 | +2.7 | 4.4e-02 |
| medium | 422/1188 | 513/1188 | **+7.7** | **1.6e-04** |
| hard | 101/1188 | 132/1188 | +2.6 | 3.8e-02 |
| blind | 300/1200 | 386/1200 | **+7.2** | **1.2e-04** |

All 8 cells (4 suites x 2 seeds) favour instruments-off; sign test p = 0.0078.

## Decision
Accept that the four instrument channels cost roughly 7 pp on medium and blind
as *policy inputs*. Keep them as a player-facing readout — that is what they
were designed for (ADR 0013) — but stop treating the cockpit display and the
agent's observation vector as the same channel. `SPACENAV_INSTRUMENTS` already
makes them separable, so the default for training flips to off.

## Consequences
- Removes the instruments half of the ADR 0029 decomposition as a live puzzle,
  and leaves the contested-mix half, which is a chosen difficulty increase and
  not a defect.
- **The mechanism is not measured.** What *is* measured is the channel content.
  Reading `observe` against `instrument_panel`: the observation already carries
  `rot(g)/SHIP_ACCEL` (so |g|), `dv_left = fuel * accel`,
  `1 - t/MAX_EPISODE_TIME` (so t_left) and a `dv_left / dv_needed` ratio.
  So `budget = (fuel*accel)/(|g|*t_left)` is a ratio of three quantities
  already present and partly duplicates an existing ratio, and
  `demand = eta/budget` is redundant given the two. Only `eta` — which needs
  the `topk=1` dominant-body decomposition — and `stretch`, the tidal
  gradient, carry information not already in the vector.
- That makes a targeted follow-up cheaper and more informative than the
  all-or-nothing flag: ablate the four channels individually (4 arms x 3.4 h on
  the 1080 Ti) and keep `eta`/`stretch` if the cost sits in the redundant
  ratios. Not run; this record does not claim which channel is responsible.
- The competing mechanism is the capacity account of ADR 0029 — four more input
  dims against a 285k-param net at its budget. Redundancy and capacity are not
  exclusive, and both predict the cost shrinks at larger `d_model`, which needs
  the hardware of ADR 0026.
- **n = 2 seeds, where ADR 0012 asks for >= 5.** The seed spread is a
  two-point range, not a variance estimate, and the pooled Fisher test treats
  seeds as fixed and so understates variance. The conservative and the
  anti-conservative reads agree on medium and blind, which is why this is
  accepted; easy (+2.1) and hard (+2.6) sit near episode noise and are not
  claimed. Hard is independently uninterpretable per ADR 0025.
- Every suite number here is scored against a baseline that crashes on 21.6% of
  comfortable-tank missions (ADR 0030), so the `win_rate_vs_pilot` margins in
  these runs are soft even though the success rates are not.

## Evidence
- `data/rl/ppo_v4/eval.jsonl`, `ppo_v4_s1`, `ppo_v5_noinstr`,
  `ppo_v5_noinstr_s1` — four runs, 3000 updates each, ~600 episodes per suite
  per checkpoint; success is the mean of each run's last 3 checkpoints, pooled
  tests use the final checkpoint.
- `spacenav/env.py:494` `instrument_panel`, `spacenav/env.py:529` `observe` —
  the redundancy reading.
- ADR 0013 — why the channels exist. ADR 0029 — the caveat this resolves.
  ADR 0012 — the seed requirement still not met.
