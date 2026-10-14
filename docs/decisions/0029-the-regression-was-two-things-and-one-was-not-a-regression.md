# 0029 — The v4 regression was two roughly equal effects, and the larger one was a deliberate difficulty increase

**Date** 2026-10-14 · **Status** accepted

## Context
`ppo_v4` finished 7.5/18.5/11.3/17.7 pp below `ppo_v3` on easy/medium/hard/blind
and this was read as a regression needing a cause. ADR 0027 narrowed the cause
space to two changes acting from update 1: the four instrument channels
(ADR 0013) and `contested_end = 0.35`, the only differing config key. ADR 0024
voided the first attempt to separate them. `ppo_v5_noinstr` reruns v4 with
instruments off at a matched 3000-update horizon, which isolates them:

| run | instruments | contested | easy | medium | hard | blind |
|---|---|---|---|---|---|---|
| ppo_v3 | off | no | 95.1% | 52.0% | 19.6% | 41.2% |
| ppo_v4 | **ON** | yes | 87.6% | 33.5% | 8.2% | 23.6% |
| ppo_v5_noinstr | off | yes | 89.5% | 42.9% | 10.7% | 31.4% |

Mean of each run's last 3 checkpoints; ~600 episodes per suite.

| effect | easy | medium | hard | blind |
|---|---|---|---|---|
| remove instruments (v4->v5) | +1.9 | **+9.4** | +2.5 | **+7.9** |
| remove contested mix (v5->v3) | **+5.6** | **+9.1** | **+8.9** | **+9.8** |
| total | +7.5 | +18.5 | +11.3 | +17.7 |
| episode 95% CI | +-2.5 | +-4.0 | +-2.5 | +-3.7 |

Neither effect is the whole story and the contested mix is the larger of the
two. Critically, **the contested half is not a regression**: contested
waypoints sample the eta > 0.2 shell near the equal-pull surface on purpose
(ADR 0013), so v4 and v5 were trained on a harder distribution and scored on
unchanged frozen suites. Asking the agent to do more and scoring it on the old
exam is not a defect.

## Decision
Stop treating the v3 -> v4 gap as a regression to be fixed. Record it as two
measured effects: a real instruments cost on medium and blind, and a
difficulty-mix cost that is the price of a curriculum change we chose. Keep
`contested_end`; it is doing what it was added to do.

## Consequences
- Answers the standing architecture question in a useful direction. Two
  independent manipulations — *adding* training difficulty and *adding* four
  input channels — each cost performance elsewhere at fixed 285k parameters and
  fixed 393M steps. That is the signature of a learner at its resource budget,
  not one with the wrong structure, and it favours the under-training
  hypothesis over the representation hypothesis.
- But it does not establish that. A competing reading is plain distribution
  mismatch: contested levels are not merely harder, they are different, so
  training on them moves mass away from what the suites test. Capacity and
  mismatch predict different things under scale — if capacity, a larger
  d_model recovers the contested arm; if mismatch, it does not. That test needs
  the hardware of ADR 0026.
- **n = 1 training seed per arm.** The quoted CIs are episode-level and do not
  include training-seed variance, which ADR 0012 says must be separated. Seed 1
  of both arms is queued. Until it lands, the +9.4 pp instruments effect on
  medium is a single-seed observation outside episode noise, not a result.
- The instruments cost is concentrated on medium and blind and is within noise
  on easy and hard. The hard column is additionally uninterpretable per
  ADR 0025, since that tier is infeasible.
- Early checkpoints badly misled: at ckpt 100 the instruments gap read
  +14/+5/+1/+4 and at ckpt 200 it read +15/+1/+1/+2, so the medium and blind
  effects appeared, vanished and reappeared over the run. Reading an ablation
  before it finishes produced two wrong summaries here.

## Evidence
- `data/rl/ppo_v5_noinstr/eval.jsonl` — 31 checkpoints, 3000 updates, 393.2M
  env steps at 16,953 steps/s, `SPACENAV_INSTRUMENTS=0`.
- `data/rl/ppo_v4/eval.jsonl`, `data/rl/ppo_v3/eval.jsonl` — the baselines.
- ADR 0027 — the config diff that left exactly these two candidates.
- ADR 0024 — why the first instruments arm was void and the horizon is matched.
- ADR 0013 — the instrument channels and the contested shell.
