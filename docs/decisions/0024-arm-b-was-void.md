# 0024 — An A/B between runs of different length is void; schedules must be pinned

**Date** 2026-10-14 · **Status** accepted

## Context
To test whether the four instrument channels (ADR 0013) hurt the agent as
*information*, arm B was launched as `ppo_v4_noinstr` with
`SPACENAV_INSTRUMENTS=0` for **1000** updates, against the `ppo_v4` baseline's
**3000**. `scripts/train.py` did
`cfg = dataclasses.replace(cfg, total_updates=args.updates)`, so every schedule
is driven by `frac = u / args.updates`. The two arms were therefore never
comparable:

| compare at | arm B | ppo_v4 | why it is not an A/B |
|---|---|---|---|
| same update u=100 | frac 0.10, wp_max 1.40 | frac 0.033, wp_max 1.13 | arm B faces a harder task mix |
| same frac 0.10 | u=100, 13M steps | u=300, 39M steps | ppo_v4 has 3x the data |

At matched update arm B looks better (0.41 vs 0.12 train success); at matched
curriculum fraction it looks worse (0.41 vs 0.59). Both readings are artifacts.
Arm B also died silently at u~100 of 1000, so it has no checkpoint at any
matched point regardless.

## Decision
Add `--schedule-updates` to `scripts/train.py`. It sets the horizon the
curriculum, entropy anneal and lr anneal are written against, independently of
how many updates the arm actually runs. Any arm compared against a baseline of
a different length must pass the baseline's length here. The chosen horizon is
recorded in the run's `config.json` as `schedule_updates`.

Arm B is discarded, not reinterpreted. It is not evidence about instruments in
either direction.

## Consequences
- Buys comparability: a 300-update probe can now be read against a 3000-update
  baseline at the same update number, which makes cheap ablations possible.
- Costs nothing at default (`--schedule-updates 0` follows `--updates`), so
  every existing run's recorded config still describes what it ran.
- Does **not** fix the deeper problem that v3-vs-v4 is n=1 vs n=1, which
  ADR 0012 says cannot resolve a 5 pp difference. Pinning schedules makes an
  arm comparable; it does not make one seed sufficient.
- The instruments ablation is still unanswered and must be rerun at 3000
  updates.

## Evidence
- `data/rl/ppo_v4/train.jsonl`, `data/rl/ppo_v4_noinstr/train.jsonl` — the
  matched-update and matched-frac tables above, n=1 run per arm.
- `data/rl/ppo_v4_noinstr/eval.jsonl` — 2 checkpoints (u=0, u=50); the run died
  at u~100 with no traceback.
- `scripts/train.py:57` (before this change) is the single line responsible.
