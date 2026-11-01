# 0037 — The channel cost was capacity: it vanishes at twice the embedding width

**Date** 2026-11-01 · **Status** accepted · **Resolves the open mechanism in 0033**

## Context
0031 and 0033 established that feeding the policy the two redundant
instrument channels (`budget`, `demand`) costs ~7–10 pp against feeding it
`eta,stretch` or nothing, and left the mechanism open between two
accounts: a **capacity** account (four extra input dimensions against
285,322 parameters) and a **conditioning** account (a ratio with a division
in it is heavy-tailed even after symlog). The two predict different
behavior under scale, and 0026 had priced the scaling test out of reach on
the 1080 Ti. On the 3090 it is 2 h 16 min per run.

`queues/scale.txt`: `d_model` 256 (698,634 parameters), 3000 updates, 150 s
suites, `eta,stretch` versus all four channels, three seeds each. The first
attempt (`spacenav-scale-1`) failed in every run with
`RESOURCE_EXHAUSTED` at the 0.45 memory cap; rerun at 0.80
(`spacenav-scale-2`, 13.6 h, every run rc=0).

## Measurement
Mean of each run's last three checkpoints, 150 s suites.

| width | arm | n | easy | medium | hard | blind |
|---|---|---|---|---|---|---|
| 128 | `eta,stretch` | 5 | 90.4 ± 1.8 | 41.9 ± 2.3 | 11.1 ± 1.0 | 29.2 ± 1.2 |
| 128 | all four | 5 | 86.3 ± 1.9 | 31.6 ± 5.2 | 5.8 ± 3.4 | 22.2 ± 3.0 |
| 256 | `eta,stretch` | 3 | 89.8 ± 0.9 | 43.6 ± 2.0 | 12.2 ± 0.4 | 32.3 ± 1.0 |
| 256 | all four | 3 | 90.1 ± 0.9 | 41.2 ± 0.4 | 11.4 ± 1.2 | 29.6 ± 0.5 |

The channel cost (`eta,stretch` minus all four):

| width | easy | medium | hard | blind |
|---|---|---|---|---|
| 128 | +4.1 | +10.3 | +5.3 | +7.0 |
| 256 | −0.3 | +2.4 | +0.8 | +2.7 |
| Welch p at 256 | 0.74 | 0.18 | 0.38 | 0.026 |

Width alone, `eta,stretch` 256 vs 128: −0.6 / +1.7 / +1.1 / +3.1.

## Decision
Record the mechanism as **capacity**. The conditioning account predicted
persistence at any width and is not supported. The wider network does not
perform better with good inputs; it tolerates bad ones.

## Consequences
- Closes the one limitation the paper could not previously address, and in
  the direction 0029 argued from: two independent manipulations costing
  performance at fixed size was the signature of a learner at its budget.
- Sharpens the design guidance of 0033. At d_model 128, feed `eta,stretch`;
  at 256 the choice is nearly free, so the default stays `eta,stretch` for
  the cheaper network and the cockpit can carry all four regardless.
- The blind-tier difference at 256 (+2.7, p = 0.026) is the one residue;
  with n = 3 it is not pressed.
- n = 3 per arm at 256 against 0012's five; the direction is safe (every
  tier's cost shrinks by 3–8 pp and the d256 seed SDs are ≤ 2.0).
- Doubling the width doubles the wall clock per run (67 → 136 min) for a
  gain of 0–3 pp on good inputs at this budget, which is the practical
  reason d_model 128 remains the working configuration.
- Not tested: d_model 384 (0026 priced it at 22.9 h per run on the 1080 Ti;
  ~5 h on the 3090), and whether the 10k-update budget (0035) interacts
  with width.

## Evidence
- `data/razer_scale/rl/d256_{info,all}_s{0,1,2}/eval.jsonl`, fetched via
  `evals_scale.tgz` on the cluster share.
- `queues/scale.txt`; `scripts/run_queue.sh` `dmodel=` token; cluster jobs
  `spacenav-scale-1` (OOM at 0.45) and `spacenav-scale-2` (0.80 cap,
  19:06 UTC Oct 31 – 08:44 UTC Nov 1).
- 0033 — the d128 arms and the open mechanism. 0029 — the capacity
  argument. 0026 — the scaling cost that made this unaffordable before the
  3090.
