# 0033 — The instrument cost splits exactly on the redundancy line: eta and stretch are free, budget and demand cost everything

**Date** 2026-10-25 · **Status** accepted · **Extends 0031**

## Context
ADR 0031 established that all four instrument channels cost ~7 pp as policy
inputs, and read the mechanism off the code rather than measuring it: `observe`
already carries `rot(g)/SHIP_ACCEL` (hence `|g|`), `dv_left = fuel * accel`,
`1 - t/MAX_EPISODE_TIME` (hence `t_left`) and a `dv_left / dv_needed` ratio, so

- `budget = (fuel*accel) / (|g| * t_left)` is a ratio of three quantities the
  policy already had, and partly duplicates an existing ratio;
- `demand = eta / budget` is redundant given those;
- `eta` needs the `topk=1` dominant-body split and `stretch` is the tidal
  gradient — neither is otherwise in the vector.

0031 predicted that splitting the panel along that line would put the cost on
the redundant half, and said so as a hypothesis with n=2 behind it. Five seeds
per arm, 3000 updates each, all on the cluster 3090:

| configuration | n | easy | medium | hard | blind |
|---|---|---|---|---|---|
| all four off | 5 | 89.4 +- 1.50 | 41.7 +- 3.04 | 10.7 +- 1.24 | 30.2 +- 2.27 |
| **`eta,stretch`** | 5 | **90.4 +- 1.81** | **41.9 +- 2.30** | **11.1 +- 1.03** | **29.2 +- 1.23** |
| `budget,demand` | 5 | 86.6 +- 1.22 | 33.2 +- 2.61 | 6.2 +- 1.68 | 22.7 +- 2.52 |
| all four on | 5 | 86.3 +- 1.87 | 31.6 +- 5.25 | 5.8 +- 3.39 | 22.2 +- 2.99 |

Welch t on seed means, two contrasts that matter:

| contrast | easy | medium | hard | blind |
|---|---|---|---|---|
| `eta,stretch` vs all-off | p = 0.40 | p = 0.90 | p = 0.60 | p = 0.42 |
| `budget,demand` vs all-on | p = 0.77 | p = 0.59 | p = 0.81 | p = 0.79 |
| **`eta,stretch` vs `budget,demand`** | **+3.74 pp, p = 0.0064, d = 2.42** | **+8.75 pp, p = 0.0005, d = 3.56** | **+4.86 pp, p = 0.0011, d = 3.49** | **+6.53 pp, p = 0.0022, d = 3.30** |

Adding the two informative channels is indistinguishable from adding nothing on
all four suites. Adding the two redundant ones is indistinguishable from adding
all four. The panel partitions cleanly.

## Decision
Feed the policy **`eta,stretch`** and keep all four dials in the player cockpit.
`SPACENAV_INSTRUMENTS=eta,stretch` becomes the training default in place of
`0`, since the two physics channels are free and may yet prove useful on a task
that needs them — the point of adding them in ADR 0013 — while the two
arithmetic ratios are what the policy was choking on.

## Consequences
- Resolves the cockpit-versus-observation tension from 0031 without a
  compromise: the display and the observation can now differ by exactly the two
  channels that are redundant for a policy and still legible to a person.
- Confirms a mechanism that was read off the source and could have been wrong.
  Worth noting which way the evidence runs: the prediction was made from code
  inspection in 0031 and then held at n=5, so the redundancy account earns
  weight it did not have before.
- `eta,stretch` is nominally *above* all-off on easy, medium and hard
  (+0.9 / +0.2 / +0.4 pp) and below on blind (-1.0 pp). None of that is
  significant. **There is no evidence the two channels help**, only that they
  do not hurt.
- Does **not** explain *why* redundant inputs cost 7-9 pp. Four extra
  dimensions against 285k parameters is a capacity story; a ratio with a
  division in it is heavy-tailed even after symlog, which is a conditioning
  story. Both remain untested, and ADR 0026's scaling test would separate them.
- The hard-tier figures still carry ADR 0032's caveat, and every margin against
  the pilot still carries ADR 0030's.
- All ten runs behind the two per-channel arms ran on the 3090 while seeds 0-1
  of the headline arms ran on the 1080 Ti. ADR 0031 checked for a hardware
  effect within the headline arms and found none on the all-off side.

## Evidence
- `data/rl/abl_info_s{0..4}`, `abl_redundant_s{0..4}`,
  `ppo_v5_noinstr{,_s1..s4}`, `ppo_v4{,_s1..s4}` — 20 runs, 3000 updates and
  393M env steps each, ~600 eval episodes per suite per checkpoint, success as
  the mean of each run's last 3 checkpoints.
- `queues/ablation.txt`, `queues/ablation2.txt` — the arms as run.
- `spacenav/env.py` `instrument_panel` / `observe` — the redundancy reading.
- ADR 0031 — the n=2 result and the prediction this tests. ADR 0013 — why the
  channels exist. ADR 0012 — the 5-seed requirement, met here.
