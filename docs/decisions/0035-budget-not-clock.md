# 0035 — Ten thousand updates buys 5–15 points; a 600 s clock at the old discount loses 20–40

**Date** 2026-10-31 · **Status** accepted · **Builds on 0034, 0033, 0026**

## Context
0034 queued two experiments on the cluster 3090 to address the paper's
threats to validity: the best configuration (`eta,stretch`) trained at a
600 s clock on the long-horizon pool (3 seeds), and the same configuration
trained for 10,000 updates instead of 3000 (2 seeds). Both finished
(`spacenav-longclock-1`, 11.4 h, every run rc=0).

## Measurements

**Budget.** 10,000 updates (1.31 B env steps, ~3.5 h per run) against the
five-seed 3000-update mean, 150 s suites, mean of each run's last three
checkpoints:

| suite | 3000 (n=5) | 10,000 (n=2) | diff |
|---|---|---|---|
| easy | 90.4 ± 1.8 | 95.1 ± 0.9 | +4.7 |
| medium | 41.9 ± 2.3 | 57.2 ± 2.5 | +15.3 |
| hard | 11.1 ± 1.0 | 18.9 ± 0.6 | +7.8 |
| blind | 29.2 ± 1.2 | 40.9 ± 0.1 | +11.7 |

The 10k runs' own values at update 3000 were 92.7 / 47.8 / 13.4 / 34.8, and
the medium curve rose 47.8 → 50.2 → 52.4 → 53.9 → 54.1 → 55.9 → 57.7 from
3k to 10k in steps of 1000, with no plateau. At 10k the policy exceeds the
reference pilot on every tier including blind (40.9 vs 31.8), which the 3k
policy lost.

**Clock.** 600 s clock, `long` pool, `suites_long.pkl`, γ = 0.999, 3 seeds:

| seed | easy | medium | hard | blind | medium timeouts |
|---|---|---|---|---|---|
| 0 | 68.1 | 26.0 | 4.8 | 18.4 | 42 % |
| 1 | 74.2 | 31.9 | 7.1 | 19.8 | 37 % |
| 2 | 4.4 | 1.5 | 0.0 | 1.0 | 73 % |
| mean ± sd | 48.9 ± 38.6 | 19.8 ± 16.1 | 3.9 ± 3.6 | 13.1 ± 10.5 | |

Against 90.4 / 41.9 / 11.1 / 29.2 at 150 s (same families and specs,
different levels; blind not blind at 600 s). Seed 2 never learned (easy
peak 19 % at u2300). Seeds 0–1 learned and still timed out on 37–42 % of
medium tasks at 600 s, against roughly a third at 150 s: four times the
clock, more timeouts.

## Decision
1. **Report the 10k-update result as the paper's ceiling**, and state
   that every 3000-update comparison is between under-trained policies whose
   effect sizes should not be read as task properties. The direction of the
   channel effect (0033) is expected to survive; its size is not.
2. **Do not report the 600 s clock as a repair.** State it as a third case
   beside the optimizer (helped) and the pilot (not helped): a learner at an
   unchanged discount is hurt, badly and with one seed in three failing to
   learn.
3. **Test the discount.** γ = 0.999 is ~65 s of lookahead, a ninth of a 600 s
   episode. `queues/long600_gamma.txt` queues γ = 0.9995 (~130 s) and 0.9998
   (~330 s) at 600 s, 3 seeds each, behind `queues/scale.txt`
   (`spacenav-gamma-1`, launched automatically when the scale queue ends).
   `scripts/run_queue.sh` gained `gamma=` and `dmodel=` tokens.

## Consequences
- Answers the author's "more training budget (by a lot)" with a number that
  is larger than most effects in the paper. ADR 0026's framing — that scale
  and steps compete for the 1080 Ti — is moot on the 3090, where 10k updates
  is 3.5 h.
- n = 2 for the budget result and n = 3 for the clock result, both below
  0012's five. The budget result's seed SD (0.1–2.5) makes its direction
  safe; its size wants more seeds. The clock result's SD (39 on easy) is the
  finding.
- The γ hypothesis is untested. Alternatives: the time-remaining observation
  feature rescales with the clock (a 600 s policy sees a different input
  distribution), and per-step reward terms accumulate four times longer.
  Both are confounded with γ in the queued runs; if γ = 0.9998 recovers
  150 s performance, γ was sufficient, not necessarily the whole cause.
- The 600 s blind suite is seen-family; its number is not a generalization
  measure.

## Evidence
- `data/razer_long/data/rl/{long600_s0,s1,s2,budget10k_s0,s1}/eval.jsonl`,
  fetched from `/mnt/x/cluster/data/spaceship-navigator/results/evals_long.tgz`.
- Cluster log of `spacenav-longclock-1`: five runs, all rc=0, 00:38–10:39 UTC
  2026-10-31.
- 0034 — the clock machinery and the pilot-at-600 s result. 0033 — the
  configuration. 0026 — the 1080 Ti scaling cost this supersedes in practice.
