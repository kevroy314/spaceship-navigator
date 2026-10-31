# 0036 — The clock and the discount are one parameter: γ = 0.9998 recovers the 600 s learner

**Date** 2026-10-31 · **Status** accepted · **Resolves the open question in 0035**

## Context
0035 reported that training at a 600 s clock with γ = 0.999 lost 20–40
points against the 150 s reference, with one seed in three never learning
and the seeds that learned timing out on 37–42 % of medium tasks. It named
the discount as the untested suspect: γ = 0.999 is ~67 s of lookahead at
15 Hz, a ninth of a 600 s episode. `queues/long600_gamma.txt` tested it:
γ = 0.9995 (~133 s) and γ = 0.9998 (~333 s), 600 s clock, `long` pool,
`suites_long.pkl`, `eta,stretch`, 3000 updates, three seeds each
(`spacenav-gamma-1`, 8.0 h, every run rc=0).

## Measurement
Mean of each run's last three checkpoints; medium-suite timeout share at
the final checkpoint.

| γ | lookahead | easy | medium | hard | blind | medium timeouts |
|---|---|---|---|---|---|---|
| 0.999 (0035) | 67 s | 48.9 ± 38.7 | 19.8 ± 16.1 | 4.0 ± 3.6 | 13.1 ± 10.5 | 37–42 % (learned seeds) |
| 0.9995 | 133 s | 47.7 ± 38.1 | 20.0 ± 16.2 | 4.2 ± 4.1 | 13.1 ± 11.1 | 45–62 % |
| **0.9998** | **333 s** | **83.7 ± 1.4** | **42.4 ± 1.4** | **14.1 ± 0.7** | **29.7 ± 0.1** | **26–35 %** |
| 150 s reference (0033) | 67 s of a 150 s episode | 90.4 ± 1.8 | 41.9 ± 2.3 | 11.1 ± 1.0 | 29.2 ± 1.2 | ~33 % |

Per seed at γ = 0.9995: 65.7 / 73.5 / 4.0 on easy — the same seed 2 that
failed at γ = 0.999 fails again, and the two that learned land within two
points of their γ = 0.999 values. At γ = 0.9998 the three seeds are 82.1,
84.1, 84.8 on easy and 42.3, 43.9, 41.0 on medium.

## Decision
Treat the episode clock and the discount as a single setting. A policy
trained at clock T needs a value horizon 1/(1−γ) that reaches a substantial
fraction of T; 1/9 of the episode fails, 2/9 fails identically, 5/9 works.
The paper's clock section now says so, and the recommended pairing for a
600 s clock is γ = 0.9998.

## Consequences
- Closes 0035's open question in the direction it guessed. The alternatives
  it listed (time-remaining feature rescaling; per-step reward terms
  accumulating) are not excluded as contributors, but neither is needed to
  explain the result, since changing only γ moves the learner from failure
  to the 150 s level.
- The threshold between 133 s and 333 s was not resolved further; a sweep
  over γ at fixed clock (or over the clock at fixed γ) would locate it and
  would be the natural next measurement if the paper's claim is to be a
  rule rather than two points.
- 600 s vs 150 s comparisons remain same-family, not paired (`long` vs
  `val_seen` levels), and the 600 s blind suite is seen-family. Easy at
  600 s (83.7) sits below the 150 s easy (90.4); the 600 s easy suite's own
  pilot is 80.0 against 85.3, so part of that gap is the levels.
- n = 3 per arm, below 0012's five. The γ = 0.9998 seed SD (≤ 1.4) makes the
  direction safe.
- The seed-2 failure at both lower discounts, with the same seed, suggests
  the failure is a property of the initialisation-plus-horizon rather than
  of chance; not investigated.

## Evidence
- `data/razer_gamma/rl/long600_g{9995,9998}_s{0,1,2}/eval.jsonl`, fetched
  via `evals_gamma.tgz` on the cluster share.
- `queues/long600_gamma.txt`; `scripts/run_queue.sh` `gamma=` token;
  cluster job `spacenav-gamma-1`, 11:05–19:00 UTC 2026-10-31.
- 0035 — the γ = 0.999 runs and the hypothesis. 0034 — the clock machinery.
