# 0034 — The episode clock is a per-process setting, and a longer one helps a navigator only if it can use the time

**Date** 2026-10-30 · **Status** accepted · **Builds on 0032**

## Context
0032 left the clock as a global constant and listed a per-pool clock as
unbuilt. The author's review of the replay videos made it the first
substantive point of the paper rewrite: several agents looked as if they
would have arrived had the simulation not stopped them, and time is the
least interesting of the three constraints (time, fuel, exposure). Two things
were therefore needed at once — a way to run a longer clock without touching
every existing checkpoint and suite, and a measurement of what a longer clock
actually buys.

## Decision
1. **`SPACENAV_EPISODE_S`** sets `MAX_EPISODE_TIME` per process, read once at
   import in `spacenav/constants.py`; default is exactly `150.0`, so every
   existing checkpoint, suite and test is bit-identical. Everything downstream
   derives from it (`MAX_EPISODE_TICKS`, the time-remaining features in
   `observe()` and `instrument_panel()`, `levels/build.N_SNAPSHOTS`).
2. **`SPACENAV_SUITES`** selects the suite file; `train.py` refuses to
   auto-build a non-default path, so a typo cannot silently evaluate a 600 s
   policy against 150 s suites. Each suite now stores the clock it was built
   at, and `load_suites()` warns on a mismatch.
3. **Suites for a 600 s clock come from a new `long_val` pool**, not from
   `long`. `long` holds exactly 200 levels per family, and the `easy` spec
   takes 200 from each of its three families, so suites drawn from `long`
   would have been the training set. `long_val` is the same recipe (training
   families, 200 per family, 16 snapshots, validated to 900 s) under seed
   block 40M; `scripts/build_long_suites.py` builds both.
4. `scripts/run_queue.sh` takes optional trailing tokens per run
   (`pool=`, `updates=`, `UPPER=value`), backward-compatible with the
   three-column queues.

## The measurement
The scripted reflex pilot, flown on the 600 s suites:

| suite | arrivals after the 150 s mark | 150 s timeouts → 600 s outcome |
|---|---|---|
| easy | 1 / 480 | — |
| medium | 3 / 197 | timeouts 102 → 17; lost 53 → 105; arrivals 206 → 197 |
| hard | 1 / 47 | timeouts 146 → (mostly lost/destroyed) |
| blind | 0 / 136 | timeouts 147 → (mostly lost/destroyed) |

For the pilot, a longer clock converts timeouts into fuel exhaustion and
collisions, not into arrivals. For the cross-entropy optimizer in 0032, the
same lengthening converted timeouts into arrivals (5/32 → 13/32 at 600 s).
Thus, what a longer clock buys is a property of the navigator and not of the
clock: a navigator on a path that would have arrived gets to arrive, and a
navigator that was never going to gets to run its tank dry instead of timing
out. The paper carries both results side by side rather than the optimizer
result alone.

## Consequences
- The 150 s results remain the paper's results; `long600_s0..s2` (3 seeds,
  `eta,stretch`, `long` pool, 600 s) and `budget10k_s0..s1` (10,000 updates,
  defaults) are queued as `spacenav-longclock-1` on the cluster 3090, ~10.5 h,
  and will say which of the two the learned policy is.
- **`blind` is not blind at 600 s.** No long-horizon pool carries the three
  holdout families, so the 600 s blind suite is a seen-family suite with the
  blind task mix. Comparisons on that tier across clocks are not meaningful.
- 600 s and 150 s suites are different levels (`long_val` vs `val_seen`), so
  comparisons are same-family and same-spec, not paired.
- The pilot yardstick in `suites_long.pkl` was flown at 600 s, so `win`
  against it is against a 600 s pilot.
- `gamma = 0.999` is ~65 s of lookahead and is unchanged; a 600 s policy
  still discounts anything beyond a minute. If the 600 s runs underperform,
  this is the first suspect.
- n = 3 and n = 2 seeds fall short of 0012's five; lines appended to
  `queues/longclock.txt` and a relaunched worker (it skips completed runs)
  lift them.
- `scripts/twostop_clock.py` still patches the constant by hand; it should
  use the env var.
- The pilot's medium-suite numbers are from different levels than the 150 s
  figure, so the "102 → 17" row is indicative, not a paired difference.

## Evidence
- `spacenav/constants.py`, `spacenav/rl/evaluate.py`, `scripts/train.py`,
  `scripts/run_queue.sh`, `scripts/build_long_suites.py`,
  `scripts/razer_long.sh`, `queues/longclock.txt`.
- `python -m pytest tests/ -q` with no env vars: 126 passed (defaults
  unchanged). `SPACENAV_EPISODE_S=600` gives `MAX_EPISODE_TICKS = 9000`; a
  coasting ship times out at tick 9000 (t = 599.9 s).
- Pilot numbers from `scripts/build_long_suites.py` at `--clock 600`,
  n per suite as in the table; `data/eval/suites_long.pkl` (gitignored, also
  on the cluster share).
- 0032 — the optimizer sweep this contrasts with. 0012 — the seed count.
