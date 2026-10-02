# 0014 — Four measurement bugs the test suite found

**Date** 2026-10-02 · **Status** accepted

## Context
Test coverage went from 35 to 118 tests. Writing the tests found more than the
tests did: two live bugs, one latent, and two assumptions of mine that were
simply false. Recorded because several of today's measurements were affected.

## 1. `p_contested` was a no-op (fixed)
`_sample_waypoint` binds `moving = take_station | take_body` *before* the
contested-placement override and never recomputes it. The override set
`take_body |= chaotic`, but `anchor` and `pos` read the stale flag — so a
contested waypoint was computed and then discarded in favour of a free-space
point.

Measured on a hand-built level: `p_contested=1.0` gave results **bit-identical**
to `p_contested=0.0`, all anchors -1, median waypoint eta 0.0114. With one line
recomputing `moving`: real secondaries, median eta **0.554 — 49x higher**.

Scope: broken entirely when `p_station=0` (e.g. spec `1w0mc`), and `(1-p_station)`
of the time otherwise. The curve build used `100m` specs, where `want_moving` is
always true, so **those measurements stand** — but the contested curriculum has
been running at a fraction of its intended strength, and the RL ramp
(`p_contested` up to 0.35 with `p_station` well below 1) was mostly inert.

## 2. PRNG key collision in `sample_tour` (fixed)
`split(key, K+5)` gave the waypoint loop `keys[2 .. K+1]` while the loadout drew
from `keys[K]` (accel) and `keys[K+1]` (budget) — an overlap. The ship's engine
and tank were a deterministic function of where the last two waypoints landed.
At the old `MAX_WAYPOINTS=4` this corrupted 3- and 4-stop tours; at 6 it
corrupts 5- and 6-stop tours. Now `split(key, K+7)` with the loadout at K+5/K+6.

**This changes every sampled task.** Recorded human runs and episode IDs no
longer reproduce. Accepted deliberately: carrying a sampler whose difficulty and
loadout are correlated is worse than losing a handful of recorded runs, and
checkpoints were already invalidated today by the observation change.

## 3. `thrust_authority` returned NaN (fixed)
`jnp.median` *propagates* NaN, and the function used NaN to mask post-flight
ticks — so any flight that ended before the horizon, i.e. nearly all of them,
reported NaN. Now `jnp.nanmedian`. Latent; only consumed inside `probe()`.

## 4. Two of my assumptions were false
- **`gravity_at(topk=k)` is not pointwise monotone in k.** A coarse model is a
  truncated *vector* sum and discarded terms can cancel: measured 1 of 24 probe
  points where k=2 is more accurate than k=3. What does hold, and what the
  model ladder actually relies on, is that the discarded *magnitude budget*
  falls strictly to zero and bounds the residual by the triangle inequality;
  mean residual falls ~58x per rung.
- **eta is not bounded by 1.** `|a_full - a_dom| / |a_full|` reaches **4.3** at a
  star-giant saddle where the resultant nearly cancels, and sampled contested
  waypoints hit 2.5. So eta is not "the share of the force that is not the
  dominant body" once the full field is small. Where eta > 1, D = eta/A is
  measuring near-cancellation of the resultant, which is a different
  phenomenon and probably deserves its own treatment.

## Also found, not a bug
Calling `E.step` eagerly in a Python loop recompiles the substep `fori_loop`
every iteration (the body is a fresh lambda each call): 180 eager steps took
over 12 minutes, ~2 s when jitted. Worth auditing `scripts/` for.

`make_causal.py` never populated `concentration` or `n_critical` into the feature
dict, so the `one-exact-boost` lesson could **never** match. That — not the
absence of chaos — is why that branch was empty. Fixed with a two-stage pass:
a window under a human reaction time is a necessary condition, so necessity is
measured only on candidates that already have one.
