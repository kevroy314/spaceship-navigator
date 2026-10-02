# 0013 — Longer episodes

**Date** 2026-10-02 · **Status** proposed

## Context
Requested as a direction to explore, on the grounds that players will enjoy
levels with broader decision trees. It also turns out to be the keystone for
several stalled threads.

## Why it is the keystone
[0008](0008-chaos-is-aspirational.md) measured lambda ~ 0.007 1/s, a Lyapunov
time of ~140 s against 150 s episodes: about **one e-folding per episode**. That
is why the chaos machinery found nothing — alpha unmeasurable,
`rolling-with-it` and `lottery` empty, the prediction cone never expiring.

Episode length is the cheapest dial on all of it. At 600 s an episode spans
~4.2 e-foldings; at 900 s, ~6.4. Chaos stops being an aspiration and becomes a
property, which in turn populates the branches that are currently empty and makes
the predictability cone informative instead of inert.

It also does what was actually asked: more time means more reachable waypoints,
more viable routes between them, and genuinely branching decision trees rather
than one transfer with a tolerance.

## Costs and risks
- Rollout cost is linear: 600 s = 9,000 ticks at 15 Hz, **4x** current. Affordable
  for measurement; meaningful for training, and it interacts with the discount —
  a longer horizon with gamma = 0.999 weights ~1,000 steps, so credit assignment
  over 9,000 needs either shaping (we have potential-based shaping) or a
  hierarchical/option structure ([0010](0010-planner-as-teacher.md)).
- Re-check the fractal-objective threshold after the change. More e-foldings
  means a larger effective lambda over the episode, which pushes toward
  lambda > log(1/gamma)/dt — the condition [0008](0008-chaos-is-aspirational.md)
  cleared with only 2x margin. **This change could move us into the regime the
  alarm was about.** Measure lambda again at the new length before training.
- Human playability: a 10-minute flight needs the time controls to be good
  (they exist) and probably needs save/resume.

## Decision
Explore at 600 s as a measurement-only change first: rebuild the demand survey
and re-measure lambda and alpha at the new length. Only then consider training.
