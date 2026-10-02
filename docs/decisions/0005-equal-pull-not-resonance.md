# 0005 — The equal-pull surface, not resonance overlap

**Date** 2026-10-01 · **Status** accepted

## Context
To raise eta ([0004](0004-model-demand.md)) we needed a constructive knob. The
obvious candidate from the literature is the Chirikov resonance-overlap
criterion and Wisdom's 2/7 law for the chaotic zone around a secondary — clean
closed forms with generator knobs (mass ratio, spacing, eccentricity).

## Decision
Rejected on timescale grounds. Use the **equal-pull surface** instead: the
radius where a secondary's pull equals its host's, d = a*sqrt(m/M), about half a
Hill radius.

## Why
Resonance overlap governs orbital evolution over *many* periods. An episode here
lasts a fraction of one orbit, so asymptotic chaos never gets time to bite.
Measured directly: placing waypoints in the Wisdom band lifted eta from 0.0026
to 0.0172 — sixfold and still negligible, because a few Hill radii out is an
eta ~ 0.02 desert.

Force competition is instantaneous, which is why it is the scale that matters at
this episode length. Measured eta against distance from a secondary, in units of
the equal-pull radius: 0.57 at 0.3x, **0.69 at 1.0x**, 0.24 at 2x, 0.064 at 4x,
0.019 at 8x. Shell placement was later tightened to 0.8–1.3x for this reason,
roughly doubling eta.

## Consequences
- Family choice is the larger lever: median eta is 0.0012 for `sol_like` and
  0.0006 for `compact_system`, against 0.41 `binary_star`, 0.46 `trinary`,
  0.54 `star_cluster`. Comparable masses are contested everywhere.
- Keep the Chirikov machinery on the shelf for a version with multi-orbit
  missions. It is the right tool for a different game.

## Sources
Chirikov 1979; Wisdom 1980; Deck, Payne & Holman 2013 (arXiv:1307.8119);
Laplace sphere of influence vs Hill radius.
