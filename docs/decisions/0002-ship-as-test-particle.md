# 0002 — The ship is a test particle

**Date** 2026-09-21 · **Status** accepted

## Context
A browser client needs to show the same universe the trainer sees, over bad
internet, and `tests/test_js_parity.py` has to be able to prove the two agree.

## Decision
The ship feels every body but moves none. Body trajectories are therefore fully
determined by the level's initial snapshot.

## Consequences
- The client receives ~1.5 KB (`init`: initial body state) and integrates the
  ephemeris itself, instead of downloading ~1 MB of precomputed positions.
- Exact JS/JAX parity is achievable and tested, because both sides integrate the
  same deterministic system from the same numbers.
- Costs: no ship-on-body back-reaction (physically wrong, irrelevant at game
  masses), and any physics change must be mirrored in `web/js/sim.js` or the
  parity test fails. That mirroring requirement is a standing obligation.
