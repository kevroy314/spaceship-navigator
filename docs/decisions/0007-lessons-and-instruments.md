# 0007 — Discrete lessons, measured signatures, earned instruments

**Date** 2026-10-02 · **Status** accepted

## Context
D ([0004](0004-model-demand.md)) gives a continuum. A continuum does not tell a
player what they are supposed to be learning.

## Decision
Cut the space into named skills, each defined by a **measurable signature**, and
assign a level by what it demands — never by which generator produced it, which
would make the taxonomy unfalsifiable. Each lesson earns the cockpit instrument
that makes its lesson legible.

| lesson | signature | instrument |
|---|---|---|
| Gravity Is Cool | coasting alone completes the tour | none |
| Dead Heading | one burn wins from a broad spread | target marker |
| Careful Leading | burn time and heading are coupled | lead indicator |
| Thinking In Conics | one-body model cannot fly it (D >= 0.1) | tidal discriminator (eta) |
| All About Fuel | tank below gravity's impulse (A < 1) | budget gauge |
| Rolling With It | open-loop loses, re-planning recovers | predictability cone |
| One Exact Boost | one decision, window under a reaction time | flight computer |

## Consequences
- **Level 0 is a feature, not a bug.** An earlier version of this code treated a
  free ride as a generator defect and discarded it. It is the opening lesson and
  it works like "hold right" in a platformer. `levels/coastpath.py` builds them
  by inverting the problem — roll a ballistic arc, then anchor targets *onto*
  it. Measured: 108/108 targets collected by pure coasting across six families,
  arcs winding 1.5–6.2 turns.
- The instrument for the superhuman branch is what lets a *human* fly it at all,
  which is a nicer resolution than gating it off.
- **Known weakness:** the signatures overlap rather than discriminate. All four
  coast levels also satisfy `gravity-assist`, `conics` and `dead-heading`;
  `free-ride` wins only by match order. Needs orthogonalising.
- Three of eight lessons have no instance yet (`rolling-with-it`,
  `one-exact-boost`, `lottery`) — see [0008](0008-chaos-is-aspirational.md).
