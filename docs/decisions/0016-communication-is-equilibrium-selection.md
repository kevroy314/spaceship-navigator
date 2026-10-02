# 0016 — Communication is equilibrium selection, not information transfer

**Date** 2026-10-02 · **Status** accepted · **Corrects** [0015](0015-architecture-pick.md)

## Context
[0015](0015-architecture-pick.md) closes with a risk note claiming that because
our environment is fully observed — `obs` already contains every body's state —
a plan-conditioned policy would learn to ignore the plan and read the answer off
the observation, making any announcement vacuous (concept-bottleneck leakage in
the control path).

That reasoning is wrong as a general claim about the communication channel, and
it contradicts a finding recorded in [0011](0011-callouts-and-legibility.md) in
the same session: shared weights **cannot break symmetry**, and on multi-modal
reward landscapes shared policies provably converge to averaged solutions where
independent ones reach optima (Fu et al., ICML 2022). Both statements cannot be
the whole story.

## The distinction
A message can carry two kinds of content, and observability bears on only one:

1. **Derivable facts.** "Retrograde burn in 4 s" is a function of the world state.
   Any receiver that can see the state can compute it. Full observability makes
   this content redundant, and the leakage argument in 0015 holds **for it**.
2. **Joint choices.** "You take left, I'll take right" is not in the world state
   at all. Two agents with perfect, identical observations still do not know
   which of several equally-good role assignments they are playing. This is
   equilibrium selection; observability cannot supply it, and it is the classic
   coordination case.

The switch is **mutual exclusivity of roles**, which is what the battle
constraints introduce. Without them there is no role to assign, so in the
current single-ship navigation task the channel really is redundant. The error in
0015 was generalising a property of the task in hand to the task being built.

## The test, and it is one we already have
Whether a channel can matter is not "is the environment observed" but **"does the
task admit several mutually exclusive optimal joint strategies"**. One winning
strategy means nothing to coordinate; k equally good exclusive assignments need
log2(k) bits of pre-commitment.

`opt/necessity.py` already measures this as `sufficiency` — the fraction of the
strategy family that wins. Measured on current levels: **median 0.0029**, i.e.
effectively a single winning strategy. That is an independent confirmation that
communication cannot help yet, and it is the same number that made three of the
causal maps look blank ([0009](0009-probability-field-and-alpha.md)).

## Consequences
- **Design requirement for battle levels**, not an aspiration: they must admit
  multiple viable role assignments, verified by lifting the sufficiency
  measurement to the joint action space. A combat level with one optimal joint
  plan teaches nothing about coordinating.
- The leakage risk in 0015 is **scoped to the single-agent navigation phase**. It
  is still the right thing to watch there, and the 5–20% return cost for an
  announcement that survives an intervention test still applies to the
  *derivable-fact* channel.
- A human teammate does not share a training convention, so "learn a private
  convention in self-play" is unavailable for the human case. This is a second,
  independent reason for the name-first vocabulary in
  [0011](0011-callouts-and-legibility.md) — not a stylistic preference.
- ~~Pre-play commitment is a distinct channel from in-flight callouts~~ —
  **withdrawn by [0017](0017-no-pre-play.md): there is no pre-play.** Every
  episode is novel to every participant, so coordination is in-band from a cold
  start. Participants share a protocol, never a plan.

## Process note
This record exists because an assumption was asserted from a property that did
not bear on the mechanism. The standing mandate in CLAUDE.md covers measuring
before believing; it should be read as covering assumptions about *which factors
matter*, not only numerical claims.
