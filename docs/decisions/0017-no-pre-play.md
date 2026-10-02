# 0017 — No pre-play: every episode is novel to every participant

**Date** 2026-10-02 · **Status** accepted · **Corrects** [0016](0016-communication-is-equilibrium-selection.md)

## Context
[0016](0016-communication-is-equilibrium-selection.md) closed by proposing
pre-play commitment as a separate channel — "one bit exchanged before the
episode may be worth more than a stream during it."

There is no before. Every episode is a level neither participant has seen. There
is no lobby, no briefing, and no shared prior about *this* geometry.

## Decision
Coordination is **in-band, from a cold start, in a novel field**, under the same
compute and bandwidth budget as flying. Participants may share a **protocol**
(the vocabulary, the tie-break rule, who speaks first); they can never share a
**plan** (who takes left here), because the level is new to both.

## What follows, and it splits by partner

**Agent to agent, fully observed, novel level: role assignment is mostly a
zero-bit problem.** Both agents run the same deterministic function of the
observed state — e.g. "whoever is counterclockwise of the barycentre takes the
inner objective" — and coordinate without transmitting anything. This is the
defensible version of the leakage intuition that [0015](0015-architecture-pick.md)
got wrong and [0016](0016-communication-is-equilibrium-selection.md) over-corrected:
the message is not redundant because the world is observed; the **tie-break** is
computable because the geometry is observed *and the rule is shared*.

**Agent to human: no shared rule exists.** The human will not run the tie-break
function and cannot be handed it mid-flight. This is the only place a channel
reliably earns its keep, which also means the human case — not the agent case —
is what should drive the communication design.

## Consequences
- **Legible motion is promoted from an option to the primary mechanism.** With no
  pre-play and no shared convention, the fastest way to say "I am taking left" is
  to commit left early and visibly. Dragan's legibility objective is exactly
  this: maximise the observer's posterior on your goal given the trajectory
  prefix. The opening seconds are the coordination window, which coincides with
  where the necessity profile puts the high-causal-effect decisions.
- **The design requirement in 0016 sharpens.** Not "multiple viable role
  assignments" but **does the geometry break the tie**. Measure the
  *role-assignment margin*: the cost difference between (A inner, B outer) and
  the swap. A large margin means a shared rule suffices and a channel adds
  nothing; a small margin is where communication pays. Near-mirror starting
  positions relative to the objectives is therefore a generator knob.
- Zero-shot coordination with a *novel partner on a novel level* is strictly
  harder than the Hanabi/ZSC setting, where the task is fixed and only the
  partner varies. Expect the published cross-play numbers to be optimistic for
  our case.
- A shared tie-break rule is a convention, so it must be **stated and fixed by
  us**, not learned in self-play — or two independently trained agents will hold
  incompatible rules and the zero-bit coordination silently fails.

## Record of revisions
This question was revised three times in one session: 0015 (channel vacuous
because fully observed — wrong), 0016 (channel essential for equilibrium
selection — right in general, wrong about pre-play), and this record. The stable
formulation is the partner split above. Logged because the oscillation is itself
evidence that the relevance of a factor needs checking before it is reasoned
from, which is now the standing mandate in CLAUDE.md.
