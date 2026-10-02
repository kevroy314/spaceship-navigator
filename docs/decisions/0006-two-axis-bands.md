# 0006 — Model scale and execution margin stay separate axes

**Date** 2026-10-01 · **Status** accepted

## Context
We wanted to name the levels where human intuition fails. Two literatures,
reviewed independently, both insisted the same thing: a policy's *rate* (how few
bits of state-to-action mapping it needs) and its *precision* (how finely it must
be timed) are different quantities. A bang-bang controller on a razor-thin
switching surface is about one bit of policy sitting on an impossible trigger.

## Decision
Never collapse them into one difficulty number. Band a level on both:
- **model scale** — the coarsest `gravity_at(topk=k)` under which a planner still
  arrives in the true world. k=1 is patched conics.
- **execution margin** — timing slack against measured human limits.

Bands: `trivial`, `human`, `calculated` (intuition fails, computation works),
`precision` (no hand can hold it), `opaque` (nobody; rejected as unfair).

## Consequences
- `calculated` and `precision` are *different kinds* of impossible, and need
  opposite fixes: widen a `precision` level's window and it becomes
  `calculated`; simplify a `calculated` level's field and it becomes `human`.
- The human thresholds are measured, not invented: ~20 ms motor onset jitter for
  a practised discrete action (Repp 2005), ~200 ms visual reaction time, ~3 Hz
  closed-loop correction ceiling (Franklin et al. 2019). Control here runs at
  15 Hz, so one tick is 67 ms — already coarser than a trained hand, which makes
  the test generous rather than rigged.
- Collapsing the axes would hide exactly the levels we are hunting.

## Sources
Lai & Gershman, policy compression (PLoS CB 2024); Furuta et al., Policy
Information Capacity (ICML 2021); Repp 2005; Franklin et al., J Neurosci 2019.
