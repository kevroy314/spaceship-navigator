# 0021 — A ~1% win rate is the measurement, not a defect

**Date** 2026-10-09 · **Status** accepted · **Corrects** [0019](0019-revert-the-clock-for-measurement.md)

## Context
[0019](0019-revert-the-clock-for-measurement.md) called the near-blank causal
maps a failure of the probe — "the one-burn strategy family is the binding
constraint" — and prescribed a two-burn family (five or six parameters) as the
fix. That prescription has now been built and measured, and the premise was wrong.

## Measurement
Same level (`long-14-466303-1w100m`), same 150 s clock, `scratchpad/tbtime.py`:

| family | parameters | win fraction |
|---|---|---|
| one burn | 3 | 0.0064 |
| two burns | 6 | **0.0098** |

1.5x, not 10x, and at 1024 draws the standard error at p ~ 0.01 is +-0.003, so
the gap is barely resolvable. **Enriching the family does not materially change
the win rate.**

## The reframe
A ~1% win rate is the correct answer to the question the probe asks. The question
is "what fraction of *randomly chosen* burns arrive", and arrival means passing
within 12 u of a target at scales of several hundred u, from any of 450 s of
burn times across 120 degrees of heading. One percent is a plausible, honest
number for that, not a broken instrument.

What the map was wanted for was the *shape* of the winning set, and 1% of 2880
cells is still ~29 lit cells with structure in them. The right response to that
was the visualisation — stacked planes with the empty cells transparent, so the
winners read as clusters rather than as a near-black square — not a richer
strategy family.

## Consequences
- **Keep the one-burn volume as the visual object.** It is three parameters, so
  it is the only member of this family that can be drawn at all, and enriching it
  buys ~1.5x on a number that is not the point.
- **Keep the two-burn probe, once, as the evidence** for this record. At 1024
  draws it costs ~28 s per level, which is affordable; 4096 draws would cost 112 s
  per level for a standard error that still does not resolve a 1.5x gap.
- `sufficiency` should be read as "how forgiving is this level to a guess",
  which is a genuine axis: the coast levels measured 0.27-0.58 against ~0.006
  for the rest. That two-orders-of-magnitude spread is the signal, and it was
  always there.
- Three of [0019](0019-revert-the-clock-for-measurement.md)'s four stated causes
  have now been corrected or withdrawn: the clock claim
  ([0020](0020-the-clock-was-not-the-cause.md)), the family claim (here), and the
  chaos claim was already superseded by
  [0018](0018-chaos-needs-close-encounters.md). The one that stands is the timing
  axis being a hundredfold coarser than the threshold it was compared against —
  that was a real defect and is fixed by using `delay_window`.

## Performance note, recorded because it cost 40 minutes
`two_burn_sufficiency` first used `jax.lax.map`, which reads like a parallel map
and is a **sequential** scan: 4096 draws of 2250 ticks is 9.2M sequential steps,
about an hour for a single level. Chunked into `lax.map` over N chunks with
`vmap` inside each, the sequential depth drops to one episode per chunk.
