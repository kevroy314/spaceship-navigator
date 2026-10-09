# 0018 — Longer episodes do not produce chaos; close encounters do

**Date** 2026-10-04 · **Status** accepted · **Corrects** [0013](0013-longer-episodes.md)

## Context
[0013](0013-longer-episodes.md) proposed 600 s episodes on the reasoning that at
lambda ~ 0.007 1/s a 150 s episode spans about one e-folding, so 600 s would span
~4.2 and "chaos stops being an aspiration and becomes a property". It required
re-measuring lambda at the new length before training. That measurement is in,
and the premise was wrong.

## Measurement
A one-tick-of-thrust velocity perturbation, 16 levels x 8 families, coasted for a
full episode; `scratchpad/lam600.py`. Fitting sep = a*t*exp(lambda*t) to divide
out ballistic growth:

| | lambda median | e-foldings per episode | slope ratio (0.50 = linear) |
|---|---|---|---|
| 150 s clock | 0.0045 – 0.0086 | ~1 | 0.33 – 0.64 |
| **600 s clock** | **-0.0001 – 0.0013** | **-0.09 – 0.80** | **0.14 – 0.80** |

Longer episodes made the system **more** predictable, and growth is *sub*-linear.

## Why
These are **bound** systems. A velocity perturbation puts the ship on a nearby
bound orbit, so the separation **oscillates with the orbital period** rather than
growing. At 150 s you observe the linear phase of a partial orbit; at 600 s you
observe several orbits and the separation saturates. Extrapolating a lambda fitted
inside one orbital period across many was the error.

The system is **quasi-integrable**, and the cause is our own generator:
`planetary_system` enforces 7 mutual Hill radii of separation and e <= 0.06,
which deliberately produces well-separated, non-interacting orbits. Chaos
requires close encounters.

## The real tension
**Level validity and chaos are in direct conflict under the current generator.**
A level is accepted because it stays well-behaved for the horizon and its bodies
keep clear of one another; that is exactly the condition that prevents the
scattering which would make it chaotic. No episode length resolves this.

Note what this means for [0005](0005-equal-pull-not-resonance.md): rejecting the
resonance-overlap framing on timescale grounds was right for the *Wisdom band
placement* knob, but the **spacing-relaxation** knob from the same research
(`chaos_kappa`, relaxing the 7-Hill-radii guard toward the overlap threshold) was
never implemented or tested. That is the knob that produces close encounters, and
it remains the open route to chaos.

## Where chaos already exists
In the tail, in the clusters: p90 e-foldings per episode reach **3.58
(bh_cluster)** and **2.88 (star_cluster)** against medians near zero. So chaotic
levels are being generated, rarely, in the loosely-bound families. Selecting on
measured p90 divergence is a cheaper route to a chaotic subset than changing the
generator.

## Scope of the lambda measurement (added 2026-10-09)
These numbers are for a **coasting ship in a bound orbit**, which is what the
probe rolls. A *thrusting* flight that escapes diverges far faster: a JS/JAX
parity diagnostic on `val_seen-900-5-t` -- a 304 s pilot flight ending `lost` --
shows float32-vs-float64 differences amplifying at about **0.036 1/s**, five
times the coasting figure, with the two implementations terminating 5 ticks
apart. So lambda here is trajectory-dependent and 0.007 1/s should not be quoted
as "the game's lambda"; it is the bound-coasting value, which is the right one
for the claim this record makes and the wrong one for anything about escapes.

## Consequences
- The `rolling-with-it` and `lottery` lessons will stay empty at 600 s. They were
  never blocked by episode length.
- 600 s is retained anyway, for a different and better reason: it gives a single
  burn time to reach a target, which is the fix for the near-blank causal maps
  (median sufficiency was 0.0029, see [0009](0009-probability-field-and-alpha.md)).
  Whether it delivers that is a separate measurement.
- The 900 s validation horizon cost nothing: acceptance stayed at 100% for eight
  of nine families, 93% for `star_cluster` — the loosest-bound family, as expected.
- Before claiming chaos anywhere, measure lambda **over the horizon actually
  used**. A rate fitted inside one orbital period does not extrapolate across many.
