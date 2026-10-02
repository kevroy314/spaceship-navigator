# 0009 — Win fractions, not binary volumes; and alpha needs samples

**Date** 2026-10-02 · **Status** accepted

## Context
To show *necessary* versus *sufficient* decisions we sweep the single-burn family
— when to burn, which way, how much delta-v — which is a 3-D volume that can be
enumerated. But an n-body basin boundary can be fractal, and then a grid does not
sample the map, it aliases it.

## Decision
1. Each cell holds a **win fraction** over jittered draws inside its own
   footprint, not a yes/no. Well defined whatever the boundary looks like, and it
   converges with sampling rather than with grid refinement.
2. Report the **uncertainty exponent** alpha (Grebogi–McDonald–Ott–Yorke:
   f(eps) ~ eps^alpha) alongside, so nobody reads crisp regions into a map that
   cannot have them.
3. alpha reports `measurable: false` — and propagates as `None`, meaning
   *unknown*, not *fractal* — unless there is enough signal to fit it.

## Consequences, and two mistakes worth keeping
- **Aliasing, found the mundane way.** The arrival radius is 12 u at a ~500 u
  scale, so the required heading precision is ~3 degrees while the first grid
  stepped 15 degrees. The sweep was *missing* solutions, not finding fractal
  ones. Fixed by jittered cells plus spending resolution where solutions live
  (+-60 degrees about the bearing to target, not a full circle).
- **alpha = 0.00 was not fractality, it was no signal.** With `eps_samples = 96`
  the flip rates are tiny integer counts (0/1/3/5/5); one fitted alpha of 1.297
  was a two-point slope between 2 and 6 flips. Bootstrapped CIs straddle the
  0.45 threshold, so **0 of 5 exponents was trustworthy** and the
  fractal-vs-resolvable split was not established. The fix is `eps_samples` up
  one to two orders of magnitude — a memory/parallelism problem, which is why a
  24 GB card matters more here than FLOPS.
- A saturation check is still required before trusting any alpha: if f(eps_min)
  approaches 2p(1-p), two points are effectively independent and the slope is
  meaningless. (Checked: *not* the failure mode in this run, cleanly refuted.)
- The single-burn family is too weak for hard levels — median sufficiency 0.0029,
  18 of 20 non-coast levels below 0.01, three of five causal maps effectively
  blank. Those maps report the limits of the probe, not the shape of the level.
  Needs a two-burn family (5 parameters).

## Sources
Grebogi, McDonald, Ott & Yorke, Phys. Lett. A 99:415 (1983); Daza et al., basin
entropy, Sci. Rep. 6:31416.
