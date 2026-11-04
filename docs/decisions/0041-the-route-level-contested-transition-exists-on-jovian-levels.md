# 0041 — A route-level well-behaved/contested transition exists on jovian levels, and the raw FTLE labels 40 % of such a flight chaotic

**Date** 2026-11-04 · **Status** accepted · **Extends 0039**

## Context
0039 measured the regime states along flights on star-cluster (contested
throughout) and asteroid-field/sol-like (never contested) levels and noted
that a transition along a route, into and out of an equal-pull shell, had
not been measured. The author's review asked for a regimes figure on a
flight that visits more than two states.

## Measurement
`scripts/fig_regimes.py`, same thresholds and policy as 0039
(`ppo_v5_noinstr` checkpoint 3000, 16 sampled rollouts, `PRNGKey(7)`), on
a screen of all 528 hard- and medium-suite tasks of the jovian, saturnian,
sol_like and bh_cluster families (η and hazard per tick; 45 min CPU), then
both exponent readings on seven candidates.

| family | n tasks | well | contested | hazard | tasks contested > 5 % |
|---|---|---|---|---|---|
| jovian | 132 | 90.1 | 7.1 | 2.8 | 38 |
| saturnian | 132 | 93.6 | 6.1 | 0.3 | 43 |
| sol_like | 132 | 62.6 | 1.1 | 36.3 | 10 |
| bh_cluster | 132 | 27.0 | 23.5 | 49.5 | 79 |

(rollout-1 means, % of ticks). Chosen flight hard:339 (jovian, level 309,
R = 919 u), ballistic reading: 55.0 / 28.7 / 1.3 / 15.0 % well / contested
/ chaotic / hazard on rollout 1, 53.0 / 27.1 / 1.9 / 18.0 over 16 rollouts;
raw 20 s FTLE reading 31.7 / 11.6 / 41.6 / 15.0 and 30.4 / 10.4 / 41.2 /
18.0. The contested segment (37–79 s) is a moon pass with η up to 2.0 at
26 u; the hazard segments are three radiation-belt crossings from 102 s;
the two chaotic samples (λ = 0.0167, 0.0263 s⁻¹) are single 1 s windows at
23 u and 60 u from moons and may be tidal.

## Decision
The paper's regimes figure uses this flight. The §2.4 text states the
fractions, that the contested state appears on 38/132 jovian tasks, and
that on the present levels the machine reduces to the mass hierarchy with
hazard entered from either state. The chaotic hue is changed to #9B5DE5:
the dataviz validator fails the previous purple against the state blue
under deutan simulation (ΔE 1.2).

## Consequences
- 0039's "not measured" note is now measured: the route-level transition
  exists and is common in hierarchical families with an equal-pull shell on
  the route.
- The raw FTLE trap is worse beside a giant than beside a star: ~40 % of a
  jovian flight reads chaotic under the raw exponent against 1–2 % under
  the corrected one.
- The strict "top three by contested time" tasks reach no targets and skim
  a star for their hazard time; the figure's flight was chosen for
  annotatable causes (moon pass, belts) and for reaching two targets, and
  the choice is recorded here so it is not read as representative of the
  family means above.

## Evidence
- `scripts/fig_regimes.py` (defaults `--task hard:339 --chaos ballistic`;
  `--all` prints the table); screen scripts and pickles in the session
  scratchpad.
- 0039 — the first measurement and the FTLE trap.
