# 0012 — Evaluation statistics we actually need

**Date** 2026-10-02 · **Status** accepted

## Context
Our outcome is near-binary (arrived or not). We have been quoting per-condition
results at n = 3 and n = 8.

## The arithmetic
95% CI half-width ~ 1.96*sqrt(p(1-p)/n). At p = 0.5:
- **n ~ 384** for +-5 percentage points
- **n ~ 9,600** for +-1 pp
- **~1,500 per arm** to detect a 5 pp difference at 80% power

Two variance sources must be separated and never reported as one another:
**training-seed** variance (needs >= 5–10 independent runs, bootstrapped) and
**episode/initial-condition** variance (needs thousands of held-out ICs).

## Decision
- Freeze >= 2,000 held-out initial conditions; run >= 5 training seeds.
- Report **IQM** with stratified-bootstrap intervals (the right aggregator for
  bimodal scores), plus worst-case-over-solvable-levels.
- Estimate the **irreducible miss rate** by Monte Carlo over perturbations of a
  near-optimal planner reference, and score the agent as a fraction of that
  ceiling. A ceiling below 100% is not a training failure.
- Curriculum difficulty must be a **repeated-rollout** estimate of p, never a
  single rollout or a value-loss proxy. Existing UED approximations correlate
  with success rate rather than regret and fail to find levels the agent "can
  sometimes solve but not always" — which is our exact setting.

## What this invalidates in our own work
- The *Two Routes Per Rung* per-rung figures (n = 8) are not significant. Only
  the pooled comparison survives (24 per band, Fisher p = 0.005). The published
  page should say so.
- **Monotonic improvement across versions is currently unverifiable.** An early
  project requirement was that checkpoint versions improve monotonically, as an
  overfitting check. But v3 changed the suites and the observation shape, and
  2026-10-02 changed it again (MAX_WAYPOINTS 4->6, four instrument channels), so
  old checkpoints cannot be re-scored. Either freeze an evaluation interface with
  an observation adapter, or accept that version comparisons restart at each
  break and say so explicitly. Right now we have neither.

## Sources
Agarwal et al., Statistical Precipice (arXiv:2108.13264); Colas, Sigaud &
Oudeyer (arXiv:1806.08295); Rutherford et al., No Regrets / SFL
(arXiv:2408.15099); Jiang et al., aleatoric UED (arXiv:2207.05219).
