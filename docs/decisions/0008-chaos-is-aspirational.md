# 0008 — The game is not chaotic yet; chaos must be built

**Date** 2026-10-02 · **Status** accepted

## Context
Several design threads leaned on the n-body field being chaotic: fractal basin
boundaries, an uncertainty exponent, a `lottery` band, a predictability cone, and
a worry that a fractal policy objective would have no descent direction (Wang,
Herbert & Gao, NeurIPS 2023: J is alpha-Hoelder with alpha = log(1/gamma)/lambda,
and fractal when lambda > log(1/gamma)).

At gamma = 0.999 and 15 Hz that threshold is lambda_crit = 0.0150 1/s, i.e. the
Lyapunov time must exceed 67 s. Our measured predictability horizon — where one
control tick of thrust error grows past the 12 u arrival tolerance — is 52–74 s
across eight families, which looked alarmingly close.

## Decision
Measured it properly instead of assuming. **The divergence is ballistic, not
exponential**, so the alarm was false and gamma = 0.999 is fine.

The discriminator: d ln(sep)/dt decays as 1/t for linear growth and stays flat
for exponential. Pure linear predicts a 150 s/75 s slope ratio of exactly 0.50;
we measured 0.33–0.64, median ~0.56. Fitting sep = a*t*exp(lambda*t) to divide
out the unavoidable ballistic factor gives lambda = 0.0045–0.0086 1/s median
(p90 ~0.010), i.e. **about 2x margin** below lambda_crit. Only `star_cluster`'s
tail exceeds it (p90 0.028).

## Consequences
- Lyapunov time ~140 s against 150 s episodes is roughly **one e-folding per
  episode**, ~2000 control steps per Lyapunov time. The literature's successful
  chaotic-control results sit at 10–100 steps per e-folding; we are far inside
  the comfortable regime, not the hard one.
- This explains three loose ends at once: alpha was unmeasurable because flips
  are genuinely rare; `rolling-with-it` and `lottery` found zero instances
  because there is no chaos to roll with; the 20 s prediction cone never expires.
- **Retracted:** an earlier claim in this project that run v3's plateau (2% for
  13M steps, 21% final on the hard tier) was explained by a fractal objective.
  It is not. That plateau needs another explanation.
- The 67 s predictability horizon is still a real and useful design number. It
  just comes from linear drift of a velocity error, not from chaos.
- To get chaos we must **construct** it: closer encounters, tighter binaries, and
  above all longer episodes relative to orbital periods. See
  [0013](0013-longer-episodes.md).

## Evidence
`scratchpad/ischaotic.py`, `scratchpad/lam.py`, 16 levels x 8 families, one tick
of thrust (0.107 u/s) as the perturbation.
