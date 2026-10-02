# 0010 — Search over the true simulator; distil the policy, not the value

**Date** 2026-10-02 · **Status** proposed

## Context
We are not committed to PPO; we want whatever reaches near-human or superhuman.
The decisive fact about our setup is that **we already have an exact, fast,
differentiable simulator.** Most model-based RL exists to work around not having
one — DreamerV3 and MuZero spend their capacity learning dynamics we know.

Separately: the CEM planner currently *beats* the PPO agent, which is why it
serves as the reference flight. Its work is discarded after each use.

## Decision (to try, not yet built)
Close the loop: a learned policy and value **seed and shape a search over the
true simulator**, and the search's output is distilled back into the policy —
AlphaZero's recipe with our simulator in place of a learned model.
- receding horizon of roughly one predictability horizon, re-planned often
- a discrete skill vocabulary for temporal abstraction over 2250-step episodes
- plan at 1–2 Hz; the fast policy fills in the ticks between re-plans

## Why this shape
1. Largest available gain and weeks not months: the planner's answers are
   already better than the policy's and are currently thrown away.
2. It is the mechanism that goes superhuman on precision. A search can evaluate
   thousands of candidate burn times; a reactive policy must have memorised the
   reflex.
3. It unifies with explainability: if you plan at 1–2 Hz, **the plan is the
   object you announce**, so a callout is faithful by construction rather than a
   post-hoc rationalisation. See [0011](0011-callouts-and-legibility.md).

## Distil the policy, not the value
Izzo & Oeztuerk's ablation on Pontryagin-optimal low-thrust transfers is unusually
clean: propellant penalty **+0.015% for a policy network, +0.30% for
value-gradients, +38% for value-function-only**. Regressing V is far worse than
regressing pi. Combined with the fractal-landscape result, uniform value
approximation is the wrong target.

## Expectations to hold ourselves to
RL does **not** beat a good optimiser on optimality — every honest
astrodynamics comparison has RL losing by 0.3–10% (Bonasera et al.: 0.117 vs
0.129 m/s, ~10% worse; Scorsoglio et al.: 8.1% worse than GPOPS). So the planner
stays our quality ceiling and the agent's job is millisecond closed-loop
evaluation and graceful degradation. Frame the planner as **teacher**, not as a
baseline to surpass.

## Sources
Izzo & Oeztuerk (arXiv:2002.09063); Wang, Herbert & Gao (arXiv:2310.15418);
MBPO (arXiv:1906.08253, k=1 rollouts hard to beat); PETS (arXiv:1805.12114).
