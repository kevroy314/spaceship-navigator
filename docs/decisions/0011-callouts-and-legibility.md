# 0011 — Hand-designed callout vocabulary; legible trajectories for humans

**Date** 2026-10-02 · **Status** proposed

## Context
Target: a wingmate that privately calls out the manoeuvre it is about to make,
with the announcement causally tied to what it then does. A flat policy has
nothing to announce — PPO maps observation to action with no intermediate object
that could be named. Announcement requires discrete structure.

## Decisions (to build, not yet built)

**1. Do not learn the message.** Fix ~8–12 manoeuvre tokens in the orbital frame
(`prograde-burn`, `retrograde-burn`, `radial-in/out`, `periapsis-raise`,
`intercept`, `break-left/right`, `reload-run`, `bomb-run`), each defined by a
deterministic labeller `L(next 1–3 s of trajectory) -> token`. This buys grounded
semantics, a human-readable alphabet, and immunity to the private-code problem.

The emergent-communication literature does not support learning it: without an
explicit length penalty agents assign the *longest* codes to the *most frequent*
meanings (anti-Zipf); compositionality is neither necessary nor sufficient for
generalisation; natural language does not emerge unless forced.

**2. Faithfulness by construction.** An auxiliary intent head predicting its own
token at horizon H, trained with a supervised consistency loss against
`L(realised trajectory)` — self-labelling, so free data — plus a commitment
penalty so the policy either follows through or stays quiet.

**3. Measure what the field does not.** Per eval: announcement accuracy,
commitment-breaking rate, and **positive listening** by counterfactual token
swap. Lowe et al. show positive signalling and positive listening are
*independent* — agents exist whose messages perfectly predict their own next
action with zero causal effect on anyone. Most reported emergent communication
satisfies only the first.

**4. Legible trajectories for the human channel.** Dragan's formalism separates
*predictability* (your path matches what I expect given your goal) from
*legibility* (I can infer your goal from your path); they can conflict. A ship
flying a slightly exaggerated arc needs zero bandwidth, zero latency and no
vocabulary. Candidate goals here are discrete, so the inverse-planning observer
is cheap. Expectation: legible flying for humans, faithful tokens for
agent-to-agent, tokens surfaced to the human only as low-rate confirmation.

## Rejected or demoted
- **Weight sharing as a coordination channel.** It gives mutual predictability
  for free (each ship can simulate the other), but conveys no runtime
  information, cannot break symmetry, and on multi-modal reward landscapes
  shared policies provably converge to *averaged* solutions where independent
  ones reach optima (Fu et al., ICML 2022). "Who flanks, who baits" is exactly
  multi-modal, so sharing will hurt role assignment. Useless for the human case.
- **Ensembles of differently-biased policies as a deployed team.** No supporting
  evidence. Diversity is for generating **training partners**: best-response to a
  population plus mid-training checkpoints (Fictitious Co-Play) beats training
  against human data and humans prefer the result.
- **Explanations as a performance aid.** Bansal et al. (CHI 2021): explanations
  raised the probability a human accepted the AI's recommendation *regardless of
  correctness*, without improving accuracy. Pre-register joint score as primary;
  treat trust as secondary.

## Mandatory baseline
Before any channel: **full observation sharing** (concatenate both ships'
states), with network capacity held constant across arms. Much of the comms
literature's reported gains dissolve into "bigger network with a comms path".

## Sources
Lowe et al. (arXiv:1903.05168); Foerster et al. DIAL (arXiv:1605.06676); NDQ
(arXiv:1910.05366); Kim et al. Intention Sharing (ICLR 2021); Fu et al.
(arXiv:2206.07505); Strouse et al. FCP (arXiv:2110.08176); Dragan, Lee &
Srinivasa (HRI 2013); Bansal et al. (arXiv:2006.14779); Pope et al.
AlphaDogfight (arXiv:2105.00990).
