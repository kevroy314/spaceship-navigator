# 0015 — Architecture: search the true simulator with mctx; reduce the horizon

**Date** 2026-10-02 · **Status** proposed

## Context
Four parallel literature reviews on modern RL, chaos, language-over-policy, and
multi-agent communication. Supersedes nothing, but makes
[0010](0010-planner-as-teacher.md) concrete and kills several candidates.

## What the reviews settled

**PPO is not obsolete for our shape.** It still owns wall-clock on massively
parallel GPU sim, but its edge is contingent on environment count and the
benchmark lead belongs to the regularised off-policy family (SimbaV2 0.892
aggregate over 57 tasks vs TD-MPC2 0.749, DreamerV3 0.397). Caution: two
confounds (action-bound conventions of +-100, and rewards tuned for PPO)
invalidate much of the published PPO-vs-SAC record.

**Don't learn a world model when you own a simulator.** LightZero, verbatim:
*"employing AlphaZero directly is advantageous when an environment simulator is
available."* DeepMind's **`mctx`** is actively maintained, runs AlphaZero /
Gumbel MuZero batched on GPU, and its `recurrent_fn` accepts a real simulator
step. That is [0010](0010-planner-as-teacher.md) already implemented. Gumbel-Top-k
plus Sequential Halving guarantees policy improvement at low budgets — MiniZero
trains it at **2 and 18 simulations**.

**Horizon reduction is the cheapest large win.** On 4000-step humanoidmaze-giant,
flat n-step SAC+BC with **n=50** scores 91+-2 against HIQL's 72+-2; DEAS finds
H=1-2 "fails to achieve meaningful performance". Our episodes are 2250 steps.
1-3 engineer-days for an n-step change.

## Decision (to try, in order)
1. **n-step returns, n ~ 25-50.** Cheapest, highest expected return per day.
2. **`mctx` Gumbel MuZero with `recurrent_fn` = our real step**, policy as prior,
   search output distilled back into the policy. Benchmark a fixed 16-32
   simulation budget *first*: the sequential tree depth serialises kernel
   launches, which is our real constraint, not arithmetic.
3. **SimbaV2 normalisation in the encoder and critic** — ablate, do not assume;
   every number comes from off-policy UTD>=1 settings.

## Rejected, with reasons
- **Trajectory-diffusion planners.** On AntMaze — the stitching task — Diffuser
  13.3, Decision Diffuser 3.0 against single-step DQL 80.5; 0.2-2.8 s per
  decision on a 3090 (so 5-70 s on Pascal, against our 66 ms); and their
  generated actions are causally inconsistent with their own generated states.
  With 9 discrete actions a softmax already represents every distribution, so
  the multimodality argument does not even apply.
- **Decision Transformer as RL.** A 2-layer MLP matches it, and return-conditioned
  SL provably needs near-determinism — contraindicated for a chaotic tour.
- **Hierarchical RL with a learned manager.** Flat n-step beats HIQL on the
  longest-horizon benchmark published; option-critic collapses structurally;
  URLB finds no competence-based method SOTA on any task.
- **Learned skill codebooks as a semantic inventory.** Vocabulary size is a free
  parameter (LISA scores 47% at K=10, 20 and 50 alike), utilisation caps near
  1024 codes, and the only ground-truth boundary measurement puts prior skill
  discovery at 0.19-0.27 precision. Nobody publishes a 10-30 code vocabulary with
  50-300 step spans. **Name first** ([0011](0011-callouts-and-legibility.md)).
- **An LLM narrating the policy.** Measured faithfulness of naive LLM
  explanation of an RL agent: **0.14**, against 0.99 fluency — rising only to
  0.46 with an explicit verification pipeline. Explanations run *as* policies
  score -69.2 on LunarLander. Post-hoc narration fails Rudin's critique
  empirically, not just philosophically.

## The one thing to design against
> **Corrected by [0016](0016-communication-is-equilibrium-selection.md).** The
> argument below is scoped to the *single-agent* navigation phase and to messages
> whose content is derivable from the observation. It does not apply to
> coordinating messages ("you left, I'll right"), which carry a joint choice that
> is not in the world state and that full observability cannot supply.

Our environment is fully observed: `obs` already contains every body's state. A
plan-conditioned policy will therefore probably learn to infer the right action
from `obs` and **ignore the plan** — concept-bottleneck leakage, in the control
path. The announcement then becomes vacuous, and our own necessity test will
correctly report near-zero counterfactual divergence. Partial defences (drop the
plan during training so obedience is required; shrink the policy; add levels
whose timing a reflex cannot resolve) all trade reward for announceability.
Expect to pay **5-20% of return** for an announcement that passes an intervention
test, and treat a configuration that pays nothing as evidence of leakage rather
than success.

## Scale note
Kinetix — 2D JAX physics with a PureJaxRL-style PPO much like ours — trained at
**376B env steps across 1M parallel environments**. At 400M steps we are three
orders of magnitude below where that architecture class was shown to generalise,
so some of what looks like an architecture problem may be a throughput problem.
Also: **PureJaxRL is stale** (last push 2024-09); vendor the loop rather than
depend on it.
