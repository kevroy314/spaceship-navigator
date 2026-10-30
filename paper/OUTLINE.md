# Outline — authoritative structure for the paper

This is the author's structure, written before reading the first draft and
therefore the thing the paper has to match. The first draft told *our* story
(what we tried, what we retracted). This paper tells *the* story: here is a
task that is hard to define, hard to judge and hard to learn; here is how one
can think about it; here is an invitation to try.

Working title direction: direct, about the task, not about "missions".
Candidates: *N-Body Navigation: …*, *Learning to Navigate an N-Body Field: …*.
Subtitle from the first draft is fine.

The word "mission" is retired except where a historical mission (Voyager,
Parker, Artemis) is the point. Say *task*, *task instance*, *level*.

## 0. Abstract
Sells the work. What is the task, why is it interesting, what did we find.
Not a list of our stumbles.

## 1. Navigating an n-body field
Brief treatment of the problem: a thrusting craft in a field made by several
moving bodies, reaching targets that are themselves moving. Nod to the
literature on n-body dynamics and spacecraft guidance. Call out the
**Braintruffle** video series as a major inspiration for the project. Keep it
short; the reader should feel the shape of the problem, not get a course.

Three ways to picture the problem, which the rest of the paper returns to:

1. **A plain reinforcement-learning loop.** Observe, act, be rewarded. This
   works until the craft enters a chaotic regime, and then it works only if the
   algorithm happens to be compatible with what chaos does to the objective.
2. **A state machine with noise filters.** Chaotic regions are filters the
   craft passes *through*. The goal inside one is not to predict the exit
   trajectory precisely but to **bound** it tightly enough that adaptation on
   the far side is possible. Plan to adapt, not to predict.
3. **Stackable strategies.** Conics (one body matters), dead heading (point and
   burn), leading a moving target, gravity assists, up through carefully learned
   models. Each is a rung; a learner climbs them.

## 2. Why designing a level of a given difficulty is hard
Given the three framings, deciding whether a particular level is "difficult" or
"interesting" is genuinely tricky. This section argues that in general before
any of our machinery appears.

Include the computational-complexity nod: checking whether a solution is even
correct is, in the general case, out of reach. The action space is continuous
in direction and magnitude and the horizon is T steps, so an exhaustive search
is O(a^T) with a continuous. Exhaustively checking even a tiny level is
impossible without simplifying assumptions, and once a chaotic regime is
entered, precision of the check itself becomes a problem. [CITE complexity of
control / reachability; chaos and finite precision]

## 3. Heuristics for difficulty and interestingness, and where we landed
The quantities we actually use, introduced from the framings above rather
than as a list of knobs:

- **Representational (nontrivial) difficulty** — does a coarse model of the
  field suffice? `eta` (field a one-body model discards), `A` (fuel versus the
  impulse gravity delivers), their ratio `D = eta/A` (share of the tank a wrong
  model costs). Closed form, simulation-free, invertible. The equal-pull
  surface as the place a coarse model breaks. Level families as the biggest
  lever, each introduced *with the reason it exists*.
- **Reaction-time (trivial) difficulty** — timing slack against human limits.
  Mentioned, kept separate, not belaboured. It is the less interesting axis.
- **Chaos as a property to measure, not assume** — Lyapunov time versus the
  episode, the uncertainty exponent, the Hölder threshold for the RL objective.
  The field is quasi-integrable on episode timescales; chaos has to be built.
- **Solvability** — a level nobody can fly earns no label. A reference
  optimiser as the available-but-imperfect yardstick, and what it means that
  the true optimum is never available.
- **The episode clock is not a difficulty knob.** Time is the least interesting
  constraint. Fuel is harder; damage (radiation, debris, later projectiles) is
  the most interesting because of its dynamics. A craft that takes a long,
  energy-efficient path and is stopped by the clock has not failed the task; it
  has failed a hyperparameter. Voyager was not judged on reaching Jupiter in a
  year; Parker bleeds velocity over many assists; Artemis boosted over
  progressive passes. The clock sweep result belongs here as evidence that
  time was mis-set, not as a story about us.

Tables of the quantities. Figures that show them on real levels.

## 4. The RL training loop as a lens on the levels and the agent
Using a learner to understand the levels *and* to evaluate the learner.
Methods, briefly: the environment specifics (physics, observation, actions,
families, pools), the network, PPO and the curriculum, the frozen suites, the
reference pilot. Examples.

## 5. Results
Across training configurations. Correlations between the measures of interest
(eta, D, family, clock, success, the per-instance spread). The
instrument-channel finding belongs here as a result about *what to tell a
learner* — redundant versus new information — not as a surprise narrative.
The ghost fleets as the picture of what a single success rate hides.

## 6. Discussion
Pedagogical progression: learning this *class* of simulation rather than
overfitting one instance. RL over chaotic simulations. Then the caveats: how
arbitrary several of our definitions are, and why better intuitions and
definitions here help us understand both the task environment and the agents
that run over it.

## 7. Future directions
Brief. Mention only, and vaguely: adversarial situations such as in a video
game where one ship may be attempting to catch, target, or otherwise react to
another. Nothing more specific.

## Appendix
- The observation vector in full.
- Theoretical notes: a finite speed of light; a *projected* observation of body
  and goal positions versus a real-time measurement; what each would mean for
  a learner.
- The goal conditions in detail (flythrough, rendezvous, station, attached to a
  body, order-free tours).
- Reproduction.

## Throughout
Tables and visuals carry the intuition. One complex, possibly full-page task
figure early. Phase-space portraits computed on real levels, not schematics.
Dense citation into complex systems, complexity theory, dynamical systems,
RL and curriculum learning. Where a citation is not yet in hand, a
`[CITE: claim]` placeholder, to be resolved before the next review.
