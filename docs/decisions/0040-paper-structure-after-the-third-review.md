# 0040 — The paper's structure after the third review: twelve pages of primary text, the clock as a choice, the orbital-period line

**Date** 2026-11-04 · **Status** accepted · **Revises 0038**

## Context
The author's third review (2026-11-04) accepted sections 1–3 of the
previous draft in substance and asked for a different shape:

- Primary content limited to **12 pages excluding figures**; move
  argument detail to appendices and state conclusions in the discussion
  with pointers.
- Sections as Problem/Intro/Task · Difficulty · Pilots/Agents/Solvers ·
  Results · Discussion · Future directions. The short "defining
  difficulty" section is split between the intro and the difficulty
  section; the navigators section must describe the reference pilot and
  the optimizer as well as the learner; the suites table belongs with
  evaluation, not with the agent.
- **The clock.** The author's position: the 150 s timeout is clearly too
  short; the failure mode should almost never be timeout; "levels on the
  order of one orbit" was never a constraint he gave — it emerged from how
  the simulations were built and is not one he wants to keep. The clock
  material (formerly §3.6, 1,563 words, the longest subsection) is "an
  assumption and a choice more than a finding" and goes to an appendix,
  disentangled from difficulty. At the same time he found the discussion's
  argument for episodes of order one orbit compelling and asked that the
  line be drawn carefully.
- "What ___ buys" and "one <thing>, <statement>" are AI-coded; the
  regimes figure needs a flight that visits more than two states, a
  complete legend, annotated hazards and distinct colours.
- Future directions should be longer; the paper is an invitation to
  propose better definitions and better pilots.

## Decision
1. **Structure.** 1 Navigating an n-body field (history, RL, task, two
   pictures, why) · 2 Defining difficulty (reachability derivation;
   environment; representational; execution; chaos; solvability; clock) ·
   3 Navigators (pilot; optimizer; learning agent) · 4 Results
   (evaluation; vs pilot; budget; channels; per-instance; relations) · 5
   Discussion · 6 Future directions. Appendices: A observation · B
   theoretical notes · C complexity · D measuring chaos · E equal-pull
   surface · F the episode clock (the whole former §3.6 with its two
   figures and the multi-target clock paragraph) · G observation channels
   (argument and width experiment) · H goals · I reproduction (now with
   the PPO hyperparameters).
2. **Page budget.** Measured by compiling the main text with every figure
   and table environment removed: 13 pages of which the last is a third
   full, with the title block on the first, i.e. ~12.2 pages of body text
   at 7,857 words. Each further cut is recorded in the commit.
3. **The clock.** The main text (§2.6, ~200 words) states that the clock is
   a choice inherited from the game, that the evidence says it is short
   (optimizer arrivals 5/32 → 13/32 from 150 to 600 s; a third of
   medium-suite failures at 150 s are timeouts), that the position taken
   is that timing out should rarely be the way an episode ends, and the
   two caveats (the discount must match the clock; no finite clock settles
   solvability, and hazards supply the stopping criterion here). The
   measurements live in Appendix F.
4. **The orbital-period line** (§5.3) is drawn between two tasks, not two
   clock values: a task on the timescale of an orbit, in which difficulty
   is representational, the equal-pull surface is the right scale, the
   field is quasi-integrable, and the clock should be long enough within
   that timescale that timeouts are rare; and a task over many orbits, in
   which resonance overlap and constructed chaos make difficulty
   dynamical and the present measures are starting points at best. The
   paper studies the first and argues it is the right first step. This
   supersedes any reading of earlier drafts in which "episodes shorter
   than an orbit" was presented as a constraint of the task.
5. **Abstract** rewritten to sell the work (the question, the derivation,
   D with its prediction, the quasi-integrable field, the clock as a
   confound, the agent's three results, the invitation).

## Consequences
- The former sections 2 and 3.6 no longer exist as main-text sections;
  every cross-reference was redirected (sec:complexity → app:complexity,
  the clock → sec:clock (short) and app:clock, the suites → sec:evaluation).
- The suites table moved from the agent section to §4.1.
- The width experiment's prose is in Appendix G; its table and the
  ablation figure stay in §4.4.
- `paper/drafts/spacenav_v2.tex` preserves the previous draft.

## Evidence
- The author's review text of 2026-11-04 (session record).
- Word counts per section from the scratch script used during the
  restructure; the figure-free compile described above.
