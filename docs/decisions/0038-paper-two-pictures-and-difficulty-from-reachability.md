# 0038 — The paper pictures the task two ways, not three, and derives difficulty from reachability

**Date** 2026-11-01 · **Status** accepted · **Revises the structure recorded in 0034–0037's paper sections**

## Context
The author's review of the second draft (sections 1–3) made four
structural points, each a decision about how the paper frames the task
rather than a line edit:

1. The three "pictures" of the task (RL loop; state machine over regimes;
   stack of strategies) are really two. The RL loop and the stack are both
   points in one space of models of the field, from strategies a person
   writes down (dead heading, leading, conics, assists, low-energy
   transfers) to a policy learned with no structure imposed, with
   model-based methods between; the analogy is to forecasting (mean model,
   ARIMA, transformer), where models differ in which priors they commit to.
   The state machine over regimes (bound the exit of a chaotic region, do
   not predict it) is the other picture. The text must not call pictures
   "correct"; each has a utility and none is sufficient.
2. The six-reason list of why a level is hard was example-based. It is to be
   derived: a task is a path through a graph whose nodes are regions of the
   craft's state (position, velocity, time, fuel, health), whose edges are
   the transitions the field and engine admit, each with a cost in fuel,
   time and exposure. Difficulty is then a path failing to exist (budget
   exceeded; degenerate target geometry), failing to be found (wrong
   representation; scattering encounter makes edges distributions), or
   failing to be flown (timing window narrower than the controller).
3. D = η/A is a first-order measure that implies a brute-force navigator
   (one that spends fuel to correct its representation's error) and credits
   nothing to a navigator that uses the field (bets on an encounter's
   outcome; waits out a long timescale), which is the trick of every mission
   in the introduction. The paper must say so where D is defined.
4. The families are samples of the configuration space, not a taxonomy, and
   the text must present them as the means by which the level-defining
   parameters are bootstrapped (mass hierarchy; what besides gravity
   constrains the route; whether the dominant structure persists).

Plus: "delta-v budget" not "tank"; Braintruffle to the acknowledgements
unless there is a paper to cite; the intro must not carry the methods
(N, T, H stated generically in §1, the values 32 / 150 s / 15 Hz in §3.1);
a final §1 paragraph on why (the configuration space is effectively
unbounded; a difficulty scale allows pedagogical bootstrapping for a person
or an agent); an early schematic figure of the three historical missions;
the simulation assumptions as a table; no X.1 without an X.2.

## Decision
All of the above, applied in `paper/spacenav.tex` sections 1–3, with the
knock-on references in §3.3, §6 and Appendix B updated (the "first picture"
is now the state machine; the "second" is the model space). Two bib entries
support the forecasting analogy (`box2015`, `vaswani2017`) and one the
model-based middle of the spectrum (`moerland2023`). The missions figure is
`scripts/fig_missions.py` → `paper/figures/missions.pdf`; the assumptions
table is `paper/table_assumptions.tex`, every value traced to code.

Points 1 and 2 are the author's framing and are recorded as such; they are
not measurements. Point 3 is a consequence of D's definition (η is an
instantaneous force ratio; nothing in it locates divergence or credits a
transfer gravity performs) and is stated in the paper as a limitation of
the measure, with the chaos, solvability and clock sections named as the
beginnings of the higher-order terms.

## Consequences
- Every later section that said "the third picture" or "the stack of
  strategies" now says "the model space"; "the second picture" (regimes)
  is now "the first". The companion site (`docs/index.html`) must follow.
- The sentence "Of the three pictures, this is the one that yields a
  closed-form measure" survives as "of the two".
- The abstract still says "three" in spirit and "tank" in letter; the author
  is reviewing it separately and it is left for that pass.
- `paper/README.md` carried the first draft's title and a paragraph about
  "retractions"; both contradicted `paper/STYLE.md` and are replaced.

## Evidence
- The author's review, pasted into the session on 2026-11-01 (paragraph
  references 1-1 through section 3), and the preceding review of the first
  draft.
- `paper/spacenav.tex` at the commit that cites this record; `paper/style/
  antipatterns.md` §4–6 for the audits of the earlier drafts.
