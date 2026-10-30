# Writing policy for the paper

Every sentence of prose in `paper/` is written against this file. Two further
files, produced by the style pass, extend it: `style/voice.md` (what the
author's own writing does) and `style/antipatterns.md` (what rejected writing
does). When they exist, read them too; when they conflict with this file, this
file wins.

## Whose story

Tell *the* story, not *ours*. The reader wants to understand a task and a way
of thinking about it. They do not want a retrospective of what we tried, what
we got wrong, or what we corrected. Nothing in this paper has been published
before, so there is nothing to "retract"; a measurement that moved our
thinking is reported as what it shows about the task, in the place the
argument needs it, with no narrative about us.

Concretely: no "our own earlier conclusion", no "we previously reported", no
"turned out to be an artifact", no "we got it wrong in public". If a result
corrects a plausible-but-wrong intuition, state the intuition as *the natural
expectation* and then the result.

## Sentences

The rejected register is short, stunted, declarative sentences that open with
"We <verb> a/the <noun>": *We give a closed-form score. We argue that it must
never be collapsed. We release the environment.* That cadence is immediately
recognisable as machine prose and the author will reject it on sight.

Instead: sentences that connect, qualify and carry their reasoning in-line;
varied length; subordinate clauses where the logic has them; the occasional
long sentence that earns its length. Hedge where the evidence is thin, in the
sentence, not in a separate caveat. Read `style/voice.md` for how the author
actually does this.

Other constructions to avoid:

- The "X, and the Y" section title that stacks two noun phrases
  (*The planner as yardstick, and the solvability gate*). Titles name what the
  section is about.
- Colon-then-reveal, "not X but Y" framing, em-dash asides used as a tic,
  scare quotes around invented labels, "worth noting", "honest caveat",
  "it turns out", "importantly".
- Three-item parallel lists dressed as rhetoric (*cheap to simulate,
  deterministic, fully specified*). Lists are for content, not cadence.
- Anthropomorphism of the field or the agent. Say *representation space*,
  not *mental model*. The agent *is told*, *observes*, *is rewarded*; it does
  not *think*, *believe* or *want*.

## Words

- *task*, *task instance*, *level* — never *mission*, except for a real
  historical mission used as an example.
- *trivial difficulty* (reaction time, execution margin) versus *nontrivial
  difficulty* (representational, logical, conceptual). The nontrivial kind is
  what the paper is about; the trivial kind gets a paragraph.
- Every symbol and every acronym is defined at first use, in words, before it
  is used in a formula. PPO, GAE, CEM, IQM, CI are all defined or avoided.
- Units everywhere a number appears. "150 s", not "150".

## Claims and citations

Dense citation is the goal, into complex systems, complexity theory,
dynamical systems, reinforcement learning and curriculum learning. Where the
citation is not yet in hand, write the claim the way a human author would and
mark it `[CITE: what the citation needs to establish]`. The citation pass
resolves these; do not invent references, and do not soften a claim just
because its citation is missing.

Every number comes from a file in this repository or from a cited source, and
carries enough context to be checked: a sample size, a script, a seed count.

References are numbered, `\cite` with a numeric style.

## Figures and tables

Visuals and tables carry the intuition; prose connects them. Every figure is
looked at, as an image, before it is referenced in text, and again after any
change. Legends do not overlap data. Lines do not cross boxes they were not
meant to cross. A figure that needs a paragraph of caption to be read is a
figure that needs redrawing.

Families, quantities and symbols appear in a figure only after the text has
introduced them.

## Process

Draft with placeholders rather than guesses. A paragraph that needs a figure
that does not yet exist says `[FIG: what it shows]`. A claim that needs a
measurement we have not run says `[MEASURE: what and how]`. The author would
rather see the gap than a confident filler.
