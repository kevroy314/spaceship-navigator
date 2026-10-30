# Antipatterns: what marks prose as machine-written

A checklist for the drafter and the auditor. It extends `paper/STYLE.md`,
which wins on any conflict. Section 1 is the catalogue, section 2 the
vocabulary blocklist, section 3 the author's own complaints verbatim, section
4 an audit of the rejected first draft (`paper/spacenav.tex`, 961 lines, at
commit `1b7b238`) showing what the catalogue catches.

Two caveats before the list, both from the sources.

- **Vocabulary tells drift; constructions last.** "Delve" rose 1,375% in
  PubMed abstracts 2020-2024 ([Juzek & Ward 2024, Table 2](https://arxiv.org/abs/2412.11385))
  and then fell to 1 in 1,000 ChatGPT messages by July 2025 once it became a
  known tell ([Washington Post corpus of 328,744 messages, via Boston Globe](https://www.bostonglobe.com/2025/11/13/business/chatgpt-writing-style-clues/)).
  The sentence-shape and rhetoric entries below are the durable ones; the word
  list is a snapshot.
- **No single hit is proof.** Wikipedia's guide says its signs are "only
  potential signs of a problem, not the problem itself"
  ([Wikipedia:Signs of AI writing](https://en.wikipedia.org/wiki/Wikipedia:Signs_of_AI_writing)).
  Pangram's multipliers are for the pattern's *rate*, and humans produce every
  one of them. What the author rejected was density: a draft in which most
  paragraphs trip several entries at once. The audit in section 4 should be
  read that way.

Sources cited by short name below:

| Short name | Source |
|---|---|
| Wikipedia | [Wikipedia:Signs of AI writing](https://en.wikipedia.org/wiki/Wikipedia:Signs_of_AI_writing), WikiProject AI Cleanup field guide |
| Pangram | [9 Signs of AI Writing, Backed by Data](https://www.pangram.com/signs-of-ai-writing), rates per 10k words, human vs AI, "millions" of documents |
| WaPo | [Washington Post analysis of 328,744 ChatGPT messages, May 2024-Jul 2025](https://www.bostonglobe.com/2025/11/13/business/chatgpt-writing-style-clues/) (Boston Globe republication) |
| Kobak | [Kobak, González-Márquez, Horvát, Lause, *Science Advances* 2025; arXiv:2406.07016](https://arxiv.org/abs/2406.07016), 15M PubMed abstracts |
| Liang | [Liang et al., COLM 2024; arXiv:2404.01268](https://arxiv.org/abs/2404.01268), 950,965 papers |
| Juzek | [Juzek & Ward, COLING 2025; arXiv:2412.11385](https://arxiv.org/abs/2412.11385), 21 focal words |
| Kousha | [Kousha & Thelwall, arXiv:2509.09596](https://arxiv.org/abs/2509.09596), six databases plus 2.4M PMC full texts |
| tessl | [jbaruch blog-writer "AI anti-patterns" reference](https://tessl.io/registry/jbaruch/blog-writer/1.1.0/files/references/ai-anti-patterns.md), 37 numbered patterns with before/after examples; practitioner catalogue, not measured |
| aismells | [aismells.pages.dev AI writing tells](https://aismells.pages.dev/ai-writing-tells.html), unattributed practitioner list |
| Sapling | [ChatGPT's favorite phrases](https://sapling.ai/devblog/chatgpt-phrases/), n-gram log-odds on HC3 (24K QA pairs) |
| Firth | [Thesis Whisperer, "De-stuffing your writing"](https://thesiswhisperer.com/2018/02/14/de-stuffing-your-writing-or-the-the-bumper-list-of-words-and-phrases-you-could-delete-to-make-your-writing-more-concise/), Katherine Firth 2018; pre-LLM editing advice |
| STYLE | `paper/STYLE.md` in this repository |

Reliability, in the order I would trust them: Kobak, Liang, Juzek and Kousha
are peer-reviewed or preprinted measurements on millions of documents, but
they measure *words*, not constructions. WaPo and Pangram are corpus counts of
constructions (em dashes, "not just X but Y", triads) with stated rates; Pangram
is a detection vendor and does not state n per sign. Wikipedia is an edited
consensus of many editors' observations with real examples, not measured. tessl
and aismells are one person's catalogue each, unmeasured, but they are the only
sources that name the sentence-shape patterns (fragment chains, colon reveals,
labelled devices) that the measured studies do not reach. Firth is ordinary
pre-LLM editing advice, included because many "AI tells" are simply old filler
that models learned from us.

---

## 1. Checklist

Each entry: the pattern, why it reads as machine prose, an example (from the
draft where one exists, marked *draft L<n>*), and the fix.

### 1a. Sentence shape

**S1. Runs of stunted "We <verb> a/the <noun>" declaratives.**
Why: every sentence has the same subject, length and shape; nothing connects to
the sentence before it. Low sentence-length variance ("burstiness") is the
most-cited stylometric tell ([tessl #14](https://tessl.io/registry/jbaruch/blog-writer/1.1.0/files/references/ai-anti-patterns.md); Wikipedia's "Syntax" sign lists plain, varied human syntax as the counter-signal). STYLE names this as the rejected register.
Example (*draft L43-59*): "We give a closed-form, simulation-free score ... We
report this and two other self-corrections ... We release the environment, the
levels, the trained policies and the analysis code."
Fix: make the reasoning the subject. "Because the score needs no simulation,
millions of candidate tasks can be scored before any is flown, and because it
inverts, a task can be placed at a chosen difficulty by solving for its fuel."

**S2. Fragment chains and one-line reveals.**
Why: a two- or three-word sentence placed after a build-up ("The opposite
happened." "It does.") is rhythm standing in for content; tessl bans "three or
more fragments in a row" and the single-fragment kicker ([tessl #3, #4](https://tessl.io/registry/jbaruch/blog-writer/1.1.0/files/references/ai-anti-patterns.md)).
Example (*draft L112*): "...would learn faster. The opposite happened."
(*draft L701*): "...on the redundant half. It does."
Fix: fold the reveal into the sentence that carries the number. "Across five
seeds per condition the four channels cost 3.1 to 10.1 percentage points of
final success, the reverse of the expectation that motivated them."

**S3. Colon-then-reveal.**
Why: the colon promises a punchline; the text before it is a label for what
follows rather than a claim. STYLE bans it; tessl calls the same move
"labeling the device" and "telegraphing" ([tessl #28, #30](https://tessl.io/registry/jbaruch/blog-writer/1.1.0/files/references/ai-anti-patterns.md)).
Example (*draft L100*): "The result was wrong: the three budgets had been run
against the same twelve missions."
(*draft L633*): "These are intended as a cockpit: the quantities a
well-instrumented pilot would read."
Fix: state the fact and let the reader draw the conclusion, or subordinate.
"Because all three budgets were run against the same twelve tasks, they gave
one chance to observe a success, not three."

**S4. Em-dash pair as an aside tic.**
Why: AI text uses em dashes about 10x the human rate ([Pangram](https://www.pangram.com/signs-of-ai-writing): 2 vs 17 per 10k words); over half of ChatGPT replies contained one by summer 2025 ([WaPo](https://www.bostonglobe.com/2025/11/13/business/chatgpt-writing-style-clues/)); Wikipedia lists it as a historical indicator 2022-2026. Spaced em dashes (` --- ` in LaTeX source) are the specific form Wikipedia names. STYLE bans the tic.
Example (*draft L109-111*): "four physically meaningful readouts --- $\eta$, the
fuel-versus-gravity ratio, their quotient $\dd$, and a local tidal stretching
rate --- on the hypothesis that..."
Fix: parentheses for a true aside, commas for a short one, a separate sentence
for a long one. Keep at most one dash per paragraph and never a matched pair
around a list.

**S5. Symmetrical antithesis ("feels every body but moves none").**
Why: mirrored A/B clauses with the same skeleton are a cadence device, the
"fortune-cookie" sentence ([tessl #2, #5](https://tessl.io/registry/jbaruch/blog-writer/1.1.0/files/references/ai-anti-patterns.md)). One is fine; a draft full of them reads as generated.
Example (*draft L203*): "The ship is a test particle: it feels every body but
moves none." (*draft L291-292*): "Near zero, patched conics is right; near one,
it is lying."
Fix: one plain statement. "The ship is a test particle, so it exerts no force
on the bodies and their trajectories are fixed by the initial snapshot."

**S6. Trailing participle tail ("..., highlighting the importance of").**
Why: a dangling "-ing" clause that restates the sentence as analysis. Wikipedia
"Superficial analyses" (watch words: highlighting, underscoring, emphasizing,
ensuring, reflecting, contributing to, fostering, enhancing); [tessl #19](https://tessl.io/registry/jbaruch/blog-writer/1.1.0/files/references/ai-anti-patterns.md).
Example (Wikipedia's): "further enhancing its significance as a dynamic hub of
activity and culture".
Fix: delete the tail, or give the implication its own sentence with evidence.

**S7. Copula avoidance ("serves as", "boasts", "stands as", "functions as").**
Why: Wikipedia lists "Avoidance of basic copulatives" as a language sign;
"boasts" rose 918% in PubMed abstracts ([Juzek Table 2](https://arxiv.org/abs/2412.11385)).
Fix: "is", "has".

**S8. Self-answering question ("The result? A cleaner architecture.").**
Why: a question that exists only to set up its own answer ([tessl #6](https://tessl.io/registry/jbaruch/blog-writer/1.1.0/files/references/ai-anti-patterns.md); [aismells](https://aismells.pages.dev/ai-writing-tells.html) "self-posed questions"). The draft's title is a question (*L21*) and the introduction poses "is this particular mission hard?" twice (*L39, L67*).
Fix: state the claim. A question is allowed only when the paper does not
answer it.

**S9. Low burstiness across a paragraph.**
Why: three or more consecutive sentences within about five words of each other
in length ([tessl #14](https://tessl.io/registry/jbaruch/blog-writer/1.1.0/files/references/ai-anti-patterns.md)). The human counter-signal is "short punchy sentences followed by longer flowing ones".
Example (*draft L391-393*): "$\dd$ measures whether a coarse model suffices. It
says nothing about whether a correct plan can be executed. We therefore band
every mission on two axes."
Fix: one subordinated sentence. "Because $\dd$ says only whether a coarse model
suffices, and nothing about whether the resulting plan can be executed, every
task is banded on two axes rather than one."

### 1b. Vocabulary

**V1. Excess-vocabulary words.**
Why: measured. "delves" +6,697%, "showcasing" +1,396%, "underscores" +904%,
"intricate" +611%, "realm" +381%, "groundbreaking" +330% in PubMed abstracts
2020-2024 ([Juzek Table 2](https://arxiv.org/abs/2412.11385)); "delve" +1,500%,
"underscore" +1,000%, "intricate" +700% across six databases 2022-2024
([Kousha](https://arxiv.org/abs/2509.09596)); "realm, intricate, showcasing,
pivotal" named for arXiv CS ([Liang Fig. 2](https://arxiv.org/abs/2404.01268));
Kobak's 2024 excess vocabulary "consisted almost entirely of style words".
Fix: section 2 blocklist; grep before submitting.

**V2. Empty intensifiers ("genuinely", "entirely", "truly", "enormous",
"remarkably", "deliberately", "exactly").**
Why: the adverb asserts what the evidence should show ([tessl #33](https://tessl.io/registry/jbaruch/blog-writer/1.1.0/files/references/ai-anti-patterns.md); [Firth](https://thesiswhisperer.com/2018/02/14/de-stuffing-your-writing-or-the-the-bumper-list-of-words-and-phrases-you-could-delete-to-make-your-writing-more-concise/): "adjectives or adverbs that describe information instead of providing evidence").
Example (*draft L57*): "the two that carry genuinely new physics are free";
(*draft L115*): "the cost is \emph{entirely} carried by".
Fix: delete the adverb; the number carries it. "The two channels not
computable from the rest of the observation cost nothing measurable."

**V3. Unproven assessment adjectives ("honest", "natural", "robust",
"elegant", "comprehensive", "blunt").**
Why: a verdict without the fact that earns it ([tessl #34](https://tessl.io/registry/jbaruch/blog-writer/1.1.0/files/references/ai-anti-patterns.md)). "Honest" in particular is a self-award.
Example (*draft L98*): "The honest way to ask whether a mission class is
feasible is to let a strong optimiser attempt it."
Fix: "Whether a task class is feasible is settled by whether a strong
optimiser can fly it."

**V4. Significance inflation ("the single most consequential design decision",
"pivotal", "fundamental").**
Why: Wikipedia "Undue emphasis on significance" (watch: pivotal, vital, key
role, underscores its importance, setting the stage); [tessl #20](https://tessl.io/registry/jbaruch/blog-writer/1.1.0/files/references/ai-anti-patterns.md).
Example (*draft L204*): "This is the single most consequential design decision
in the environment."
Fix: state the consequence and let its size speak. "Because the ship exerts no
force, the bodies' trajectories are fixed by the snapshot, which is what lets
a 1.5 KB snapshot reproduce the field in a browser."

**V5. Elegant variation / synonym cycling.**
Why: models avoid repeating a noun and cycle synonyms, which in a technical
paper changes the referent ([Wikipedia](https://en.wikipedia.org/wiki/Wikipedia:Signs_of_AI_writing) historical indicator "Lexical diversity"; [tessl #17](https://tessl.io/registry/jbaruch/blog-writer/1.1.0/files/references/ai-anti-patterns.md)). The draft uses mission / task / level / tour for overlapping things.
Fix: one term per concept, defined once. STYLE fixes *task*, *task instance*,
*level*.

**V6. Scare quotes around invented labels.**
Why: STYLE bans them; the move invents a term and distances from it in the
same breath. Source: `paper/STYLE.md` ("scare quotes around invented
labels"). No external measurement found; Wikipedia's nearest sign is
quotation-mark typography ([Wikipedia](https://en.wikipedia.org/wiki/Wikipedia:Signs_of_AI_writing), "Curly quotation marks"), which is not the same thing.
Example (*draft L84-85*): "``only the body I am orbiting matters''"; (*draft
L431*): "a ``lottery'' difficulty band".
Fix: name the concept in prose or define it as a term in italics once.

**V7. Anthropomorphism of the field, the model or the agent.**
Why: STYLE: "Say *representation space*, not *mental model*. The agent *is
told*, *observes*, *is rewarded*; it does not *think*, *believe* or *want*."
Source: `paper/STYLE.md`. No external corpus measure; the nearest external
sign is Wikipedia's "Undue emphasis on significance" ([Wikipedia](https://en.wikipedia.org/wiki/Wikipedia:Signs_of_AI_writing)), which is about inflating a subject, not animating it.
Example (*draft L292*): "near one, it is lying"; (*draft L458*): "The
uncertainty exponent told the same story the hard way"; (*draft L741*): "an
agent confidently executing a plan".
Fix: "near one, the one-body model discards most of the field"; "the
uncertainty exponent gives the same answer"; "a policy that executes a plan
which cannot complete the task".

### 1c. Rhetoric

**R1. Negative parallelism: "not X but Y", "X, not Y", "Y rather than X".**
Why: Wikipedia names all three forms and calls it the single most-flagged tell;
3x the human rate ([Pangram](https://www.pangram.com/signs-of-ai-writing)); in
6% of ChatGPT chats in July 2025 ([WaPo](https://www.bostonglobe.com/2025/11/13/business/chatgpt-writing-style-clues/)); [tessl #1](https://tessl.io/registry/jbaruch/blog-writer/1.1.0/files/references/ai-anti-patterns.md). STYLE bans it. The negated half is usually a claim nobody made.
Example (*draft L444*): "the divergence is \emph{ballistic, not exponential}";
(*draft L851-852*): "not on whether the information is \emph{true} but on
whether it is \emph{new}"; (*draft L512-513*): "an unremarkable outcome, not
evidence of impossibility".
Fix: assert Y. "The divergence is ballistic: separation grows linearly with
time, and the fitted exponential rate is a factor of two below the threshold."

**R2. Rule of three.**
Why: triads at 4x the human rate ([Pangram](https://www.pangram.com/signs-of-ai-writing)); Wikipedia: "groups items in threes to make superficial analysis look comprehensive"; the "asyndetic tricolon with a kicker" ([tessl #3](https://tessl.io/registry/jbaruch/blog-writer/1.1.0/files/references/ai-anti-patterns.md)). STYLE: "Lists are for content, not cadence."
Example (*draft L847*): "cheap to simulate, deterministic, fully specified";
(*draft L123*): "elegant, confident and wrong conclusions"; (*draft L65-66*):
three semicoloned clauses.
Fix: keep the one property the argument needs, or give the real count.

**R3. Three-part skeleton announced up front ("We make three claims."
"First... Second... Third...").**
Why: the preamble announcement ([tessl #9](https://tessl.io/registry/jbaruch/blog-writer/1.1.0/files/references/ai-anti-patterns.md)) combined with the triad; the structure is visible before the content.
Example (*draft L39-52*): "resists a single answer in three distinct ways.
First, ... Second, ... Third, ..."; (*draft L81*): "We make three claims.";
(*draft L469*): "Three loose ends resolve at once."
Fix: let the claims follow in paragraphs whose first sentences are the claims.

**R4. Aphoristic paragraph closers.**
Why: a short, quotable sentence that restates what the paragraph already
proved ([tessl #29](https://tessl.io/registry/jbaruch/blog-writer/1.1.0/files/references/ai-anti-patterns.md) "restating the proven point"; #31 "incomplete assertions").
Example (*draft L522*): "More time buys more exposure."; (*draft L852*): "None
of those three is specific to gravity."; (*draft L105*): "a benchmark whose
difficulty estimates are this fragile is a fact about the benchmark."
Fix: end on the number or the mechanism, not on the moral.

**R5. Telegraphing ("matters more than it sounds", "the way it was wrong is
instructive", "Two observations deserve emphasis").**
Why: announcing that something is important instead of showing it ([tessl #30](https://tessl.io/registry/jbaruch/blog-writer/1.1.0/files/references/ai-anti-patterns.md); [aismells](https://aismells.pages.dev/ai-writing-tells.html) "padded transitions").
Example (*draft L490*): "Including the zero-thrust candidate matters more than
it sounds."; (*draft L506*): "That conclusion was wrong, and the way it was
wrong is instructive."; (*draft L615*): "Two observations deserve emphasis."
Fix: delete the sentence; start with the content.

**R6. Labelling the device ("The resolution is that", "One practical
caution:", "The honest way", "The concern with sharpest teeth was").**
Why: naming the rhetorical move before making it ([tessl #28](https://tessl.io/registry/jbaruch/blog-writer/1.1.0/files/references/ai-anti-patterns.md)).
Example (*draft L444*): "The resolution is that the divergence is..."; (*draft
L322*): "One practical caution: when retuning..."
Fix: make the move.

**R7. Our-story / retrospective narration.**
Why: STYLE, "Whose story": "no 'our own earlier conclusion', no 'we previously
reported', no 'turned out to be an artifact', no 'we got it wrong in public'."
Also the "fabricated experience" tell ([tessl #15](https://tessl.io/registry/jbaruch/blog-writer/1.1.0/files/references/ai-anti-patterns.md)): machine drafts manufacture a narrator with a journey. Nothing here has been published, so there is nothing to retract.
Example (*draft L46-48*): "Our own published conclusion ... turned out to be an
artifact"; (*draft L499*): "We previously reported that"; (*draft L472-473*):
"a previously published claim of ours ... is retracted".
Fix: state the natural expectation, then the result. "Twelve tasks under
three budgets with zero arrivals looks like infeasibility; at thirty-two tasks
the class is flown 5/32 of the time, and zero of twelve at that rate has
probability 0.13."

**R8. Second-person "you" and assistant tone.**
Why: "What you tell the agent", "it cannot tell you", "if you ask" are the
chatbot's address to a user ([aismells](https://aismells.pages.dev/ai-writing-tells.html) "assistant tone"; Wikipedia "Collaborative communication"). A paper addresses no one.
Example (*draft L107*): "What you tell the agent changes what it can learn";
(*draft L480*): "it cannot tell you whether a mission is flyable at all".
Fix: "The observation changes what the agent can learn"; "it does not say
whether a task is flyable".

**R9. Vague attribution ("the literature's obvious candidate", "the successful
chaotic-control results in the literature").**
Why: Wikipedia "Vague attributions and overgeneralization" (watch: experts
argue, observers have cited, industry reports); [tessl #25](https://tessl.io/registry/jbaruch/blog-writer/1.1.0/files/references/ai-anti-patterns.md). STYLE requires `[CITE: ...]` instead.
Example (*draft L454-455*): "the successful chaotic-control results in the
literature sit at 10-100 steps per e-folding".
Fix: cite or mark `[CITE: chaotic-control results at 10-100 steps per
e-folding]`.

### 1d. Structure

**D1. Uniform paragraphs.**
Why: every paragraph the same length, opening on a claim, closing on a moral
([aismells](https://aismells.pages.dev/ai-writing-tells.html) "uniform paragraphs"). The draft's introduction is three `\paragraph` blocks of near-identical shape (*L83-119*).
Fix: let paragraph length follow the argument; merge the two that belong
together.

**D2. Section summaries and closing restatement.**
Why: Wikipedia "Section summaries" ("In summary", "In conclusion",
"Overall"); [tessl #29](https://tessl.io/registry/jbaruch/blog-writer/1.1.0/files/references/ai-anti-patterns.md).
Example (*draft L846-852*): the final paragraph restates the three claims of
the abstract.
Fix: delete; the abstract already does this job.

**D3. Challenge-optimism sandwich and outline-like limitations.**
Why: Wikipedia "Outline-like conclusions about challenges and future
prospects"; [tessl #21](https://tessl.io/registry/jbaruch/blog-writer/1.1.0/files/references/ai-anti-patterns.md). The form "Despite X, challenges remain; nevertheless ..." or a numbered First/Second/Third of limitations.
Example (*draft L831-844*): "The most substantive limitation is ... Second, ...
Third, ..."
Fix: a limitations section is fine; it should read as prose about what the
results do not show, each item tied to the measurement that bounds it.

**D4. Inline-header lists ("**Model scale** --- the coarsest k ...").**
Why: bold term, dash or colon, definition; a glossary dressed as prose
(Wikipedia "Inline-header vertical lists"; [tessl #22](https://tessl.io/registry/jbaruch/blog-writer/1.1.0/files/references/ai-anti-patterns.md); Pangram: bullet lists 9x).
Example (*draft L395-404*).
Fix: two sentences of prose, or a table if the content is tabular.

**D5. Caption and body text duplicated verbatim.**
Why: the same sentence appears in a caption and in the body, which a human
drafter would notice. Found in the audit; no external source. The nearest is
the restated-point tell ([tessl #29](https://tessl.io/registry/jbaruch/blog-writer/1.1.0/files/references/ai-anti-patterns.md)). STYLE: "A figure
that needs a paragraph of caption to be read is a figure that needs
redrawing."
Example (*draft L412-414* and *L421-426*): "different kinds of impossible and
need opposite repairs ... merge exactly the two cases a designer most needs to
tell apart" appears twice.
Fix: the caption says what is plotted; the body says what it means.

**D6. Stacked data points for one effect.**
Why: several statistics restating one improvement ([tessl #32](https://tessl.io/registry/jbaruch/blog-writer/1.1.0/files/references/ai-anti-patterns.md)).
Example (*draft L643-646* and *L701-706*): the instrument result is given as a
point range, a p range, an effect-size range, then again as a point range, a
p range, an effect-size range.
Fix: table once, prose once.

### 1e. Hedging and emphasis

**H1. Hedging clichés ("We note that", "We note explicitly", "We emphasise
that", "worth noting", "importantly", "honest caveat", "a caveat we attach").**
Why: STYLE bans "worth noting", "honest caveat", "it turns out",
"importantly". The forms "it is important to note/know/remember" are among the
top ChatGPT n-grams ([Sapling](https://sapling.ai/devblog/chatgpt-phrases/)); "sycophantic hedging" ([tessl #10](https://tessl.io/registry/jbaruch/blog-writer/1.1.0/files/references/ai-anti-patterns.md)). STYLE: "Hedge where the evidence is thin, in the sentence, not in a separate caveat."
Example (*draft L336-337*): "We note explicitly that the per-rung figures, at
$n = 8$, are not significant"; (*draft L647*): "We emphasise that these are
tests on seed means"; (*draft L619-620*): "a caveat we attach to every such
comparison".
Fix: "At $n=8$ per rung no single rung is significant; the pooled comparison
is."

**H2. Italic emphasis as a tic (`\emph{costs}`, `\emph{entirely}`,
`\emph{not}`, `\emph{inverts}`, `\emph{why}`).**
Why: typographic emphasis doing the work the sentence should (Wikipedia
"Overuse of boldface"; [tessl #26](https://tessl.io/registry/jbaruch/blog-writer/1.1.0/files/references/ai-anti-patterns.md)). The draft has 78 `\emph{}`, most of them outside term definitions.
Example (*draft L54*): "\emph{costs} it roughly ten percentage points";
(*draft L711*): "We do \emph{not} have evidence".
Fix: reserve `\emph` for a term at its definition. If a word needs stress,
restructure so it lands at the end of the clause.

**H3. Hedge-then-assert ("we think", "arguably", "in our view" wrapped around
a strong claim).**
Why: a hedge that does not change the claim's strength is decoration. Firth
distinguishes real hedges ("may", "it is likely that"), which are legitimate,
from decorative ones ([Firth](https://thesiswhisperer.com/2018/02/14/de-stuffing-your-writing-or-the-the-bumper-list-of-words-and-phrases-you-could-delete-to-make-your-writing-more-concise/)).
Example (*draft L68-69*): "are, we think, more interesting than the
environment itself"; (*draft L846*): "What we think transfers beyond".
Fix: either make the claim or make the hedge real: "more interesting than the
environment itself" or "may transfer".

**H4. Filler adverbs ("essentially", "simply", "at all", "exactly the kind
of", "comfortably", "uncomfortably close").**
Why: fill the space without changing meaning ([tessl #10](https://tessl.io/registry/jbaruch/blog-writer/1.1.0/files/references/ai-anti-patterns.md) lists "basically", "essentially"; [Firth](https://thesiswhisperer.com/2018/02/14/de-stuffing-your-writing-or-the-the-bumper-list-of-words-and-phrases-you-could-delete-to-make-your-writing-more-concise/)).
Example (*draft L313*): "essentially for free"; (*draft L455-456*): "We are
comfortably inside the benign regime."; (*draft L545*): "exactly the kind of
structural detail".
Fix: delete.

**H5. "And"/"But"/"And yet" sentence openers used for drama.**
Why: the dramatic conjunction opener is a cadence device; STYLE wants
"sentences that connect, qualify and carry their reasoning in-line", which a
subordinating conjunction does and a dramatic "And yet" does not. Source:
`paper/STYLE.md` ("Sentences"). Practitioner observation; not measured. Related:
the "telegraphing transitions" tell ([tessl #30](https://tessl.io/registry/jbaruch/blog-writer/1.1.0/files/references/ai-anti-patterns.md)).
Example (*draft L66*): "And yet if you ask..."; (*draft L93*): "But $\dd$ says
nothing about timing"; (*draft L714*): "And we have not isolated the
mechanism."
Fix: subordinate. "Although $\dd$ says nothing about timing, ..."

### 1f. Titles and headings

**T1. Stacked noun phrases joined by ", and the" ("The planner as yardstick,
and the solvability gate").**
Why: STYLE: "Titles name what the section is about." The form stacks two
topics with a rhetorical comma. Source: `paper/STYLE.md`. No external
measurement; Wikipedia's heading signs ([Wikipedia](https://en.wikipedia.org/wiki/Wikipedia:Signs_of_AI_writing), "Heading errors", "Title case") concern format, not this shape.
Example (*draft L478*): exactly the STYLE example; (*L294*): "Authority, and
the ratio that matters"; (*L388*): "The second axis, and why it must stay
separate"; (*L144*): "Chaos, and whether it is present"; (*L496*): "A
retraction, and what it says about the benchmark".
Fix: "Solvability"; "Authority"; "Keeping the axes separate"; "Chaos on episode
timescales"; "Sample size and the two-stop class".

**T2. Claim-sentence `\paragraph` headings with a comma clause.**
Why: a heading that is a full sentence with a dependent clause is a tweet, not
a heading (same device as T1 applied to `\paragraph`). Source: `paper/STYLE.md`
("Titles name what the section is about"); for the sentence-as-heading form see
[tessl #28, #30](https://tessl.io/registry/jbaruch/blog-writer/1.1.0/files/references/ai-anti-patterns.md).
Example (*draft L83*): "Difficulty needs at least two axes, and one of them has
a closed form."; (*L97*): "Measuring difficulty is itself hard, and we got it
wrong in public."; (*L107-108*): "What you tell the agent changes what it can
learn, in a direction that is easy to get backwards."
Fix: a noun phrase. "Two axes of difficulty"; "Measurement"; "Observation
design".

**T3. Colon titles ("Where to put a target: the equal-pull surface").**
Why: colon-reveal (S3) applied to a heading. Source: `paper/STYLE.md` bans the
construction generally; [tessl #28](https://tessl.io/registry/jbaruch/blog-writer/1.1.0/files/references/ai-anti-patterns.md) ("labeling the device").
Example (*draft L341*).
Fix: "The equal-pull surface".

**T4. Question titles and teaser headings ("What Makes a Mission Hard?",
"What a single number hides").**
Why: the self-posed question (S8) at the level of a title; the teaser heading
withholds the topic for effect. Practitioner observation ([aismells](https://aismells.pages.dev/ai-writing-tells.html) "self-posed questions"); STYLE's "Titles name what the section is about".
Example (*draft L21*, *L719*, *L623* "What the agent should be told").
Fix: "Difficulty in a gravitational navigation environment"; "Per-task
variation under sampled actions"; "Observation channels".

**T5. Gerund triad subtitles ("Defining, Measuring, and Learning Difficulty").**
Why: rule of three (R2) in a title, with the Oxford-comma cadence (triads at
4x the human rate, [Pangram](https://www.pangram.com/signs-of-ai-writing); `paper/STYLE.md`).
Example (*draft L22-23*).
Fix: drop the subtitle or name the one thing: "A two-axis measure of task
difficulty and its effect on learning".

**T6. Title Case in section headings.**
Why: Wikipedia "Title case" sign; [tessl #27](https://tessl.io/registry/jbaruch/blog-writer/1.1.0/files/references/ai-anti-patterns.md). The draft's `\section` headings are already sentence case, which is right; the `\title` is title case, which is conventional for a paper title and not a tell. Listed so the auditor does not flag it.
Fix: none needed in the draft.

---

## 2. Vocabulary blocklist

Grep the source for these before every pass. Three in one paragraph is a
"contamination event" ([tessl #12](https://tessl.io/registry/jbaruch/blog-writer/1.1.0/files/references/ai-anti-patterns.md)).

### Measured excess words (scientific abstracts, 2020-2024)

From [Juzek & Ward, Table 2](https://arxiv.org/abs/2412.11385), 21 focal words with
percentage rise in PubMed abstracts 2020-2024: delves (+6,697%), delved
(+2,240%), delving (+1,817%), showcasing (+1,396%), delve (+1,375%), boasts
(+918%), underscores (+904%), comprehending (+899%), intricacies (+773%),
surpassing (+667%), intricate (+611%), underscoring (+537%), garnered (+437%),
showcases (+422%), emphasizing (+397%), underscore (+391%), realm (+381%),
surpasses (+368%), groundbreaking (+330%), advancements (+278%), aligns (+267%).

From [Kobak et al.](https://arxiv.org/abs/2406.07016): rare words with strong excess,
delves, underscores, showcasing; common words with strong excess, potential,
findings, crucial; the ten-word "common" set used for the lower-bound estimate:
across, additionally, comprehensive, crucial, enhancing, exhibited, insights,
notably, particularly, within. Note that "findings", "across", "within" and
"particularly" cannot be banned from a paper; they are flagged as *excess*, so
use them only where the plain word is right.

From [Liang et al., Fig. 2](https://arxiv.org/abs/2404.01268) (arXiv CS abstracts):
realm, intricate, showcasing, pivotal.

From [Kousha & Thelwall](https://arxiv.org/abs/2509.09596) (PMC full text 2022-2025,
papers using the term six or more times): underscore (+10,000%), intricate
(+5,400%), meticulous (+2,800%); in 2024 "underscore" co-occurs with "pivotal"
(r = 0.449) and "delve" (r = 0.311).

### Wikipedia watch lists (by sign)

From [Wikipedia:Signs of AI writing](https://en.wikipedia.org/wiki/Wikipedia:Signs_of_AI_writing):

- *AI vocabulary*: additionally, crucial, delve, enduring, intricate, pivotal,
  showcase, tapestry, testament, underscore, vibrant.
- *Undue significance*: stands as, serves as, is a testament, is a reminder,
  pivotal, vital, significant, key role, key moment, underscores its
  importance, highlights its importance, setting the stage for, evolving
  landscape, indelible mark, deeply rooted.
- *Superficial analysis tails*: highlighting, underscoring, emphasizing,
  ensuring, reflecting, symbolizing, contributing to, fostering, enhancing,
  valuable insights, align with, resonate with.
- *Promotional*: boasts a, vibrant, rich, profound, showcasing, exemplifies,
  commitment to, nestled, in the heart of, groundbreaking, renowned, diverse
  array.
- *Vague connection*: in connection with, associated with, in association with.
- *Vague attribution*: industry reports, observers have cited, experts argue,
  some critics argue, such as (as a non-exhaustive dodge).
- *Copula avoidance*: serves as, stands as, marks, functions as, boasts,
  features, maintains, offers, refers to.
- *Section summaries*: in summary, in conclusion, overall.
- *Hedging disclaimers*: while specific details are limited, based on
  available information, not publicly available, not widely documented.

### Phrases (corpus counts)

From [Pangram](https://www.pangram.com/signs-of-ai-writing): "delve into" (45x
human rate), "navigate the tapestry of" (18x), "in today's fast-paced world"
(35x), "past, present, and future" (triad example), "not only ... but also".

From [Sapling](https://sapling.ai/devblog/chatgpt-phrases/) (HC3 n-grams, top of
list): "there are a couple", "it's also important to think", "is always a good
thing", "it is also important that", "it is important to know", "can vary
depending on what", "I hope this helps", "one way to think of", "to keep in
mind", "best course of action is", "it is important to remember however".

From [WaPo](https://www.bostonglobe.com/2025/11/13/business/chatgpt-writing-style-clues/)
(2025 drift): "core" (5x its prior rate), "modern" (in 8% of messages), "not
just X, but Y" (6% of chats); words the model has *dropped*: ensure, various,
crucial, significant, approach, delve.

### This paper's own banned words (from STYLE)

worth noting, honest caveat, it turns out / turned out, importantly, mental
model, mission (except a real historical one), retract / retraction, "we
previously reported", "our own earlier conclusion", "we got it wrong".

### Added from the audit (no external source; author-adjacent)

genuinely, entirely (as emphasis), honest, natural (as in "the natural
question"), blunt, instructive, deserve emphasis, comfortably, uncomfortably,
essentially, simply, at all, exactly the kind of, dressed up as, littered
with, sharpest teeth, the hard way, razor-thin, a place to stand.

---

## 3. This author's complaints, verbatim from `paper/STYLE.md`

Each quoted from STYLE, with the catalogue entry it corresponds to and what
the audit found.

1. **"We <verb> a/the <noun>" openers.** STYLE: "The rejected register is
   short, stunted, declarative sentences that open with 'We <verb> a/the
   <noun>': *We give a closed-form score. We argue that it must never be
   collapsed. We release the environment.* That cadence is immediately
   recognisable as machine prose and the author will reject it on sight."
   Entry S1. The draft has 24 sentences beginning "We <verb>" (`grep -nE
   '(^|[.:;] )We [a-z]+' paper/spacenav.tex`); the audit lists them
   individually.

2. **Stacked-noun-phrase section titles.** STYLE: "The 'X, and the Y' section
   title that stacks two noun phrases (*The planner as yardstick, and the
   solvability gate*). Titles name what the section is about." Entry T1. The
   draft has five such `\subsection`/`\paragraph` titles plus three
   claim-sentence `\paragraph` titles (T2).

3. **Retrospective / our-story framing.** STYLE: "Tell *the* story, not
   *ours*. The reader wants to understand a task and a way of thinking about
   it. They do not want a retrospective of what we tried, what we got wrong,
   or what we corrected. Nothing in this paper has been published before, so
   there is nothing to 'retract'." And: "no 'our own earlier conclusion', no
   'we previously reported', no 'turned out to be an artifact', no 'we got it
   wrong in public'. If a result corrects a plausible-but-wrong intuition,
   state the intuition as *the natural expectation* and then the result."
   Entry R7. This is the draft's largest single problem: the abstract, the
   second intro claim, a whole subsection (L496-536), part of the chaos
   section (L458-473), two captions, the reproduction section and the
   acknowledgements are all narrated as self-correction.

4. **"Mental model" anthropomorphism.** STYLE: "Anthropomorphism of the field
   or the agent. Say *representation space*, not *mental model*. The agent *is
   told*, *observes*, *is rewarded*; it does not *think*, *believe* or
   *want*." Entry V7. "mental model" appears at L41, L146, L274; "lying" L292;
   "told the same story" L458; "confidently" L741.

5. **Three-item rhetorical lists.** STYLE: "Three-item parallel lists dressed
   as rhetoric (*cheap to simulate, deterministic, fully specified*). Lists
   are for content, not cadence." Entry R2/R3. The STYLE example is at L847
   verbatim. The whole paper is organised as a triad (three claims, three
   self-corrections, three loose ends, three limitations).

Also from STYLE, caught by entries S3, R1, S4, V6, H1: "Colon-then-reveal,
'not X but Y' framing, em-dash asides used as a tic, scare quotes around
invented labels, 'worth noting', 'honest caveat', 'it turns out',
'importantly'."

---

## 4. Audit of `paper/spacenav.tex`

Line numbers from `cat -n paper/spacenav.tex` at commit `1b7b238`. Multi-line
sentences cite the line where the quoted fragment begins. Pattern codes refer
to section 1. Rewrites follow STYLE (*task* for *mission*, the natural
expectation in place of our story) and are suggestions, not drop-ins; several
depend on decisions the coordinator owns (whether the self-correction material
stays at all).

Counts first, so the shape is visible. Measured on the source with grep
(commands in the report that accompanied this file): 24 sentences begin
"We <verb>" (lines 35, 43, 49, 58, 81, 86, 94, 99, 103, 109, 166, 276, 345,
363, 392, 455, 466, 499, 544, 553, 630, 638, 708, 711); 29 spaced em dashes
(` --- `), 11 of them in matched pairs around an aside; 78 `\emph{}`; 13 lines
containing "rather than"; "mental model" at lines 41, 146, 274; "mission" 73
times. The rows below are counted by hand and list each `\emph` passage once
rather than each instance.

### 4a. Title and abstract

| Line | Quoted | Pattern | Suggested rewrite |
|---|---|---|---|
| 21 | "What Makes a Mission Hard?" | T4 question title; STYLE word *mission* | "Difficulty in a gravitational navigation task" |
| 22-23 | "Defining, Measuring, and Learning Difficulty in a Game-Scale Gravitational Navigation Environment" | T5 gerund triad | "A two-axis measure of task difficulty and its effect on learning" |
| 35 | "We present \emph{spacenav}, a simulated navigation environment" | S1 | "Spacenav is a navigation environment in which a thrusting spacecraft visits moving targets inside a many-body gravitational field." |
| 37-39 | "cheap to simulate and easy to state, and yet we find that the ordinary question ``how hard is this mission?'' resists a single answer in three distinct ways." | H5 "and yet"; R3 triad announced; S8 self-posed question; V6 scare quotes | "Although the environment is cheap to simulate, the difficulty of a task in it has no single measure, for reasons that bear on any benchmark." |
| 40-42 | "\emph{First}, difficulty is not one quantity: a mission can be hard because a simple mental model ... or because the window for acting is narrow, and these two failure modes require opposite fixes." | S3 colon; V7 "mental model"; R3 | "A task can be hard because a one-body approximation of the field fails, or because the window for acting is narrow, and the two call for opposite repairs." |
| 43-45 | "We give a closed-form, simulation-free score, $\dd = \eta / A$, that captures the first and can be \emph{inverted} to construct missions at a chosen difficulty, and we argue that it must never be collapsed with the second." | S1; H2 `\emph{inverted}`; "must never" | "The first has a closed form, $\dd = \eta/A$, which needs no simulation and inverts to place a task at a chosen difficulty; the second does not, and collapsing them hides exactly the cases a designer must separate." |
| 46 | "\emph{Second}, difficulty is hard to measure even once defined." | S2 stunted; R3 | Merge into the next sentence. |
| 46-49 | "Our own published conclusion that a class of two-stop missions was infeasible turned out to be an artifact of twelve correlated samples" | R7 our-story; "turned out" banned | "Twelve correlated samples with no arrival look like infeasibility; at thirty-two samples the same class is flown 16% of the time." |
| 49-51 | "We report this and two other self-corrections as findings about the measurement problem rather than footnotes." | S1; R1 "rather than"; R7 | Delete. |
| 52 | "\emph{Third}, difficulty interacts with what the learner is told." | S2; R3 | "The observation itself changes what can be learned." |
| 53-58 | "we show that adding four hand-designed physical readouts ... \emph{costs} it roughly ten percentage points ... the two that carry genuinely new physics are free" | S1; H2 `\emph{costs}`; V2 "genuinely" | "Four physical readouts added to the observation cost roughly ten percentage points of success over five seeds per condition, and the whole cost sits on the two readouts that are algebraically redundant given the rest of the observation." |
| 58-59 | "We release the environment, the levels, the trained policies and the analysis code." | S1 stunted closer | Fold into the author footnote (L26-27 already says it). |

### 4b. Introduction (L62-125)

| Line | Quoted | Pattern | Suggested rewrite |
|---|---|---|---|
| 65-66 | "The physics is Newtonian and deterministic; the simulation runs in milliseconds; a competent undergraduate can write the equations of motion from memory." | R2 semicoloned tricolon | "The physics is Newtonian and deterministic and the simulation runs in milliseconds." |
| 66-69 | "And yet if you ask ``is this particular mission hard?'' the question turns out to have no single well-posed answer, and the ways in which it fails to have one are, we think, more interesting" | H5 "And yet"; R8 "you"; S8; "turns out" banned; H3 "we think" | "Whether a particular task is hard has no single well-posed answer, and the ways in which it fails to have one are the subject of this paper." |
| 71 | "This paper is about that failure, using one concrete environment as the subject." | D2 meta-announcement | Delete (covered by the rewrite above). |
| 72-73 | "\emph{Spacenav} is a two-dimensional, game-scale gravitational navigation task: a ship with a steerable engine..." | S3 colon | "In spacenav, a ship with a steerable engine and a finite tank must..." |
| 77-79 | "designed as a testbed for reinforcement learning, but the difficulties we report are not specific to learning: a search-based trajectory optimiser meets most of them too." | S3 colon; R1 "not X: Y" | "...a testbed for reinforcement learning, though a search-based optimiser meets most of the same difficulties." |
| 81 | "We make three claims." | S1; R3 announced triad | Delete. |
| 83 | "\paragraph{Difficulty needs at least two axes, and one of them has a closed form.}" | T2 claim-sentence heading with comma | "\paragraph{Two axes.}" or no paragraph heading. |
| 84-86 | "The natural mesoscale model ... --- ``only the body I am orbiting matters'', known in astrodynamics as \emph{patched conics} --- is wrong by a measurable amount" | S4 dash pair; V6 scare quotes; V3 "natural" | "Patched conics, the approximation in which only the dominant body's pull is kept, is wrong by a measurable amount at every point in space." |
| 86 | "We define $\eta$ as the fraction" | S1 | "Let $\eta$ be the fraction..." |
| 90-92 | "It requires no simulation, so millions of candidate missions can be scored for free, and it inverts: given a target difficulty one can solve for the fuel" | S3 colon | "Because it needs no simulation, millions of candidate tasks can be scored before any is flown, and because it inverts, the fuel that places a task at a chosen difficulty can be solved for directly." |
| 93-95 | "But $\dd$ says nothing about \emph{timing}, and a mission can be trivial to plan and impossible to execute. We keep the two separate throughout" | H5 "But"; H2; S5 antithesis; S1 | "$\dd$ says nothing about timing, however, and a task whose plan is trivial may still be impossible to execute; the paper keeps the two axes separate and shows what collapsing them hides." |
| 97 | "\paragraph{Measuring difficulty is itself hard, and we got it wrong in public.}" | T2; R7 ("got it wrong in public" banned verbatim) | "\paragraph{Measurement.}" |
| 98 | "The honest way to ask whether a mission class is feasible is to let a strong optimiser attempt it." | V3 "honest"; V4 | "Whether a task class is feasible is settled by whether a strong optimiser can fly it." |
| 99-100 | "We did that, with three search budgets, and reported that a class of two-stop missions never succeeded. The result was wrong: the three budgets" | R7; S3 colon | "Twelve two-stop tasks under three search budgets produced no arrival, which reads as infeasibility; but the three budgets were run on the same twelve tasks, so they gave one chance to observe a success, not three." |
| 103-105 | "We describe this and two further self-corrections in Section~\ref{sec:measuring}, because a benchmark whose difficulty estimates are this fragile is a fact about the benchmark." | S1; R7; R4 aphorism | "Section~\ref{sec:measuring} treats the fragility of such estimates as a property of the benchmark." |
| 107-108 | "\paragraph{What you tell the agent changes what it can learn, in a direction that is easy to get backwards.}" | T2; R8 "you" | "\paragraph{Observation design.}" |
| 109-111 | "We instrumented the environment with four physically meaningful readouts --- $\eta$, the fuel-versus-gravity ratio, their quotient $\dd$, and a local tidal stretching rate --- on the hypothesis" | S1; S4 dash pair around a list | "Four readouts ($\eta$, the authority $A$, their ratio $\dd$ and a local tidal stretching rate) were added to the observation on the expectation that an agent told..." |
| 112 | "The opposite happened." | S2 one-line reveal | Fold into the next sentence. |
| 115-118 | "the cost is \emph{entirely} carried by the two channels ... the two that carry information not otherwise present are statistically indistinguishable from supplying nothing at all" | H2; V2 "entirely"; H4 "at all" | "the cost falls on the two channels that are algebraic combinations of quantities already observed; the two that add information are indistinguishable from the no-channel arm." |
| 121-125 | "Throughout, we have tried to report what we measured rather than what we expected ... development history is littered with elegant, confident and wrong conclusions, each caught only by a measurement, and we have found that record more useful than the successes." | R7; R1 "rather than"; R2 triad; V "littered with" | Delete the paragraph. |

### 4c. Related work (L127-188)

| Line | Quoted | Pattern | Suggested rewrite |
|---|---|---|---|
| 130 | "\paragraph{Astrodynamics and the coarse model.}" | T1 stacked | "\paragraph{Patched conics.}" |
| 141-142 | "our setting differs in being game-scale, partially observed and adversarially generated rather than drawn from a mission catalogue." | R2 triad; R1 "rather than" | "our tasks are generated to a target difficulty rather than drawn from a catalogue, and are partially observed." |
| 144 | "\paragraph{Chaos, and whether it is present.}" | T1 | "\paragraph{Chaos on episode timescales.}" |
| 145-146 | "Our initial design intent was for the environment to be chaotic, since chaotic regimes are where coarse mental models fail most interestingly." | R7; V7 "mental models" | "A chaotic field is the natural place to look for coarse-model failure." |
| 149-151 | "We rejected this route on timescale grounds and verified the rejection empirically (Section~\ref{sec:chaos}): resonance overlap governs" | R7; S3 colon | "Resonance overlap governs evolution over many orbital periods, and an episode here is a fraction of one (Section~\ref{sec:chaos})." |
| 162-163 | "and becomes fractal --- losing any useful descent direction --- when $\lambda$ exceeds" | S4 | "and becomes fractal, with no useful descent direction, when..." |
| 166-167 | "We measured against it rather than assuming either side of it." | S1; R1; R7 | Delete; Section 5 reports the measurement. |
| 170-171 | "That ``how simple is the rule'' and ``how precisely must it be timed'' are different quantities is argued from two directions:" | V6 scare quotes; S3 | "Policy simplicity and timing precision are argued to be distinct quantities by..." |
| 174-175 | "A bang-bang controller on a razor-thin switching surface is a one-bit policy on an impossible trigger." | R4 aphorism; V "razor-thin" | "A bang-bang controller with a narrow switching surface has a one-bit policy and an execution margin below human reach." |
| 184-185 | "This discipline is the reason several results in this paper are reported as null." | R7-adjacent | "Several results below are null at this protocol." |
| 186-188 | "both of which make clear that our training budget is small by current standards --- a limitation we return to in Section~\ref{sec:limits}." | S4 | "...small by current standards (Section~\ref{sec:limits})." |

### 4d. Environment (L190-267)

| Line | Quoted | Pattern | Suggested rewrite |
|---|---|---|---|
| 203-204 | "The ship is a \emph{test particle}: it feels every body but moves none. This is the single most consequential design decision in the environment." | S3; S5 antithesis; V4 significance inflation | "The ship is a test particle, exerting no force on the bodies, so a level's trajectories are fixed by its initial snapshot. This is what allows..." |
| 218 | "A \emph{mission} is a short tour: between one and six targets" | STYLE word; S3 | "A task is a tour of one to six targets, each..." |
| 247 | "The action space is deliberately the set a person can press: turn" | S3; V2 "deliberately" | "The action space is the nine combinations a person can press on a gamepad: ..." (colon here introduces a set, which is content; acceptable if "deliberately" goes) |
| 257-259 | "if it remains well-behaved --- no collisions, no ejections, bounded energy drift --- for $900$\,s" | S4 pair around a triad; R2 | "if it has no collisions or ejections and bounded energy drift for 900 s" |

### 4e. Defining difficulty (L269-426)

| Line | Quoted | Pattern | Suggested rewrite |
|---|---|---|---|
| 272 | "\subsection{The coarse model and what it discards}" | T1 | "\subsection{The one-body model}" |
| 274-276 | "The mental model a person brings to a gravitational field is almost always the one-body model: identify the dominant attractor, treat its field as the whole field, and reason about conic sections." | V7 "mental model"; S3; R2 triad | "The coarse approximation of a gravitational field is the one-body model, in which the dominant attractor's field stands for the whole and trajectories are conic sections." |
| 276-277 | "We instantiate exactly this model in code." | S1 stunted; H4 "exactly" | "The function \code{gravity\_at(..., topk=k)} instantiates it, restricting..." |
| 279-280 | "$k=1$ is patched conics, and $k=0$ is the truth. Bodies always move truthfully --- only the test particle's felt force is restricted --- so" | S5 parallel binary; S4 | "$k=1$ is patched conics and $k=0$ the full field; because only the test particle's felt force is restricted and the bodies move under the full field, a plan made..." |
| 284 | "This gives the first axis of difficulty a natural definition." | V3 "natural"; D2 announcement | Delete; go to "Let $a_{\mathrm{full}}$ be..." |
| 291-292 | "Near zero, patched conics is right; near one, it is lying." | S5; V7 "lying" | "Near zero the one-body model is adequate; near one it discards most of the field." |
| 294 | "\subsection{Authority, and the ratio that matters}" | T1 (STYLE's exact form) | "\subsection{Authority and the demand $\dd$}" |
| 310-311 | "which has a direct reading: \emph{the share of the fuel tank that flying the wrong model would cost}." | S3; H2 | "the share of the tank that flying the wrong model would cost." |
| 313 | "thousands of candidate missions can be scored essentially for free" | H4 "essentially" | "thousands of candidate tasks can be scored without simulation" |
| 317 | "$\dd$ also \emph{inverts}." | S2 stunted; H2 | "Solving for the tank," (run straight into the equation) |
| 321-324 | "by solving rather than by generate-and-test. One practical caution: when retuning a mission onto a rung we only ever \emph{add} fuel. Reaching a high rung by starving the tank produces an infeasible mission dressed up as a demanding one, which is a different thing." | R1; R6 labelled caution; S3; H2; V "dressed up" | "...by solving rather than by generate-and-test, with the restriction that retuning only ever adds fuel, since a task starved onto a high rung is infeasible rather than demanding." |
| 329-330 | "That is why patched conics works here, and why breaking it requires deliberate construction rather than luck." | R1 "rather than"; R4 | "Patched conics therefore works on an ordinary task, and $\eta$ has to be raised by construction." |
| 332 | "\paragraph{Validation.} We compared a planner restricted to the one-body model" | S1 | "A planner restricted to the one-body model and one with the true field were compared on..." |
| 336-339 | "We note explicitly that the \emph{per-rung} figures, at $n = 8$, are not significant; only the pooled comparison is. This is the first of several places where our own early reporting outran its sample size." | H1; H2; R7 | "At $n=8$ per rung no single rung is significant; the pooled comparison is." |
| 341 | "\subsection{Where to put a target: the equal-pull surface}" | T3 colon title | "\subsection{The equal-pull surface}" |
| 343 | "To raise $\eta$ constructively one needs a place to stand." | R4 aphorism | "Raising $\eta$ by construction needs a region of the level where the one-body model fails." |
| 343-345 | "The literature's obvious candidate is the chaotic zone ... We tried it and rejected it." | R9 vague attribution; R7; S2 | "The natural candidate is the chaotic zone around a secondary, delimited by resonance overlap [CITE]; it does not work on this timescale, because..." |
| 347 | "The measurement was blunt:" | S3; V3 "blunt"; R6 | "Placing targets inside the Wisdom band lifted $\eta$ from 0.0026 to 0.0172, still negligible, because..." |
| 363 | "We note that the peak value depends strongly on how one samples it." | H1 "We note" | "The peak depends on how it is sampled: along the line joining..." (colon here introduces the mechanism; acceptable) |
| 371-373 | "--- a spread of three orders of magnitude. Systems of comparable masses are contested everywhere; hierarchical systems are contested only in thin shells" | S4; S5 | "..., three orders of magnitude. Systems of comparable masses are contested throughout, whereas hierarchical systems are contested only in thin shells that must be aimed at." |
| 388 | "\subsection{The second axis, and why it must stay separate}" | T1 | "\subsection{Execution margin}" |
| 391-393 | "$\dd$ measures whether a coarse \emph{model} suffices. It says nothing about whether a correct plan can be \emph{executed}. We therefore band every mission on two axes and never reduce them to one number:" | S9 staccato; H2; S1; S3 | "Because $\dd$ says only whether a coarse model suffices, and nothing about whether the resulting plan can be executed, every task is banded on two axes." |
| 395-404 | "\item \textbf{Model scale} --- the coarsest $k$ ... \item \textbf{Execution margin} --- timing slack" | D4 inline-header list; S4 | Two prose sentences. |
| 401-403 | "already coarser than a trained hand, which makes the comparison generous rather than rigged." | R1 "rather than" | "already coarser than a trained hand, so the comparison favours the agent." |
| 412-414 | caption: "different kinds of impossible and need opposite repairs, which is why a single scalar difficulty would merge exactly the two cases a designer most needs to tell apart." | D5 duplicated at L421-426; H4 "exactly" | Caption: "Bands on the two axes; the horizontal axis is model demand $\dd$, the vertical is execution margin." |
| 418-420 | "\emph{trivial}, \emph{human}, \emph{calculated} (...), \emph{precision} (...) and \emph{opaque} (nothing we have succeeds; rejected as unfair)" | H2 (defining terms, acceptable); R7 "nothing we have" | "...and opaque (no available method succeeds; excluded)." |

### 4f. Chaos (L428-473)

| Line | Quoted | Pattern | Suggested rewrite |
|---|---|---|---|
| 431-434 | "Several of our design threads --- fractal basin boundaries, a ``lottery'' difficulty band, and a worry that ... --- presupposed that the field was chaotic on episode timescales. It is not, and establishing that changed the plan." | R7; S4 pair around a triad; V6 scare quotes; S2 | "Whether the field is chaotic on episode timescales decides whether basin boundaries are fractal and whether the Hölder result of \citet{wang2023} bites. It is not chaotic on that timescale, as follows." |
| 436 | "The concern with sharpest teeth was the H\"older result" | R6 labelled device; V metaphor | "The H\"older result of \citet{wang2023} sets the threshold: at..." |
| 439-442 | "Our measured predictability horizon --- the time at which one control tick's worth of thrust error grows past the $12$\,u arrival tolerance --- is $52$--$74$\,s across eight families, uncomfortably close to the $67$\,s" | S4; H4 "uncomfortably" | "The predictability horizon (the time for one tick of thrust error to exceed the 12 u arrival tolerance) is 52-74 s across eight families, close to the 67 s that corresponds to $\lambda_{\mathrm{crit}}$." |
| 444 | "The resolution is that the divergence is \emph{ballistic, not exponential}." | R6; R1 "X, not Y"; H2 | "The divergence is ballistic: $d\ln(\mathrm{sep})/dt$ falls as $1/t$..." |
| 455-456 | "We are comfortably inside the benign regime." | H4 "comfortably"; S2 | "The environment is in the regime where the objective is not fractal." |
| 458 | "The uncertainty exponent told the same story the hard way." | V7; R7; V "the hard way" | "The uncertainty exponent gives the same answer." |
| 458-464 | "Our first estimates of $\alpha$ were computed from too few samples ... Zero of five exponents was trustworthy, and we had reported a fractal/resolvable split that the data did not support." | R7 | "At 96 perturbation samples the boundary-flip counts are small integers (0, 1, 3, 5, 5) and the bootstrapped intervals straddle the fractal/resolvable threshold, so nothing can be concluded; at eight times the samples $\alpha$ measures 0.53-1.02, resolvable." |
| 466-467 | "We also added an explicit \code{measurable} flag so that an unmeasurable exponent propagates as \emph{unknown} rather than as \emph{fractal}." | R7 implementation diary; R1; H2 | Delete, or one clause: "an unmeasurable exponent is reported as unknown." |
| 469 | "Three loose ends resolve at once." | R3 triad announced | Delete. |
| 472-473 | "And a previously published claim of ours, that a training plateau was caused by a fractal objective, is retracted: that plateau needs a different explanation." | H5 "And"; R7 ("retracted", "previously published"); S3 | "A training plateau cannot therefore be attributed to a fractal objective." |

### 4g. Measuring difficulty (L475-546)

| Line | Quoted | Pattern | Suggested rewrite |
|---|---|---|---|
| 478 | "\subsection{The planner as yardstick, and the solvability gate}" | T1 (STYLE's exact example) | "\subsection{Solvability}" |
| 480 | "but it cannot tell you whether a mission is flyable at all." | R8 "you"; H4 "at all" | "but not whether a task is flyable." |
| 486-488 | "Missions that nothing can fly are excluded from difficulty labelling: a level nobody can fly earns no label, since its measured difficulty would describe the probe rather than the level." | S3 colon restating; R1 | "Tasks that no method can fly receive no label, since their measured difficulty would describe the probe." |
| 490 | "Including the zero-thrust candidate matters more than it sounds." | R5 telegraph | Delete; begin "Throttle is parameterised through a sigmoid, so..." |
| 496 | "\subsection{A retraction, and what it says about the benchmark}" | T1; R7 "retraction" | "\subsection{Sample size and the two-stop class}" |
| 499 | "We previously reported that two-stop missions --- tours with two sequential targets, typically with at least one rendezvous --- were \emph{infeasible}" | R7 (banned verbatim); S4; H2 | "Two-stop tasks (two sequential targets, usually one a rendezvous) look infeasible at the 150 s clock on a small sample: twelve such tasks under three search budgets..." |
| 506 | "That conclusion was wrong, and the way it was wrong is instructive." | R7; R5 telegraph | Delete. |
| 512-513 | "$(1-5/32)^{12} = 0.13$ --- an unremarkable outcome, not evidence of impossibility." | S4; R1 "X, not Y" | "$(1-5/32)^{12} = 0.13$, so zero of twelve is unremarkable." |
| 522 | "More time buys more exposure." | R4 aphorism; S2 | Fold into the previous sentence: "...so a longer clock buys hazard exposure as often as it buys arrivals." |
| 532-534 | caption: "An earlier report of ours, based on twelve correlated samples at the $150$\,s clock, concluded that this mission class was infeasible." | R7 in a caption | Delete the sentence. |
| 538 | "\subsection{What the episode clock is, structurally}" | T4 teaser heading with comma tag | "\subsection{The episode clock}" |
| 541-544 | "This is a limitation we discovered while running the sweep above --- each clock required a separate process with the constant patched at import --- and it is the reason" | R7; S4 | "Each clock in the sweep above required a separate process with the constant patched at import, and the same constant is why the environment cannot run long tasks for play while measuring on a short clock." |
| 544-546 | "We note it because it is exactly the kind of structural detail that silently constrains which experiments are cheap." | H1 "We note"; H4 "exactly the kind of"; R4 | Delete. |

### 4h. Learning and instruments (L548-717)

| Line | Quoted | Pattern | Suggested rewrite |
|---|---|---|---|
| 553 | "We train with Proximal Policy Optimization" | S1 (methods voice; mild) | "Training uses Proximal Policy Optimization (PPO) \citep{schulman2017}..." |
| 585-586 | "a scripted reflex pilot --- proportional navigation with gravity compensation --- evaluated" | S4 | "a scripted reflex pilot (proportional navigation with gravity compensation) evaluated" |
| 615 | "Two observations deserve emphasis. First, the learned policy \emph{loses} to" | R5; R3; H2 | "The learned policy loses to a hand-written reflex controller on the held-out families, and the reflex pilot is a weaker yardstick than it appears: on..." |
| 617-619 | "is a weaker yardstick than it appears: on a comfortable-tank mission set it arrives $63.3\%$ of the time, and its dominant failure is \emph{collision} at $21.6\%$, not fuel exhaustion ($4.7\%$) or timeout ($7.8\%$)." | S3; R1 "X, not Y"; H2 | "...arrives 63.3% of the time on a comfortable-tank set, failing by collision (21.6%) far more often than by fuel exhaustion (4.7%) or timeout (7.8%)." |
| 619-621 | "Any margin quoted against this opponent is correspondingly soft, a caveat we attach to every such comparison. Absolute success rates are unaffected." | H1 "caveat"; S2 | "Margins against it are therefore soft; absolute rates are not affected." |
| 628-630 | "the natural question is whether an agent told those quantities learns better. We added four channels to the observation:" | V3 "natural"; S1; S3 (introduces a list; acceptable) | "Whether an agent told those quantities learns better was tested by adding four channels to the observation: ..." |
| 633-634 | "These are intended as a cockpit: the quantities a well-instrumented pilot would read." | S3; metaphor label | "These are the quantities an instrumented pilot would read." |
| 638-639 | "We verified separately that toggling them does not perturb the dynamics." | S1; R7 | "Toggling them does not change the dynamics (verified by \code{[script]})." |
| 647-649 | "We emphasise that these are tests on \emph{seed means}: the quoted uncertainties are training-seed standard deviations, not episode-level binomial intervals, which are much smaller and would overstate confidence." | H1 "We emphasise"; H2; S3; R1 "X, not Y" | "The tests are on seed means; the quoted uncertainties are training-seed standard deviations, since episode-level binomial intervals are much smaller and would overstate confidence." |
| 689 | "The explanation was visible in the observation code before it was measured." | R7 narration | "The observation already carries..." |
| 700-701 | "This generated a prediction: splitting the panel along the redundancy line should put the entire cost on the redundant half. It does." | S3; S2 "It does."; R7 | "If redundancy is the cause, splitting the panel along that line should put the whole cost on the redundant half, and it does: supplying only..." (acceptable with the fragment folded in) |
| 708 | "We draw a narrow conclusion and resist a broad one. Narrowly:" | S1; S5 antithesis; R6 label; S3 | "In this environment, at this network size and step budget, appending algebraically redundant features measurably degrades a policy, while..." |
| 711-713 | "genuinely new physics is free. We do \emph{not} have evidence that the informative channels help --- $\eta$ and stretch are nominally above ... none significantly." | V2 "genuinely"; H2 `\emph{not}`; S4 | "...is free. There is no evidence that the informative channels help: $\eta$ and stretch are nominally above the no-channel arm on three suites and below on one, none significantly." |
| 714-717 | "And we have not isolated the mechanism. Four extra input dimensions against $285{,}322$ parameters is a capacity story; a ratio with a division in it is heavy-tailed even after log compression, which is a conditioning story. Both remain untested." | H5 "And"; S5 parallel "X story; Y story"; S2 closer | "The mechanism is not isolated: four extra input dimensions against 285,322 parameters suggests capacity, while a ratio with a division in it is heavy-tailed even after log compression, which suggests conditioning; neither has been tested." |

### 4i. Ghosts, threats, reproduction, limitations (L719-860)

| Line | Quoted | Pattern | Suggested rewrite |
|---|---|---|---|
| 719 | "\section{What a single number hides}" | T4 teaser | "\section{Per-task variation}" |
| 723-725 | "That number conceals both the policy's own stochasticity and the enormous variation between individual missions. Figure~\ref{fig:ghosts} makes both visible:" | V2 "enormous"; S3 | "That number conceals the policy's stochasticity and the variation between tasks, which Figure~\ref{fig:ghosts} shows: ..." |
| 740-742 | caption: "an agent confidently executing a plan that cannot complete the mission." | V7 "confidently" | "a policy executing a tight plan that cannot complete the task." |
| 746 | "Two things follow. First, sampled success is far below greedy success --- the \emph{easy} mission shown arrives..." | R5/R3; S4; H2 | "Sampled success is far below greedy success (the easy task shown arrives 52% of the time when sampled, against 89% greedy on that suite), so..." |
| 754-756 | "Any claim about this environment that rests on a handful of missions is therefore fragile by construction, which is the lesson of Section~\ref{sec:retraction} restated in pictures." | R4; R7 self-reference | "A claim resting on a handful of tasks is therefore fragile by construction (compare Section~\ref{sec:retraction})." |
| 764-766 | caption: "reading an ablation before it finishes is actively misleading; we did so twice during this work and drew the wrong conclusion both times." | R7 in a caption | "Mid-run checkpoints are noisy enough that an ablation read before it finishes can invert." |
| 793-795 | "Several phenomena that matter in real mission design --- plane changes, out-of-plane perturbations --- cannot occur." | S4 | "Plane changes and out-of-plane perturbations cannot occur." |
| 803-807 | "including the ones that were later retracted; most of the self-corrections reported in this paper are traceable to a specific record and the measurement that forced it." | R7 "retracted", "self-corrections" | "...a numbered, append-only series of decision records giving the reasoning and the measurement behind each design choice." |
| 831-833 | "we do not know \emph{why} redundant observation features cost a policy ten points. Two mechanisms --- capacity pressure and the conditioning of heavy-tailed ratio features --- predict" | H2; S4 | "the mechanism by which redundant features cost ten points is not known. Capacity pressure and the conditioning of heavy-tailed ratio features predict..." |
| 837 | "Second, the environment is not chaotic, and we would like it to be." | R3 enumeration; R7 wish | "The environment is not chaotic on episode timescales; making it so would need..." |
| 838-839 | "suggests the route: closer encounters, tighter binaries, and above all episodes that are long relative to orbital periods." | S3; R2 triad | "...would need episodes long relative to orbital periods, which in turn needs a per-task episode clock." |
| 846-852 | "What we think transfers beyond this particular environment is the shape of the problem. A task can be cheap to simulate, deterministic, fully specified, and still resist ... in ways that depend not on whether the information is \emph{true} but on whether it is \emph{new}. None of those three is specific to gravity." | H3 "we think"; R2 (STYLE's exact triad); R1 "not X but Y"; H2; R4 closer; D2 summary | Delete the paragraph; the abstract carries it. |
| 856-860 | "include the agent's own retracted conclusions" | R7 "retracted" | "...were written during that process." |

### 4j. What the audit did not flag

- `\emph` used to define a term at first use (L203 *test particle*, L221
  *rendezvous*, L418-420 band names) is correct and not counted.
- Content lists with real cardinality: the three consequences of the test
  particle at L206-209 (i)-(iii), the four pools at L263-267, the four suites
  at L575-582. Lists are for content.
- Colons that introduce an equation, a set or an enumeration (L247, L559,
  L628) rather than a reveal.
- Sentence-case `\section` headings (T6) are already right.
- The Related work and Threats sections are mostly clean of S1 because their
  sentences have cited authors or named quantities as subjects. That is the
  model for the rest.


---

## 5. Audit of draft2.tex

Line numbers from `cat -n paper/draft2.tex` (1042 lines, untracked, read
2026-10-30 after the coordinator's message). Same table format as section 4.
This audit allows for `paper/style/voice.md`: sentence-initial "However,",
"Thus,", "Moreover,", "Of course,", "Note that", "Taken together,", "Firstly,
... Secondly,", "Here, we examine", long sentences with parenthetical
definitions, and inline "1) ... 2)" lists are the author's own habits and are
not flagged. Colons that introduce an equation, a specification or an
enumeration are not flagged (voice.md item 6); colons that set up a reveal
are.

### 5a. Measured counts, draft 1 against draft 2

Same greps as section 4 (`grep -oE '(^|[.:;] )We [a-z]+'`, `grep -oE ' --- '`,
`grep -oE '\\emph\{'`, `grep -o 'rather than'`, `grep -c 'mental model'`,
`grep -oi 'mission'`).

| Measure | spacenav.tex (draft 1) | draft2.tex |
|---|---|---|
| Sentences beginning "We <verb>" | 24 | 0 |
| Spaced em dashes ` --- ` | 29 | 0 |
| `\emph{}` | 78 | 0 |
| "rather than" | 13 | **28** (lines 60, 70, 73, 75, 85, 127, 182, 253, 407, 509, 528, 552, 571, 590, 611, 620, 621, 647, 720, 840, 912, 916, 928, 932, 962, 971, 982, 983) |
| "mental model" | 3 | 0 |
| "mission" | 73 | 4 (lines 6, 122, 191, 602: a comment, "mission designer" twice, "historical missions"; all permitted by STYLE) |
| "exactly" | 6 | 10 (lines 203, 277, 293, 370, 531, 567, 609, 908, 929, 1020) |

The five surface tells STYLE named are gone. What is left is the residue
that a word-level pass does not reach: the draft now carries its rhetoric in
"rather than" (one every 37 lines, roughly twice the first draft's rate),
in "not X but Y" with the negation spread across a clause, in colons that
set up a reveal, in paragraph-closing aphorisms, in mirrored A/B clauses,
and in a handful of surviving triads, including STYLE's own example at
L900. Several arguments are also made three or four times in nearly the
same words (the two-repairs chiasmus, the Voyager/Parker/Artemis triad,
"constructed rather than assumed"), which reads as generated even when each
instance is clean. The author's own corpus uses "rather than" as a real
contrast between two named alternatives (voice.md 1.6: "rather than focus
on one type of spatial relational error (e.g. swaps), we sought to ...");
most instances below are the rhetorical form in which the rejected
alternative was never on the table.

Rows are ordered by line. Priority: **H** = would be flagged by a reader as
machine prose on its own; M = contributes to density; L = minor, fix if
touching the sentence anyway.

### 5b. Title and abstract

| Line | Quoted | Pattern | Pri | Suggested rewrite |
|---|---|---|---|---|
| 37-38 | "Defining, Measuring, and Learning Difficulty in a Game-Scale Gravitational Navigation Environment" | T5 gerund triad (unchanged from draft 1) | H | Drop the subtitle, or "Difficulty in a game-scale gravitational navigation environment". The comment at L6-10 already lists (a) as the main title; the subtitle is draft 1's. |
| 50-53 | "a problem with a long history in astrodynamics and a short one in reinforcement learning, and the two literatures do not agree" | S5 mirrored long/short | L | Acceptable as a real contrast; keep if the "do not agree" claim gets its `\CITE`. |
| 62-64 | "a representational component, i.e. the degree to which a coarse one-body model of the field is wrong, which this paper is about, and a reaction-time component, which it is not." | S5 parallel binary with kicker "which it is not" | M | "...and a reaction-time component (the width of the window in which an action must be taken), of which only the first is treated here." |
| 69-70 | "so that a level can be constructed at a chosen difficulty rather than searched for" | R1 "rather than" (1 of 5 in the abstract) | L | "so that a level can be constructed at a chosen difficulty." |
| 72-75 | "where chaos is wanted it must be built into a level rather than assumed, and the episode clock is shown to be the least informative of the constraints, since a craft stopped by the clock on an energy-efficient path has failed a hyperparameter rather than the task." | R1 x2; R4 aphorism ("failed a hyperparameter rather than the task") | M | "...so chaos has to be built into a level deliberately; and the episode clock is the least informative of the constraints, since at 150 s a substantial share of scored failures are timeouts on energy-efficient paths." |
| 83-87 | "Taken together, these results suggest that the difficulty of an n-body navigation task is a property to be measured on several axes rather than a knob to be set" | R1; R4 antithesis "property ... knob" | M | "Taken together, these results suggest that the difficulty of an n-body navigation task has several separable axes and that each has to be measured, and the environment..." |

### 5c. Section 1, Navigating an n-body field (L91-210)

| Line | Quoted | Pattern | Pri | Suggested rewrite |
|---|---|---|---|---|
| 100-101 | "managed the problem for sixty years by never confronting it whole: a trajectory is designed as a sequence of two-body arcs" | S3 colon reveal | M | "...has nonetheless managed the problem for sixty years without confronting it whole, since a trajectory is designed as a sequence of two-body arcs..." |
| 108-113 | "Voyager 2 ..., Parker Solar Probe ..., and Artemis~I ... and in every case the design was judged on fuel, on exposure, and on arrival, and not on how long it took." | R2 triad with anaphora ("on fuel, on exposure, and on arrival"); R1 "and not" kicker; repeated at L601-606 (D5) | H | Keep the three missions here (they are content and cited); end "...and in every case the design was judged on fuel, exposure and arrival rather than elapsed time." Then do not repeat the three at L601-606. |
| 115 | "Reinforcement learning has a much shorter history with the same field." | S2 short thesis sentence opening a paragraph (voice.md 1.1: short sentences are qualifications, not theses) | L | "Reinforcement learning has a much shorter history with the same field, in which learning-based guidance..." (merge with the next sentence). |
| 123-128 | "The present work grew out of a series of visual explorations of gravitational dynamics by Braintruffle ..., whose treatment of what a many-body field looks like from the inside of a trajectory, rather than from above it, is the picture of the problem this paper tries to make precise." | R7 provenance narrative (voice.md 6.3: never narrates the research process); R1 | M | Coordinator's call. If the credit stays: "Braintruffle's visual treatment of a many-body field from inside a trajectory \CITE{} is the picture this paper tries to make precise." The "grew out of" is the narrated origin. |
| 162-164 | "three pictures of the task, each of which is correct, each of which suggests a different notion of difficulty, and none of which is sufficient alone." | R2 tricolon with anaphora ("each of which ... each of which ... none of which") | H | "three pictures of the task, which are each correct and each suggest a different notion of difficulty, and no one of which is sufficient alone." |
| 169-170 | "This picture works, in the sense that an agent trained this way does learn to fly (Section~\ref{sec:results}), and it fails in a specific way: when the craft enters a region" | S5 works/fails antithesis; S3 colon reveal | M | "An agent trained this way does learn to fly (Section~\ref{sec:results}), but when the craft enters a region in which nearby trajectories diverge exponentially, the objective..." |
| 174-176 | "a threshold that the environment, not the algorithm, sets" | R1 "X, not Y" interjected | M | "a threshold set by the environment's Lyapunov exponent rather than by the algorithm" (one "rather than" that names both alternatives is fine; the interjected "not the algorithm" is the tic). |
| 177-178 | "Whether the field of this environment actually crosses that threshold is a measurable question, and Section~\ref{sec:chaos} measures it." | R4 echo closer (question/measures) | L | "Whether this environment's field crosses that threshold is measured in Section~\ref{sec:chaos}." |
| 184-189 | "the right goal inside such a region is not to predict the exit trajectory precisely (which is impossible by definition) but to bound it ... a competent navigator plans to adapt after it rather than to be right through it" | R1 "not X but Y"; R1 "rather than"; R4 aphorism ("adapt after it rather than be right through it"), repeated at L939-940 | H | "inside such a region the exit trajectory cannot be predicted, by definition, but it can be bounded, i.e. the region can be entered in a way that constrains the set of exits to something a plan on the far side can absorb. On this view a chaotic region is a noise filter, and the navigator's plan is for what comes after it." |
| 192 | "makes "difficulty" depend on what lies after a region" | V6 scare quotes on the paper's own term (voice.md 7) | M | "makes difficulty depend on..." |
| 202-204 | "a level is hard on this picture exactly when the rung a navigator is standing on is the wrong one for it." | H4 "exactly"; R4 | L | "a level is hard on this picture when the rung a navigator stands on is the wrong one for it." |

### 5d. Section 2, Why a level of a given difficulty is hard to design (L213-269)

| Line | Quoted | Pattern | Pri | Suggested rewrite |
|---|---|---|---|---|
| 218-224 | "A level can be hard because the field is contested ...; because an encounter scatters ...; because the window ...; because the tank is short; because the hazards are dense; or because the targets happen to be placed so that any one of these is unavoidable." | R2-style anaphora over six items (content, but the repeated "because" is cadence; voice.md 1.8 numbers inline lists) | M | "A level can be hard for at least six reasons: 1) the field is contested and a coarse representation of it is wrong (the third picture); 2) an encounter scatters trajectories so widely that nothing on the far side can be planned for (the second); 3) the window in which a decisive action must be taken is narrower than a human hand can hold; 4) the tank is short; 5) the hazards are dense; or 6) the targets are placed so that one of these is unavoidable." |
| 224 | "These are not points on a line." | S2 one-line thesis reveal | H | Delete; the next sentence says it. |
| 226-228 | "widening a timing window does nothing for a contested field, and simplifying the field does nothing for a narrow window." | S5 chiasmus; repeated at L500-504 (D5) | M | Keep one of the two statements; here: "and the repair for one does nothing for another." |
| 234-235 | "The problem is compounded by the fact that none of these notions can be checked by exhaustion." | R5 one-sentence teaser paragraph into the subsection | M | Fold into the subsection's first sentence: "None of these notions can be checked by exhaustion. Consider the most naive possible procedure..." or drop, since L240 opens with the worked case anyway. |
| 247-248 | "Formal results in motion planning make the same point without the arithmetic:" | S3 (introduces three results, so partly enumerative) | L | "Formal results in motion planning make the same point: the generalized mover's problem is..." is acceptable; the "without the arithmetic" flourish is the tell. |
| 253-254 | "well inside the hard regime rather than at its boundary" | R1 | L | "well inside the hard regime." |
| 266-267 | "Precision of the check, not merely its cost, becomes the problem." | R1 "X, not merely Y"; S2 short thesis | H | "...so that the check's precision, and not only its cost, is the limit." Or fold into the preceding sentence: "...a trajectory the craft cannot actually fly, so that precision of the check becomes the problem before its cost does." |

### 5e. Section 3, Heuristics (L272-635)

| Line | Quoted | Pattern | Pri | Suggested rewrite |
|---|---|---|---|---|
| 281 | "\subsection{The environment, briefly}" | T4 comma-tagged adverb heading (draft 1's "What the episode clock is, structurally") | M | "\subsection{The environment}" |
| 290-291 | "The craft is a test particle: it feels every body and moves none, so a level's body trajectories are determined entirely by their initial snapshot" | S3 colon; S5 antithesis (draft 1 L203 survived) | H | "The craft is a test particle (it is accelerated by every body and accelerates none), so a level's body trajectories are determined by the initial snapshot, which is what allows..." |
| 364 | "\subsection{Nontrivial difficulty: whether a coarse representation suffices}" | T3 colon title | L | "\subsection{Nontrivial difficulty}" with the gloss in the first sentence. (STYLE names *nontrivial difficulty* as a term, so the head noun is right.) |
| 367-369 | "is almost always the one-body representation: identify the dominant attractor, treat its field as the whole field, and reason in conic sections." | S3 colon; R2 imperative triad (draft 1 L274-276 survived) | H | "is almost always the one-body representation, in which the dominant attractor is identified, its field is taken for the whole field, and trajectories are reasoned about as conic sections." (Three steps are content; the imperative triad after a colon is the cadence.) |
| 369-370 | "This is instantiated exactly in the environment by a function" | H4 "exactly" | L | "This is instantiated in the environment by a function" |
| 377-378 | "This gives the representational component of difficulty a natural definition." | V3 "natural"; D2 announcement sentence (draft 1 L284 survived) | M | Delete; go straight to "With $a_{\mathrm{full}}$ the true local acceleration..." |
| 386-387 | "a craft with a large enough engine can absorb the error without noticing it" | V7 (a craft noticing) | L | "can absorb the error." |
| 407-410 | "by solving rather than by generating and testing (with the caution that fuel is only ever added when retuning, since reaching a rung by starving the tank produces an infeasible level dressed as a demanding one)" | R1; V "dressed as" (blocklist); R6 labelled caution | M | "by solving for the tank (fuel is only ever added when retuning, since a level starved onto a high rung is infeasible, not demanding)." |
| 412-416 | "this is why patched conics works, and why breaking it requires deliberate construction." | R4 closer; "deliberate" | L | "...so patched conics is adequate on an ordinary level and $\eta$ has to be raised by construction." |
| 426 | "Where in a level a coarse representation breaks is not arbitrary." | S2 short thesis opener | M | Merge: "Where in a level a coarse representation breaks is governed by instantaneous force competition rather than by the resonance structure, although the latter is the natural candidate from the literature on constructed chaos \CITE{}: resonance overlap..." (one real "rather than", both alternatives named). |
| 429-430 | "and it is the wrong scale for this task: resonance overlap describes what happens over many orbital periods, while an episode here is a fraction of one" | S3 colon reveal | M | "...and it is the wrong scale for this task, since resonance overlap describes what happens over many orbital periods while an episode here is a fraction of one;" |
| 433-434 | "is an $\eta \approx 0.02$ desert" | V invented metaphor label | L | "has $\eta \approx 0.02$ throughout." |
| 490 | "\subsection{Trivial difficulty: reaction time}" | T3 colon title (pair with L364) | L | "\subsection{Trivial difficulty}" |
| 500-504 | "since a level that is hard for this reason is repaired by widening its window and a level hard for the representational reason is repaired by simplifying its field, and a single number would merge the two cases a designer most needs to tell apart." | D5 third statement of the two-repairs argument (L226-228, caption-free now); S5 | M | Since L226-228 made the point, here: "and the two are never combined (Section~\ref{sec:hard})." |
| 504-507 | "This kind of difficulty is called trivial here not because it is easy but because it is uninteresting: it says nothing about the field, and this paper has nothing further to say about it." | R1 "not because X but because Y"; S3 colon; R4 closer ("nothing further to say") | H | "This kind of difficulty is called trivial here because it says nothing about the field, and it is not treated further." |
| 509 | "\subsection{Chaos, measured rather than assumed}" | T1 comma-stacked heading; R1 in a title | H | "\subsection{Chaos on episode timescales}" |
| 514 | "sharp enough to be worth checking against" | H1-adjacent "worth" | L | "sharp enough that the field has to be checked against it." |
| 525-526 | "close enough to that figure that the question had to be settled directly." | R7-adjacent narration ("had to be settled") | L | "close enough to that figure that the exponent has to be measured rather than bounded." |
| 528 | "The divergence is ballistic rather than exponential." | S2 short thesis opener; R1 (draft 1 L444 "ballistic, not exponential" survived with the connective swapped) | M | Keep the sentence but attach the discriminator: "The divergence is ballistic, since $d\ln(\mathrm{sep})/dt$, which falls as $1/t$ for linear growth and stays flat for exponential growth, ..." |
| 549-552 | "it has to be built into a level, by closer encounters, tighter binaries, and above all episodes long relative to orbital periods, rather than assumed to be present." | R2 triad with "above all"; R1; repeated at L72-73, L944-945, L981-983 (D5: four statements of "constructed rather than assumed") | M | State the route once, here, and refer to it elsewhere: "it has to be built into a level, chiefly by episodes long relative to orbital periods (closer encounters and tighter binaries help less)." |
| 554 | "\subsection{Solvability and the missing optimum}" | T1 "X and the Y" | M | "\subsection{Solvability}" |
| 566-567 | "which caused the optimizer to fail on exactly the levels where coasting is optimal" | H4 "exactly" | L | "to fail on the levels where coasting is optimal" |
| 573-574 | "Two properties of this yardstick have to be kept in view whenever it is used. Firstly, its failure is weak evidence." | R3 announcement + S2 (the "Firstly" itself is voice-approved) | L | "Two properties of this yardstick bear on every use of it. Firstly, its failure is weak evidence, since a given optimizer configuration..." (merge the short sentence into the one that explains it). |
| 586-588 | "This is not a limitation that more computation removes; it is the condition of the problem, and it is the reason the paper treats a learning agent ... as a second, independent probe of the same levels rather than as the thing being graded." | R1 "not X; it is Y"; R1 "rather than" | H | "More computation does not remove this limitation, which is the condition of the problem, and it is why the paper treats a learning agent (Section~\ref{sec:learning}) as a second, independent probe of the same levels as well as the thing being graded." |
| 592 | "\subsection{The episode clock is not a difficulty knob}" | T2 sentence heading, negated (voice.md 6.7) | H | "\subsection{The episode clock}" |
| 601-606 | "the historical missions remind us how perverse that penalty is: Voyager's grand tour, Parker's seven Venus encounters and Artemis~I's progressive orbit-raising were each judged on fuel, exposure and arrival, and a victory condition that failed Voyager for not reaching Jupiter within a year would have been absurd." | S3 colon reveal; R8 "remind us"; D5 repeat of L108-113; V4 "perverse", "absurd" | H | "The clock, by contrast, mostly penalizes a craft that chooses an energy-efficient path over a fast one, which is the choice every mission in Section~\ref{sec:field} was praised for making." |
| 608-609 | "binding in exactly that way" | H4 | L | "binding in that way" |
| 617-619 | "More time converts timeouts into attempts, and about half of those attempts into exposure." | R4 aphorism (draft 1 L522 "More time buys more exposure" survived, lengthened) | M | Fold into the preceding numbers: "...so that of the thirteen timeouts a longer clock removes, about half become arrivals and half become hazard endings." |

### 5f. Sections 4-5, Learner and results (L638-894)

| Line | Quoted | Pattern | Pri | Suggested rewrite |
|---|---|---|---|---|
| 638 | "\section{A learner as a lens on the levels}" | T4 metaphor heading (voice.md 3: headings are noun phrases naming content) | M | "\section{The learning agent}" |
| 641-643 | "for two purposes that are usually kept separate: it is evaluated on the levels, as any agent would be, and the levels are evaluated through it" | S3 colon; S5 chiasmus (evaluated on / evaluated through) | M | "A trained agent is used here both as a subject, evaluated on the levels as any agent would be, and as an instrument, since where and how it fails is a measurement of the levels that no closed-form quantity provides." |
| 646-647 | "attributable to the levels rather than to the procedure" | R1 (real contrast; both named) | L | Acceptable. |
| 717-722 | "Note that the reference pilot is a weaker yardstick than its rates suggest: on a comfortable-tank instance set it arrives 63.3\% of the time ... every margin quoted against it is softened by an opponent that flies into bodies one time in five" | S3 colon before evidence (borderline; "Note that" is voice) | L | "Note that the reference pilot is a weaker yardstick than its rates suggest, since on a comfortable-tank instance set it arrives 63.3% of the time and..." |
| 746 | "\subsection{What the agent should be told}" | T4 teaser (draft 1 L623 survived) | M | "\subsection{Observation channels}" |
| 768 | "The reason is visible in the observation vector before it is measured." | R6 labelled reveal (draft 1 L689 survived in present tense) | L | "The observation already carries the local gravity vector..." (start there). |
| 819 | "The conclusion to be drawn is narrow." | S2 short thesis sentence (draft 1 L708 survived) | M | "The conclusion is narrow: in this environment..." is still a colon reveal; better "In this environment, at this network size and step budget, appending features..." and let the scope clauses carry the narrowness. |
| 819-830 | one sentence, L819-830, "In this environment ... \MEASURE{}" | S9 inverse: a 120-word sentence with five coordinated "and"/"while"/"since"/"which" clauses | L | Voice-consistent in length; split once after "...none significantly)." |
| 832 | "\subsection{What a single success rate conceals}" | T4 teaser (draft 1 L719 survived) | M | "\subsection{Per-instance variation}" |
| 842-843 | "Early policies spray across the field and late policies converge into bundles, and the bundle is not always the right one: on the hard-suite instance shown" | S5 parallel; S3 colon reveal | M | "Early policies spray across the field and late policies converge into bundles, which are not always the right ones; on the hard-suite instance shown..." |
| 844 | "converges onto a confident shuttle between its first two targets" | V7 "confident" (draft 1 L741 survived) | M | "converges onto a tight shuttle between its first two targets" |
| 846 | "Two consequences follow. Firstly, sampled success..." | R3 two-word announcement (the Firstly/Secondly that follows is voice) | L | "Firstly, sampled success is far below greedy success..." (drop the announcement; the enumeration announces itself). |
| 854-856 | "Thus, any claim about this environment that rests on a handful of instances is fragile by construction, which is the argument of Section~\ref{sec:solvable} restated in pictures." | R4 self-referential closer (draft 1 L754-756 survived) | M | "Thus, any claim about this environment that rests on a handful of instances is fragile by construction (compare Section~\ref{sec:solvable})." |
| 873-876 | "from which one practical point follows: mid-run checkpoints are noisy enough that" | S3 colon reveal | L | "...learning curves, which show that mid-run checkpoints are noisy enough that..." |

### 5g. Discussion and future directions (L897-989)

| Line | Quoted | Pattern | Pri | Suggested rewrite |
|---|---|---|---|---|
| 900-902 | "The task examined here is cheap to simulate, fully specified, and deterministic, and it still does not admit a one-dimensional notion of difficulty" | R2: STYLE's own banned triad ("cheap to simulate, deterministic, fully specified"), reordered, survived from draft 1 L847 | H | "Although the task examined here is fully specified and cheap to simulate, it does not admit a one-dimensional notion of difficulty, which is..." (determinism is already implied by "fully specified" and stated in Section 3). |
| 907-910 | "are consistent with exactly that: strong on the hierarchical families, weak on the contested ones, and worst on the held-out families" | H4 "exactly"; S3 colon; R2 strong/weak/worst | M | "are consistent with that account, the policy being strong on the hierarchical families, weak on the contested ones and worst on the held-out families where..." |
| 912 | "\subsection{Learning a class of simulations rather than one of them}" | R1 in a heading | L | "\subsection{Learning a class of levels}" |
| 914-917 | "the representation it needs on the next, rather than a policy fitted to that level's particular geometry" | R1 (real contrast) | L | Acceptable. |
| 919 | "is a crude version of this, and its crudeness shows in the results:" | S3 colon reveal; "crude ... crudeness" echo | M | "is a crude version of this, since the contested share was held at zero until the agent could fly at all, which is to say..." |
| 926-930 | "correlate with an agent's success rate rather than with its regret, and so fail to surface exactly the instances an agent can sometimes solve but not reliably" | R1; H4 "exactly" | L | "...rather than with its regret, and so miss the instances an agent can sometimes solve but not reliably," |
| 931-935 | "a curriculum axis that can be set rather than discovered: a generator can place an instance ..., which is a different thing from being shown harder levels in the hope that they are hard in the right way." | R1; S3 colon reveal; R4 closer ("which is a different thing", draft 1 L324) | H | "What the closed-form $\dd$ offers to this problem is a curriculum axis that a generator can set directly, placing an instance at a chosen share-of-tank cost of a wrong representation, so that an agent can be walked up that axis instead of being shown harder levels in the hope that they are hard in the right way." |
| 939-940 | "a policy that plans to adapt after a region it cannot plan through" | D5 repeat of L188-189 aphorism | L | "a policy that bounds its exit from a region it cannot plan through." |
| 943 | "a limitation of the environment and not of the framing" | R1 "X and not Y" | M | "a limitation of the environment rather than of the framing" (one named contrast) or simply "a limitation of the environment". |
| 945-949 | "the relevant question is not whether a policy gradient can be computed inside the region, since the H\"older result says it cannot be computed usefully, but whether a policy can learn the bounding behavior" | R1 "not whether X ... but whether Y" | H | "Since the H\"older result says a policy gradient cannot be computed usefully inside such a region, the relevant question is whether a policy can learn the bounding behavior at the region's entrance and the adaptive behavior at its exit while treating the interior as noise, which is..." |
| 952-955 | "Whether the learned representation survives passage through such a region, or whether the region destroys it, is the question that would decide between a model-free and a model-based approach, and it is open." | R4 closer with "and it is open" kicker | L | "Whether the learned representation survives passage through such a region is the question that would decide between a model-free and a model-based approach, and the present data do not reach it." |
| 957 | "\subsection{How arbitrary the definitions are}" | T4 (heading phrased as an indirect question) | L | "\subsection{Choices in the definitions}" |
| 960-968 | "The equal-pull surface is ... The 150\,s clock is inherited rather than derived ... The banding ... is ... The reference optimizer is ... The twelve families were chosen ..., not derived from any taxonomy" | S9 five consecutive sentences of the same shape (subject, "is", verdict); R1 x2 | M | voice.md 1.8: enumerate inline, "Firstly, the equal-pull surface ...; secondly, the 150 s clock ...; ..." or vary the shape by attaching each verdict to the section that made the choice. |
| 968-971 | "None of this makes the measurements wrong, but it does mean that what the paper offers is a set of intuitions and a set of definitions that make those intuitions checkable, rather than a theory of difficulty for this task." | R1; R4 | L | "These choices do not make the measurements wrong; they mean that the paper offers definitions that make a set of intuitions checkable, not a theory of difficulty." (still one contrast; acceptable) |
| 981-983 | "the ones the measurements call for: a longer clock with hazard exposure rather than time as the binding constraint, levels in which chaos is constructed rather than assumed, and training budgets and seed counts" | R1 x2; D5 fourth "constructed rather than assumed"; R2 (three items, content) | M | "...the ones the measurements call for: a longer clock, levels with constructed chaotic regions (Section~\ref{sec:chaos}), and training budgets and seed counts large enough that..." |

### 5h. What draft 2 fixed, and what the catalogue still catches

Fixed, and verified by count: every "We <verb>" opener, every em dash, every
`\emph`, "mental model", "mission" as a synonym for task, the self-correction
narrative (the one survivor at L574-580 is stated as a property of the
yardstick, as the coordinator says), the `\paragraph{Claim sentence.}`
headings, the First/Second/Third abstract, the caption/body duplication,
and most of the stunted one-line reveals.

Still present, by count of rows above (81 rows; 16 H, 34 M, 31 L):

1. **"rather than" as the replacement tic (28).** The draft moved its
   negative parallelism from "not X but Y" and em dashes into "rather than".
   Where both alternatives are real and named (L646, L720, L916) it is the
   author's own usage; where the rejected alternative was never proposed
   (L70, L75, L85, L253, L509, L552, L983) it is the same rhetorical move
   with a different connective. Target: under ten, each with two named
   alternatives.
2. **"not X but Y" spread across a clause (L184, L505, L586, L943, L945).**
   The single-line grep finds one; the construction survives in the long
   form.
3. **Colon reveals (L100, L169, L290, L367, L429, L505, L601, L641, L842,
   L907, L919, L931).** Twelve; voice.md 6.6 allows colons only before lists,
   equations and specifications.
4. **Surviving draft-1 sentences:** L290 (test particle antithesis), L367
   (one-body imperative triad), L377 ("natural definition"), L528
   (ballistic/exponential), L617 (more time), L746/L832 (teaser headings),
   L768 (reason visible), L819 (narrow conclusion), L844 ("confident"),
   L854 ("restated in pictures"), L900 (STYLE's triad), L931 ("a different
   thing"). Each was flagged in section 4 and carried over with the
   connective changed.
5. **Repetition of whole arguments (D5):** two-repairs chiasmus at L226 and
   L500; Voyager/Parker/Artemis at L108 and L601; "constructed rather than
   assumed" at L73, L509, L552, L944, L983; "adapt after / plan through" at
   L188 and L939. Each argument once, then a section reference.
6. **Headings:** L281, L364, L490, L509, L554, L592, L638, L746, L832, L912,
   L957. voice.md 3: "Section headings are noun phrases that name the
   content."

Not flagged, deliberately: "However,", "Thus,", "Moreover,", "Note that",
"Taken together,", "Firstly/Secondly", "Here, we examine" (L53, L130),
"the natural expectation" (L750; STYLE's own phrase), the "Consider the
most naive possible procedure" worked case (L240; voice.md 1.7), the
parenthetical definitions throughout, the six-reason and three-picture
enumerations as content, and the 100-word sentences, which are the
author's length.
