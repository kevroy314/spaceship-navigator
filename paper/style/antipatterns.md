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


---

## 6. Audit of the complete draft

Line numbers from `cat -n` on the four files at commit `c33e6ec` (2026-10-31):
`paper/spacenav.tex` (1320 lines) and the three fragments it inputs,
`paper/appendix_obs.tex` (184), `paper/appendix_goals.tex` (124) and
`paper/appendix_repro.tex` (115). The whole text was read end to end; the
tables below cover the parts section 5 did not see (the recurrence argument
and the zero-thrust measurement in the clock section, the 600 s training
result, "Training budget" and Table 4, "Relations between the measures" and
its figure, Appendix B, and the three appendices) and re-check the rest.
Allowances are those of section 5: sentence-initial "However,", "Thus,",
"Moreover,", "Note that", "Taken together,", "Firstly, ... Secondly,",
"Here, we examine", long sentences with parenthetical definitions, inline
"1) ... 2)" lists, "the natural expectation", the "Consider ..." worked case,
"rather than" where both alternatives are named, and colons before an
equation, a specification or an enumeration. Thirty-six of the cited lines
were re-read with `sed -n Np` before the tables were written (6k).

### 6a. Measured counts, draft 1 against draft 2 against the complete draft

Same greps as section 5a, run on flattened text (comment lines dropped,
paragraphs joined, so a sentence start or a "rather than" split across a
line break is counted). The raw per-line grep gives 17 "rather than" in
`spacenav.tex`; the two extra are at L339-340 and L428-429, where the phrase
crosses a line.

| Measure | spacenav.tex (draft 1) | draft2.tex | spacenav.tex (now) | appendix_obs | appendix_goals | appendix_repro |
|---|---|---|---|---|---|---|
| Sentences beginning "We <verb>" | 24 | 0 | 0 | 0 | 0 | 0 |
| Spaced em dashes ` --- ` | 29 | 0 | 0 | 0 | 0 | 0 |
| `\emph{}` | 78 | 0 | 0 | 0 | 0 | 0 |
| "rather than" | 13 | 28 | 19 (L108, 174, 180, 254, 339, 429, 446, 651, 665, 670, 681, 719, 785, 863, 1023, 1037, 1130, 1173, 1186) | 1 (L12) | 1 (L115) | 2 (L86, 107) |
| "mental model" | 3 | 0 | 0 | 0 | 0 | 0 |
| "mission" | 73 | 4 | 4 (L115, 189 "mission designer(s)"; L638 "mission-design source"; L645 "every mission in Section 1", the historical ones; all permitted) | 0 | 0 | 0 |
| "exactly" | 6 | 10 | 6 (L288, 305, 563, 707, 1296, 1307; three are literal and three are filler, rows below) | 0 | 1 (L107, literal) | 0 |

The five surface tells remain at zero. "rather than" fell from 28 to 23 over
the four files and most of the survivors now name both alternatives (L108,
L254, L339, L651, L785, L1037, L1186 are real contrasts); the rhetorical form
survives at L429, L719, L1023 and L1130, listed below. The main residue in the
new third of the text is of a different kind from section 5's: the new
material is cleaner at the sentence level than the old (one colon reveal per
page rather than three, no triads announced) but it carries more numbers,
and several of them disagree with each other or with the appendices (6j).
Those rows are the ones to fix first.

Priority as in section 5: **H** = a reader would flag it on its own, or the
text contradicts itself; M = contributes to density or leaves a number
unsupported; L = fix if touching the sentence.

### 6b. Title and abstract (spacenav.tex L31-84)

| Line | Quoted | Pattern | Pri | Suggested rewrite |
|---|---|---|---|---|
| 32-33 | "Defining, Measuring, and Learning Difficulty in a Game-Scale Gravitational Navigation Environment" | T5 gerund triad, flagged H in section 5b; the comment at L5-6 says the subtitle is kept at the author's request | -- | Author's call; listed so the count is honest, not re-flagged. |
| 66-72 | "the episode clock is argued to be the least informative of the constraints, with the caveat that what a longer clock buys depends on the navigator: a trajectory optimizer converts timeouts into arrivals as the clock lengthens, while a scripted reflex pilot converts them into fuel exhaustion and collisions" | S3 colon before a mirrored pair; S5; and incomplete against the body, which has a third navigator (the policy trained at 600 s times out more, L688-691) and the zero-action measurement (L728-751) | M | "...the least informative of the constraints, though what a longer clock buys depends on the navigator, since between 150 s and 900 s a trajectory optimizer converts timeouts into arrivals (5/32 to 13/32 at 600 s), a scripted reflex pilot converts them into fuel exhaustion and collisions, and a policy trained at 600 s times out more often than one trained at 150 s." |
| 68 | "chaos has to be built into a level deliberately" | V2 "deliberately" | L | "chaos has to be built into a level." |
| 72-78 | "a proximal policy optimization agent trained across 20 runs (5 seeds per condition) ... cost 3.1 to 10.1 percentage points of success" | Inconsistent with Section 5.2 (L892-912): the paper now has 25 runs (20 at 3000 updates, 3 at the 600 s clock, 2 at 10,000 updates), the 10,000-update result is called "the ceiling this paper can report" (L908) and is absent here, and the channel cost is quoted without the caveat at L905-908 that it was measured between under-trained policies | H | "Finally, a proximal policy optimization agent trained for 3000 updates across 20 runs (5 seeds per condition) is used both to evaluate the levels and to be evaluated by them; of four physically meaningful readouts added to its observation, the two that are algebraic combinations of quantities it already observes cost 3.1 to 10.1 percentage points of success depending on the evaluation tier (Welch's $t$ on seed means, $p \le 0.029$ on every tier), while the two that carry new information cost nothing; training the default configuration to 10,000 updates at two seeds gains a further 4.7 to 15.3 points, so the channel costs are differences between under-trained policies, and per-level replay of the trained policies shows how much a single success rate conceals." |

### 6c. Section 1, Navigating an n-body field (L87-211)

| Line | Quoted | Pattern | Pri | Suggested rewrite |
|---|---|---|---|---|
| 145-146 | caption: "and then circle the star to the clock without matching the giant's velocity; it is shown as the failure it is." | R4 aphorism in a caption | M | "...and then circle the star to the clock without matching the giant's velocity, so the flight fails on the third target." |
| 153-154 | caption: "(g) The five ways an episode ends, the one way it does not, and the units." | R2 cadence with a teaser ("the one way it does not") | L | "(g) The five episode endings, the empty tank (which is not one), and the units." |
| 187-188 | "On this view a chaotic region is a noise filter, and the navigator's plan is for what comes after it." | R4 closer; this is section 5c's own suggested wording | L | Keep unless the paragraph is reopened; if it is, "On this view a chaotic region is a noise filter and the plan is made for the far side of it." |
| 209-211 | `\FIG{the three pictures side by side ...}` | Editorial placeholder visible in the PDF of a paper described as complete; and because it is not a float, "Figure 2" in the PDF is `fig:eta`, not the figure the text promises | H | Draw it (three panels, `scripts/fig_pictures.py`) or delete the placeholder; the three pictures are already carried by prose. |
| 229 | "which is what a level designer ultimately wants" | H4 "ultimately" | L | "which is what a level designer wants" |

### 6d. Section 2, Why a level of a given difficulty is hard to design (L214-280)

| Line | Quoted | Pattern | Pri | Suggested rewrite |
|---|---|---|---|---|
| 214 | `\section{Why a level of a given difficulty is hard to design}` | T2/T4 heading as an indirect clause (voice.md 3: headings are noun phrases); section 5d let it pass | M | "Designing a level to a difficulty" |
| 235 | `\subsection{What it would take to check an answer}` | T4 teaser heading, same form | M | "The cost of checking solvability" |
| 274-275 | "The same literature offers the consolation on which this paper's method rests" | V metaphor ("consolation") | L | "The same literature supplies the property on which this paper's method rests, which is that ensemble statistics..." |

### 6e. Section 3, Heuristics (L283-774)

| Line | Quoted | Pattern | Pri | Suggested rewrite |
|---|---|---|---|---|
| 286-288 | "and the reader is asked to hold them as such" | R8 address to the reader | L | Delete the clause. |
| 288-290 | "What follows is the environment in enough detail to define them, then the quantities, then the properties of the field that bear on whether they mean anything." | D2 roadmap sentence; "then, then" cadence | L | Delete, or "The environment is described first, in enough detail to define them." |
| 365 | Table 1 cell: "the hazard, not the gravity, shapes the route" | R1 "X, not Y" in a table | M | "extended debris zones that shape the route; the gravity does not" |
| 370 | Table 1 cell: "a luminous body whose radiation, not its gravity, is the constraint" | R1 | M | "a luminous body whose radiation is the constraint" |
| 394-395 | "$k = 0$ the truth; bodies always move truthfully" | V5/V7: "truth", "truthfully" for the full field | L | "$k = 0$ the full field; bodies always move under the full field" |
| 423 | "thousands of candidate levels can be scored for free" | H4 informal | L | "scored without simulation" |
| 431-435 | "An ordinary level in this environment sits at $\dd \approx 0.0014$, with $\eta = 0.0036$ and $A = 3.8$" | Number without its n or source (the figures are in `docs/decisions/0004-model-demand.md`), and the three do not satisfy the definition: $0.0036/3.8 = 0.00095$, not $0.0014$, so they are presumably separate medians and the median of a ratio is not the ratio of the medians; the paper does not say | H | "A typical level (medians over the $n = $ [N] levels of the training pool, \code{docs/decisions/0004}) has $\eta = 0.0036$ and $A = 3.8$, and a median $\dd$ of $0.0014$ (the median of the ratio, which exceeds the ratio of the medians because $\dd$ is heavy-tailed), which is to say that..." |
| 436-443 | "was checked by comparing a planner restricted to the one-body representation against one with the true field on levels pooled by $\dd$ band (24 levels per band, two bands)" | Number without its script (voice.md 4: a measurement names its script) | L | Append "(\code{scripts/[validation script]}, 48 levels)". |
| 450 | "placing targets in the Wisdom band raises $\eta$ only from $0.0026$ to $0.0172$" | Term used before definition ("Wisdom band" is never defined; `wisdom1980` is cited two lines earlier for resonance overlap, not for the band); number without n or script | M | "placing targets in the resonance-overlap band around a secondary (the band whose width Wisdom \cite{wisdom1980} gave as the $2/7$ power of the mass ratio) raises $\eta$ only from $0.0026$ to $0.0172$ ($n = $ [N] levels, \code{scripts/[probe]})" |
| 471 | "thin shells that must be aimed at deliberately" | V2 "deliberately" | L | "thin shells that must be aimed at" |
| 496-497 | "The outcome row does not separate them, and the reason is instructive: a single-burn decision space is shaped by the task" | R6 labelled device; S3 colon reveal; "instructive" is on the section 2 blocklist (added from the section 4 audit) | H | "The outcome row does not separate them, since a single-burn decision space is shaped by the task (the start orbit, the target's motion, the authority $A$) at least as much as by the family, so that the star cluster shown has a clean diagonal winning stripe (...) and the binary star a single narrow band, while two other levels from the same families would show different shapes." |
| 517-518 | caption: "Row (c) is the decision space of one task, not a property of the family." | R1 "X, not Y" in a caption | M | "Row (c) is the decision space of the one task shown and differs between tasks of the same family." |
| 535-536 | "Every level is banded on this axis as well as on $\dd$, and the two are never combined" | Claim without its measurement: no figure, table or script in the paper shows the trivial-difficulty banding (draft 1's bands figure is gone; `--which ... bands` survives in Appendix D L49) | M | Either restore the banding figure or "The generator bands every level on this axis as well as on $\dd$ (\code{scripts/[probe]}; the bands are not reported here), and the two are never combined". |
| 574-575 | "the environment is well inside the benign regime" | H4: draft 1's "comfortably inside" (section 4f) with the adverb swapped | L | "the environment is inside the regime in which the objective is not fractal, by about a factor of two in $\lambda$" |
| 588-589 | "chiefly by episodes long relative to orbital periods (closer encounters and tighter binaries help less)" | Claim without a measurement; "help less" is asserted, and the only related number (L450-452) is for $\eta$, not $\lambda$ | M | "chiefly, on the timescale argument above, by episodes long relative to orbital periods \MEASURE{$\lambda$ on levels with tighter binaries and closer encounters}" |
| 608 | "this is a design decision, not a result" | R1 "X, not Y" | M | "this is a design decision, and it is the nearest relative in this paper of..." |
| 636-637 | "time is the least interesting, and it is also the one most easily mistaken for difficulty. This is a position, not a finding, since" | R1; S2 short sentence as a thesis | M | "That ranking is a position, since deadlines and launch windows are real constraints in every mission-design source cited above, but it is the position the measurement below supports." |
| 645-646 | "which is the choice every mission in Section~\ref{sec:field} was praised for making" | Overstates Section 1, which says the missions were "judged on fuel, exposure and arrival" (L107-108), not praised | L | "which is the choice each of the historical missions in Section~\ref{sec:field} made." |
| 662-663 | "The same experiment on a weaker navigator gives the opposite reading, and the contrast is the point." | R6 labelling the device; R4 | H | "The same experiment on a weaker navigator gives the opposite reading." |
| 671-674 | "A longer clock, that is, gives a navigator the time to finish a path it could not otherwise finish, or the time to exhaust its tank and fall into a body, and which of the two it gets depends on the navigator and not on the clock." | S5 mirrored A/B; R1 "and not"; R4 closer | H | "Thus, a longer clock gives the optimizer time to finish paths it could not otherwise finish and gives the reflex pilot time to exhaust its tank and fall into a body, so that what the clock buys is a property of the navigator." |
| 688-691 | "Note that the dominant ending for the policies that did learn was still the clock: 37--42\% of medium-suite tasks timed out at 600\,s, against roughly a third at 150\,s, so that four times the time produced more timeouts, not fewer." | R1 kicker "not fewer"; "roughly a third" has no n (the figure is in `docs/decisions/0035-budget-not-clock.md`) | M | "Note that the dominant ending for the two policies that did learn was still the clock, with 37--42\% of medium-suite tasks timing out at 600\,s against [33]\% ($n = 594$, \code{docs/decisions/0035}) at 150\,s for the same configuration, so that four times the clock produced more timeouts." |
| 695-696 | "a policy trained at the longer clock is rewarded for progress it cannot see the end of and has every reason to drift" | V7 colloquial ("has every reason") | L | "...is rewarded for progress it cannot see the end of, and drifting is not penalized." |
| 699-701 | "a longer clock is not a free parameter for a learner either, and that the sense in which 150\,s was arbitrary (Section~\ref{sec:discussion}) does not make 600\,s correct." | R4 chiastic closer | M | "...a longer clock is not a free parameter for a learner either; 150\,s was inherited (Section~\ref{sec:discussion}) and 600\,s has not been derived." |
| 703-705 | "There is a reason the clock cannot simply be set long enough to settle the question, and it is worth stating because it bears on what solvability means. Consider a navigator that does nothing at all." | R5 telegraph; H1 "worth"; H4 "simply", "at all" | H | "The clock cannot be set long enough to settle the question, for a reason that bears on what solvability means. Consider a navigator that does nothing." |
| 707 | "in the sense Poincar\'e established for exactly this problem" | H4 "exactly" | L | "in the sense Poincar\'e established for this problem" |
| 713-718 | "Thus, for a flythrough tour, the zero-action policy succeeds with probability approaching one as the clock lengthens ... (in the limit of a system collapsing gravitationally to a point, every path is that path)." | The recurrence claim is stated without its conditions (CLAUDE.md mandate 3): the craft's own Hamiltonian is time-dependent, so recurrence applies to the autonomous bodies-plus-craft system on a bounded energy surface, with respect to Liouville measure rather than the task's start distribution, and "succeeds" for a tour means arriving in order; the measurement that follows shows contact flat by 9,000 s because the craft is dead (L742-745). The parenthetical is a flourish. | M | "Thus, if the bodies-plus-craft system stays bounded, a coasting craft returns arbitrarily near its starting state (the recurrence is for the autonomous system on a bounded energy surface and with respect to Liouville measure, not the task's start distribution), so that for a flythrough tour the zero-action policy's contact probability has no bound other than the one the hazards set, and the only stopping criterion that is not a choice is divergence, i.e. the craft leaving the system." Delete the parenthetical about gravitational collapse. |
| 739-741 | "a pool of 900 levels was validated to 9{,}000\,s (ten episodes; 80--100\% of generated levels survive that long, so bounded systems from these generators stay bounded)" | "ten episodes" is wrong at either clock (sixty at 150 s, fifteen at 600 s); it is ten validation horizons. "stay bounded" overreads 80--100\% survival | M | "(ten times the 900\,s validation horizon, or sixty 150\,s episodes; 80--100\% of generated levels survive that long, so most levels from these generators stay bounded over it)" |
| 745-751 | "Thus, patience alone buys a coasting craft three to five points of contact ... and the race between the improbable path and exposure is won by exposure; in this environment the stopping criterion that recurrence fails to supply is supplied instead by the hazards, and the constraint the author ranks most interesting is also the one that bounds the clock." | R4 aphoristic triple closer; metaphor ("race ... won"); self-reference ("the author ranks"); R7-adjacent | H | "Thus, coasting gains three to five points of contact between 150\,s and 900\,s and another three to six between 900\,s and 9{,}000\,s, while the fraction of craft destroyed over the same intervals rises to 26--27\% and then to 78--80\%, so that in this environment the stopping criterion recurrence does not supply is supplied by the hazards." |
| 756 | caption: "A craft that never thrusts." | Headline fragment opening a caption | L | "Contact and death curves for a craft that never thrusts." |
| 763 | caption: "the two rows are same-specification, not paired." | R1 in a caption | L | "the two rows are drawn to the same suite specifications on different levels." |
| 753-774 | `fig:noop` is placed before `fig:clock` in the source, but `fig:clock` is referenced first (L652 against L729), so the PDF cites Figure 6 before Figure 5 | Figure numbered out of reference order | M | Move the `fig:clock` float above `fig:noop`. |

### 6f. Sections 4-5, The learning agent and Results (L777-1151)

| Line | Quoted | Pattern | Pri | Suggested rewrite |
|---|---|---|---|---|
| 800 | "$\gamma = 0.999$; $\lambda = 0.95$; clip $0.2$" | Symbol collision: $\lambda$ is the maximal Lyapunov exponent from L548 onward (and $\lambda_{\mathrm{crit}}$ at L552) | M | "the advantage-estimation parameter $\lambda_{\mathrm{GAE}} = 0.95$" |
| 803-804 | "which on the hardware used takes about 67 minutes" | Disagrees with Appendix D L89-96, where four of the twenty runs took 6.4 h on the other card | L | "which takes about 67 minutes on the faster of the two cards used (Appendix~\ref{app:repro})" |
| 838-840 | Table 2 caption: "(a station or a target carried by a body, drawn with probability 0.45--0.6 per target)" | Disagrees with Appendix C L14 ("\code{p\_station}, 0.35 by default and set per suite") unless the reader is told the suites set it | L | "(a station or a target carried by a body, drawn with a per-suite probability of 0.45--0.6; the training default is 0.35, Appendix~\ref{app:goals})" |
| 852-853 | "the best configuration found (the policy told $\eta$ and the stretching rate, for reasons given below)" | "best" is nominal: Table 5 gives $p = 0.40$--$0.90$ against the no-channel arm; and the paper uses three different agents as "the agent" (see 6j item 3) | H | "the default configuration (the policy told $\eta$ and the stretching rate, which is statistically indistinguishable from the no-channel arm, Table~\ref{tab:instr})", and state at each later use which arm is flown and why. |
| 861 | "on a comfortable-tank instance set it arrives 63.3\% of the time" | Number without n or script (carried over from draft 2) | L | "on a comfortable-tank instance set ($n = $ [N], \code{scripts/[pilot eval]})" |
| 883-885 | caption: `\TODO{the first draft's table used the no-channel arm ...; confirm which the author prefers as "the" result}` | Editorial placeholder in a table caption of the complete draft | H | Resolve (6j item 3) and delete. |
| 867-887, 914-934 | Table 3 and Table 4 | D6: Table 3 (3000-update row, pilot row) is a strict subset of Table 4 (the same two rows plus the 10,000-update row) | M | One table of three rows, or make Table 3 the no-channel arm as the TODO proposes. |
| 900-901 | "rises from $47.8\%$ at update 3000 through $52.4\%$ at 6000 to $57.7\%$ at 10,000" | Two values for one quantity: Table 4 gives $57.2 \pm 2.5$ for the 10,000-update medium result (mean of the last three checkpoints), the text $57.7$ (final checkpoint), without saying so | L | "to $57.7\%$ at the final checkpoint ($57.2\%$ as the mean of the last three, Table~\ref{tab:budget})" |
| 905-908 | "the direction of an effect measured there is likely to survive more training, as the budget runs show the same ordering across tiers, but its size should not be read as a property of the task" | The factor does not bear on the mechanism (CLAUDE.md mandate 3): the ordering of the four tiers under one configuration says nothing about the contrast between arms, and only the $\eta$,stretch arm was trained to 10,000 updates | H | "whether the direction of an effect measured there survives more training has not been tested, since only the $\eta$,stretch arm was trained to 10,000 updates \MEASURE{the all-four arm at 10,000 updates, two seeds}, and its size should not be read as a property of the task." |
| 908-912 | "The 10,000-update result is the ceiling this paper can report, and the one against which the reference pilot of Table~\ref{tab:main} should be compared: the learned policy then exceeds the pilot by $9.8$, ..." | S3 colon reveal; and the abstract does not carry this comparison (6b) | M | "...should be compared, since the learned policy then exceeds the pilot by $9.8$, $22.5$, $11.3$ and $9.1$ points on the four tiers, including the holdout tier on which the 3000-update policy fell short of it." |
| 947 | "toggling them was verified not to perturb the dynamics" | Number-free claim without its script (carried over) | L | "(\code{tests/[test name]})" |
| 967-969 | "(it is also a good proxy for the quantity one would actually want, since on the four levels of Figure~\ref{fig:phase} its log--log correlation with the measured 20\,s Lyapunov exponent is $0.96$--$0.98$)" | "the quantity one would actually want" is vague; $n = 4$ levels | L | "(on the four levels of Figure~\ref{fig:phase} its log--log correlation over the playable area with the 20\,s finite-time Lyapunov exponent is $0.96$--$0.98$, so it stands in for the local divergence rate)" |
| 970 | "puts the entire cost on the redundant half" | V2 "entire" | L | "puts the cost on the redundant half" |
| 1017-1021 | "Four extra input dimensions against 285,322 parameters is a capacity account, a ratio with a division in it is heavy-tailed even after log compression, which is a conditioning account, and the two predict different behavior under scale" | S5 parallel "X account / Y account" (draft 1's "X story; Y story", section 4h, with the noun swapped); and the first clause argues against capacity pressure, not for it | M | "The cost could be a capacity effect, though four extra inputs against 285,322 parameters makes that an unlikely one, or a conditioning effect, since a ratio with a division in it is heavy-tailed even after log compression; the two predict different behavior as the network widens \MEASURE{...}." |
| 1022-1023 | "an observation channel's value to a learner depends on whether it is new rather than on whether it is true" | R1: draft 1 L851 "not on whether the information is true but on whether it is new" (section 4i) with the connective swapped | M | "an observation channel's value to a learner depends on whether it adds information the observation does not already determine, and a true quantity that is already determined costs capacity without buying anything" |
| 1049-1050 | "the channel effect of Section~\ref{sec:instruments} is seven to ten points" | Contradicts L949 and Table 5: the all-four cost is $3.1$, $10.1$, $4.9$ and $8.0$ points on the four tiers | H | "since the channel effect of Section~\ref{sec:instruments} is three to ten points" |
| 1044, 1047-1048, 1064-1066 | "the easy-suite instance shown arrives 52\%"; "(60\%, 52\%, 8\% and 0\% on the four shown)"; caption names medium task 1, medium task 0 and hard task 0 only | The fourth (easy) instance is unnamed in the caption and the four rates are not assigned to instances; if the order is easy, medium 1, medium 0, hard 0 then medium task 1 (60\%) outperforms the easy instance (52\%), which the text should say | L | Caption: "Easy task [k] (52\% sampled success), medium task 1 (60\%), medium task 0 (8\%) and hard task 0 (0\%)." |
| 1078-1079 | caption: "one seed of each headline configuration" | Configurations unnamed | L | "(no channels, $\eta$,stretch, $A$,$\dd$ and all four; seed 0)" |
| 1086-1088 | "the test of that is whether they predict who arrives" | V7 colloquial "who" | L | "whether they predict which instances are completed" |
| 1089-1090 | "the fraction of four 3000-update policies (two no-channel seeds and two all-channel seeds) that arrived" | Mixes the two arms Table 5 separates by $3$--$10$ points; `fig:corr` caption (L1144) says only "mean of four 3000-update runs" | M | Use one arm (five seeds of the no-channel or $\eta$,stretch arm), or state why the mixture is harmless ("the two arms rank instances identically, $\rho = $ [x]"), and name the arms in the caption. |
| 1102-1104 | "Thus, $\dd$ predicts success better than family alone and better than the suite's own design levers, as it was meant to." | R4 tag "as it was meant to" | L | End at "design levers." |
| 1106 | "However, it does so for the wrong half of itself." | S2 one-line reveal; metaphor | H | "However, the prediction is carried by the authority $A$ rather than by $\eta$, since $A$ alone is the stronger predictor ($\rho = +0.754$; held-out area $0.946$ with the design levers) while $\eta$ at the targets is weak ($\rho = -0.248$) and, once $A$ is in the model, carries no information (deviance drop $0.2$ on one degree of freedom, $p = 0.67$; coefficient $-0.02$ against $+2.09$ for $A$)." |
| 1108, 1146-1147 | "$\eta$ at the targets" (text) against "the maximum $\eta$ over a task's targets" (caption) | Two descriptions of the per-instance $\eta$ | L | Use "the maximum $\eta$ over a task's targets" in both. |
| 1109-1110 | "carries no information at all" | H4 | L | "carries no information" |
| 1115-1116 | "where the field is simple and the route is not" | S5 antithesis | M | "where the field is simple and the hazards shape the route" |
| 1116-1120 | "What the learned policy's success tracks at this training budget is the tank against the tour, i.e. whether a direct flight is affordable, and not whether a one-body representation of the field is wrong; the representational component of difficulty that the dial was built to capture is not yet the component that limits the learner." | R1 "and not"; V5 "the dial" ($\dd$ is called a dial nowhere else) | H | "At this training budget the learned policy's success tracks the tank against the tour, i.e. whether a direct flight is affordable, and the representational component that $\dd$ was built to capture is not yet the component that limits the learner." |
| 1124-1126 | "is a question the 10,000-update policies could answer and the present ones cannot" | S5 mirrored | L | "is a question for the 10,000-update policies \MEASURE{$\eta$ against per-instance success of the 10,000-update runs}" |
| 1126-1127 | "the picture is the same from the other side: a family's median $\dd$ predicts its success rate" | S3 colon reveal | M | "the picture is the same, since a family's median $\dd$ predicts its success rate ($\rho = -0.734$ over the twelve families, $p = 0.007$) and the families that break the ordering are the ones whose difficulty is hazard rather than field." |
| 1132-1133 | "The expectation that a longer clock matters most on multi-target tasks does not survive normalization." | S2 thesis opener; the paragraph is also a clock measurement filed under "Relations between the measures" | M | Merge with the next sentence ("...does not survive normalization, since on the 600\,s suites a coasting craft...") and move the paragraph to Section~\ref{sec:clock} after the zero-action measurement. |
| 1137-1138 | "more targets are more chances, and the tour structure adds nothing" | R4 closer | M | "so that the multi-target excess is the number of chances and not anything about the tour" is the same tic; use "so that the multi-target excess is accounted for by the number of chances." |

### 6g. Discussion, Future directions, Acknowledgements (L1154-1257)

| Line | Quoted | Pattern | Pri | Suggested rewrite |
|---|---|---|---|---|
| 1164-1167 | "the policy being strong on the hierarchical families, weak on the contested ones and worst on the held-out families where the representation learned elsewhere does not transfer" | Contradicted by Section 5.4 L1113-1116, where four hierarchical families (\code{debris\_maze} 25\%, \code{asteroid\_field} 28\%, \code{compact\_system} 31\%, \code{pulsar} 39\%; all "hierarchical" in Table 1) are among the worst; also R2 strong/weak/worst | H | "and the results of Section~\ref{sec:results} are consistent with that account only in part, since the policy is strong on the hierarchical families whose routes are not shaped by hazards, weak on the contested ones, and weakest on the hazard families and the held-out families, where what limits it is the route or the transfer and not the field (Section~\ref{sec:correlations})." |
| 1180-1181 | "which is to say that the agent learned a one-body representation first and was then asked to extend it" | An inference stated as a fact: nothing in the paper measures what representation was learned | M | "which is to say that the agent was trained on levels where a one-body representation suffices before being shown levels where it does not" |
| 1199-1202 | "which limits the environment and not the picture, and the route to building them is known (Section~\ref{sec:chaos})" | R1 "and not" (section 5g L943 survived with the words changed); "known" overstates a timescale argument with no measurement | M | "which is a limitation of the environment; the route to building them (Section~\ref{sec:chaos}) has not been tried." |
| 1209-1210 | "the gradient through a chaotic system is correct and useless at once" | S5 aphorism | M | "the gradient through a chaotic system is exact and carries no usable descent direction" |
| 1217-1218 | "Several of the definitions in this paper are choices, and the reader should know which." | R8 address to the reader; R5 | M | "Several of the definitions in this paper are choices." |
| 1225-1226 | "the regimes the authors could think of" | Single author (L35); "the author" at L750 | L | "the regimes that came to mind" |
| 1232-1236 | "The value of better definitions here is that they let the environment and the agents that run over it be understood together, with the same quantities describing what a level demands and what a learner has managed, and that is a more useful thing to have than either description alone." | R4 closer | M | End at "...what a learner has managed." |
| 1247-1248 | "one craft may be attempting to catch, target, or otherwise react to another" | R2 cadence | L | "one craft may be attempting to catch or evade another" |

### 6h. Appendix B, Theoretical notes on observation (spacenav.tex L1263-1311)

| Line | Quoted | Pattern | Pri | Suggested rewrite |
|---|---|---|---|---|
| 1268-1271 | "Two departures from this are worth setting down, because each would change what a learner could and could not infer, and because the environment's construction makes one of them unusually clean to study." | H1 "worth"; R3 announced pair | M | "Two departures from this would each change what a learner could infer, and the environment's construction makes the second unusually clean to study." |
| 1275 | "at game scale this is not negligible: with a signal speed chosen so that" | S3 colon reveal | L | "and at game scale this is not negligible, since with a signal speed chosen so that..." |
| 1292-1294 | "and here the environment's design has a consequence that is easy to overlook." | R5 telegraph | M | Delete the clause. |
| 1301-1303 | "In a real setting the two differ: a measured position carries sensor noise, a projected one carries model error that grows with the projection horizon, and a navigator must weigh one against the other." | S3; S5 mirrored | L | "In a real setting a measured position carries sensor noise while a projected one carries model error that grows with the horizon, and a navigator must weigh the two." |
| 1304-1305 | "which is a simplification to be aware of when reading the results" | R8 | L | Delete the clause. |
| 1307 | "since $\eta$ is exactly a measure of how wrong" | H4 | L | "since $\eta$ is a measure of how wrong" |
| 1263-1311 | the whole appendix | D3: two untested proposals with no measurement, figure or number from the repository; it reads as a future-work essay placed among the data appendices | L | Coordinator's call: fold into Future directions as two sentences, or keep and say in its first sentence that it proposes two experiments the environment supports and none was run. |

### 6i. Appendices A, C and D (the fragment files)

| File:line | Quoted | Pattern | Pri | Suggested rewrite |
|---|---|---|---|---|
| obs:28 | "``the target'' is the waypoint currently being flown to" | V6 quotes on the paper's own term (voice.md 7) | L | "the target (the waypoint currently being flown to, which in a fixed-order tour ...)" |
| obs:169 | "with a success rate that simply looks wrong" | H4 "simply" | L | "with a success rate that looks wrong" |
| goals:14 | "with probability \code{p\_station}, 0.35 by default and set per suite" | See 6f L838-840; the two statements of the station probability should point at each other | L | "(0.35 in training; 0.45--0.6 on the evaluation suites, Table~\ref{tab:suites})" |
| goals:70-72 | "so an order-free tour is neither easier nor harder by construction; it is different in that the navigator, and not the instance, chooses the sequence." | R1 "and not" | M | "so an order-free tour is neither easier nor harder by construction, and differs only in that the navigator chooses the sequence." |
| goals:91-93 | "the budget multiple $\beta$ is drawn uniformly from $[0.55, 1.35]$ (\code{DELTAV\_BUDGET\_RANGE}, narrowed per suite as Table~\ref{tab:suites} lists)" | The easy suite's $1.0$--$1.5\times$ (Table 2 L829; `spacenav/rl/evaluate.py:54` sets `deltav_budget=(1.0, 1.5)`) lies outside that range, so the suites set $\beta$ rather than narrow it | M | "drawn uniformly from $[0.55, 1.35]$ in training (\code{DELTAV\_BUDGET\_RANGE}); each evaluation suite sets its own range (Table~\ref{tab:suites}), and the easy suite's $1.0$--$1.5\times$ extends past the training range" |
| goals:117-118 | "An empty tank is not among them." | Contradicted by the body, which counts "fuel exhaustions" as endings for the optimizer sweep (L654-655), the reflex pilot at 600 s (L669-671, "fuel exhaustions 53 to 105") and the pilot's failure breakdown (L862-863); `spacenav/env.py:483` selects a status over crashed, health, finished and lost only, and `docs/decisions/0032` reports the sweep with a column "crashed / destroyed / out of fuel" | H | In Section~\ref{sec:clock}, at first use: "an episode that ends by any code with the tank empty is counted here as a fuel exhaustion (the environment itself has no such ending, Appendix~\ref{app:goals})", and keep the appendix sentence. |
| repro:8-10 | "Everything reported in this paper is produced by scripts in the repository ... in the order below" | The listing omits \code{scripts/noop\_clock.py} (`fig:noop`), \code{scripts/correlations.py} (`fig:corr`), the 10,000-update runs (Table 4), the reflex pilot on the 600\,s suites (L666-671) and the 9,000\,s pool (L739; \code{data/pools/long10x.npz}); and `--which ... bands` (L49) draws a figure that is no longer in the paper | H | Add the five commands; drop `bands`. |
| repro:11-15 | "from $3 \times 10^7$ and $4 \times 10^7$ for the two long-horizon pools" | The body names one long-horizon pool of 1,800 levels (L346-351) and a separate 900-level pool validated to 9,000\,s (L739); \code{data/pools/} holds \code{long}, \code{long\_val} and \code{long10x} | M | "from $3 \times 10^7$ and $4 \times 10^7$ for the two 900\,s long-horizon pools (training and suites) and from [seed] for the 9{,}000\,s pool" |
| repro:21-22 | `--train 2000 --val 300 --holdout 400`, `--only long --n 200` | Per-family counts against the body's totals (18,000 / 2,700 / 1,200 / 1,800, L346-349); the paper never says the flags are per family | L | Comment: "# counts are per family: nine training families, three holdout families" |
| repro:77-82 | "Python 3.12 against JAX 0.9.2 ..., Flax 0.12.6 ..., Optax 0.2.8; ... NumPy 2.4.3 and SciPy 1.17.1, ... matplotlib 3.11.2" | voice.md 4: methods cite the libraries | L | Add `\cite` or `[CITE]` for JAX, Flax, Optax, NumPy, SciPy and matplotlib. |
| repro:83-87 | "the replays on the companion site" | The site is not introduced anywhere in the paper (L306 says "a browser client") | L | "the replays on the companion site (\url{...})" |
| repro:98-99 | "no difference attributable to the card was found" | Claim without its statistic ($n = 2$ against $3$ seeds) | L | "no difference larger than the seed spread (easy-suite seed means [x] and [y])" |

### 6j. Internal inconsistencies

Numbered in the order the coordinator should take them; the first four are
the ones a reader stops on.

1. **An empty tank is and is not an ending.** Section 3.1 (L320-321) and
   Appendix C (goals:117-118) say an empty tank does not end an episode and
   the status set is five codes; Section 3.6 counts "fuel exhaustions" among
   the endings for the optimizer sweep (L654-655), the reflex pilot at 600 s
   (L669-671, "fuel exhaustions 53 to 105") and the pilot's failure breakdown
   (L862-863, "fuel exhaustion (4.7\%)"). `spacenav/env.py:483` has no fuel
   code; the category is a post-hoc classification (an ending with the tank
   empty, presumably) that the paper never defines.
2. **The abstract reports the under-trained numbers without saying so.**
   L72-78 gives "20 runs", the 3000-update pilot comparison and the 3.1--10.1
   point channel cost; L892-912 says every such comparison is between
   under-trained policies, that the 10,000-update result (two further seeds)
   is "the ceiling this paper can report" and the one the pilot "should be
   compared" against, and that the margins over the pilot are then
   9.8--22.5 points. None of that reaches the abstract.
3. **Three different agents are "the agent".** Tables 3 and 4 report the
   $\eta$,stretch arm as "the best configuration" (L852), though Table 5 puts
   it at $p = 0.40$--$0.90$ against the no-channel arm; `fig:task` and the
   ghost fleets are flown by \code{ppo\_v5\_noinstr}, the no-channel arm
   (repro:45, 50; L1044 "89\% for the greedy policy on that suite" is the
   no-channel 89.4); the correlations average two no-channel and two
   all-channel seeds (L1089-1090), the arms the paper separates by 3--10
   points. The TODO at L883-885 is this question, unresolved.
4. **"Seven to ten points" (L1049-1050) against 3.1--10.1 (L949, Table 5).**
   The per-tier all-four cost is 3.1, 10.1, 4.9 and 8.0.
5. **The Discussion's family story (L1164-1167) against Section 5.4
   (L1113-1116).** "Strong on the hierarchical families" is false for four of
   the six hierarchical families in Table 1, which Section 5.4 names as the
   low-$\eta$ failures at 25--39\%.
6. **$\dd \approx 0.0014$ with $\eta = 0.0036$ and $A = 3.8$ (L431-432).**
   $0.0036/3.8 = 0.00095$. If the three are separate medians the text should
   say so; no $n$ or script is given (the source is ADR 0004).
7. **How many long-horizon pools.** Body: 1,800 levels validated to 900 s
   (L349) and 900 levels validated to 9,000 s (L739). Appendix D: "the two
   long-horizon pools" (repro:11-15) and one build command (repro:22).
   On disk: \code{long}, \code{long\_val}, \code{long10x}. And "ten episodes"
   (L740) for 9,000 s is sixty at 150 s or fifteen at 600 s.
8. **The budget range.** Appendix C says $\beta \in [0.55, 1.35]$ "narrowed
   per suite" (goals:91-93); Table 2 gives the easy suite $1.0$--$1.5\times$
   (L829), and `spacenav/rl/evaluate.py:54` sets it directly.
9. **The station probability.** Table 2 caption: 0.45--0.6 per target
   (L839-840); Appendix C: 0.35 by default (goals:14). Compatible, but
   neither points at the other.
10. **$\lambda$ is two things**: the maximal Lyapunov exponent (L548, 552,
    566) and the advantage-estimation parameter (L800).
11. **Figure order.** `fig:noop` (source L753) precedes `fig:clock` (L767)
    while `fig:clock` is referenced first (L652 against L729), so Figure 6 is
    cited before Figure 5; and the `\FIG` placeholder at L209 means the PDF's
    Figure 2 is `fig:eta`, not the three-pictures figure the text promises.
12. **The reproduction appendix is not complete** (repro:8-10 against the
    figures and tables; 6i row repro:8-10): five of the paper's results have
    no command, and one listed figure (`bands`) is not in the paper.
13. **Trivial-difficulty banding (L535-536)** is asserted for every level and
    shown nowhere; the abstract (L59-60) says the reaction-time component is
    not treated.
14. **57.7\% (L901) against $57.2$ (Table 4)** for the 10,000-update medium
    result: final checkpoint against mean of the last three, unsaid.
15. **67 minutes (L803-804) against 6.4 h** for four of the twenty runs
    (repro:89-93).
16. **"the author" (L750), "the authors" (L1226), one author (L35).**
17. **The ghost figure's fourth instance.** The text's "easy-suite instance"
    (L1044) and the four rates (L1047-1048) are not matched to the three
    instances the caption names (L1064-1066).
18. **Per-instance $\eta$**: "at the targets" (L1108) against "the maximum
    $\eta$ over a task's targets" (L1146-1147).
19. **"Praised" (L645-646) against "judged" (L107-108)** for the historical
    missions.
20. **Placeholders in a complete draft**: `\FIG` at L209, `\TODO` at
    L883-885, `\MEASURE` at L697-698 and L1021-1022. The two `\MEASURE`s are
    what STYLE asks for; the `\FIG` and `\TODO` are not.

One repository note, not a paper inconsistency: CLAUDE.md's trap table says
the instrument default "is now off (ADR 0031)"; `spacenav/env.py:73` defaults
\code{SPACENAV\_INSTRUMENTS} to \code{eta,stretch}, and the paper (obs:161-163,
repro:68) and ADR 0033 agree with the code. The CLAUDE.md row is the stale
one.

### 6k. Spot-checks

Thirty-six cited lines re-read with `sed -n Np` at `c33e6ec`; every one
matched the quotation in its row. spacenav.tex: 68 ("built into a level
deliberately"), 145 ("the failure it"), 229 ("ultimately wants"), 365
("the hazard, no[t]"), 394 ("$k = 0$ the truth"), 423 ("scored for free"),
431 ("$\dd \approx 0.0014$"), 450 ("Wisdom band"), 497 ("the reason is
instructive:"), 518 ("not a property of the family"), 535 ("Every level is
banded"), 608 ("not a result"), 637 ("a position, not a finding"), 663
("the contrast is the point"), 674 ("and not on the clock"), 691 ("not
fewer"), 704 ("worth stating"), 740 ("ten episodes"), 750 ("the author
ranks"), 800 ("$\lambda = 0.95$"), 884 (the TODO), 901 ("$57.7\%$ at
10,000"), 906 ("likely to survive more training"), 1050 ("seven to ten
points"), 1106 ("the wrong half of itself"), 1119 ("the dial"), 1133 ("not
survive normalization"), 1165 ("strong on the hierarchical families"), 1226
("the authors"), 1269 ("worth setting down"), 1293 ("easy to overlook");
appendix_goals.tex: 92 ("narrowed per suite"), 117 ("An empty tank is not
among"); appendix_obs.tex: 169 ("simply looks wrong"); appendix_repro.tex: 8
("Everything reported in this paper"), 12 ("from $10^7$").

### 6l. Summary

One hundred rows: 17 H, 37 M, 45 L, and the subtitle left to the author.
Of the seventeen H, ten are contradictions between numbers or claims (6j
items 1-6 and 12, the unsupported inference at L905-908, and the two
placeholders) rather than prose tics, which is the reverse of section 5's
mix, and it is the reason the first three fixes are not sentence rewrites.
First, settle which agent the paper reports and make the abstract, Tables 3
and 4, the ghost fleets and the correlations agree (6j items 2, 3 and 4, and
the TODO at L883): the abstract should carry the 10,000-update result and
the under-training caveat, and "seven to ten" at L1050 should read "three to
ten". Second, define "fuel exhaustion" where it is first counted (L654) and
reconcile it with the five endings of Appendix C, since as written the
paper's own appendix says the category does not exist. Third, fix the
Discussion's family claim at L1164-1167 against Section 5.4 and the
$\dd$/$\eta$/$A$ medians at L431, since both are places where a reader who
has followed the numbers will stop. After those, the prose rows that matter
most are the five aphoristic closers in the clock section (L663, L671-674,
L703-705, L745-751) and the correlations section (L1106, L1116-1120), which
are the only passages in the new third of the text that still read as
generated.
