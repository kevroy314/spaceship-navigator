# End-to-end requirements review

Every requirement given so far, with an honest status. Reviewed 2026-10-02.

Status key: **done** · **partial** (works, with a named gap) · **designed**
(reviewed and specified, nothing built) · **not started** · **broken**

---

## 1. Gravity environment and level design

| Requirement | Status | Notes |
|---|---|---|
| GPU n-body sim of gravitationally bound systems | **done** | JAX, leapfrog, ~0.5–1M env steps/s batched |
| Traditional and exotic families (jovian/saturnian moons, multi-star, black holes) | **done** | 9 training + 3 held-out families |
| Game-like, not realistic, scales | **done** | G=1, masses are GM, 150 s episodes |
| Ship health: debris, atmosphere skimming, radiation | **done** | Split damage sources in `hazard_terms` |
| Levels that are "interesting" rather than boring | **partial** | Measured two ways (D, control MDL). But see §2 gaps |
| Work backwards from a path to derive the level | **done** | `levels/coastpath.py`; this is Level 0 |
| **Longer episodes, broader decision trees** | **not started** | Newly requested. See [0013](decisions/0013-longer-episodes.md) — this is the keystone, below |

**Gap.** The generator cannot currently produce three of the eight defined
lesson types. Root cause is §2.

## 2. Bootstrapping and curriculum

| Requirement | Status | Notes |
|---|---|---|
| Train across many scenarios; generalise to blind ones | **done** | `val_holdout` is three unseen families |
| Difficulty ramp that tracks interestingness | **done** | D = eta/A rungs, solved for rather than searched |
| A human path plus non-human / impossible branches | **partial** | Bands defined and validated in the pooled test; branches unpopulated |
| Discrete lessons, graduating skill to skill | **done** | `lessons.py`, 6 path lessons + 1 branch + 1 excluded |
| Level 0: "touch nothing and watch" | **done** | 108/108 targets by pure coasting, 6 families |
| Challenge branches escalating to superhuman | **designed** | Signatures written; **zero instances found** |
| Checkpointed versions with monotonic improvement | **broken** | See below |

**Gap 1 — the branches are empty, and we now know why.** `rolling-with-it`,
`one-exact-boost` and `lottery` found no instances. [0008](decisions/0008-chaos-is-aspirational.md):
the game is not chaotic at 150 s. Measured lambda 0.0045–0.0086 1/s, a Lyapunov
time of ~140 s, about one e-folding per episode. There is no chaos to roll with
and no fractal boundary to be defeated by.

**Gap 2 — monotonic improvement is unverifiable.** An early requirement was that
checkpoint versions improve monotonically as an overfitting check. v3 changed the
suites and observation shape; 2026-10-02 changed it again (MAX_WAYPOINTS 4->6,
four instrument channels). Old checkpoints cannot be re-scored, so no version
comparison currently exists. Fix: freeze an evaluation interface with an
observation adapter, or declare explicit version epochs. We have neither.

**Gap 3 — the signatures overlap.** All four coast levels also match
`gravity-assist`, `conics` and `dead-heading`; `free-ride` wins by match order
alone. The taxonomy classifies but does not discriminate.

**Gap 4 — evaluation is underpowered.** Per-rung n = 8 against the n ~ 384 needed
for +-5 pp ([0012](decisions/0012-evaluation-statistics.md)). The pooled
difficulty-curve result (p = 0.005) stands; the per-rung table does not.

## 3. Battle mechanics

| Requirement | Status |
|---|---|
| Bullets: small damage, rapid fire | **not started** |
| Bombs: proximity-triggered blast radius | **not started** |
| Reload by flying to targets | **not started** |
| Agents fighting in a gravitational field while servicing objectives | **not started** |

Correctly gated: this was to follow core navigation balance and training. The
design input that exists is from the multi-agent review — the published
AlphaDogfight work is hierarchical (a selector over shaped low-level policies),
which is nearly isomorphic to the manoeuvre vocabulary in
[0011](decisions/0011-callouts-and-legibility.md), so the two phases share
structure. Self-play needs historical opponent sampling and a league from the
first day or the ships cycle non-transitively and evaluation means nothing.

## 4. Gamification

| Requirement | Status | Notes |
|---|---|---|
| Playable in a browser, mobile-friendly | **done** | Tested at ~400 px |
| Time controls: pause, slow, fast-forward, rewind to revise | **done** | Steering divided by the time multiplier |
| Selectable number of waypoints, mix of moving and stationary | **done** | Mission spec `<stops>w<moving%>m[d][c][l]` |
| Sensors / "advanced flight computer" exposing emergence and causality | **partial** | eta, A, D, stretch on the HUD, gated per lesson |
| Predictability cone showing where forecasting fails | **broken in practice** | Correct code, but it never fires — see below |
| Live training dashboard, no refresh, LAN-visible | **done** | Port 8767, SSE |
| Notifications when work completes | **done** | ntfy (public topic — anyone who guesses the name can read it) |

**Gap.** The cone is the instrument that was supposed to make chaos legible, and
it does not trigger: a one-tick perturbation reaches only 1.2–3.5 u against a
12 u tolerance over 20 s. That is not a rendering bug, it is §2 Gap 1. Longer
episodes fix it.

## 5. Explainability and cooperation

| Requirement | Status | Notes |
|---|---|---|
| A policy that is both planned and explainable | **designed** | Plan at 1–2 Hz, fast policy fills ticks; the plan *is* the explanation |
| Wingmate calling out its next manoeuvre privately | **designed** | Hand-designed vocabulary + intent head + self-supervised consistency |
| Weight sharing as a coordination channel | **reviewed, demoted** | Gives mutual predictability; cannot break symmetry, and provably averages on multi-modal role assignment |
| Ensembles with distinct model biases | **reviewed, redirected** | No evidence as a deployed team; correct use is generating training partners |
| Fixed compute/energy budget per decision | **designed** | Slow deliberator + fast reflex; no LLM in the 15 Hz loop |
| Human-agent collaboration | **designed** | Legible trajectories preferred over text callouts |

**Nothing here is built.** The reviews changed the design substantially (don't
learn the message; faithfulness by construction; legibility may beat messaging),
which is worth more than a premature implementation, but it remains design.

## 6. Communication

Covered by §5. The one novel measurement obligation: announcement accuracy,
commitment-breaking rate, and positive listening via counterfactual token swap.
The MARL literature reports none of these, and they are cheap.

## 7. Process and infrastructure

| Requirement | Status |
|---|---|
| Bind to 0.0.0.0 | **done** |
| Published reports as shareable pages | **done** — report, replay, proposal, observability, difficulty curve |
| Decision records so reasoning survives | **done** — `docs/decisions/`, 13 records, standing mandate in CLAUDE.md |
| Version control | **done** — committed 2026-10-02 after running untracked far too long |

---

## The keystone

**Longer episodes unlock four stalled items at once**, which is why it should go
first:

1. Chaos becomes real. At 600 s an episode spans ~4.2 e-foldings instead of ~1.
2. The three empty lesson slots become reachable — `rolling-with-it` in
   particular, which is the closed-loop skill that transfers to combat.
3. The predictability cone starts firing, so the chaos instrument earns its place.
4. Decision trees genuinely branch: more reachable waypoints, more viable routes,
   real choices rather than one transfer with a tolerance.

With one warning recorded in [0013](decisions/0013-longer-episodes.md): more
e-foldings per episode pushes toward lambda > log(1/gamma)/dt, the fractal-objective
condition we currently clear with only 2x margin. **Re-measure lambda at the new
episode length before training on it.** Making the environment interesting enough
to need chaos may be the thing that makes the policy objective hard to optimise —
and that trade is the real design question for the next phase.

## Recommended order

1. **Longer episodes, measurement only.** Rebuild the demand survey at 600 s;
   re-measure lambda and alpha. Cheap, and it decides much of the rest.
2. **Fix alpha's sample starvation** (`eps_samples` x100) and move to a two-burn
   strategy family. Both gated on GPU memory, hence the 3090.
3. **Planner as teacher** ([0010](decisions/0010-planner-as-teacher.md)). Largest
   performance lever and the architecture that makes callouts faithful.
4. **Evaluation harness** ([0012](decisions/0012-evaluation-statistics.md)).
   Without it, improvements in step 3 are not measurable.
5. **Two ships, no comms, honest baselines** — including full observation sharing
   as the ceiling any channel must beat.
6. **Battle mechanics.**
