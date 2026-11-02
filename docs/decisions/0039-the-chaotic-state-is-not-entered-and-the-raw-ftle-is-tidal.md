# 0039 — Along real flights the chaotic state is not entered, and the raw 20 s FTLE is tidal shear

**Date** 2026-11-01 · **Status** accepted · **Extends 0008 (ballistic drift) and the chaos measurement in the paper's §3.3**

## Context
The author asked for a figure of the state-machine picture (regimes of
predictability as states) and for a clear statement of what it adds and
where. The paper had said only that the field is quasi-integrable on
episode timescales (λ = 0.0045–0.0086 s⁻¹ against λ_crit = 0.0150) and, in
the discussion, that the environment "does not yet contain such regions".
Neither had been checked along flights a policy actually flies.

## Measurement
`scripts/fig_regimes.py`: state membership per control tick along sampled
rollouts of `ppo_v5_noinstr` (checkpoint 3000, 16 rollouts, `PRNGKey(7)`),
with contested = η > 0.1 at the craft, hazard = inside a harmful zone or an
atmosphere or damage > 0.0667 health/s, chaotic = λ > 0.0150 s⁻¹ measured
from the craft's own state, precedence hazard > chaotic > contested >
well-behaved. Two readings of λ: the raw 20 s finite-time exponent from a
position displacement (as in `fig_phase.py` row b), and the paper's own
exponent (one tick of thrust as the perturbation, `sep = a t e^{λt}` fitted
over 2–20 s, ballistic factor divided out).

| task | reading | well | contested | chaotic | hazard |
|---|---|---|---|---|---|
| hard:556, star_cluster, mean of 16 | ballistic | 0.0 | 86.5 | 1.7 | 11.8 |
| hard:556 | raw FTLE | 0.0 | 80.8 | 7.4 | 11.8 |
| hard:0, asteroid_field, mean of 16 | ballistic | 73.9 | 0.0 | 0.0 | 26.1 |
| hard:0 | raw FTLE | 52.9 | 0.0 | 21.0 | 26.1 |
| easy:0, sol_like, mean of 16 | ballistic | 44.1 | 0.1 | 6.4 | 49.4 |
| easy:0 | raw FTLE | 26.0 | 0.0 | 24.6 | 49.4 |
| medium:0, asteroid_field | ballistic | 74.6 | 0.0 | 0.0 | 25.4 |
| medium:1, asteroid_field | ballistic | 25.5 | 0.0 | 0.0 | 74.5 |

Screen of all 132 binary_star and star_cluster tasks of the hard suite (η
and hazard only, 16 rollouts each): every task crosses η > 0.1, 36–44 % are
contested on every tick, and none has a well-behaved segment above 5 %.

Along every flight the raw FTLE's logarithm tracks the tidal closed form
√(Σ 2GM/r³) at correlation 0.86–1.00, and every raw crossing of λ_crit lies
within 290–434 u of a star. The 1.7 % chaotic reading on hard:556 is one
2 s window on a dive toward three stars and may itself be tidal.

## Decision
1. The paper states that on the levels the environment currently generates
   the state machine has in effect two states per family (well-behaved or
   contested by the mass hierarchy, with hazard entered from either), that
   the chaotic state is not entered on episode timescales, and that the
   first picture therefore does not pay off on the present levels; its value
   is recorded as prospective (a specification for constructed-chaos levels,
   and the entry/exit-set quantities such a navigator would need).
2. The published figure colours the state bar by the ballistic reading and
   shows both exponents beneath it, because the disagreement is the lesson.
3. The phase-portrait caption now says its FTLE row maps tidal shear beside
   the bodies as much as divergence.

## Consequences
- A state criterion based on a raw short-window FTLE labels the near field
  of every star chaotic. This is 0008's ballistic-drift error in another
  guise and is mandate 3's "wrong relevance" trap; any future use of an FTLE
  field as a chaos indicator must divide out the ballistic factor or use a
  window long against the local orbital period.
- At η > 0.1 a comparable-mass companion contests the whole disc, so the
  contested state is a property of the family, not of the route, on these
  levels; a route-level contested/well-behaved transition needs a
  hierarchical level with an equal-pull shell on the path (jovian) and was
  not measured.
- The hazard threshold (10 health per episode) is a choice; on easy:0 the
  policy skims the star and half the flight is "hazard" at that threshold.
- The companion figure `paper/figures/regimes.pdf` and the paper's §3.3
  paragraph carry the numbers; `scripts/fig_regimes.py --all` reprints the
  table.

## Evidence
- `scripts/fig_regimes.py` (defaults `--task hard:556 --chaos ballistic`),
  CPU backend, `data/eval/suites.pkl`, `data/rl/ppo_v5_noinstr/ckpt_03000.msgpack`.
- Screen log in the session scratchpad (`screen.log`, 22 min CPU).
- 0008 — ballistic drift; 0005 — the wrong length scale; the paper's §3.3.
