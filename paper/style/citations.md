# Citations: claims in `paper/OUTLINE.md` mapped to `paper/references.bib`

Produced by the citation pass, 2026-10-30. The bib has 114 entries; every one
was checked against a machine-readable record (Crossref for DOIs, the arXiv
API for eprints, Open Library for ISBNs, NASA NTRS, or the YouTube page
itself) and the trail for each is in the verification table at the end. The
drafter should resolve `[CITE: ...]` placeholders from the claim table, read
the "what it actually says" column before citing, and treat the **flag**
column as a warning that the literature disputes, narrows or does not support
the outline's wording.

Conventions (from `STYLE.md`): numbered references, `\cite{key}`,
`\bibliographystyle{unsrt}`, `\bibliography{references}`. Several entries use
`\url{}` in `note`/`howpublished`, so the main document must load the `url`
package (or `hyperref`).

Key naming: `surnameYEAR`, lowercase; `a` suffix disambiguates (`jiang2021`,
`jiang2021a`; `braintruffle2024`, `braintruffle2024a`).

## Claim table

| Section | Claim (short) | Keys | What each source actually says | Flag |
|---|---|---|---|---|
| 1 | The n-body problem has no closed-form solution and is where chaos was first seen | `poincare1890`, `henon1964`, `szebehely1967`, `koon2011` | Poincaré 1890: the three-body memoir; homoclinic tangles, non-integrability. Hénon & Heiles 1964: a two-degree-of-freedom galactic potential that is regular at low energy and chaotic at high energy, i.e. "quasi-integrable until it is not". Szebehely 1967: the standard CR3BP monograph (Jacobi constant, zero-velocity curves, Hill regions). Koon et al. 2011: modern textbook on invariant manifolds and mission design in the CR3BP. | — |
| 1 | Spacecraft guidance literature: patched conics, sphere of influence, Hill sphere, gravity assist | `bate1971`, `battin1999`, `vallado2013`, `hill1878`, `flandro1966`, `belbruno1993`, `marsden2006` | Bate/Mueller/White 1971 ch. 7–8: patched conics and the Laplace sphere of influence. Battin 1999 and Vallado 2013: reference texts. Hill 1878: the Hill region and (implicitly) the Hill radius from the lunar problem. Flandro 1966: the Jupiter gravity-assist Grand Tour trajectory that became Voyager. Belbruno & Miller 1993: weak-stability-boundary ballistic capture. Marsden & Ross 2006: tube dynamics, the interplanetary transport network, review for mathematicians. | Laplace's sphere of influence and the Hill radius are *not* the project's equal-pull surface (ADR 0005). Attribute the equal-pull surface to this paper, not to these sources. |
| 1 | Learning-based guidance exists and is close to optimal-control quality | `izzo2019`, `izzo2021`, `gaudet2020`, `lafarge2021`, `bonasera2023` | Izzo, Märtens & Pan 2019: survey of AI in GNC. Izzo & Öztürk 2021: DNNs imitate optimal low-thrust transfers in real time. Gaudet et al. 2020: RL for 6-DoF powered descent. LaFarge et al. 2021: RL closed-loop guidance in a low-thrust CR3BP (Earth–Moon) environment, PPO. Bonasera et al. 2023: RL stationkeeping on a Sun–Earth L2 halo; the 0.117 vs 0.129 m/s number in `CLAUDE.md` is from this paper (JGCD 46(2), 2023 — not 2022). | — |
| 1 | The braintruffle video series is a major inspiration | `braintruffle2024`, `braintruffle2024a`, `braintruffle2025` | Channel: *braintruffle*, Torsten Heinze, "former nonlinear dynamics researcher (PhD), now science educator"; 11 videos since 2021, on fluids, traffic, crowds, markets, pendulums and orbits. "Master the complexity of spaceflight" (2024-01-25, 32 min) is the one that matters here: probability tracing versus ray tracing of reachable sets in phase space, continuous versus impulsive firing, Oberth efficiency of parabolic approaches, effective potential and zero-velocity curves in the rotating frame, Lagrange points, manifold hopping, weak stability boundaries, interplanetary transport network, symplectic integration. "Simulating Jupiter's comet shield" (2024-11-07): Lagrange-point gateways and Jupiter's effect on comets. "The code that revolutionized orbital simulation" (2025-12-11): symplectic (staggered-update) integrators and phase-space geometry. | **Not a series.** These are three standalone videos on a general nonlinear-dynamics channel. Say "the braintruffle videos on spaceflight and orbital simulation", not "series". `research/proposal.py` also cites "The Manipulator's Sneaky Math to Beat Chaos" as visualisation inspiration; that video is about chaos in a market-manipulation simulator, not spaceflight. Do not cite it for orbital mechanics. |
| 1 | Framing 1: a plain RL loop works until the craft enters a chaotic regime, then only if the algorithm is compatible with what chaos does to the objective | `wang2023`, `metz2021`, `parmas2018`, `suh2022` | Wang, Herbert & Gao 2023: the policy objective is α-Hölder with α = log(1/γ)/λ, fractal (no descent direction) when the maximal Lyapunov exponent λ exceeds log(1/γ); deterministic MDPs included. Metz et al. 2021: backprop-through-time gradients in chaotic dynamics have variance that grows exponentially with horizon; the gradient is "correct" and useless. Parmas et al. 2018: the same "curse of chaos" in model-based policy search; likelihood-ratio gradients avoid it. Suh et al. 2022: when differentiable-simulator gradients are worse than zeroth-order ones (discontinuities, chaos). | ADR 0008 measured λ ≈ 1/140 s⁻¹ against 150 s episodes: *this* field is at roughly one e-folding per episode, below Wang's threshold at γ = 0.999. The claim is correct as a conditional; the paper should not imply the regime is reached by default. |
| 1 | Chaotic systems can nonetheless be controlled, and by RL | `ott1990`, `shinbrot1990`, `gadaleta1999`, `bucci2019`, `zeng2021`, `rabault2019`, `vignon2023` | OGY 1990: small perturbations stabilise unstable periodic orbits embedded in a chaotic attractor. Shinbrot et al. 1990: sensitivity is exploited to *steer* to targets with tiny controls. Gadaleta & Dangelmayr 1999: Q-learning for OGY-style chaos control on maps. Bucci et al. 2019: deep RL (DDPG) stabilises the Kuramoto–Sivashinsky equation. Zeng & Graham 2021: symmetry-reduced deep RL on KS. Rabault et al. 2019: deep RL discovers active flow control at Re = 100. Vignon, Rabault & Vinuesa 2023: review of RL flow control. | **Dispute of tone:** Shinbrot and OGY frame chaos as an *asset* for control (exponential sensitivity means cheap steering), not an obstacle. ADR 0008 notes the successful RL results sit at 10–100 control steps per e-folding; this environment has ~2000. |
| 1 | Framing 2: chaotic regions are filters; bound the exit rather than predict it; plan to adapt | `grebogi1983`, `mcdonald1985`, `daza2016`, `mitchell2005`, `mayne2005`, `kaelbling1998`, `lorenz1963` | Grebogi et al. 1983: final-state sensitivity; the uncertainty exponent α says how little an improvement in initial precision buys when basin boundaries are fractal (uncertain fraction ∝ ε^α). McDonald et al. 1985: fractal basin boundaries, the long version. Daza et al. 2016: basin entropy as a scalar measure of exit uncertainty. Mitchell, Bayen & Tomlin 2005: reachable sets of a continuous dynamic game by a Hamilton–Jacobi PDE, i.e. the rigorous version of "bound where you can end up". Mayne et al. 2005: tube MPC, bounding the effect of disturbances and re-planning inside the tube. Kaelbling et al. 1998: belief-state planning in POMDPs. Lorenz 1963: the origin of "predictability horizon". | Nothing in the literature uses the "state machine with noise filters" phrasing; it is the paper's own framing. Cite the pieces, not the metaphor. |
| 1 | Framing 3: stackable strategies (conics, dead heading, lead pursuit, gravity assist, learned models) | `sutton1999`, `bate1971`, `flandro1966`, `belbruno1993`, `koon2000`, `narvekar2020` | Sutton, Precup & Singh 1999: options, temporal abstraction, the RL formalisation of a strategy hierarchy. Bate 1971: the conic rung. Flandro 1966: the gravity-assist rung. Belbruno 1993, Koon et al. 2000: the manifold / low-energy rung. Narvekar et al. 2020: curriculum learning as climbing a task sequence. | No source orders these exactly as rungs; the ladder is the paper's own. |
| 2 | Deciding whether a level is difficult or interesting is hard in general | `furuta2021`, `dennis2020`, `rutherford2024` | Furuta et al. 2021: policy information capacity, an information-theoretic task-difficulty measure that correlates with learnability across benchmarks. Dennis et al. 2020: difficulty defined by *regret* against the best policy, which must be estimated. Rutherford et al. 2024: common regret approximations do not track learnability; proposes sampling for learnability. | — |
| 2 | Checking whether a solution is correct is, in general, out of reach | `papadimitriou1987`, `madani2003`, `blondel2000` | Papadimitriou & Tsitsiklis 1987: finite-horizon POMDPs are PSPACE-complete; MDPs P-complete. Madani, Hanks & Condon 2003: infinite-horizon unobservable/partially observable planning is undecidable. Blondel & Tsitsiklis 2000: survey of complexity and undecidability in systems and control (null controllability, stability of hybrid systems, etc.). | **Narrow it.** These are results about problem *classes*. A single deterministic, finite-horizon, finite-precision instance is checkable by simulation; what is out of reach is checking *optimality* or searching the policy class. Say "intractable" or "no efficient general procedure", not "impossible". |
| 2 | Motion planning is PSPACE-hard | `reif1979`, `schwartz1983`, `hopcroft1984`, `canny1988`, `reif1994`, `donald1993`, `lavalle2006` | Reif 1979: the generalised mover's problem is PSPACE-hard. Schwartz & Sharir 1983: doubly-exponential exact algorithm. Hopcroft, Schwartz & Sharir 1984: PSPACE-hard for multiple rectangles (warehouseman's problem). Canny 1988: singly-exponential roadmap, PSPACE-complete. Reif & Sharir 1994: *moving obstacles* make it PSPACE-hard even in 3D with a point robot and bounded velocity; closest analogue to moving targets. Donald et al. 1993: kinodynamic planning. LaValle 2006: textbook. | **Partial dispute:** Donald, Xavier, Canny & Reif 1993 give a *polynomial-time ε-approximation* for kinodynamic planning with bounded velocity and acceleration. Exhaustive search is not the only option; approximate search is polynomial in 1/ε. The outline's `O(a^T)` is an argument about naive enumeration, not a theorem; no citation supports the count itself. |
| 2 | Reachability in continuous / hybrid dynamics is undecidable | `moore1990`, `koiran1994`, `henzinger1998`, `asarin1995`, `blondel1999` | Moore 1990: a smooth map in R³ (and an ODE) can simulate a universal Turing machine, so reachability and long-term prediction are undecidable even with exact initial conditions. Koiran et al. 1994: piecewise-linear maps in 2D already suffice. Henzinger et al. 1998: reachability is undecidable for most hybrid automata beyond the initialised rectangular class. Asarin, Maler & Pnueli 1995: undecidable for piecewise-constant derivatives in 3D, decidable in 2D. Blondel & Tsitsiklis 1999: stability and controllability of elementary hybrid systems NP-hard / undecidable. | These results use pathological constructions; they bound what a *general* checker can do, not what happens on this field. |
| 2 | Once a chaotic regime is entered, the precision of the check itself becomes a problem | `hammel1987`, `sauer1997`, `sauer2002`, `boekholt2015`, `portegieszwart2018`, `rein2015`, `wisdom1991` | Hammel, Yorke & Grebogi 1987: numerical chaotic orbits are shadowed by true orbits for long times (shadowing lemma, numerically). Sauer, Grebogi & Yorke 1997: shadowing *breaks down* in finite time when the dynamics has fluctuating Lyapunov exponents; the valid time scales as a power of the precision. Sauer 2002: the same, with large errors in physical simulations. Boekholt & Portegies Zwart 2015: an n-body solution converges only when the integration precision exceeds the phase-space divergence; statistics are reliable even when individual orbits are not. Portegies Zwart & Boekholt 2018: time-reversibility test, exponential divergence of n-body integrations. Rein & Spiegel 2015: IAS15, machine-precision integration for a billion orbits. Wisdom & Holman 1991: symplectic maps for the n-body problem (the family the simulator's kick-drift-kick leapfrog belongs to). | Boekholt & Portegies Zwart's point cuts both ways: ensemble statistics (success rates) are robust where individual verdicts are not. That supports the paper's use of pools and suites. |
| 3 | Representational difficulty: does a coarse (one-body) model suffice? `eta`, `A`, `D` | `bate1971`, `battin1999`, `vallado2013`, `hill1878` | Patched-conic and sphere-of-influence constructions are exactly a one-body approximation with a switching rule; these are the textbook sources for "the field a one-body model discards". | `eta`, `A`, `D` and the equal-pull surface are the paper's own; cite the sources for the *coarse model*, not for the quantities. |
| 3 | Level families as the biggest lever | `cobbe2020`, `justesen2018`, `cobbe2019` | Cobbe et al. 2020: Procgen, 16 procedurally generated games; generalisation is measured across levels. Justesen et al. 2018: procedural level generation during training improves generalisation; difficulty-adaptive generation. Cobbe et al. 2019: train/test level gap quantified. | — |
| 3 | Reaction-time (trivial) difficulty against human limits | `woods2015`, `franklin2011`, `franklin2014`, `repp2005`, `repp2013` | Woods et al. 2015: simple visual reaction time ≈ 200–250 ms in healthy adults, with the determinants. Franklin & Wolpert 2011: review of sensorimotor control, feedback delays (visual ~ 100–200 ms). Franklin, Franklin & Wolpert 2014: visuomotor feedback responses begin ~ 120–180 ms after a perturbation and are fractionated by direction. Repp 2005: synchronisation tapping, asynchrony tolerance and the ~ 1.8 s upper limit of the synchronisation range. Repp & Su 2013: update. | The draft's "Franklin et al. 2019, J Neurosci" does not exist; the paper is 2014, J Neurophysiol. |
| 3 | Chaos measured, not assumed: Lyapunov time versus episode | `benettin1980`, `wolf1985`, `ott2002`, `lichtenberg1992`, `strogatz2015` | Benettin et al. 1980: the standard algorithm for Lyapunov exponents. Wolf et al. 1985: from time series. Ott 2002, Lichtenberg & Lieberman 1992, Strogatz 2015: textbooks for Lyapunov time and predictability horizons. | — |
| 3 | Uncertainty exponent and basin entropy | `grebogi1983`, `mcdonald1985`, `aguirre2009`, `daza2016` | As above; Aguirre, Viana & Sanjuán 2009 is the Rev. Mod. Phys. review of fractal structures (basins, exits, Wada). | ADR 0009: the exponent was fitted to an empty winning set once; cite Grebogi for the definition, not for a value. |
| 3 | The Hölder threshold for the RL objective | `wang2023` | α = log(1/γ)/λ; fractal when λ > log(1/γ). | — |
| 3 | The field is quasi-integrable on episode timescales; chaos has to be built | `kolmogorov1954`, `arnold1963`, `nekhoroshev1977`, `chirikov1979`, `wisdom1980`, `deck2013`, `laskar1989`, `sussman1992` | KAM (Kolmogorov 1954, Arnold 1963): most invariant tori survive small perturbation. Nekhoroshev 1977: in near-integrable Hamiltonian systems the actions drift only over exponentially long times. Chirikov 1979: resonance overlap as the criterion for the onset of global chaos. Wisdom 1980: the overlap criterion applied to the restricted three-body problem, width ∝ μ^{2/7}. Deck et al. 2013: first-order overlap for two planets. Laskar 1989, Sussman & Wisdom 1992: the solar system is chaotic with a Lyapunov time of ~ 5 Myr, i.e. millions of orbits — chaos is real yet irrelevant on short horizons. | ADR 0005: the Wisdom overlap band was the wrong length scale for episodes shorter than one orbit. Cite Wisdom for *where* chaos lives, not as evidence the levels are chaotic. |
| 3 | Solvability: a level nobody can fly earns no label; a reference optimiser as an imperfect yardstick | `rubinstein1999`, `williams2017`, `dennis2020`, `jiang2021a`, `rutherford2024` | Rubinstein 1999: the cross-entropy method (the planner is CEM). Williams et al. 2017: MPPI, sampling-based MPC on GPUs. Dennis 2020 / Jiang 2021a: regret-based difficulty needs an estimate of the best achievable return. Rutherford 2024: "learnability" (p(1−p)) as a curriculum signal — levels with success 0 or 1 carry no information. | **No support** for "earns no label" as a rule; it is a design decision (state it as one, per CLAUDE.md mandate 1). Rutherford's learnability is the nearest cousin. |
| 3 | The episode clock is not a difficulty knob; time is the least interesting constraint | — | — | **No literature says this.** Deadlines, launch windows and mission lifetime are real constraints in every mission-design source cited here. State it as the paper's position, with the clock-sweep measurement as the evidence. |
| 3 | Voyager was not judged on reaching Jupiter in a year | `kohlhase1977`, `flandro1966` | Kohlhase & Penzo 1977: mission description; launch 1977, Jupiter 1979, Saturn 1980–81, gravity assist at each; the mission is specified by encounter geometry and science, not by elapsed time. Flandro 1966: the alignment that made the Grand Tour possible. | — |
| 3 | Parker bleeds velocity over many assists | `fox2016`, `guo2010` | Fox et al. 2016: mission overview, seven Venus gravity assists over ~ 7 years lowering perihelion to 9.86 R☉. Guo 2010: the trade study that chose the seven-Venus-flyby (V7GA) trajectory over Jupiter-assist alternatives. | — |
| 3 | Artemis boosted over progressive passes | `batcha2020` | Batcha et al. 2020 (AAS 20-649): Artemis I trajectory: trans-lunar injection, an *outbound powered flyby* of the Moon, insertion into a distant retrograde orbit, departure, a *return powered flyby*, ~ 25 days. | **Wording is wrong.** Artemis I did not raise its orbit over progressive passes; it used two lunar powered flybys and a DRO. Progressive apogee-raising describes the ICPS perigee-raise burn and missions such as SMART-1, not Orion. Rewrite as "Artemis I used two powered lunar flybys and a distant retrograde orbit". |
| 3 | Fuel is harder; damage (radiation, debris) is the most interesting constraint | — | — | **Nothing found** that ranks constraints this way. Position, not citation. |
| 4 | Environment: physics, observation, actions, families, pools | `wisdom1991`, `matthews2025`, `kaelbling1998` | Wisdom & Holman 1991 for the symplectic-leapfrog lineage. Matthews et al. 2025 (Kinetix): JAX-native physics environments at scale, the same engineering pattern. Kaelbling 1998 if the observation is discussed as partial. | — |
| 4 | PPO and GAE | `schulman2017`, `schulman2016` | PPO clipped objective; GAE λ-weighted advantage. | — |
| 4 | Network and normalisation at scale | `lee2025` | Hyperspherical normalisation (SimbaV2): unit-norm projection of weights and features, scaling to larger networks and update ratios. | The draft's author list for this paper was wrong (it was the Simba v1 list). |
| 4 | Curriculum | `bengio2009`, `narvekar2020`, `portelas2020`, `florensa2018`, `matiisen2020` | Bengio 2009: curriculum learning. Narvekar 2020: RL curriculum framework and survey. Portelas 2020: automatic curriculum survey. Florensa 2018: goals of intermediate difficulty (GoalGAN). Matiisen 2020: teacher–student curriculum by learning progress. | — |
| 4 | Unsupervised environment design, regret, replay | `dennis2020`, `jiang2021`, `jiang2021a`, `parkerholder2022`, `jiang2022`, `rutherford2024` | PAIRED; PLR; Robust PLR; ACCEL (editing levels); grounding aleatoric uncertainty (adversaries can exploit stochasticity); No Regrets (SFL). | — |
| 4 | Frozen suites, seeds, statistics | `agarwal2021`, `henderson2018`, `colas2018` | Agarwal 2021: IQM, stratified bootstrap CIs, few-run regime. Henderson 2018: reproducibility, seed variance. Colas 2018: power analysis, how many seeds. | ADR 0012 numbers (n ≈ 384 for ±5 pp) are the project's own binomial arithmetic, not from these papers. |
| 5 | Correlations between `eta`, `D`, family, clock and success | — | — | Project measurements; no external citation. |
| 5 | What a single success rate hides; per-instance spread | `agarwal2021`, `cobbe2019`, `kirk2023` | Agarwal: point estimates mislead. Cobbe 2019: performance varies strongly across procedurally generated levels. Kirk 2023: survey of zero-shot generalisation, taxonomy of what varies between instances. | — |
| 5 | Instrument channel: redundant versus new information to a learner | `furuta2021`, `lai2024`, `kaelbling1998` | Furuta 2021: PIC is mutual information between policy parameters and return — a *task* measure, not an observation-channel measure. Lai & Gershman 2024: humans trade reward for policy compressibility; a channel that does not change the policy is "free". Kaelbling 1998: the observation function defines the belief state. | Do not cite Furuta as if it measured channel redundancy. ADR 0016's lesson (argue from the mechanism, not from observability) applies. |
| 5 | Ghost fleets as the picture of hidden variance | — | — | Visual; no citation. |
| 6 | Learning the class rather than overfitting one instance | `cobbe2019`, `cobbe2020`, `justesen2018`, `kirk2023`, `finn2017`, `duan2016`, `beck2025` | Procgen family for generalisation across instances; MAML, RL², meta-RL tutorial for learning-to-learn across task distributions. | — |
| 6 | RL over chaotic simulations | `wang2023`, `metz2021`, `parmas2018`, `suh2022`, `bucci2019`, `zeng2021`, `gadaleta1999` | As in §1. | — |
| 6 | Definitions are arbitrary; better definitions help understand both task and agent | `furuta2021`, `narvekar2020`, `rutherford2024` | Task-difficulty measures and curricula depend on the definition of difficulty; Rutherford shows the measure changes what the curriculum does. | — |
| 7 | Adversarial pursuit / evasion between ships | `mitchell2005`, `reif1994`, `dennis2020` | Mitchell 2005: reachable sets for pursuit–evasion games. Reif & Sharir 1994: planning among moving obstacles. Dennis 2020: adversary-designed environments. | Mention only, per the outline. |
| App. | Finite speed of light; projected versus measured observation | `kaelbling1998` | Partial observability and belief state is the formal home for delayed or projected observations. | No source specific to light-delay in RL observations was found. |
| App. | Goal conditions, reproduction | — | — | Project-specific. |

## What the literature disputes or does not support (summary)

1. "Exhaustively checking even a tiny level is impossible" — overstated. Finite-horizon, finite-precision instances are checkable; the hardness results (`reif1979`, `hopcroft1984`, `papadimitriou1987`, `madani2003`, `moore1990`) bound general procedures, and `donald1993` gives a polynomial ε-approximation for kinodynamic planning. Write "intractable in general" and cite the pair.
2. "Chaos breaks the RL loop" — conditional on λ > log(1/γ) (`wang2023`), which ADR 0008 shows this field does not reach by default; and the control literature (`ott1990`, `shinbrot1990`) treats sensitivity as an asset.
3. "Artemis boosted over progressive passes" — factually wrong for Artemis I (`batcha2020`).
4. "The braintruffle series" — three standalone videos, not a series; one video cited in `research/proposal.py` is about markets.
5. "Time is the least interesting constraint", "a level nobody can fly earns no label", "damage is the most interesting constraint" — positions with no literature behind them; must be stated as such (CLAUDE.md mandate 1).
6. The `O(a^T)` count — an argument, not a citation.

## First-draft bibliography: what was wrong

| Draft key | Problem | Correction (new key) |
|---|---|---|
| `kinetix2024` | Author list wrong (seven names; four are from other FLAIR papers); venue missing | Matthews, Beukman, Lu, Foerster; ICLR 2025 (oral). `matthews2025` |
| `simbav2` | Title wrong ("SimbaV2: hyperspherical normalization…"); author list is the Simba v1 list, not this paper's | "Hyperspherical normalization for scalable deep reinforcement learning", Lee, Lee, Seno, Kim, Stone, Choo; ICML 2025. `lee2025` |
| `wang2023` | Title wrong ("On the fractal nature of the reinforcement learning objective landscape") | "Fractal landscapes in policy optimization", NeurIPS 2023, arXiv:2310.15418. `wang2023` |
| `franklin2019` | Year and journal wrong (J. Neurosci. 2019); no such paper | Franklin, Franklin & Wolpert, *J. Neurophysiol.* 112(9):2218–2233, 2014. `franklin2014` |
| `izzo2021` | Pages missing | 44(2):315–327. `izzo2021` |
| `lai2024` | Volume/article missing | 20(4):e1012057. `lai2024` |
| `agarwal2021` | Fine (NeurIPS 2021) | `agarwal2021` |
| `chirikov1979`, `daza2016`, `deck2013`, `grebogi1983`, `repp2005`, `schulman2017`, `wisdom1980`, `colas2018`, `furuta2021` | Correct as written | same keys |
| (format) | Draft used natbib `\citep`/`\citet` with a hand-written `thebibliography` | Switch to `\cite`, `\bibliographystyle{unsrt}`, `\bibliography{references}`, and load `url`. |

`CLAUDE.md` cites "Bonasera et al., JGCD 2022"; the paper is JGCD 46(2):301–311, **2023** (`bonasera2023`). Not my file to edit.

## Dropped, and why

- Minovitch 1961 (JPL TM 312-130, the gravity-assist memo): no DOI, not on NTRS under that title in the time available; `flandro1966` carries the gravity-assist history.
- Moser 1962 (Nachr. Akad. Wiss. Göttingen) and Oseledets 1968 (Trudy Mosk. Mat. Obshch.): no machine-readable record found; KAM is covered by `kolmogorov1954` + `arnold1963`, Lyapunov exponents by `benettin1980` + `wolf1985`.
- Guo & Thompson, IAC 2019 (Parker in-flight trajectory control): conference-only, not indexed; `guo2010` + `fox2016` suffice.
- Harpold et al. and Eckman et al., AAS 2023 (Artemis I off-nominal and as-flown): one Artemis source is enough.
- Wang et al. 2016 "Learning to reinforcement learn": redundant with `duan2016`.
- Hairer–Lubich–Wanner (geometric integration) and Rein & Tamayo 2015 (WHFast): `wisdom1991` + `rein2015` cover the integrator point.

## Verification table

Codes: **CR** = Crossref API record for the DOI, fields matched; **AX** = arXiv API record for the eprint; **S2** = Semantic Scholar batch record for venue; **OL** = Open Library ISBN/search record; **NTRS** = NASA Technical Reports Server citation page; **YT** = YouTube `ytInitialPlayerResponse` / oEmbed / channel RSS feed fetched 2026-10-30; **WEB** = secondary web sources only (flagged). Helper scripts and raw responses are in the scratch directory `/tmp/claude-1000/-home-kevin-spaceship-navigator/2a556e21-84d3-4ec9-9ff8-86b965c8840d/scratchpad/agents/citations/`.

| Key | How verified | Notes |
|---|---|---|
| poincare1890 | CR (10.1007/BF02392506, -507, -508: Acta Math 13, 1890, Poincaré, chapters of the memoir) | Crossref splits the memoir into chapters; no single DOI for pp. 1–270, so none given. |
| hill1878 | CR 10.2307/2369430 | Am. J. Math. 1(1):5, 1878; end page 26 from the JSTOR record. |
| szebehely1967 | OL (Academic Press 1967, ISBN 0126806500) | |
| bate1971 | OL (Dover 1971, ISBN 9780486600611) | |
| battin1999 | OL ISBN 9781563473425 (AIAA, 1999, rev. ed.) | |
| vallado2013 | OL ISBN 9781881883180 (Microcosm 2013, 4th ed.) | |
| koon2000 | CR 10.1063/1.166509 | Title is "…in celestial mechanics", not "…in the restricted three-body problem". |
| koon2011 | Authors' page ross.aoe.vt.edu/books (Marsden Books 2011, ISBN 978-0-615-24095-4) + OL for the Springer 2007 edition (9780387495156) | WEB-level for the 2011 printing. |
| marsden2006 | CR 10.1090/S0273-0979-05-01085-2 | Crossref dates online publication 2005; print volume 43(1) is 2006. |
| belbruno1993 | CR 10.2514/3.21079 | |
| flandro1966 | WEB: Wikipedia "Grand Tour program" and "Gary Flandro", Caltech alumni page, all give Astronautica Acta 12:329–337, 1966 | **No publisher record.** Astronautica Acta 1966 is not online. Issue number 4 from ADS bibcode 1966AcAau..12..329F, not re-fetched. |
| kohlhase1977 | CR 10.1007/BF00200846 (Space Sci. Rev. 21(2), 1977) | Pages 77–101 from the ADS record, not in Crossref. |
| fox2016 | CR 10.1007/s11214-015-0211-6 | 14 authors as in Crossref. |
| guo2010 | CR 10.1016/j.actaastro.2010.06.007 | |
| batcha2020 | NTRS 20205005150 (title, nine authors, AAS 20-649, 2020 AAS/AIAA Astrodynamics Specialist Conference) | |
| henon1964 | CR 10.1086/109234 | End page 79 standard; Crossref gives start page only. |
| wisdom1980 | CR 10.1086/112778 | End page 1133 standard; Crossref gives start page only. |
| deck2013 | CR 10.1088/0004-637X/774/2/129; AX 1307.8119 | |
| wisdom1991 | CR 10.1086/115978 | End page 1538 standard; Crossref gives start page only. |
| rein2015 | CR 10.1093/mnras/stu2164 | |
| boekholt2015 | CR 10.1186/s40668-014-0005-3 | Article number 2. |
| portegieszwart2018 | CR 10.1016/j.cnsns.2018.02.002 | |
| izzo2019 | CR 10.1007/s42064-018-0053-6; AX 1812.02948 | |
| izzo2021 | CR 10.2514/1.G005254 | |
| gaudet2020 | CR 10.1016/j.asr.2019.12.030; AX 1810.08719 | arXiv title differs ("…Powered Descent and Landing"); journal title used. |
| lafarge2021 | CR 10.1016/j.actaastro.2021.05.014 | |
| bonasera2023 | CR 10.2514/1.G006783 | |
| braintruffle2024 | YT: oEmbed (title, author "braintruffle"); video page (id dhYqflvJMXc, 1932 s, published 2024-01-25, description and chapter list); channel RSS feed | Channel metadata: "I'm Torsten Heinze, former nonlinear dynamics researcher (PhD), now science educator." |
| braintruffle2024a | YT: video page (id 1zu41rrc_Ng, 1156 s, 2024-11-07); RSS | |
| braintruffle2025 | YT: video page (id nCg3aXn5F3M, 1698 s, 2025-12-11); RSS | |
| lorenz1963 | CR 10.1175/1520-0469(1963)020<0130:DNF>2.0.CO;2 | |
| kolmogorov1954 | CR 10.1007/BFb0021737 for the English translation (LNP 93, pp. 51–56) | The 1954 Doklady citation (98:527–530) is from standard references, not a machine record. |
| arnold1963 | CR 10.1070/RM1963v018n05ABEH004130 | |
| nekhoroshev1977 | CR 10.1070/RM1977v032n06ABEH003859 | |
| chirikov1979 | CR 10.1016/0370-1573(79)90023-1 | |
| lichtenberg1992 | OL (Springer 1992, ISBN 9780387977454) | |
| ott2002 | OL ISBN 9780521010849 (CUP 2002, 2nd ed.) | |
| strogatz2015 | OL (ISBN 9780813349107 listed under the title; Westview) | Edition/year from the ISBN, which is the 2015 second edition. |
| benettin1980 | CR 10.1007/BF02128236 | |
| wolf1985 | CR 10.1016/0167-2789(85)90011-9 | |
| grebogi1983 | CR 10.1016/0375-9601(83)90945-3 | |
| mcdonald1985 | CR 10.1016/0167-2789(85)90001-6 | |
| daza2016 | CR 10.1038/srep31416 | Article 31416. |
| aguirre2009 | CR 10.1103/RevModPhys.81.333 | |
| hammel1987 | CR 10.1016/0885-064X(87)90024-0 | |
| sauer1997 | CR 10.1103/PhysRevLett.79.59 | |
| sauer2002 | CR 10.1103/PhysRevE.65.036220 | |
| laskar1989 | CR 10.1038/338237a0 | |
| sussman1992 | CR 10.1126/science.257.5066.56 | |
| ott1990 | CR 10.1103/PhysRevLett.64.1196 | |
| shinbrot1990 | CR 10.1103/PhysRevLett.65.3215 | |
| reif1979 | CR 10.1109/SFCS.1979.10 | |
| schwartz1983 | CR 10.1016/0196-8858(83)90014-3 | |
| hopcroft1984 | CR 10.1177/027836498400300405 | |
| canny1988 | OL (MIT Press 1988, ISBN 0262031361) | |
| reif1994 | CR 10.1145/179812.179911 | |
| donald1993 | CR 10.1145/174147.174150 | |
| lavalle2006 | OL ISBN 9780521862059 (CUP, May 2006) | |
| moore1990 | CR 10.1103/PhysRevLett.64.2354 | |
| koiran1994 | CR 10.1016/0304-3975(94)90229-1 | |
| blondel1999 | CR 10.1016/S0005-1098(98)00175-7 | |
| blondel2000 | CR 10.1016/S0005-1098(00)00050-9 | |
| henzinger1998 | CR 10.1006/jcss.1998.1581 | |
| asarin1995 | CR 10.1016/0304-3975(94)00228-B | |
| papadimitriou1987 | CR 10.1287/moor.12.3.441 | |
| madani2003 | CR 10.1016/S0004-3702(02)00378-8 | |
| mitchell2005 | CR 10.1109/TAC.2005.851439 | |
| mayne2005 | CR 10.1016/j.automatica.2004.08.019 | |
| kaelbling1998 | CR 10.1016/S0004-3702(98)00023-X | |
| sutton2018 | OL ISBN 9780262039246 (MIT Press / Bradford, Nov 2018) | |
| sutton1999 | CR 10.1016/S0004-3702(99)00052-1 | |
| schulman2016 | AX 1506.02438; S2 venue ICLR | |
| schulman2017 | AX 1707.06347; S2 (arXiv only) | Never formally published. |
| wang2023 | AX 2310.15418; S2 venue NeurIPS; proceedings.neurips.cc hash 0d21f257… | |
| metz2021 | AX 2111.05803; S2 (arXiv only) | |
| parmas2018 | AX 1902.01240; S2 venue ICML | ICML 2018 (PMLR 80). |
| suh2022 | AX 2202.00817 (journal_ref "ICML 2022"); S2 | |
| gadaleta1999 | CR 10.1063/1.166451 | |
| bucci2019 | CR 10.1098/rspa.2019.0351 | |
| zeng2021 | CR 10.1103/PhysRevE.104.014210 | |
| rabault2019 | CR 10.1017/jfm.2019.62 | |
| vignon2023 | CR 10.1063/5.0143913 | |
| bengio2009 | CR 10.1145/1553374.1553380 | |
| narvekar2020 | AX 2003.04960 (journal_ref JMLR 21(181):1–50) | |
| portelas2020 | AX 2003.04664; WEB ijcai.org/proceedings/2020/671 (pp. 4819–4825, DOI 10.24963/ijcai.2020/671) | |
| florensa2018 | AX 1705.06366; S2 venue ICML | ICML 2018. |
| matiisen2020 | CR 10.1109/TNNLS.2019.2934906; AX 1707.00183 | |
| dennis2020 | AX 2012.02096; S2 venue NeurIPS 2020 | |
| jiang2021 | AX 2010.03934; S2 venue ICML | ICML 2021. |
| jiang2021a | AX 2110.02439; S2 venue NeurIPS 2021 | |
| parkerholder2022 | AX 2203.01302; S2 venue ICML 2022 | |
| jiang2022 | AX 2207.05219; S2 venue NeurIPS 2022 | |
| rutherford2024 | AX 2408.15099; S2 venue NeurIPS 2024 | |
| cobbe2019 | AX 1812.02341; S2 venue ICML | ICML 2019. |
| cobbe2020 | AX 1912.01588; S2 venue ICML | ICML 2020. |
| justesen2018 | AX 1806.10729 | Workshop venue (NeurIPS 2018 Deep RL workshop) not confirmed by a machine record, so the bib says "arXiv preprint" only. |
| kirk2023 | CR 10.1613/jair.1.14174; AX 2111.09794 | |
| finn2017 | AX 1703.03400; S2 venue ICML 2017 | |
| duan2016 | AX 1611.02779; S2 (arXiv only) | |
| beck2025 | CR 10.1561/2200000080; AX 2301.08028 | |
| henderson2018 | AX 1709.06560; S2 venue AAAI | AAAI 2018. |
| colas2018 | AX 1806.08295 | |
| agarwal2021 | AX 2108.13264 | NeurIPS 2021 (outstanding paper); venue from the NeurIPS proceedings, widely indexed. |
| matthews2025 | AX 2410.23208; S2 venue ICLR; GitHub/PyPI "accepted at ICLR 2025 as an oral" | |
| lee2025 | AX 2502.15280; icml.cc/virtual/2025/poster/44241; S2 venue ICML 2025 | |
| furuta2021 | AX 2103.12726; S2 venue ICML 2021 | |
| lai2024 | CR 10.1371/journal.pcbi.1012057 | |
| rubinstein1999 | CR 10.1023/A:1010091220143 | |
| williams2017 | CR 10.2514/1.G001921 | |
| repp2005 | CR 10.3758/BF03206433 | |
| repp2013 | CR 10.3758/s13423-012-0371-2 | |
| franklin2011 | CR 10.1016/j.neuron.2011.10.006 | |
| franklin2014 | CR 10.1152/jn.00377.2013 | |
| woods2015 | CR 10.3389/fnhum.2015.00131 | Article 131. |

## Compile check

Minimal document (`\nocite{*}`, `unsrt`, packages `inputenc`, `fontenc`, `url`) in the scratch directory, compiled with `/home/kevin/anaconda3/envs/spacenavtex/bin/tectonic --keep-logs --keep-intermediates test.tex`: exit 0; `test.blg` has no BibTeX warnings; 114 `\bibitem`s for 114 `@` entries; `test.log` has no "Citation … undefined" and no "Empty thebibliography"; only two underfull-hbox warnings from long URLs.
