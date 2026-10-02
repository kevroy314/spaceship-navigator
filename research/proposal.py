"""Build the 'interesting levels' proposal page (figures embedded as data URIs)."""

import base64
import json
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "research" / "out"


def img(name):
    return "data:image/png;base64," + base64.b64encode((OUT / name).read_bytes()).decode()


HTML = """<title>Interesting by Construction</title>
<link rel="preconnect" href="https://fonts.googleapis.com">
<link rel="preconnect" href="https://fonts.gstatic.com" crossorigin>
<link rel="stylesheet" href="https://fonts.googleapis.com/css2?family=IBM+Plex+Mono:wght@400;500&family=IBM+Plex+Sans:wght@400;500;600&family=Saira+Semi+Condensed:wght@500;600;700&display=swap">
<style>
:root {
  color-scheme: dark;
  --ground: #0a0e16; --panel: #0f1520; --panel2: #141c2a; --line: #1f2a3b;
  --ink: #e4ebf5; --ink2: #adbacd; --muted: #74839a; --accent: #6da7ec;
  --good: #3fcf6a; --bad: #f07272; --warn: #ffc247; --flag: #d55181;
  --display: "Saira Semi Condensed", "Arial Narrow", system-ui, sans-serif;
  --body: "IBM Plex Sans", system-ui, -apple-system, "Segoe UI", sans-serif;
  --mono: "IBM Plex Mono", ui-monospace, SFMono-Regular, Menlo, Consolas, monospace;
}
* { box-sizing: border-box; }
html, body { background: var(--ground); color: var(--ink); }
body { font: 16px/1.62 var(--body); padding-inline: 20px; padding-block: 30px 80px; }
.wrap { max-width: 860px; margin: 0 auto; display: flex; flex-direction: column; gap: 34px; min-width: 0; }
.wrap > *, section > * { min-width: 0; }   /* flex children default to min-width:auto and refuse to shrink */
h1 { font-family: var(--display); font-size: clamp(34px, 6vw, 54px); line-height: 1.02; font-weight: 700; margin: 0; text-wrap: balance; }
h2 { font-family: var(--display); font-size: 27px; line-height: 1.15; margin: 0 0 2px; letter-spacing: .01em; }
h3 { font-family: var(--display); font-size: 19px; margin: 18px 0 4px; letter-spacing: .03em; }
p { margin: 0 0 12px; max-width: 70ch; color: var(--ink2); }
p.lead { color: var(--ink); font-size: 17.5px; }
section { display: flex; flex-direction: column; min-width: 0; }
.eyebrow { font: 500 12px/1 var(--mono); letter-spacing: .15em; text-transform: uppercase; color: var(--accent); }
.mono, code, table { font-family: var(--mono); }
code { background: var(--panel2); border: 1px solid var(--line); border-radius: 4px; padding: 1px 5px; font-size: .88em; }
b, strong { color: var(--ink); }
ul, ol { margin: 0 0 12px; padding-left: 20px; color: var(--ink2); max-width: 70ch; }
li { margin-bottom: 6px; }
li::marker { color: var(--muted); }
a { color: var(--accent); text-decoration-color: rgba(109,167,236,.4); text-underline-offset: 2px; }
.card { background: var(--panel); border: 1px solid var(--line); border-radius: 10px; padding: 16px 18px; }
.card.flag { border-color: rgba(213,81,129,.45); }
.card.good { border-color: rgba(63,207,106,.35); }
.card h3 { margin-top: 0; }
figure { margin: 0; display: flex; flex-direction: column; gap: 8px; }
figure img { width: 100%; border: 1px solid var(--line); border-radius: 8px; display: block; }
figcaption { font: 13px/1.5 var(--mono); color: var(--muted); }
.table-wrap { overflow-x: auto; max-width: 100%; border: 1px solid var(--line); border-radius: 8px; background: var(--panel); }
table { border-collapse: collapse; width: 100%; font-size: 13.5px; font-variant-numeric: tabular-nums; }
th, td { padding: 7px 12px; text-align: right; border-bottom: 1px solid var(--line); white-space: nowrap; }
th { color: var(--muted); font-weight: 500; font-size: 11px; letter-spacing: .07em; text-transform: uppercase; background: var(--panel2); }
td:first-child, th:first-child { text-align: left; }
tr:last-child td { border-bottom: 0; }
.win { color: var(--good); } .lose { color: var(--muted); } .mid { color: var(--warn); }
.formula { max-width: 100%; background: var(--panel2); border: 1px solid var(--line); border-left: 3px solid var(--accent); border-radius: 6px;
  padding: 11px 14px; font: 14px/1.7 var(--mono); color: var(--ink); overflow-x: auto; margin: 4px 0 12px; }
.formula em { color: var(--muted); font-style: normal; }
.steps { counter-reset: s; list-style: none; padding: 0; display: flex; flex-direction: column; gap: 10px; }
.steps li { counter-increment: s; padding-left: 38px; position: relative; }
.steps li::before { content: counter(s); position: absolute; left: 0; top: 1px; width: 26px; height: 26px; border-radius: 50%;
  background: var(--panel2); border: 1px solid var(--line); color: var(--accent); font: 600 13px/25px var(--mono); text-align: center; }
.two { min-width: 0; display: grid; grid-template-columns: repeat(auto-fit, minmax(min(280px, 100%), 1fr)); gap: 14px; }
.tag { display: inline-block; font: 500 10px/1.7 var(--mono); letter-spacing: .08em; text-transform: uppercase;
  border-radius: 999px; padding: 1px 9px; margin-right: 6px; }
.tag.keep { background: rgba(63,207,106,.16); color: var(--good); }
.tag.drop { background: rgba(240,114,114,.16); color: var(--bad); }
.tag.maybe { background: rgba(255,194,71,.16); color: var(--warn); }
hr { border: 0; border-top: 1px solid var(--line); margin: 6px 0; }
.foot { font-size: 13.5px; color: var(--muted); }
.foot a { color: var(--ink2); }
</style>

<div class="wrap">
<header>
  <div class="eyebrow">spacenav · level design research</div>
  <h1>Interesting by construction</h1>
  <p class="lead">A complicated star system does not make a complicated flight. Measurements on our own
  trained policy say the levels we generate are dynamically bland almost everywhere, and that the
  descriptors we would reach for first are the ones that fail. This proposes a definition of
  "interesting" that is measurable, a curriculum that tracks it, and a generator that works backwards
  from the solution instead of forwards from the star system.</p>
</header>

<section>
  <h2>1. The problem, stated precisely</h2>
  <p>Your intuition: take the solution path, smooth it, measure its complexity. If the winning path is
  simple, the level was boring however baroque its potential field. The three-dimensional time-unfolded
  potential picture explains why that happens so often — a landscape can be intricate while the
  <em>cheapest route across it</em> is a straight roll downhill.</p>
  <p>Our run-3 data says exactly this. Across 960 missions flown by the final policy, the geometric
  deviation of the path barely moves, while the fuel bookkeeping moves a lot:</p>
  <div class="table-wrap"><table>
    <thead><tr><th>descriptor (median)</th><th>easy</th><th>medium</th><th>hard</th><th>blind</th></tr></thead>
    <tbody>
      <tr><td>detour = path ÷ straight-line tour</td><td>1.07</td><td>1.06</td><td>1.08</td><td>1.04</td></tr>
      <tr><td>Δv used ÷ direct-flight estimate</td><td>0.62</td><td>0.64</td><td>0.61</td><td>0.58</td></tr>
      <tr><td>ballistic fraction</td><td>0.56</td><td>0.57</td><td>0.53</td><td>0.65</td></tr>
    </tbody></table></div>
  <p>A detour ratio of 1.04–1.08 on <em>every</em> tier, including missions the scripted pilot cannot
  fly at all, means the measure is saturated, not that the missions are easy. There is a reason, and it
  is not a measurement artefact: <b>a gravity assist is cheap precisely because it does not require a
  detour.</b> It converts angular momentum into orbital energy, changing the velocity history while
  leaving the position history nearly as short as a straight line. Shape-based complexity is blind to
  the thing we care about.</p>
</section>

<section>
  <h2>2. What does discriminate</h2>
  <p>Splitting those missions into ones <b>only the agent can fly</b> (the pilot fails — a greedy trap)
  versus ones <b>both fly</b> (boring by construction), and asking how well each descriptor separates
  the two. AUC 0.5 means no separation; 1.0 means perfect.</p>
  <div class="table-wrap"><table>
    <thead><tr><th>descriptor</th><th>agent-only</th><th>both solved</th><th>AUC</th><th>verdict</th></tr></thead>
    <tbody>
      <tr><td>turning (velocity winding, turns)</td><td>1.23</td><td>0.68</td><td class="win">0.83</td><td>best single geometric signal</td></tr>
      <tr><td>flight time (s)</td><td>104.6</td><td>34.5</td><td class="win">0.81</td><td>strong, but confounds "slow" with "clever"</td></tr>
      <tr><td>fuel margin left</td><td>0.17</td><td>0.47</td><td class="win">0.79 <em>(inverted)</em></td><td>tight budget is the whole point</td></tr>
      <tr><td>detour ratio (smoothed)</td><td>1.17</td><td>1.04</td><td class="mid">0.72</td><td>weak; saturates</td></tr>
      <tr><td>burns (after 1 s smoothing)</td><td>11.5</td><td>6.0</td><td class="mid">0.67</td><td>meaningless unsmoothed — policy chatters</td></tr>
      <tr><td>Δv ÷ direct estimate</td><td>0.71</td><td>0.58</td><td class="mid">0.67</td><td>informative, wrong sign vs expectation</td></tr>
      <tr><td>ballistic fraction</td><td>0.43</td><td>0.47</td><td class="lose">0.49</td><td>useless alone</td></tr>
    </tbody></table></div>
  <p>Two things to take from this. First, <b>smoothing is not a detail</b>: unsmoothed, our policy
  registers 42–120 "burns" per flight, which measures throttle dithering, not manoeuvres. Second,
  <b>no single descriptor is enough</b> — ballistic fraction alone is noise, but ballistic fraction
  <em>paired with</em> an energy gain is the signature of a timed swing-by.</p>
</section>

<section>
  <h2>3. Seeing where the structure is</h2>
  <p>Braintruffle's trick is to stop looking at the bodies and look at the field: the effective
  potential in a co-rotating frame (gravitational plus centrifugal), whose level sets bound where a
  ship of a given energy can go at all, with the Lagrange points as the passes between basins
  (<a href="https://youtu.be/dhYqflvJMXc">Master the Complexity of Spaceflight</a>, ~20:30 and ~22:10;
  the weak-stability-boundary and manifold views at ~26:50). I re-implemented the idea against our own
  simulator rather than borrowing the pictures — and with the honest caveat that the clean version of
  that construction assumes the circular restricted three-body problem, which our eccentric,
  many-bodied levels are not.</p>
  <p>So the middle panel below is the one to trust: a <b>finite-time Lyapunov exponent</b> field. Launch
  a ballistic test particle from every pixel with the local circular velocity, integrate 40 s, and
  measure how fast neighbouring launches separate. Bright ridges are transport barriers; dark basins are
  forgiving. FTLE is designed for exactly our case — finite-time, aperiodic, no conserved energy —
  whereas Jacobi constants, zero-velocity curves and Lagrange points quietly assume a system we do not
  have.</p>
  <figure>
    <img src="__FIG_SOL__" alt="Pseudo-potential, FTLE and escape-time maps for a sun-like level">
    <figcaption>A typical <b>sol_like</b> level. Median FTLE 0.0075 s⁻¹: smooth almost everywhere, thin
    ridges hugging the planets, 1% of launch points die within 40 s. This is a boring landscape — most
    of its area offers no structure to exploit.</figcaption>
  </figure>
  <figure>
    <img src="__FIG_BH__" alt="Pseudo-potential, FTLE and escape-time maps for a black-hole cluster level">
    <figcaption>A <b>bh_cluster</b> level, same computation. Median FTLE 0.0243 s⁻¹ — 3× the stretching —
    with wide ridge structures where trajectories tear apart, and a companion star whose Hill region
    stands out as a coherent island. 5% of launches die within 40 s.</figcaption>
  </figure>
  <p>This gives a per-level scalar we do not currently have: <b>how much usable structure exists at
  all</b>, and how much of the play area is lethal. It costs one ballistic integration per pixel, which
  on our GPU is seconds per level.</p>
</section>

<section>
  <h2>4. A definition of "interesting"</h2>
  <p>The literature is blunt about one thing: solution-path complexity predicts <em>difficulty</em>
  reasonably well and <em>enjoyment</em> badly (<a href="https://dl.acm.org/doi/10.1145/3649921.3659846">Biemer
  &amp; Cooper 2024</a> found path heuristics failed to beat a trivial baseline at predicting fun).
  So path complexity should be a descriptor, never the objective. The objective should be
  <b>learnability</b>, and the recent result here is unambiguous: the widely used regret proxies
  (positive value loss, MaxMC) correlate poorly with what actually helps an agent, while plain success
  variance beats them (<a href="https://arxiv.org/html/2408.15099v1">Rutherford et al., "No Regrets"</a>).</p>
  <div class="formula">interest(level) = <b>learnability</b> × <b>depth</b> × <b>structure</b> × <b>fairness</b>
<br><br>learnability &nbsp;ℒ = p̂ (1 − p̂) · n/(n−1) &nbsp; <em>— p̂ = success rate of the current policy over n≈32 rollouts</em>
<br>depth &nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp; D = (J_pilot − J_planner) / |J_planner| &nbsp; <em>— how much a cleverer plan buys; policy-independent</em>
<br>structure &nbsp;&nbsp;&nbsp; S = Σ_b |Δε_b| / Δv_used &nbsp; <em>— orbital energy the geometry gave away for free, per unit of engine</em>
<br>fairness &nbsp;&nbsp;&nbsp;&nbsp; F = 1 / (1 + λ̄ T) &nbsp; <em>— penalty for aleatoric difficulty: mean FTLE along the reference flight</em></div>
  <p>Each term rules out a specific way of being fake-interesting:</p>
  <ul>
    <li><b>ℒ</b> is zero both when the agent always wins and when it never does. It is the only term that
    moves with training, and it is what makes the curriculum track the agent rather than a hand-set ramp.</li>
    <li><b>D</b> is the <a href="https://link.springer.com/chapter/10.1007/978-3-319-16549-3_30">relative-algorithm-performance</a>
    idea: a level is deep if a strong planner beats a greedy one on it. It is measured from two reference
    solvers we already have, so it does not drift as the policy changes.</li>
    <li><b>S</b> is the flyby energy bookkeeping: per encounter, the change in specific orbital energy
    relative to that body, measured between bracketing apoapses with the engine off. One burn and coast
    scores zero by construction; a swing-by scores 0.2–1.0. <b>This is the term that most directly
    encodes your intuition</b>, and it survives where the geometric measures saturate.</li>
    <li><b>F</b> excludes levels that are hard because they are unforgiving rather than because they need
    a plan. This distinction has a name and a cost: minimax-regret curricula provably over-select levels
    whose difficulty is aleatoric (<a href="https://arxiv.org/abs/2207.05219">SAMPLR/CICS</a>), which
    for us means debris fields and black-hole grazes crowding out everything else.</li>
  </ul>
  <p>And a hard gate before any of it: a level counts only if <b>some</b> reference solver finishes it.
  Unsolvable levels are the classic failure of regret-maximising generators.</p>
</section>

<section>
  <h2>5. The curriculum</h2>
  <p>Replace the hand-tuned ramp (tour length, then rendezvous share, then fuel budget) with a buffer
  that selects on measured interest — the SFL recipe, which suits us because sampling levels is nearly
  free on GPU:</p>
  <ol class="steps">
    <li>Every K updates, sample a large fresh batch of candidate levels (10⁴ is affordable) and roll the
    current policy briefly on each. Keep p̂ per level.</li>
    <li>Score with ℒ, gate by solvability, and (for the top slice only) pay for D, S and F, which need
    the planner and an FTLE pass.</li>
    <li>Keep the top-K in a buffer; train on a mix of buffer levels and fresh ones. Levels age out as p̂
    saturates, which is the ramp — no schedule to hand-tune.</li>
    <li>Maintain a <a href="https://arxiv.org/abs/1504.04909">MAP-Elites</a> archive over 3 descriptors
    (structure S, fuel margin, winding) with ℒ as the cell fitness, so the curriculum cannot collapse
    onto one family of trick.</li>
  </ol>
  <p>What this buys over what we have: the ramp stops being a guess. Run 3's ramp was set by hand, and
  the first attempt at it stalled at 2% success for 13M steps because the constraint bit before the
  agent could fly at all. A learnability-driven buffer would have caught that within one refresh.</p>
</section>

<section>
  <h2>6. Working backwards from the path</h2>
  <p>You asked whether deriving a system to justify a chosen path beats designing initial conditions and
  hoping. I think yes, and I would do it in two tiers, cheapest first.</p>
  <div class="two">
    <div class="card good">
      <h3>Tier 1 — assist-first placement</h3>
      <p>Choose the <em>solution</em> analytically, then place bodies to make it true. Pick a start
      state and a target; decide the flight is "coast, one powered flyby of a body at periapsis r_p
      giving turn angle δ, coast to target". Two-body geometry fixes what mass at what phase produces δ,
      and the Tisserand relation says what the flyby does to the orbit. Place that body, fill the rest of
      the system around it with the existing generator, and run the existing validator.</p>
      <p><b>Cost:</b> days. Reuses everything. <b>Risk:</b> levels become formulaic — one intended trick
      per level, which an agent may learn to pattern-match.</p>
    </div>
    <div class="card">
      <h3>Tier 2 — differentiable level fitting</h3>
      <p>Sample a desired solution skeleton (burn times, coast arcs, which waypoints), then optimise
      <em>level parameters</em> — body masses, orbital elements, phases — so that skeleton is both
      feasible and near-optimal: minimise ‖ballistic endpoint − target‖ plus a margin term that pushes
      the greedy pilot's cost <em>up</em> and the skeleton's cost <em>down</em>. Our simulator is
      differentiable with respect to these parameters.</p>
      <p><b>Cost:</b> weeks, and it inherits the gradient pathologies we already hit — long-horizon
      n-body gradients are dominated by flyby chaos, so this needs the same CEM fallback the planner
      uses. <b>Risk:</b> over-fitting levels to one intended solution.</p>
    </div>
  </div>
  <p>The honest counter-argument to both: forward generation plus rejection is <em>already</em> a
  path-first method if the acceptance test is strong enough, and it costs nothing to try. Our current
  test asks only "is this level stable and non-degenerate". Adding "does the planner beat the pilot by
  D, using structure S, at fairness F" turns the existing generator into an interest-seeking one
  overnight. <b>I would do that first and use it as the baseline any inverse design must beat.</b></p>
</section>

<section>
  <h2>7. Experiments, in order</h2>
  <div class="table-wrap"><table>
    <thead><tr><th>#</th><th>experiment</th><th>settles</th><th>cost</th></tr></thead>
    <tbody>
      <tr><td>1</td><td>Compute S, D, F on 2k existing levels; correlate with agent-only-solvable</td><td>whether the proposed terms predict depth better than the 0.83 AUC we get from winding alone</td><td>~1 h GPU</td></tr>
      <tr><td>2</td><td>Rejection-sample levels on interest; train identically to run 3</td><td>whether interest-filtered levels beat plain generation at equal steps — the cheap win</td><td>1 run (~5 h)</td></tr>
      <tr><td>3</td><td>Learnability buffer (SFL) replacing the hand ramp</td><td>whether the curriculum can set itself</td><td>1 run + buffer code</td></tr>
      <tr><td>4</td><td>Tier-1 assist-first generator, 500 levels; measure S and pilot failure rate</td><td>whether designed flybys actually force structure use</td><td>~3 days build</td></tr>
      <tr><td>5</td><td>FTLE sweep over integration time T on 50 levels</td><td>whether our FTLE ridges are stable or an artefact of T and of our softened gravity</td><td>~2 h</td></tr>
      <tr><td>6</td><td>Tier-2 differentiable fitting on a single hand-chosen skeleton</td><td>feasibility only — does the optimiser find a system that makes the skeleton optimal</td><td>~1 week</td></tr>
    </tbody></table></div>
  <p>Experiment 5 matters more than it looks. Our gravity is softened inside each body's radius, which
  removes the true singularity that drives real flyby chaos, and our leapfrog error grows near periapsis.
  FTLE numbers are therefore <em>sim-specific</em> and only comparable within our own levels — worth
  knowing before any of this becomes a fitness function.</p>
</section>

<section>
  <h2>8. What I would not do</h2>
  <ul>
    <li><span class="tag drop">reject</span><b>Fractal dimension / entropy of the path.</b> Box-counting
    dimension is near-meaningless for a smooth finite curve, and the entropy family (permutation, sample,
    spectral, Lempel-Ziv) is a minefield of preprocessing choices: permutation entropy on oversampled
    smooth data measures integrator noise, and its ordering can invert under 1% noise. Lempel-Ziv is worse
    for us specifically — under the textbook n/log n normalisation it is biased +14% at 200 samples and
    still +1.7% at 100,000, converging only as 1/log n, so two flights of different duration are not
    comparable unless each is normalised against an empirical random reference at its own length. Our
    flights run 200-2,000 ticks, squarely in the drifting regime. If we ever use one of these, it needs
    arc-length resampling, tie dithering, per-length normalisation, and a surrogate distribution beside
    every number.</li>
    <li><span class="tag drop">reject</span><b>Positive value loss / MaxMC as the primary score.</b>
    Documented weak correlation with learning, and in a system with debris and close passes they select
    for chaos — our exact failure mode.</li>
    <li><span class="tag drop">reject</span><b>PAIRED-style adversarial generation.</b> Needs a second
    student we do not have, and is reported to underperform plain randomisation.</li>
    <li><span class="tag maybe">care</span><b>Jacobi constants, zero-velocity curves, Lagrange points.</b>
    Beautiful and correct in the circular restricted three-body problem; in our eccentric multi-body
    levels they exist only instantaneously. Fine as an overlay, wrong as a metric.</li>
  </ul>
</section>

<section>
  <h2>9. Open questions for you</h2>
  <ul>
    <li><b>Interesting for whom?</b> Everything above optimises for the <em>agent's</em> learning. A
    level that is interesting to fly by hand is a different object — that one probably does want the
    shape-complexity and the readable trick. If the game matters as much as the training, we should
    score both and keep the levels that satisfy either.</li>
    <li><b>Do we want one intended solution or many?</b> Inverse design naturally produces levels with a
    single intended trick. Multi-solution levels are richer but cannot be generated by construction in
    the same way.</li>
    <li><b>How much compute per level is acceptable?</b> D and F need a planner run and an FTLE pass per
    level — perhaps 2–5 s. That is fine for a curated buffer of thousands, not for 10⁴ fresh levels per
    refresh. The tiering above assumes we pay only for the top slice.</li>
  </ul>
</section>

<section class="foot">
  <hr>
  <p><b>Sources.</b> Curriculum and interestingness:
  <a href="https://arxiv.org/html/2408.15099v1">No Regrets (SFL)</a> ·
  <a href="https://arxiv.org/pdf/2110.02439">Robust PLR</a> ·
  <a href="https://arxiv.org/abs/2203.01302">ACCEL</a> ·
  <a href="https://arxiv.org/abs/2207.05219">SAMPLR / CICS</a> ·
  <a href="https://arxiv.org/abs/1504.04909">MAP-Elites</a> ·
  <a href="https://arxiv.org/abs/1802.00048">Deceptive Games</a> ·
  <a href="https://arxiv.org/abs/1403.7373">Sudoku difficulty</a> ·
  <a href="https://dl.acm.org/doi/10.1145/3649921.3659846">solution-path heuristics vs enjoyment</a>.
  Dynamics: <a href="https://ross.aoe.vt.edu/books/Ross_3BodyProblem_Book_2022.pdf">Koon, Lo, Marsden &amp; Ross</a> ·
  <a href="https://georgehaller.com/reprints/annurev-fluid-010313-141322.pdf">Haller on Lagrangian coherent structures</a> ·
  <a href="https://web.ma.utexas.edu/mp_arc/c/12/12-19.pdf">Belbruno on weak stability boundaries</a>.
  Signal traps: <a href="https://arxiv.org/abs/1905.06443">permutation-entropy delay selection</a> ·
  <a href="https://arxiv.org/abs/2305.17109">compression-cost action priors</a>.
  Visualisation inspiration: braintruffle,
  <a href="https://youtu.be/dhYqflvJMXc">Master the Complexity of Spaceflight</a> and
  <a href="https://youtu.be/-jF9gW2r_bk">The Manipulator's Sneaky Math to Beat Chaos</a> — figures here are
  our own, computed with our simulator.</p>
  <p>Transcripts, reference frames and the measurement scripts behind every number are in
  <code>research/</code> in the repo.</p>
</section>
</div>
"""


def main():
    html = (HTML.replace("__FIG_SOL__", img("fields_val_seen_12.png"))
                .replace("__FIG_BH__", img("fields_val_seen_1210.png")))
    out = OUT / "interesting_levels_proposal.html"
    out.write_text(html)
    print(f"wrote {out} ({len(html) / 1e6:.2f} MB)")


if __name__ == "__main__":
    main()
