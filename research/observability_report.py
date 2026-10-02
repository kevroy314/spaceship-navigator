"""Build the minimal-observability report page."""

import json
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "research" / "out"

CSS = (ROOT / "research" / "proposal.py").read_text().split("<style>")[1].split("</style>")[0]

HTML = """<title>Seeing Enough to Fly</title>
<link rel="preconnect" href="https://fonts.googleapis.com">
<link rel="preconnect" href="https://fonts.gstatic.com" crossorigin>
<link rel="stylesheet" href="https://fonts.googleapis.com/css2?family=IBM+Plex+Mono:wght@400;500&family=IBM+Plex+Sans:wght@400;500;600&family=Saira+Semi+Condensed:wght@500;600;700&display=swap">
<style>__CSS__</style>

<div class="wrap">
<header>
  <div class="eyebrow">spacenav · observability research</div>
  <h1>Seeing enough to fly</h1>
  <p class="lead">If the ship could only see a noisy ring of bearings around itself, what would it still
  need in order to navigate? There is a clean answer, it is sharper than expected, and one part of it
  is counter-intuitive: <b>finite light speed does not cost the ship information — it hands the ship
  information it otherwise could not have.</b> I measured this on our own simulator rather than
  arguing it.</p>
</header>

<section>
  <h2>1. The exact thing bearings cannot see</h2>
  <p>Newtonian gravity has a symmetry at fixed time: scale every position and velocity by λ and every
  mass by λ³, and the dynamics are unchanged. Bearings are angles, so they are blind to scale by
  construction. Combined with the two symmetries any relative measurement has — translating the whole
  system, and adding a constant velocity to the whole system (Galilean boost), since
  <span class="mono">r(t) → r(t) + ut</span> for everything leaves all relative geometry untouched —
  that predicts <b>five blind directions</b> for a coasting, bearings-only observer in 2D.</p>
  <div class="formula">unknown snapshot θ = (ship position, ship velocity, body positions, body velocities, masses)
<em>for our test level: 4 + 5×8 = 44 unknowns</em>

predicted blind directions: translation (2) + boost (2) + scale (1) = 5</div>
  <p>The formal machinery behind this is the
  <a href="https://doi.org/10.1109/TAC.1977.1101601">nonlinear observability rank condition</a>: stack
  the derivatives of the measurements with respect to the unknowns and ask whether they span everything.
  The classical bearings-only result is that a constant-velocity observer <em>cannot</em> localise a
  constant-velocity target — <a href="https://doi.org/10.1109/TAES.1981.309141">Nardone &amp; Aidala</a>,
  generalised by <a href="https://doi.org/10.1109/7.192098">Fogel &amp; Gavish</a> to "the observer must
  out-manoeuvre the target by one order." Hence the folklore that a bearings-only ship must burn fuel to
  see.</p>
</section>

<section>
  <h2>2. Measured, not assumed</h2>
  <p>Our simulator is differentiable, so this is directly checkable. I took a real level (8 bodies),
  treated the whole snapshot as unknown, recorded bearings to every body over a 25 s window at 40
  sample times, and formed the Jacobian of that bearing history with respect to all 44 unknowns. The
  singular value spectrum gives the blind subspace; projecting each candidate symmetry onto it says
  which ones are invisible. Numbers are "how much of this symmetry is unobservable": 1.000 means the
  ship cannot detect that change at all.</p>
  <div class="table-wrap"><table>
    <thead><tr><th>what the ship does</th><th>blind dims</th><th>translation</th><th>boost</th><th>scale (r, v, 3m)</th></tr></thead>
    <tbody>
      <tr><td>coasting, bearings only</td><td>5 of 44</td><td>1.000</td><td>1.000</td><td>1.000</td></tr>
      <tr><td>thrusting at 2 u/s²</td><td>4 of 44</td><td>1.000</td><td>1.000</td><td class="win">0.007</td></tr>
      <tr><td>coasting, light delay c = 3000 u/s</td><td>2 of 44</td><td>1.000</td><td class="win">0.000</td><td class="win">0.007</td></tr>
      <tr><td>coasting, light delay c = 800 u/s</td><td>2 of 44</td><td>1.000</td><td class="win">0.000</td><td class="win">0.007</td></tr>
    </tbody></table></div>
  <p>The prediction lands exactly. Coasting leaves precisely those five directions blind. <b>Firing the
  engine removes exactly one of them — the scale gauge</b> — because a known thrust is a known
  acceleration in absolute units, which is a ruler. Translation and boost stay invisible forever, and
  that is fine: our levels are defined up to those anyway, and nothing in the task depends on them.</p>
  <div class="card good">
    <h3>The light-speed result</h3>
    <p>Finite light speed breaks <b>two</b> symmetries: scale <em>and</em> boost. A retarded bearing
    depends on how long the light took, and that time depends on the distance — so <b>light delay is a
    disguised range measurement</b>, and it is also sensitive to motion relative to the signal, which
    is what kills the boost gauge. A coasting ship that sees delayed light knows strictly more about
    its universe than a coasting ship that sees instantaneous light, and more than a thrusting ship
    that sees instantaneous light.</p>
    <p>So your instinct was right but the sign is the other way round: in a deterministic world,
    causality delay is not a handicap to be corrected away. It is the only free ruler in the problem.
    The delay is also <em>per-body and state-dependent</em>, which has a practical consequence — the
    clean trick of folding a constant observation delay into an augmented state
    (<a href="https://doi.org/10.1109/TAC.2003.809799">Katsikopoulos &amp; Engelbrecht</a>) does not
    apply; it needs a filter or a recurrent policy.</p>
  </div>
  <p class="foot">Caveats worth stating: this is a local, linearised result at one configuration on one
  level; full rank does not guarantee global identifiability (mirror ambiguities can survive). Our
  levels are built from orbit trees, so they are deliberately non-generic — hierarchical, near-resonant
  — which is exactly where generic rank arguments are weakest. The computation needed float64 and a
  masked self-interaction term, since differentiating √0 on the diagonal makes every gradient NaN.</p>
</section>

<section>
  <h2>3. Occlusion is not bad luck — it is the same event</h2>
  <p>A body is occluded when it lines up behind another. Bearings-only observability degenerates
  precisely when two targets share a bearing (<a href="https://arxiv.org/abs/2507.14765">collinearity
  kills the rank</a>). <b>Occlusion and rank loss are the same geometric condition</b>, not two
  problems that happen to co-occur. That makes "infer the hidden one from the perturbations of the
  visible ones" the right framing: when a body is hidden, the only channel left is its gravitational
  effect on what you can still see — the Le Verrier argument that found Neptune. The modern cautionary
  version is <a href="https://doi.org/10.3847/0004-6256/151/2/22">Planet Nine</a>: inferring unseen
  mass from visible perturbations is weakly identifiable, and its mass-versus-distance degeneracy is
  our λ³ gauge wearing a different hat.</p>
</section>

<section>
  <h2>4. Does the space of sufficient observations have a shape?</h2>
  <p>This was your actual question, and the honest answer is split.</p>
  <div class="two">
    <div class="card flag">
      <h3>Over bodies, arcs and frames — no</h3>
      <p>Sufficiency is upward-closed (adding observations never hurts), so the insufficient sets form a
      simplicial complex, which is true and useless. The property that would buy something — exchange,
      making minimal sufficient sets equicardinal — fails: two frames of three bodies and five frames of
      one body can both be minimal and have different sizes. There is no matroid at this level.</p>
    </div>
    <div class="card good">
      <h3>Over measurement rows — yes</h3>
      <p>Fix a nominal trajectory and treat each scalar measurement as a row of the Jacobian. Sufficient
      sets are the row sets of full rank: a <b>linear matroid over ℝ</b>. That gives real structure —
      all minimal sufficient sets have the same size, greedy computes rank exactly, and there is a clean
      split between generic behaviour and a measure-zero degenerate stratum (collinearities, occultations,
      symmetric configurations). It does <em>not</em> give greedy optimality for accuracy.</p>
    </div>
  </div>
  <p>That last caveat is the practical one. If you build a controller that decides where to point a
  limited sensor, the objective matters more than the algorithm:
  <a href="https://www.jmlr.org/papers/v9/krause08a.html">mutual information is submodular</a> and greedy
  gets you the (1−1/e) guarantee; so is log-det. But
  <a href="https://arxiv.org/abs/2007.05377">λ_min (worst-direction uncertainty) is not submodular</a>,
  and <a href="https://arxiv.org/abs/1711.01920">minimising Kalman error covariance has no constant-factor
  approximation</a> — greedy can be arbitrarily bad. Since weak observability is exactly when λ_min is
  the informative measure, the thing we care about is the thing with no guarantee.</p>
</section>

<section>
  <h2>5. The partial-arc belief, borrowed from astronomy</h2>
  <p>Astronomers have the quantitative version of "a short glimpse is not enough". A short optical arc
  yields an <b>attributable</b> — two angles and two angle rates — leaving range and range-rate
  completely undetermined. Physical constraints (the object is bound, it is not a satellite) confine
  those two unknowns to a compact
  <a href="https://doi.org/10.1007/s10569-004-6593-5"><b>admissible region</b></a>, which is then sampled
  and propagated. The analogy to a ring-arc glimpse of one of our bodies is exact, and our levels come
  with the constraints that make the region compact: bound systems, known play-area radius, validated
  energy drift. That is the right belief object for a partial-observation agent, and a far better
  baseline than hoping a recurrent net invents it.</p>
</section>

<section>
  <h2>6. What I would do in our simulator</h2>
  <p>One caveat dominates the experiment design: <b>our policy network is feedforward with no memory</b>
  (<span class="mono">spacenav/rl/network.py</span>). Parallax is a statement about consecutive frames,
  so a memoryless agent cannot do it in principle. Any bearings-only result on the current architecture
  measures the absence of memory, not the absence of information. Frame-stacking or a recurrent core
  comes first, or the whole ladder is confounded.</p>
  <ol class="steps">
    <li><b>Extend the Jacobian sweep</b> (the experiment above, already built:
    <span class="mono">research/observability.py</span>). Sweep window length, ring coverage, angular
    resolution and number of visible bodies; report λ_min and the condition number, which are the
    informative measures when observability is weak. Costs minutes, no training, and maps the frontier
    analytically.</li>
    <li><b>Feature-ablation ladder.</b> Full state → drop masses → drop velocities → bearings only, each
    with 1, 4 and 16 stacked frames. Separates "information absent" from "memory absent".</li>
    <li><b>Coverage × resolution grid.</b> Mask the ring to 360°/180°/90°/45°/20° and quantise bearings.
    Look for the knee. Control for the fact that a narrow window is also an attention aid by matching
    total information (narrow-and-fine vs wide-and-coarse).</li>
    <li><b>Occlusion, structured vs random.</b> Occlude physically (a body behind another) versus
    randomly at the same rate. If structured occlusion is worse at matched rate, that is direct evidence
    that the rank-degeneracy framing is the right one. Stratify by family, since occultation rate tracks
    crowding.</li>
    <li><b>Light delay, two flavours.</b> Constant delay (recoverable by frame-stacking) versus true
    per-body retardation. Our measurement predicts the second should <em>help</em> a coasting
    bearings-only agent, which would be a satisfying thing to see an agent discover.</li>
    <li><b>k-nearest-body ablation</b> with a random-k control, to test whether gravitational navigation
    is effectively local — I would bet three bodies recovers most of the performance, since attraction
    falls off as 1/r².</li>
  </ol>
</section>

<section class="foot">
  <hr>
  <p><b>Sources.</b> Observability:
  <a href="https://doi.org/10.1109/TAC.1977.1101601">Hermann &amp; Krener 1977</a> ·
  <a href="https://doi.org/10.1109/TAES.1981.309141">Nardone &amp; Aidala 1981</a> ·
  <a href="https://doi.org/10.1109/7.192098">Fogel &amp; Gavish 1988</a> ·
  <a href="https://arxiv.org/abs/1405.6412">empirical observability Gramians</a> ·
  <a href="https://arxiv.org/abs/2401.17117">bearing-angle measurements without lateral motion</a>.
  Rigidity: <a href="https://arxiv.org/abs/1408.6552">bearing rigidity</a> ·
  <a href="https://arxiv.org/abs/1703.04035">Laman graphs and generic bearing rigidity</a> ·
  <a href="https://doi.org/10.1007/BF01534980">Laman 1970</a>.
  Orbit determination: <a href="https://doi.org/10.1007/s10569-004-6593-5">admissible regions</a> ·
  <a href="https://doi.org/10.1007/BF00049379">Gooding IOD</a> ·
  <a href="https://arxiv.org/abs/1801.04004">modern practice</a>.
  Sensor selection: <a href="https://www.jmlr.org/papers/v9/krause08a.html">Krause et al.</a> ·
  <a href="https://arxiv.org/abs/1711.01920">limits of greedy</a> ·
  <a href="https://doi.org/10.1109/TCNS.2015.2453711">log-det submodularity</a>.
  Partial observability: <a href="https://arxiv.org/abs/1207.4167">predictive state representations</a> ·
  <a href="https://doi.org/10.1145/2382559.2382563">hardness of finite-memory control</a> ·
  <a href="https://doi.org/10.1109/TAC.2003.809799">delayed observations</a>.
  Light time: <a href="https://doi.org/10.1086/367593">Klioner 2003</a>.</p>
  <p>The measurement in §2 is <span class="mono">research/observability.py</span>; it reproduces in
  about a minute on CPU.</p>
</section>
</div>
"""


def main():
    html = HTML.replace("__CSS__", CSS)
    out = OUT / "observability_report.html"
    out.write_text(html)
    print(f"wrote {out} ({len(html) / 1e3:.0f} kB)")


if __name__ == "__main__":
    main()
