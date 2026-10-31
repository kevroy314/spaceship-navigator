# Decision records

One file per decision that carries reasoning we would otherwise lose. Numbered,
append-only: if a decision is reversed, write a **new** record that supersedes
the old one and add a `Superseded by` line to the original. Never edit history —
the wrong turns are the most useful part of the record.

Each record states the measurement or paper that forced the decision. A decision
with no evidence behind it is a preference, and should say so.

| # | decision |
|---|---|
| [0001](0001-jax-on-pascal.md) | JAX with CUDA 12 and the Pascal kernel gap |
| [0002](0002-ship-as-test-particle.md) | The ship is a test particle |
| [0003](0003-task-rebuilt-around-gravity.md) | The task was rebuilt because thrust dwarfed gravity |
| [0004](0004-model-demand.md) | Model demand D = eta / A as the generator's dial |
| [0005](0005-equal-pull-not-resonance.md) | The equal-pull surface, not resonance overlap |
| [0006](0006-two-axis-bands.md) | Model scale and execution margin stay separate axes |
| [0007](0007-lessons-and-instruments.md) | Discrete lessons, measured signatures, earned instruments |
| [0008](0008-chaos-is-aspirational.md) | The game is not chaotic yet; chaos must be built |
| [0009](0009-probability-field-and-alpha.md) | Win fractions, not binary volumes; alpha needs samples |
| [0010](0010-planner-as-teacher.md) | Search over the true simulator, distil the policy |
| [0011](0011-callouts-and-legibility.md) | Hand-designed callout vocabulary; legible trajectories |
| [0012](0012-evaluation-statistics.md) | Evaluation statistics we actually need |
| [0013](0013-longer-episodes.md) | Longer episodes as the keystone |
| [0014](0014-sampler-bugs.md) | Four measurement bugs the test suite found |
| [0015](0015-architecture-pick.md) | Search the true simulator with mctx; reduce the horizon |
| [0016](0016-communication-is-equilibrium-selection.md) | Communication is equilibrium selection (corrects 0015) |
| [0017](0017-no-pre-play.md) | No pre-play; coordination is in-band (corrects 0016) |
| [0018](0018-chaos-needs-close-encounters.md) | Longer episodes do not produce chaos (corrects 0013) |
| [0019](0019-revert-the-clock-for-measurement.md) | The 600 s clock hurt the atlas; fix the instruments |
| [0020](0020-the-clock-was-not-the-cause.md) | The candidate mix, not the clock (corrects 0019) |
| [0021](0021-sparsity-is-the-answer.md) | A ~1% win rate is the measurement, not a defect |
| [0022](0022-solvability-gate.md) | A level nobody can fly gets no lesson |
| [0023](0023-planner-fails-multi-stop.md) | The planner flies one stop, not two |
| [0024](0024-arm-b-was-void.md) | An A/B between runs of different length is void; pin the schedules |
| [0025](0025-two-stop-is-infeasible-not-hard.md) | Two-stop infeasible at 150 s — **wrong, superseded by 0032** |
| [0026](0026-the-1080-cannot-buy-both.md) | Scale and steps compete for one card; 285k params is the budget, not the design |
| [0027](0027-the-prng-collision-did-not-shift-the-distribution.md) | The PRNG collision did not shift the training distribution (p=0.997) |
| [0028](0028-the-solvability-test-was-underpowered.md) | The pilot-solvability test was underpowered (superseded by 0030) |
| [0029](0029-the-regression-was-two-things-and-one-was-not-a-regression.md) | The v4 regression was two effects; the larger was a deliberate difficulty increase |
| [0031](0031-the-instrument-channels-cost-seven-points.md) | The instrument channels cost ~7 pp as policy inputs; two of four are redundant |
| [0032](0032-two-stop-is-flyable-and-the-clock-helps.md) | Two-stop tours are flyable and the clock helps to 600 s (supersedes 0025) |
| [0033](0033-the-cost-splits-exactly-on-the-redundancy-line.md) | eta and stretch are free; budget and demand cost the whole 7 pp |
| [0034](0034-the-clock-is-a-process-setting-and-it-helps-some-navigators.md) | The clock is a per-process setting; a longer one helps a navigator only if it can use the time |
| [0035](0035-budget-not-clock.md) | 10k updates buys 5-15 points; a 600 s clock at the old discount loses 20-40 |
