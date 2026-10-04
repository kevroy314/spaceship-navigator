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
