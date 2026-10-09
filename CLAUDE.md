# spacenav — working agreement

A GPU n-body gravitational navigation game and RL testbed. Two developers, one
of whom forgets everything between sessions, so the rules below exist to make
the forgetful one useful rather than fast.

## Standing mandates

**1. Write a decision record.** Any decision that carries reasoning — a design
choice, a rejected approach, a measurement that changed the plan — gets a file in
`docs/decisions/`. Numbered, append-only. If a decision is reversed, write a new
record and add `Superseded by` to the old one; never edit the history, because
the wrong turns are the most useful part. Cite the measurement or the paper that
forced it. A decision with no evidence behind it is a preference and must say so.
Use the `decision-record` skill.

**2. Cite the paper, name the number.** Claims in records, reports and commit
messages carry their source: an arXiv/DOI link for literature, a script path and
sample size for anything measured here. "RL does worse than an optimiser" is
useless; "0.117 vs 0.129 m/s over 500 ICs (Bonasera et al., JGCD 2022)" is not.

**3. Measure before believing, including your own ideas — and check that the
factor you are reasoning from actually bears on the mechanism.** Several errors
here were not wrong numbers but wrong relevance: arguing from full observability
about a channel whose content is a joint choice (0016), or from resonance overlap
about episodes shorter than one orbit (0005). This project has
produced several confident, elegant, wrong conclusions, each caught only by
checking it against the physics. Recent examples, all recorded: the Wisdom
resonance band was the wrong length scale (0005); a 40 s planning horizon against
130 s missions produced a flat null result that looked like a finding; an
uncertainty exponent was fitted to an empty winning set (0009); a fractal-objective
alarm about gamma turned out to be ballistic drift (0008). When a result is
dramatic, try to kill it before reporting it.

**4. Report the negative result.** State plainly when the data does not support
the claim, when n is too small, and when a number measures the probe rather than
the thing. Do not publish a page that looks conclusive when it is not.

**5. Statistics are not optional.** Our outcome is near-binary: +-5 pp needs
n ~ 384, detecting a 5 pp gap needs ~1,500 per arm. Separate training-seed
variance from episode variance. See `docs/decisions/0012-evaluation-statistics.md`.

## Environment traps (each has cost real time)

| Trap | Rule |
|---|---|
| `conda run` buffers all stdout until exit, so a long job looks hung | Call `/home/kevin/anaconda3/envs/spaceship/bin/python` directly, with `-u` when redirecting |
| `pkill -f` / `pgrep -f` match the killing shell's own argv and kill the session (happened 5x) | **Never type those letters.** Use `ps -eo pid,args \| grep "scrip[t].py" \| awk '{print $1}'` |
| The bracketed-`ps` trick still killed the session once | It fails if the *same command* also mentions the target unbracketed — e.g. killing a job and relaunching it in one line. **Kill and relaunch in separate calls**, always |
| Two JAX processes on one GPU trigger an out-of-memory reap | One GPU job at a time; poll `nvidia-smi --query-compute-apps=pid --format=csv,noheader` before launching |
| Python block-buffers stdout to a file | `python -u` |
| A `python - <<'PY'` heredoc job that gets backgrounded loses stdin and dies silently, leaving a zero-byte log | Write long jobs to a file in the scratchpad and run the file. Heredocs are for edits and quick checks only |
| JAX silently ran CPU-only on Pascal | `jax[cuda12]`; suspect it whenever throughput looks wrong |

## Invariants

- **Physics changes must be mirrored in `web/js/sim.js`.** `tests/test_js_parity.py`
  guards it. The browser integrates its own ephemeris from a ~1.5 KB snapshot
  (`docs/decisions/0002-ship-as-test-particle.md`).
- Tunables live in `spacenav/constants.py`.
- Changing `MAX_WAYPOINTS` or the observation layout invalidates every existing
  checkpoint. Say so when you do it.
- `data/` is gitignored. Pools are reproducible via `scripts/build_pools.py`.

## Orientation

- `spacenav/` physics, env, levels, RL, planner, demand, lessons
- `spacenav/opt/` planner (CEM), probe (two-axis bands), routes, necessity
- `web/` browser client, exact physics port
- `scripts/` pools, training, reports, curriculum and causal builds
- `docs/decisions/` why anything is the way it is — **read this first**
- `.claude/skills/` training-run, decision-record
