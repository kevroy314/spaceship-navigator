"""Model demand: one closed-form number that says how much a coarse plan costs.

A patched-conic pilot ("only the body I am orbiting matters") does not fly a
wrong *path* so much as accumulate a wrong *velocity*.  The force it discards has
magnitude eta*|g| (see `env.contested_zone` for eta), so over a flight of
duration T it ends up off by about

    dv_error ~ eta * |g| * T

and the only way to fix that is to spend fuel.  Compare it to everything the ship
has to spend, dv_total = fuel * accel:

    D = eta * |g| * T / (fuel * accel)

D is the fraction of the entire tank that using the wrong model would cost.
Below ~0.1 a coarse plan is comfortably recoverable mid-course; near and above 1
no amount of flying skill rescues it, because correcting the model error would
burn the whole budget.  That is the line between "a person can learn to fly
this" and "a person has to work out the numbers first".

Two things make this worth more than a diagnostic:

* **It needs no simulation.**  Every term is available when the task is sampled,
  so a generator can evaluate it on thousands of candidates for free, instead of
  planning each one to find out whether it was interesting.
* **It inverts.**  D is linear in 1/fuel and in eta, so given a placement you can
  *solve* for the tank that puts a level on a chosen rung (`retune`), rather than
  sampling and hoping.  That is what makes a difficulty curve constructible.

It also absorbs thrust authority, which is otherwise a separate knob.  Writing
A = dv_total / (|g| T) — the ship's budget measured in units of the total impulse
gravity delivers over the flight — the whole thing collapses to

    D = eta / A

which is the shortest honest statement of the dual: *how wrong the coarse model
is, divided by how easily you can afford to be wrong*.  A ship that can
overpower the field (large A) never has to understand it, however contested the
space it flies through.

Measured on this game as it stands: an ordinary mission sits at A ~ 3.8, so
gravity hands the ship about a quarter of its tank for free - the field is not a
small perturbation at all.  What keeps D low is the other term.  eta is 0.0036,
meaning a *single* body accounts for 99.6% of that force, which is exactly why
patched conics works and why it took deliberate placement to break it.

So the two knobs are not interchangeable.  Lowering A (a weaker engine, a
smaller tank) makes the field matter more in total; raising eta makes it matter
in a way no one-body model can absorb.  Only the second is about knowledge, and
only the second cannot be flown around by someone with a full tank.
"""

from typing import NamedTuple

import jax
import jax.numpy as jnp

from spacenav import constants as C
from spacenav import env as E
from spacenav import physics as P
from spacenav.types import Level, Task

# Rungs of the difficulty curve, by D.  Half a decade apart, which is about the
# resolution the measured coarse-versus-fine outcome can actually resolve.  An
# ordinary level with an ordinary tank sits near 1e-4, off the bottom of this
# ladder entirely: that is the finding, not a calibration error.
RUNGS = (0.003, 0.01, 0.03, 0.1, 0.3, 1.0)
RUNG_NAMES = ("negligible", "faint", "slight", "noticeable", "costly", "prohibitive")


class Terms(NamedTuple):
    eta: jnp.ndarray      # discarded-force fraction at the places the ship must go
    g: jnp.ndarray        # local gravity magnitude there
    T: jnp.ndarray        # estimated flight time for the tour
    dv: jnp.ndarray       # total delta-v the tank affords
    D: jnp.ndarray        # eta / A
    A: jnp.ndarray        # dv / (g T): the tank in units of gravity's impulse
    impulse: jnp.ndarray  # g T: how much velocity the field hands you for free
    feasible: jnp.ndarray  # does the estimated tour fit inside the episode clock?
    authority: jnp.ndarray  # engine acceleration over local gravity (instantaneous)


def terms(level: Level, task: Task) -> Terms:
    """Closed-form model demand for a task.  No rollout, no planning."""
    bp, _ = E.snapshot_state(level, task.snapshot)
    wpos, _ = E.waypoint_states(task, bp, bp * 0.0)
    # sample the route at its waypoints and at the midpoint of every leg: the
    # places the ship is obliged to pass through
    chain = jnp.concatenate([task.start_pos[None], wpos], axis=0)
    mids = 0.5 * (chain[1:] + chain[:-1])
    pts = jnp.concatenate([wpos, mids], axis=0)
    live = jnp.concatenate([task.wp_active, task.wp_active], axis=0)

    full = jax.vmap(lambda x: P.gravity_at(x[None], bp, level, 0)[0])(pts)
    dom = jax.vmap(lambda x: P.gravity_at(x[None], bp, level, 1)[0])(pts)
    gmag = jnp.sqrt(jnp.sum(full * full, -1) + 1e-12)
    eta = jnp.sqrt(jnp.sum((full - dom) ** 2, -1) + 1e-18) / gmag

    n = jnp.maximum(jnp.sum(live), 1.0)
    eta_bar = jnp.sum(jnp.where(live, eta, 0.0)) / n
    g_bar = jnp.sum(jnp.where(live, gmag, 0.0)) / n
    # the flight cannot last longer than the episode, however slow the ship is:
    # a tour whose own estimate overruns the clock is infeasible, not difficult
    t_est = task.ref_scale[0]
    T = jnp.clip(t_est, C.CTRL_DT, C.MAX_EPISODE_TIME)
    dv = jnp.maximum(task.fuel * task.accel, 1e-6)
    impulse = jnp.maximum(g_bar * T, 1e-9)
    A = dv / impulse
    return Terms(eta=eta_bar, g=g_bar, T=T, dv=dv, D=eta_bar / A, A=A, impulse=impulse,
                 feasible=t_est < 0.85 * C.MAX_EPISODE_TIME,
                 authority=task.accel / jnp.maximum(g_bar, 1e-9))


def retune(level: Level, task: Task, target_D) -> Task:
    """Set the tank so the task lands on `target_D`, as far as fuel limits allow.

    The exact inversion of D in the one knob that has no effect on geometry:
    dv_total = eta*g*T / D, and fuel = dv_total / accel.  Clipped to the ship's
    tank limits, so `terms(...).D` afterwards is the achieved value, which may
    differ for a level whose geometry cannot reach the requested rung.
    """
    t = terms(level, task)
    fuel = jnp.clip(t.eta * t.impulse / (jnp.maximum(target_D, 1e-6) * task.accel),
                    *C.SHIP_FUEL_LIMITS)
    return task._replace(fuel=fuel)


def rung(D):
    """Index of the rung a demand falls in (0 = easiest, len(RUNGS) = beyond the top)."""
    return int(sum(1 for r in RUNGS if float(D) > r))


def survey(level: Level, task: Task):
    """Terms as plain floats, for logging and reports."""
    t = jax.tree.map(float, terms(level, task))
    return dict(D=t.D, eta=t.eta, g=t.g, T=t.T, dv=t.dv, A=t.A, impulse=t.impulse,
                feasible=bool(t.feasible), authority=t.authority, rung=rung(t.D),
                rung_name=RUNG_NAMES[min(rung(t.D), len(RUNG_NAMES) - 1)])
