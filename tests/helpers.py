"""Shared helpers: hand-built levels with known structure.

Built with `custom_level` rather than `build_levels` so the geometry is exact
and the tests stay fast — no generate-and-validate loop, no GPU batch.
"""

import numpy as np

from spacenav import constants as C
from spacenav.levels.build import custom_level
from spacenav.levels.tree import Node, Orbit


def hierarchy_level(level_radius=1500.0):
    """Star + giant (+ moon) + outer rocky: three well-separated gravity scales.

    Luminosity is left at zero so nothing in these tests dies of radiation; the
    hazard field has its own tests.  The giant's equal-pull radius is
    a*sqrt(m/M) = 400*sqrt(500/5e4) = 40 u, comfortably clear of its 10 u
    surface, so a waypoint can be parked where a one-body model is worst.
    """
    star = Node(C.KIND_STAR, 5e4, 20.0)
    giant = star.add(Node(C.KIND_GIANT, 500.0, 10.0), Orbit(400.0, phase=0.0, omega=0.0))
    giant.add(Node(C.KIND_MOON, 4.0, 4.0), Orbit(70.0, phase=0.0, omega=0.0))
    star.add(Node(C.KIND_ROCKY, 20.0, 6.0), Orbit(850.0, phase=np.pi / 2, omega=0.0))
    return custom_level(star, level_radius=level_radius)


def body_index(level, kind):
    """Index of the first active body of `kind` in a level."""
    k = np.asarray(level.kind)
    a = np.asarray(level.active)
    hits = np.nonzero((k == kind) & a)[0]
    assert len(hits), f"no active body of kind {kind}"
    return int(hits[0])
