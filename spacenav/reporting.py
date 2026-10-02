"""Shared helpers for building report pages: episode geometry as plain JSON.

Both the static report (`scripts/make_report.py`) and the animated replay
(`scripts/make_replay.py`) need the same picture of an episode — bodies and
where they travel, zones, A and B — sampled coarsely enough to ship to a page.
"""

import jax
import numpy as np

from spacenav import constants as C
from spacenav.physics import roll_bodies

_roll = jax.jit(roll_bodies, static_argnums=3)


def r1(a):
    """Round to 0.1 u: plenty for drawing, and much smaller as JSON."""
    return np.round(np.asarray(a, np.float64), 1).tolist()


def episode_geometry(level, task, n_ticks, body_stride=8):
    """Static level description plus each body's path over `n_ticks` control ticks.

    Paths are sampled every `body_stride` ticks; a page interpolates between them.
    """
    lv = jax.tree.map(np.asarray, level)
    n = int(lv.active.sum())
    steps = max(n_ticks, 1) * C.SUBSTEPS
    ps, _ = _roll(level, level.snap_pos[task.snapshot], level.snap_vel[task.snapshot], steps)
    ps = np.asarray(ps)[:: C.SUBSTEPS * body_stride, :n]              # (T, n, 2)
    bodies = [dict(kind=C.KIND_NAMES[int(lv.kind[i])], radius=float(lv.radius[i]),
                   atmo=float(lv.atmo_h[i]), lum=float(lv.lum[i]), path=r1(ps[:, i]))
              for i in range(n)]
    zones = []
    for z in range(len(lv.zone_active)):
        if not lv.zone_active[z]:
            continue
        a = int(lv.zone_anchor[z])
        zones.append(dict(kind=C.ZONE_NAMES[int(lv.zone_kind[z])], anchor=a,
                          r_in=float(lv.zone_r_in[z]), r_out=float(lv.zone_r_out[z]),
                          centre=r1(ps[0, a]) if a >= 0 else [0.0, 0.0],
                          exposure=bool(task.exposure_mask[z])))
    waypoints = []
    for w in range(len(task.wp_active)):
        if not task.wp_active[w]:
            continue
        a = int(task.wp_anchor[w])
        off = np.asarray(task.wp_offset[w]) if hasattr(task, "wp_offset") else np.zeros(2)
        waypoints.append(dict(anchor=a, radius=float(task.wp_radius[w]),
                              rendezvous=bool(task.wp_v_tol[w] < C.FLYTHROUGH_V_TOL / 2),
                              pos=r1(ps[0, a] + off) if a >= 0 else r1(task.wp_pos[w]),
                              path=r1(ps[:, a] + off[None]) if a >= 0 else None))
    return dict(
        bodies=bodies, zones=zones, level_radius=float(lv.level_radius), body_stride=body_stride,
        start=r1(task.start_pos), waypoints=waypoints,
        order_free=bool(task.order_free), accel=float(task.accel), fuel=float(task.fuel),
        weights=np.round(np.asarray(task.weights), 2).tolist(),
    )
