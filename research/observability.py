"""Is a bearings-only observer in our simulator actually unobservable in scale?

Newtonian gravity has an exact symmetry at fixed time:  r -> λr,  v -> λv,  m -> λ³m
leaves the accelerations consistent, and bearings are scale-invariant.  So a ship
that only measures directions to bodies, and does not know their masses, should be
unable to determine the scale of the system --- unless something supplies an
absolute length or acceleration.  Two candidates: firing the engine (a known
acceleration in absolute units) and finite light speed (the delay is a disguised
range measurement).

This measures it directly: build the Jacobian of the bearing history with respect
to the unknown snapshot, and look at its singular values.  A zero singular value
is an unobservable direction; we then check whether that direction is the
predicted gauge (positions and velocities scaling together with 3x the masses).

    python -m research.observability
"""

import json
from pathlib import Path

import jax
import jax.numpy as jnp
import numpy as np

from spacenav import constants as C
from spacenav import env as E
from spacenav import physics as P
from spacenav.levels.build import index_level, load_pool

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "research" / "out"
jax.config.update("jax_enable_x64", True)      # singular values near zero need the precision


def safe_bodies_step(bp, bv, mass, radius, h):
    """Leapfrog for the bodies, with the self-interaction masked out rather than
    softened.  Identical values to physics.bodies_step, but differentiable with
    respect to body positions and masses: sqrt(0) on the diagonal otherwise makes
    every gradient NaN."""
    eye = jnp.eye(bp.shape[0], dtype=bool)

    def acc(p):
        d = p[None, :, :] - p[:, None, :]
        r2 = jnp.sum(d * d, -1)
        r = jnp.sqrt(jnp.where(eye, 1.0, r2))
        soft = jnp.maximum(r, radius[None, :])
        w = jnp.where(eye, 0.0, mass[None, :] / soft**3)
        return jnp.sum(w[:, :, None] * d, axis=1)

    v = bv + 0.5 * h * acc(bp)
    p = bp + h * v
    return p, v + 0.5 * h * acc(p)


def ship_accel(x, bp, mass, radius):
    d = bp - x[None]
    r = jnp.sqrt(jnp.sum(d * d, -1) + 1e-12)
    soft = jnp.maximum(r, radius)
    return jnp.sum((mass / soft**3)[:, None] * d, axis=0)


def make_theta(level, snapshot, ship_pos, ship_vel, n):
    bp, bv = E.snapshot_state(level, snapshot)
    return jnp.concatenate([ship_pos, ship_vel, bp[:n].ravel(), bv[:n].ravel(),
                            level.mass[:n]]).astype(jnp.float64)


def bearings(theta, level, n, thrust_dir, thrust_mag, seconds, samples, c_light=None):
    """Bearing to every body at `samples` times, given the unknown snapshot theta."""
    x, v = theta[:2], theta[2:4]
    bp = theta[4:4 + 2 * n].reshape(n, 2)
    bv = theta[4 + 2 * n:4 + 4 * n].reshape(n, 2)
    m = theta[4 + 4 * n:]
    lv = level._replace(mass=level.mass.at[:n].set(m).astype(jnp.float64),
                        radius=level.radius.astype(jnp.float64),
                        atmo_h=level.atmo_h.astype(jnp.float64),
                        atmo_rho=level.atmo_rho.astype(jnp.float64),
                        lum=level.lum.astype(jnp.float64))
    # only the active bodies: the padded slots all sit at the origin, so their mutual
    # distances are zero and every gradient through them is NaN
    bp_full, bv_full = bp, bv

    steps = int(seconds / C.PHYS_DT)
    every = max(steps // samples, 1)

    mass_full = m
    rad_full = lv.radius[:n].astype(jnp.float64)

    def step(carry, i):
        bp, bv, x, v = carry
        h = C.PHYS_DT
        a0 = ship_accel(x, bp, mass_full, rad_full) + thrust_mag * thrust_dir
        vh = v + 0.5 * h * a0
        x1 = x + h * vh
        bp1, bv1 = safe_bodies_step(bp, bv, mass_full, rad_full, h)
        v1 = vh + 0.5 * h * (ship_accel(x1, bp1, mass_full, rad_full) + thrust_mag * thrust_dir)
        d = bp1 - x1[None]
        if c_light is not None:
            # retarded position: where the body was when the light left it
            tau = jnp.sqrt(jnp.sum(d * d, -1)) / c_light
            d = d - bv1 * tau[:, None]
        return (bp1, bv1, x1, v1), jnp.arctan2(d[:, 1], d[:, 0])

    _, h = jax.lax.scan(step, (bp_full, bv_full, x, v), jnp.arange(steps))
    return jnp.unwrap(h[::every][:samples], axis=0).ravel()


def spectrum(level, n, ship_pos, ship_vel, thrust_mag, seconds=25.0, samples=40, c_light=None):
    theta = make_theta(level, 0, ship_pos, ship_vel, n)
    dirn = jnp.array([0.6, 0.8])
    f = lambda th: bearings(th, level, n, dirn, thrust_mag, seconds, samples, c_light)
    J = np.asarray(jax.jacfwd(f)(theta))
    _, s, Vt = np.linalg.svd(J)
    return s, np.asarray(theta), Vt


def symmetries(theta, n):
    """The directions in snapshot-space that bearings should be blind to.

    translation: move everything (ship included) - relative geometry unchanged.
    boost:       add a constant velocity to everything - relative geometry unchanged
                 at every time, since r_i(t) -> r_i(t) + u t for all i.
    scale:       r -> λr, v -> λv, m -> λ³m - the Newtonian gauge; bearings are
                 scale-free, so only an absolute acceleration or length breaks it.
    """
    z = lambda: np.zeros_like(theta)
    out = {}
    for axis, name in [(0, "x"), (1, "y")]:
        t = z(); t[axis] = 1.0; t[4 + axis::2][: 2 * n][::1] = 0.0
        t[4:4 + 2 * n].reshape(n, 2)[:, axis] = 1.0
        out[f"translation {name}"] = t
        b = z(); b[2 + axis] = 1.0
        b[4 + 2 * n:4 + 4 * n].reshape(n, 2)[:, axis] = 1.0
        out[f"boost {name}"] = b
    sc = np.concatenate([theta[:4], theta[4:4 + 4 * n], 3.0 * theta[4 + 4 * n:]])
    out["scale (r,v,3m)"] = sc
    return {k: v / np.linalg.norm(v) for k, v in out.items()}


def captured(vec, null_basis):
    """How much of `vec` lies in the unobservable subspace (1 = invisible, 0 = observable)."""
    return float(np.linalg.norm(null_basis @ vec))


def main():
    OUT.mkdir(parents=True, exist_ok=True)
    levels, meta = load_pool(ROOT / "data" / "pools" / "val_seen.npz")
    level = index_level(levels, 12)
    n = int(np.sum(np.asarray(level.active)))
    bp, bv = E.snapshot_state(level, 0)
    ship_pos = (np.asarray(bp)[0] + np.array([260.0, 130.0])).astype(np.float64)
    ship_vel = np.array(E.parking_velocity(level, bp, bv, jnp.asarray(ship_pos), 1.0), np.float64)

    cases = {
        "coasting (no thrust)": dict(thrust_mag=0.0),
        "thrusting at 2 u/s^2": dict(thrust_mag=2.0),
        "coasting + light delay (c=3000 u/s)": dict(thrust_mag=0.0, c_light=3000.0),
        "coasting + light delay (c=800 u/s)": dict(thrust_mag=0.0, c_light=800.0),
    }
    print(f"level val_seen[12] ({meta[12]['family']}), {n} bodies -> {4 + 5 * n} unknowns, "
          f"bearings only, 25 s window, 40 samples\n")
    syms = None
    report = {}
    header = None
    for name, kw in cases.items():
        s, theta, Vt = spectrum(level, n, jnp.asarray(ship_pos), jnp.asarray(ship_vel), **kw)
        tol = s[0] * 1e-9
        null_dim = int(np.sum(s <= tol))
        null_basis = Vt[len(s) - null_dim:] if null_dim else np.zeros((0, len(theta)))
        syms = syms or symmetries(theta, n)
        caps = {k: captured(v, null_basis) for k, v in syms.items()}
        if header is None:
            header = list(caps)
            print(f"{'case':34s} {'null dim':>8s}  " + "  ".join(f"{k:>14s}" for k in header))
        print(f"{name:34s} {null_dim:8d}  " + "  ".join(f"{caps[k]:14.3f}" for k in header))
        report[name] = dict(null_dim=null_dim, unknowns=int(len(theta)),
                            captured={k: caps[k] for k in header},
                            singular_values=[float(x) for x in s[-8:]])
    (OUT / "observability.json").write_text(json.dumps(report, indent=2))
    print("\nEach number is how much of that symmetry lies in the unobservable subspace:")
    print("1.000 = bearings cannot see it at all, 0.000 = fully determined.")


if __name__ == "__main__":
    main()
