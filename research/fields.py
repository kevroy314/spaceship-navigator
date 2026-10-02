"""Potential and chaos maps for a spacenav level.

Three views of the same level, all computed with the real simulator:

  1. Instantaneous pseudo-potential in a rotating frame fitted to the dominant
     pair (gravitational + centrifugal).  In the circular restricted 3-body
     problem this is exact and its level sets are the zero-velocity curves that
     gate transport; in our eccentric, many-body levels it is only an
     instantaneous approximation, so it is drawn as context, not as truth.
  2. Finite-time Lyapunov exponent (FTLE) over launch positions: release a
     ballistic test particle at each pixel with the local circular velocity,
     integrate, and measure how fast neighbouring launches separate.  Ridges are
     the transport barriers — the honest "where does it get chaotic" picture for
     a non-autonomous system.
  3. Escape/impact time, which says where the level is simply lethal.

    python -m research.fields --pool val_seen --level 12 --grid 256 --seconds 40
"""

import argparse
from pathlib import Path

import jax
import jax.numpy as jnp
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

from spacenav import constants as C
from spacenav import env as E
from spacenav import physics as P
from spacenav.levels.build import index_level, load_pool

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "research" / "out"


def pseudo_potential(level, bp, bv, X, Y):
    """Gravitational + centrifugal potential in a frame co-rotating with the dominant pair."""
    m = np.asarray(level.mass) * np.asarray(level.active)
    order = np.argsort(-m)
    i, j = order[0], order[1] if len(order) > 1 else order[0]
    r = np.asarray(bp)[j] - np.asarray(bp)[i]
    v = np.asarray(bv)[j] - np.asarray(bv)[i]
    sep = np.linalg.norm(r) + 1e-9
    omega = np.cross(r, v) / sep**2                      # instantaneous angular rate
    centre = (m[i] * np.asarray(bp)[i] + m[j] * np.asarray(bp)[j]) / (m[i] + m[j])
    pts = np.stack([X, Y], -1)
    U = np.zeros_like(X)
    for k in range(len(m)):
        if m[k] <= 0:
            continue
        d = np.linalg.norm(pts - np.asarray(bp)[k], axis=-1)
        U -= m[k] / np.maximum(d, np.asarray(level.radius)[k])
    U -= 0.5 * omega**2 * np.sum((pts - centre) ** 2, -1)
    return U, float(omega)


def ftle_and_escape(level, snapshot, X, Y, seconds):
    """Ballistic launches on a grid: FTLE from neighbour separation, plus escape time."""
    bp0, bv0 = E.snapshot_state(level, snapshot)
    pts = jnp.stack([jnp.asarray(X).ravel(), jnp.asarray(Y).ravel()], -1)
    vel = jax.vmap(lambda x: E.parking_velocity(level, bp0, bv0, x, 1.0))(pts)
    steps = int(seconds / C.PHYS_DT)
    R = level.level_radius * C.OUT_OF_BOUNDS_FACTOR

    def step(carry, _):
        bp, bv, x, v, alive, t_end = carry
        bp1, bv1, x1, v1 = jax.vmap(
            lambda xx, vv: P.substep(level, bp, bv, xx, vv, jnp.zeros(2)), in_axes=(0, 0))(x, v)
        bp1, bv1 = bp1[0], bv1[0]                     # bodies are shared across particles
        d = jnp.sqrt(jnp.sum((x1[:, None, :] - bp1[None]) ** 2, -1))
        hit = jnp.any((d < level.radius[None]) & level.active[None], axis=-1)
        gone = jnp.sum(x1 * x1, -1) > R ** 2
        dead = hit | gone
        t_end = jnp.where(alive & dead, t_end + C.PHYS_DT, t_end)
        alive = alive & ~dead
        t_end = jnp.where(alive, t_end + C.PHYS_DT, t_end)
        # frozen once dead, so the map shows where it died
        keep = lambda a, b: jnp.where(alive[:, None], a, b)
        return (bp1, bv1, keep(x1, x), keep(v1, v), alive, t_end), None

    init = (bp0, bv0, pts, vel, jnp.ones(len(pts), bool), jnp.zeros(len(pts)))
    (_, _, xf, _, alive, t_end), _ = jax.lax.scan(step, init, None, length=steps)
    return np.asarray(xf), np.asarray(alive), np.asarray(t_end)


def ftle_from_flowmap(xf, shape, dx, seconds):
    """sigma = (1/T) log sqrt(lambda_max(J^T J)) from finite differences of the flow map."""
    fx = xf[:, 0].reshape(shape)
    fy = xf[:, 1].reshape(shape)
    dfx_dx, dfx_dy = np.gradient(fx, dx, dx)
    dfy_dx, dfy_dy = np.gradient(fy, dx, dx)
    a = dfx_dx**2 + dfy_dx**2
    b = dfx_dx * dfx_dy + dfy_dx * dfy_dy
    d = dfx_dy**2 + dfy_dy**2
    tr, det = a + d, a * d - b * b
    lam = tr / 2 + np.sqrt(np.maximum(tr**2 / 4 - det, 0))
    return np.log(np.maximum(lam, 1e-12)) / (2 * seconds)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--pool", default="val_seen")
    ap.add_argument("--level", type=int, default=12)
    ap.add_argument("--grid", type=int, default=256)
    ap.add_argument("--seconds", type=float, default=40.0)
    ap.add_argument("--snapshot", type=int, default=0)
    ap.add_argument("--tag", default=None)
    args = ap.parse_args()
    OUT.mkdir(parents=True, exist_ok=True)

    levels, meta = load_pool(ROOT / "data" / "pools" / f"{args.pool}.npz")
    level = index_level(levels, args.level)
    family = meta[args.level]["family"]
    bp, bv = E.snapshot_state(level, args.snapshot)
    R = float(level.level_radius)
    g = np.linspace(-R, R, args.grid)
    X, Y = np.meshgrid(g, g)

    U, omega = pseudo_potential(level, bp, bv, X, Y)
    xf, alive, t_end = ftle_and_escape(level, args.snapshot, X, Y, args.seconds)
    sigma = ftle_from_flowmap(xf, X.shape, g[1] - g[0], args.seconds)
    surv = alive.reshape(X.shape)
    tend = t_end.reshape(X.shape)

    fig, axes = plt.subplots(1, 3, figsize=(16.5, 5.6), facecolor="#0a0e16")
    bpn, rad, act = np.asarray(bp), np.asarray(level.radius), np.asarray(level.active)
    for ax in axes:
        ax.set_facecolor("#070a11")
        ax.set_xticks([]); ax.set_yticks([]); ax.set_aspect("equal")
        for k in np.nonzero(act)[0]:
            if rad[k] > 0:
                ax.add_patch(plt.Circle(bpn[k], max(rad[k], R * 0.006), color="#e4ebf5", zorder=5))

    v = U[np.isfinite(U)]
    lo, hi = np.percentile(v, 2), np.percentile(v, 75)
    axes[0].contourf(X, Y, np.clip(U, lo, hi), levels=40, cmap="magma")
    axes[0].contour(X, Y, np.clip(U, lo, hi), levels=18, colors="#ffffff", linewidths=0.35, alpha=0.35)
    axes[0].set_title(f"pseudo-potential (rotating frame, ω={omega:.3f} rad/s)",
                      color="#a9b6c9", fontsize=10)

    s = np.where(surv, sigma, np.nan)
    axes[1].imshow(s, origin="lower", extent=[-R, R, -R, R], cmap="inferno",
                   vmin=np.nanpercentile(s, 5), vmax=np.nanpercentile(s, 99))
    axes[1].set_title(f"FTLE over launch points ({args.seconds:.0f} s ballistic)",
                      color="#a9b6c9", fontsize=10)

    axes[2].imshow(np.where(surv, np.nan, tend), origin="lower", extent=[-R, R, -R, R],
                   cmap="viridis", vmin=0, vmax=args.seconds)
    axes[2].set_title("time to impact or escape (blank = survives)", color="#a9b6c9", fontsize=10)

    fig.suptitle(f"{args.pool} level {args.level} · {family}", color="#e4ebf5", fontsize=12)
    fig.tight_layout()
    tag = args.tag or f"{args.pool}_{args.level}"
    path = OUT / f"fields_{tag}.png"
    fig.savefig(path, dpi=110, facecolor=fig.get_facecolor())
    frac = float(np.mean(~surv))
    print(f"wrote {path}  (family={family}, {frac:.0%} of launch points die within {args.seconds:.0f}s, "
          f"FTLE median {np.nanmedian(s):.4f} /s, p95 {np.nanpercentile(s, 95):.4f} /s)")


if __name__ == "__main__":
    main()
