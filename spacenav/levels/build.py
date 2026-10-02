"""Sample levels from a family, validate them on the GPU, and store snapshots.

A level is accepted only if, over LEVEL_HORIZON seconds, no solid bodies (or a
station and a solid body) touch, every orbiting body stays within a band
around its initial distance from its parent, and the integrator conserves
energy well.  Accepted levels keep body snapshots every SNAPSHOT_INTERVAL
seconds; episodes start from one of the snapshots that leaves a full
MAX_EPISODE_TIME before the horizon.
"""

from functools import partial

import jax
import jax.numpy as jnp
import numpy as np

from spacenav import constants as C
from spacenav.levels.families import ALL_FAMILIES, FAMILY_IDS
from spacenav.levels.tree import flatten, hyperbolic_state, to_arrays
from spacenav.physics import bodies_step, energy
from spacenav.types import Level

N_SNAPSHOTS = int((C.LEVEL_HORIZON - C.MAX_EPISODE_TIME) / C.SNAPSHOT_INTERVAL) + 1
STEPS_PER_SNAPSHOT = int(round(C.SNAPSHOT_INTERVAL / C.PHYS_DT))
N_CHUNKS = int(round(C.LEVEL_HORIZON / C.SNAPSHOT_INTERVAL))

# validation thresholds
PARENT_BAND = (0.6, 1.6)
ENERGY_TOL = 2e-3
CLEARANCE = 3.0


def sample_raw(family: str, seed: int):
    """One unvalidated level as a dict of padded numpy arrays."""
    rng = np.random.default_rng(seed)
    root, extras = ALL_FAMILIES[family](rng)
    bodies, zones = flatten(root, rng)
    exclude = [False] * len(bodies)
    if extras:
        m_sys = sum(b["mass"] for b in bodies)
        for ex in extras:
            fb = ex.pop("flyby")
            pos, vel = hyperbolic_state(m_sys + ex["mass"], fb["r_peri"], fb["v_inf"],
                                        fb["r_start"], rng)
            bodies.append(dict(ex, pos=pos, vel=vel))
            exclude.append(True)
    arr = to_arrays(bodies, zones)
    arr["radius_exclude"] = np.zeros(C.MAX_BODIES, bool)
    arr["radius_exclude"][: len(exclude)] = exclude
    arr["family"] = np.int32(FAMILY_IDS[family])
    arr["seed"] = np.int64(seed)
    arr["n_bodies"] = len(bodies)
    return arr


def _static_level(arr, pos, vel):
    """A Level whose snapshots are just the given state (used while validating)."""
    return Level(
        mass=arr["mass"], radius=arr["radius"], atmo_h=arr["atmo_h"], atmo_rho=arr["atmo_rho"],
        lum=arr["lum"], kind=arr["kind"], parent=arr["parent"], active=arr["active"],
        zone_anchor=arr["zone_anchor"], zone_r_in=arr["zone_r_in"], zone_r_out=arr["zone_r_out"],
        zone_kind=arr["zone_kind"], zone_strength=arr["zone_strength"], zone_spin=arr["zone_spin"],
        zone_active=arr["zone_active"], snap_pos=pos[None], snap_vel=vel[None],
        level_radius=jnp.float32(0), family=arr["family"])


def _pair_masks(arr):
    solid = arr["active"] & (arr["radius"] > 0) & (arr["kind"] != C.KIND_TRACER) \
        & (arr["kind"] != C.KIND_STATION)
    station = arr["active"] & (arr["kind"] == C.KIND_STATION)
    reach = arr["radius"] + arr["atmo_h"]
    need = reach[:, None] + reach[None, :] + CLEARANCE
    pair = (solid[:, None] & solid[None, :]) | (solid[:, None] & station[None, :]) \
        | (station[:, None] & solid[None, :])
    pair = pair & ~jnp.eye(pair.shape[0], dtype=bool)
    return pair, need


@jax.jit
def _validate_one(arr):
    level = _static_level(arr, arr["pos"], arr["vel"])
    pair, need = _pair_masks(arr)
    # tracers only carry debris clouds; letting them drift is harmless
    has_parent = arr["active"] & (arr["parent"] >= 0) & (arr["kind"] != C.KIND_TRACER)
    par = jnp.maximum(arr["parent"], 0)

    def parent_dist(p):
        return jnp.sqrt(jnp.sum((p - p[par]) ** 2, axis=-1) + 1e-9)

    d0 = parent_dist(arr["pos"])
    e0 = energy(level, arr["pos"], arr["vel"])

    def stats(p):
        d = jnp.sqrt(jnp.sum((p[:, None] - p[None, :]) ** 2, axis=-1) + 1e-9)
        margin = jnp.min(jnp.where(pair, d - need, jnp.inf))
        ratio = parent_dist(p) / d0
        rmin = jnp.min(jnp.where(has_parent, ratio, jnp.inf))
        rmax = jnp.max(jnp.where(has_parent, ratio, -jnp.inf))
        return margin, rmin, rmax, jnp.sqrt(jnp.sum(p * p, axis=-1))

    def step(carry, _):
        p, v, margin, rmin, rmax, rad = carry
        p, v = bodies_step(p, v, level)
        m, lo, hi, r = stats(p)
        return (p, v, jnp.minimum(margin, m), jnp.minimum(rmin, lo),
                jnp.maximum(rmax, hi), jnp.maximum(rad, r)), None

    def chunk(carry, _):
        carry, _ = jax.lax.scan(step, carry, None, length=STEPS_PER_SNAPSHOT)
        return carry, (carry[0], carry[1])

    m0, lo0, hi0, r0 = stats(arr["pos"])
    carry = (arr["pos"], arr["vel"], m0, lo0, hi0, r0)
    carry, (ps, vs) = jax.lax.scan(chunk, carry, None, length=N_CHUNKS)
    p, v, margin, rmin, rmax, rad = carry
    drift = jnp.abs((energy(level, p, v) - e0) / e0)
    snap_pos = jnp.concatenate([arr["pos"][None], ps[: N_SNAPSHOTS - 1]], axis=0)
    snap_vel = jnp.concatenate([arr["vel"][None], vs[: N_SNAPSHOTS - 1]], axis=0)
    finite = jnp.all(jnp.isfinite(p)) & jnp.isfinite(drift)
    return dict(margin=margin, parent_min=rmin, parent_max=rmax, max_r=rad, drift=drift,
                finite=finite, snap_pos=snap_pos, snap_vel=snap_vel)


_validate_batch = jax.jit(jax.vmap(_validate_one))

_ARR_KEYS = ("mass", "radius", "atmo_h", "atmo_rho", "lum", "kind", "parent", "active", "pos",
             "vel", "zone_anchor", "zone_r_in", "zone_r_out", "zone_kind", "zone_strength",
             "zone_spin", "zone_active", "family")


def level_radius(arr, max_r):
    """Play-area radius: everything that matters, plus a margin."""
    reach = arr["radius"] + arr["atmo_h"]
    zr = np.zeros(C.MAX_BODIES, np.float32)
    for a, r, act in zip(arr["zone_anchor"], arr["zone_r_out"], arr["zone_active"]):
        if act and a >= 0:
            zr[a] = max(zr[a], r)
    keep = arr["active"] & ~arr["radius_exclude"]
    return float(np.max(np.where(keep, max_r + np.maximum(reach, zr), 0.0)) + 100.0)


def build_levels(family: str, n: int, seed0: int = 0, batch: int = 128, max_tries: int = 20,
                 parent_band=PARENT_BAND, verbose=False):
    """Returns (Level stacked over n, list of per-level metadata dicts, rejection stats)."""
    if family == "rogue_flyby":
        parent_band = (0.3, 3.0)
    accepted, meta = [], []
    reasons = dict(tried=0, collision=0, orbit=0, energy=0, nonfinite=0)
    seed = seed0
    for _ in range(max_tries):
        if len(accepted) >= n:
            break
        raws = []
        for _ in range(batch):
            raws.append(sample_raw(family, seed))
            seed += 1
        stacked = {k: jnp.asarray(np.stack([r[k] for r in raws])) for k in _ARR_KEYS}
        out = jax.device_get(_validate_batch(stacked))
        for i, raw in enumerate(raws):
            reasons["tried"] += 1
            ok = True
            if not out["finite"][i]:
                reasons["nonfinite"] += 1; ok = False
            elif out["margin"][i] <= 0:
                reasons["collision"] += 1; ok = False
            elif not (parent_band[0] < out["parent_min"][i] and out["parent_max"][i] < parent_band[1]):
                reasons["orbit"] += 1; ok = False
            elif out["drift"][i] > ENERGY_TOL:
                reasons["energy"] += 1; ok = False
            if ok and len(accepted) < n:
                raw = dict(raw)
                raw["snap_pos"] = out["snap_pos"][i]
                raw["snap_vel"] = out["snap_vel"][i]
                raw["level_radius"] = np.float32(level_radius(raw, out["max_r"][i]))
                accepted.append(raw)
                meta.append(dict(family=family, seed=int(raw["seed"]), n_bodies=int(raw["n_bodies"]),
                                 drift=float(out["drift"][i]), margin=float(out["margin"][i]),
                                 level_radius=float(raw["level_radius"])))
        if verbose:
            print(f"{family}: {len(accepted)}/{n} accepted after {reasons['tried']} tries {reasons}")
    if len(accepted) < n:
        raise RuntimeError(f"{family}: only {len(accepted)}/{n} levels accepted: {reasons}")
    return stack_levels(accepted), meta, reasons


def stack_levels(raws) -> Level:
    def s(k, dtype=None):
        return jnp.asarray(np.stack([r[k] for r in raws]), dtype=dtype)
    return Level(mass=s("mass"), radius=s("radius"), atmo_h=s("atmo_h"), atmo_rho=s("atmo_rho"),
                 lum=s("lum"), kind=s("kind"), parent=s("parent"), active=s("active"),
                 zone_anchor=s("zone_anchor"), zone_r_in=s("zone_r_in"), zone_r_out=s("zone_r_out"),
                 zone_kind=s("zone_kind"), zone_strength=s("zone_strength"),
                 zone_spin=s("zone_spin"), zone_active=s("zone_active"),
                 snap_pos=s("snap_pos", jnp.float32), snap_vel=s("snap_vel", jnp.float32),
                 level_radius=s("level_radius", jnp.float32), family=s("family"))


def save_pool(path, levels: Level, meta):
    np.savez_compressed(path, meta=np.array(meta, dtype=object),
                        **{k: np.asarray(v) for k, v in levels._asdict().items()})


def load_pool(path):
    z = np.load(path, allow_pickle=True)
    levels = Level(**{k: jnp.asarray(z[k]) for k in Level._fields})
    return levels, list(z["meta"])


def index_level(levels: Level, i) -> Level:
    return jax.tree.map(lambda x: x[i], levels)


def concat_levels(pools):
    return jax.tree.map(lambda *xs: jnp.concatenate(xs, axis=0), *pools)


def custom_level(root, rng=None, level_radius=1000.0, n_snapshots=1) -> Level:
    """A single, unvalidated Level from an orbit tree (for tests and sandboxes).

    Snapshots are taken every SNAPSHOT_INTERVAL by integrating the bodies."""
    from spacenav.physics import roll_bodies
    rng = rng or np.random.default_rng(0)
    bodies, zones = flatten(root, rng)
    arr = to_arrays(bodies, zones)
    arr["family"] = np.int32(-1)
    lv = _static_level({k: jnp.asarray(v) for k, v in arr.items()}, jnp.asarray(arr["pos"]),
                       jnp.asarray(arr["vel"]))
    pos, vel = [arr["pos"]], [arr["vel"]]
    for _ in range(n_snapshots - 1):
        ps, vs = roll_bodies(lv, jnp.asarray(pos[-1]), jnp.asarray(vel[-1]), STEPS_PER_SNAPSHOT)
        pos.append(np.asarray(ps[-1])); vel.append(np.asarray(vs[-1]))
    return lv._replace(snap_pos=jnp.asarray(np.stack(pos)), snap_vel=jnp.asarray(np.stack(vel)),
                       level_radius=jnp.float32(level_radius))
