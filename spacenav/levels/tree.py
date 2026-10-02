"""Hierarchical orbit trees -> padded Level arrays (numpy, host side).

A `Node` is a body plus the things orbiting it.  Children are placed from the
inside out, each on a Kepler orbit around the barycentre of everything already
placed in that subsystem (Jacobi coordinates), so binaries, circumbinary
planets and moons-of-planets-of-stars all come out close to their intended
orbits.
"""

from dataclasses import dataclass, field

import numpy as np

from spacenav import constants as C


@dataclass
class Zone:
    r_in: float
    r_out: float
    kind: int
    strength: float
    spin: float = 0.0


@dataclass
class Orbit:
    a: float
    e: float = 0.0
    sign: float = 1.0          # +1 prograde (counter-clockwise), -1 retrograde
    phase: float | None = None  # true anomaly; random if None
    omega: float | None = None  # argument of periapsis; random if None


@dataclass
class Node:
    kind: int
    mass: float = 0.0
    radius: float = 0.0
    atmo_h: float = 0.0
    atmo_rho: float = 0.0
    lum: float = 0.0
    zones: list = field(default_factory=list)
    children: list = field(default_factory=list)   # list of (Node, Orbit)
    orbit_a: float = float("inf")                  # set when attached to a host
    host_mass: float = 0.0

    def add(self, child: "Node", orbit: Orbit) -> "Node":
        child.orbit_a = orbit.a
        child.host_mass = self.total_mass()
        self.children.append((child, orbit))
        return child

    def hill(self) -> float:
        """Hill radius around this body (infinite for the root)."""
        if not np.isfinite(self.orbit_a) or self.host_mass <= 0:
            return float("inf")
        return self.orbit_a * (self.mass / (3 * self.host_mass)) ** (1 / 3)

    def clear_radius(self) -> float:
        return self.radius + self.atmo_h

    def free_orbit(self, rng, a_lo, a_hi, own_extent=0.0, pad=8.0):
        """A random orbit radius in [a_lo, a_hi] that keeps clear of existing
        children (their extent plus a few Hill radii); None if there is none."""
        a_hi = min(a_hi, 0.35 * self.hill())
        bands = []
        for c, o in self.children:
            h = c.hill() if c.mass > 0 else 0.0
            half = c.extent() + 3 * min(h, 1e6) + own_extent + pad
            bands.append((o.a * (1 - o.e) - half, o.a * (1 + o.e) + half))
        for _ in range(64):
            if a_hi <= a_lo:
                return None
            a = rng.uniform(a_lo, a_hi)
            if all(not (lo < a < hi) for lo, hi in bands):
                return float(a)
        return None

    def total_mass(self) -> float:
        return self.mass + sum(c.total_mass() for c, _ in self.children)

    def extent(self) -> float:
        """Rough outer radius of this subsystem (for spacing decisions)."""
        r = self.radius + self.atmo_h
        for c, o in self.children:
            r = max(r, o.a * (1 + o.e) + c.extent())
        for z in self.zones:
            r = max(r, z.r_out)
        return r


def kepler_state(mu, orbit: Orbit, rng):
    f = rng.uniform(0, 2 * np.pi) if orbit.phase is None else orbit.phase
    w = rng.uniform(0, 2 * np.pi) if orbit.omega is None else orbit.omega
    p = orbit.a * (1 - orbit.e**2)
    r = p / (1 + orbit.e * np.cos(f))
    pos = r * np.array([np.cos(f), np.sin(f)])
    vel = np.sqrt(mu / p) * np.array([-np.sin(f), orbit.e + np.cos(f)])
    rot = np.array([[np.cos(w), -np.sin(w)], [np.sin(w), np.cos(w)]])
    pos, vel = rot @ pos, rot @ vel
    if orbit.sign < 0:
        pos, vel = pos * [1, -1], vel * [1, -1]
    return pos, vel


def hyperbolic_state(mu, r_peri, v_inf, r_start, rng):
    """State on a hyperbolic flyby, inbound at distance r_start."""
    a = mu / v_inf**2                      # |semi-major axis|
    e = 1 + r_peri / a
    p = a * (e**2 - 1)
    f_inf = np.arccos(-1 / e)
    # solve r(f) = r_start on the inbound leg
    cf = (p / r_start - 1) / e
    f = -np.arccos(np.clip(cf, -1, 1))
    f = max(f, -f_inf + 1e-3)
    orbit = Orbit(a=-a, e=e, phase=f, omega=rng.uniform(0, 2 * np.pi),
                  sign=rng.choice([-1, 1]))
    r = p / (1 + e * np.cos(f))
    pos = r * np.array([np.cos(f), np.sin(f)])
    vel = np.sqrt(mu / p) * np.array([-np.sin(f), e + np.cos(f)])
    w = orbit.omega
    rot = np.array([[np.cos(w), -np.sin(w)], [np.sin(w), np.cos(w)]])
    pos, vel = rot @ pos, rot @ vel
    if orbit.sign < 0:
        pos, vel = pos * [1, -1], vel * [1, -1]
    return pos, vel


def flatten(root: Node, rng):
    """Returns a list of body dicts (barycentric pos/vel) and zone dicts."""
    bodies, zones = [], []

    def build(node, parent_idx):
        idx = len(bodies)
        bodies.append(dict(kind=node.kind, mass=node.mass, radius=node.radius,
                           atmo_h=node.atmo_h, atmo_rho=node.atmo_rho, lum=node.lum,
                           parent=parent_idx, pos=np.zeros(2), vel=np.zeros(2)))
        for z in node.zones:
            zones.append(dict(anchor=idx, r_in=z.r_in, r_out=z.r_out, kind=z.kind,
                              strength=z.strength, spin=z.spin))
        members = [idx]
        m_sys = node.mass
        for child, orbit in sorted(node.children, key=lambda co: co[1].a):
            first = len(bodies)
            m_c = build(child, idx)
            child_members = list(range(first, len(bodies)))
            ms = np.array([bodies[i]["mass"] for i in members])
            ps = np.array([bodies[i]["pos"] for i in members])
            vs = np.array([bodies[i]["vel"] for i in members])
            if m_sys > 0:
                cm_p, cm_v = (ms[:, None] * ps).sum(0) / m_sys, (ms[:, None] * vs).sum(0) / m_sys
            else:
                cm_p, cm_v = bodies[idx]["pos"], bodies[idx]["vel"]
            mu = m_sys + m_c
            rel_p, rel_v = kepler_state(mu, orbit, rng)
            frac_c = m_c / mu if mu > 0 else 0.0
            for i in members:            # recoil of the inner subsystem
                bodies[i]["pos"] = bodies[i]["pos"] - frac_c * rel_p
                bodies[i]["vel"] = bodies[i]["vel"] - frac_c * rel_v
            off_p = cm_p + (1 - frac_c) * rel_p
            off_v = cm_v + (1 - frac_c) * rel_v
            for i in child_members:      # child subsystem is built around its own barycentre
                bodies[i]["pos"] = bodies[i]["pos"] + off_p
                bodies[i]["vel"] = bodies[i]["vel"] + off_v
            members += child_members
            m_sys += m_c
        # recentre this subsystem on its barycentre
        ms = np.array([bodies[i]["mass"] for i in members])
        if ms.sum() > 0:
            ps = np.array([bodies[i]["pos"] for i in members])
            vs = np.array([bodies[i]["vel"] for i in members])
            cp, cv = (ms[:, None] * ps).sum(0) / ms.sum(), (ms[:, None] * vs).sum(0) / ms.sum()
            for i in members:
                bodies[i]["pos"] = bodies[i]["pos"] - cp
                bodies[i]["vel"] = bodies[i]["vel"] - cv
        return m_sys

    build(root, -1)
    return bodies, zones


def to_arrays(bodies, zones, n_max=C.MAX_BODIES, z_max=C.MAX_ZONES):
    if len(bodies) > n_max or len(zones) > z_max:
        raise ValueError(f"too many bodies/zones: {len(bodies)}/{len(zones)}")
    n, z = len(bodies), len(zones)
    out = dict(
        mass=np.zeros(n_max, np.float32), radius=np.zeros(n_max, np.float32),
        atmo_h=np.zeros(n_max, np.float32), atmo_rho=np.zeros(n_max, np.float32),
        lum=np.zeros(n_max, np.float32), kind=np.zeros(n_max, np.int32),
        parent=np.full(n_max, -1, np.int32), active=np.zeros(n_max, bool),
        pos=np.zeros((n_max, 2), np.float32), vel=np.zeros((n_max, 2), np.float32),
        zone_anchor=np.full(z_max, -1, np.int32), zone_r_in=np.zeros(z_max, np.float32),
        zone_r_out=np.zeros(z_max, np.float32), zone_kind=np.zeros(z_max, np.int32),
        zone_strength=np.zeros(z_max, np.float32), zone_spin=np.zeros(z_max, np.float32),
        zone_active=np.zeros(z_max, bool),
    )
    for i, b in enumerate(bodies):
        for k in ("mass", "radius", "atmo_h", "atmo_rho", "lum", "kind", "parent", "pos", "vel"):
            out[k][i] = b[k]
    out["active"][:n] = True
    for i, zz in enumerate(zones):
        out["zone_anchor"][i] = zz["anchor"]
        out["zone_r_in"][i] = zz["r_in"]
        out["zone_r_out"][i] = zz["r_out"]
        out["zone_kind"][i] = zz["kind"]
        out["zone_strength"][i] = zz["strength"]
        out["zone_spin"][i] = zz["spin"]
    out["zone_active"][:z] = True
    return out
