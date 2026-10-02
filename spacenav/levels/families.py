"""Scenario families.  Each generator returns (root Node, extras) for one level.

`extras` may carry free bodies (e.g. a rogue on a hyperbolic flyby) that are
not part of the orbit tree.

Families are split into TRAIN and HOLDOUT.  Held-out families are never seen in
training and serve as the "blind new scenario" benchmark.
"""

import numpy as np

from spacenav import constants as C
from spacenav.levels.tree import Node, Orbit, Zone

C_CLEAR = 4.0   # clearance beyond the validator's minimum


def U(rng, lo, hi):
    return float(rng.uniform(lo, hi))


def logU(rng, lo, hi):
    return float(np.exp(rng.uniform(np.log(lo), np.log(hi))))


def hill(a, m, M):
    return a * (m / (3 * M)) ** (1 / 3)


# ---------------------------------------------------------------------------
# building blocks
# ---------------------------------------------------------------------------

def make_star(rng, gm=None, lum_k=0.15):
    gm = gm or logU(rng, 2.5e4, 9e4)
    r = 14 + 8 * (gm / 5e4) ** 0.5
    return Node(C.KIND_STAR, gm, r, lum=lum_k * gm)


def make_rocky(rng, M_host):
    m = M_host * logU(rng, 2e-5, 4e-4)
    atmo = rng.random() < 0.55
    return Node(C.KIND_ROCKY, m, U(rng, 4, 7),
                atmo_h=U(rng, 3, 7) if atmo else 0.0, atmo_rho=1.0 if atmo else 0.0)


def make_giant(rng, M_host):
    m = M_host * logU(rng, 8e-4, 6e-3)
    return Node(C.KIND_GIANT, m, U(rng, 8, 15), atmo_h=U(rng, 5, 9), atmo_rho=1.5)


def make_moon(rng, planet: Node, big=False):
    m = planet.mass * (logU(rng, 3e-3, 2e-2) if big else logU(rng, 3e-4, 5e-3))
    atmo = big and rng.random() < 0.4
    return Node(C.KIND_MOON, m, U(rng, 3.5, 6) if big else U(rng, 2, 4),
                atmo_h=U(rng, 3, 6) if atmo else 0.0, atmo_rho=1.2 if atmo else 0.0)


def add_moons(rng, planet: Node, n_max, big_frac=0.3):
    """Moons inside the planet's Hill sphere (the planet must already be attached)."""
    a = planet.clear_radius() + C_CLEAR + U(rng, 8, 14)
    for _ in range(int(rng.integers(0, n_max + 1))):
        moon = make_moon(rng, planet, big=rng.random() < big_frac)
        a_moon = a + moon.clear_radius()
        if a_moon > 0.35 * planet.hill():
            break
        planet.add(moon, Orbit(a_moon, e=U(rng, 0, 0.03), sign=1 if rng.random() < 0.85 else -1))
        a = a_moon * U(rng, 1.5, 1.9) + moon.clear_radius()
    return planet


def add_station(rng, host: Node, a_lo=None, a_hi=None):
    a_lo = a_lo or host.clear_radius() + C_CLEAR + U(rng, 6, 12)
    a_hi = a_hi or a_lo + 25
    a = host.free_orbit(rng, a_lo, a_hi)
    if a is None:
        return None
    return host.add(Node(C.KIND_STATION, 0.0, 0.0),
                    Orbit(a, e=U(rng, 0, 0.05), sign=1 if rng.random() < 0.8 else -1))


def add_debris_cloud(rng, host: Node, a_lo, a_hi, strength=None, size=None):
    size = size or U(rng, 25, 60)
    a = host.free_orbit(rng, a_lo, a_hi, own_extent=size * 0.5)
    if a is None:
        return None
    tracer = Node(C.KIND_TRACER, 0.0, 0.0,
                  zones=[Zone(0.0, size, C.ZONE_DEBRIS, strength or U(rng, 0.4, 1.2), 0.0)])
    return host.add(tracer, Orbit(a, e=U(rng, 0, 0.1)))


def add_sensor(rng, host: Node, r_out=None):
    host.zones.append(Zone(0.0, r_out or host.radius + U(rng, 40, 110), C.ZONE_SENSOR, 1.0))


def planetary_system(rng, star: Node, a0, a_max, n_max, moons=2, giant_frac=0.35):
    """Fill `star` with planets from a0 outwards; returns [(a, planet)]."""
    placed = []
    a = a0
    prev = None
    while len(placed) < n_max and a < a_max:
        p = make_giant(rng, star.mass) if rng.random() < giant_frac else make_rocky(rng, star.mass)
        if prev is not None:     # keep >= ~7 mutual Hill radii from the previous planet
            pa, pp = prev
            r_hm = ((p.mass + pp.mass) / (3 * star.mass)) ** (1 / 3) * (a + pa) / 2
            a = max(a, pa + 7 * r_hm, pa + 3 * pp.extent() + 3 * p.clear_radius())
        if a >= a_max:
            break
        star.add(p, Orbit(a, e=U(rng, 0, 0.06), sign=1))
        add_moons(rng, p, moons if p.kind == C.KIND_GIANT else 1)
        placed.append((a, p))
        prev = (a, p)
        a = a * U(rng, 1.35, 1.75)
    return placed


def sprinkle(rng, root: Node, placed, n_stations=(1, 4), n_sensors=(0, 2), root_band=None):
    """Stations around random bodies (where an orbit fits) and sensor zones."""
    hosts = [root] + [p for _, p in placed]
    for p in list(hosts[1:]):
        hosts += [c for c, _ in p.children if c.kind == C.KIND_MOON]
    root_band = root_band or (root.clear_radius() + 60, max([a for a, _ in placed] + [300]) * 1.1)
    want = int(rng.integers(n_stations[0], n_stations[1] + 1))
    made = 0
    for _ in range(want * 6):
        if made >= want:
            break
        h = hosts[int(rng.integers(len(hosts)))]
        st = add_station(rng, root, *root_band) if h is root else add_station(rng, h)
        made += st is not None
    if made == 0:
        add_station(rng, root, *root_band)
    for _ in range(int(rng.integers(n_sensors[0], n_sensors[1] + 1))):
        h = hosts[1 + int(rng.integers(len(hosts) - 1))] if len(hosts) > 1 else root
        add_sensor(rng, h)


# ---------------------------------------------------------------------------
# training families
# ---------------------------------------------------------------------------

def belt_in_gap(rng, star, placed, strength=(0.15, 0.4)):
    if len(placed) < 3:
        return
    i = int(rng.integers(1, len(placed)))
    a0, a1 = placed[i - 1][0], placed[i][0]
    lo = a0 + 3 * placed[i - 1][1].extent() + 20
    hi = a1 - 3 * placed[i][1].extent() - 20
    if hi - lo > 30:
        mid, half = (lo + hi) / 2, (hi - lo) * U(rng, 0.25, 0.45)
        star.zones.append(Zone(mid - half, mid + half, C.ZONE_DEBRIS, U(rng, *strength), 1.0))


def sol_like(rng):
    star = make_star(rng)
    placed = planetary_system(rng, star, U(rng, 70, 150), 950, int(rng.integers(3, 9)),
                              giant_frac=U(rng, 0.15, 0.55))
    if rng.random() < 0.6:
        belt_in_gap(rng, star, placed)
    for _ in range(int(rng.integers(0, 3))):
        add_debris_cloud(rng, star, 150, 850)
    sprinkle(rng, star, placed)
    return star, []


def moon_system(rng, planet: Node, a0, a_max, n, mass_range, radius_range, atmo_p=0.25,
                retro_p=0.1):
    placed = []
    a = a0
    for _ in range(n):
        if a > a_max:
            break
        m = Node(C.KIND_MOON, planet.mass * logU(rng, *mass_range), U(rng, *radius_range))
        if rng.random() < atmo_p:
            m.atmo_h, m.atmo_rho = U(rng, 4, 9), 1.3
        planet.add(m, Orbit(a, e=U(rng, 0, 0.04), sign=1 if rng.random() > retro_p else -1))
        placed.append((a, m))
        a = a * U(rng, 1.35, 1.7) + 4 * m.hill() + 2 * m.clear_radius()
    return placed


def jovian(rng):
    gm = logU(rng, 1.2e4, 3.5e4)
    planet = Node(C.KIND_GIANT, gm, U(rng, 28, 42), atmo_h=U(rng, 10, 16), atmo_rho=2.0)
    r = planet.clear_radius() + U(rng, 5, 30)
    for _ in range(int(rng.integers(1, 3))):             # radiation belts
        w = U(rng, 25, 60)
        planet.zones.append(Zone(r, r + w, C.ZONE_RADIATION, U(rng, 1.0, 3.0)))
        r += w + U(rng, 10, 40)
    placed = moon_system(rng, planet, U(rng, 100, 140), 850, int(rng.integers(3, 8)),
                         (1e-3, 8e-3), (5, 10))
    for _ in range(int(rng.integers(0, 3))):
        add_debris_cloud(rng, planet, 150, 800)
    sprinkle(rng, planet, placed, n_stations=(1, 3), n_sensors=(0, 2))
    return planet, []


def saturnian(rng):
    gm = logU(rng, 1.5e4, 2.5e4)
    planet = Node(C.KIND_GIANT, gm, U(rng, 25, 32), atmo_h=U(rng, 8, 12), atmo_rho=1.5)
    r = planet.clear_radius() + U(rng, 4, 10)
    r_end = planet.radius * U(rng, 2.8, 3.6)
    while r < r_end and len(planet.zones) < 4:           # ring bands with gaps
        w = U(rng, 8, 25)
        planet.zones.append(Zone(r, min(r + w, r_end), C.ZONE_DEBRIS, U(rng, 0.8, 2.5), 1.0))
        r += w
        gap = U(rng, 6, 16)
        if gap > 11 and r + gap < r_end and rng.random() < 0.6:   # shepherd moonlet in the gap
            planet.add(Node(C.KIND_MOON, gm * 1e-5, U(rng, 1.5, 2.5)), Orbit(r + gap / 2))
        r += gap
    placed = moon_system(rng, planet, max(r_end + 60, U(rng, 150, 200)), 850,
                         int(rng.integers(2, 6)), (5e-4, 8e-3), (4, 10), atmo_p=0.35, retro_p=0.0)
    sprinkle(rng, planet, placed, n_stations=(1, 3), n_sensors=(0, 1),
             root_band=(r_end + 20, 850))
    return planet, []


def binary_star(rng):
    a_gm = logU(rng, 3e4, 6e4)
    b_gm = a_gm * U(rng, 0.2, 1.0)
    root = make_star(rng, a_gm)
    a_bin = U(rng, 70, 130)
    root.add(make_star(rng, b_gm), Orbit(a_bin, e=U(rng, 0, 0.12)))
    # circumbinary planets orbit the binary's barycentre (Jacobi builder)
    placed = planetary_system(rng, root, a_bin * U(rng, 3.3, 4.0), 1100, int(rng.integers(2, 6)))
    for _ in range(int(rng.integers(0, 3))):
        add_debris_cloud(rng, root, 2.5 * a_bin, 1000)
    sprinkle(rng, root, placed, root_band=(2.5 * a_bin, 1000))
    return root, []


def bh_cluster(rng):
    gm = logU(rng, 2e5, 7e5)
    bh = Node(C.KIND_BLACK_HOLE, gm, 8.0, lum=U(rng, 2e4, 4e4),
              zones=[Zone(12.0, U(rng, 55, 80), C.ZONE_RADIATION, U(rng, 4, 8))])
    a = U(rng, 260, 360)
    placed = []
    for _ in range(int(rng.integers(3, 7))):
        if a > 1700:
            break
        s = make_star(rng, logU(rng, 1e4, 4e4))
        bh.add(s, Orbit(a, e=U(rng, 0, 0.06), sign=1 if rng.random() < 0.85 else -1))
        if rng.random() < 0.7:
            pa = s.clear_radius() + C_CLEAR + U(rng, 12, 18)
            for _ in range(int(rng.integers(1, 3))):
                p = make_rocky(rng, s.mass)
                if pa + p.clear_radius() > 0.35 * s.hill():
                    break
                s.add(p, Orbit(pa + p.clear_radius()))
                pa = (pa + p.clear_radius()) * 1.6 + p.clear_radius()
        placed.append((a, s))
        a = a + max(8 * s.hill(), a * U(rng, 0.3, 0.55))
    for _ in range(int(rng.integers(0, 3))):
        add_debris_cloud(rng, bh, 200, 1500, size=U(rng, 40, 90))
    sprinkle(rng, bh, placed, n_stations=(1, 4), n_sensors=(0, 2), root_band=(150, 1600))
    return bh, []


def asteroid_field(rng):
    star = make_star(rng)
    placed = planetary_system(rng, star, U(rng, 90, 140), 700, int(rng.integers(2, 4)), moons=1)
    a_last = placed[-1][0] if placed else 200
    belt_in = a_last + 3 * (placed[-1][1].extent() if placed else 0) + U(rng, 60, 100)
    belt_out = belt_in + U(rng, 120, 250)
    star.zones.append(Zone(belt_in, belt_out, C.ZONE_DEBRIS, U(rng, 0.15, 0.4), 1.0))
    for _ in range(int(rng.integers(3, 9))):               # big rocks with real gravity
        rock = Node(C.KIND_ASTEROID, logU(rng, 0.5, 4), U(rng, 3, 6))
        a = star.free_orbit(rng, belt_in, belt_out, own_extent=rock.radius, pad=6)
        if a is not None:
            star.add(rock, Orbit(a, e=U(rng, 0, 0.03)))
    for _ in range(int(rng.integers(2, 6))):
        add_debris_cloud(rng, star, belt_in * 0.6, belt_out * 1.1)
    sprinkle(rng, star, placed, n_stations=(1, 3), n_sensors=(0, 2),
             root_band=(star.clear_radius() + 60, belt_out))
    return star, []


def compact_system(rng):
    """A tight, fast inner system: short periods, strong radiation, little room."""
    star = make_star(rng, logU(rng, 2e4, 5e4))
    placed = planetary_system(rng, star, U(rng, 45, 75), 450, int(rng.integers(4, 8)),
                              moons=1, giant_frac=U(rng, 0.1, 0.4))
    for _ in range(int(rng.integers(0, 3))):
        add_debris_cloud(rng, star, 60, 420, size=U(rng, 15, 35))
    sprinkle(rng, star, placed, n_stations=(1, 3), n_sensors=(0, 2),
             root_band=(star.clear_radius() + 30, 460))
    return star, []


def debris_maze(rng):
    """Mostly empty of planets, full of things that hurt: belts, clouds, patrol zones."""
    star = make_star(rng)
    placed = planetary_system(rng, star, U(rng, 90, 160), 800, int(rng.integers(1, 4)), moons=1)
    r = U(rng, 180, 260)
    for _ in range(int(rng.integers(1, 3))):                 # belts with a gap between them
        w = U(rng, 60, 140)
        star.zones.append(Zone(r, r + w, C.ZONE_DEBRIS, U(rng, 0.15, 0.5), 1.0))
        r += w + U(rng, 70, 160)
    for _ in range(int(rng.integers(3, 6))):
        add_debris_cloud(rng, star, 120, 900, strength=U(rng, 0.6, 1.6), size=U(rng, 30, 70))
    sprinkle(rng, star, placed, n_stations=(1, 3), n_sensors=(1, 2))
    return star, []


def star_cluster(rng):
    """Several stars bound in a hierarchy: deep wells, fast orbits, little room.

    Flying here on a small tank means riding the wells rather than crossing them."""
    root = make_star(rng, logU(rng, 3e4, 7e4))
    a = U(rng, 70, 120)
    root.add(make_star(rng, root.mass * U(rng, 0.4, 1.0)), Orbit(a, e=U(rng, 0, 0.1)))
    placed = []
    for _ in range(int(rng.integers(2, 5))):
        a = a * U(rng, 3.0, 4.2)
        if a > 1400:
            break
        st = make_star(rng, logU(rng, 1.5e4, 5e4))
        root.add(st, Orbit(a, e=U(rng, 0, 0.12), sign=1 if rng.random() < 0.85 else -1))
        if rng.random() < 0.6:                      # a planet or two in its own well
            pa = st.clear_radius() + C_CLEAR + U(rng, 12, 20)
            for _ in range(int(rng.integers(1, 3))):
                p = make_rocky(rng, st.mass)
                if pa + p.clear_radius() > 0.35 * st.hill():
                    break
                st.add(p, Orbit(pa + p.clear_radius(), e=U(rng, 0, 0.04)))
                pa = (pa + p.clear_radius()) * 1.7 + p.clear_radius()
        placed.append((a, st))
    for _ in range(int(rng.integers(0, 3))):
        add_debris_cloud(rng, root, 200, 1500, size=U(rng, 30, 70))
    sprinkle(rng, root, placed, n_stations=(2, 4), n_sensors=(0, 2), root_band=(150, 1500))
    return root, []


# ---------------------------------------------------------------------------
# held-out families (never used for training)
# ---------------------------------------------------------------------------

def trinary(rng):
    a_gm = logU(rng, 3e4, 5e4)
    root = make_star(rng, a_gm)
    a_bin = U(rng, 55, 85)
    root.add(make_star(rng, a_gm * U(rng, 0.3, 0.8)), Orbit(a_bin, e=U(rng, 0, 0.08)))
    a_c = U(rng, 6.5, 8.5) * a_bin
    c = root.add(make_star(rng, logU(rng, 1.5e4, 3e4)), Orbit(a_c, e=U(rng, 0, 0.06)))
    pa = c.clear_radius() + C_CLEAR + U(rng, 12, 20)
    if pa + 8 < 0.35 * c.hill():
        c.add(make_rocky(rng, c.mass), Orbit(pa + 7))
    placed = [(a_c, c)]
    far = root.add(make_giant(rng, root.total_mass()), Orbit(a_c * U(rng, 1.9, 2.2), e=0.02))
    add_moons(rng, far, 2)
    placed.append((far.orbit_a, far))
    sprinkle(rng, root, placed, root_band=(2.5 * a_bin, far.orbit_a * 1.1))
    return root, []


def pulsar(rng):
    ns = Node(C.KIND_NEUTRON, logU(rng, 3e4, 6e4), 5.0, lum=U(rng, 4e4, 8e4))
    placed = planetary_system(rng, ns, U(rng, 180, 240), 1000, int(rng.integers(3, 6)))
    sprinkle(rng, ns, placed, root_band=(150, 1000))
    return ns, []


def rogue_flyby(rng):
    star, _ = sol_like(rng)
    rogue_gm = star.mass * U(rng, 0.15, 0.4)
    rogue = dict(kind=C.KIND_STAR, mass=rogue_gm, radius=14 + 8 * (rogue_gm / 5e4) ** 0.5,
                 atmo_h=0.0, atmo_rho=0.0, lum=0.15 * rogue_gm, parent=-1,
                 flyby=dict(r_peri=U(rng, 550, 850), v_inf=U(rng, 10, 16), r_start=U(rng, 1300, 1600)))
    return star, [rogue]


TRAIN_FAMILIES = {
    "sol_like": sol_like,
    "jovian": jovian,
    "saturnian": saturnian,
    "binary_star": binary_star,
    "bh_cluster": bh_cluster,
    "asteroid_field": asteroid_field,
    "compact_system": compact_system,
    "debris_maze": debris_maze,
    "star_cluster": star_cluster,
}
HOLDOUT_FAMILIES = {
    "trinary": trinary,
    "pulsar": pulsar,
    "rogue_flyby": rogue_flyby,
}
ALL_FAMILIES = {**TRAIN_FAMILIES, **HOLDOUT_FAMILIES}
FAMILY_IDS = {name: i for i, name in enumerate(ALL_FAMILIES)}
FAMILY_NAMES = {i: name for name, i in FAMILY_IDS.items()}
