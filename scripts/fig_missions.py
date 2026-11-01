r"""Three missions that used the n-body field instead of fighting it: the
schematic that opens the paper.

    python scripts/fig_missions.py

No JAX, no GPU, no data: every curve is drawn by hand from the published
mission profiles.  Writes paper/figures/missions.pdf and missions.png.

  (a) Voyager 2, heliocentric view.  Four gravity assists, Jupiter, Saturn,
      Uranus, Neptune, each adding heliocentric speed, so that a trip that
      would take 30 years took 12 (NASA, Voyager planetary voyage page).  At
      Neptune the flyby was used to steer, not to speed up: the craft was
      sent over Neptune's north pole and bent below the ecliptic to pass
      Triton.
  (b) Parker Solar Probe, heliocentric view.  Seven Venus gravity assists,
      each removing heliocentric speed, walked the perihelion from
      35.7 solar radii (first perihelion, 2018-11-05) down to 9.86 solar
      radii (2024-12-24).  Perihelia are drawn x3 so the inner orbits can be
      told apart; aphelia are at their true fractions of 1 AU.
  (c) Artemis I in the Earth-Moon rotating frame.  Outbound powered flyby at
      ~130 km over the far side (2022-11-21), coast out to a distant
      retrograde orbit (DRO) of the Moon (insertion 2022-11-25), half a
      revolution of the DRO, retrograde, i.e. against the Moon's own motion
      around Earth, departure (2022-12-01), return powered flyby at ~130 km
      (2022-12-05) and home (splashdown 2022-12-11).

Schematic, not to scale: orbit radii are compressed, perihelia in (b) are
exaggerated, the DRO in (c) is drawn large.  Orbits, the order of the
flybys, the direction of travel and the sense of each trick (speed added or
removed, perihelion lowered, powered flyby at perilune, retrograde loop) are
as flown.  Dates are closest approach (a, b) or burn date (c), UTC, from the
NASA mission pages and the Artemis I flight-day blog.

Draft caption (the coordinator owns the final wording):

  Figure 1.  Three missions that solved an n-body problem by using the
  field.  Schematic, not to scale; blue is the craft, orange marks the
  trick.  (a) Voyager 2 (launched 1977-08-20) took gravity assists at
  Jupiter (Jul 1979), Saturn (Aug 1981), Uranus (Jan 1986) and Neptune
  (Aug 1989), each adding heliocentric speed that no tank could have
  supplied; the assists cut the flight to Neptune from about 30 years to 12
  \cite{kohlhase1977}.  At Neptune the assist was spent on steering: the
  craft was sent over the north pole and bent below the ecliptic to pass
  Triton.  (b) Parker Solar Probe (launched 2018-08-12) used seven Venus
  gravity assists (2018-10-03 to 2024-11-06), each removing heliocentric
  speed, to lower its perihelion from 35.7 solar radii (2018-11-05) to
  9.86 solar radii, 6.2 million km above the photosphere, on 2024-12-24,
  where it moved at about 192 km/s \cite{fox2016,guo2010}; perihelia are
  drawn three times too large so that the inner orbits can be told apart.
  (c) Artemis I (launched 2022-11-16) in the Earth--Moon rotating frame: an
  outbound powered flyby about 130 km over the far side of the Moon
  (2022-11-21), coast to a distant retrograde orbit about 70,000 km from the
  Moon (insertion 2022-11-25), half a revolution of that orbit, traversed
  against the Moon's own motion about Earth and reaching 432,210 km from
  Earth (2022-11-28), departure (2022-12-01), a return powered flyby at
  about 130 km (2022-12-05) and splashdown (2022-12-11) \cite{batcha2020}.
"""
import pathlib

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.patches import Circle, Ellipse, FancyArrowPatch

ROOT = pathlib.Path(__file__).resolve().parents[1]
OUT = ROOT / "paper" / "figures"
OUT.mkdir(parents=True, exist_ok=True)

# house style (scripts/paper_figures.py, scripts/fig_task.py)
BLUE, ORANGE, GREEN, RED, GREY = "#2a78d6", "#eb6834", "#0ca30c", "#d03b3b", "#6b7280"
BODY_FACE, BODY_EDGE = "#d6dae2", "#6b7280"
STAR_FACE = "#f2b134"
INK = "#1f2430"
ORBIT = "#b9bfca"
plt.rcParams.update({
    "figure.dpi": 150, "savefig.dpi": 200, "savefig.bbox": "tight",
    "font.family": "serif", "font.serif": ["DejaVu Serif"], "font.size": 8.0,
    "axes.labelsize": 8.0, "axes.titlesize": 8.5, "legend.fontsize": 7.5,
    "lines.solid_capstyle": "round", "pdf.fonttype": 42,
})
SMALL = 5.6
BOX = dict(boxstyle="round,pad=0.12", fc="white", ec="none", alpha=0.85)


# ------------------------------------------------------------------ helpers
def letter(fig, x, y, s):
    fig.text(x, y, s, fontsize=8.5, fontweight="bold", va="top", ha="left", color=INK)


def frame(ax, half, cx=0.0, cy=0.0):
    ax.set_xlim(cx - half, cx + half); ax.set_ylim(cy - half, cy + half)
    ax.set_aspect("equal"); ax.set_xticks([]); ax.set_yticks([])
    for sp in ax.spines.values():
        sp.set_visible(True); sp.set_color("#c9ced9"); sp.set_linewidth(0.5)


def body(ax, x, y, r, face=BODY_FACE, edge=BODY_EDGE, z=6):
    ax.add_patch(Circle((x, y), r, facecolor=face, edgecolor=edge, lw=0.5, zorder=z))


def orbit(ax, r, cx=0.0, cy=0.0, ls="-", lw=0.5, z=1):
    ax.add_patch(Circle((cx, cy), r, facecolor="none", edgecolor=ORBIT, lw=lw, ls=ls, zorder=z))


def path(ax, pts, color=BLUE, lw=1.0, z=5, ls="-"):
    pts = np.asarray(pts)
    ax.plot(pts[:, 0], pts[:, 1], color=color, lw=lw, zorder=z, ls=ls)


def head(ax, p, q, color=BLUE, ms=5, lw=0.9, z=7):
    """An arrowhead at q, pointing from p to q."""
    ax.add_patch(FancyArrowPatch(p, q, arrowstyle="-|>", mutation_scale=ms, color=color,
                                 lw=lw, zorder=z, shrinkA=0, shrinkB=0))


def note(ax, text, where, ha="left", va="center", color=INK, bold=False, box=True,
         leader=None, leader_from=None):
    """A label at a hand-placed spot; `leader` is a point a thin grey line is
    drawn to, from the label's anchor or from `leader_from`."""
    if leader is not None:
        ax.annotate("", leader, leader_from if leader_from is not None else where,
                    arrowprops=dict(arrowstyle="-", color=GREY, lw=0.45, shrinkA=0, shrinkB=0),
                    zorder=4)
    ax.text(where[0], where[1], text, fontsize=SMALL, color=color, ha=ha, va=va,
            fontweight="bold" if bold else "normal", zorder=12, bbox=BOX if box else None,
            linespacing=1.15)


def accent(ax, p, d, length, label, where, ha="left", va="center", ms=6, lw=1.1,
           leader=None, leader_from=None):
    """The trick: an orange arrow from p along unit direction d, and its label."""
    p = np.asarray(p, float); d = np.asarray(d, float); d = d / np.hypot(*d)
    q = p + d * length
    ax.add_patch(FancyArrowPatch(p, q, arrowstyle="-|>", mutation_scale=ms, color=ORANGE,
                                 lw=lw, zorder=9, shrinkA=0, shrinkB=0))
    note(ax, label, where, ha=ha, va=va, color=ORANGE, leader=leader, leader_from=leader_from)


def bezier(p0, p1, p2, p3=None, n=60):
    t = np.linspace(0, 1, n)[:, None]
    p0, p1, p2 = (np.asarray(p, float) for p in (p0, p1, p2))
    if p3 is None:
        return (1 - t) ** 2 * p0 + 2 * (1 - t) * t * p1 + t ** 2 * p2
    p3 = np.asarray(p3, float)
    return ((1 - t) ** 3 * p0 + 3 * (1 - t) ** 2 * t * p1 + 3 * (1 - t) * t ** 2 * p2
            + t ** 3 * p3)


def pol(r, deg):
    a = np.radians(deg)
    return np.array([r * np.cos(a), r * np.sin(a)])


# ------------------------------------------------------------ (a) Voyager 2
def panel_voyager(ax):
    # orbit radii compressed; heliocentric longitudes roughly as flown
    # (Earth at launch ~327 deg, Jupiter Jul 1979 ~140, Saturn Aug 1981 ~190,
    # Uranus Jan 1986 ~255, Neptune Aug 1989 ~280), so the path sweeps almost
    # a full turn counterclockwise, outward; the whole picture is turned by
    # +40 deg so that no planet sits against the frame
    R = dict(E=0.42, J=1.05, S=1.62, U=2.22, N=2.78)
    TH = dict(E=10.0, J=180.0, S=232.0, U=290.0, N=322.0)
    for k in R:
        orbit(ax, R[k])
    P = {k: pol(R[k], TH[k]) for k in R}
    ax.add_patch(Circle((0, 0), 0.12, facecolor=STAR_FACE, edgecolor=ORANGE, lw=0.5, zorder=6))
    note(ax, "Sun", (0.0, -0.17), ha="center", va="top", color=GREY, box=False)

    # Earth -> Jupiter: half an ellipse with perihelion at Earth
    rp, ra = R["E"], 1.07
    a = (rp + ra) / 2; e = (ra - rp) / (ra + rp)
    th0 = np.radians(TH["E"])
    f = np.linspace(0, np.radians(170.0), 80)
    r = a * (1 - e * e) / (1 + e * np.cos(f))
    leg_ej = np.stack([r * np.cos(th0 + f), r * np.sin(th0 + f)], 1)
    leg_ej[-1] = P["J"]
    path(ax, leg_ej)

    # Jupiter -> Saturn -> Uranus -> Neptune: nearly straight legs, each
    # leaving its planet turned prograde (the assist)
    def leg(p, q, dep_dir, k=0.45):
        d = np.asarray(dep_dir, float); d /= np.hypot(*d)
        return bezier(p, p + d * k * np.hypot(*(q - p)), q)

    tan = lambda key: np.array([-np.sin(np.radians(TH[key])), np.cos(np.radians(TH[key]))])
    rad = lambda key: np.array([np.cos(np.radians(TH[key])), np.sin(np.radians(TH[key]))])
    legs = [leg(P["J"], P["S"], 0.55 * tan("J") + 0.85 * rad("J")),
            leg(P["S"], P["U"], 0.75 * tan("S") + 0.65 * rad("S")),
            leg(P["U"], P["N"], 0.75 * tan("U") + 0.65 * rad("U"))]
    for L in legs:
        path(ax, L)
    # beyond Neptune, out of the ecliptic: dashed
    d_out = 0.45 * tan("N") + 0.9 * rad("N")
    out = leg(P["N"], P["N"] + 0.6 * d_out, d_out, k=0.3)
    path(ax, out, ls=(0, (2, 1.5)), lw=0.9)
    head(ax, out[-2], out[-1])

    # direction of travel on the first leg
    i = 30
    head(ax, leg_ej[i - 1], leg_ej[i])

    # bodies
    body(ax, *P["E"], 0.05)
    body(ax, *P["J"], 0.10)
    body(ax, *P["S"], 0.085)
    body(ax, *P["U"], 0.07)
    body(ax, *P["N"], 0.07)

    # the trick: speed added along the outgoing direction at J, S, U
    for key, L, lab, where, ha, va in (
            ("J", legs[0], "+v  Jul 1979", (-1.32, -0.42), "right", "center"),
            ("S", legs[1], "+v  Aug 1981", (-1.22, -1.56), "right", "center"),
            ("U", legs[2], "+v  Jan 1986", (0.95, -2.5), "center", "top")):
        d = L[6] - L[0]
        accent(ax, P[key] + d / np.hypot(*d) * 0.13, d, 0.42, lab, where, ha=ha, va=va)
    note(ax, "Jupiter", P["J"] + (0.0, 0.16), ha="center", va="bottom")
    note(ax, "Saturn", P["S"] + (0.13, 0.0), ha="left", va="center")
    note(ax, "Uranus", P["U"] + (-0.12, 0.05), ha="right", va="bottom")
    note(ax, "Neptune", P["N"] + (-0.02, 0.13), ha="center", va="bottom")
    note(ax, "Earth\nAug 1977", P["E"] + (0.10, 0.0), ha="left")
    note(ax, "Aug 1989: over the\nnorth pole, bent\nbelow the ecliptic\nto pass Triton",
         (2.9, -1.32), ha="right", va="bottom", color=ORANGE, leader=P["N"] + (0.0, 0.07),
         leader_from=(2.3, -1.32))
    note(ax, "four assists: 30 years\nto Neptune became 12", (-2.9, 2.85), ha="left", va="top",
         color=INK)
    frame(ax, 3.0)


# ------------------------------------------------------- (b) Parker Solar Probe
def panel_parker(ax):
    EX = 3.0   # perihelia drawn this many times too large
    ax.add_patch(Circle((0, 0), 0.07, facecolor=STAR_FACE, edgecolor=ORANGE, lw=0.5, zorder=6))
    note(ax, "Sun", (0.0, -0.1), ha="center", va="top", color=GREY, box=False)
    orbit(ax, 1.0)
    orbit(ax, 0.723)
    # (perihelion AU, aphelion AU): orbit 1 (after VGA1; period 150 d) and the
    # orbits after VGA3, VGA5, VGA7 (final: period 88 d, aphelion 0.73 AU)
    orbs = [(0.166, 0.946), (0.095, 0.82), (0.062, 0.76), (0.046, 0.726)]
    for n, (rp, ra) in enumerate(orbs):
        rp *= EX
        a = (rp + ra) / 2; c = a - rp; b = np.sqrt(a * a - c * c)
        lw = 1.0 if n in (0, 3) else 0.7
        ax.add_patch(Ellipse((rp - a, 0), 2 * a, 2 * b, facecolor="none", edgecolor=BLUE,
                             lw=lw, zorder=5, alpha=1.0 if n in (0, 3) else 0.55))
    # direction of travel: counterclockwise, marked on the first orbit at the top
    rp, ra = orbs[0][0] * EX, orbs[0][1]
    a = (rp + ra) / 2; c = a - rp; b = np.sqrt(a * a - c * c)
    t = np.radians(100.0)
    p = np.array([rp - a + a * np.cos(t), b * np.sin(t)])
    q = np.array([rp - a + a * np.cos(t + 0.03), b * np.sin(t + 0.03)])
    head(ax, p, q)
    # Earth at launch, on the aphelion side
    E = pol(1.0, 205.0)
    body(ax, *E, 0.045)
    note(ax, "Earth\n2018-08-12", E + (0.02, -0.07), ha="left", va="top")
    # Venus, where the orbits touch its orbit: the flybys, each removing speed
    V = np.array([-0.723, 0.0])
    body(ax, *V, 0.04)
    note(ax, "Venus", V + (-0.06, 0.03), ha="right", va="bottom")
    # velocity at the leftmost point is downward (counterclockwise orbit);
    # the assist takes speed out, so the accent points the other way
    accent(ax, V + (0.0, 0.07), (0, 1), 0.28, "−v  seven Venus flybys\n2018-10-03 … 2024-11-06",
           (-1.05, 1.1), ha="left", va="top")
    # perihelia
    x1, x7 = orbs[0][0] * EX, orbs[3][0] * EX
    note(ax, "first perihelion\n2018-11-05, 35.7 $R_\\odot$", (0.42, -0.45), ha="left", va="top",
         color=INK, leader=(x1, -0.03))
    note(ax, "final perihelion\n2024-12-24: 9.86 $R_\\odot$\n6.2 million km up,\n≈192 km/s",
         (0.42, 0.45), ha="left", va="bottom", color=ORANGE, leader=(x7, 0.03))
    note(ax, "perihelion stepped down after\neach flyby; perihelia drawn\n×3 to keep them apart",
         (-1.05, -1.13), ha="left", va="bottom", color=GREY)
    frame(ax, 1.22, cx=0.2, cy=0.0)


# ---------------------------------------------------------------- (c) Artemis I
def panel_artemis(ax):
    """Earth-Moon rotating frame, Earth at the origin, Moon at (1, 0); the Moon
    moves counterclockwise about Earth, so a retrograde orbit of the Moon runs
    clockwise here.  Topology as NASA's Artemis I mission map."""
    E = np.array([0.0, 0.0]); M = np.array([1.0, 0.0])
    body(ax, *E, 0.075, face="#9fc2ea", edge=BLUE)
    orbit(ax, 1.0, ls=(0, (2, 2)))
    body(ax, *M, 0.035)
    # the DRO: a loop around the Moon, longer across the Earth-Moon line
    ax_, ay_ = 0.21, 0.30
    dro = lambda deg: M + np.array([ax_ * np.cos(np.radians(deg)), ay_ * np.sin(np.radians(deg))])
    ins, dep = 55.0, 235.0           # insertion, departure: the ends of the half traversed
    th = np.linspace(ins, ins - 180.0, 80)        # clockwise from insertion to departure
    half = np.array([dro(t) for t in th])
    rest = np.array([dro(t) for t in np.linspace(ins, ins + 180.0, 80)])
    path(ax, rest, color=ORBIT, lw=0.6, ls=(0, (2, 2)), z=2)
    path(ax, half, color=BLUE, lw=1.0)
    head(ax, half[41], half[42])
    # outbound: from Earth, under the Moon, perilune on the lower far side
    peri1 = M + 0.06 * pol(1.0, -55.0)
    out1 = bezier(E + (0.07, -0.03), (0.45, -0.34), (0.98, -0.30), peri1)
    out2 = bezier(peri1, M + 0.13 * pol(1.0, -20.0), M + (0.33, 0.12), dro(ins))
    path(ax, out1); path(ax, out2)
    head(ax, out1[22], out1[23])
    # return: departure at the lower left, perilune on the far side, home over the top
    peri2 = M + 0.06 * pol(1.0, 10.0)
    ret1 = bezier(dro(dep), M + (-0.08, -0.26), M + (0.17, -0.18), peri2)
    ret2 = bezier(peri2, M + (0.0, 0.17), (0.45, 0.30), E + (0.07, 0.04))
    path(ax, ret1); path(ax, ret2)
    head(ax, ret2[42], ret2[43])
    # the tricks: the two powered flybys at perilune, the retrograde loop
    a1 = peri1 + 0.03 * pol(1.0, -55.0)
    accent(ax, a1, pol(1.0, 20.0), 0.14,
           "outbound powered flyby\n2022-11-21, ≈130 km up", (-0.15, -0.63),
           ha="left", va="bottom", leader=a1 + 0.05 * pol(1.0, 20.0), leader_from=(0.46, -0.53))
    a2 = peri2 + 0.03 * pol(1.0, 10.0)
    accent(ax, a2, pol(1.0, 100.0), 0.14,
           "return powered flyby\n2022-12-05, ≈130 km up", (1.40, 0.63),
           ha="right", va="top", leader=a2 + 0.08 * pol(1.0, 100.0), leader_from=(1.1, 0.5))
    note(ax, "DRO insertion\n2022-11-25", dro(ins) + (-0.03, 0.06), ha="right", va="bottom",
         color=INK, leader=dro(ins))
    note(ax, "DRO departure\n2022-12-01", (0.92, -0.36), ha="left", va="top", color=INK,
         leader=dro(dep) + (0.01, -0.01), leader_from=(0.95, -0.36))
    note(ax, "half a distant retrograde\norbit (DRO), ≈70,000 km from\nthe Moon, run against the\n"
         "Moon's own motion", (-0.15, 0.63), ha="left", va="top", color=ORANGE)
    note(ax, "farthest from Earth:\n432,210 km, 2022-11-28", (1.40, -0.63), ha="right", va="bottom",
         color=INK, leader=dro(-8.0) + (0.01, 0.0), leader_from=(1.36, -0.52))
    note(ax, "Earth\nlaunch 2022-11-16\nsplashdown 2022-12-11", E + (-0.15, -0.1), ha="left", va="top")
    note(ax, "Moon", M + (-0.05, -0.02), ha="right", va="top")
    ax.set_xlim(-0.17, 1.41); ax.set_ylim(-0.65, 0.65)
    ax.set_aspect("equal"); ax.set_xticks([]); ax.set_yticks([])
    for sp in ax.spines.values():
        sp.set_visible(True); sp.set_color("#c9ced9"); sp.set_linewidth(0.5)


# --------------------------------------------------------------------- main
def main():
    W, H = 6.4, 2.6
    fig = plt.figure(figsize=(W, H))

    def axes(x, y, w, h):
        return fig.add_axes([x / W, y / H, w / W, h / H])

    top = H - 0.22
    s = 1.9
    gap = 0.1
    wc = W - 0.08 - 2 * s - 2 * gap
    xa, xb, xc = 0.04, 0.04 + s + gap, 0.04 + 2 * (s + gap)
    ax_a = axes(xa, top - s, s, s)
    ax_b = axes(xb, top - s, s, s)
    ax_c = axes(xc, top - s, wc, s)
    panel_voyager(ax_a)
    panel_parker(ax_b)
    panel_artemis(ax_c)
    letter(fig, xa / W, (top + 0.19) / H, "(a)  Voyager 2")
    letter(fig, xb / W, (top + 0.19) / H, "(b)  Parker Solar Probe")
    letter(fig, xc / W, (top + 0.19) / H, "(c)  Artemis I, rotating frame")

    fig.savefig(OUT / "missions.pdf")
    fig.savefig(OUT / "missions.png")
    plt.close(fig)
    print("wrote", OUT / "missions.pdf", OUT / "missions.png", flush=True)


if __name__ == "__main__":
    main()
