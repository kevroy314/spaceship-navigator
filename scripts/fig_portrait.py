"""Phase portrait of the difficulty space: task demand D against what the learned
policy does there, with the twelve level families and the four evaluation
suites overlaid.

Reads paper/figures/corr_data.npz (written by scripts/correlations.py: for every
task of the four frozen 150 s suites, the closed-form demand D = eta / A at the
150 s clock, the family, and the final-checkpoint outcome of four 3000-update
runs) and draws paper/figures/portrait.{pdf,png}.  No JAX, no GPU.

    python scripts/fig_portrait.py

Panel (a): every instance as a jittered point in D x success (success is the
fraction of the four runs that arrived, so it takes five values and the cloud is
banded), coloured by the regime Table 2 assigns the family (hierarchical /
intermediate / contested / transient); over the cloud, a running success curve
per family (Gaussian kernel in log10 D, sigma = 0.3 decade, drawn only between
the family's 5th and 95th percentile of D and where the effective n is at least
15), labelled with the family's code name at its median D.  Panel (b): the same
axes with one curve per suite (kernel as in (a), 95 % Wilson band on the
effective n) and a marker at the suite's median D; the cloud is repeated in grey
for context.

Labels are placed by a small scorer (`place_labels`) that measures each label's
rendered box and keeps it off the curves, dots, other labels and the legend,
adding a thin leader when the chosen spot is not adjacent to the dot.

Draft caption (every number below is printed by the script):

  The difficulty space.  Task demand $\\dd$ (log scale) against the learned
  policy's per-instance success (the fraction of four 3000-update runs that
  arrived: two seeds each of the no-channel and all-four arms) over the 2,388
  tasks of the four 150 s suites.  (a) Every instance as a jittered point
  (success takes five values, so the cloud is banded), coloured by the regime
  of its family (Table 2), with a running success curve per family (Gaussian
  kernel in $\\log_{10}\\dd$, $\\sigma = 0.3$ decade, drawn between the family's
  5th and 95th percentiles of $\\dd$ where the effective $n \\ge 15$) and a dot
  at the family's median $\\dd$.  The hierarchical families span half a decade
  of median $\\dd$ (0.0007 for debris_maze to 0.0037 for compact_system) at
  success from 25 % to 73 %; the contested families sit at median $\\dd$ of
  0.29--1.45 with success 17--22 %; bh_cluster (0.52) is the hardest at 5 %.
  (b) The same axes with one curve per suite, its 95 % Wilson band on the
  effective $n$, and a dot at the suite's median $\\dd$ (easy 0.0005, medium
  0.0091, hard 0.023, blind 0.044).  Spearman $\\rho$ of success with $\\dd$ is
  $-0.585$ pooled ($n = 2{,}388$) and $-0.37$, $-0.44$, $-0.36$, $-0.48$ within
  easy, medium, hard and blind ($n = 600$, 594, 594, 600; every $p < 10^{-18}$).
  Seven tasks with $\\dd < 10^{-5}$ are drawn at the left edge.  Computed by
  scripts/fig_portrait.py from the data of scripts/correlations.py.
"""
import pathlib

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.patheffects as pe
import numpy as np
from scipy.stats import spearmanr

ROOT = pathlib.Path(__file__).resolve().parents[1]
OUT = ROOT / "paper" / "figures"
DATA = OUT / "corr_data.npz"

SUITES = ["easy", "medium", "hard", "blind"]
# Table 2 of the paper: the regime each family is listed under
REGIME = {
    "sol_like": "hierarchical", "jovian": "hierarchical", "saturnian": "hierarchical",
    "compact_system": "hierarchical", "asteroid_field": "hierarchical",
    "debris_maze": "hierarchical", "pulsar": "hierarchical",
    "bh_cluster": "intermediate",
    "binary_star": "contested", "trinary": "contested", "star_cluster": "contested",
    "rogue_flyby": "transient",
}
REGIMES = ["hierarchical", "intermediate", "contested", "transient"]
# Okabe-Ito blue / orange / bluish green / vermilion: passes the dataviz
# validator on all pairs (light surface); the orange is below 3:1 contrast,
# which the direct labels discharge.
REGIME_COL = {"hierarchical": "#0072B2", "intermediate": "#E69F00",
              "contested": "#009E73", "transient": "#D55E00"}
# suites: the three validation suites are an ordered set (one hue, light to
# dark); the held-out suite is neutral and dashed.  Distinct from the regime
# hues so that nothing is the same colour in (a) and (b).
SUITE_COL = {"easy": "#b39ddb", "medium": "#7e57c2", "hard": "#3b1f8f", "blind": "#222222"}
SUITE_LS = {"easy": "-", "medium": "-", "hard": "-", "blind": (0, (3.2, 1.6))}
LBL = {"easy": "easy", "medium": "medium", "hard": "hard", "blind": "blind (held-out)"}

XLO, XHI = 1e-5, 30.0        # 7 of 2,388 tasks have D < 1e-5; they are drawn at the edge
SIGMA = 0.3                  # kernel width in decades of D
GRID = np.linspace(np.log10(XLO), np.log10(XHI), 400)

plt.rcParams.update({
    "figure.dpi": 150, "savefig.dpi": 150, "savefig.bbox": "tight",
    "font.family": "serif", "font.serif": ["DejaVu Serif"], "font.size": 8.5,
    "axes.labelsize": 8.5, "axes.titlesize": 9, "legend.fontsize": 8,
    "xtick.labelsize": 7.5, "ytick.labelsize": 7.5,
    "axes.spines.top": False, "axes.spines.right": False,
    "axes.linewidth": 0.6, "xtick.major.width": 0.6, "ytick.major.width": 0.6,
    "lines.solid_capstyle": "round", "pdf.fonttype": 42,
})


def load():
    d = np.load(DATA, allow_pickle=False)
    rows = {}
    for k in ("D", "family", "agent_arrived"):
        rows[k] = np.concatenate([d[f"suite_{s}/{k}"] for s in SUITES], 0)
    rows["suite"] = np.concatenate([[s] * len(d[f"suite_{s}/D"]) for s in SUITES])
    rows["agent"] = rows["agent_arrived"].mean(1)
    rows["logD"] = np.log10(rows["D"].astype(float))
    return rows


def wilson(p, n, z=1.96):
    den = 1 + z * z / n
    c = (p + z * z / (2 * n)) / den
    h = z * np.sqrt(p * (1 - p) / n + z * z / (4 * n * n)) / den
    return c - h, c + h


def running(x, y, grid, sigma=SIGMA, n_min=15, q=(5, 95)):
    """Kernel-weighted mean of y along x (log10 D) on `grid`.  Returns the
    estimate, its Wilson 95 % band on the Kish effective n, and a mask of the
    points that are drawn: inside the q-percentile range of x and with
    effective n >= n_min."""
    w = np.exp(-0.5 * ((grid[:, None] - x[None, :]) / sigma) ** 2)
    sw = w.sum(1)
    p = (w * y[None, :]).sum(1) / np.maximum(sw, 1e-12)
    n_eff = sw ** 2 / np.maximum((w * w).sum(1), 1e-12)
    lo, hi = wilson(np.clip(p, 0, 1), np.maximum(n_eff, 1e-6))
    qlo, qhi = np.percentile(x, q)
    ok = (n_eff >= n_min) & (grid >= qlo) & (grid <= qhi)
    return p, lo, hi, ok, n_eff


def at(grid, p, ok, x0):
    """The curve's value at log10 D = x0 (nearest drawn grid point)."""
    idx = np.where(ok)[0]
    j = idx[np.argmin(np.abs(grid[idx] - x0))]
    return grid[j], p[j]


def crossing(grid, p, ok, level=0.5):
    """First D (left to right, drawn range only) at which the running success
    falls through `level`; None if the curve never starts above it or never
    falls below it."""
    idx = np.where(ok)[0]
    g, v = grid[idx], p[idx]
    if v[0] < level:
        return None, "starts below"
    below = np.where(v < level)[0]
    if len(below) == 0:
        return None, "never falls below"
    j = below[0]
    # linear interpolation between j-1 and j
    t = (v[j - 1] - level) / (v[j - 1] - v[j])
    return 10 ** (g[j - 1] + t * (g[j] - g[j - 1])), "crosses"


def place_labels(fig, ax, texts, dots, curves, legends, pad=1.5, passes=4):
    """Put each label beside its dot where it collides least.  Candidates are
    16 directions x 7 radii (5-56 pt) around the dot; a candidate is scored in
    display space by the curve samples its (padded) box covers, hard penalties
    for covering a dot, another label, a legend or another leader, or for
    leaving the axes, and a small cost per point of distance.  A label more
    than 12 pt from its dot gets a thin leader, and the curve samples, dots,
    other labels and other leaders that leader would cross count against the
    candidate.  Labels are placed greedily, the most hemmed-in dot first, then
    each is re-placed with the others held fixed for a few passes; a label still
    paying for a long leader or a collision then tries moving each neighbour to
    that neighbour's next-best spots and keeps the cheapest pair."""
    r = fig.canvas.get_renderer()
    px = fig.dpi / 72.0
    tr = ax.transData
    inv = tr.inverted()
    cpts = np.concatenate([tr.transform(np.c_[cx, cy]) for cx, cy in curves])
    dpts = tr.transform(np.array(dots))
    axbox = ax.get_window_extent(r)
    fixed = [lg.get_window_extent(r) for lg in legends]
    dirs = [(np.cos(a), np.sin(a)) for a in np.arange(0, 2 * np.pi, np.pi / 8)]
    radii = [5, 9, 14, 20, 28, 40, 56]
    n = len(texts)
    # the most hemmed-in dots choose first: curve samples and other dots within
    # 30 pt of the dot, then the longer label
    crowd = [np.sum(np.hypot(*(cpts - dpts[i]).T) < 30 * px)
             + 40 * (np.sum(np.hypot(*(dpts - dpts[i]).T) < 30 * px) - 1) for i in range(n)]
    order = sorted(range(n), key=lambda i: (-crowd[i], -len(texts[i].get_text())))

    def crosses(p1, p2, p3, p4):
        def side(a, b, c):
            return (b[0] - a[0]) * (c[1] - a[1]) - (b[1] - a[1]) * (c[0] - a[0])
        return (side(p1, p2, p3) * side(p1, p2, p4) < 0) and (side(p3, p4, p1) * side(p3, p4, p2) < 0)

    def through(p1, p2, box):
        q = np.c_[np.linspace(p1[0], p2[0], 24), np.linspace(p1[1], p2[1], 24)][1:-1]
        return bool(np.any((q[:, 0] > box.x0) & (q[:, 0] < box.x1) & (q[:, 1] > box.y0) & (q[:, 1] < box.y1)))

    # each label's candidate geometry and the part of its cost that does not
    # depend on the other labels are computed once
    geo = []
    for i in range(n):
        t, (dx, dy) = texts[i], dpts[i]
        g = []
        for rad in radii:
            for ux, uy in dirs:
                ha = "left" if ux > 0.3 else ("right" if ux < -0.3 else "center")
                va = "bottom" if uy > 0.3 else ("top" if uy < -0.3 else "center")
                t.set_ha(ha); t.set_va(va)
                t.set_position(inv.transform((dx + ux * rad * px, dy + uy * rad * px)))
                bb = t.get_window_extent(r).padded(pad * px)
                cost = 0.8 * rad
                cost += np.sum((cpts[:, 0] > bb.x0) & (cpts[:, 0] < bb.x1)
                               & (cpts[:, 1] > bb.y0) & (cpts[:, 1] < bb.y1))
                hit_d = (dpts[:, 0] > bb.x0 - 3 * px) & (dpts[:, 0] < bb.x1 + 3 * px) \
                        & (dpts[:, 1] > bb.y0 - 3 * px) & (dpts[:, 1] < bb.y1 + 3 * px)
                cost += 300 * hit_d.sum()
                cost += 1000 * sum(bb.overlaps(b) for b in fixed)
                if not (bb.x0 >= axbox.x0 and bb.x1 <= axbox.x1 and bb.y0 >= axbox.y0 - 2 * px
                        and bb.y1 <= axbox.y1):
                    cost += 1000
                lead = None
                if rad > 12:
                    ex = np.clip(dx, bb.x0, bb.x1); ey = np.clip(dy, bb.y0, bb.y1)
                    lead = ((dx, dy), (ex, ey))
                    seg = np.c_[np.linspace(dx, ex, 24), np.linspace(dy, ey, 24)][2:]
                    dmin = np.min(np.hypot(cpts[:, 0][None] - seg[:, :1], cpts[:, 1][None] - seg[:, 1:]), 1)
                    cost += 1.5 * np.sum(dmin < 1.5 * px)
                    dd = np.min(np.hypot(dpts[:, 0][None] - seg[:, :1], dpts[:, 1][None] - seg[:, 1:]), 0)
                    dd[i] = np.inf
                    cost += 300 * np.sum(dd < 4 * px)
                g.append((cost, ha, va, t.get_position(), bb, lead))
        geo.append(g)

    boxes = [None] * n          # padded bbox of each placed label
    leaders = [None] * n        # ((dot), (box point)) or None
    costs = [0.0] * n

    def score(i, c):
        """The static cost of candidate c of label i plus its collisions with the
        other labels as currently placed."""
        cost, ha, va, pos, bb, lead = c
        others = [b for j, b in enumerate(boxes) if j != i and b is not None]
        olead = [l for j, l in enumerate(leaders) if j != i and l is not None]
        cost += 1000 * sum(bb.overlaps(b) for b in others)
        cost += 300 * sum(through(a, b, bb) for a, b in olead)
        if lead is not None:
            cost += 300 * sum(crosses(lead[0], lead[1], a, b) for a, b in olead)
            cost += 300 * sum(through(lead[0], lead[1], b) for b in others)
        return (cost, ha, va, pos, bb, lead)

    def candidates(i):
        return sorted((score(i, c) for c in geo[i]), key=lambda c: c[0])

    def apply(i, c):
        cost, ha, va, pos, bb, lead = c
        texts[i].set_ha(ha); texts[i].set_va(va); texts[i].set_position(pos)
        boxes[i], leaders[i], costs[i] = bb, lead, cost

    def place(i):
        apply(i, candidates(i)[0])
        return costs[i]

    def sweep():
        for _ in range(passes - 1):
            moved = False
            for i in order:
                old = boxes[i]
                place(i)
                moved |= not np.allclose([old.x0, old.y0], [boxes[i].x0, boxes[i].y0])
            if not moved:
                break

    for i in order:
        place(i)
    sweep()
    # pairwise relief: a label still paying for a long leader or a collision
    # tries each neighbour's next-best spots and keeps the pair that is cheapest
    for i in order:
        if costs[i] <= 30:
            continue
        best = (costs[i], None, None)
        keep_i = (costs[i], texts[i].get_ha(), texts[i].get_va(), texts[i].get_position(), boxes[i], leaders[i])
        for j in range(n):
            if j == i:
                continue
            keep_j = (costs[j], texts[j].get_ha(), texts[j].get_va(), texts[j].get_position(), boxes[j], leaders[j])
            for cj in candidates(j)[:8]:
                apply(j, cj)
                ci = candidates(i)[0]
                tot = ci[0] + cj[0] - keep_j[0]
                if tot < best[0] - 1e-6:
                    best = (tot, (j, cj), ci)
            apply(j, keep_j); apply(i, keep_i)
        if best[1] is not None:
            (j, cj), ci = best[1], best[2]
            apply(j, cj); apply(i, ci)
    sweep()
    for i, lead in enumerate(leaders):
        if lead is None:
            continue
        (dx, dy), (ex, ey) = lead
        # start 2.5 pt out from the dot's centre so the line does not touch the marker
        v = np.array([ex - dx, ey - dy]); v = v / max(np.hypot(*v), 1e-9)
        p0 = inv.transform((dx + v[0] * 2.5 * px, dy + v[1] * 2.5 * px)); p1 = inv.transform((ex, ey))
        ax.plot([p0[0], p1[0]], [p0[1], p1[1]], color="#555", lw=0.45, zorder=4.5,
                solid_capstyle="butt")


def main():
    rows = load()
    x, y = rows["logD"], rows["agent"]
    n = len(x)
    rng = np.random.default_rng(0)
    xc = np.clip(x, np.log10(XLO), np.log10(XHI))
    yj = 100 * y + rng.uniform(-3.5, 3.5, n)

    # ------------------------------------------------------------ numbers
    print(f"tasks: {n}; agent success = fraction of 4 runs arrived (values {np.unique(y)})")
    rho, p = spearmanr(x, y)
    print(f"Spearman rho(success, D) pooled: {rho:+.3f} (p={p:.1e}, n={n})")
    print(f"{'suite':8s} {'n':>4s} {'median D':>9s} {'IQR':>17s} {'success':>8s} {'rho':>7s}")
    suite_med = {}
    for s in SUITES:
        sel = rows["suite"] == s
        r, pv = spearmanr(x[sel], y[sel])
        suite_med[s] = np.median(rows["D"][sel])
        print(f"{s:8s} {sel.sum():4d} {suite_med[s]:9.4f} "
              f"{np.percentile(rows['D'][sel], 25):8.4f}-{np.percentile(rows['D'][sel], 75):8.4f} "
              f"{100 * y[sel].mean():7.1f}% {r:+.3f} (p={pv:.1e})")
    fams = sorted(set(rows["family"]), key=lambda f: np.median(rows["D"][rows["family"] == f]))
    print(f"\n{'family':15s} {'regime':13s} {'n':>4s} {'median D':>9s} {'success':>8s} {'rho':>7s} "
          f"{'50% crossing':>14s}  suites")
    fam_curves = {}
    for f in fams:
        sel = rows["family"] == f
        pc, lo, hi, ok, ne = running(x[sel], y[sel], GRID)
        fam_curves[f] = (pc, lo, hi, ok)
        dx, how = crossing(GRID, pc, ok)
        r = spearmanr(x[sel], y[sel])[0]
        su = {s: int((rows["suite"][sel] == s).sum()) for s in SUITES if (rows["suite"][sel] == s).any()}
        print(f"{f:15s} {REGIME[f]:13s} {sel.sum():4d} {np.median(rows['D'][sel]):9.4f} "
              f"{100 * y[sel].mean():7.1f}% {r:+.3f} "
              f"{(f'{dx:.2g}' if dx is not None else '-- ' + how):>14s}  {su}")
    print(f"\nkernel sigma {SIGMA} decade; family curves drawn for n_eff >= 15 within the 5-95th "
          f"percentile of the family's D; suite curves for n_eff >= 20 within the 2.5-97.5th")
    print(f"tasks with D < {XLO:g}: {(x < np.log10(XLO)).sum()} (drawn at the left edge); "
          f"max D {rows['D'].max():.2f}")

    # ------------------------------------------------------------ figure
    fig, (ax_a, ax_b) = plt.subplots(1, 2, figsize=(6.4, 3.0), sharey=True,
                                     gridspec_kw=dict(width_ratios=[1.45, 1]))
    halo = [pe.withStroke(linewidth=1.8, foreground="white")]
    ticks = [1e-5, 1e-4, 1e-3, 1e-2, 1e-1, 1, 10]
    ticklabels = [r"$10^{-5}$", "", r"$10^{-3}$", "", r"$10^{-1}$", "", "10"]

    # (a) the cloud, coloured by regime, and the per-family curves
    for g in REGIMES:
        sel = np.array([REGIME[f] == g for f in rows["family"]])
        ax_a.scatter(10 ** xc[sel], yj[sel], s=2.2, color=REGIME_COL[g], alpha=0.28,
                     linewidths=0, zorder=1, rasterized=True)
    curves_a, dots_a, labels_a = [], [], []
    for f in fams:
        sel = rows["family"] == f
        pc, lo, hi, ok = fam_curves[f]
        col = REGIME_COL[REGIME[f]]
        seg = np.where(ok, 100 * pc, np.nan)
        ax_a.plot(10 ** GRID, seg, color=col, lw=1.0, zorder=3)
        ax_a.plot(10 ** GRID, seg, color="white", lw=2.2, zorder=2, alpha=0.9)
        curves_a.append((10 ** GRID[ok], 100 * pc[ok]))
        gx, gy = at(GRID, pc, ok, np.median(x[sel]))
        ax_a.scatter([10 ** gx], [100 * gy], s=11, color=col, edgecolor="white",
                     linewidths=0.6, zorder=4)
        dots_a.append((10 ** gx, 100 * gy))
        labels_a.append(ax_a.text(10 ** gx, 100 * gy, f, fontsize=5.6, family="monospace",
                                  color="#222", ha="left", va="center", zorder=5,
                                  path_effects=halo))
    ax_a.set_xscale("log"); ax_a.set_xlim(XLO, XHI); ax_a.set_ylim(-10, 106)
    ax_a.set_yticks([0, 25, 50, 75, 100])
    ax_a.set_xticks(ticks); ax_a.set_xticklabels(ticklabels)
    ax_a.grid(lw=0.4, alpha=0.4)
    ax_a.set_ylabel("arrived (% of 4 runs)")
    ax_a.set_xlabel(r"task demand $\mathcal{D}$")
    ax_a.set_title("(a) families, by regime", loc="left")
    handles = [plt.Line2D([], [], color=REGIME_COL[g], lw=1.2, marker="o", ms=3,
                          markeredgecolor="white", markeredgewidth=0.5, label=g) for g in REGIMES]
    # the upper-right corner above ~55 % and beyond D ~ 0.5 holds no curve
    leg_a = ax_a.legend(handles=handles, frameon=False, loc="upper right",
                        bbox_to_anchor=(1.0, 0.74), fontsize=6.6, handlelength=1.4,
                        handletextpad=0.5, borderaxespad=0.2, labelspacing=0.3)

    # (b) the cloud in grey; one curve per suite with its Wilson band
    ax_b.scatter(10 ** xc, yj, s=2.0, color="#9a9a9a", alpha=0.22, linewidths=0,
                 zorder=1, rasterized=True)
    curves_b, dots_b, labels_b = [], [], []
    for s in SUITES:
        sel = rows["suite"] == s
        pc, lo, hi, ok, ne = running(x[sel], y[sel], GRID, n_min=20, q=(2.5, 97.5))
        col = SUITE_COL[s]
        seg = np.where(ok, 100 * pc, np.nan)
        ax_b.fill_between(10 ** GRID, np.where(ok, 100 * lo, np.nan), np.where(ok, 100 * hi, np.nan),
                          color=col, alpha=0.13, lw=0, zorder=2)
        ax_b.plot(10 ** GRID, seg, color=col, lw=1.2, ls=SUITE_LS[s], zorder=3,
                  label=LBL[s])
        curves_b.append((10 ** GRID[ok], 100 * pc[ok]))
        gx, gy = at(GRID, pc, ok, np.log10(suite_med[s]))
        ax_b.scatter([10 ** gx], [100 * gy], s=18, color=col, edgecolor="white",
                     linewidths=0.7, zorder=4)
        dots_b.append((10 ** gx, 100 * gy))
        labels_b.append(ax_b.text(10 ** gx, 100 * gy, s, fontsize=6.4, color="#222",
                                  ha="left", va="center", zorder=5, path_effects=halo))
    ax_b.set_xscale("log"); ax_b.set_xlim(XLO, XHI)
    ax_b.set_xticks(ticks); ax_b.set_xticklabels(ticklabels)
    ax_b.grid(lw=0.4, alpha=0.4)
    ax_b.set_xlabel(r"task demand $\mathcal{D}$")
    ax_b.set_title("(b) suites", loc="left")
    leg_b = ax_b.legend(frameon=False, loc="upper right", bbox_to_anchor=(1.0, 0.82),
                        fontsize=6.6, handlelength=1.8, handletextpad=0.5, borderaxespad=0.2,
                        labelspacing=0.3)
    ax_b.tick_params(axis="y", length=0)

    fig.tight_layout(w_pad=0.6)
    fig.canvas.draw()
    place_labels(fig, ax_a, labels_a, dots_a, curves_a, [leg_a])
    place_labels(fig, ax_b, labels_b, dots_b, curves_b, [leg_b])
    fig.savefig(OUT / "portrait.pdf"); fig.savefig(OUT / "portrait.png")
    plt.close(fig)
    print("wrote", OUT / "portrait.pdf", "and portrait.png")


if __name__ == "__main__":
    main()
