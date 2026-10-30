"""Figures for the spacenav navigation paper.  Vector PDF, no GPU needed except
where marked.

    python scripts/paper_figures.py --which ghosts,ablation,clock,curves
    python scripts/paper_figures.py --which eta        # touches the env/GPU
"""
import argparse
import json
import pathlib

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.colors import Normalize
from matplotlib.cm import ScalarMappable

ROOT = pathlib.Path(__file__).resolve().parents[1]
OUT = ROOT / "paper" / "figures"
OUT.mkdir(parents=True, exist_ok=True)

BLUE, ORANGE, GREEN, RED, GREY = "#2a78d6", "#eb6834", "#0ca30c", "#d03b3b", "#6b7280"
SUITES = ["easy", "medium", "hard", "blind"]
LBL = {"easy": "Easy", "medium": "Medium", "hard": "Hard", "blind": "Blind"}

plt.rcParams.update({
    "figure.dpi": 150, "savefig.dpi": 150, "savefig.bbox": "tight",
    "font.family": "serif", "font.serif": ["DejaVu Serif"], "font.size": 8.5,
    "axes.labelsize": 8.5, "axes.titlesize": 9, "legend.fontsize": 8,
    "xtick.labelsize": 7.5, "ytick.labelsize": 7.5,
    "axes.spines.top": False, "axes.spines.right": False,
    "axes.linewidth": 0.6, "xtick.major.width": 0.6, "ytick.major.width": 0.6,
    "lines.solid_capstyle": "round", "pdf.fonttype": 42,
})


def evals(run):
    for base in ("data/rl", "data/razer/rl"):
        p = ROOT / base / run / "eval.jsonl"
        if p.exists():
            return [json.loads(l) for l in p.read_text().splitlines() if l.strip()]
    return None


def final(run, k=3):
    r = evals(run)
    if not r or r[-1]["update"] < 3000:
        return None
    t = r[-k:]
    return {s: 100 * float(np.mean([x["metrics"][s]["success"] for x in t])) for s in SUITES}


ARMS = {
    "off":  ["ppo_v5_noinstr", "ppo_v5_noinstr_s1", "ppo_v5_noinstr_s2",
             "ppo_v5_noinstr_s3", "ppo_v5_noinstr_s4"],
    "info": ["abl_info_s0", "abl_info_s1", "abl_info_s2", "abl_info_s3", "abl_info_s4"],
    "red":  ["abl_redundant_s0", "abl_redundant_s1", "abl_redundant_s2",
             "abl_redundant_s3", "abl_redundant_s4"],
    "on":   ["ppo_v4", "ppo_v4_s1", "ppo_v4_s2", "ppo_v4_s3", "ppo_v4_s4"],
}


# ---------------------------------------------------------------- ghost fleets
def fig_ghosts():
    tracks = [("medium_1", "Medium, task 1"), ("easy_0", "Easy, task 0"),
              ("medium_0", "Medium, task 0"), ("hard_0", "Hard, task 0")]
    fig, axes = plt.subplots(2, 2, figsize=(7.0, 7.25))
    norm = Normalize(0, 3000)
    cmap = plt.get_cmap("viridis")
    for ax, (key, title) in zip(axes.ravel(), tracks):
        d = json.loads((ROOT / "data" / "ghosts" / f"{key}.json").read_text())
        for b in d["bodies"]:
            pa = np.array(b["path"], float)
            if len(pa) > 1:
                ax.plot(pa[:, 0], pa[:, 1], color=GREY, lw=0.35, alpha=0.45, zorder=1)
            ax.add_patch(plt.Circle((pa[0, 0], pa[0, 1]), max(3.0, b["r"]),
                                    facecolor="#d6dae2", edgecolor=GREY, lw=0.4, zorder=2))
        g = sorted(d["ghosts"], key=lambda x: x["u"])
        for gh in g:
            p = np.array(gh["p"], float)
            if len(p) < 2:
                continue
            t = gh["u"] / 3000.0
            ax.plot(p[:, 0], p[:, 1], color=cmap(norm(gh["u"])), lw=0.22 + 0.5 * t,
                    alpha=0.16 + 0.34 * t, zorder=3, solid_capstyle="round")
        for wp in d["waypoints"]:
            pa = np.array(wp["path"], float)
            ax.plot(pa[:, 0], pa[:, 1], color=GREEN, lw=0.5, alpha=0.35, zorder=4)
            ax.add_patch(plt.Circle((pa[0, 0], pa[0, 1]), max(7.0, wp["r"]),
                                    facecolor="none", edgecolor=GREEN, lw=1.0, zorder=5))
        arr = sum(1 for x in d["ghosts"] if x["s"] == "arrived")
        n = len(d["ghosts"])
        ax.set_title(f"{title}\n{arr}/{n} arrived ({100*arr/n:.0f}%), "
                     f"{len(d['waypoints'])} waypoint" + ("s" if len(d["waypoints"]) > 1 else ""),
                     fontsize=8.5)
        # frame on where the ship and its targets actually go.  The stored box is
        # sized to contain every body's full orbit, which on most levels leaves the
        # flown region a small island in the middle.
        pts = [np.array(x["p"], float) for x in d["ghosts"] if len(x["p"])]
        pts += [np.array(w["path"], float) for w in d["waypoints"]]
        allp = np.concatenate(pts, 0)
        lo, hi = allp.min(0), allp.max(0)
        c = (lo + hi) / 2.0
        half = max(hi[0] - lo[0], hi[1] - lo[1]) / 2.0 * 1.10 + 12
        ax.set_xlim(c[0] - half, c[0] + half)
        ax.set_ylim(c[1] + half, c[1] - half)
        ax.set_aspect("equal"); ax.set_xticks([]); ax.set_yticks([])
        for sp in ax.spines.values():
            sp.set_visible(True); sp.set_color("#c9ced9"); sp.set_linewidth(0.5)
    cb = fig.colorbar(ScalarMappable(norm=norm, cmap=cmap), ax=axes,
                      orientation="horizontal", fraction=0.035, pad=0.035, aspect=48)
    cb.set_label("training update at which the attempt was flown")
    cb.outline.set_linewidth(0.4)
    fig.savefig(OUT / "ghosts.pdf")
    fig.savefig(OUT / "ghosts.png")
    plt.close(fig)
    print("ghosts.pdf")


# -------------------------------------------------------------- ablation, n=5
def fig_ablation():
    V = {k: [x for x in (final(r) for r in runs) if x] for k, runs in ARMS.items()}
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(7.0, 2.9),
                                   gridspec_kw=dict(width_ratios=[1.05, 1]))
    # left: all-off vs all-on, every seed
    y = np.arange(len(SUITES))
    for k, col, lab, dy in [("off", BLUE, "no instrument channels", -0.17),
                            ("on", ORANGE, "all four channels", 0.17)]:
        for i, s in enumerate(SUITES):
            v = [x[s] for x in V[k]]
            ax1.scatter(v, np.full(len(v), y[i] + dy), s=11, color=col, alpha=0.5,
                        linewidths=0, zorder=3)
            m, sd = np.mean(v), np.std(v, ddof=1)
            ax1.plot([m - sd, m + sd], [y[i] + dy] * 2, color=col, lw=1.4, alpha=0.5, zorder=2)
            ax1.plot([m, m], [y[i] + dy - 0.1, y[i] + dy + 0.1], color=col, lw=2.0, zorder=4)
            if i == 0:
                ax1.plot([], [], color=col, lw=2.0, label=lab)
    ax1.set_yticks(y); ax1.set_yticklabels([LBL[s] for s in SUITES])
    ax1.invert_yaxis(); ax1.set_xlabel("arrived (%)")
    ax1.set_xlim(0, 100); ax1.grid(axis="x", lw=0.4, alpha=0.5)
    # the Hard row has nothing beyond x~15, so the legend sits in its empty right
    ax1.legend(loc="center right", bbox_to_anchor=(1.0, 0.38), frameon=False)
    ax1.set_title("(a) five training seeds per arm", loc="left")

    # right: the four-way split
    order = [("off", "none"), ("info", r"$\eta$, stretch"),
             ("red", "budget, demand"), ("on", "all four")]
    for i, s in enumerate(SUITES):
        base = np.mean([x[s] for x in V["off"]])
        for j, (k, nm) in enumerate(order):
            v = [x[s] for x in V[k]]
            col = BLUE if k in ("off", "info") else ORANGE
            ax2.scatter([i + (j - 1.5) * 0.19], [np.mean(v)], s=26, color=col,
                        zorder=3, linewidths=0)
            ax2.plot([i + (j - 1.5) * 0.19] * 2,
                     [np.mean(v) - np.std(v, ddof=1), np.mean(v) + np.std(v, ddof=1)],
                     color=col, lw=1.0, alpha=0.55, zorder=2)
    for col, lab in [(BLUE, "matches the no-channel arm"), (ORANGE, "matches the all-four arm")]:
        ax2.scatter([], [], s=26, color=col, label=lab, linewidths=0)
    ax2.set_xticks(range(len(SUITES))); ax2.set_xticklabels([LBL[s] for s in SUITES])
    ax2.set_ylabel("arrived (%)"); ax2.grid(axis="y", lw=0.4, alpha=0.5)
    ax2.set_ylim(0, 100)
    # the band between the Easy group (~85-90) and the Medium group (~30-45) is
    # empty on the left, so the legend and the dot order live there
    leg = ax2.legend(loc="center left", bbox_to_anchor=(0.0, 0.61), frameon=False,
                     title="dots in each group, left to right:\n"
                           r"none $\cdot$ $\eta$, stretch $\cdot$ budget, demand $\cdot$ all four",
                     title_fontsize=7, alignment="left")
    ax2.set_title("(b) which channels cost", loc="left")
    fig.tight_layout()
    fig.savefig(OUT / "ablation.pdf"); fig.savefig(OUT / "ablation.png")
    plt.close(fig); print("ablation.pdf")


# ------------------------------------------------------------------ clock sweep
def fig_clock():
    rows = []
    for p in sorted((ROOT / "data" / "razer" / "curriculum" / "clock").glob("*.json"),
                    key=lambda q: json.loads(q.read_text())["clock"]):
        d = json.loads(p.read_text())
        o = d["outcomes"] if "outcomes" in d else {}
        if not o:
            o = {}
            for r in d["rows"]:
                o[r["status"]] = o.get(r["status"], 0) + 1
        rows.append((d["clock"], d["arrived"], o.get("timeout", 0),
                     o.get("crashed", 0) + o.get("destroyed", 0) + o.get("lost", 0), d["n"]))
    fig, ax = plt.subplots(figsize=(3.4, 3.0))
    x = np.arange(len(rows)); w = 0.62
    arr = np.array([r[1] for r in rows], float)
    tmo = np.array([r[2] for r in rows], float)
    died = np.array([r[3] for r in rows], float)
    ax.bar(x, arr, w, color=GREEN, label="arrived", zorder=3)
    ax.bar(x, tmo, w, bottom=arr, color=BLUE, label="ran out of clock", zorder=3)
    ax.bar(x, died, w, bottom=arr + tmo, color=RED,
           label="crashed, destroyed\nor out of fuel", zorder=3)
    for i, r in enumerate(rows):
        ax.text(i, r[1] + 0.6, f"{r[1]}", ha="center", va="bottom",
                fontsize=8, color="#0a6b0a", fontweight="bold")
    ax.set_xticks(x); ax.set_xticklabels([f"{int(r[0])} s" for r in rows])
    ax.set_xlabel("episode clock"); ax.set_ylabel("two-stop tours (of 32)")
    ax.set_ylim(0, 34); ax.grid(axis="y", lw=0.4, alpha=0.5)
    # every bar is stacked to the full 32, so there is no empty space inside the
    # axes: the legend goes below the x label
    fig.legend(frameon=False, loc="lower center", ncol=3, fontsize=7,
               bbox_to_anchor=(0.5, 0.0), handlelength=1.2, columnspacing=1.0)
    fig.tight_layout(rect=(0, 0.11, 1, 1))
    fig.savefig(OUT / "clock.pdf"); fig.savefig(OUT / "clock.png")
    plt.close(fig); print("clock.pdf")


# --------------------------------------------------------------- learning curves
def fig_curves():
    fig, axes = plt.subplots(1, 4, figsize=(7.0, 1.95), sharex=True)
    for ax, s in zip(axes, SUITES):
        for run, col, lab in [("ppo_v5_noinstr", BLUE, "no channels"),
                              ("ppo_v4", ORANGE, "all four")]:
            r = evals(run)
            if not r:
                continue
            u = [x["update"] for x in r]
            v = [100 * x["metrics"][s]["success"] for x in r]
            ax.plot(u, v, color=col, lw=1.1, label=lab)
        ax.set_title(LBL[s], loc="left")
        ax.grid(lw=0.4, alpha=0.5)
        ax.set_xlim(0, 3000); ax.set_xticks([0, 1500, 3000])
        # each suite keeps its own y range (Hard tops out near 10 %), but all
        # start at 0 and tick in whole percent
        ax.set_ylim(0, None); ax.margins(y=0.08)
        ax.yaxis.set_major_locator(matplotlib.ticker.MaxNLocator(nbins=5, integer=True))
        if s == "easy":
            ax.set_ylabel("arrived (%)"); ax.legend(frameon=False, loc="lower right", fontsize=7)
        ax.set_xlabel("training update")
    fig.tight_layout(); fig.savefig(OUT / "curves.pdf"); fig.savefig(OUT / "curves.png")
    plt.close(fig); print("curves.pdf")




# ------------------------------------------------- eta vs the equal-pull surface
def fig_eta():
    """Measured eta against distance from a secondary, in units of the radius
    where that secondary's pull equals its primary's.  Touches the env/GPU."""
    import jax, jax.numpy as jnp
    from spacenav import physics as P
    from spacenav import env as E
    from spacenav.levels.build import load_pool
    from spacenav.levels.families import FAMILY_NAMES

    pool, meta = load_pool(ROOT / "data" / "pools" / "long.npz")
    mass = np.asarray(pool.mass); active = np.asarray(pool.active)
    mults = np.array([0.2, 0.3, 0.5, 0.75, 1.0, 1.5, 2.0, 3.0, 4.0, 6.0, 8.0, 12.0])
    NANG = 24
    rows, fam_eta = [], {}

    def eta_at(level, bp, pts):
        full = P.gravity_at(pts, bp, level, 0)
        dom = P.gravity_at(pts, bp, level, 1)
        num = jnp.sqrt(jnp.sum((full - dom) ** 2, -1))
        den = jnp.sqrt(jnp.sum(full * full, -1)) + 1e-12
        return num / den
    eta_j = jax.jit(eta_at, static_argnums=())

    # the pool is laid out family by family, so a prefix would only ever see the
    # first one or two; take an even sample from each family instead
    by_fam = {}
    for i, mt in enumerate(meta):
        by_fam.setdefault(mt["family"], []).append(i)
    per = max(1, 300 // max(1, len(by_fam)))
    picks = [i for v in by_fam.values() for i in v[:per]]
    for i in picks:
        level = jax.tree.map(lambda x: x[i], pool)
        bp, _ = E.snapshot_state(level, 0)
        bp = np.asarray(bp); m = mass[i] * active[i]
        if (m > 0).sum() < 2:
            continue
        primary = int(np.argmax(m))
        order = np.argsort(m)[::-1]
        sec = int(order[1])
        if m[sec] <= 0:
            continue
        a = float(np.linalg.norm(bp[sec] - bp[primary]))
        if a <= 0:
            continue
        d_eq = a * np.sqrt(m[sec] / m[primary])
        ang = np.linspace(0, 2 * np.pi, NANG, endpoint=False)
        for mu in mults:
            r = mu * d_eq
            pts = bp[sec][None, :] + r * np.stack([np.cos(ang), np.sin(ang)], -1)
            e = np.asarray(eta_j(level, jnp.asarray(bp), jnp.asarray(pts)))
            e = e[np.isfinite(e)]
            if len(e):
                rows.append((mu, float(np.median(e))))
        # family-level median eta, sampled over the whole playable disc
        fam = FAMILY_NAMES[int(np.asarray(level.family))]
        rr = np.sqrt(np.random.RandomState(i).uniform(0.02, 1.0, 160)) * float(level.level_radius)
        th = np.random.RandomState(i + 7).uniform(0, 2 * np.pi, 160)
        pts = np.stack([rr * np.cos(th), rr * np.sin(th)], -1)
        e = np.asarray(eta_j(level, jnp.asarray(bp), jnp.asarray(pts)))
        e = e[np.isfinite(e)]
        if len(e):
            fam_eta.setdefault(fam, []).append(float(np.median(e)))

    rows = np.array(rows)
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(7.0, 2.65),
                                   gridspec_kw=dict(width_ratios=[1, 1.15]))
    med = [(mu, np.median(rows[rows[:, 0] == mu, 1]),
            np.percentile(rows[rows[:, 0] == mu, 1], 25),
            np.percentile(rows[rows[:, 0] == mu, 1], 75)) for mu in mults]
    med = np.array(med)
    ax1.fill_between(med[:, 0], med[:, 2], med[:, 3], color=BLUE, alpha=0.18, lw=0)
    ax1.plot(med[:, 0], med[:, 1], color=BLUE, lw=1.6, marker="o", ms=3.2)
    ax1.axvline(1.0, color=RED, lw=0.9, ls="--")
    # label above the interquartile band, not on it
    top = float(med[:, 3].max())
    ax1.set_ylim(0, top * 1.28)
    ax1.text(1.08, top * 1.25, "equal-pull\nsurface", color=RED, fontsize=7.2, va="top")
    ax1.set_xscale("log"); ax1.set_xticks([0.2, 0.5, 1, 2, 4, 8])
    ax1.get_xaxis().set_major_formatter(matplotlib.ticker.ScalarFormatter())
    ax1.set_xlabel(r"distance from the secondary ($\times$ equal-pull radius)")
    ax1.set_ylabel(r"$\eta$  (field a one-body model discards)")
    ax1.grid(lw=0.4, alpha=0.5)
    ax1.set_title("(a) where the coarse model breaks", loc="left")

    fams = sorted(fam_eta, key=lambda f: np.median(fam_eta[f]))
    vals = [np.median(fam_eta[f]) for f in fams]
    cols = [ORANGE if v > 0.1 else BLUE for v in vals]
    ax2.barh(range(len(fams)), vals, color=cols, height=0.68)
    ax2.set_yticks(range(len(fams)))
    ax2.set_yticklabels([f.replace("_", " ") for f in fams])
    ax2.set_xscale("log")
    ax2.set_xlabel(r"median $\eta$ over the playable area")
    ax2.grid(axis="x", lw=0.4, alpha=0.5)
    ax2.set_title("(b) contested everywhere, or only in a shell", loc="left")
    fig.tight_layout(); fig.savefig(OUT / "eta.pdf"); fig.savefig(OUT / "eta.png")
    plt.close(fig)
    print("eta.pdf  (families: " + ", ".join(f"{f}={np.median(fam_eta[f]):.4f}" for f in fams) + ")")


# ------------------------------------------------------- the two difficulty axes
def fig_bands():
    fig, ax = plt.subplots(figsize=(4.6, 3.1))
    # model scale (x) against execution margin (y); bands are regions, not points.
    # Three columns of width 0.26 with 0.02 gutters; the two repair arrows run in
    # the top and right margins so that no line crosses a band.
    bands = [
        ("trivial",    0.03, 0.56, 0.26, 0.29, "#cfe3f7"),
        ("human",      0.31, 0.36, 0.26, 0.49, "#9ec5f4"),
        ("calculated", 0.59, 0.36, 0.26, 0.49, "#f6c5ac"),
        ("precision",  0.03, 0.03, 0.54, 0.30, "#f3b0b0"),
        ("opaque",     0.59, 0.03, 0.26, 0.30, "#d9d9d9"),
    ]
    for name, x, y, w, h, col in bands:
        ax.add_patch(plt.Rectangle((x, y), w, h, facecolor=col, edgecolor="white", lw=1.4))
        ax.text(x + w / 2, y + h / 2, name, ha="center", va="center", fontsize=8.5)
    # calculated -> human: a coarser field, drawn above the two columns
    ax.annotate("", xy=(0.44, 0.905), xytext=(0.72, 0.905),
                arrowprops=dict(arrowstyle="->", color="#333", lw=1.0))
    ax.text(0.58, 0.925, "simplify the field", ha="center", va="bottom", fontsize=7, color="#333")
    # precision -> human: more slack, drawn to the right of the bands
    ax.annotate("", xy=(0.905, 0.60), xytext=(0.905, 0.18),
                arrowprops=dict(arrowstyle="->", color="#333", lw=1.0))
    ax.text(0.925, 0.39, "widen the timing window", ha="left", va="center", rotation=90,
            fontsize=7, color="#333")
    ax.set_xlim(0, 1); ax.set_ylim(0, 1)
    ax.set_xlabel(r"model scale $\longrightarrow$ coarse model fails ($\mathcal{D}$ rises)")
    ax.set_ylabel(r"execution margin $\longrightarrow$ more slack")
    ax.set_xticks([]); ax.set_yticks([])
    fig.tight_layout(); fig.savefig(OUT / "bands.pdf"); fig.savefig(OUT / "bands.png")
    plt.close(fig); print("bands.pdf")

if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--which", default="ghosts,ablation,clock,curves")
    a = ap.parse_args()
    fns = dict(ghosts=fig_ghosts, ablation=fig_ablation, clock=fig_clock,
               curves=fig_curves, eta=fig_eta, bands=fig_bands)
    for w in a.which.split(","):
        w = w.strip()
        if w in fns:
            fns[w]()
