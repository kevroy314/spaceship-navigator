"""Render a ghost fleet as an MP4: either all attempts flying on one mission
clock, or the trails accumulating as training proceeds.

    python scripts/ghost_video.py --track medium_1 --mode replay --out docs/media/replay_medium_1.mp4
"""
import argparse, json, pathlib
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.animation import FFMpegWriter
from matplotlib.collections import LineCollection, EllipseCollection
from matplotlib.colors import Normalize

ROOT = pathlib.Path(__file__).resolve().parents[1]
GREY, GREEN = "#7c8595", "#19b355"


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--track", default="medium_1")
    ap.add_argument("--mode", default="replay", choices=["replay", "learning"])
    ap.add_argument("--seconds", type=float, default=20.0)
    ap.add_argument("--fps", type=int, default=30)
    ap.add_argument("--tail", type=int, default=16)
    ap.add_argument("--title", default=None)
    ap.add_argument("--out", required=True)
    a = ap.parse_args()

    d = json.loads((ROOT / "data" / "ghosts" / f"{a.track}.json").read_text())
    ghosts = sorted(d["ghosts"], key=lambda g: g["u"])
    stride = d["stride"]
    paths = [np.array(g["p"], float) for g in ghosts]
    ups = np.array([g["u"] for g in ghosts], float)
    norm = Normalize(0, 3000)
    cmap = plt.get_cmap("viridis")
    cols = cmap(norm(ups))

    pts = np.concatenate([p for p in paths if len(p)] +
                         [np.array(w["path"], float) for w in d["waypoints"]], 0)
    lo, hi = pts.min(0), pts.max(0)
    c = (lo + hi) / 2
    half = max(hi[0] - lo[0], hi[1] - lo[1]) / 2 * 1.06 + 10

    fig, ax = plt.subplots(figsize=(7.2, 7.2), dpi=110)
    fig.patch.set_facecolor("#0c0e13"); ax.set_facecolor("#0c0e13")
    ax.set_xlim(c[0] - half, c[0] + half); ax.set_ylim(c[1] + half, c[1] - half)
    ax.set_aspect("equal"); ax.set_xticks([]); ax.set_yticks([])
    for sp in ax.spines.values():
        sp.set_visible(False)

    for b in d["bodies"]:
        pa = np.array(b["path"], float)
        if len(pa) > 1:
            ax.plot(pa[:, 0], pa[:, 1], color="#39404e", lw=0.5, zorder=1)
        ax.add_patch(plt.Circle((pa[0, 0], pa[0, 1]), max(4.0, b["r"]),
                                facecolor="#2b313d", edgecolor="#5d6675", lw=0.6, zorder=2))
    wrings = []
    for w in d["waypoints"]:
        pa = np.array(w["path"], float)
        ax.plot(pa[:, 0], pa[:, 1], color=GREEN, lw=0.6, alpha=0.30, zorder=3)
        ring = plt.Circle((pa[0, 0], pa[0, 1]), max(8.0, w["r"]), facecolor="none",
                          edgecolor=GREEN, lw=1.6, zorder=9)
        ax.add_patch(ring); wrings.append((ring, pa))

    lc = LineCollection([], linewidths=1.0, zorder=5)
    ax.add_collection(lc)
    heads = ax.scatter([], [], s=7, zorder=8, linewidths=0)
    label = ax.text(0.02, 0.975, "", transform=ax.transAxes, color="#aeb6c6",
                    fontsize=11, va="top", family="monospace")
    title = ax.text(0.02, 0.025, a.title or d.get("label", a.track), transform=ax.transAxes,
                    color="#e8ebf2", fontsize=12, va="bottom", family="monospace")

    nframes = int(a.seconds * a.fps)
    maxT = max((len(p) - 1) * stride for p in paths)
    writer = FFMpegWriter(fps=a.fps, bitrate=3200,
                          metadata=dict(title=f"spacenav ghost fleet: {a.track}"))
    out = pathlib.Path(a.out)
    if not out.is_absolute():
        out = ROOT / out
    out.parent.mkdir(parents=True, exist_ok=True)

    with writer.saving(fig, str(out), dpi=110):
        for fr in range(nframes):
            u = fr / max(1, nframes - 1)
            segs, scols, hx, hy, hc = [], [], [], [], []
            if a.mode == "replay":
                T = u * maxT
                f = T / stride; idx = int(f); frac = f - idx
                for i, p in enumerate(paths):
                    n = len(p)
                    if f >= n - 1:
                        continue
                    hxy = p[idx] + (p[idx + 1] - p[idx]) * frac
                    j0 = max(0, idx - a.tail)
                    seg = np.vstack([p[j0:idx + 1], hxy])
                    if len(seg) > 1:
                        segs.append(seg); scols.append(cols[i])
                        hx.append(hxy[0]); hy.append(hxy[1]); hc.append(cols[i])
                for ring, pa in wrings:
                    k = min(len(pa) - 1, int(T / (stride * 4)))
                    ring.center = (pa[k][0], pa[k][1])
                label.set_text(f"t = {T*d['ctrl_dt']:5.1f} s   {len(segs):3d} flying")
            else:
                cut = u * 3000
                for i, p in enumerate(paths):
                    if ups[i] > cut or len(p) < 2:
                        continue
                    segs.append(p); scols.append(cols[i])
                label.set_text(f"training update {int(cut):5d}   {len(segs):3d} attempts")
            lc.set_segments(segs)
            if segs:
                cc = np.array(scols); cc[:, 3] = 0.16 + 0.5 * norm(ups[:len(cc)] * 0 + 1500)
                cc = np.array(scols); cc[:, 3] = 0.55 if a.mode == "replay" else 0.22
                lc.set_color(cc)
            heads.set_offsets(np.c_[hx, hy] if hx else np.empty((0, 2)))
            if hc:
                heads.set_color(np.array(hc))
            writer.grab_frame()
    plt.close(fig)
    print(f"{out}  {out.stat().st_size/1e6:.1f} MB  {nframes} frames")


if __name__ == "__main__":
    main()
