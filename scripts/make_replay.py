"""Build the animated replay page: every checkpoint flying the same task at once.

    python scripts/make_replay.py --run ppo_v2 [--per-tier 4] [--stride 2]

For each showcase task it collects one flight per checkpoint (the "ghost fleet"),
plus the scripted pilot and, where one exists, the planned reference flight, and
the motion of every body.  The page plays them all on a shared clock.
"""

import argparse
import json
import pickle
from pathlib import Path

import jax
import numpy as np

from spacenav import constants as C
from spacenav.levels.build import index_level, load_pool
from spacenav.levels.families import FAMILY_NAMES
from spacenav.reporting import episode_geometry, r1
from spacenav.rl.evaluate import _pilot_rollout, load_suites
from spacenav.types import REASONS

ROOT = Path(__file__).resolve().parents[1]
TIER_ORDER = ["easy", "medium", "hard", "blind"]


def sample(path, stride):
    """Every `stride`-th point, always keeping the last one (where the flight ended)."""
    keep = np.unique(np.r_[np.arange(0, len(path), stride), len(path) - 1])
    return r1(np.asarray(path)[keep])


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--run", default="ppo_v2")
    ap.add_argument("--per-tier", type=int, default=4)
    ap.add_argument("--stride", type=int, default=2, help="trajectory sampling, in control ticks")
    ap.add_argument("--body-stride", type=int, default=4)
    args = ap.parse_args()

    run_dir = ROOT / "data" / "rl" / args.run
    evals = [json.loads(l) for l in (run_dir / "eval.jsonl").read_text().splitlines()]
    ckpts = [e["update"] for e in evals]
    steps = {e["update"]: e["env_steps"] for e in evals}
    showcase = {}
    for u in ckpts:
        d = pickle.load(open(run_dir / f"traj_{u:05d}.pkl", "rb"))
        showcase[u] = d["showcase"] if isinstance(d, dict) and "showcase" in d else d

    suites = load_suites()
    ref_path = ROOT / "data" / "eval" / "references.pkl"
    refs = pickle.load(open(ref_path, "rb")) if ref_path.exists() else {}
    pools, episodes = {}, []

    for tier in [t for t in TIER_ORDER if t in suites]:
        s = suites[tier]
        pools.setdefault(s["pool"], load_pool(ROOT / "data" / "pools" / f"{s['pool']}.npz")[0])
        for j, i in list(enumerate(s["showcase"]))[: args.per_tier]:
            i = int(i)
            level = index_level(pools[s["pool"]], int(s["level_idx"][i]))
            task = jax.tree.map(lambda x: np.asarray(x[i]), s["task"])
            start = np.asarray(task.start_pos)[None]

            ghosts, longest = [], 1
            for u in ckpts:
                t = showcase[u][tier][j]
                pos = np.concatenate([start, t["pos"]])
                longest = max(longest, len(pos))
                ghosts.append(dict(update=u, steps=steps[u], status=t["status"],
                                   objective=round(float(t["objective"]), 3),
                                   ticks=len(t["pos"]), path=sample(pos, args.stride)))

            _, ptr = _pilot_rollout(level, jax.tree.map(jax.numpy.asarray, task))
            pst = np.asarray(ptr["status"])
            p_end = int(np.argmax(pst != 0)) + 1 if (pst != 0).any() else C.MAX_EPISODE_TICKS
            ppos = np.concatenate([start, np.asarray(ptr["pos"])[:p_end]])
            longest = max(longest, len(ppos))
            pilot = dict(status=REASONS[int(s["pilot"]["status"][i])], ticks=p_end,
                         objective=round(float(s["pilot"]["objective"][i]), 3),
                         path=sample(ppos, args.stride))

            reference = None
            r = refs.get(tier)
            if r is not None and i in r.get("pos", {}):
                rp = np.concatenate([start, np.asarray(r["pos"][i])])
                longest = max(longest, len(rp))
                k = int(np.nonzero(r["idx"] == i)[0][0])
                reference = dict(path=sample(rp, args.stride), ticks=len(rp),
                                 cost=round(float(r["cost"][k]), 3))

            geo = episode_geometry(level, task, longest + 4, args.body_stride)
            episodes.append(dict(tier=tier, family=FAMILY_NAMES[int(np.asarray(level.family))],
                                 level=int(s["level_idx"][i]), ticks=int(longest), **geo,
                                 ghosts=ghosts, pilot=pilot, reference=reference))
            print(f"  {tier}/{FAMILY_NAMES[int(np.asarray(level.family))]}: "
                  f"{len(ghosts)} ghosts, {longest} ticks", flush=True)

    data = dict(run=args.run, checkpoints=ckpts, steps=[steps[u] for u in ckpts],
                episodes=episodes, stride=args.stride, ctrl_dt=C.CTRL_DT,
                max_ticks=max(e["ticks"] for e in episodes),
                final={t: evals[-1]["metrics"][t]["success"] for t in suites})
    html = (ROOT / "scripts" / "replay_template.html").read_text().replace(
        "/*__DATA__*/null", json.dumps(data, separators=(",", ":")))
    out = ROOT / "data" / "reports" / f"{args.run}_replay.html"
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(html)
    print(f"wrote {out} ({len(html) / 1e6:.2f} MB, {len(episodes)} episodes x {len(ckpts)} ghosts)")


if __name__ == "__main__":
    main()
