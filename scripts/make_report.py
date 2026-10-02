"""Build the training report page for a run.

    python scripts/make_report.py --run ppo_v1

Collects eval.jsonl, train.jsonl and every checkpoint's showcase trajectories,
re-derives each showcase episode's geometry (bodies, zones, body paths over the
episode, A and B, the pilot's path), and embeds it all as JSON into
scripts/report_template.html -> data/reports/<run>_report.html.
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
STRIDE = 4          # keep every 4th tick of a trajectory (15 Hz -> 3.75 Hz)
BODY_STRIDE = 8     # body paths are smoother, sample more sparsely


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--run", default="ppo_v1")
    ap.add_argument("--note", action="append", default=[], help="run-specific note for the report")
    args = ap.parse_args()
    run_dir = ROOT / "data" / "rl" / args.run
    evals = [json.loads(l) for l in (run_dir / "eval.jsonl").read_text().splitlines()]
    train = [json.loads(l) for l in (run_dir / "train.jsonl").read_text().splitlines()]
    config = json.loads((run_dir / "config.json").read_text())
    suites = load_suites()
    pools = {}
    ref_path = ROOT / "data" / "eval" / "references.pkl"
    refs = pickle.load(open(ref_path, "rb")) if ref_path.exists() else {}
    prev_path = ROOT / "data" / "eval" / "ppo_v1_final.json"
    previous = json.loads(prev_path.read_text())["metrics"] if prev_path.exists() else None

    ckpts = [e["update"] for e in evals]
    trajs, per_task = {}, {}
    for u in ckpts:
        d = pickle.load(open(run_dir / f"traj_{u:05d}.pkl", "rb"))
        if isinstance(d, dict) and "showcase" in d:        # newer runs also store per-task costs
            trajs[u], per_task[u] = d["showcase"], d["per_task"]
        else:
            trajs[u] = d

    gallery = {}
    for name, s in suites.items():
        if s["pool"] not in pools:
            pools[s["pool"]] = load_pool(ROOT / "data" / "pools" / f"{s['pool']}.npz")[0]
        eps = []
        for j, i in enumerate(s["showcase"]):
            i = int(i)
            level = index_level(pools[s["pool"]], int(s["level_idx"][i]))
            task = jax.tree.map(lambda x: np.asarray(x[i]), s["task"])
            _, ptr = _pilot_rollout(level, jax.tree.map(jax.numpy.asarray, task))
            pst = np.asarray(ptr["status"])
            p_end = int(np.argmax(pst != 0)) + 1 if (pst != 0).any() else C.MAX_EPISODE_TICKS
            runs = []
            longest = p_end
            for u in ckpts:
                t = trajs[u][name][j]
                longest = max(longest, len(t["pos"]))
                pos = np.concatenate([np.asarray(task.start_pos)[None], t["pos"]])
                keep = np.unique(np.r_[np.arange(0, len(pos), STRIDE), len(pos) - 1])
                runs.append(dict(update=u, status=t["status"], objective=round(t["objective"], 3),
                                 ticks=len(t["pos"]), path=r1(pos[keep])))
            ppos = np.concatenate([np.asarray(task.start_pos)[None], np.asarray(ptr["pos"])[:p_end]])
            keep = np.unique(np.r_[np.arange(0, len(ppos), STRIDE), len(ppos) - 1])
            geo = episode_geometry(level, task, longest, BODY_STRIDE)
            ref_pos = refs.get(name, {}).get("pos", {}).get(i)
            ref_cost = None
            if name in refs and i in set(int(x) for x in refs[name]["idx"]):
                j = int(np.nonzero(refs[name]["idx"] == i)[0][0])
                if np.isfinite(refs[name]["cost"][j]):
                    ref_cost = round(float(refs[name]["cost"][j]), 3)
            eps.append(dict(
                family=FAMILY_NAMES[int(np.asarray(level.family))], level=int(s["level_idx"][i]),
                pool=s["pool"], **geo, runs=runs,
                pilot=dict(status=REASONS[int(s["pilot"]["status"][i])],
                           objective=round(float(s["pilot"]["objective"][i]), 3), path=r1(ppos[keep])),
                reference=(dict(cost=ref_cost, path=r1(np.asarray(ref_pos)[::STRIDE]))
                           if ref_pos is not None else (dict(cost=ref_cost, path=None) if ref_cost else None))))
        gallery[name] = eps

    # cost against the best known flight (planner or pilot), on the reference subset
    ref_summary = {}
    for name, r in refs.items():
        have = r["have"]
        ref_summary[name] = dict(
            n=int(have.sum()), planner_arrived=float(np.mean(r["plan_arrived"])),
            planner_beats_pilot=float(np.mean(r["plan_cost"][r["plan_arrived"] & r["pilot_arrived"]]
                                              < r["pilot_cost"][r["plan_arrived"] & r["pilot_arrived"]]))
            if (r["plan_arrived"] & r["pilot_arrived"]).any() else None,
            mean_cost=float(np.mean(r["cost"][have])), mean_time=float(np.mean(r["plan_time"][r["plan_arrived"]]))
            if r["plan_arrived"].any() else None,
            by_update={})
        for u in ckpts:
            if u not in per_task or name not in per_task[u]:
                continue
            pt = per_task[u][name]
            sub_obj, sub_st = pt["objective"][r["idx"]], pt["status"][r["idx"]]
            ok = (sub_st == 1) & have
            if ok.any():
                ratio = sub_obj[ok] / np.maximum(r["cost"][ok], 1e-6)
                ref_summary[name]["by_update"][str(u)] = dict(
                    median_ratio=float(np.median(ratio)), arrived=float(np.mean(sub_st[have] == 1)),
                    beats_best_known=float(np.mean(ratio < 1)))

    notes = list(args.note)
    for name, r in refs.items():
        if not r["plan_arrived"].any():
            notes.append(f"No planner reference exists for the {name} suite: an open-loop plan can fly to a "
                         f"moving station but cannot match its velocity on arrival, so the best known flight "
                         f"there is the scripted pilot's.")
    pilot_cost = {}
    for name, s in suites.items():
        ok = s["pilot"]["status"] == 1
        pilot_cost[name] = float(s["pilot"]["objective"][ok].mean()) if ok.any() else None
    train_levels = len(load_pool(ROOT / "data" / "pools" / "train.npz")[1])
    data = dict(run=args.run, config=config, checkpoints=ckpts, pilot_cost=pilot_cost,
                train_levels=train_levels,
                references=ref_summary, previous=previous, notes=notes,
                suite_size=int(sum(evals[-1]["metrics"][next(iter(evals[-1]["metrics"]))]["outcomes"].values())),
                evals=[dict(update=e["update"], env_steps=e["env_steps"], regressions=e["regressions"],
                            metrics={k: {kk: m[kk] for kk in ("success", "pilot_success", "objective",
                                                              "cost_ratio_vs_pilot", "win_rate_vs_pilot",
                                                              "damage", "outcomes", "by_family")}
                                     for k, m in e["metrics"].items()}) for e in evals],
                train=[{k: t[k] for k in ("update", "env_steps", "success", "success_flythrough",
                                          "success_rendezvous", "ep_return", "p_rendezvous", "sps",
                                          "entropy", "crash", "destroyed", "lost", "timeout", "wall")}
                       for t in train],
                gallery=gallery, stride=STRIDE, body_stride=BODY_STRIDE, ctrl_dt=C.CTRL_DT)
    blob = json.dumps(data, separators=(",", ":"))
    html = (ROOT / "scripts" / "report_template.html").read_text().replace("/*__DATA__*/null", blob)
    out = ROOT / "data" / "reports" / f"{args.run}_report.html"
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(html)
    print(f"wrote {out} ({len(html) / 1e6:.2f} MB, {len(ckpts)} checkpoints)")


if __name__ == "__main__":
    main()
