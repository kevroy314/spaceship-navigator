"""Frozen evaluation suites and checkpoint evaluation.

Suites are built once (fixed levels, fixed tasks) and saved, so every
checkpoint of every run is scored on exactly the same problems:

  easy    val_seen, planetary/moon systems, short fly-throughs
  medium  val_seen, all training families, fly-throughs
  hard    val_seen, all training families, station rendezvous
  blind   val_holdout families (never trained on), half fly-through / half rendezvous

The scripted pilot is scored on the same tasks and serves as the yardstick:
"win" = the agent arrives and the pilot fails, or both arrive and the agent's
weighted cost is lower.
"""

import collections
import pickle
from pathlib import Path

import jax
import jax.numpy as jnp
import numpy as np

from spacenav import baselines
from spacenav import constants as C
from spacenav import env as E
from spacenav.levels.build import load_pool
from spacenav.rl.network import ACTIONS
from spacenav.types import ARRIVED, REASONS

DATA = Path(__file__).resolve().parents[2] / "data"
SUITES_PATH = DATA / "eval" / "suites.pkl"

SUITE_SPECS = {
    # one target, a roomy tank and a strong engine: the old "point and burn" job
    "easy": dict(pool="val_seen", families=("sol_like", "jovian", "saturnian"),
                 cfg=E.TaskConfig(waypoints=(1, 1), p_rendezvous=0.0, min_dist=100.0,
                                  max_dist=350.0, deltav_budget=(1.0, 1.5), accel_range=(1.5, 3.0))),
    # short tours on a budget that leaves little to waste
    "medium": dict(pool="val_seen", families=None,
                   cfg=E.TaskConfig(waypoints=(1, 2), p_station=0.45, p_rendezvous=0.25,
                                    max_dist=800.0, deltav_budget=(0.8, 1.25))),
    # Voyager-style: three stops, half of them docks, and usually not enough fuel
    # to fly it directly - gravity has to do some of the work
    "hard": dict(pool="val_seen", families=None,
                 cfg=E.TaskConfig(waypoints=(2, 3), p_station=0.6, p_rendezvous=0.5,
                                  max_dist=600.0, deltav_budget=(0.7, 1.05),
                                  accel_range=(0.8, 2.0))),
    # never-trained families, and sometimes the tour order is the pilot's to choose
    "blind": dict(pool="val_holdout", families=None,
                  cfg=E.TaskConfig(waypoints=(1, 3), p_order_free=0.3, p_station=0.5,
                                   p_rendezvous=0.35, max_dist=700.0,
                                   deltav_budget=(0.75, 1.15))),
}
N_PER_SUITE = 600   # 600 tasks -> ~2 points of binomial noise on a success rate
N_SHOWCASE = 4


def _gather(levels, idx):
    return jax.tree.map(lambda x: x[idx], levels)


@jax.jit
def _pilot_rollout(lv, task):
    return E.rollout(lv, task, baselines.pilot(lv, task))


def build_suites(seed=0):
    suites = {}
    for name, spec in SUITE_SPECS.items():
        levels, meta = load_pool(DATA / "pools" / f"{spec['pool']}.npz")
        fams = collections.defaultdict(list)
        for i, m in enumerate(meta):
            fams[m["family"]].append(i)
        use = spec["families"] or sorted(fams)
        per = N_PER_SUITE // len(use)
        idx, tasks, fam_of = [], [], []
        key = jax.random.PRNGKey(seed * 1000 + sum(map(ord, name)))  # stable across processes
        sample = jax.jit(jax.vmap(lambda k, l: E.sample_task(k, l, spec["cfg"])))
        for fam in use:
            cand = np.asarray(fams[fam][:per])
            got_idx, got_task = [], []
            for attempt in range(8):
                key, sub = jax.random.split(key)
                lv = _gather(levels, cand)
                t, ok = sample(jax.random.split(sub, len(cand)), lv)
                ok = np.array(ok)
                if spec["cfg"].p_rendezvous >= 1.0:   # demand a real rendezvous leg
                    ok &= np.any(np.asarray(t.wp_v_tol) < C.FLYTHROUGH_V_TOL / 2, axis=-1)
                for j in np.nonzero(ok)[0]:
                    got_idx.append(cand[j]); got_task.append(jax.tree.map(lambda x: np.asarray(x[j]), t))
                if len(got_idx) >= per:
                    break
            got_idx, got_task = got_idx[:per], got_task[:per]
            idx += got_idx; tasks += got_task; fam_of += [fam] * len(got_idx)
        task_b = jax.tree.map(lambda *xs: np.stack(xs), *tasks)
        # pilot yardstick on the same tasks
        lv = _gather(levels, np.asarray(idx))
        fin, _ = jax.vmap(_pilot_rollout)(lv, task_b)
        pilot = dict(status=np.asarray(fin.status),
                     objective=np.asarray(jax.vmap(E.objective_cost)(task_b, fin.costs)),
                     damage=np.asarray(fin.costs.damage), time=np.asarray(fin.costs.time))
        # showcase: one task from each of the first N_SHOWCASE families (padded from the start)
        show, seen = [], set()
        for j, f in enumerate(fam_of):
            if f not in seen and len(show) < N_SHOWCASE:
                show.append(j); seen.add(f)
        while len(show) < N_SHOWCASE:
            show.append(len(show))
        suites[name] = dict(pool=spec["pool"], level_idx=np.asarray(idx), task=task_b,
                            family=np.asarray(fam_of), pilot=pilot, showcase=np.asarray(show))
        print(f"suite {name}: {len(idx)} tasks, pilot success {np.mean(pilot['status'] == ARRIVED):.0%}")
    SUITES_PATH.parent.mkdir(parents=True, exist_ok=True)
    with open(SUITES_PATH, "wb") as f:
        pickle.dump(suites, f)
    return suites


def load_suites():
    with open(SUITES_PATH, "rb") as f:
        return pickle.load(f)


class Evaluator:
    """Greedy-policy evaluation of checkpoints on the frozen suites."""

    def __init__(self, net, suites=None):
        self.suites = suites or load_suites()
        self.pools = {s["pool"]: load_pool(DATA / "pools" / f"{s['pool']}.npz")[0]
                      for s in self.suites.values()}
        self.net = net

        def run(params, lv, task):
            def policy(obs, state):
                logits, _ = net.apply(params, obs)
                return ACTIONS[jnp.argmax(logits)]
            return E.rollout(lv, task, policy)
        self._run = jax.jit(jax.vmap(run, in_axes=(None, 0, 0)))

    def evaluate(self, params):
        """Returns (aggregate metrics, showcase trajectories, per-task costs)."""
        metrics, trajs, per_task = {}, {}, {}
        for name, s in self.suites.items():
            lv = _gather(self.pools[s["pool"]], s["level_idx"])
            task = jax.tree.map(jnp.asarray, s["task"])
            fin, tr = self._run(params, lv, task)
            st = np.asarray(fin.status)
            obj = np.asarray(jax.vmap(E.objective_cost)(task, fin.costs))
            ok = st == ARRIVED
            p_ok = s["pilot"]["status"] == ARRIVED
            # ratios only where the pilot's cost is meaningfully non-zero (an exposure-only
            # objective can cost exactly 0)
            both = ok & p_ok & (s["pilot"]["objective"] > 0.02)
            win = (ok & ~p_ok) | (both & (obj < s["pilot"]["objective"]))
            m = dict(
                success=float(ok.mean()),
                pilot_success=float(p_ok.mean()),
                objective=float(obj[ok].mean()) if ok.any() else None,
                cost_ratio_vs_pilot=float(np.median(obj[both] / s["pilot"]["objective"][both])) if both.any() else None,
                win_rate_vs_pilot=float(win.mean()),
                damage=float(np.asarray(fin.costs.damage).mean()),
                fuel=float(np.asarray(fin.costs.fuel)[ok].mean()) if ok.any() else None,
                time=float(np.asarray(fin.costs.time)[ok].mean()) if ok.any() else None,
                outcomes={REASONS[k]: int(v) for k, v in collections.Counter(st.tolist()).items()},
                by_family={f: float(ok[s["family"] == f].mean()) for f in np.unique(s["family"])},
            )
            metrics[name] = m
            # per-task costs: lets a later optimiser reference be applied without re-running
            per_task[name] = dict(status=st.astype(np.int8), objective=obj.astype(np.float32))
            sc = s["showcase"]
            pos = np.asarray(tr["pos"])[sc]
            status = np.asarray(tr["status"])[sc]
            trajs[name] = []
            for j, i in enumerate(sc):
                end = int(np.argmax(status[j] != 0)) + 1 if (status[j] != 0).any() else C.MAX_EPISODE_TICKS
                trajs[name].append(dict(task=int(i), status=REASONS[int(st[i])], objective=float(obj[i]),
                                        pos=pos[j, :end].astype(np.float32)))
        return metrics, trajs, per_task
