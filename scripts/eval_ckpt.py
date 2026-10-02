"""Score a saved checkpoint on the current frozen suites.

    python scripts/eval_ckpt.py data/rl/ppo_v1/ckpt_02000.msgpack --tag ppo_v1_final

Writes data/eval/<tag>.json so older runs can be compared on the same tasks.
"""

import argparse
import json
import pickle
from pathlib import Path

import jax
import numpy as np
from flax import serialization

from spacenav.rl.evaluate import DATA, Evaluator
from spacenav.rl.network import ActorCritic

ROOT = Path(__file__).resolve().parents[1]


def load_params(path, net, d_model=128):
    ev = Evaluator(net)
    suite = next(iter(ev.suites.values()))
    lv = jax.tree.map(lambda x: x[0], ev.pools[suite["pool"]])
    task = jax.tree.map(lambda x: jax.numpy.asarray(x[0]), suite["task"])
    from spacenav import env as E
    obs = E.observe(lv, task, E.reset(lv, task))
    template = net.init(jax.random.PRNGKey(0), obs)
    return ev, serialization.from_bytes(template, Path(path).read_bytes())


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("ckpt")
    ap.add_argument("--tag", default=None)
    ap.add_argument("--d-model", type=int, default=128)
    args = ap.parse_args()
    net = ActorCritic(d=args.d_model)
    ev, params = load_params(args.ckpt, net, args.d_model)
    metrics, trajs, per_task = ev.evaluate(params)
    tag = args.tag or Path(args.ckpt).stem
    out = DATA / "eval" / f"{tag}.json"
    out.write_text(json.dumps(dict(ckpt=str(args.ckpt), metrics=metrics), indent=2))
    with open(DATA / "eval" / f"{tag}_traj.pkl", "wb") as f:
        pickle.dump(dict(showcase=trajs, per_task=per_task), f)
    for k, m in metrics.items():
        print(f"{k:8s} success {m['success']:.0%}  pilot {m['pilot_success']:.0%}  "
              f"win {m['win_rate_vs_pilot']:.0%}")


if __name__ == "__main__":
    main()
