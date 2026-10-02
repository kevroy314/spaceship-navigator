"""Train the navigation policy with PPO.

    python scripts/train.py --run ppo_v1 --updates 1500 --ckpt-every 50

Writes to data/rl/<run>/:
  config.json            hyper-parameters
  train.jsonl            one line of training metrics per logged update
  eval.jsonl             one line per checkpoint: metrics on every frozen suite
  ckpt_XXXXX.msgpack     parameters
  traj_XXXXX.pkl         showcase trajectories for the report
"""

import argparse
import dataclasses
import json
import pickle
import time
from pathlib import Path

import jax
import numpy as np
from flax import serialization

from spacenav.levels.build import load_pool
from spacenav.rl.evaluate import SUITES_PATH, Evaluator, build_suites
from spacenav.rl.ppo import PPOConfig, make

ROOT = Path(__file__).resolve().parents[1]


def to_py(x):
    """JSON-safe: NaN (a rate with no episodes behind it) becomes null."""
    if isinstance(x, dict):
        return {k: to_py(v) for k, v in x.items()}
    if isinstance(x, (np.ndarray, jax.Array)):
        x = x.item() if x.ndim == 0 else x.tolist()
    if isinstance(x, float) and not np.isfinite(x):
        return None
    if isinstance(x, list):
        return [to_py(v) for v in x]
    return x


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--run", default="ppo_v1")
    ap.add_argument("--updates", type=int, default=1500)
    ap.add_argument("--ckpt-every", type=int, default=50)
    ap.add_argument("--log-every", type=int, default=10)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--resume", action="store_true",
                    help="continue from state_latest.msgpack in the run directory")
    for f in dataclasses.fields(PPOConfig):
        ap.add_argument(f"--{f.name.replace('_', '-')}", type=type(f.default), default=f.default)
    args = ap.parse_args()
    cfg = PPOConfig(**{f.name: getattr(args, f.name) for f in dataclasses.fields(PPOConfig)})
    cfg = dataclasses.replace(cfg, total_updates=args.updates)   # schedules follow the real run length

    out = ROOT / "data" / "rl" / args.run
    out.mkdir(parents=True, exist_ok=True)
    (out / "config.json").write_text(json.dumps(dict(dataclasses.asdict(cfg), updates=args.updates,
                                                     seed=args.seed), indent=2))
    if not SUITES_PATH.exists():
        build_suites()

    pool, _ = load_pool(ROOT / "data" / "pools" / "train.npz")
    net, runner, update = make(cfg, pool, jax.random.PRNGKey(args.seed))
    evaluator = Evaluator(net)
    n_params = sum(x.size for x in jax.tree.leaves(runner.params))
    print(f"run {args.run}: {n_params:,} params, batch {cfg.batch:,}, {args.updates} updates "
          f"= {args.updates * cfg.batch / 1e6:.0f}M env steps", flush=True)

    # resume: restore params + optimiser state and carry on where the run stopped
    start = 0
    best = {}
    state_path = out / "state_latest.msgpack"
    ckpts_on_disk = sorted(out.glob("ckpt_*.msgpack"))
    if args.resume and (state_path.exists() or ckpts_on_disk):
        if state_path.exists():
            blob = serialization.from_bytes(
                dict(params=runner.params, opt_state=runner.opt_state, update=0),
                state_path.read_bytes())
            runner = runner._replace(params=blob["params"], opt_state=blob["opt_state"])
            start = int(blob["update"])
        else:   # older checkpoints hold parameters only; Adam state restarts
            last = ckpts_on_disk[-1]
            runner = runner._replace(params=serialization.from_bytes(runner.params, last.read_bytes()))
            start = int(last.stem.split("_")[1])
            print(f"no optimiser state saved; resuming from {last.name} with a fresh Adam state",
                  flush=True)
        for line in (out / "eval.jsonl").read_text().splitlines():
            rec = json.loads(line)
            if rec["update"] <= start:
                for k, m in rec["metrics"].items():
                    best[k] = max(best.get(k, 0.0), m["success"])
        print(f"resumed at update {start} ({start * cfg.batch / 1e6:.0f}M steps)", flush=True)

    def checkpoint(u, steps):
        t = time.time()
        (out / f"ckpt_{u:05d}.msgpack").write_bytes(serialization.to_bytes(runner.params))
        state_path.write_bytes(serialization.to_bytes(
            dict(params=runner.params, opt_state=runner.opt_state, update=u)))
        metrics, trajs, per_task = evaluator.evaluate(runner.params)
        with open(out / f"traj_{u:05d}.pkl", "wb") as f:
            pickle.dump(dict(showcase=trajs, per_task=per_task), f)
        # monotonicity, with a noise band: 2 binomial standard errors on the suite size
        regress = {}
        for k, m in metrics.items():
            n = sum(m["outcomes"].values())
            band = max(0.02, 2 * (m["success"] * (1 - m["success"]) / max(n, 1)) ** 0.5)
            if k in best and m["success"] < best[k] - band:
                regress[k] = round(best[k] - m["success"], 3)
        for k, m in metrics.items():
            best[k] = max(best.get(k, 0.0), m["success"])
        rec = dict(update=u, env_steps=steps, metrics=metrics, regressions=regress)
        with open(out / "eval.jsonl", "a") as f:
            f.write(json.dumps(rec) + "\n")
        line = " ".join(f"{k}={m['success']:.0%}/win{m['win_rate_vs_pilot']:.0%}" for k, m in metrics.items())
        print(f"[ckpt {u:5d}] {line}  ({time.time() - t:.0f}s){'  REGRESSION ' + str(regress) if regress else ''}",
              flush=True)

    if start == 0:
        checkpoint(0, 0)
    t0 = time.time()
    for u in range(start + 1, args.updates + 1):
        frac = u / args.updates
        cur, ent = cfg.curriculum(frac), cfg.entropy_coef(frac)
        runner, m = update(runner, cur["p_rdv"], ent, cur["wp_max"], cur["p_order_free"],
                           cur["budget"], cur["accel"], cur["p_contested"])
        if u % args.log_every == 0 or u == 1:
            m = to_py(jax.device_get(m))
            steps = u * cfg.batch
            m.update(update=u, env_steps=steps, p_rendezvous=cur["p_rdv"],
                     p_contested=cur["p_contested"],
                     wp_max=cur["wp_max"], p_order_free=cur["p_order_free"],
                     budget_lo=cur["budget"][0], accel_lo=cur["accel"][0],
                     sps=(u - start) * cfg.batch / (time.time() - t0), wall=time.time() - t0)
            with open(out / "train.jsonl", "a") as f:
                f.write(json.dumps(m) + "\n")
            print(f"u{u:5d} {steps / 1e6:7.1f}M  sps {m['sps']:,.0f}  ret {m['ep_return']:6.2f}  "
                  f"succ {m['success']:.2f} (fly {m['success_flythrough']:.2f} rdv {m['success_rendezvous']:.2f})  "
                  f"wp {m['waypoints']:.1f}/{m['reached_frac']:.2f}  "
                  f"len {m['ep_length']:5.0f}  crash {m['crash']:.2f} dest {m['destroyed']:.2f} "
                  f"lost {m['lost']:.2f} tout {m['timeout']:.2f}  ent {m['entropy']:.2f} kl {m['approx_kl']:.4f}",
                  flush=True)
        if u % args.ckpt_every == 0 or u == args.updates:
            checkpoint(u, u * cfg.batch)


if __name__ == "__main__":
    main()
