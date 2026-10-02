"""Generate and validate the level pools used for training and evaluation.

    python scripts/build_pools.py [--train 2000] [--val 200] [--holdout 300]

Pools (data/pools/):
  train.npz        TRAIN_FAMILIES, seeds [0, ...)
  val_seen.npz     TRAIN_FAMILIES, seeds [10_000_000, ...)   same families, unseen levels
  val_holdout.npz  HOLDOUT_FAMILIES, seeds [20_000_000, ...) families never trained on
"""

import argparse
import json
import time
from pathlib import Path

from spacenav.levels.build import build_levels, concat_levels, save_pool
from spacenav.levels.families import HOLDOUT_FAMILIES, TRAIN_FAMILIES

OUT = Path(__file__).resolve().parents[1] / "data" / "pools"


def build(name, families, n, seed0):
    pools, meta, stats = [], [], {}
    for fam in families:
        t = time.time()
        lv, m, reasons = build_levels(fam, n, seed0=seed0, batch=256, max_tries=100)
        pools.append(lv)
        meta += m
        rejected = sum(v for k, v in reasons.items() if k != "tried")
        stats[fam] = dict(reasons, accept_rate=round(1 - rejected / reasons["tried"], 3),
                          seconds=round(time.time() - t, 1))
        print(f"  {name}/{fam}: {n} levels, accept {stats[fam]['accept_rate']:.0%}, "
              f"{stats[fam]['seconds']}s", flush=True)
    save_pool(OUT / f"{name}.npz", concat_levels(pools), meta)
    return stats


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--train", type=int, default=2000)
    ap.add_argument("--val", type=int, default=200)
    ap.add_argument("--holdout", type=int, default=300)
    args = ap.parse_args()
    OUT.mkdir(parents=True, exist_ok=True)
    report = {
        "train": build("train", TRAIN_FAMILIES, args.train, 0),
        "val_seen": build("val_seen", TRAIN_FAMILIES, args.val, 10_000_000),
        "val_holdout": build("val_holdout", HOLDOUT_FAMILIES, args.holdout, 20_000_000),
    }
    (OUT / "report.json").write_text(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
