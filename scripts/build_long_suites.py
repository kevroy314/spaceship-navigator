"""Evaluation suites that are valid at a long episode clock.

    SPACENAV_EPISODE_S=600 python scripts/build_long_suites.py
    python scripts/build_long_suites.py --clock 600            # same thing

The frozen suites in data/eval/suites.pkl draw from val_seen / val_holdout, whose
levels carry 8 snapshots to 140 s and were validated to 300 s.  A 600 s episode
started from a late snapshot runs to 740 s, well past where the level was checked
to be well behaved (docs/decisions/0019).  The `long` family of pools is built for
this: 16 snapshots to 300 s, validated to LEVEL_HORIZON = 900 s, so an episode of
up to 600 s from any snapshot stays inside the validated window.

This writes data/eval/suites_long.pkl: the same four suites, same SUITE_SPECS task
configs and sizes, but every suite drawn from one long-horizon pool.  Two caveats
that the 150 s suites do not have:

  * `blind` is not blind.  The long pools hold the nine TRAINING families only
    (HOLDOUT_FAMILIES were never built at the 900 s horizon), so `blind` here
    measures seen families with the blind task mix, not generalisation.
  * `long` itself (what `train.py --pool long` trains on) has 200 levels per
    family, and the `easy` spec would take all 600 levels of its three families,
    so suites drawn from `long` sit entirely inside the training set.  The default
    is therefore `long_val`: the same recipe as `long` with a fresh seed block,
    built here on first use so the suites play the role val_seen plays at 150 s.

The pilot yardstick stored with each suite is flown at the clock this script
runs under, and the clock is stored with the suite; load_suites() warns when a
process at a different clock reads it.  Point training at the file with
SPACENAV_SUITES=data/eval/suites_long.pkl.
"""

import argparse
import os
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
LONG_VAL_SEED0 = 40_000_000     # train 0, val_seen 10M, val_holdout 20M, long 30M


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--clock", type=float, default=600.0,
                    help="episode clock (s) the pilot yardstick is flown at; sets "
                         "SPACENAV_EPISODE_S for this process")
    ap.add_argument("--pool", default="long_val",
                    help="long-horizon pool to draw every suite from (`long_val` is built on "
                         "first use; `long` is the training pool and overlaps it entirely)")
    ap.add_argument("--val-n", type=int, default=200, help="levels per family when building long_val")
    ap.add_argument("--out", default="data/eval/suites_long.pkl")
    ap.add_argument("--seed", type=int, default=0)
    args = ap.parse_args()

    # The clock is read once, at import of spacenav.constants, so it has to be in the
    # environment before anything from spacenav is imported.
    env_clock = os.environ.get("SPACENAV_EPISODE_S")
    if env_clock is not None and abs(float(env_clock) - args.clock) > 1e-6:
        sys.exit(f"SPACENAV_EPISODE_S={env_clock} disagrees with --clock {args.clock}; set one")
    os.environ["SPACENAV_EPISODE_S"] = repr(float(args.clock))

    from spacenav import constants as C
    from spacenav.levels.families import TRAIN_FAMILIES
    from spacenav.rl.evaluate import DATA, N_PER_SUITE, build_suites
    sys.path.insert(0, str(ROOT / "scripts"))
    from build_pools import build as build_pool

    if C.MAX_EPISODE_TIME > C.LEVEL_HORIZON - 0.0:
        sys.exit(f"clock {C.MAX_EPISODE_TIME} s exceeds LEVEL_HORIZON {C.LEVEL_HORIZON} s")

    pool_path = DATA / "pools" / f"{args.pool}.npz"
    if not pool_path.exists():
        if args.pool != "long_val":
            sys.exit(f"{pool_path} does not exist")
        # N_SNAPSHOTS in levels/build.py is derived from the clock at import, so this
        # stores (900 - clock)/20 + 1 snapshots: 16 at 600 s, matching `long`.
        print(f"building {args.pool}: {args.val_n}/family x {len(TRAIN_FAMILIES)} training "
              f"families, seeds from {LONG_VAL_SEED0:,}, validated to {C.LEVEL_HORIZON:.0f} s",
              flush=True)
        build_pool(args.pool, TRAIN_FAMILIES, args.val_n, LONG_VAL_SEED0)

    out = Path(args.out)
    out = out if out.is_absolute() else ROOT / out
    print(f"suites from {args.pool} at a {C.MAX_EPISODE_TIME:.0f} s clock "
          f"({C.MAX_EPISODE_TICKS} ticks), {N_PER_SUITE} tasks each -> {out}", flush=True)
    build_suites(seed=args.seed, pool=args.pool, path=out)
    print("NOTE: `blind` here is drawn from training families (the long pools have no held-out "
          "families); it is not a generalisation measurement.", flush=True)


if __name__ == "__main__":
    main()
