#!/usr/bin/env python
"""Probe a spread of episodes, band them, and write a curriculum.

Each episode is placed on two measured axes — the coarsest force model under
which it can still be flown, and how much timing slack the right plan leaves —
and lands in one of the bands in `spacenav.curriculum`.  The human path ramps
the number of gravitational domains a flight crosses; the branches hang off the
end and are meant to be unreachable by extrapolating from it.

    python scripts/build_curriculum.py --out v1 --n 24 --batch 8

The probe is the expensive part: three cross-entropy searches with mid-course
replanning per episode, so budget roughly a minute of GPU per episode and run it
in the background.
"""

import argparse
import json
import time
from pathlib import Path

import jax
import numpy as np

import spacenav.curriculum as CU
from spacenav import episodes as EP
from spacenav.levels.families import FAMILY_NAMES
from spacenav.opt.probe import ProbeConfig, probe_batch

OUT = Path(__file__).resolve().parents[1] / "data" / "curriculum"


def candidates(pools, specs, n, rng):
    """Episode ids worth probing: valid tasks, spread over levels and missions."""
    out = []
    for pool in pools:
        levels, _ = EP.pool(pool)
        n_levels = int(levels.active.shape[0])
        for spec in specs:
            got, tries = 0, 0
            while got < n and tries < n * 12:
                tries += 1
                idx = int(rng.integers(n_levels))
                seed = int(rng.integers(1, 10 ** 6))
                eid = EP.episode_id(pool, idx, seed, spec)
                try:
                    level, task, ok = EP._level_task(eid)
                except Exception:
                    continue
                if not ok:
                    continue
                out.append((eid, level, task, FAMILY_NAMES[int(level.family)]))
                got += 1
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default="v1")
    ap.add_argument("--pools", nargs="+", default=["train"])
    ap.add_argument("--specs", nargs="+",
                    default=["1w0m", "2w50m", "3w50m", "2w100mc", "3w50mc"])
    ap.add_argument("--n", type=int, default=16, help="episodes per (pool, spec)")
    ap.add_argument("--batch", type=int, default=8)
    ap.add_argument("--seed", type=int, default=0)
    args = ap.parse_args()

    rng = np.random.default_rng(args.seed)
    cfg = ProbeConfig()
    cands = candidates(args.pools, args.specs, args.n, rng)
    print(f"{len(cands)} candidate episodes", flush=True)

    probe = jax.jit(lambda l, t, k: probe_batch(l, t, k, cfg))
    rows, ids, families, t0 = [], [], [], time.time()
    for i in range(0, len(cands), args.batch):
        chunk = cands[i: i + args.batch]
        levels = jax.tree.map(lambda *a: np.stack(a), *[c[1] for c in chunk])
        tasks = jax.tree.map(lambda *a: np.stack(a), *[c[2] for c in chunk])
        r = jax.tree.map(np.asarray, probe(levels, tasks, jax.random.PRNGKey(i)))
        rows.append(r)
        ids += [c[0] for c in chunk]
        families += [c[3] for c in chunk]
        done = i + len(chunk)
        rate = (time.time() - t0) / done
        print(f"  {done}/{len(cands)}  {rate:.1f}s/episode  "
              f"eta~{np.median(r['eta_mean']):.3f}", flush=True)

    merged = {k: np.concatenate([r[k] for r in rows]) for k in rows[0]}
    assigned = CU.assign(merged, ids, cfg)
    for a, fam in zip(assigned, families):
        a["family"] = fam
    summary = CU.summary(assigned)
    print(json.dumps(summary, indent=2))

    OUT.mkdir(parents=True, exist_ok=True)
    path = OUT / f"{args.out}.json"
    path.write_text(json.dumps(dict(
        summary=summary, config=cfg._asdict(), human=CU.HUMAN._asdict(),
        path=[s._asdict() for s in CU.PATH],
        branches=[s._asdict() for s in CU.BRANCH_STAGES],
        episodes=assigned), indent=1))
    print("wrote", path)


if __name__ == "__main__":
    main()
