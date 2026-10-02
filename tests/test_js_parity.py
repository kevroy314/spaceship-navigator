"""The browser's ship physics (web/js/sim.js) must match JAX on the same actions."""

import json
import shutil
import subprocess
from pathlib import Path

import numpy as np
import pytest

from spacenav import episodes as EP

ROOT = Path(__file__).resolve().parents[1]
pytestmark = pytest.mark.skipif(shutil.which("node") is None or not EP.pool_names(),
                                reason="needs node and built pools")


@pytest.mark.parametrize("bodies", ["jax", "local"])
@pytest.mark.parametrize("eid", ["val_seen-3-1-f", "val_seen-450-2-r", "val_seen-900-5-t",
                                 "val_holdout-10-0-t", "val_holdout-700-3-r"])
def test_js_matches_jax(eid, bodies, tmp_path):
    """bodies="jax": JAX ephemeris shipped to JS; "local": JS integrates bodies (the game's path)."""
    pilot = EP.simulate_baseline(eid, "pilot")
    traj = np.asarray(pilot["trajectory"])
    # replay the pilot's own action sequence open-loop in both engines
    lvl, task, _ = EP._level_task(eid)
    import jax
    from spacenav import baselines, env as E
    _, trace = jax.jit(lambda l, t: E.rollout(l, t, baselines.pilot(l, t)))(lvl, task)
    actions = np.stack([np.asarray(trace["turn"]), np.asarray(trace["thrust"])], -1)[: pilot["ticks"]]
    ref = EP.simulate_actions(eid, actions)

    (tmp_path / "desc.json").write_text(json.dumps(EP.describe(eid)))
    (tmp_path / "eph.bin").write_bytes(EP.ephemeris(eid))
    (tmp_path / "act.json").write_text(json.dumps(actions.tolist()))
    eph = str(tmp_path / "eph.bin") if bodies == "jax" else "local"
    out = subprocess.run(["node", str(ROOT / "tests/js_replay.mjs"), str(tmp_path / "desc.json"),
                          eph, str(tmp_path / "act.json")],
                         capture_output=True, text=True, check=True)
    js = json.loads(out.stdout)

    a = np.asarray(js["trajectory"])[:, :2]
    b = np.asarray(ref["trajectory"])[:, :2]
    n = min(len(a), len(b), 20 * 15)     # first 20 s: float32 vs float64 drift stays tiny
    err = np.linalg.norm(a[:n] - b[:n], axis=-1).max()
    assert err < 0.05, f"max position error {err:.4f} over {n} ticks"
    assert js["status"] == ref["status"], (js["status"], ref["status"])
    for k in ("time", "fuel"):
        assert js["costs"][k] == pytest.approx(ref["costs"][k], rel=1e-3, abs=1e-3)
