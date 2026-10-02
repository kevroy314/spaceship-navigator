"""Local web server: serves the game client, episodes, and recorded runs.

    python -m spacenav.server [--port 8765]
"""

import argparse
import json
import os
import time
from pathlib import Path

# The server's JAX work is tiny (one task sample per episode, baseline rollouts on
# request).  Keep it on the CPU so it never queues behind a training run on the GPU.
os.environ.setdefault("JAX_PLATFORMS", "cpu")

import numpy as np
from fastapi import FastAPI, HTTPException
from fastapi.middleware.gzip import GZipMiddleware
from fastapi.responses import FileResponse, Response
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel

from spacenav import episodes as EP

ROOT = Path(__file__).resolve().parents[1]
WEB = ROOT / "web"
RUNS = EP.DATA / "runs"

CURRICULA = EP.DATA / "curriculum"

app = FastAPI(title="spacenav")
app.add_middleware(GZipMiddleware, minimum_size=1000)


@app.middleware("http")
async def no_stale_client(request, call_next):
    """Never let a browser keep an old copy of the client: a cached main.js against a
    newer API is a blank screen with a null-property error, which looks like a broken
    game rather than a stale cache."""
    response = await call_next(request)
    if not request.url.path.startswith("/api/"):
        response.headers["Cache-Control"] = "no-cache, must-revalidate"
    return response


@app.on_event("startup")
def warmup():
    """Compile the JAX paths in the background so the first page load is quick."""
    import threading

    def go():
        names = EP.pool_names()
        for name in names:
            EP.pool(name)
        for mode in ("1w0m", "3w50m"):
            eid = EP.episode_id(names[0], 0, 0, mode)
            EP.describe(eid)
            EP.simulate_baseline(eid, "pilot")
    threading.Thread(target=go, daemon=True).start()


@app.get("/api/pools")
def pools():
    out = []
    for name in EP.pool_names():
        _, meta = EP.pool(name)
        fams = {}
        for i, m in enumerate(meta):
            fams.setdefault(m["family"], []).append(i)
        out.append(dict(name=name, size=len(meta),
                        families={f: dict(first=ix[0], count=len(ix)) for f, ix in fams.items()}))
    return out


@app.get("/api/curriculum")
def curriculum(name: str = "v1"):
    """A banded curriculum: the human path, plus the branches that hang off it.

    Each entry carries the measurements the band was decided from, so the client
    can say *why* a level is on a branch rather than just that it is.
    """
    path = CURRICULA / f"{name}.json"
    if not path.exists():
        avail = sorted(p.stem for p in CURRICULA.glob("*.json")) if CURRICULA.exists() else []
        raise HTTPException(404, f"no curriculum {name!r}; have {avail}")
    return json.loads(path.read_text())


@app.get("/api/curricula")
def curricula():
    return sorted(p.stem for p in CURRICULA.glob("*.json")) if CURRICULA.exists() else []


@app.get("/api/episode/{eid}")
def episode(eid: str):
    try:
        return EP.describe(eid)
    except (ValueError, KeyError, IndexError, FileNotFoundError) as e:
        raise HTTPException(404, f"bad episode id {eid}: {e}")


@app.get("/api/episode/{eid}/ephemeris")
def ephemeris(eid: str):
    return Response(EP.ephemeris(eid), media_type="application/octet-stream",
                    headers={"Cache-Control": "max-age=86400"})


@app.get("/api/episode/{eid}/baseline/{name}")
def baseline(eid: str, name: str):
    if name not in ("pilot", "coast"):
        raise HTTPException(404, "unknown baseline")
    return dict(EP.simulate_baseline(eid, name), source=f"baseline:{name}", name=name)


class Run(BaseModel):
    episode: str
    name: str = "human"
    actions: list            # [[turn, throttle], ...] per control tick
    client: dict = {}        # outcome/costs/trajectory as computed in the browser


@app.post("/api/runs")
def save_run(run: Run):
    """Store a human run and re-simulate its actions in JAX as a parity check."""
    jax_result = EP.simulate_actions(run.episode, run.actions)
    rec = dict(episode=run.episode, name=run.name, source="human", saved=time.time(),
               actions=run.actions, client=run.client, jax=jax_result)
    d = RUNS / run.episode
    d.mkdir(parents=True, exist_ok=True)
    rid = f"{int(time.time())}-{EP.run_hash(json.dumps(run.actions).encode())}"
    (d / f"{rid}.json").write_text(json.dumps(rec))
    parity = None
    ct = (run.client or {}).get("trajectory")
    if ct:
        a, b = np.asarray(ct)[:, :2], np.asarray(jax_result["trajectory"])[:, :2]
        n = min(len(a), len(b))
        parity = float(np.max(np.linalg.norm(a[:n] - b[:n], axis=-1))) if n else None
    return dict(id=rid, jax_status=jax_result["status"], parity_max_err=parity)


@app.get("/api/runs/{eid}")
def list_runs(eid: str):
    d = RUNS / eid
    out = []
    for f in sorted(d.glob("*.json")) if d.exists() else []:
        rec = json.loads(f.read_text())
        res = rec["jax"]
        out.append(dict(id=f.stem, name=rec["name"], source=rec["source"], saved=rec["saved"],
                        **{k: res[k] for k in ("status", "ticks", "costs", "objective",
                                               "trajectory", "columns")}))
    return out


@app.get("/")
def index():
    return FileResponse(WEB / "index.html")


app.mount("/", StaticFiles(directory=WEB), name="web")


def main():
    import uvicorn
    ap = argparse.ArgumentParser()
    ap.add_argument("--host", default="0.0.0.0")
    ap.add_argument("--port", type=int, default=8765)
    args = ap.parse_args()
    uvicorn.run(app, host=args.host, port=args.port)


if __name__ == "__main__":
    main()
