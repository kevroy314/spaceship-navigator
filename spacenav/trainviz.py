"""Live training dashboard.

    python -m spacenav.trainviz [--port 8766]

Reads the JSONL that scripts/train.py writes (data/rl/<run>/train.jsonl,
eval.jsonl, config.json) and streams it to the browser over server-sent events,
so the page updates itself without polling or reloading.  Deliberately imports
no JAX: the dashboard must stay responsive while the GPU is busy training.
"""

import argparse
import json
import os
import time
from pathlib import Path

from fastapi import FastAPI, HTTPException, Request
from fastapi.responses import FileResponse, StreamingResponse
from fastapi.middleware.gzip import GZipMiddleware

ROOT = Path(__file__).resolve().parents[1]
RUNS = ROOT / "data" / "rl"
WEB = ROOT / "web" / "train"
STALL_AFTER = 180.0          # s without a new line before a run counts as stalled
TICK = 2.0                   # s between stream updates

app = FastAPI(title="spacenav training")
app.add_middleware(GZipMiddleware, minimum_size=1000)


def clean(o):
    """NaN/Inf are not valid JSON: a metric with no episodes of that kind (e.g. the
    rendezvous success rate before any rendezvous finishes) arrives as NaN."""
    if isinstance(o, float):
        return o if o == o and o not in (float("inf"), float("-inf")) else None
    if isinstance(o, dict):
        return {k: clean(v) for k, v in o.items()}
    if isinstance(o, list):
        return [clean(v) for v in o]
    return o


def _read_jsonl(path: Path, start=0):
    """Lines from `start` onwards; bad trailing lines (mid-write) are skipped."""
    if not path.exists():
        return []
    out = []
    with open(path) as f:
        for i, line in enumerate(f):
            if i < start:
                continue
            line = line.strip()
            if not line:
                continue
            try:
                out.append(clean(json.loads(line)))
            except json.JSONDecodeError:
                break        # the trainer is mid-write; pick it up next tick
    return out


def run_names():
    return sorted(p.name for p in RUNS.glob("*") if (p / "train.jsonl").exists()
                  or (p / "config.json").exists())


def summarise(name):
    d = RUNS / name
    cfg = json.loads((d / "config.json").read_text()) if (d / "config.json").exists() else {}
    train = _read_jsonl(d / "train.jsonl")
    evals = _read_jsonl(d / "eval.jsonl")
    last = train[-1] if train else {}
    total_updates = cfg.get("updates", 0)
    batch = cfg.get("n_envs", 0) * cfg.get("n_steps", 0)
    done_updates = last.get("update", 0)
    age = time.time() - (d / "train.jsonl").stat().st_mtime if (d / "train.jsonl").exists() else 1e9

    # rate from the recent window, so a resumed run does not average in the gap
    window = train[-6:]
    rate = 0.0
    if len(window) > 1:
        dt = window[-1]["wall"] - window[0]["wall"]
        if dt > 0:
            rate = (window[-1]["env_steps"] - window[0]["env_steps"]) / dt
    remaining = max(total_updates - done_updates, 0) * batch
    eta = remaining / rate if rate > 0 else None
    status = ("finished" if total_updates and done_updates >= total_updates
              else "stalled" if age > STALL_AFTER else "running")

    best = {}
    for e in evals:
        for suite, m in e["metrics"].items():
            best[suite] = max(best.get(suite, 0.0), m["success"])
    return dict(
        name=name, status=status, updates=done_updates, total_updates=total_updates,
        env_steps=last.get("env_steps", 0), total_steps=total_updates * batch,
        progress=(done_updates / total_updates) if total_updates else 0.0,
        rate=rate, eta=eta, elapsed=last.get("wall", 0.0), age=age,
        latest=last, checkpoints=len(evals), best=best,
        suites=(evals[-1]["metrics"] if evals else {}),
        regressions=(evals[-1].get("regressions", {}) if evals else {}),
        config=cfg, n_train=len(train), n_eval=len(evals),
        started=(d / "config.json").stat().st_mtime if (d / "config.json").exists() else None,
    )


@app.get("/api/runs")
def api_runs():
    return clean([summarise(n) for n in run_names()])


@app.get("/api/run/{name}")
def api_run(name: str, since: int = 0, since_eval: int = 0):
    if name not in run_names():
        raise HTTPException(404, f"no run {name}")
    d = RUNS / name
    return clean(dict(summary=summarise(name), points=_read_jsonl(d / "train.jsonl", since),
                      evals=_read_jsonl(d / "eval.jsonl", since_eval)))


@app.get("/api/stream")
async def api_stream(request: Request, run: str = ""):
    """Server-sent events: run summaries every tick, plus new rows for `run`."""
    async def gen():
        import asyncio
        cursor, eval_cursor, current = 0, 0, run
        while True:
            if await request.is_disconnected():
                return
            names = run_names()
            name = current if current in names else (names[-1] if names else "")
            if name != current:
                current, cursor, eval_cursor = name, 0, 0
            payload = dict(runs=[summarise(n) for n in names], run=name, reset=(cursor == 0))
            if name:
                d = RUNS / name
                points = _read_jsonl(d / "train.jsonl", cursor)
                evals = _read_jsonl(d / "eval.jsonl", eval_cursor)
                cursor += len(points)
                eval_cursor += len(evals)
                payload.update(points=points, evals=evals)
            yield f"data: {json.dumps(clean(payload), allow_nan=False)}\n\n"
            await asyncio.sleep(TICK)
    return StreamingResponse(gen(), media_type="text/event-stream",
                             headers={"Cache-Control": "no-cache", "X-Accel-Buffering": "no"})


@app.get("/")
def index():
    return FileResponse(WEB / "index.html")


def main():
    import uvicorn
    ap = argparse.ArgumentParser()
    ap.add_argument("--host", default="0.0.0.0")
    ap.add_argument("--port", type=int, default=8767)   # 8766 is taken by the local nginx
    args = ap.parse_args()
    print(f"training dashboard on http://{args.host}:{args.port} (runs in {RUNS})")
    uvicorn.run(app, host=args.host, port=args.port)


if __name__ == "__main__":
    main()
