"""Phase 5 dashboard backend: static UI + a thin bootstrap API.

Run it next to the Round 1 model server:

    python serve.py --port 8000                 # the model (repo root)
    python dashboard/app.py --port 8001         # this dashboard

The browser POSTs every rollout straight to ``http://localhost:8000/simulate``
itself (see static/app.js); this process only serves the page, the synthetic
analysis windows and the preset/shock metadata, plus a server-side probe of the
model server so the UI can say whether it is reachable. Nothing is cached or
pre-rendered: each shock drag is a fresh inference call.
"""
from __future__ import annotations

import argparse
import json
import os
import sys
import urllib.error
import urllib.request
from pathlib import Path

from fastapi import FastAPI, HTTPException, Request
from fastapi.responses import FileResponse, JSONResponse
from fastapi.staticfiles import StaticFiles
from starlette.concurrency import run_in_threadpool

try:                      # python -m uvicorn dashboard.app:app
    from . import state as state_mod
except ImportError:       # python dashboard/app.py
    import state as state_mod

HERE = Path(__file__).resolve().parent
ROOT = HERE.parent
STATIC = HERE / "static"
if str(ROOT) not in sys.path:      # the impact package lives at the repo root
    sys.path.insert(0, str(ROOT))

from impact import assess as assess_impact  # noqa: E402
API_BASE = os.environ.get("MODEL_API", "http://localhost:8000").rstrip("/")

# Shock primitives the browser turns into a [6,32,36] perturbation delta.
# mode "absolute" adds `value` (times unit_scale) in the channel's own units,
# mode "percent" scales the current field of that channel by value/100.
SHOCK_TYPES = {
    "thermal": {
        "label": "Temperature", "unit": "K", "channels": [0], "mode": "absolute",
        "min": -8.0, "max": 8.0, "step": 0.25, "default": 3.0,
    },
    "moisture": {
        "label": "Water vapour", "unit": "%", "channels": [5], "mode": "percent",
        "min": -60.0, "max": 60.0, "step": 1.0, "default": 15.0,
    },
    "wind": {
        "label": "Wind speed", "unit": "%", "channels": [2, 3], "mode": "percent",
        "min": -80.0, "max": 80.0, "step": 1.0, "default": -40.0,
    },
    "pressure": {
        "label": "Pressure", "unit": "hPa", "channels": [4], "mode": "absolute",
        "unit_scale": 100.0,
        "min": -12.0, "max": 12.0, "step": 0.25, "default": -4.0,
    },
    "rain": {
        "label": "Rainfall", "unit": "mm/day", "channels": [1], "mode": "absolute",
        "min": -10.0, "max": 20.0, "step": 0.5, "default": 8.0,
    },
}

PRESETS = [
    {
        "id": "bob_warm",
        "label": "Bay of Bengal SST Warming +3 K",
        "detail": "+3 K on t2m, +15% of tcwv",
        "center": {"lat": 13.5, "lon": 88.0},
        "radius": 6.0,
        "components": [
            {"type": "thermal", "value": 3.0},
            {"type": "moisture", "value": 15.0},
        ],
    },
    {
        "id": "arabian_stall",
        "label": "Arabian Sea Monsoon Wind Stall -40%",
        "detail": "-40% on u10 and v10",
        "center": {"lat": 14.0, "lon": 70.0},
        "radius": 5.5,
        "components": [{"type": "wind", "value": -40.0}],
    },
    {
        "id": "himalaya_cold",
        "label": "Himalayan Cold Surge -5 K",
        "detail": "-5 K on t2m",
        "center": {"lat": 30.5, "lon": 78.0},
        "radius": 5.0,
        "components": [{"type": "thermal", "value": -5.0}],
    },
    {
        "id": "null",
        "label": "Null control (zero delta)",
        "detail": "perturbation = zeros; must equal Timeline Prime",
        "center": {"lat": 20.0, "lon": 80.0},
        "radius": 6.0,
        "components": [{"type": "thermal", "value": 0.0}],
    },
]

app = FastAPI(title="multiverse dashboard", docs_url=None, redoc_url=None)


@app.get("/api/bootstrap")
def bootstrap():
    return {
        "api_base": API_BASE,
        "grid": {
            "nlat": state_mod.NLAT,
            "nlon": state_mod.NLON,
            "lat0": float(state_mod.LAT[0]),
            "lon0": float(state_mod.LON[0]),
            "dlat": 1.0,
            "dlon": 1.0,
        },
        "channels": list(state_mod.CHANNELS),
        "windows": [{"id": k, "label": v["label"]} for k, v in state_mod.WINDOWS.items()],
        "default_window": "monsoon_peak",
        "presets": PRESETS,
        "shock_types": SHOCK_TYPES,
        "impact": True,
        "stats": {
            "mean": [float(x) for x in state_mod.CHANNEL_MEAN],
            "std": [float(x) for x in state_mod.CHANNEL_STD],
        },
    }


@app.get("/api/window/{window_id}")
def window(window_id: str):
    if window_id not in state_mod.WINDOWS:
        raise HTTPException(status_code=404, detail=f"unknown window '{window_id}'")
    return state_mod.payload(window_id)


@app.post("/api/impact")
async def impact(request: Request):
    """Agricultural impact assessment for one live rollout.

    Body: {prime: [steps,6,32,36], multi: optional perturbed rollout,
           init_states: [3,6,32,36], doy_sincos: [sin, cos]}
    Returns CRDI, crop water stress and wet-bulb heat stress per cell and
    per agro-climatic district, plus the prime/multiverse comparison.
    """
    try:
        body = await request.json()
    except (ValueError, TypeError):
        raise HTTPException(status_code=400, detail="invalid JSON body")
    if not isinstance(body, dict):
        raise HTTPException(status_code=400, detail="body must be a JSON object")
    prime = body.get("prime")
    init = body.get("init_states")
    doy = body.get("doy_sincos")
    if prime is None or init is None or doy is None:
        raise HTTPException(
            status_code=400,
            detail="prime, init_states and doy_sincos are required",
        )
    try:
        return await run_in_threadpool(assess_impact, prime, init, doy, body.get("multi"))
    except (ValueError, TypeError) as exc:
        raise HTTPException(status_code=400, detail=f"invalid rollout: {exc}")


@app.get("/api/health")
def health():
    """Server-side probe of the model server (the browser could also do this)."""
    req = urllib.request.Request(
        API_BASE + "/health",
        data=b"{}",
        method="POST",
        headers={"Content-Type": "application/json"},
    )
    try:
        with urllib.request.urlopen(req, timeout=2.5) as resp:
            body = json.loads(resp.read().decode() or "{}")
        return {
            "reachable": True,
            "status": body.get("status"),
            "model_name": body.get("model_name"),
            "api_base": API_BASE,
        }
    except (urllib.error.URLError, OSError, ValueError) as exc:
        return {"reachable": False, "error": f"{type(exc).__name__}: {exc}", "api_base": API_BASE}


@app.get("/")
def index():
    return FileResponse(STATIC / "index.html")


app.mount("/static", StaticFiles(directory=str(STATIC)), name="static")


def main() -> None:
    import uvicorn

    ap = argparse.ArgumentParser(description="Phase 5 multiverse dashboard")
    ap.add_argument("--port", type=int, default=8001)
    ap.add_argument("--host", default="127.0.0.1")
    args = ap.parse_args()
    print(f"[dashboard] http://{args.host}:{args.port}  -> model API {API_BASE}", flush=True)
    uvicorn.run(app, host=args.host, port=args.port, log_level="warning")


if __name__ == "__main__":
    main()
