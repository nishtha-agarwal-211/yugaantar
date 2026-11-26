"""Offline inference server for the monsoon digital twin (Round 1 contract).

The judge runs exactly:

    python serve.py --port 8000

so everything this process needs is loaded once, at import time, from files
next to this one and never from the network:

    model_weights.pt   trained MonsoonUNet checkpoint (state_dict + arch +
                       mean/std + the 2 static maps + validation metrics)
    stats.json         per-channel mean/std (fallback if the checkpoint has none)

The server is read-only by design: nothing is written to disk (no caches, no
temporary checkpoints, no logs on disk) and no socket is opened outward.
Weights load in well under a second, leaving the 120 s grader warm-up ample
margin; a 10-step rollout costs a few tens of milliseconds.

Wire format (see API_CONTRACT.md): physical units in and out, plain nested
JSON lists. `simulate()` below is the exact serving path.
"""
from __future__ import annotations

import argparse
import base64
import json
import math
import sys
import time
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from fastapi import FastAPI, Request
from fastapi.responses import JSONResponse

# --------------------------------------------------------------------------- #
# contract constants
# --------------------------------------------------------------------------- #
HERE = Path(__file__).resolve().parent
WEIGHTS_PATH = HERE / "model_weights.pt"
STATS_PATH = HERE / "stats.json"

NC, NLAT, NLON = 6, 32, 36
HIST = 3                       # init_states days, oldest first
MAX_LEAD = 10                  # lead normalisation used in training
T2M_MIN_K, T2M_MAX_K = 233.16, 333.14   # just inside -40 C .. +60 C
THREADS = 4                    # judge container: 4 CPU cores
DEFAULT_STATIC_FILES = ("static_maps.npy", "static_maps.npz")

_T_IMPORT = time.perf_counter()

torch.set_num_threads(THREADS)
try:                            # only settable before any parallel work starts
    torch.set_num_interop_threads(1)
except RuntimeError:            # pragma: no cover - already initialised
    pass


# --------------------------------------------------------------------------- #
# model (identical to train.py, kept local so serving needs no xarray)
# --------------------------------------------------------------------------- #
class ConvBlock(nn.Module):
    def __init__(self, cin: int, cout: int, groups: int = 8):
        super().__init__()
        self.body = nn.Sequential(
            nn.Conv2d(cin, cout, 3, padding=1, padding_mode="replicate", bias=False),
            nn.GroupNorm(groups, cout),
            nn.SiLU(inplace=True),
            nn.Conv2d(cout, cout, 3, padding=1, padding_mode="replicate", bias=False),
            nn.GroupNorm(groups, cout),
            nn.SiLU(inplace=True),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.body(x)


class MonsoonUNet(nn.Module):
    """One-day forecaster: 3-day window + statics + season + lead -> next day."""

    def __init__(self, n_dyn: int = NC, n_static: int = 2, width: int = 32,
                 depth: int = 3, max_step_delta: float = 2.0):
        super().__init__()
        self.n_dyn = n_dyn
        self.width = width
        self.depth = depth
        self.max_step_delta = float(max_step_delta)
        in_ch = 3 * n_dyn + n_static + 3      # history + statics + doy sin/cos + lead
        widths = [width * (2 ** k) for k in range(depth)]

        enc = [ConvBlock(in_ch, widths[0])]
        for k in range(1, depth):
            enc.append(ConvBlock(widths[k - 1], widths[k]))
        self.enc = nn.ModuleList(enc)
        self.pool = nn.MaxPool2d(2)
        self.bottleneck = ConvBlock(widths[-1], widths[-1])

        dec = []
        for k in range(depth - 1, 0, -1):
            dec.append(ConvBlock(widths[k] + widths[k - 1], widths[k - 1]))
        self.dec = nn.ModuleList(dec)

        self.head = nn.Conv2d(widths[0], n_dyn, 3, padding=1, padding_mode="replicate")
        nn.init.zeros_(self.head.weight)
        nn.init.zeros_(self.head.bias)

    def forward(self, window: torch.Tensor, static: torch.Tensor,
                sincos: torch.Tensor, lead: torch.Tensor) -> torch.Tensor:
        b, _, _, h, w = window.shape
        x = window.reshape(b, -1, h, w)
        st = static.expand(b, -1, -1, -1).to(x)
        extra = torch.cat([sincos.to(x).view(b, 2, 1, 1).expand(b, 2, h, w),
                           lead.to(x).view(b, 1, 1, 1).expand(b, 1, h, w)], dim=1)
        x = torch.cat([x, st, extra], dim=1)

        feats = []
        for k, block in enumerate(self.enc):
            if k:
                x = self.pool(x)
            x = block(x)
            feats.append(x)
        x = self.bottleneck(feats[-1])
        for k, block in enumerate(self.dec):
            skip = feats[-2 - k]
            x = F.interpolate(x, size=skip.shape[-2:], mode="nearest")
            x = block(torch.cat([x, skip], dim=1))

        delta = self.max_step_delta * torch.tanh(self.head(x) / self.max_step_delta)
        return window[:, -1] + delta


def advance_sincos(sincos: torch.Tensor, k: int) -> torch.Tensor:
    """Rotate the day-of-year phase forward k days."""
    theta = 2.0 * math.pi / 365.0
    s0, c0 = sincos[:, 0], sincos[:, 1]
    cs, sn = math.cos(k * theta), math.sin(k * theta)
    return torch.stack([s0 * cs + c0 * sn, c0 * cs - s0 * sn], dim=1)


# --------------------------------------------------------------------------- #
# one-time load (module import time, >= the /health call)
# --------------------------------------------------------------------------- #
def _load_stats() -> tuple[np.ndarray, np.ndarray]:
    """stats.json per-channel mean/std; raises if unusable."""
    with STATS_PATH.open() as fh:
        stats = json.load(fh)
    mean = np.asarray(stats["mean"], dtype=np.float32).reshape(-1)
    std = np.asarray(stats["std"], dtype=np.float32).reshape(-1)
    if mean.shape != (NC,) or std.shape != (NC,) or not np.all(std > 0):
        raise ValueError("stats.json must hold 6 finite mean/std values with std > 0")
    if not (np.isfinite(mean).all() and np.isfinite(std).all()):
        raise ValueError("stats.json contains NaN/Inf")
    return mean, std


def _load_static(payload: dict) -> np.ndarray:
    """Static maps: bundled in the checkpoint, else a side file, else zeros."""
    raw = payload.get("static")
    if raw is not None:
        st = np.asarray(raw, dtype=np.float32)
        if st.shape != (2, NLAT, NLON):
            raise ValueError(f"checkpoint static maps have shape {st.shape}, expected (2,{NLAT},{NLON})")
        return np.ascontiguousarray(st)
    for name in DEFAULT_STATIC_FILES:
        path = HERE / name
        if not path.exists():
            continue
        st = np.load(path, allow_pickle=False)
        st = np.asarray(st["static"] if getattr(st, "files", None) else st, dtype=np.float32)
        if st.shape != (2, NLAT, NLON):
            raise ValueError(f"{name} has shape {st.shape}, expected (2,{NLAT},{NLON})")
        return np.ascontiguousarray(st)
    print("[serve] WARNING: no static maps found; serving with zero statics", file=sys.stderr)
    return np.zeros((2, NLAT, NLON), dtype=np.float32)


class _State:
    """Everything the model needs, resolved once at import."""

    def __init__(self) -> None:
        self.model: MonsoonUNet | None = None
        self.static: torch.Tensor | None = None
        self.mean: torch.Tensor | None = None
        self.std: torch.Tensor | None = None
        self.name = "fallback-persistence (model_weights.pt MISSING)"
        self.ready = False
        self.load_s = 0.0

    def load(self) -> None:
        t0 = time.perf_counter()
        if not WEIGHTS_PATH.exists():
            # Degraded but contract-valid: the container must still answer
            # /health and return finite, physically bounded forecasts.
            print(f"[serve] WARNING: {WEIGHTS_PATH.name} not found next to serve.py; "
                  "serving persistence of the perturbed last day", file=sys.stderr)
            self.ready = True          # still able to answer every call
            self.load_s = time.perf_counter() - t0
            return

        payload = torch.load(WEIGHTS_PATH, map_location="cpu", weights_only=False)
        arch = dict(payload.get("arch") or {})
        arch.setdefault("n_dyn", NC)
        arch.setdefault("n_static", 2)
        arch.setdefault("width", 32)
        arch.setdefault("depth", 3)
        arch.setdefault("max_step_delta", 2.0)
        model = MonsoonUNet(**arch)
        model.load_state_dict(payload["state_dict"], strict=True)
        model.eval()
        for p in model.parameters():
            p.requires_grad_(False)

        mean = np.asarray(payload.get("mean", []), dtype=np.float32).reshape(-1)
        std = np.asarray(payload.get("std", []), dtype=np.float32).reshape(-1)
        if mean.shape != (NC,) or std.shape != (NC,) or not np.all(std > 0):
            mean, std = _load_stats()          # fall back to stats.json

        static_np = _load_static(payload)

        self.mean = torch.from_numpy(mean).view(1, NC, 1, 1)
        self.std = torch.from_numpy(std).view(1, NC, 1, 1)
        self.static = torch.from_numpy(static_np)[None]
        self.model = model
        self.ready = True
        epoch = payload.get("epoch")
        self.name = f"MonsoonUNet-w{arch['width']}d{arch['depth']}" + (
            f"-ep{int(epoch)}" if isinstance(epoch, int) else "")
        self.load_s = time.perf_counter() - t0
        print(f"[serve] loaded {WEIGHTS_PATH.name} ({self.name}, "
              f"{sum(p.numel() for p in model.parameters())/1e6:.2f}M params) "
              f"in {self.load_s*1000:.0f} ms", flush=True)


STATE = _State()
try:
    STATE.load()
except Exception as exc:      # keep the process alive; /health reports not ready
    STATE.model = None
    STATE.ready = False
    STATE.name = f"load-failed: {type(exc).__name__}: {exc}"
    print(f"[serve] FATAL: could not load weights: {exc}", file=sys.stderr)

MODEL_NAME = STATE.name
LOAD_S = STATE.load_s
app = FastAPI(title="monsoon digital twin")


# --------------------------------------------------------------------------- #
# inference
# --------------------------------------------------------------------------- #
def _arr(x):
    """Accept nested lists (the contract) or {'b64': ..., 'shape': [...]} (tolerated)."""
    if isinstance(x, dict):
        return np.frombuffer(base64.b64decode(x["b64"]), dtype="<f4").reshape(
            x["shape"]).astype("f4")
    return np.asarray(x, dtype="f4")


def _clamp_physical(out: np.ndarray, fallback_last: np.ndarray) -> np.ndarray:
    """Validity gate: t2m in [-40, +60] C, tp >= 0, tcwv >= 0, everything finite.

    Done in float64 so the bound itself survives serialisation: 233.15 as a
    float32 rounds to 233.14999, which a strict `>= 233.15` check rejects.
    """
    out = np.array(out, dtype=np.float64, copy=True)
    out[:, 0] = np.clip(out[:, 0], T2M_MIN_K, T2M_MAX_K)
    out[:, 1] = np.clip(out[:, 1], 0.0, None)
    out[:, 5] = np.clip(out[:, 5], 0.0, None)
    bad = ~np.isfinite(out)
    if bad.any():                       # never emit NaN/Inf: use the last input day
        out[bad] = np.broadcast_to(fallback_last, out.shape)[bad]
    return out


def _fallback_rollout(init_states: np.ndarray, steps: int,
                      perturbation) -> np.ndarray:
    """No checkpoint: persistence of the (perturbed) last day."""
    last = init_states[-1].astype(np.float32)
    if perturbation is not None:
        last = last + perturbation
    return np.repeat(last[None], steps, axis=0)


@torch.no_grad()
def simulate(init_states, doy_sincos, steps: int, perturbation) -> np.ndarray:
    """init_states [3,6,32,36] physical, oldest first; perturbation [6,32,36] or None.

    The perturbation is added to init_states[-1] before the first step, exactly
    as the contract specifies. Returns [steps,6,32,36] in physical units.
    """
    init = np.asarray(init_states, dtype=np.float32)
    if init.shape != (HIST, NC, NLAT, NLON):
        raise ValueError(f"init_states shape {init.shape}, expected {(HIST, NC, NLAT, NLON)}")
    if not np.isfinite(init).all():
        raise ValueError("init_states contains NaN/Inf")
    steps = int(steps)
    if not 1 <= steps <= 64:
        raise ValueError(f"steps must be in [1, 64], got {steps}")

    delta_phys = None
    if perturbation is not None:
        delta_phys = np.asarray(perturbation, dtype=np.float32)
        if delta_phys.shape != (NC, NLAT, NLON):
            raise ValueError(f"perturbation shape {delta_phys.shape}, "
                             f"expected {(NC, NLAT, NLON)}")
        if not np.isfinite(delta_phys).all():
            raise ValueError("perturbation contains NaN/Inf")

    if STATE.model is None:
        return _clamp_physical(_fallback_rollout(init, steps, delta_phys), init[-1])

    window_z = (torch.from_numpy(init)[None] - STATE.mean) / STATE.std
    if delta_phys is not None:
        delta_z = torch.from_numpy(delta_phys)[None] / STATE.std
        window_z = torch.cat([window_z[:, :-1], window_z[:, -1:] + delta_z[:, None]], dim=1)

    sincos = torch.tensor([list(map(float, doy_sincos))], dtype=torch.float32)
    if sincos.shape != (1, 2) or not torch.isfinite(sincos).all():
        raise ValueError("doy_sincos must be two finite numbers [sin, cos]")

    outs = []
    w = window_z
    for k in range(steps):
        lead = torch.full((1,), min(k + 1, MAX_LEAD) / float(MAX_LEAD),
                          dtype=torch.float32)
        nxt = STATE.model(w, STATE.static, advance_sincos(sincos, k + 1), lead)
        outs.append(nxt)
        w = torch.cat([w[:, 1:], nxt[:, None]], dim=1)

    z = torch.stack(outs, dim=1)[0]                        # [steps,6,H,W] normalised
    phys = (z * STATE.std[0] + STATE.mean[0]).numpy()
    return _clamp_physical(phys, init[-1])


# --------------------------------------------------------------------------- #
# endpoints
# --------------------------------------------------------------------------- #
@app.post("/health")
def health():
    if not STATE.ready:
        return JSONResponse(status_code=503,
                            content={"status": "loading", "model_name": MODEL_NAME})
    return {"status": "ready", "model_name": MODEL_NAME}


@app.post("/simulate")
async def simulate_endpoint(request: Request):
    t0 = time.perf_counter()
    try:
        body = await request.json()
    except Exception:
        return JSONResponse(status_code=400, content={"error": "body must be JSON"})
    try:
        init = _arr(body["init_states"])
        pert = body.get("perturbation")
        pert = None if pert is None else _arr(pert)
        out = simulate(init, body["doy_sincos"], int(body["steps"]), pert)
    except (KeyError, TypeError, ValueError) as exc:
        return JSONResponse(status_code=400, content={"error": str(exc)})
    return {"rollout": np.asarray(out, dtype=np.float64).tolist(),
            "inference_ms": (time.perf_counter() - t0) * 1000.0}


if __name__ == "__main__":
    import uvicorn

    ap = argparse.ArgumentParser()
    ap.add_argument("--port", type=int, default=8000)
    args = ap.parse_args()
    print(f"[serve] ready on 0.0.0.0:{args.port} in "
          f"{(time.perf_counter() - _T_IMPORT)*1000:.0f} ms (model: {MODEL_NAME})",
          flush=True)
    uvicorn.run(app, host="0.0.0.0", port=args.port, log_level="warning")
