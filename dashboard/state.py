"""Synthetic 3-day analysis windows for the Phase 5 dashboard.

The Round 1 benchmark keeps its windows hidden; a desk review has none either,
so the dashboard builds three reproducible analysis windows on the contract
grid (32 x 36, lat 6.5..37.5, lon 65.5..100.5). They are shaped by the bundled
coastline (static/geo/india.json), the per-channel statistics in stats.json and
a handful of monsoon features (rain band, Western Ghats, monsoon trough).

These are plausible synthetic analyses, not ERA5 reanalyses, and the UI says so.
Each window is deterministic: the same id always yields the same 3-day state.
"""
from __future__ import annotations

import json
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
ROOT = HERE.parent
GEO_PATH = HERE / "static" / "geo" / "india.json"
STATS_PATH = ROOT / "stats.json"

NC, NLAT, NLON, HIST = 6, 32, 36, 3
LAT = 6.5 + np.arange(NLAT, dtype=np.float64)
LON = 65.5 + np.arange(NLON, dtype=np.float64)
LON2D, LAT2D = np.meshgrid(LON, LAT)

CHANNELS = ("t2m", "tp", "u10", "v10", "msl", "tcwv")

# name -> (label, day-of-year for the last input day)
WINDOWS = {
    "monsoon_peak": {
        "label": "Monsoon peak (late Aug)",
        "doy": 232,
        "rain": 1.00,
        "trough_hpa": 5.4,
        "moist": 1.00,
        "temp_bias": 0.0,
        "flow": 1.00,
    },
    "break_phase": {
        "label": "Break phase (weak rain)",
        "doy": 245,
        "rain": 0.30,
        "trough_hpa": 2.1,
        "moist": 0.74,
        "temp_bias": 1.4,
        "flow": 0.72,
    },
    "pre_monsoon": {
        "label": "Pre-monsoon heat (mid May)",
        "doy": 138,
        "rain": 0.22,
        "trough_hpa": 1.2,
        "moist": 0.62,
        "temp_bias": 2.8,
        "flow": 0.55,
    },
}


def _ring_mask(lon, lat, ring):
    """Vectorised ray casting: True where (lon, lat) is inside one polygon ring."""
    inside = np.zeros(lon.shape, dtype=bool)
    n = len(ring)
    j = n - 1
    for i in range(n):
        xi, yi = ring[i]
        xj, yj = ring[j]
        crosses = (yi > lat) != (yj > lat)
        with np.errstate(divide="ignore", invalid="ignore"):
            x_cross = (xj - xi) * (lat - yi) / ((yj - yi) if yj != yi else 1e-12) + xi
        inside ^= crosses & (lon < x_cross)
        j = i
    return inside


def _land_mask():
    """1.0 over land, 0.0 over sea, from the simplified coastline in static/geo."""
    try:
        doc = json.loads(GEO_PATH.read_text())
    except (OSError, ValueError):
        return np.zeros((NLAT, NLON), dtype=np.float64)
    rings = []
    for feature in doc.get("features", []):
        geom = feature.get("geometry") or {}
        polys = geom.get("coordinates") or []
        if geom.get("type") == "Polygon":
            polys = [polys]
        for poly in polys:
            if poly:
                rings.append(np.asarray(poly[0], dtype=np.float64))
    lon2d = np.broadcast_to(LON[None, :], (NLAT, NLON))
    lat2d = np.broadcast_to(LAT[:, None], (NLAT, NLON))
    mask = np.zeros((NLAT, NLON), dtype=bool)
    for ring in rings:
        mask |= _ring_mask(lon2d, lat2d, ring)
    return mask.astype(np.float64)


LAND = _land_mask()


def _stats():
    try:
        raw = json.loads(STATS_PATH.read_text())
        return (np.asarray(raw["mean"], dtype=np.float64),
                np.asarray(raw["std"], dtype=np.float64))
    except (OSError, KeyError, ValueError):
        return (np.array([292.4, 3.6, 0.9, 0.2, 101122.0, 30.0]),
                np.array([13.5, 7.6, 2.9, 2.4, 619.0, 20.5]))


CHANNEL_MEAN, CHANNEL_STD = _stats()


def build_window(name: str):
    """Return (states[3,6,32,36] float32, [sin, cos] of the last input day)."""
    cfg = WINDOWS[name]
    ocean = 1.0 - LAND
    states = np.zeros((HIST, NC, NLAT, NLON), dtype=np.float64)

    for d in range(HIST):
        phase = 0.85 * d

        plateau = -7.5 * np.clip((LAT2D - 31.0) / 4.5, 0.0, 1.0)   # Himalayan cooling
        t2m = (302.4 - 0.25 * (LAT2D - 10.0) + cfg["temp_bias"] + 1.5 * LAND
               + 3.2 * np.exp(-(((LAT2D - 28.5) ** 2 + (LON2D - 71.0) ** 2) / 26.0))
               + plateau
               + 0.9 * np.sin(0.55 * LON2D / 6.0 + 0.8 * LAT2D / 7.0 + phase))

        band = (13.0 * np.exp(-((LAT2D - 19.0) / 6.5) ** 2)
                * (0.55 + 0.45 * np.cos((LON2D - 80.0) / 7.0)))
        ghats = 10.0 * np.exp(-(((LAT2D - 16.0) ** 2 + (LON2D - 74.2) ** 2) / 6.5))
        northeast = 5.5 * np.exp(-(((LAT2D - 25.0) ** 2 + (LON2D - 92.0) ** 2) / 9.0))
        tp = (cfg["rain"] * (band + ghats + northeast)
              * (0.85 + 0.15 * np.sin(phase + LAT2D / 5.0)))

        u10 = (cfg["flow"] * (7.2 * np.exp(-((LAT2D - 13.0) / 8.5) ** 2)
                              - 1.6 * LAND * np.exp(-((LAT2D - 26.0) / 8.0) ** 2))
               + 0.9 * np.sin(LON2D / 5.0 + phase))
        v10 = (cfg["flow"] * 2.6 * np.exp(-((LON2D - 85.0) / 7.0) ** 2)
               * np.sin((LAT2D - 12.0) / 4.5)
               + 0.5 * np.cos(LAT2D / 6.0 + phase))

        trough = cfg["trough_hpa"] * np.exp(-(((LAT2D - 25.5) ** 2 + (LON2D - 79.0) ** 2) / 55.0))
        msl_hpa = 1009.4 - trough + 2.6 * np.cos(LAT2D / 6.0) - 1.2 * LAND

        tcwv = (cfg["moist"] * (24.0 + 28.0 * np.exp(-((LAT2D - 14.0) / 9.0) ** 2) + 6.0 * ocean)
                + 1.5 * np.sin(phase + LON2D / 8.0))

        states[d, 0] = t2m
        states[d, 1] = np.clip(tp, 0.0, None)
        states[d, 2] = u10
        states[d, 3] = v10
        states[d, 4] = msl_hpa * 100.0
        states[d, 5] = np.clip(tcwv, 0.0, None)

    # Keep every channel inside the physical ranges the contract validates.
    states[:, 0] = np.clip(states[:, 0], 233.16, 333.14)
    states[:, 1] = np.clip(states[:, 1], 0.0, None)
    states[:, 4] = np.clip(states[:, 4], 87000.0, 108500.0)
    states[:, 5] = np.clip(states[:, 5], 0.0, None)

    doy = float(cfg["doy"])
    theta = 2.0 * np.pi * doy / 365.0
    return np.round(states, 3).astype(np.float32), [float(np.sin(theta)), float(np.cos(theta))]


def payload(name: str) -> dict:
    """JSON-ready window description for the browser."""
    cfg = WINDOWS[name]
    states, sincos = build_window(name)
    return {
        "id": name,
        "label": cfg["label"],
        "doy": cfg["doy"],
        "doy_sincos": sincos,
        "shape": list(states.shape),
        "channels": list(CHANNELS),
        "init": states.tolist(),
    }
