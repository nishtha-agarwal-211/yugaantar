"""Grid geometry, land mask, cell areas and agro-climatic districts.

The impact module shares the 1 degree / 32 x 36 contract grid used by the
model server and the dashboard.  Districts are simplified agro-climatic
boxes (not official administrative boundaries): the point is to aggregate
the 1 degree indices into regions a decision maker recognises, and to attach
a representative rural population density to each region so exposure can be
estimated without shipping a census raster.

Every quantity here is static and fits in a few kilobytes of source.
"""
from __future__ import annotations

import json
from pathlib import Path

import numpy as np

NC, NLAT, NLON = 6, 32, 36
LAT = 6.5 + np.arange(NLAT, dtype=np.float64)
LON = 65.5 + np.arange(NLON, dtype=np.float64)
LAT2D, LON2D = np.meshgrid(LAT, LON, indexing="ij")

ROOT = Path(__file__).resolve().parent.parent
GEO_PATH = ROOT / "dashboard" / "static" / "geo" / "india.json"

# 1 degree cell: 110.57 km meridional, 111.32*cos(lat) km zonal.
DEG_LAT_KM = 110.57
DEG_LON_KM = 111.32
CELL_AREA_KM2 = (DEG_LAT_KM * DEG_LON_KM * np.cos(np.radians(LAT)))[:, None] * np.ones(
    (1, NLON), dtype=np.float64
)

# Simplified agro-climatic districts.  bbox = (lat0, lat1, lon0, lon1).
# density = representative rural population density, persons / km2.
DISTRICT_DEFS = (
    dict(id="punjab_haryana", name="Punjab & Haryana", state="Punjab / Haryana",
         bbox=(29.0, 32.6, 73.8, 77.8), density=520, crops="rice, wheat"),
    dict(id="west_up", name="West Uttar Pradesh", state="Uttar Pradesh",
         bbox=(26.0, 29.4, 77.4, 81.2), density=800, crops="rice, sugarcane"),
    dict(id="bihar_jharkhand", name="Bihar & Jharkhand", state="Bihar / Jharkhand",
         bbox=(24.4, 27.2, 82.8, 88.2), density=880, crops="rice, maize"),
    dict(id="rajasthan", name="Rajasthan & Marwar", state="Rajasthan",
         bbox=(23.6, 29.2, 69.4, 74.8), density=190, crops="cotton, millets"),
    dict(id="kutch", name="Kutch & North Gujarat", state="Gujarat",
         bbox=(22.6, 24.8, 67.8, 72.4), density=120, crops="cotton, groundnut"),
    dict(id="saurashtra", name="Saurashtra", state="Gujarat",
         bbox=(20.0, 22.6, 69.2, 72.4), density=310, crops="cotton, groundnut"),
    dict(id="east_gujarat", name="East Gujarat & Malwa", state="Gujarat / MP",
         bbox=(21.6, 24.6, 72.2, 76.2), density=340, crops="soybean, cotton"),
    dict(id="mp_bundelkhand", name="Madhya Pradesh & Bundelkhand", state="Madhya Pradesh",
         bbox=(22.4, 26.2, 74.2, 80.6), density=240, crops="soybean, rice"),
    dict(id="vidarbha", name="Vidarbha", state="Maharashtra",
         bbox=(18.8, 22.2, 75.8, 80.8), density=250, crops="cotton, soybean"),
    dict(id="marathwada", name="Marathwada", state="Maharashtra",
         bbox=(17.2, 20.2, 74.2, 78.2), density=300, crops="cotton, soybean"),
    dict(id="telangana", name="Telangana", state="Telangana",
         bbox=(15.8, 19.6, 77.0, 81.6), density=320, crops="cotton, rice"),
    dict(id="coastal_ap", name="Coastal Andhra", state="Andhra Pradesh",
         bbox=(14.8, 18.6, 79.8, 83.2), density=480, crops="rice"),
    dict(id="rayalaseema", name="Rayalaseema & North Karnataka", state="AP / Karnataka",
         bbox=(13.2, 16.4, 75.8, 80.2), density=260, crops="cotton, millets"),
    dict(id="karnataka", name="Interior Karnataka & Cauvery", state="Karnataka",
         bbox=(11.2, 15.2, 75.0, 79.0), density=390, crops="rice, cotton"),
    dict(id="odisha_chhattisgarh", name="Odisha & Chhattisgarh", state="Odisha / Chhattisgarh",
         bbox=(18.8, 23.2, 81.0, 85.8), density=270, crops="rice"),
    dict(id="gangetic_wb", name="Gangetic West Bengal", state="West Bengal",
         bbox=(21.8, 26.6, 86.0, 89.4), density=900, crops="rice, jute"),
    dict(id="tamil_nadu", name="Tamil Nadu", state="Tamil Nadu",
         bbox=(8.2, 12.2, 76.8, 80.4), density=540, crops="rice, cotton"),
    dict(id="kerala", name="Kerala & Coastal Karnataka", state="Kerala / Karnataka",
         bbox=(8.2, 12.6, 74.4, 77.6), density=860, crops="rice, spices"),
    dict(id="assam", name="Assam & Brahmaputra", state="Assam",
         bbox=(24.8, 28.2, 89.2, 96.0), density=440, crops="rice, tea"),
)


def _ring_mask(lon, lat, ring):
    """Vectorised ray casting: True where (lon, lat) lies inside one ring."""
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


def _fallback_land_mask():
    """Coarse land approximation used only if the bundled coastline is absent."""
    mask = np.ones((NLAT, NLON), dtype=bool)
    mask &= ~((LAT2D < 22.0) & (LON2D < 73.0))       # Arabian Sea
    mask &= ~((LAT2D < 20.0) & (LON2D > 80.5))       # Bay of Bengal
    mask &= LAT2D > 7.0                              # Indian Ocean
    return mask


def _land_mask():
    try:
        doc = json.loads(GEO_PATH.read_text())
    except (OSError, ValueError):
        return _fallback_land_mask()
    rings = []
    for feature in doc.get("features", []):
        geom = feature.get("geometry") or {}
        polys = geom.get("coordinates") or []
        if geom.get("type") == "Polygon":
            polys = [polys]
        for poly in polys:
            if poly:
                rings.append(np.asarray(poly[0], dtype=np.float64))
    if not rings:
        return _fallback_land_mask()
    lon2d = np.broadcast_to(LON[None, :], (NLAT, NLON))
    lat2d = np.broadcast_to(LAT[:, None], (NLAT, NLON))
    mask = np.zeros((NLAT, NLON), dtype=bool)
    for ring in rings:
        mask |= _ring_mask(lon2d, lat2d, ring)
    return mask


LAND = _land_mask()


def _build_district_index():
    """Assign each land cell to the nearest containing district box.

    Overlapping boxes are resolved by distance to the box centre; land cells
    outside every box fall back to the nearest box centre, so the whole land
    mask is covered exactly once.
    """
    idx = np.full((NLAT, NLON), -1, dtype=np.int16)
    best = np.full((NLAT, NLON), np.inf, dtype=np.float64)
    for k, d in enumerate(DISTRICT_DEFS):
        lat0, lat1, lon0, lon1 = d["bbox"]
        cy, cx = (lat0 + lat1) / 2.0, (lon0 + lon1) / 2.0
        dy = LAT2D - cy
        dx = (LON2D - cx) * np.cos(np.radians(LAT2D))
        dist = dy * dy + dx * dx
        inside = (LAT2D >= lat0) & (LAT2D <= lat1) & (LON2D >= lon0) & (LON2D <= lon1) & LAND
        take = inside & (dist < best)
        idx[take] = k
        best[take] = dist[take]

    todo = LAND & (idx < 0)
    if todo.any():
        centres = []
        for d in DISTRICT_DEFS:
            cy = (d["bbox"][0] + d["bbox"][1]) / 2.0
            cx = (d["bbox"][2] + d["bbox"][3]) / 2.0
            dy = LAT2D - cy
            dx = (LON2D - cx) * np.cos(np.radians(LAT2D))
            centres.append(dy * dy + dx * dx)
        centres = np.stack(centres, axis=0)
        nearest = np.argmin(centres, axis=0).astype(np.int16)
        idx[todo] = nearest[todo]
    return idx


DISTRICT_INDEX = _build_district_index()

DISTRICTS = [
    {
        "id": d["id"],
        "name": d["name"],
        "state": d["state"],
        "crops": d["crops"],
        "bbox": [float(v) for v in d["bbox"]],
        "center": [(d["bbox"][0] + d["bbox"][1]) / 2.0, (d["bbox"][2] + d["bbox"][3]) / 2.0],
        "pop_density": float(d["density"]),
    }
    for d in DISTRICT_DEFS
]
