"""Agricultural monsoon-vulnerability and heatwave impact indices.

Consumes the live 10-day rollout of ``tp`` (mm/day), ``t2m`` (K) and ``tcwv``
(kg/m2) from ``POST /simulate`` and turns it into decision-scale impact:

* **CRDI** - Cumulative Rainfall Deficit Index.  Forecast rainfall accumulated
  over the rollout is compared with the fitted 2005-2019 harmonic climatology
  for the same day-of-year.  A strongly negative CRDI with a long dry spell is
  a monsoon-break signature.
* **Kharif crop water stress** - Hargreaves reference evapotranspiration
  demand against cumulative rainfall, consecutive dry days and a Tmax proxy,
  with separate rice / cotton / soybean susceptibility and an area-weighted
  composite.
* **Wet-bulb heat stress** - Stull (2011) wet-bulb temperature from ``t2m`` and
  a dew point inferred from ``tcwv`` (Reitan 1963 precipitable-water relation).
  This is the heatwave danger metric: humid heat is what kills, and dry-bulb
  temperature alone misses it.

Everything is vectorised numpy over the 32 x 36 contract grid, per lead day.
No network, no calendar lookup: the phase comes from the wire's ``doy_sincos``.

Run a self-contained smoke test with::

    python -m impact.indices
"""
from __future__ import annotations

import math
from functools import lru_cache

import numpy as np

from .geo import (
    CELL_AREA_KM2,
    DISTRICTS,
    DISTRICT_INDEX,
    LAND,
    LAT,
    LAT2D,
    LON2D,
    NC,
    NLAT,
    NLON,
    ROOT,
)

STEPS_DEFAULT = 10
DRY_DAY_MM = 1.0                 # a day below this is a dry day
MODERATE_CROP = 35.0             # composite crop stress, moderate-to-severe
SEVERE_CROP = 65.0
RISK_MODERATE = 45.0
HEAT_WARN_C = 28.0               # wet-bulb caution
HEAT_DANGER_C = 31.0
HEAT_EXTREME_C = 35.0
YEAR_DAYS = 365.0

# Kharif area shares used for the composite stress (rice, cotton, soybean).
CROP_SHARE = {"rice": 0.40, "cotton": 0.30, "soybean": 0.30}


# --------------------------------------------------------------------------- #
# climatology
# --------------------------------------------------------------------------- #
@lru_cache(maxsize=1)
def _climatology():
    """Load the repository's fitted harmonic climatology, or None."""
    try:
        from climatology import HarmonicClimatology
    except Exception:
        return None
    path = ROOT / "artifacts" / "climatology_coefs.npz"
    if not path.exists():
        return None
    try:
        return HarmonicClimatology.load(path)
    except Exception:
        return None


def _doy_from_sincos(sin_component: float, cos_component: float) -> float:
    theta = math.atan2(float(sin_component), float(cos_component)) % (2.0 * math.pi)
    return theta * YEAR_DAYS / (2.0 * math.pi)


def _climatology_rain(sin_component: float, cos_component: float, steps: int):
    """Climatological daily rain [steps, NLAT, NLON] for forecast days 1..steps."""
    clim = _climatology()
    if clim is not None:
        out = clim.predict_steps(sin_component, cos_component, steps)
        rain = np.asarray(out[:, 1], dtype=np.float64)
        return np.clip(rain, 0.0, None), {
            "source": "harmonic climatology (2-frequency, OLS)",
            "training_period": [clim.train_start, clim.train_end],
            "file": "artifacts/climatology_coefs.npz",
            "rain_space": clim.rain_space,
        }

    # Fallback: smooth seasonal monsoon envelope, used only if the fitted
    # coefficients are missing from the repository.
    doy = _doy_from_sincos(sin_component, cos_component) + np.arange(1, steps + 1)
    season = 0.5 * (1.0 + np.cos(2.0 * np.pi * (doy - 215.0) / YEAR_DAYS))
    lat_shape = 0.2 + 0.8 * np.exp(-(((LAT - 19.0) / 7.0) ** 2))
    daily = 12.0 * season[:, None, None] * lat_shape[None, :, None]
    return np.clip(daily, 0.0, None), {
        "source": "analytic seasonal fallback (fitted coefficients unavailable)",
        "training_period": None,
        "file": None,
        "rain_space": "linear",
    }


# --------------------------------------------------------------------------- #
# physics
# --------------------------------------------------------------------------- #
def _extraterrestrial_radiation_mm(doy):
    """Hargreaves Ra in mm/day equivalent, shape [steps, NLAT]."""
    doy = np.atleast_1d(np.asarray(doy, dtype=np.float64))
    phi = np.radians(LAT)[None, :]
    decl = 0.409 * np.sin(2.0 * np.pi * doy / YEAR_DAYS - 1.39)[:, None]
    dr = 1.0 + 0.033 * np.cos(2.0 * np.pi * doy / YEAR_DAYS)
    ws = np.arccos(np.clip(-np.tan(phi) * np.tan(decl), -1.0, 1.0))
    ra = (
        (24.0 * 60.0 / np.pi)
        * 0.0820
        * dr[:, None]
        * (ws * np.sin(phi) * np.sin(decl) + np.cos(phi) * np.cos(decl) * np.sin(ws))
    )
    return ra * 0.408


def _reference_et(t2m_c, doy):
    """Hargreaves reference evapotranspiration, mm/day, clipped to [0.2, 14]."""
    ra = _extraterrestrial_radiation_mm(doy)[:, :, None]
    # Diurnal range is not in the contract channels; 10 K is a conservative
    # inland-monsoon assumption and only enters as sqrt(TR).
    pet = 0.0023 * ra * (t2m_c + 17.8) * math.sqrt(10.0)
    return np.clip(pet, 0.2, 14.0)


def _wet_bulb(t2m_c, tcwv):
    """Stull (2011) wet-bulb temperature in C from T and column water vapour.

    Dew point is inferred from precipitable water with Reitan (1963):
    ln(W[cm]) = 0.1102 + 0.0614 * Td[C].
    """
    w_cm = np.clip(tcwv, 0.1, None) / 10.0
    td = (np.log(w_cm) - 0.1102) / 0.0614
    e = 6.112 * np.exp(17.67 * td / (td + 243.5))
    es = 6.112 * np.exp(17.67 * t2m_c / (t2m_c + 243.5))
    rh = np.clip(100.0 * e / es, 1.0, 100.0)
    tw = (
        t2m_c * np.arctan(0.151977 * np.sqrt(rh + 8.313659))
        + np.arctan(t2m_c + rh)
        - np.arctan(rh - 1.676331)
        + 0.00391838 * rh ** 1.5 * np.arctan(0.023101 * rh)
        - 4.686035
    )
    return tw, rh


def _risk_category(risk: float) -> str:
    if risk >= 65.0:
        return "severe"
    if risk >= RISK_MODERATE:
        return "high"
    if risk >= 25.0:
        return "moderate"
    return "low"


# --------------------------------------------------------------------------- #
# per-scenario metrics
# --------------------------------------------------------------------------- #
def _metrics(rollout, init_states, doy_sincos):
    roll = np.nan_to_num(np.asarray(rollout, dtype=np.float64), nan=0.0, posinf=0.0, neginf=0.0)
    init = np.nan_to_num(np.asarray(init_states, dtype=np.float64), nan=0.0, posinf=0.0, neginf=0.0)
    if roll.ndim != 4 or roll.shape[1:] != (NC, NLAT, NLON):
        raise ValueError(f"rollout shape {roll.shape} != [steps, {NC}, {NLAT}, {NLON}]")
    if init.shape != (3, NC, NLAT, NLON):
        raise ValueError(f"init_states shape {init.shape} != [3, {NC}, {NLAT}, {NLON}]")
    if len(doy_sincos) != 2:
        raise ValueError("doy_sincos must be [sin, cos]")

    steps = roll.shape[0]
    sin_component, cos_component = float(doy_sincos[0]), float(doy_sincos[1])
    doy_last = _doy_from_sincos(sin_component, cos_component)
    doy = doy_last + np.arange(1, steps + 1, dtype=np.float64)

    t2m_c = roll[:, 0] - 273.15
    tp = np.clip(roll[:, 1], 0.0, None)
    tcwv = np.clip(roll[:, 5], 0.0, None)

    pet = _reference_et(t2m_c, doy)
    rain_cum = np.cumsum(tp, axis=0)
    pet_cum = np.cumsum(pet, axis=0)

    clim_rain, clim_meta = _climatology_rain(sin_component, cos_component, steps)
    clim_cum = np.cumsum(clim_rain, axis=0)

    # CRDI: 0 = climatological rain, -100 = no rain at all.
    crdi = 100.0 * (clim_cum - rain_cum) / np.maximum(clim_cum, 10.0)
    crdi = np.clip(crdi, -100.0, 100.0)

    # Consecutive dry days, seeded with the last observed input day.
    cdd = np.zeros((steps, NLAT, NLON), dtype=np.float64)
    streak = (init[-1, 1] < DRY_DAY_MM).astype(np.float64)
    for k in range(steps):
        streak = np.where(tp[k] >= DRY_DAY_MM, 0.0, streak + 1.0)
        cdd[k] = streak

    deficit_ratio = np.clip((pet_cum - rain_cum) / np.maximum(pet_cum, 1e-3), 0.0, 1.0)
    cdd_factor = np.clip(cdd / 12.0, 0.0, 1.0)
    tmax_proxy = t2m_c + 5.0
    heat_factor = np.clip((tmax_proxy - 34.0) / 10.0, 0.0, 1.0)

    # Monsoon moisture transport: sustained low-level inflow is the conveyor
    # that resupplies soil water between rain events.  A stall relative to the
    # 3-day analysis wind is a leading indicator of crop water stress even
    # before rainfall totals react, so it enters the crop index as a supply
    # term (a 40% slowdown gives a transport deficit of 0.8).
    wind_speed = np.hypot(roll[:, 2], roll[:, 3])
    ref_wind = np.hypot(init[:, 2], init[:, 3]).mean(axis=0)
    ratio = np.where(ref_wind > 0.3, wind_speed / np.maximum(ref_wind, 0.3), 1.0)
    transport_deficit = np.clip((1.0 - ratio) / 0.4, 0.0, 1.0)

    def crop_stress(w_deficit, w_dry, w_heat, w_flow):
        base = (w_deficit * deficit_ratio + w_dry * cdd_factor
                + w_heat * heat_factor + w_flow * transport_deficit)
        return 100.0 * np.clip(base, 0.0, 1.0)

    rice = crop_stress(0.35, 0.20, 0.10, 0.35)
    cotton = crop_stress(0.30, 0.15, 0.20, 0.35)
    soybean = crop_stress(0.33, 0.18, 0.14, 0.35)
    crop = (CROP_SHARE["rice"] * rice + CROP_SHARE["cotton"] * cotton
            + CROP_SHARE["soybean"] * soybean)

    wetbulb, rh = _wet_bulb(t2m_c, tcwv)
    heat = 100.0 * np.clip((wetbulb - 24.0) / 10.0, 0.0, 1.0)

    risk = 0.50 * crop + 0.30 * np.clip(-crdi, 0.0, 100.0) + 0.20 * heat

    return {
        "steps": steps,
        "t2m_c": t2m_c,
        "tp": tp,
        "crdi": crdi,
        "crop": crop,
        "rice": rice,
        "cotton": cotton,
        "soybean": soybean,
        "heat": heat,
        "wetbulb": wetbulb,
        "rh": rh,
        "transport": transport_deficit,
        "risk": risk,
        "cdd": cdd,
        "rain_cum": rain_cum,
        "clim_cum": clim_cum,
        "pet_cum": pet_cum,
        "climatology": clim_meta,
    }


# --------------------------------------------------------------------------- #
# aggregation
# --------------------------------------------------------------------------- #
def _wmean(arr, weights):
    """Area-weighted daily mean over a cell subset, arr [steps, cells]."""
    total = float(weights.sum())
    if total <= 0:
        return np.zeros(arr.shape[0])
    return (arr * weights[None, :]).sum(axis=1) / total


def _aggregate_districts(m):
    out = []
    steps = m["steps"]
    for k, d in enumerate(DISTRICTS):
        mask = (DISTRICT_INDEX == k) & LAND
        if not mask.any():
            continue
        w = CELL_AREA_KM2[mask]
        sel = {key: m[key][:, mask] for key in
               ("crdi", "crop", "rice", "cotton", "soybean", "heat", "wetbulb", "risk")}
        area_total = float(w.sum())
        cell_pop = w * d["pop_density"]

        risk_series = _wmean(sel["risk"], w)
        crop_series = _wmean(sel["crop"], w)
        crdi_series = _wmean(sel["crdi"], w)
        heat_series = _wmean(sel["heat"], w)
        affected = (sel["crop"] >= MODERATE_CROP)
        severe = (sel["crop"] >= SEVERE_CROP)
        affected_km2 = (affected * w[None, :]).sum(axis=1)
        severe_km2 = (severe * w[None, :]).sum(axis=1)
        exposed = (affected * cell_pop[None, :]).sum(axis=1)

        out.append({
            "index": k,
            "id": d["id"],
            "name": d["name"],
            "state": d["state"],
            "crops": d["crops"],
            "bbox": d["bbox"],
            "center": d["center"],
            "pop_density": d["pop_density"],
            "land_cells": int(mask.sum()),
            "area_km2": round(area_total, 0),
            "population": round(area_total * d["pop_density"], 0),
            "risk": round(float(risk_series[-1]), 1),
            "category": _risk_category(float(risk_series[-1])),
            "crop": round(float(crop_series[-1]), 1),
            "crdi": round(float(crdi_series[-1]), 1),
            "heat": round(float(heat_series[-1]), 1),
            "wetbulb": round(float(_wmean(sel["wetbulb"], w)[-1]), 1),
            "rice": round(float(_wmean(sel["rice"], w)[-1]), 1),
            "cotton": round(float(_wmean(sel["cotton"], w)[-1]), 1),
            "soybean": round(float(_wmean(sel["soybean"], w)[-1]), 1),
            "peak_crop": round(float(crop_series.max()), 1),
            "peak_risk": round(float(risk_series.max()), 1),
            "affected_km2": round(float(affected_km2[-1]), 0),
            "severe_km2": round(float(severe_km2[-1]), 0),
            "population_exposed": round(float(exposed[-1]), 0),
            "series": {
                "risk": [round(float(v), 1) for v in risk_series],
                "crop": [round(float(v), 1) for v in crop_series],
                "crdi": [round(float(v), 1) for v in crdi_series],
                "heat": [round(float(v), 1) for v in heat_series],
                "wetbulb": [round(float(v), 1) for v in _wmean(sel["wetbulb"], w)],
                "affected_km2": [round(float(v), 0) for v in affected_km2],
                "severe_km2": [round(float(v), 0) for v in severe_km2],
                "population_exposed": [round(float(v), 0) for v in exposed],
            },
        })
    return out


def _global_summary(m):
    steps = m["steps"]
    weights = CELL_AREA_KM2[LAND]
    sel = {key: m[key][:, LAND] for key in ("crop", "risk", "crdi", "heat", "wetbulb")}
    affected = sel["crop"] >= MODERATE_CROP
    severe = sel["crop"] >= SEVERE_CROP
    affected_km2 = (affected * weights[None, :]).sum(axis=1)
    severe_km2 = (severe * weights[None, :]).sum(axis=1)

    land_pop = CELL_AREA_KM2[LAND] * 0.0
    for k, d in enumerate(DISTRICTS):
        mask = (DISTRICT_INDEX == k) & LAND
        land_pop[mask[LAND]] = CELL_AREA_KM2[mask] * d["pop_density"]
    exposed = (affected * land_pop[None, :]).sum(axis=1)

    crop_series = _wmean(sel["crop"], weights)
    risk_series = _wmean(sel["risk"], weights)
    series = {
        "day": list(range(1, steps + 1)),
        "affected_km2": [round(float(v), 0) for v in affected_km2],
        "severe_km2": [round(float(v), 0) for v in severe_km2],
        "population_exposed": [round(float(v), 0) for v in exposed],
        "mean_crop": [round(float(v), 1) for v in crop_series],
        "mean_risk": [round(float(v), 1) for v in risk_series],
        "max_wetbulb_c": [round(float(v), 1) for v in sel["wetbulb"].max(axis=1)],
    }
    peak_day = int(np.argmax(affected_km2))
    return {
        "day": steps,
        "affected_km2": round(float(affected_km2[-1]), 0),
        "severe_km2": round(float(severe_km2[-1]), 0),
        "population_exposed": round(float(exposed[-1]), 0),
        "mean_crop": round(float(crop_series[-1]), 1),
        "mean_risk": round(float(risk_series[-1]), 1),
        "mean_crdi": round(float(_wmean(sel["crdi"], weights)[-1]), 1),
        "max_wetbulb_c": round(float(sel["wetbulb"][-1].max()), 1),
        "land_area_km2": round(float(weights.sum()), 0),
        "population_total": round(float(land_pop.sum()), 0),
        "peak": {
            "day": peak_day + 1,
            "affected_km2": round(float(affected_km2[peak_day]), 0),
            "severe_km2": round(float(severe_km2[peak_day]), 0),
            "population_exposed": round(float(exposed[peak_day]), 0),
        },
        "series": series,
    }


def _cells(m):
    steps = m["steps"]
    out = []
    for k in range(steps):
        out.append({
            "crdi": np.round(m["crdi"][k], 1).tolist(),
            "crop": np.round(m["crop"][k], 1).tolist(),
            "heat": np.round(m["heat"][k], 1).tolist(),
            "risk": np.round(m["risk"][k], 1).tolist(),
            "wetbulb": np.round(m["wetbulb"][k], 1).tolist(),
            "rain_cum": np.round(m["rain_cum"][k], 1).tolist(),
            "clim_cum": np.round(m["clim_cum"][k], 1).tolist(),
            "cdd": np.round(m["cdd"][k], 0).tolist(),
        })
    return out


# --------------------------------------------------------------------------- #
# scenario comparison
# --------------------------------------------------------------------------- #
def _pct(a, b):
    base = max(abs(a), 1e-6)
    return round(100.0 * (b - a) / base, 1)


def _comparison(prime_summary, multi_summary, prime_districts, multi_districts):
    ps, ms = prime_summary["series"], multi_summary["series"]
    steps = min(len(ps["day"]), len(ms["day"]))
    day = min(7, steps)
    k = day - 1
    p_aff, m_aff = ps["affected_km2"][k], ms["affected_km2"][k]
    p_exp, m_exp = ps["population_exposed"][k], ms["population_exposed"][k]
    p_crop, m_crop = ps["mean_crop"][k], ms["mean_crop"][k]
    p_wb, m_wb = ps["max_wetbulb_c"][k], ms["max_wetbulb_c"][k]

    by_id = {d["id"]: d for d in prime_districts}
    deltas = []
    newly = []
    for d in multi_districts:
        p = by_id.get(d["id"])
        if not p:
            continue
        p_c = p["series"]["crop"][k]
        m_c = d["series"]["crop"][k]
        p_a = p["series"]["affected_km2"][k]
        m_a = d["series"]["affected_km2"][k]
        deltas.append({
            "id": d["id"],
            "name": d["name"],
            "state": d["state"],
            "delta_crop": round(m_c - p_c, 1),
            "delta_affected_km2": round(m_a - p_a, 0),
        })
        if m_a - p_a >= 500.0:
            newly.append(d["name"])
    hotspots = [d for d in sorted(deltas, key=lambda x: x["delta_crop"], reverse=True)
                if d["delta_crop"] > 0.5][:3]

    pct = _pct(p_aff, m_aff)
    delta_aff = m_aff - p_aff
    delta_exp = m_exp - p_exp
    verb = "expands" if delta_aff >= 0 else "contracts"
    spotlight = ""
    if hotspots:
        names = [h["name"] for h in hotspots]
        if len(names) == 1:
            spotlight = f" Led by {names[0]}."
        else:
            spotlight = f" Led by " + ", ".join(names[:-1]) + f" and {names[-1]}."
    narrative = (
        f"Moderate-to-severe crop water stress {verb} by {abs(pct):.0f}% "
        f"({delta_aff:+,.0f} km²) by day {day}."
        + spotlight
    )
    if abs(delta_exp) >= 50_000:
        direction = "rises" if delta_exp > 0 else "falls"
        narrative += f" Population exposure {direction} by {abs(delta_exp) / 1e6:.1f} M."

    return {
        "day": day,
        "affected_km2": {
            "prime": round(float(p_aff), 0),
            "multi": round(float(m_aff), 0),
            "delta": round(float(delta_aff), 0),
            "delta_pct": pct,
        },
        "population_exposed": {
            "prime": round(float(p_exp), 0),
            "multi": round(float(m_exp), 0),
            "delta": round(float(delta_exp), 0),
            "delta_pct": _pct(p_exp, m_exp),
        },
        "mean_crop": {
            "prime": round(float(p_crop), 1),
            "multi": round(float(m_crop), 1),
            "delta": round(float(m_crop - p_crop), 1),
        },
        "max_wetbulb_c": {
            "prime": round(float(p_wb), 1),
            "multi": round(float(m_wb), 1),
            "delta": round(float(m_wb - p_wb), 1),
        },
        "newly_affected_districts": newly,
        "hotspots": hotspots,
        "narrative": narrative,
    }


# --------------------------------------------------------------------------- #
# public entrypoint
# --------------------------------------------------------------------------- #
def assess(rollout, init_states, doy_sincos, multi_rollout=None):
    """Assess one rollout, optionally compared with a perturbed rollout.

    Parameters
    ----------
    rollout : nested list [steps, 6, 32, 36], physical units
    init_states : nested list [3, 6, 32, 36], physical units
    doy_sincos : [sin, cos] of the last input day
    multi_rollout : optional perturbed rollout, same shape as ``rollout``
    """
    prime = _metrics(rollout, init_states, doy_sincos)
    prime_districts = _aggregate_districts(prime)
    prime_summary = _global_summary(prime)

    response = {
        "meta": {
            "grid": {"nlat": NLAT, "nlon": NLON, "lat0": 6.5, "lon0": 65.5},
            "steps": prime["steps"],
            "day_of_year_last_input": round(_doy_from_sincos(*doy_sincos), 1),
            "thresholds": {
                "dry_day_mm": DRY_DAY_MM,
                "moderate_crop": MODERATE_CROP,
                "severe_crop": SEVERE_CROP,
                "risk_moderate": RISK_MODERATE,
                "wetbulb_warn_c": HEAT_WARN_C,
                "wetbulb_danger_c": HEAT_DANGER_C,
                "wetbulb_extreme_c": HEAT_EXTREME_C,
            },
            "crop_shares": CROP_SHARE,
            "climatology": prime["climatology"],
            "methods": {
                "crdi": "forecast cumulative rain vs harmonic 2005-2019 climatology "
                        "for the same day-of-year, normalised by max(climatology, 10 mm)",
                "crop": "Hargreaves PET vs cumulative rain, consecutive dry days "
                        "(<1 mm/day), a Tmax proxy and a monsoon moisture-transport "
                        "term (wind relative to the 3-day analysis), with "
                        "rice/cotton/soybean weights",
                "heat": "Stull (2011) wet-bulb from t2m and a Reitan (1963) "
                        "TCWV-derived dew point; danger thresholds 28/31/35 C",
            },
        },
        "district_index": DISTRICT_INDEX.tolist(),
        "districts": prime_districts,
        "cells": _cells(prime),
        "summary": prime_summary,
        "cells_multi": None,
        "districts_multi": None,
        "summary_multi": None,
        "comparison": None,
    }

    if multi_rollout is not None:
        multi = _metrics(multi_rollout, init_states, doy_sincos)
        multi_districts = _aggregate_districts(multi)
        multi_summary = _global_summary(multi)
        response["cells_multi"] = _cells(multi)
        response["districts_multi"] = multi_districts
        response["summary_multi"] = multi_summary
        response["comparison"] = _comparison(
            prime_summary, multi_summary, prime_districts, multi_districts
        )
    return response


def _demo():
    """Deterministic smoke test on a synthetic monsoon rollout."""
    rng = np.random.default_rng(7)
    steps = 10
    init = np.zeros((3, NC, NLAT, NLON), dtype=np.float64)
    lat = LAT2D
    init[:, 0] = 300.0 - 0.2 * (lat - 10.0) + 3.0 * LAND
    init[:, 1] = 8.0 * np.exp(-((lat - 19.0) / 6.0) ** 2)
    init[:, 5] = 35.0 + 20.0 * np.exp(-((lat - 14.0) / 9.0) ** 2)
    rollout = np.repeat(init[-1][None], steps, axis=0)
    rollout += rng.normal(0.0, 0.2, rollout.shape)
    rollout[:, 1] = np.clip(rollout[:, 1], 0.0, None)
    rollout[:, 5] = np.clip(rollout[:, 5], 0.0, None)
    theta = 2.0 * np.pi * 232.0 / 365.0
    out = assess(rollout, init, [math.sin(theta), math.cos(theta)],
                 multi_rollout=rollout * 1.0 + 0.0)
    s = out["summary"]
    c = out["comparison"]
    print(f"steps={out['meta']['steps']} districts={len(out['districts'])}")
    print(f"climatology={out['meta']['climatology']['source']}")
    print(f"affected={s['affected_km2']:,.0f} km2  exposed={s['population_exposed']:,.0f}")
    print(f"mean CRDI={s['mean_crdi']:+.1f}  max wetbulb={s['max_wetbulb_c']} C")
    print(f"comparison: {c['narrative']}")


if __name__ == "__main__":
    _demo()
