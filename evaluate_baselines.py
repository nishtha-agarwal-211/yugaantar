#!/usr/bin/env python3
"""Phase 1: offline reference baselines + internal grading harness (Round 1 contract).

Two references are built and scored exactly the way ``monsoon_r1.scoring`` scores a
submission, so that any model trained later can be compared against the real bar:

  persistence          repeat init_states[-1] for all 10 forecast steps.
  harmonic_climatology annual harmonics (mean + K sin/cos pairs) per grid cell and
                       channel, fitted on 2005-2019 only, evaluated for the
                       day-of-year of every forecast day with phase = 2*pi*doy/365
                       (the convention in ``monsoon_r1.contract.doy_sincos``).

Metrics are imported, never re-implemented, from the kit:
  * latitude-weighted RMSE        ``monsoon_r1.scoring.rmse_lat``  (cos(lat) weights)
  * heavy-rain CSI at 10 mm/day   ``monsoon_r1.scoring.csi``
  * day-8 spatial variance ratio  ``monsoon_r1.scoring.variance_ratio`` /
                                  ``anti_blur_points``
  * physical validity gate        ``monsoon_r1.scoring.validity``

Split: train 2005-01-01..2019-12-31, validation 2020-01-01..2021-12-31. Forecast
windows are 3 observed input days + 10 target days, with every target day inside
the validation period, sampled every ``--stride`` days of June, July/August,
October and January of both validation years (the four regimes the judges probe:
monsoon onset, peak monsoon, post-monsoon cyclone, winter cold wave).

Usage:
    python evaluate_baselines.py                 # writes baseline_scores.json
    python evaluate_baselines.py --harmonics 3   # different climatology order
    python evaluate_baselines.py --no-rain-floor # keep raw (possibly negative) tp

Results are deterministic apart from the ``generated_at_utc`` timestamp.
"""
from __future__ import annotations

import argparse
import calendar
import hashlib
import json
import math
import time
import warnings
from dataclasses import dataclass
from datetime import date, timedelta
from pathlib import Path

import numpy as np
import xarray as xr

try:
    from monsoon_r1.contract import CHANNELS, LAT, LON, NC, NLAT, NLON, LEAD_DAYS, SCORE_STEPS
    from monsoon_r1.scoring import (
        BLUR_MAX_RATIO,
        BLUR_MIN_RATIO,
        RAIN_THRESH,
        STD_DEFAULT,
        anti_blur_points,
        csi,
        rmse_lat,
        validity,
        variance_ratio,
    )
except ImportError as exc:  # pragma: no cover - environment guard
    raise SystemExit(
        "monsoon_r1 is not importable. Install the team kit into this environment:\n"
        "    pip install monsoon_r1_kit-1.0.0-py3-none-any.whl\n"
        f"(original error: {exc})"
    )

DATA_FILE = "india_climate_train.nc"
STATS_FILE = "stats.json"
OUT_FILE = "baseline_scores.json"

TRAIN_START, TRAIN_END = date(2005, 1, 1), date(2019, 12, 31)
VAL_START, VAL_END = date(2020, 1, 1), date(2021, 12, 31)

UNITS = ("K", "mm/day", "m/s", "m/s", "Pa", "kg/m2")
UNITS_SHORT = ("K", "mm/d", "m/s", "m/s", "Pa", "kg/m2")

# (key, label, months) — the four validation regimes, in chronological order.
SEASONS: tuple[tuple[str, str, tuple[int, ...]], ...] = (
    ("monsoon_onset_june", "Monsoon onset (June)", (6,)),
    ("peak_monsoon_jul_aug", "Peak monsoon (July-August)", (7, 8)),
    ("post_monsoon_cyclone_oct", "Post-monsoon cyclone (October)", (10,)),
    ("winter_cold_wave_jan", "Winter cold wave (January)", (1,)),
)

DAY8 = 7  # zero-based index of forecast day 8 in a 10-step rollout

WIRE_NOTE = (
    "Predictions are cast to float32, the dtype of the simulation wire format; "
    "all metrics are then the kit's own numpy implementations."
)


# --------------------------------------------------------------------------- #
# small helpers
# --------------------------------------------------------------------------- #
def _nanmean(a, axis=0):
    """NaN-aware mean that stays silent on all-NaN slices."""
    a = np.asarray(a, dtype=np.float64)
    fin = np.isfinite(a)
    cnt = fin.sum(axis=axis)
    tot = np.where(fin, a, 0.0).sum(axis=axis)
    return np.where(cnt > 0, tot / np.maximum(cnt, 1), np.nan)


def _sha256(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as fh:
        for chunk in iter(lambda: fh.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def _jsonable(obj):
    """Recursively convert numpy/types to strict-JSON values (NaN/Inf -> null)."""
    if isinstance(obj, dict):
        return {str(k): _jsonable(v) for k, v in obj.items()}
    if isinstance(obj, (list, tuple)):
        return [_jsonable(v) for v in obj]
    if isinstance(obj, np.ndarray):
        return _jsonable(obj.tolist())
    if isinstance(obj, (bool, np.bool_)):
        return bool(obj)
    if isinstance(obj, (np.integer,)):
        return int(obj)
    if isinstance(obj, (float, np.floating)):
        v = float(obj)
        return v if math.isfinite(v) else None
    if isinstance(obj, (date,)):
        return obj.isoformat()
    return obj


def _num(x, width=9, sig=4):
    if x is None or not np.isfinite(x):
        return "nan".rjust(width)
    return f"{float(x):>{width}.{sig}g}"


def _pct(x, width=8):
    """Format a value that is already a percentage."""
    if x is None or not np.isfinite(x):
        return "nan".rjust(width)
    return f"{float(x):>+{width - 1}.2f}%"


# --------------------------------------------------------------------------- #
# data + split
# --------------------------------------------------------------------------- #
@dataclass
class Dataset:
    data: np.ndarray            # [days, 6, 32, 36] float32, physical units
    days: list[date]            # calendar date per time index
    index: dict[date, int]
    sha256: str
    path: str

    def __len__(self) -> int:
        return len(self.days)


def load_dataset(path: str | Path) -> Dataset:
    path = Path(path)
    if not path.exists():
        raise SystemExit(f"dataset not found: {path}")
    path = Path(path)
    if not path.exists() and (Path(__file__).parent / ".venv" / path.name).exists():
        path = Path(__file__).parent / ".venv" / path.name
    with xr.open_dataset(path) as ds:
        if tuple(ds["channel"].values) != tuple(CHANNELS):
            raise SystemExit(f"channel order {tuple(ds['channel'].values)} != contract {CHANNELS}")
        if tuple(ds["data"].dims) != ("time", "channel", "lat", "lon"):
            raise SystemExit(f"unexpected data dims {ds['data'].dims}")
        if not np.allclose(ds["lat"].values, LAT, atol=1e-6) or not np.allclose(ds["lon"].values, LON, atol=1e-6):
            raise SystemExit("lat/lon coordinates do not match the Round 1 contract grid")
        data = np.asarray(ds["data"].values, dtype=np.float32)
        raw_times = ds["time"].values.astype("datetime64[D]")

    days = [d for d in raw_times.tolist()]
    if len(set(days)) != len(days):
        raise SystemExit("duplicate timestamps in the dataset")
    gaps = np.diff(raw_times).astype("timedelta64[D]").astype(int)
    if len(gaps) and not np.all(gaps == 1):
        raise SystemExit("dataset is not gap-free daily data")
    if not np.isfinite(data).all():
        raise SystemExit("dataset contains NaN/Inf")
    if data.shape[1:] != (NC, NLAT, NLON):
        raise SystemExit(f"unexpected data shape {data.shape}")
    return Dataset(data=data, days=days, index={d: i for i, d in enumerate(days)},
                   sha256=_sha256(path), path=str(path))


def load_std(path: str | Path = STATS_FILE) -> tuple[np.ndarray, str]:
    """Per-channel scale for reporting error in sigma units; falls back to the kit default."""
    path = Path(path)
    if path.exists():
        with path.open() as fh:
            stats = json.load(fh)
        std = np.asarray(stats["std"], dtype=np.float64)
        if std.shape == (NC,) and np.all(std > 0):
            return std, f"{path.name} (shipped normalisation statistics)"
    return np.asarray(STD_DEFAULT, dtype=np.float64), "monsoon_r1.scoring.STD_DEFAULT"


# --------------------------------------------------------------------------- #
# consistency checks between consecutive days (sanity, not used for scoring)
# --------------------------------------------------------------------------- #
def continuity_stats(data: np.ndarray) -> dict:
    """Mean 1-day abs increment per channel — how wrong 'repeat yesterday' is on average."""
    diffs = np.abs(np.diff(data, axis=0))
    return {"mean_abs_day_to_day_change": {ch: float(v) for ch, v in zip(CHANNELS, diffs.mean(axis=(0, 2, 3)))}}


# --------------------------------------------------------------------------- #
# climatology
# --------------------------------------------------------------------------- #
def harmonic_design(doy: np.ndarray, k: int) -> np.ndarray:
    """[1, cos(2*pi*h*doy/365), sin(...)] columns for h = 1..k."""
    angle = 2.0 * np.pi * np.asarray(doy, dtype=np.float64) / 365.0
    cols = [np.ones_like(angle)]
    for h in range(1, k + 1):
        cols.append(np.cos(h * angle))
        cols.append(np.sin(h * angle))
    return np.stack(cols, axis=1)


@dataclass
class HarmonicClimatology:
    k: int
    coef: np.ndarray                 # [1 + 2k, 6, 32, 36] float64
    rain_floor: float | None
    train_rmse: np.ndarray           # [6] unweighted, physical units
    train_r2: np.ndarray             # [6]
    n_train_days: int

    def predict(self, dates: list[date]) -> tuple[np.ndarray, dict]:
        doy = np.array([d.timetuple().tm_yday for d in dates], dtype=np.float64)
        A = harmonic_design(doy, self.k)                            # [steps, p]
        out = np.einsum("sp,pcxy->scxy", A, self.coef).astype(np.float32)
        cells, delta = 0, 0.0
        if self.rain_floor is not None:
            tp = out[:, 1]
            neg = tp < self.rain_floor
            cells = int(neg.sum())
            if cells:
                delta = float((self.rain_floor - tp[neg]).max())
            np.maximum(tp, self.rain_floor, out=tp)
        return out, {"tp_floor_cells": cells, "tp_floor_max_delta_mm": delta}


def fit_harmonic_climatology(data: np.ndarray, days: list[date], train_idx: np.ndarray,
                             k: int, rain_floor: float | None) -> HarmonicClimatology:
    """Least-squares annual harmonics per grid cell and channel, train years only."""
    doy = np.array([days[i].timetuple().tm_yday for i in train_idx], dtype=np.float64)
    A = harmonic_design(doy, k)                                     # [n, p]
    n, p = A.shape
    pinv = np.linalg.pinv(A)                                        # [p, n]
    coef = np.empty((p, NC, NLAT, NLON), dtype=np.float64)
    train_rmse = np.empty(NC)
    train_r2 = np.empty(NC)
    for c in range(NC):
        Y = data[train_idx, c].reshape(n, -1).astype(np.float64)     # [n, cells]
        B = pinv @ Y                                                # [p, cells]
        coef[:, c] = B.reshape(p, NLAT, NLON)
        resid = Y - A @ B
        ss_res = float((resid ** 2).sum())
        ss_tot = float(((Y - Y.mean(axis=0)) ** 2).sum())
        train_rmse[c] = math.sqrt(ss_res / Y.size)
        train_r2[c] = 1.0 - ss_res / ss_tot if ss_tot > 0 else float("nan")
    return HarmonicClimatology(k=k, coef=coef, rain_floor=rain_floor,
                               train_rmse=train_rmse, train_r2=train_r2, n_train_days=n)


# --------------------------------------------------------------------------- #
# baselines
# --------------------------------------------------------------------------- #
def persistence_forecast(init_states: np.ndarray, dates: list[date]) -> tuple[np.ndarray, dict]:
    """Repeat the last input day for every lead (the starter serve.py placeholder)."""
    return np.repeat(init_states[-1:], len(dates), axis=0).astype(np.float32), {}


def climatology_forecast(fit: HarmonicClimatology):
    def fn(init_states: np.ndarray, dates: list[date]) -> tuple[np.ndarray, dict]:
        return fit.predict(dates)
    return fn


# --------------------------------------------------------------------------- #
# windows
# --------------------------------------------------------------------------- #
@dataclass(frozen=True)
class Window:
    season: str
    init_last: date
    i0: int          # data index of init_states[0] (two days before init_last)
    steps: int


def build_windows(ds: Dataset, seasons=SEASONS, stride: int = 4, steps: int = SCORE_STEPS,
                  val_start: date = VAL_START, val_end: date = VAL_END) -> list[Window]:
    years = range(val_start.year, val_end.year + 1)
    out: list[Window] = []
    for key, _label, months in seasons:
        for year in years:
            for month in months:
                for day in range(1, calendar.monthrange(year, month)[1] + 1, stride):
                    last = date(year, month, day)
                    first = last - timedelta(days=2)
                    end = last + timedelta(days=steps)
                    # Only the scored target days must be inside the validation period.
                    # The 2-day history may reach into late December of the train period:
                    # those are observed inputs, not fitted values, so no leakage occurs.
                    if last < val_start or end > val_end:
                        continue
                    if first not in ds.index or end not in ds.index:
                        continue
                    out.append(Window(season=key, init_last=last, i0=ds.index[first], steps=steps))
    return out


# --------------------------------------------------------------------------- #
# evaluation
# --------------------------------------------------------------------------- #
@dataclass
class WindowResult:
    window: Window
    violations: tuple[str, ...]
    rmse: dict[int, np.ndarray]              # lead -> [6]
    csi: dict[int, float]                    # lead -> CSI
    csi_counts: dict[int, tuple[int, int, int]]   # lead -> (hits, misses, false alarms)
    observed_heavy_frac: dict[int, float]    # lead -> share of true cells above 10 mm/day
    var_ratio: np.ndarray                    # [6] at day 8
    min_tp: float
    tp_floor_cells: int
    tp_floor_max_delta: float


def evaluate_windows(forecast, windows: list[Window], ds: Dataset, leads: tuple[int, ...]) -> list[WindowResult]:
    results: list[WindowResult] = []
    for w in windows:
        steps = w.steps
        init = ds.data[w.i0:w.i0 + 3]
        dates = [w.init_last + timedelta(days=k) for k in range(1, steps + 1)]
        pred, meta = forecast(init, dates)
        pred = np.asarray(pred, dtype=np.float32)
        truth = ds.data[w.i0 + 3:w.i0 + 3 + steps]
        if pred.shape != (steps, NC, NLAT, NLON) or truth.shape != (steps, NC, NLAT, NLON):
            raise RuntimeError(f"bad rollout shape {pred.shape} for window {w.init_last}")
        # off-by-one guard: rollout[k] must be the real field of init_last + (k + 1) days
        if ds.days[w.i0 + 2] != w.init_last or ds.days[w.i0 + 3] != dates[0] or ds.days[w.i0 + 2 + steps] != dates[-1]:
            raise RuntimeError(f"date/index misalignment for window {w.init_last}")
        if not np.isfinite(pred).all():
            # keep going: validity() reports it, metrics become NaN just like the judge
            warnings.warn(f"non-finite rollout for window {w.init_last} ({w.season})", stacklevel=2)

        # LEAD_DAYS are 1-based forecast-day numbers: lead k is rollout index k - 1.
        rmse = {k: np.asarray(rmse_lat(pred[k - 1:k], truth[k - 1:k]), dtype=np.float64) for k in leads}
        cs, cc, hf = {}, {}, {}
        for k in leads:
            p_tp, t_tp = pred[k - 1, 1], truth[k - 1, 1]
            cs[k] = csi(p_tp, t_tp)
            pp, tt = p_tp > RAIN_THRESH, t_tp > RAIN_THRESH
            cc[k] = (int((pp & tt).sum()), int((~pp & tt).sum()), int((pp & ~tt).sum()))
            hf[k] = float(tt.mean())
        vr = np.asarray(variance_ratio(pred[DAY8], truth[DAY8]), dtype=np.float64)
        results.append(WindowResult(
            window=w,
            violations=tuple(validity(pred, steps)),
            rmse=rmse,
            csi=cs,
            csi_counts=cc,
            observed_heavy_frac=hf,
            var_ratio=vr,
            min_tp=float(pred[:, 1].min()),
            tp_floor_cells=int(meta.get("tp_floor_cells", 0)),
            tp_floor_max_delta=float(meta.get("tp_floor_max_delta_mm", 0.0)),
        ))
    return results


def _rmses(results: list[WindowResult], lead: int) -> np.ndarray:
    return np.stack([r.rmse[lead] for r in results]) if results else np.empty((0, NC))


def summarize(results: list[WindowResult], leads: tuple[int, ...], std: np.ndarray) -> dict:
    n = len(results)
    out: dict = {"n_windows": n}
    rmse_mean, rmse_pooled, rmse_sigma = {}, {}, {}
    csi_mean, csi_pooled, csi_undef, obs_frac = {}, {}, {}, {}
    for k in leads:
        arr = _rmses(results, k)
        m = _nanmean(arr, axis=0)
        pooled = np.sqrt(_nanmean(arr ** 2, axis=0))
        rmse_mean[str(k)] = {ch: float(v) for ch, v in zip(CHANNELS, m)}
        rmse_pooled[str(k)] = {ch: float(v) for ch, v in zip(CHANNELS, pooled)}
        rmse_sigma[str(k)] = {ch: float(v) for ch, v in zip(CHANNELS, m / std)}

        vals = np.array([r.csi[k] for r in results], dtype=np.float64)
        csi_mean[str(k)] = float(_nanmean(vals))
        csi_undef[str(k)] = int((~np.isfinite(vals)).sum())
        h = sum(r.csi_counts[k][0] for r in results)
        mm = sum(r.csi_counts[k][1] for r in results)
        f = sum(r.csi_counts[k][2] for r in results)
        csi_pooled[str(k)] = float(h / (h + mm + f)) if (h + mm + f) else float("nan")
        obs_frac[str(k)] = float(_nanmean([r.observed_heavy_frac[k] for r in results]))

    ratios = np.stack([r.var_ratio for r in results]) if results else np.empty((0, NC))
    vr_mean = _nanmean(ratios, axis=0)
    in_band = np.isfinite(ratios) & (ratios >= BLUR_MIN_RATIO) & (ratios <= BLUR_MAX_RATIO)
    out.update({
        "rmse_mean_over_windows": rmse_mean,
        "rmse_pooled_over_windows": rmse_pooled,
        "rmse_mean_over_windows_sigma_units": rmse_sigma,
        "csi_mean_over_windows": csi_mean,
        "csi_pooled_over_windows": csi_pooled,
        "csi_undefined_windows": csi_undef,
        "observed_heavy_rain_cell_frac": obs_frac,
        "variance_ratio_day8_mean": {ch: float(v) for ch, v in zip(CHANNELS, vr_mean)},
        "variance_ratio_day8_in_band_frac": {
            ch: float(v) for ch, v in zip(CHANNELS, in_band.mean(axis=0) if len(results) else np.zeros(NC))
        },
        "anti_blur_points_of_10": float(anti_blur_points(ratios)) if len(results) else float("nan"),
        "validity": {
            "n_windows": n,
            "n_failed": int(sum(1 for r in results if r.violations)),
            "failed_windows": [
                {"init_last": r.window.init_last.isoformat(), "season": r.window.season,
                 "violations": list(r.violations)}
                for r in results if r.violations
            ],
            "violation_counts": {
                v: int(sum(1 for r in results if v in r.violations))
                for v in ("NaN/Inf", "negative rain", "t2m outside [-40, +60] C")
            },
            "tp_floor_cells": int(sum(r.tp_floor_cells for r in results)),
            "tp_floor_max_delta_mm": float(max([r.tp_floor_max_delta for r in results], default=0.0)),
        },
        "min_tp_mm_day": float(min([r.min_tp for r in results], default=float("nan"))),
    })
    return out


def compare(p_results: list[WindowResult], c_results: list[WindowResult],
            summary_p: dict, summary_c: dict, leads: tuple[int, ...]) -> dict:
    """Where does each reference win, and what is the effective 'best of both' bar?"""
    per_lead: dict = {}
    cells_c = cells_p = cells_tie = 0
    for k in leads:
        p = _rmses(p_results, k)
        c = _rmses(c_results, k)
        with np.errstate(divide="ignore", invalid="ignore"):
            ss = np.where(p > 0, 1.0 - c / p, np.nan)          # >0 => climatology better
        best = np.minimum(p, c)
        best_mean = _nanmean(best, axis=0)
        win_c = np.where(np.isfinite(p) & np.isfinite(c), c < p, False)
        win_p = np.where(np.isfinite(p) & np.isfinite(c), p < c, False)
        counts = {
            "climatology": int(win_c.sum(axis=0).sum()),
            "persistence": int(win_p.sum(axis=0).sum()),
            "tie": int((~win_c & ~win_p).sum()),
        }
        cells_c += counts["climatology"]
        cells_p += counts["persistence"]
        cells_tie += counts["tie"]

        delta = np.where(np.isfinite(p) & (p > 0), 100.0 * (c - p) / p, np.nan)
        csi_p = np.array([summary_p["csi_mean_over_windows"][str(k)]], dtype=np.float64)[0]
        csi_c = np.array([summary_c["csi_mean_over_windows"][str(k)]], dtype=np.float64)[0]
        cw_p = np.array([r.csi[k] for r in p_results], dtype=np.float64)
        cw_c = np.array([r.csi[k] for r in c_results], dtype=np.float64)
        both = np.isfinite(cw_p) & np.isfinite(cw_c)
        best_csi = np.where(both, np.maximum(cw_p, cw_c), np.where(np.isfinite(cw_p), cw_p, cw_c))
        per_lead[str(k)] = {
            "rmse_mean": {
                "persistence": {ch: float(v) for ch, v in zip(CHANNELS, _nanmean(p, axis=0))},
                "climatology": {ch: float(v) for ch, v in zip(CHANNELS, _nanmean(c, axis=0))},
            },
            "delta_rmse_pct_clim_vs_persist": {ch: float(v) for ch, v in zip(CHANNELS, _nanmean(delta, axis=0))},
            "rmse_skill_of_climatology_vs_persistence": {
                ch: float(v) for ch, v in zip(CHANNELS, _nanmean(ss, axis=0))
            },
            "mean_skill_of_climatology_vs_persistence": float(_nanmean(ss.ravel())),
            "best_reference_rmse_mean": {ch: float(v) for ch, v in zip(CHANNELS, best_mean)},
            "best_reference_source_counts": {
                ch: {"climatology": int(win_c[:, i].sum()), "persistence": int(win_p[:, i].sum()),
                     "tie": int((~win_c[:, i] & ~win_p[:, i]).sum())}
                for i, ch in enumerate(CHANNELS)
            },
            "csi": {
                "persistence": csi_p,
                "climatology": csi_c,
                "delta_clim_minus_persist": csi_c - csi_p,
                "best_reference_csi_mean": float(_nanmean(best_csi)),
                "best_reference_source_counts": {
                    "climatology": int((both & (cw_c > cw_p)).sum()),
                    "persistence": int((both & (cw_p > cw_c)).sum()),
                    "tie": int((both & (cw_p == cw_c)).sum()),
                },
                "observed_heavy_rain_cell_frac": summary_p["observed_heavy_rain_cell_frac"][str(k)],
            },
        }
    total = cells_c + cells_p + cells_tie
    return {
        "per_lead": per_lead,
        "win_rate_channel_lead_cells": {
            "n_cells": int(total),
            "climatology_better": int(cells_c),
            "persistence_better": int(cells_p),
            "ties": int(cells_tie),
        },
    }


# --------------------------------------------------------------------------- #
# console report
# --------------------------------------------------------------------------- #
def print_block(title: str, summary_p: dict, summary_c: dict, cmp: dict, leads: tuple[int, ...]) -> None:
    print(f"\n{title}")
    head = f"  {'channel':<8}{'unit':<7}" + "".join(
        f"{('d%d P' % k):>9}{('d%d C' % k):>9}{'dC%':>8}" for k in leads
    )
    print("  Latitude-weighted RMSE (cos-lat, physical units)   P = persistence, C = harmonic climatology")
    print(head)
    print("  " + "-" * (len(head) - 2))
    for ch, unit in zip(CHANNELS, UNITS_SHORT):
        row = f"  {ch:<8}{unit:<7}"
        for k in leads:                    row += (_num(summary_p["rmse_mean_over_windows"][str(k)][ch])
                    + _num(summary_c["rmse_mean_over_windows"][str(k)][ch])
                    + _pct(cmp["per_lead"][str(k)]["delta_rmse_pct_clim_vs_persist"][ch]))
        print(row)

    print("\n  Heavy-rain CSI (tp > 10 mm/day, mean over windows)")
    print("  " + f"{'':<15}" + "".join(f"{('d%d' % k):>9}" for k in leads))
    for name, s in (("persistence", summary_p), ("climatology", summary_c)):
        row = f"  {name:<15}" + "".join(f"{s['csi_mean_over_windows'][str(k)]:>9.4f}" for k in leads)
        print(row)
    print("  observed >10" + " " * 3 + "".join(
        f"{summary_p['observed_heavy_rain_cell_frac'][str(k)]:>8.2%} " for k in leads))
    undef_p = [summary_p["csi_undefined_windows"][str(k)] for k in leads]
    undef_c = [summary_c["csi_undefined_windows"][str(k)] for k in leads]
    if sum(undef_p) or sum(undef_c):
        print("  undefined windows (no heavy-rain cell): "
              + " | ".join(f"d{k} P {a} C {b}" for k, a, b in zip(leads, undef_p, undef_c)))

    print(f"\n  Day-8 spatial variance ratio (anti-blur band {BLUR_MIN_RATIO:.2f}-{BLUR_MAX_RATIO:.2f})")
    print("  " + f"{'':<15}" + "".join(f"{ch:>9}" for ch in CHANNELS))
    for name, s in (("persistence", summary_p), ("climatology", summary_c)):
        print(f"  {name:<15}" + "".join(_num(s["variance_ratio_day8_mean"][ch]) for ch in CHANNELS))
    print(f"  anti-blur points (of 10): persistence {summary_p['anti_blur_points_of_10']:.2f} | "
          f"climatology {summary_c['anti_blur_points_of_10']:.2f}")

    vp, vc = summary_p["validity"], summary_c["validity"]
    extra = ""
    if vc["tp_floor_cells"]:
        extra = (f"; climatology floored {vc['tp_floor_cells']} negative tp cells "
                 f"(max {vc['tp_floor_max_delta_mm']:.3g} mm/day)")
    print(f"  Validity: persistence {vp['n_failed']}/{vp['n_windows']} failed | "
          f"climatology {vc['n_failed']}/{vc['n_windows']} failed{extra}")

    wr = cmp["win_rate_channel_lead_cells"]
    print(f"  RMSE win rate over {wr['n_cells']} (window x channel x lead) cells: "
          f"climatology {wr['climatology_better']}, persistence {wr['persistence_better']}, ties {wr['ties']}")


def print_reference_bar(overall_p: dict, overall_c: dict, cmp: dict, leads: tuple[int, ...]) -> None:
    print("\n  REFERENCE BAR for a trained model — elementwise best of P and C per window")
    print(f"  {'channel':<8}{'unit':<7}" + "".join(f"{('d%d best' % k):>10}{'source':>13}" for k in leads))
    print("  " + "-" * (15 + 23 * len(leads)))
    for ch, unit in zip(CHANNELS, UNITS_SHORT):
        row = f"  {ch:<8}{unit:<7}"
        for k in leads:
            cnt = cmp["per_lead"][str(k)]["best_reference_source_counts"][ch]
            src = "climatology" if cnt["climatology"] >= cnt["persistence"] else "persistence"
            row += _num(cmp["per_lead"][str(k)]["best_reference_rmse_mean"][ch], width=10) + f"{src:>13}"
        print(row)
    print("  mean RMSE skill of climatology vs persistence (1 - RMSE_c / RMSE_p): "
          + ", ".join(f"d{k} {cmp['per_lead'][str(k)]['mean_skill_of_climatology_vs_persistence']:+.3f}" for k in leads))

    # The f-string expressions below are kept on one line: a multi-line
    # expression inside a single-quoted f-string is a syntax error before
    # Python 3.12, and this module has to import on every scoring box.
    csi_cells = []
    for k in leads:
        csi = cmp["per_lead"][str(k)]["csi"]
        counts = csi["best_reference_source_counts"]
        best_src = max(counts, key=counts.get)[:4]
        csi_cells.append(f"d{k} {csi['best_reference_csi_mean']:.3f} ({best_src})")
    print("  heavy-rain CSI bar (better of P and C per window): " + ", ".join(csi_cells))


# --------------------------------------------------------------------------- #
# main
# --------------------------------------------------------------------------- #
def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--data", default=DATA_FILE, help=f"training NetCDF (default {DATA_FILE})")
    ap.add_argument("--out", default=OUT_FILE, help=f"JSON output (default {OUT_FILE})")
    ap.add_argument("--harmonics", type=int, default=2, help="annual harmonics K of the primary climatology (default 2)")
    ap.add_argument("--sweep", default="1,2,3", help="harmonic orders for the sensitivity sweep (default 1,2,3; '' disables)")
    ap.add_argument("--stride", type=int, default=4, help="days between window init dates within a sampled month (default 4)")
    ap.add_argument("--steps", type=int, default=SCORE_STEPS, help=f"forecast steps (default {SCORE_STEPS})")
    ap.add_argument("--rain-floor", type=float, default=0.0, help="climatology tp is floored here (default 0.0 mm/day)")
    ap.add_argument("--no-rain-floor", action="store_true", help="keep raw harmonic tp, even if negative")
    ap.add_argument("--quiet", action="store_true", help="write JSON only, no tables")
    args = ap.parse_args(argv)

    if args.harmonics < 0 or args.harmonics > 6:
        raise SystemExit("--harmonics must be between 0 and 6")
    if args.steps < 8:
        raise SystemExit("--steps must be >= 8 (day-8 anti-blur check)")
    if args.stride < 1:
        raise SystemExit("--stride must be >= 1")

    t_start = time.perf_counter()
    ds = load_dataset(args.data)
    std, std_source = load_std()
    steps = int(args.steps)
    leads = tuple(int(k) for k in LEAD_DAYS if int(k) <= steps)
    rain_floor = None if args.no_rain_floor else float(args.rain_floor)

    train_idx = np.array([i for i, d in enumerate(ds.days) if TRAIN_START <= d <= TRAIN_END], dtype=np.int64)
    val_idx = np.array([i for i, d in enumerate(ds.days) if VAL_START <= d <= VAL_END], dtype=np.int64)
    windows = build_windows(ds, SEASONS, stride=args.stride, steps=steps)
    if not windows:
        raise SystemExit("no validation windows were built")
    by_season = {key: [w for w in windows if w.season == key] for key, _, _ in SEASONS}

    if not args.quiet:
        print("Round 1 reference baselines — internal grading harness")
        print(f"dataset   {ds.path}  sha256 {ds.sha256[:16]}…  {len(ds)} daily fields, "
              f"{NC}x{NLAT}x{NLON} grid, {', '.join(CHANNELS)}")
        print(f"train     {ds.days[train_idx[0]]} .. {ds.days[train_idx[-1]]}  ({len(train_idx)} days)")
        print(f"valid     {ds.days[val_idx[0]]} .. {ds.days[val_idx[-1]]}  ({len(val_idx)} days)")
        print(f"clim      K={args.harmonics} annual harmonics, phase 2*pi*doy/365, fitted on train only, "
              f"tp floor {'off' if rain_floor is None else f'{rain_floor:g} mm/day'}")
        print(f"windows   {len(windows)} rollouts of 3 input + {steps} target days, init every "
              f"{args.stride} day(s) in Jun / Jul-Aug / Oct / Jan of {VAL_START.year}-{VAL_END.year}")

    # ---- references -------------------------------------------------------
    pers = persistence_forecast
    clim_fit = fit_harmonic_climatology(ds.data, ds.days, train_idx, args.harmonics, rain_floor)
    clim = climatology_forecast(clim_fit)

    results_p = evaluate_windows(pers, windows, ds, leads)
    results_c = evaluate_windows(clim, windows, ds, leads)

    def block(label: str, months, wins, rp, rc) -> dict:
        sp = summarize(rp, leads, std)
        sc = summarize(rc, leads, std)
        cmp_ = compare(rp, rc, sp, sc, leads)
        return {
            "label": label,
            "months": list(months),
            "n_windows": len(wins),
            "init_dates": [w.init_last.isoformat() for w in wins],
            "baselines": {"persistence": sp, "harmonic_climatology": sc},
            "comparison": cmp_,
        }

    seasons_out = {}
    for key, label, months in SEASONS:
        wins = by_season[key]
        if not wins:
            continue
        rp = [r for r in results_p if r.window.season == key]
        rc = [r for r in results_c if r.window.season == key]
        seasons_out[key] = block(label, months, wins, rp, rc)

    sum_p_all = summarize(results_p, leads, std)
    sum_c_all = summarize(results_c, leads, std)
    overall = {
        "label": f"All sampled validation windows ({', '.join(s[1] for s in SEASONS)})",
        "seasons": [s[0] for s in SEASONS],
        "n_windows": len(windows),
        "baselines": {"persistence": sum_p_all, "harmonic_climatology": sum_c_all},
        "comparison": compare(results_p, results_c, sum_p_all, sum_c_all, leads),
    }

    # ---- harmonic-order sensitivity --------------------------------------
    sweep_ks = []
    for token in str(args.sweep).replace(" ", "").split(","):
        if token.isdigit():
            k = int(token)
            if 0 <= k <= 6 and k not in sweep_ks:
                sweep_ks.append(k)
    sweep = {}
    for k in sweep_ks:
        if k == args.harmonics:
            fit_k, sum_k = clim_fit, overall["baselines"]["harmonic_climatology"]
        else:
            fit_k = fit_harmonic_climatology(ds.data, ds.days, train_idx, k, rain_floor)
            sum_k = summarize(evaluate_windows(climatology_forecast(fit_k), windows, ds, leads), leads, std)
        sweep[str(k)] = {
            "n_harmonics": k,
            "n_coefficients": 1 + 2 * k,
            "train_rmse": {ch: float(v) for ch, v in zip(CHANNELS, fit_k.train_rmse)},
            "train_r2": {ch: float(v) for ch, v in zip(CHANNELS, fit_k.train_r2)},
            "overall": sum_k,
            "mean_skill_vs_persistence": {
                str(le): float(_nanmean(np.array(
                    [1.0 - sum_k["rmse_mean_over_windows"][str(le)][ch] /
                     overall["baselines"]["persistence"]["rmse_mean_over_windows"][str(le)][ch]
                     for ch in CHANNELS], dtype=np.float64)))
                for le in leads
            },
        }

    # ---- effect of flooring climatological rain at zero --------------------
    floor_sensitivity = None
    if rain_floor is not None:
        raw_fit = fit_harmonic_climatology(ds.data, ds.days, train_idx, args.harmonics, None)
        raw_sum = summarize(evaluate_windows(climatology_forecast(raw_fit), windows, ds, leads), leads, std)
        floor_sensitivity = {
            "floor_applied_mm_day": rain_floor,
            "climatology_tp_rmse_mean_over_windows": {
                "floored": {str(le): overall["baselines"]["harmonic_climatology"]
                            ["rmse_mean_over_windows"][str(le)]["tp"] for le in leads},
                "raw": {str(le): raw_sum["rmse_mean_over_windows"][str(le)]["tp"] for le in leads},
            },
            "cells_floored": overall["baselines"]["harmonic_climatology"]["validity"]["tp_floor_cells"],
            "raw_windows_with_negative_rain": raw_sum["validity"]["violation_counts"]["negative rain"],
            "raw_most_negative_tp_mm_day": raw_sum["min_tp_mm_day"],
            "raw_windows_failing_validity": raw_sum["validity"]["n_failed"],
        }

    elapsed = time.perf_counter() - t_start
    payload = {
        "schema_version": "baseline-scores-1.0",
        "phase": "Round 1 Phase 1 — baseline evaluation & internal grading harness",
        "generated_at_utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        "generator": "evaluate_baselines.py",
        "dataset": {
            "file": Path(args.data).name,
            "sha256": ds.sha256,
            "n_days": len(ds),
            "time_range": [ds.days[0].isoformat(), ds.days[-1].isoformat()],
            "grid": {"shape": [NLAT, NLON], "lat": [float(LAT[0]), float(LAT[-1])],
                     "lon": [float(LON[0]), float(LON[-1])], "degree": 1.0},
            "channels": {ch: {"index": i, "unit": UNITS[i]} for i, ch in enumerate(CHANNELS)},
        },
        "split": {
            "train": {"start": TRAIN_START.isoformat(), "end": TRAIN_END.isoformat(), "n_days": int(len(train_idx))},
            "validation": {"start": VAL_START.isoformat(), "end": VAL_END.isoformat(), "n_days": int(len(val_idx))},
            "note": "Climatology coefficients are estimated on train days only; validation windows use observed "
                    "inputs for their 3-day history and score only the 10 target days.",
        },
        "config": {
            "forecast_steps": steps,
            "lead_days": list(leads),
            "rain_threshold_mm_day": RAIN_THRESH,
            "blur_ratio_band": [BLUR_MIN_RATIO, BLUR_MAX_RATIO],
            "latitude_weights": "cos(deg2rad(lat)) normalised to mean 1 via monsoon_r1.contract.lat_weights",
            "harmonics": args.harmonics,
            "doy_convention": "phase = 2*pi*doy/365 (monsoon_r1.contract.doy_sincos); future dates are real calendar dates",
            "climatology_rain_floor_mm_day": rain_floor,
            "window_sampling": {
                "stride_days": args.stride,
                "years": [VAL_START.year, VAL_END.year],
                "seasons": {key: list(months) for key, _, months in SEASONS},
                "rule": "init day every stride days per sampled month; the init day and all 10 target days lie "
                        "inside the validation period (the 2-day history may reach into the last observed days of "
                        "the train period, which are inputs only, never fitted values)",
            },
            "prediction_dtype": "float32 (wire format)",
            "std_source": std_source,
            "aggregation": "RMSE/CSI/variance-ratio figures are arithmetic means over windows of the per-window "
                           "kit metric (the judge averages per-window skill); pooled variants are also stored",
        },
        "metric_sources": {
            "rmse": "monsoon_r1.scoring.rmse_lat",
            "csi": "monsoon_r1.scoring.csi",
            "variance_ratio": "monsoon_r1.scoring.variance_ratio",
            "anti_blur": "monsoon_r1.scoring.anti_blur_points",
            "validity": "monsoon_r1.scoring.validity",
            "lat_weights": "monsoon_r1.contract.lat_weights",
        },
        "notes": [
            WIRE_NOTE,
            "null values mean the metric is undefined for that slice (never silently zeroed).",
            "Persistence ignores the calendar; climatology uses the real target date's day-of-year.",
            "Climatology tp is floored at the configured value to satisfy the physical validity gate; the number of "
            "floored cells is reported per baseline." if rain_floor is not None else
            "Raw harmonic tp is kept, so negative-rain violations are reported rather than clipped.",
        ],
        "seasons": seasons_out,
        "overall": overall,
        "supplementary": {
            "harmonic_sweep": sweep,
            "rain_floor_sensitivity": floor_sensitivity,
            "dataset_continuity": continuity_stats(ds.data),
        },
    }

    out_path = Path(args.out)
    out_path.write_text(json.dumps(_jsonable(payload), indent=2, allow_nan=False) + "\n")

    if not args.quiet:
        print("\n" + "=" * 100)
        print("SEASON BREAKDOWN")
        print("=" * 100)
        for key, _, _ in SEASONS:
            if key not in seasons_out:
                continue
            b = seasons_out[key]
            print_block(f"\n[{key}] {b['label']} — {b['n_windows']} windows "
                        f"({b['init_dates'][0]} .. {b['init_dates'][-1]})",
                        b["baselines"]["persistence"], b["baselines"]["harmonic_climatology"],
                        b["comparison"], leads)

        print("\n" + "=" * 100)
        print(f"OVERALL — {overall['n_windows']} validation windows")
        print("=" * 100)
        op, oc = overall["baselines"]["persistence"], overall["baselines"]["harmonic_climatology"]
        print_block("", op, oc, overall["comparison"], leads)
        print_reference_bar(op, oc, overall["comparison"], leads)

        print("\n  Climatology harmonic-order sensitivity (overall, mean RMSE by channel)")
        for k, s in sweep.items():
            ss = s["mean_skill_vs_persistence"]
            print(f"    K={k} ({s['n_coefficients']} coefs)  "
                  + "  ".join(f"d{le} SS {ss[str(le)]:+.3f}" for le in leads)
                  + "  | day-8 var ratio "
                  + " ".join(f"{ch} {s['overall']['variance_ratio_day8_mean'][ch]:.2f}" for ch in CHANNELS))
        if floor_sensitivity:
            fs = floor_sensitivity
            tp_raw = fs["climatology_tp_rmse_mean_over_windows"]["raw"]
            tp_flo = fs["climatology_tp_rmse_mean_over_windows"]["floored"]
            print("  rain floor: climatology tp RMSE "
                  + " ".join(f"d{le} {tp_raw[str(le)]:.3f}->{tp_flo[str(le)]:.3f}" for le in leads)
                  + f" | raw tp reaches {fs['raw_most_negative_tp_mm_day']:.3f} mm/day, "
                    f"raw/floored validity failures {fs['raw_windows_failing_validity']}/"
                    f"{oc['validity']['n_failed']} windows")
        print(f"\n  wrote {out_path}  ({out_path.stat().st_size / 1024:.0f} KB, {elapsed:.1f}s)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
