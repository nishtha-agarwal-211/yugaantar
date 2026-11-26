"""Compact 2-frequency harmonic climatology for the 32 x 36 India grid.

The digital twin does not forecast the absolute state: it forecasts the one-day
*increment* of the state, and it needs a smooth, cheap reference climate to know
what "normal" looks like for the current day-of-year.  A 365-day raw lookup table
of 2020-2021 fields would be a date lookup (explicitly banned by the rules), so we
fit a harmonic model instead.  Per grid cell and per channel:

    clim(doy) = mean
              + a1 * sin(2*pi*doy/365) + b1 * cos(2*pi*doy/365)
              + a2 * sin(4*pi*doy/365) + b2 * cos(4*pi*doy/365)

Fitted by ordinary least squares on 2005-01-01 .. 2019-12-31 only.  That is five
numbers per cell and channel, i.e. a [5, 6, 32, 36] tensor.  Stored as float32
for the annual mean and float16 for the four wave coefficients it is ~81 KB,
small enough to bundle with the weights and impossible to mistake for a data
archive.

Two conveniences make this usable inside the served model:

* ``predict_theta`` evaluates the model from a single angle theta = 2*pi*doy/365,
  and the second harmonic is obtained with the double-angle identities
  ``sin(2t) = 2 sin t cos t`` and ``cos(2t) = cos^2 t - sin^2 t``.  The API only
  sends ``doy_sincos`` of the last input day, and this is all the information
  needed to evaluate the model for every forecast lead.
* ``predict_steps`` advances the angle by 2*pi/365 per day, so a rollout never
  touches a calendar.

Precipitation is non-negative by construction: either the raw-space fit is
floored at 0 mm/day (default, ``rain_space="linear"``), or the coefficient set is
fitted on ``log1p(tp)`` and inverted with ``expm1`` (``rain_space="log1p"``),
which is strictly positive and cannot produce negative rain.

CLI
    python climatology.py fit  --data india_climate_train.nc --out artifacts/climatology_coefs.npz
    python climatology.py show --coefs artifacts/climatology_coefs.npz
"""
from __future__ import annotations

import argparse
import json
import math
from dataclasses import dataclass, field
from datetime import date
from pathlib import Path
from typing import Iterable, Sequence

import numpy as np

# --------------------------------------------------------------------------- #
# grid / contract constants (duplicated on purpose: the served model must not
# depend on the monsoon_r1 kit, which is not shipped inside the graded repo)
# --------------------------------------------------------------------------- #
CHANNELS: tuple[str, ...] = ("t2m", "tp", "u10", "v10", "msl", "tcwv")
NC, NLAT, NLON = 6, 32, 36
LAT = np.arange(6.5, 37.5 + 1e-9, 1.0, dtype=np.float64)
LON = np.arange(65.5, 100.5 + 1e-9, 1.0, dtype=np.float64)
YEAR_DAYS = 365.0                      # contract convention, not a leap calendar
TP_INDEX = CHANNELS.index("tp")

TRAIN_START = date(2005, 1, 1)
TRAIN_END = date(2019, 12, 31)

COEF_NAMES: tuple[str, ...] = ("mean", "sin1", "cos1", "sin2", "cos2")
N_COEF = len(COEF_NAMES)

DEFAULT_COEFS = Path("artifacts") / "climatology_coefs.npz"


# --------------------------------------------------------------------------- #
# small helpers
# --------------------------------------------------------------------------- #
def theta_from_doy(doy: np.ndarray | float) -> np.ndarray:
    """Contract phase convention: theta = 2*pi*doy/365 (doy is 1-based)."""
    return 2.0 * np.pi * np.asarray(doy, dtype=np.float64) / YEAR_DAYS


def theta_from_sincos(sin_component: float, cos_component: float) -> float:
    """Recover theta from the wire-format pair [sin(theta), cos(theta)]."""
    return float(math.atan2(float(sin_component), float(cos_component)))


def harmonic_basis(theta: np.ndarray | float) -> np.ndarray:
    """Design matrix rows [1, sin(t), cos(t), sin(2t), cos(2t)] along the last axis."""
    t = np.asarray(theta, dtype=np.float64)
    s1, c1 = np.sin(t), np.cos(t)
    return np.stack([np.ones_like(t), s1, c1, 2.0 * s1 * c1, c1 * c1 - s1 * s1], axis=-1)


def day_of_year(d: date) -> int:
    return d.timetuple().tm_yday


# --------------------------------------------------------------------------- #
# the model
# --------------------------------------------------------------------------- #
@dataclass
class HarmonicClimatology:
    """A [5, 6, 32, 36] harmonic coefficient tensor plus its provenance.

    ``coef[0]`` is the annual mean, ``coef[1:3]`` the first harmonic (sin, cos)
    and ``coef[3:5]`` the second harmonic.  ``rain_space`` is ``"linear"`` (tp
    coefficients live in mm/day, predictions are floored at 0) or ``"log1p"``
    (fitted on log1p(tp), predictions are ``expm1`` clipped at 0).
    """

    coef: np.ndarray                       # [5, 6, 32, 36] float32
    rain_space: str = "linear"
    train_start: str = TRAIN_START.isoformat()
    train_end: str = TRAIN_END.isoformat()
    n_train_days: int = 0
    train_rmse: np.ndarray = field(default_factory=lambda: np.zeros(NC, np.float64))
    train_r2: np.ndarray = field(default_factory=lambda: np.zeros(NC, np.float64))

    # -- validation -------------------------------------------------------
    def __post_init__(self) -> None:
        self.coef = np.ascontiguousarray(self.coef, dtype=np.float32)
        if self.coef.shape != (N_COEF, NC, NLAT, NLON):
            raise ValueError(f"coefficient shape {self.coef.shape} != {(N_COEF, NC, NLAT, NLON)}")
        if self.rain_space not in ("linear", "log1p"):
            raise ValueError(f"unknown rain_space {self.rain_space!r}")

    # -- evaluation -------------------------------------------------------
    def predict_theta(self, theta) -> np.ndarray:
        """Physical climatology for one or more angles.

        Returns ``[..., 6, 32, 36]`` float32.  Supported shapes: scalar, ``[n]``
        or ``[n, 1]``; the leading axes of the result mirror the input.
        """
        t = np.asarray(theta, dtype=np.float64)
        if t.ndim == 0:
            basis = harmonic_basis(t)[None, :]              # [1, 5]
            out = np.einsum("sp,pcxy->scxy", basis, self.coef)[0]
        else:
            flat = t.reshape(-1)
            basis = harmonic_basis(flat)                    # [m, 5]
            out = np.einsum("mp,pcxy->mcxy", basis, self.coef).reshape(
                (*t.shape, NC, NLAT, NLON)
            )
        out = np.asarray(out, dtype=np.float32)
        return self._physicalise(out)

    def predict_sincos(self, sin_component: float, cos_component: float) -> np.ndarray:
        """Climatology for the day whose phase is ``[sin, cos]`` -> [6, 32, 36]."""
        t = theta_from_sincos(sin_component, cos_component)
        basis = np.array([1.0, sin_component, cos_component,
                          2.0 * sin_component * cos_component,
                          cos_component ** 2 - sin_component ** 2], dtype=np.float64)
        out = np.einsum("p,pcxy->cxy", basis, self.coef)[None]
        return self._physicalise(np.asarray(out, dtype=np.float32))[0]

    def predict_steps(self, sin_component: float, cos_component: float, steps: int) -> np.ndarray:
        """Climatology [steps, 6, 32, 36] for forecast days 1..steps.

        The angle advances by 2*pi/365 per day from the phase of the LAST INPUT
        day, exactly as the contract defines (forecast step k is doy + k).
        """
        t0 = theta_from_sincos(sin_component, cos_component)
        theta = t0 + 2.0 * np.pi * np.arange(1, int(steps) + 1) / YEAR_DAYS
        return self.predict_theta(theta)

    def predict_dates(self, dates: Sequence[date]) -> np.ndarray:
        doy = np.array([day_of_year(d) for d in dates], dtype=np.float64)
        return self.predict_theta(theta_from_doy(doy))

    def climatology_latent(self, codec, theta) -> np.ndarray:
        """Climatology mapped through the state codec into model space."""
        return codec.encode(self.predict_theta(theta))

    def _physicalise(self, out: np.ndarray) -> np.ndarray:
        """Apply the per-channel physical constraints of the climatology."""
        if self.rain_space == "log1p":
            tp = np.expm1(np.clip(out[..., TP_INDEX, :, :], 0.0, 12.0))
        else:
            tp = np.maximum(out[..., TP_INDEX, :, :], 0.0)
        out = out.copy()
        out[..., TP_INDEX, :, :] = tp.astype(np.float32)
        return out

    # -- persistence ------------------------------------------------------
    def save(self, path: str | Path = DEFAULT_COEFS) -> Path:
        """Write the compact bundle (annual mean float32, waves float16)."""
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        np.savez_compressed(
            path,
            coef_mean=np.asarray(self.coef[0], dtype=np.float32),
            coef_waves=np.asarray(self.coef[1:], dtype=np.float16),
            channels=np.array(CHANNELS),
            coef_names=np.array(COEF_NAMES),
            rain_space=np.array(self.rain_space),
            train_start=np.array(self.train_start),
            train_end=np.array(self.train_end),
            n_train_days=np.array(int(self.n_train_days)),
            train_rmse=np.asarray(self.train_rmse, dtype=np.float64),
            train_r2=np.asarray(self.train_r2, dtype=np.float64),
            grid=np.array([NC, NLAT, NLON], dtype=np.int64),
        )
        return path

    @classmethod
    def load(cls, path: str | Path = DEFAULT_COEFS) -> "HarmonicClimatology":
        with np.load(Path(path), allow_pickle=False) as z:
            mean = np.asarray(z["coef_mean"], dtype=np.float32)
            waves = np.asarray(z["coef_waves"], dtype=np.float32)
            if mean.shape != (NC, NLAT, NLON) or waves.shape != (N_COEF - 1, NC, NLAT, NLON):
                raise ValueError(f"bad coefficient shapes {mean.shape} {waves.shape}")
            return cls(
                coef=np.concatenate([mean[None], waves], axis=0),
                rain_space=str(z["rain_space"]),
                train_start=str(z["train_start"]),
                train_end=str(z["train_end"]),
                n_train_days=int(z["n_train_days"]),
                train_rmse=np.asarray(z["train_rmse"], dtype=np.float64),
                train_r2=np.asarray(z["train_r2"], dtype=np.float64),
            )

    @property
    def n_bytes_float32(self) -> int:
        return int(self.coef.size * 4)

    def describe(self) -> dict:
        return {
            "channels": list(CHANNELS),
            "coefficients": list(COEF_NAMES),
            "shape": list(self.coef.shape),
            "n_parameters": int(self.coef.size),
            "bytes_float32": self.n_bytes_float32,
            "rain_space": self.rain_space,
            "train_period": [self.train_start, self.train_end],
            "n_train_days": int(self.n_train_days),
            "train_rmse": {c: float(v) for c, v in zip(CHANNELS, self.train_rmse)},
            "train_r2": {c: float(v) for c, v in zip(CHANNELS, self.train_r2)},
        }


# --------------------------------------------------------------------------- #
# fitting
# --------------------------------------------------------------------------- #
def _mask_days(days: Sequence[date], start: date, end: date) -> np.ndarray:
    return np.array([start <= d <= end for d in days], dtype=bool)


def fit_harmonic_climatology(
    data: np.ndarray,
    days: Sequence[date],
    rain_space: str = "linear",
    start: date = TRAIN_START,
    end: date = TRAIN_END,
    ridge: float = 0.0,
) -> HarmonicClimatology:
    """Least-squares 2-frequency harmonic fit per cell and channel.

    ``data`` is ``[time, 6, 32, 36]`` in physical units (float32 or float64).
    Only days inside ``[start, end]`` are used, so the fit cannot see the
    validation or the hidden test period.
    """
    data = np.asarray(data)
    if data.ndim != 4 or data.shape[1:] != (NC, NLAT, NLON) or data.shape[0] != len(days):
        raise ValueError(f"bad data shape {data.shape} for {len(days)} days")
    train = _mask_days(days, start, end)
    if train.sum() < 3 * N_COEF:
        raise ValueError(f"only {int(train.sum())} training days in {start}..{end}")
    idx = np.flatnonzero(train)
    doy = np.array([day_of_year(days[i]) for i in idx], dtype=np.float64)
    A = harmonic_basis(theta_from_doy(doy))                  # [n, 5]
    n = A.shape[0]

    if ridge > 0.0:
        gram = A.T @ A + ridge * np.eye(N_COEF)
        solver = np.linalg.solve(gram, A.T)                  # [5, n]
    else:
        solver = np.linalg.pinv(A)                           # [5, n]

    coef = np.empty((N_COEF, NC, NLAT, NLON), dtype=np.float64)
    train_rmse = np.empty(NC, dtype=np.float64)
    train_r2 = np.empty(NC, dtype=np.float64)
    for c in range(NC):
        Y = np.asarray(data[idx, c], dtype=np.float64).reshape(n, -1)
        workspace = Y.copy()
        if c == TP_INDEX and rain_space == "log1p":
            np.log1p(np.maximum(workspace, 0.0), out=workspace)
        B = solver @ workspace                              # [5, cells]
        coef[:, c] = B.reshape(N_COEF, NLAT, NLON)
        resid = workspace - A @ B
        ss_res = float((resid ** 2).sum())
        ss_tot = float(((workspace - workspace.mean(axis=0)) ** 2).sum())
        train_rmse[c] = math.sqrt(ss_res / workspace.size)
        train_r2[c] = 1.0 - ss_res / ss_tot if ss_tot > 0 else float("nan")

    return HarmonicClimatology(
        coef=coef.astype(np.float32),
        rain_space=rain_space,
        train_start=start.isoformat(),
        train_end=end.isoformat(),
        n_train_days=int(n),
        train_rmse=train_rmse,
        train_r2=train_r2,
    )


def fit_from_netcdf(
    path: str | Path = "india_climate_train.nc",
    rain_space: str = "linear",
    start: date = TRAIN_START,
    end: date = TRAIN_END,
    ridge: float = 0.0,
) -> tuple[HarmonicClimatology, list[date]]:
    """Read the training NetCDF and fit the climatology (xarray imported lazily)."""
    import xarray as xr

    with xr.open_dataset(Path(path)) as ds:
        order = tuple(str(c) for c in ds["channel"].values)
        if order != CHANNELS:
            raise ValueError(f"channel order {order} != {CHANNELS}")
        data = np.asarray(ds["data"].values, dtype=np.float32)
        raw = ds["time"].values.astype("datetime64[D]").tolist()
    days = list(raw)
    clim = fit_harmonic_climatology(data, days, rain_space=rain_space,
                                    start=start, end=end, ridge=ridge)
    return clim, days


# --------------------------------------------------------------------------- #
# CLI
# --------------------------------------------------------------------------- #
def _cmd_fit(args) -> int:
    clim, days = fit_from_netcdf(args.data, rain_space=args.rain_space,
                                 ridge=args.ridge)
    out = clim.save(args.out)
    size_kb = out.stat().st_size / 1024.0
    print(f"fitted on {clim.n_train_days} days ({clim.train_start} .. {clim.train_end})")
    space = "log1p(tp)" if clim.rain_space == "log1p" else "physical units"
    print(f"  fit space: {space} (RMSE below is in that space)")
    for c, rmse, r2 in zip(CHANNELS, clim.train_rmse, clim.train_r2):
        print(f"  {c:<5} train RMSE {rmse:9.4f}   R2 vs constant mean {r2:+.4f}")
    print(f"wrote {out}  ({size_kb:.1f} KB, {clim.coef.size} coefficients)")
    print(f"float32 tensor would be {clim.n_bytes_float32 / 1024:.1f} KB")
    if args.json:
        Path(args.json).write_text(json.dumps(clim.describe(), indent=2) + "\n")
    return 0


def _cmd_show(args) -> int:
    clim = HarmonicClimatology.load(args.coefs)
    info = clim.describe()
    info["file_bytes"] = Path(args.coefs).stat().st_size
    # verify the wire-format evaluation path: [sin, cos] of 2015-01-01 (doy 1)
    # must reproduce the calendar evaluation of the same day exactly.
    probe_day = date(2015, 1, 1)
    t = float(theta_from_doy(day_of_year(probe_day)))
    probe = clim.predict_sincos(math.sin(t), math.cos(t))
    ref = clim.predict_dates([probe_day])[0]
    info["sincos_vs_date_max_abs_diff"] = float(np.abs(probe - ref).max())
    print(json.dumps(info, indent=2))
    return 0


def main(argv: Iterable[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = ap.add_subparsers(dest="cmd", required=True)

    f = sub.add_parser("fit", help="fit the harmonic climatology on 2005-2019")
    f.add_argument("--data", default="india_climate_train.nc")
    f.add_argument("--out", default=str(DEFAULT_COEFS))
    f.add_argument("--rain-space", choices=("linear", "log1p"), default="linear")
    f.add_argument("--ridge", type=float, default=0.0)
    f.add_argument("--json", default="", help="optional path for a metadata dump")
    f.set_defaults(func=_cmd_fit)

    s = sub.add_parser("show", help="inspect a saved bundle")
    s.add_argument("--coefs", default=str(DEFAULT_COEFS))
    s.set_defaults(func=_cmd_show)

    args = ap.parse_args(list(argv) if argv is not None else None)
    return int(args.func(args))


if __name__ == "__main__":
    raise SystemExit(main())
