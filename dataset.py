"""Data pipeline for the India climate digital twin (Phase 2).

Responsibilities
----------------
1. ``StateCodec``  — maps physical states ``[..., 6, 32, 36]`` (K, mm/day, m/s,
   m/s, Pa, kg/m2) to the dimensionless space the network sees and back.  The
   five linear channels are standardised with the shipped ``stats.json``
   constants; precipitation is standardised in ``log1p`` space because daily
   rain is heavy-tailed (median ~0.05 mm/day, max 250 mm/day), and a linear
   standardisation would squeeze drizzle to nothing and still leave a 30-sigma
   outlier.  Decoding uses ``expm1`` of a clipped non-negative latent, so the
   returned rain is mathematically >= 0.0 mm/day.
2. ``StaticFields`` — bundles ``land_sea_mask``, normalised ``orography_m`` and
   two coordinate planes (a purely local convolution has no other way to know
   where it is on the globe) as a [4, 32, 36] float32 tensor.
3. ``ClimateWindows`` — the sliding-window dataset: three consecutive observed
   days in, up to ``horizon`` autoregressive days out, with the day-of-year
   phase of every forecast day, the current anomaly against the harmonic
   climatology, and both absolute-state and one-day-delta targets.

Everything here is import-time cheap: xarray is only used inside ``load_netcdf``
so the served model can reuse the codec and static bundle without paying for it.

Build the artifacts once:

    python dataset.py build --data india_climate_train.nc --stats stats.json \
        --climatology artifacts/climatology_coefs.npz --out artifacts/static_codec.npz
"""
from __future__ import annotations

import argparse
import json
import math
from dataclasses import dataclass
from datetime import date
from pathlib import Path
from typing import Iterable, Sequence

import numpy as np
import torch
from torch.utils.data import DataLoader, Dataset

from climatology import (
    CHANNELS,
    NC,
    NLAT,
    NLON,
    TP_INDEX,
    HarmonicClimatology,
    day_of_year,
    theta_from_doy,
)

LAT = np.arange(6.5, 37.5 + 1e-9, 1.0, dtype=np.float64)
LON = np.arange(65.5, 100.5 + 1e-9, 1.0, dtype=np.float64)

TRAIN_START, TRAIN_END = date(2005, 1, 1), date(2019, 12, 31)
VAL_START, VAL_END = date(2020, 1, 1), date(2021, 12, 31)
SPLITS: dict[str, tuple[date, date]] = {
    "train": (TRAIN_START, TRAIN_END),
    "val": (VAL_START, VAL_END),
    "all": (date(2005, 1, 1), date(2021, 12, 31)),
}

IN_STEPS = 3
DEFAULT_HORIZON = 10
DEFAULT_STATIC = Path("artifacts") / "static_codec.npz"
DEFAULT_CLIMATOLOGY = Path("artifacts") / "climatology_coefs.npz"

# physical plausibility guards; a correct model never touches them, but they make
# the validity gate (finite, rain >= 0, t2m in [-40, 60] C) unbreakable.
GUARDS: dict[str, tuple[float, float]] = {
    "t2m": (234.15, 332.15),      # -39 .. +59 C, one degree inside the gate
    "u10": (-60.0, 60.0),
    "v10": (-60.0, 60.0),
    "msl": (88000.0, 107000.0),
    "tcwv": (0.0, 120.0),
}
TP_MAX_MM_DAY = 500.0             # expm1 clip bound for the rain latent
ACTIVE_CHANNELS = tuple(i for i, c in enumerate(CHANNELS) if c in GUARDS)


# --------------------------------------------------------------------------- #
# backend helpers: the same code path runs on numpy arrays and torch tensors
# --------------------------------------------------------------------------- #
def _is_torch(x) -> bool:
    return isinstance(x, torch.Tensor)


def _clip(x, lo: float, hi: float):
    return x.clamp(lo, hi) if _is_torch(x) else np.clip(x, lo, hi)


def _log1p(x):
    return torch.log1p(x) if _is_torch(x) else np.log1p(x)


def _expm1(x):
    return torch.expm1(x) if _is_torch(x) else np.expm1(x)


def _as_vec(vec: Sequence[float], like):
    """Broadcast a per-channel constant to the channel axis of ``like``."""
    shape = [1] * like.ndim
    shape[-3] = len(vec)
    if _is_torch(like):
        return torch.as_tensor(list(map(float, vec)), dtype=like.dtype,
                               device=like.device).reshape(shape)
    return np.asarray(vec, dtype=np.float64).reshape(shape)


# --------------------------------------------------------------------------- #
# normalisation codec
# --------------------------------------------------------------------------- #
@dataclass
class StateCodec:
    """Physical <-> latent state mapping. ``mean``/``std`` are per channel.

    ``tp_log1p=True`` (default) standardises ``log1p(tp)``; the reverse transform
    is ``expm1`` of a clipped non-negative value, so rain can never be negative
    and a delta model can never produce fictitious dry air.
    """

    mean: np.ndarray
    std: np.ndarray
    tp_log1p: bool = True
    stats_source: str = "stats.json"

    def __post_init__(self) -> None:
        self.mean = np.asarray(self.mean, dtype=np.float64).reshape(NC)
        self.std = np.asarray(self.std, dtype=np.float64).reshape(NC)
        if not np.all(np.isfinite(self.mean)) or not np.all(self.std > 0):
            raise ValueError("codec mean must be finite and std strictly positive")

    # -- forward / inverse transforms -------------------------------------
    def _to_latent_units(self, phys):
        if not self.tp_log1p:
            return phys
        if _is_torch(phys):
            out = phys.clone()
            out[..., TP_INDEX, :, :] = torch.log1p(
                _clip(phys[..., TP_INDEX, :, :], 0.0, TP_MAX_MM_DAY))
            return out
        out = np.array(phys, copy=True)
        out[..., TP_INDEX, :, :] = _log1p(_clip(out[..., TP_INDEX, :, :], 0.0, TP_MAX_MM_DAY))
        return out

    def _from_latent_units(self, x):
        if not self.tp_log1p:
            return x
        if _is_torch(x):
            out = x.clone()
            out[..., TP_INDEX, :, :] = torch.expm1(
                _clip(x[..., TP_INDEX, :, :], 0.0, math.log1p(TP_MAX_MM_DAY)))
            return out
        out = np.array(x, copy=True)
        out[..., TP_INDEX, :, :] = _expm1(
            _clip(out[..., TP_INDEX, :, :], 0.0, math.log1p(TP_MAX_MM_DAY)))
        return out

    def encode(self, phys):
        """Physical state -> dimensionless latent state (float32)."""
        x = self._to_latent_units(phys)
        z = (x - _as_vec(self.mean, x)) / _as_vec(self.std, x)
        return z.to(torch.float32) if _is_torch(z) else np.asarray(z, dtype=np.float32)

    def decode(self, latent, guard: bool = True):
        """Latent state -> physical state, with the non-negativity/physics guards."""
        x = latent * _as_vec(self.std, latent) + _as_vec(self.mean, latent)
        x = self._from_latent_units(x)
        if not guard:
            return x.to(torch.float32) if _is_torch(x) else np.asarray(x, dtype=np.float32)
        for i, name in enumerate(CHANNELS):
            lo, hi = GUARDS.get(name, (None, None))
            if lo is None:
                continue
            if _is_torch(x):
                x[..., i, :, :] = x[..., i, :, :].clamp(lo, hi)
            else:
                x[..., i, :, :] = np.clip(x[..., i, :, :], lo, hi)
        return x.to(torch.float32) if _is_torch(x) else np.asarray(x, dtype=np.float32)

    def decode_tp(self, latent_tp):
        """Rain-only decode: exactly ``expm1(clip(z*std + mean, 0, ..))`` >= 0."""
        x = latent_tp * self.std[TP_INDEX] + self.mean[TP_INDEX]
        return _expm1(_clip(x, 0.0, math.log1p(TP_MAX_MM_DAY)))

    # -- construction ------------------------------------------------------
    @classmethod
    def from_stats(cls, data: np.ndarray, days: Sequence[date], stats_path: str | Path,
                   tp_log1p: bool = True,
                   split: tuple[date, date] = (TRAIN_START, TRAIN_END)) -> "StateCodec":
        """Build the codec: stats.json constants for linear channels, train log1p stats for tp."""
        stats_path = Path(stats_path)
        with stats_path.open() as fh:
            stats = json.load(fh)
        if tuple(stats["channels"]) != CHANNELS:
            raise ValueError(f"stats.json channel order {stats['channels']} != {CHANNELS}")
        mean = np.asarray(stats["mean"], dtype=np.float64).copy()
        std = np.asarray(stats["std"], dtype=np.float64).copy()
        if not np.all(std > 0):
            raise ValueError("stats.json std must be positive")
        if tp_log1p:
            idx = np.array([split[0] <= d <= split[1] for d in days], dtype=bool)
            rain = np.log1p(np.maximum(np.asarray(data)[idx, TP_INDEX], 0.0))
            mean[TP_INDEX] = float(rain.mean())
            std[TP_INDEX] = float(rain.std())
            if not std[TP_INDEX] > 0:
                raise ValueError("degenerate log1p(tp) statistics")
        return cls(mean=mean, std=std, tp_log1p=tp_log1p, stats_source=stats_path.name)

    # -- persistence -------------------------------------------------------
    def save(self, path: str | Path = DEFAULT_STATIC) -> Path:
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        np.savez_compressed(
            path,
            codec_mean=self.mean,
            codec_std=self.std,
            codec_tp_log1p=np.array(bool(self.tp_log1p)),
            codec_source=np.array(self.stats_source),
        )
        return path

    @classmethod
    def load(cls, path: str | Path) -> "StateCodec":
        with np.load(Path(path), allow_pickle=False) as z:
            return cls(mean=np.asarray(z["codec_mean"], dtype=np.float64),
                       std=np.asarray(z["codec_std"], dtype=np.float64),
                       tp_log1p=bool(z["codec_tp_log1p"]),
                       stats_source=str(z["codec_source"]))

    def describe(self) -> dict:
        return {
            "mean": {c: float(v) for c, v in zip(CHANNELS, self.mean)},
            "std": {c: float(v) for c, v in zip(CHANNELS, self.std)},
            "tp_transform": "log1p" if self.tp_log1p else "linear",
            "tp_mean_log1p": float(self.mean[TP_INDEX]),
            "tp_std_log1p": float(self.std[TP_INDEX]),
            "stats_source": self.stats_source,
        }


# --------------------------------------------------------------------------- #
# static fields
# --------------------------------------------------------------------------- #
@dataclass
class StaticFields:
    """[S, 32, 36] float32 static conditioning tensor bundled with the model."""

    tensor: np.ndarray
    names: tuple[str, ...] = ("land_sea_mask", "orography_m", "lat", "lon")
    orog_mean: float = 0.0
    orog_std: float = 1.0

    def __post_init__(self) -> None:
        self.tensor = np.ascontiguousarray(self.tensor, dtype=np.float32)
        if self.tensor.ndim != 3 or self.tensor.shape[1:] != (NLAT, NLON):
            raise ValueError(f"bad static tensor shape {self.tensor.shape}")
        if self.tensor.shape[0] != len(self.names):
            raise ValueError("static tensor planes != names")

    @property
    def n_planes(self) -> int:
        return int(self.tensor.shape[0])

    @classmethod
    def from_netcdf(cls, path: str | Path = "india_climate_train.nc") -> "StaticFields":
        import xarray as xr

        with xr.open_dataset(Path(path)) as ds:
            raw = np.asarray(ds["static"].values, dtype=np.float32)
            names = tuple(str(s) for s in ds["static_channel"].values)
        if names[:2] != ("land_sea_mask", "orography_m"):
            raise ValueError(f"unexpected static channels {names}")
        lsm = raw[0]                              # land fraction in [0, 1] (4-point mean)
        orog = raw[1]
        orog_mean = float(orog.mean())
        orog_std = float(orog.std()) or 1.0
        orog_norm = (orog - orog_mean) / orog_std
        lat_norm = (LAT - LAT.mean()) / (LAT.max() - LAT.min()) * 2.0
        lon_norm = (LON - LON.mean()) / (LON.max() - LON.min()) * 2.0
        lat_plane = np.repeat(lat_norm[:, None], NLON, axis=1)
        lon_plane = np.repeat(lon_norm[None, :], NLAT, axis=0)
        tensor = np.stack([lsm, orog_norm, lat_plane, lon_plane]).astype(np.float32)
        return cls(tensor=tensor,
                   names=("land_sea_mask", "orography_m", "lat", "lon"),
                   orog_mean=orog_mean, orog_std=orog_std)

    def save(self, path: str | Path = DEFAULT_STATIC, codec: StateCodec | None = None) -> Path:
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        payload = dict(
            static=self.tensor,
            static_names=np.array(self.names),
            static_orog_mean=np.array(self.orog_mean),
            static_orog_std=np.array(self.orog_std),
        )
        if codec is not None:
            payload.update(
                codec_mean=codec.mean,
                codec_std=codec.std,
                codec_tp_log1p=np.array(bool(codec.tp_log1p)),
                codec_source=np.array(codec.stats_source),
            )
        np.savez_compressed(path, **payload)
        return path

    @classmethod
    def load(cls, path: str | Path = DEFAULT_STATIC) -> "StaticFields":
        with np.load(Path(path), allow_pickle=False) as z:
            return cls(tensor=np.asarray(z["static"], dtype=np.float32),
                       names=tuple(str(s) for s in z["static_names"]),
                       orog_mean=float(z["static_orog_mean"]),
                       orog_std=float(z["static_orog_std"]))

    @property
    def n_bytes(self) -> int:
        return int(self.tensor.size * self.tensor.dtype.itemsize)


def load_static_codec(path: str | Path = DEFAULT_STATIC) -> tuple[StaticFields, StateCodec]:
    """Load the bundled static planes and state codec from one artifact."""
    static = StaticFields.load(path)
    with np.load(Path(path), allow_pickle=False) as z:
        if "codec_mean" not in z:
            raise ValueError(f"{path} has no codec arrays; rebuild with `python dataset.py build`")
        codec = StateCodec(mean=np.asarray(z["codec_mean"]), std=np.asarray(z["codec_std"]),
                           tp_log1p=bool(z["codec_tp_log1p"]),
                           stats_source=str(z["codec_source"]))
    return static, codec


# --------------------------------------------------------------------------- #
# NetCDF loading
# --------------------------------------------------------------------------- #
def load_netcdf(path: str | Path = "india_climate_train.nc") -> tuple[np.ndarray, list[date]]:
    """Read ``data`` [time, 6, 32, 36] float32 physical units and its calendar."""
    import xarray as xr

    path = Path(path)
    with xr.open_dataset(path) as ds:
        order = tuple(str(c) for c in ds["channel"].values)
        if order != CHANNELS:
            raise ValueError(f"channel order {order} != {CHANNELS}")
        data = np.asarray(ds["data"].values, dtype=np.float32)
        days = list(ds["time"].values.astype("datetime64[D]").tolist())
    if data.shape != (len(days), NC, NLAT, NLON):
        raise ValueError(f"unexpected data shape {data.shape}")
    gaps = np.diff(np.array(days, dtype="datetime64[D]").astype(np.int64))
    if len(gaps) and not np.all(gaps == 1):
        raise ValueError("dataset is not gap-free daily data")
    if not np.isfinite(data).all():
        raise ValueError("dataset contains NaN/Inf")
    return data, days


# --------------------------------------------------------------------------- #
# sliding-window dataset
# --------------------------------------------------------------------------- #
def doy_basis(theta: np.ndarray) -> np.ndarray:
    """[.., 4] = [sin t, cos t, sin 2t, cos 2t] — the day-of-year embedding."""
    s, c = np.sin(theta), np.cos(theta)
    return np.stack([s, c, 2.0 * s * c, c * c - s * s], axis=-1)


class ClimateWindows(Dataset):
    """Three input days -> ``horizon`` autoregressive days.

    Each item contains
      hist     [3, 6, 32, 36]  latent states of the last three observed days
      anom     [6, 32, 36]     latent anomaly of the last observed day
                               (latent state minus latent harmonic climatology)
      anom_seq [horizon+1, 6, 32, 36]  latent anomaly of days t .. t+horizon,
                               so teacher-forced step k can use ``anom_seq[k]``
      static   [S, 32, 36]     bundled static planes
      doy      [8, horizon]    [sin t, cos t, sin 2t, cos 2t] of the source day and
                               of the target day of every step (contract: step k
                               is the day after the source day, i.e. doy + k)
      target   [horizon, 6, 32, 36] latent absolute future states
      delta    [horizon, 6, 32, 36] latent one-day increments
      init_index, init_doy     int64 scalars (debug / provenance only)

    The delta parameterisation is explicit: ``delta[k] = target[k] - prev`` with
    ``prev = hist[-1]`` for ``k = 0``, so ``hist[-1] + delta.cumsum(0) == target``.
    A one-step-ahead training loop instead uses ``anom_seq``/``delta`` for every
    lead at once (teacher forcing), which is what makes 10-lead price per window.
    """

    def __init__(
        self,
        data: np.ndarray,
        days: Sequence[date],
        codec: StateCodec,
        climatology: HarmonicClimatology,
        static: StaticFields,
        split: str = "train",
        horizon: int = DEFAULT_HORIZON,
        stride: int = 1,
        starts: np.ndarray | None = None,
    ) -> None:
        if split not in SPLITS:
            raise ValueError(f"split {split!r} not in {sorted(SPLITS)}")
        if horizon < 1:
            raise ValueError("horizon must be >= 1")
        if stride < 1:
            raise ValueError("stride must be >= 1")
        self.data = np.asarray(data)
        if self.data.shape != (len(days), NC, NLAT, NLON):
            raise ValueError(f"bad data shape {self.data.shape}")
        self.days = list(days)
        self.codec = codec
        self.climatology = climatology
        self.static = torch.from_numpy(static.tensor)
        self.horizon = int(horizon)
        self.stride = int(stride)
        self.split = split
        self.start_date, self.end_date = SPLITS[split]

        # theta (contract phase) of every available day, precomputed once
        self._theta = theta_from_doy(np.array([day_of_year(d) for d in self.days],
                                              dtype=np.float64))
        self._doy_basis = doy_basis(self._theta).astype(np.float32)   # [n_days, 4]
        # latent climatology for every day-of-year: one 10 MB table shared by all
        # samples, so anomalies are a lookup instead of an einsum per sample.
        doys = np.arange(1, 367, dtype=np.float64)
        self.clim_latent = codec.encode(climatology.predict_theta(theta_from_doy(doys)))
        self.clim_latent.setflags(write=False)

        self._in_split = np.array([self.start_date <= d <= self.end_date for d in self.days],
                                  dtype=bool)
        if starts is None:
            starts = self._build_starts()
        self.starts = np.asarray(starts, dtype=np.int64)

    # -- window bookkeeping ------------------------------------------------
    def _build_starts(self) -> np.ndarray:
        span = IN_STEPS - 1 + self.horizon            # input days + forecast days
        last = len(self.days) - 1 - span
        if last < 0:
            return np.empty(0, dtype=np.int64)
        cand = np.arange(0, last + 1, self.stride, dtype=np.int64)
        # every input day and every target day must belong to the split, so no
        # window ever trains on (or validates against) days outside its period.
        ok = np.array([self._in_split[i] and self._in_split[i + span] for i in cand],
                      dtype=bool)
        return cand[ok]

    def __len__(self) -> int:
        return int(self.starts.size)

    def window_dates(self, item: int) -> tuple[date, list[date]]:
        i0 = int(self.starts[item])
        init_last = self.days[i0 + IN_STEPS - 1]
        targets = [self.days[i0 + IN_STEPS + k] for k in range(self.horizon)]
        return init_last, targets

    # -- one sample --------------------------------------------------------
    def __getitem__(self, item: int) -> dict[str, torch.Tensor]:
        i0 = int(self.starts[item])
        i1 = i0 + IN_STEPS                            # first forecast day index
        hist_phys = self.data[i0:i0 + IN_STEPS]       # [3, 6, 32, 36]
        fut_phys = self.data[i1:i1 + self.horizon]    # [K, 6, 32, 36]

        hist = self.codec.encode(hist_phys)                       # latent [3,6,H,W]
        target = self.codec.encode(fut_phys)                      # latent [K,6,H,W]
        delta = np.diff(np.concatenate([hist[-1:], target], axis=0), axis=0)

        # anomalies of days t .. t+horizon against the harmonic climatology of
        # each day, in latent space (well defined for log1p(tp)).
        states = np.concatenate([hist, target], axis=0)             # [3+K, 6, H, W]
        clim_seq = np.stack([self.clim_latent[day_of_year(d) - 1]
                             for d in self.days[i0 + IN_STEPS - 1:i1 + self.horizon]])
        anom_seq = states[IN_STEPS - 1:] - clim_seq                 # [K+1, 6, H, W]

        # doy embedding: source day (4) then target day (4) of every step
        bases = np.asarray(self._doy_basis[i1:i1 + self.horizon], dtype=np.float32)  # [K,4]
        doy = np.concatenate([
            np.repeat(self._doy_basis[i0 + IN_STEPS - 1][None], self.horizon, axis=0),
            bases,
        ], axis=1).T.astype(np.float32)                            # [8, K]

        return {
            "hist": torch.from_numpy(np.ascontiguousarray(hist)),
            "anom": torch.from_numpy(np.ascontiguousarray(anom_seq[0])),
            "anom_seq": torch.from_numpy(np.ascontiguousarray(anom_seq)),
            "static": self.static,
            "doy": torch.from_numpy(np.ascontiguousarray(doy)),
            "target": torch.from_numpy(np.ascontiguousarray(target)),
            "delta": torch.from_numpy(np.ascontiguousarray(delta)),
            "init_index": torch.tensor(i0, dtype=torch.int64),
            "init_doy": torch.tensor(day_of_year(self.days[i0 + IN_STEPS - 1]),
                                     dtype=torch.int64),
        }

    # -- convenience for evaluation code ----------------------------------
    def physical_truth(self, item: int) -> tuple[np.ndarray, np.ndarray]:
        """(init_states [3,6,H,W], future [K,6,H,W]) in physical units."""
        i0 = int(self.starts[item])
        i1 = i0 + IN_STEPS
        return (self.data[i0:i1].copy(),
                self.data[i1:i1 + self.horizon].copy())

    def describe(self) -> dict:
        last, targets = self.window_dates(0) if len(self) else (None, [])
        return {
            "split": self.split,
            "period": [self.start_date.isoformat(), self.end_date.isoformat()],
            "n_windows": len(self),
            "stride_days": self.stride,
            "horizon": self.horizon,
            "in_steps": IN_STEPS,
            "first_window": {"init_last": last.isoformat() if last else None,
                             "first_target": targets[0].isoformat() if targets else None,
                             "last_target": targets[-1].isoformat() if targets else None},
            "item_shapes": {"hist": [IN_STEPS, NC, NLAT, NLON],
                            "anom": [NC, NLAT, NLON],
                            "anom_seq": [self.horizon + 1, NC, NLAT, NLON],
                            "static": list(self.static.shape),
                            "doy": [8, self.horizon],
                            "target": [self.horizon, NC, NLAT, NLON],
                            "delta": [self.horizon, NC, NLAT, NLON]},
            "clim_latent_table_bytes": int(self.clim_latent.nbytes),
        }


def make_loader(dataset: ClimateWindows, batch_size: int = 32, shuffle: bool = True,
                num_workers: int = 0, drop_last: bool = False) -> DataLoader:
    """DataLoader with fork-friendly defaults (the 171 MB array is shared, not copied)."""
    return DataLoader(dataset, batch_size=batch_size, shuffle=shuffle,
                      num_workers=num_workers, drop_last=drop_last,
                      pin_memory=False, persistent_workers=num_workers > 0)


def build_datasets(
    data_path: str | Path = "india_climate_train.nc",
    climatology_path: str | Path = DEFAULT_CLIMATOLOGY,
    static_codec_path: str | Path = DEFAULT_STATIC,
    horizon: int = DEFAULT_HORIZON,
    train_stride: int = 1,
    val_stride: int = 2,
) -> tuple[ClimateWindows, ClimateWindows]:
    """Load the artifacts once and return the (train, val) window datasets."""
    data, days = load_netcdf(data_path)
    static, codec = load_static_codec(static_codec_path)
    climatology = HarmonicClimatology.load(climatology_path)
    train = ClimateWindows(data, days, codec, climatology, static,
                           split="train", horizon=horizon, stride=train_stride)
    val = ClimateWindows(data, days, codec, climatology, static,
                         split="val", horizon=horizon, stride=val_stride)
    return train, val


# --------------------------------------------------------------------------- #
# CLI
# --------------------------------------------------------------------------- #
def _cmd_build(args) -> int:
    data, days = load_netcdf(args.data)
    codec = StateCodec.from_stats(data, days, args.stats)
    static = StaticFields.from_netcdf(args.data)
    out = static.save(args.out, codec=codec)
    climatology = HarmonicClimatology.load(args.climatology)

    train = ClimateWindows(data, days, codec, climatology, static, split="train",
                           horizon=args.horizon, stride=args.stride)
    val = ClimateWindows(data, days, codec, climatology, static, split="val",
                         horizon=args.horizon, stride=args.stride)
    item = train[0]
    meta = {
        "artifact": str(out),
        "artifact_bytes": out.stat().st_size,
        "codec": codec.describe(),
        "static_planes": list(static.names),
        "static_bytes": static.n_bytes,
        "train": train.describe(),
        "val": val.describe(),
        "sample_shapes": {k: list(v.shape) for k, v in item.items()
                          if isinstance(v, torch.Tensor)},
        "sample_tp_physical_min": float(
            codec.decode(item["target"])[:, TP_INDEX].min()),
        "sample_target_abs_max_latent": float(item["target"].abs().max()),
    }
    print(json.dumps(meta, indent=2))
    if args.json:
        Path(args.json).write_text(json.dumps(meta, indent=2) + "\n")
    return 0


def main(argv: Iterable[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = ap.add_subparsers(dest="cmd", required=True)
    b = sub.add_parser("build", help="fit the codec and write the static/codec artifact")
    b.add_argument("--data", default="india_climate_train.nc")
    b.add_argument("--stats", default="stats.json")
    b.add_argument("--climatology", default=str(DEFAULT_CLIMATOLOGY))
    b.add_argument("--out", default=str(DEFAULT_STATIC))
    b.add_argument("--horizon", type=int, default=DEFAULT_HORIZON)
    b.add_argument("--stride", type=int, default=1)
    b.add_argument("--json", default="")
    b.set_defaults(func=_cmd_build)
    args = ap.parse_args(list(argv) if argv is not None else None)
    return int(args.func(args))


if __name__ == "__main__":
    raise SystemExit(main())
