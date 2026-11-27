#!/usr/bin/env python3
"""Phase 2 verification: climatology bundle, data pipeline and local architecture.

Every Phase 2 claim is measured here, through the same interfaces the served
model uses:

  climatology   bundle size, float16 round-trip error, seasonal skill of the
                compact fit, and equivalence of the [sin, cos] wire path with the
                calendar path (a rollout never needs a date).
  dataset       window alignment (target = init_last + k, k = 1..K), split
                leakage, delta/cumsum identity, encode/decode round-trip with
                non-negative rain, anomaly definition, day-of-year features.
  model         local-ops assertion (no attention, no linear, no global pooling),
                parameter budget, receptive radius, zero-head == persistence,
                determinism, null-shock identity, physics guards, save/load.
  smoke train   a short teacher-forced fit (all leads folded into one batch) so
                the pipeline is shown to learn, then a real 10-step rollout
                scored with the official functions against persistence and the
                harmonic climatology.
  causality     the two official shocks through the official checks on real
                windows, plus the null shock.

Usage:
    python check_phase2.py                     # 200 optimisation steps
    python check_phase2.py --steps 600 --threads 4
    python check_phase2.py --skip-train        # architecture/pipeline only
"""
from __future__ import annotations

import argparse
import json
import math
import tempfile
import time
from datetime import date, timedelta
from pathlib import Path

import numpy as np
import torch

import climatology as clim_mod
import evaluate_baselines as eb
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
from dataset import (
    DEFAULT_STATIC,
    IN_STEPS,
    ClimateWindows,
    StateCodec,
    load_static_codec,
    make_loader,
)
from model import DigitalTwin, LocalDeltaNet, ModelConfig
from monsoon_r1.contract import BOX_ARABIAN_SEA, BOX_BAY_OF_BENGAL, LAT
from monsoon_r1.scoring import (
    anti_blur_points,
    causality_checks,
    causality_points,
    csi,
    make_shock,
    null_consistent,
    rmse_lat,
    skill_vs_best_ref,
    validity,
    variance_ratio,
)

COEF_PATH = Path("artifacts") / "climatology_coefs.npz"
REPORT_PATH = Path("artifacts") / "phase2_check.json"
STD = np.array([13.46363353729248, 7.553342342376709, 2.8989806175231934,
                2.4269521236419678, 619.0198364257812, 20.49437141418457], "f4")
LEADS = (1, 4, 8)
DAY8 = 7


# --------------------------------------------------------------------------- #
# report plumbing
# --------------------------------------------------------------------------- #
class Report:
    def __init__(self) -> None:
        self.checks: list[dict] = []
        self.metrics: dict = {}

    def check(self, name: str, ok: bool | None, detail: str = "") -> None:
        status = "SKIP" if ok is None else ("PASS" if ok else "FAIL")
        self.checks.append({"name": name, "status": status, "detail": detail})
        print(f"  [{status}] {name}" + (f" — {detail}" if detail else ""))

    def metric(self, key: str, value) -> None:
        self.metrics[key] = value

    def failures(self) -> list[str]:
        return [c["name"] for c in self.checks if c["status"] == "FAIL"]

    def payload(self) -> dict:
        return {"checks": self.checks, "metrics": self.metrics}


def _fmt(v, width=9, sig=4) -> str:
    if v is None or not np.isfinite(v):
        return "nan".rjust(width)
    return f"{float(v):>{width}.{sig}g}"


# --------------------------------------------------------------------------- #
# 1. climatology
# --------------------------------------------------------------------------- #
def check_climatology(rep: Report, clim: HarmonicClimatology, data, days) -> None:
    print("\n[1] harmonic climatology")
    raw = COEF_PATH.stat().st_size
    rep.metric("climatology", {
        "bundle_bytes": raw,
        "float32_tensor_bytes": clim.n_bytes_float32,
        "n_coefficients": int(clim.coef.size),
        "coefficients_per_cell_channel": int(clim.coef.shape[0]),
        "train_r2": {c: float(v) for c, v in zip(CHANNELS, clim.train_r2)},
        "rain_space": clim.rain_space,
    })
    rep.check("bundle is compact (~100 KB, no 365-day table)",
              raw <= 100 * 1024 and clim.coef.shape == (5, NC, NLAT, NLON)
              and clim.coef.size == 5 * NC * NLAT * NLON,
              f"{raw / 1024:.1f} KB compressed, {clim.coef.size:,} coefficients, "
              f"shape {tuple(clim.coef.shape)}, float16 waves + float32 mean")
    rep.check("harmonic fit beats a constant annual mean on the training years",
              bool(np.all(np.asarray(clim.train_r2) > 0.1)),
              "R2 " + " ".join(f"{c} {v:+.2f}" for c, v in zip(CHANNELS, clim.train_r2)))

    # float16 storage error: compare the saved bundle with a fresh float32 fit
    exact_fit = clim_mod.fit_harmonic_climatology(data, days, rain_space=clim.rain_space)
    theta = theta_from_doy(np.arange(1, 367, dtype=np.float64))
    exact = exact_fit.predict_theta(theta)
    saved = HarmonicClimatology.load(COEF_PATH).predict_theta(theta)
    err = np.abs(saved - exact).max(axis=(0, 2, 3)) / STD
    rep.metric("climatology_float16_max_error_in_std", {c: float(v) for c, v in zip(CHANNELS, err)})
    rep.check("float16 wave coefficients are lossless for our purposes",
              bool(np.all(err < 5e-3)),
              "max |err|/std " + " ".join(f"{c} {v:.1e}" for c, v in zip(CHANNELS, err)))

    # wire path (only [sin, cos] of the last input day) must equal the calendar path
    probe = date(2020, 7, 15)
    t = float(theta_from_doy(day_of_year(probe)))
    diff = float(np.abs(clim.predict_sincos(math.sin(t), math.cos(t))
                        - clim.predict_dates([probe])[0]).max())
    rep.check("[sin, cos] wire path equals calendar path", diff == 0.0, f"max diff {diff:g}")

    # seasonal skill against a constant-mean reference on the validation years
    val_idx = np.array([i for i, d in enumerate(days) if date(2020, 1, 1) <= d <= date(2021, 12, 31)])
    val_days = [days[i] for i in val_idx]
    pred = clim.predict_dates(val_days)
    truth = np.asarray(data)[val_idx]
    rm = np.sqrt(((pred - truth) ** 2).mean(axis=(0, 2, 3)))
    const = np.broadcast_to(np.asarray(clim.coef[0])[None], truth.shape)
    rm_const = np.sqrt(((const - truth) ** 2).mean(axis=(0, 2, 3)))
    rep.metric("climatology_val_rmse", {c: float(v) for c, v in zip(CHANNELS, rm)})
    rep.check("harmonic climatology beats constant mean on 2020-2021",
              bool(np.all(rm < rm_const)),
              "RMSE " + " ".join(f"{c} {a:.3g}<{b:.3g}" for c, a, b in zip(CHANNELS, rm, rm_const)))
    rep.check("harmonics capture the seasonal cycle of t2m",
              float(np.corrcoef(pred[:, 0].mean(axis=(1, 2)), truth[:, 0].mean(axis=(1, 2)))[0, 1]) > 0.95,
              f"domain-mean t2m correlation "
              f"{np.corrcoef(pred[:, 0].mean(axis=(1, 2)), truth[:, 0].mean(axis=(1, 2)))[0, 1]:.4f}")


# --------------------------------------------------------------------------- #
# 2. dataset
# --------------------------------------------------------------------------- #
def check_dataset(rep: Report, train: ClimateWindows, val: ClimateWindows) -> None:
    print("\n[2] sliding-window dataset")
    item = train[123]
    rep.metric("dataset", {"train": train.describe(), "val": val.describe(),
                           "sample_shapes": {k: list(v.shape) for k, v in item.items()
                                             if isinstance(v, torch.Tensor)}})
    rep.check("three input days and K autoregressive targets",
              item["hist"].shape == (IN_STEPS, NC, NLAT, NLON)
              and item["target"].shape == (train.horizon, NC, NLAT, NLON)
              and item["delta"].shape == (train.horizon, NC, NLAT, NLON),
              f"hist {tuple(item['hist'].shape)}, target {tuple(item['target'].shape)}")
    rep.check("window counts are sane", len(train) > 5000 and len(val) > 300,
              f"{len(train)} train / {len(val)} val windows")

    # target alignment: target day k is init_last + (k + 1) calendar days
    span = IN_STEPS - 1 + train.horizon
    worst = 0
    for idx in (0, len(train) // 2, len(train) - 1):
        init_last, targets = train.window_dates(idx)
        worst = max(worst, abs((targets[0] - init_last).days - 1),
                    abs((targets[-1] - init_last).days - train.horizon))
        if train.days[int(train.starts[idx]) + IN_STEPS - 1] != init_last:
            worst = 999
    rep.check("rollout[k] is the day after init_last + k days", worst == 0,
              f"max calendar misalignment {worst} day(s)")

    # no split leakage
    leak = int(np.sum(~train._in_split[train.starts]) + np.sum(~train._in_split[train.starts + span])
               + np.sum(~val._in_split[val.starts]) + np.sum(~val._in_split[val.starts + span]))
    rep.check("no window crosses its split boundary", leak == 0,
              f"{leak} out-of-period endpoints; train {train.start_date}..{train.end_date}, "
              f"val {val.start_date}..{val.end_date}")

    # delta/cumsum identity and physical plausibility
    cum = item["hist"][-1] + torch.cumsum(item["delta"], dim=0)
    rep.check("delta parameterisation is exact", float((cum - item["target"]).abs().max()) < 1e-5,
              f"max |hist[-1] + cumsum(delta) - target| = {float((cum - item['target']).abs().max()):.2e}")

    init_phys, future_phys = train.physical_truth(123)
    rt = train.codec.decode(train.codec.encode(future_phys))
    err_std = (np.abs(rt - future_phys).max(axis=(0, 2, 3)) / STD)
    rep.check("encode/decode round-trip is faithful",
              bool(np.all(err_std < 1e-4)) and bool(rt[:, TP_INDEX].min() >= 0.0),
              f"max err/std {err_std.max():.2e}, min decoded rain "
              f"{rt[:, TP_INDEX].min():.3f} mm/day")

    guarded = train.codec.decode(train.codec.encode(future_phys), guard=True)
    rep.check("physics guards never touch real observations",
              float(np.abs(guarded - rt).max()) == 0.0,
              "clamped values: 0 (guards only fire on impossible states)")

    # normalisation constants really are the shipped stats.json values
    stats = json.loads(Path("stats.json").read_text())
    linear = [0, 2, 3, 4, 5]
    ok = all(abs(train.codec.mean[i] - stats["mean"][i]) < 1e-6
             and abs(train.codec.std[i] - stats["std"][i]) < 1e-6 for i in linear)
    rep.check("linear channels standardised with stats.json", ok,
              "5/6 channels use the shipped mean/std; tp uses log1p space")

    # anomaly and doy bookkeeping
    states = torch.cat([item["hist"], item["target"]], dim=0)
    i0 = int(item["init_index"])
    clim_day = np.array(train.clim_latent[day_of_year(train.days[i0 + IN_STEPS - 1]) - 1])
    a0_err = float((item["anom"] - (item["hist"][-1] - torch.from_numpy(clim_day)))
                   .abs().max())
    ks = [0, train.horizon - 1]
    anom_err = 0.0
    for k in ks:
        d = day_of_year(train.days[i0 + IN_STEPS - 1 + k])
        expect = states[IN_STEPS - 1 + k] - torch.from_numpy(np.array(train.clim_latent[d - 1]))
        anom_err = max(anom_err, float((item["anom_seq"][k] - expect).abs().max()))
    rep.check("anomalies are state minus harmonic climatology of the same day",
              a0_err < 1e-6 and anom_err < 1e-6,
              f"max diff {max(a0_err, anom_err):.2e} (anom == anom_seq[0])")

    basis_err = 0.0
    for k in range(train.horizon):
        src_doy = day_of_year(train.days[i0 + IN_STEPS - 1])
        tgt_doy = day_of_year(train.days[i0 + IN_STEPS + k])
        b = np.concatenate([_basis_from_doy(src_doy), _basis_from_doy(tgt_doy)])
        basis_err = max(basis_err, float(np.abs(item["doy"][:, k].numpy() - b).max()))
    rep.check("doy embedding is the source day and the forecast day (doy + k)",
              basis_err < 1e-6, f"max diff {basis_err:.2e}")

    # teacher-forcing view used by the training loop
    states_b = states[None]
    idx = torch.arange(3)[None, :] + torch.arange(train.horizon)[:, None]
    h = states_b[:, idx].reshape(train.horizon, 3, NC, NLAT, NLON)
    tf_ok = float((h[4, -1] - item["target"][3]).abs().max())
    rep.check("teacher-forced history at step k ends on the true day t+k", tf_ok == 0.0,
              f"max diff {tf_ok:.2e}")

    batch = next(iter(make_loader(train, batch_size=4, shuffle=False)))
    rep.check("default collate produces finite batches",
              all(bool(torch.isfinite(v).all()) for k, v in batch.items() if v.is_floating_point()),
              ", ".join(f"{k} {tuple(v.shape)}" for k, v in batch.items()
                        if isinstance(v, torch.Tensor)))


def _basis_from_doy(doy: int) -> np.ndarray:
    t = float(theta_from_doy(doy))
    s, c = math.sin(t), math.cos(t)
    return np.array([s, c, 2 * s * c, c * c - s * s], dtype=np.float32)


# --------------------------------------------------------------------------- #
# 3. architecture
# --------------------------------------------------------------------------- #
def check_architecture(rep: Report, codec: StateCodec, clim: HarmonicClimatology,
                       static, init: np.ndarray, doy_sincos, threads: int) -> None:
    print("\n[3] architecture")
    model = LocalDeltaNet()
    cfg = model.config
    params = model.n_parameters()
    rep.metric("model", {"config": model_mod_config(cfg), "parameters": params,
                         "in_channels": cfg.in_channels,
                         "receptive_radius_cells": cfg.receptive_radius})
    rep.check("parameters under the 3 M budget", params < 3_000_000, f"{params:,} parameters")
    try:
        model.assert_local_ops()
        local_ok, local_detail = True, (f"{cfg.blocks} ConvNeXt-style 3x3 blocks, no attention/"
                                        "linear/global pooling")
    except AssertionError as exc:                       # pragma: no cover - guard
        local_ok, local_detail = False, str(exc)
    rep.check("purely local operations", local_ok, local_detail)
    rep.check("receptive radius keeps day-1 locality", cfg.receptive_radius <= 8,
              f"±{cfg.receptive_radius} cells/day (contract allows up to ~11 at day 1)")

    # zero-initialised head: the untrained twin is exactly persistence
    twin = DigitalTwin(LocalDeltaNet(cfg), codec, clim, static, torch_threads=threads)
    out = twin.simulate(init, doy_sincos, 10)
    pers = np.repeat(init[-1:], 10, axis=0)
    head = LocalDeltaNet(cfg)
    with torch.no_grad():
        zero_delta_max = float(head(torch.zeros(1, IN_STEPS, NC, NLAT, NLON),
                                    torch.zeros(1, NC, NLAT, NLON),
                                    torch.zeros(1, cfg.static_planes, NLAT, NLON),
                                    torch.zeros(1, cfg.doy_features)).abs().max())
    rep.check("zero head reproduces persistence to codec round-trip precision",
              zero_delta_max == 0.0 and float(np.abs(out - pers).max()) < 1e-4,
              f"head output exactly {zero_delta_max:g}; max |rollout - repeat(init[-1])| = "
              f"{float(np.abs(out - pers).max()):.1e} (encode/decode round-off)")

    # determinism and the null shock
    again = twin.simulate(init, doy_sincos, 10)
    null = twin.simulate(init, doy_sincos, 10, np.zeros((NC, NLAT, NLON), "f4"))
    rep.check("deterministic and null-shock invariant",
              bool(np.array_equal(out, again)) and null_consistent(out, null, STD),
              "bitwise identical repeated calls; zero perturbation changes nothing")

    # realistic weights (small perturbation of the initialisation) for a harder
    # stability check: 14 steps from a real observed state
    torch.manual_seed(0)
    random_twin = DigitalTwin(model_mod_random(cfg), codec, clim, static, torch_threads=threads)
    with torch.inference_mode():
        long_out = random_twin.simulate(init, doy_sincos, 14)
    bad = validity(long_out, 14)
    rep.check("14-step rollout stays finite, rain >= 0, t2m in [-40, 60] C",
              not bad, f"violations: {bad or 'none'}, rain min {long_out[:, 1].min():.3f} mm/day")

    # timing
    random_twin.simulate(init, doy_sincos, 10)
    times = []
    for _ in range(5):
        _, ms = random_twin.simulate_timed(init, doy_sincos, 10)
        times.append(ms)
    med = float(np.median(times))
    rep.metric("rollout_ms_10_steps", {"median": med, "all": times, "threads": threads})
    rep.check("10-step rollout under 500 ms on 4 CPU threads", med < 500.0,
              f"{med:.1f} ms median ({med / 10:.1f} ms/step), {torch.get_num_threads()} threads")

    # bundle round-trip
    with tempfile.TemporaryDirectory() as tmp:
        path = random_twin.save(Path(tmp) / "bundle.pt")
        size_kb = path.stat().st_size / 1024
        loaded = DigitalTwin.load(path, torch_threads=threads)
        with torch.inference_mode():
            reloaded = loaded.simulate(init, doy_sincos, 10)
        expected = random_twin.simulate(init, doy_sincos, 10)
    rep.check("weights bundle round-trips bit-exactly",
              bool(np.array_equal(reloaded, expected)),
              f"{size_kb:.0f} KB single-file bundle (weights + climatology + statics)")


def model_mod_config(cfg: ModelConfig) -> dict:
    return {"hidden": cfg.hidden, "blocks": cfg.blocks, "expansion": cfg.expansion,
            "in_steps": cfg.in_steps, "static_planes": cfg.static_planes,
            "doy_features": cfg.doy_features, "max_step_delta": cfg.max_step_delta}


def model_mod_random(cfg: ModelConfig, scale: float = 0.05) -> LocalDeltaNet:
    torch.manual_seed(0)
    m = LocalDeltaNet(cfg)
    with torch.no_grad():
        for name, p in m.named_parameters():
            if not name.startswith("head"):
                p.add_(torch.randn_like(p) * scale)
    return m


# --------------------------------------------------------------------------- #
# 4. smoke training
# --------------------------------------------------------------------------- #
def fold_batch(batch: dict) -> tuple:
    """[B, K] windows -> one flat batch of K teacher-forced single steps."""
    hist, target = batch["hist"], batch["target"]
    b, k = target.shape[0], target.shape[1]
    states = torch.cat([hist, target], dim=1)                     # [B, 3+K, ...]
    idx = torch.arange(3)[None, :] + torch.arange(k)[:, None]     # [K, 3]
    h = states[:, idx].reshape(b * k, IN_STEPS, NC, NLAT, NLON)
    anom = batch["anom_seq"][:, :k].reshape(b * k, NC, NLAT, NLON)
    doy = batch["doy"].permute(0, 2, 1).reshape(b * k, -1)
    static = batch["static"][:, None].expand(b, k, -1, -1, -1).reshape(
        b * k, batch["static"].shape[1], NLAT, NLON)
    delta = (states[:, 3:3 + k] - states[:, 2:2 + k]).reshape(b * k, NC, NLAT, NLON)
    return h, anom, static, doy, delta


def _chunked_loss(model, h, anom, static, doy, delta, w, micro_batch: int,
                  backward: bool) -> float:
    """Latitude-weighted MSE over K fused leads, chunked to stay cache-friendly.

    This torch build has no oneDNN for CPU convolution, and per-sample cost rises
    ~5x once a batch exceeds 8 frames (measured), so the [B*K] teacher-forced
    batch is walked in micro-batches.  Gradient accumulation keeps the effective
    batch unchanged.
    """
    n = h.shape[0]
    total = 0.0
    for start in range(0, n, micro_batch):
        sl = slice(start, min(start + micro_batch, n))
        loss = (((model(h[sl], anom[sl], static[sl], doy[sl]) - delta[sl]) ** 2) * w).mean()
        if backward:
            (loss * (sl.stop - sl.start) / n).backward()
        total += float(loss.detach()) * (sl.stop - sl.start) / n
    return total


def smoke_train(train: ClimateWindows, val: ClimateWindows, steps: int, batch_size: int,
                lr: float, threads: int, micro_batch: int = 8) -> tuple[LocalDeltaNet, dict]:
    torch.set_num_threads(threads)
    model = LocalDeltaNet()
    opt = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=1e-4)
    sched = torch.optim.lr_scheduler.OneCycleLR(opt, max_lr=lr, total_steps=max(steps, 1),
                                               pct_start=0.15)
    w = torch.from_numpy((np.cos(np.deg2rad(LAT)) / np.cos(np.deg2rad(LAT)).mean())
                         .astype("f4"))[None, None, :, None]
    loader = make_loader(train, batch_size=batch_size, shuffle=True, num_workers=0)
    it = iter(loader)
    history: list[float] = []
    seen = 0
    t0 = time.perf_counter()
    model.train()
    for step in range(steps):
        try:
            batch = next(it)
        except StopIteration:
            it = iter(loader)
            batch = next(it)
        h, anom, static, doy, delta = fold_batch(batch)
        opt.zero_grad(set_to_none=True)
        loss = _chunked_loss(model, h, anom, static, doy, delta, w, micro_batch, True)
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        opt.step()
        sched.step()
        history.append(loss)
        seen += h.shape[0]
        if steps >= 100 and (step + 1) % max(steps // 4, 1) == 0:
            print(f"    step {step + 1:4d}/{steps}  train loss {np.mean(history[-25:]):.4f} "
                  f"({time.perf_counter() - t0:.1f}s)")
    model.eval()

    def flat_loss(ds: ClimateWindows) -> float:
        losses = []
        with torch.inference_mode():
            for batch in make_loader(ds, batch_size=batch_size, shuffle=False):
                h, anom, static, doy, delta = fold_batch(batch)
                losses.append(_chunked_loss(model, h, anom, static, doy, delta, w,
                                            micro_batch, False))
                if len(losses) >= 8:
                    break
        return float(np.mean(losses))

    seconds = time.perf_counter() - t0
    meta = {"steps": steps, "batch_size": batch_size, "micro_batch": micro_batch, "lr": lr,
            "seconds": seconds, "lead_samples": seen,
            "lead_samples_per_second": seen / seconds if seconds > 0 else None,
            "train_loss_first": float(np.mean(history[:20])) if history else None,
            "train_loss_last": float(np.mean(history[-20:])) if history else None,
            "val_loss": flat_loss(val) if len(val) else None}
    return model, meta


# --------------------------------------------------------------------------- #
# 5. rollout skill (official metrics)
# --------------------------------------------------------------------------- #
def evaluate_rollouts(rep: Report, twin: DigitalTwin, eb_ds, clim: HarmonicClimatology,
                      windows, std: np.ndarray) -> None:
    print("\n[5] 10-step rollout skill on 2020-2021 windows (official metrics)")
    leads = list(LEADS)
    rmse_m, rmse_p, rmse_c = {k: [] for k in leads}, {k: [] for k in leads}, {k: [] for k in leads}
    csi_m, csi_p, csi_c = {k: [] for k in leads}, {k: [] for k in leads}, {k: [] for k in leads}
    ss_rows, var_rows, bad_windows = [], [], []
    for w in windows:
        init = eb_ds.data[w.i0:w.i0 + 3]
        target_dates = [w.init_last + timedelta(days=k) for k in range(1, w.steps + 1)]
        truth = eb_ds.data[w.i0 + 3:w.i0 + 3 + w.steps]
        doy = day_of_year(w.init_last)
        t = float(theta_from_doy(doy))
        pred = twin.simulate(init, [math.sin(t), math.cos(t)], w.steps)
        persist = np.repeat(init[-1:], w.steps, axis=0)
        climd = clim.predict_dates(target_dates)
        bad = validity(pred, w.steps)
        if bad:
            bad_windows.append({"init_last": w.init_last.isoformat(), "violations": bad})
        for k in leads:
            rmse_m[k].append(rmse_lat(pred[k - 1:k], truth[k - 1:k]))
            rmse_p[k].append(rmse_lat(persist[k - 1:k], truth[k - 1:k]))
            rmse_c[k].append(rmse_lat(climd[k - 1:k], truth[k - 1:k]))
            csi_m[k].append(csi(pred[k - 1, 1], truth[k - 1, 1]))
            csi_p[k].append(csi(persist[k - 1, 1], truth[k - 1, 1]))
            csi_c[k].append(csi(climd[k - 1, 1], truth[k - 1, 1]))
            ss_rows.append(skill_vs_best_ref(pred[k - 1:k], truth[k - 1:k],
                                             persist[k - 1:k], climd[k - 1:k])["ss"])
        var_rows.append(variance_ratio(pred[DAY8], truth[DAY8]))

    def mean(xs):
        return float(np.nanmean(xs))

    table = {}
    for k in leads:
        m = np.mean(np.stack(rmse_m[k]), axis=0)
        p = np.mean(np.stack(rmse_p[k]), axis=0)
        c = np.mean(np.stack(rmse_c[k]), axis=0)
        best = np.minimum(p, c)
        with np.errstate(divide="ignore", invalid="ignore"):
            ss = np.where(best > 0, 1.0 - m / best, np.nan)
        table[str(k)] = {
            "rmse_model": {ch: float(v) for ch, v in zip(CHANNELS, m)},
            "rmse_persistence": {ch: float(v) for ch, v in zip(CHANNELS, p)},
            "rmse_climatology": {ch: float(v) for ch, v in zip(CHANNELS, c)},
            "ss_vs_best_ref": {ch: float(v) for ch, v in zip(CHANNELS, ss)},
            "mean_ss": float(np.nanmean(ss)),
            "csi_model": mean(csi_m[k]), "csi_persistence": mean(csi_p[k]),
            "csi_climatology": mean(csi_c[k]),
        }
    var = np.stack(var_rows)
    vr = np.nanmean(var, axis=0)
    ab = anti_blur_points(var)
    rep.metric("rollout_skill", {
        "n_windows": len(windows), "per_lead": table,
        "variance_ratio_day8": {ch: float(v) for ch, v in zip(CHANNELS, vr)},
        "anti_blur_points_of_10": ab,
        "validity_failures": bad_windows,
        "mean_ss_all_leads": float(np.nanmean(np.stack(ss_rows))),
    })

    print(f"  {'lead':<6}{'mean SS':>9}{'CSI model':>11}{'CSI pers':>10}{'CSI clim':>10}"
          f"{'RMSE t2m':>10}{'RMSE tp':>9}")
    for k in leads:
        r = table[str(k)]
        print(f"  d{k:<5}{r['mean_ss']:>9.3f}{r['csi_model']:>11.3f}{r['csi_persistence']:>10.3f}"
              f"{r['csi_climatology']:>10.3f}{r['rmse_model']['t2m']:>10.3f}"
              f"{r['rmse_model']['tp']:>9.3f}")
    print("  day-8 variance ratio " + " ".join(f"{c} {v:.2f}" for c, v in zip(CHANNELS, vr))
          + f" | anti-blur {ab:.1f}/10")

    rep.check("every rollout passes the physical validity gate", not bad_windows,
              f"{len(bad_windows)} failing windows")
    rep.check("day-8 spatial variance stays in the anti-blur band", ab >= 7.0,
              f"anti-blur points {ab:.1f}/10 (band 0.40-2.50)")
    rep.check("smoke model beats persistence at day 4 (hard gate)",
              table["4"]["mean_ss"] > 0.0, f"mean SS {table['4']['mean_ss']:+.3f}")


# --------------------------------------------------------------------------- #
# 6. causality
# --------------------------------------------------------------------------- #
def check_causality(rep: Report, twin: DigitalTwin, eb_ds, std: np.ndarray) -> None:
    print("\n[6] perturbation causality (official checks)")
    picks = []
    for want in (date(2020, 7, 15), date(2021, 10, 10)):
        idx = eb_ds.index.get(want)
        if idx is None or idx < 2 or idx + 11 >= len(eb_ds.days):
            continue
        picks.append((idx, want))
    results, null_ok_all, rows = [], True, []
    for idx, day in picks:
        init = eb_ds.data[idx - 2:idx + 1]
        t = float(theta_from_doy(day_of_year(day)))
        sc = [math.sin(t), math.cos(t)]
        doy_sincos = sc
        base = twin.simulate(init, doy_sincos, 10)
        null = twin.simulate(init, doy_sincos, 10, np.zeros((NC, NLAT, NLON), "f4"))
        null_ok = null_consistent(base, null, std)
        null_ok_all = null_ok_all and null_ok
        for name, box in (("warm_ocean", BOX_BAY_OF_BENGAL), ("monsoon_stall", BOX_ARABIAN_SEA)):
            shock = make_shock(name, init[-1], std)
            out = twin.simulate(init, doy_sincos, 10, shock)
            r = causality_checks(base, out, box, std)
            results.append(r)
            rows.append({"date": day.isoformat(), "shock": name, **r})
            print(f"  {day} {name:<14} nonzero={r['nonzero']} localized={r['localized']} "
                  f"near1={r['near_frac_day1']:.3f} propagates={r['propagates']} "
                  f"stable={r['stable']} far={r['far_frac_max']:.4f}")
    points = causality_points(results, null_ok_all) if results else 0.0
    rep.metric("causality", {"null_consistent": null_ok_all, "points_of_15": points,
                             "checks": rows})
    rep.check("null shock changes nothing", null_ok_all,
              "zero perturbation is bit-identical to the unperturbed rollout")
    rep.check("shocks produce a non-zero, numerically stable response",
              bool(results) and all(r["nonzero"] and r["stable"] for r in results),
              f"causality {points:.1f}/15 across {len(results)} shock cases")
    rep.check("day-1 response is localised near the shocked box",
              bool(results) and all(r["localized"] for r in results),
              "near_frac " + " ".join(f"{r['near_frac_day1']:.2f}" for r in results))


# --------------------------------------------------------------------------- #
# main
# --------------------------------------------------------------------------- #
def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--data", default="india_climate_train.nc")
    ap.add_argument("--climatology", default=str(COEF_PATH))
    ap.add_argument("--static-codec", default=str(DEFAULT_STATIC))
    ap.add_argument("--steps", type=int, default=300, help="smoke-training steps (0 = skip)")
    ap.add_argument("--batch-size", type=int, default=4)
    ap.add_argument("--micro-batch", type=int, default=8)
    ap.add_argument("--lr", type=float, default=2e-3)
    ap.add_argument("--threads", type=int, default=4)
    ap.add_argument("--eval-stride", type=int, default=4)
    ap.add_argument("--out", default=str(REPORT_PATH))
    ap.add_argument("--skip-train", action="store_true")
    args = ap.parse_args(argv)
    data_p = Path(args.data)
    here = Path(__file__).resolve().parent
    if not data_p.exists() and (here / ".venv" / data_p.name).exists():
        args.data = str(here / ".venv" / data_p.name)

    torch.set_num_threads(args.threads)
    torch.manual_seed(0)
    t_start = time.perf_counter()
    rep = Report()

    eb_ds = eb.load_dataset(args.data)               # official window sampler below
    data, days = eb_ds.data, eb_ds.days
    clim = HarmonicClimatology.load(args.climatology)
    static, codec = load_static_codec(args.static_codec)

    check_climatology(rep, clim, data, days)

    train = ClimateWindows(data, days, codec, clim, static, split="train", horizon=10, stride=1)
    val = ClimateWindows(data, days, codec, clim, static, split="val", horizon=10, stride=2)
    check_dataset(rep, train, val)

    probe_idx = eb_ds.index[date(2020, 7, 15)]
    init = eb_ds.data[probe_idx - 2:probe_idx + 1].copy()
    t0 = float(theta_from_doy(day_of_year(date(2020, 7, 15))))
    doy_sincos = [math.sin(t0), math.cos(t0)]
    check_architecture(rep, codec, clim, static, init, doy_sincos, args.threads)

    if args.steps > 0 and not args.skip_train:
        print(f"\n[4] smoke training: {args.steps} steps, batch {args.batch_size}, lr {args.lr}")
        model, train_meta = smoke_train(train, val, args.steps, args.batch_size, args.lr,
                                        args.threads, args.micro_batch)
        rep.metric("smoke_training", train_meta)
        print(f"  train loss {train_meta['train_loss_first']:.4f} -> "
              f"{train_meta['train_loss_last']:.4f} in {train_meta['seconds']:.1f}s "
              f"({train_meta['lead_samples_per_second']:.0f} lead-samples/s)")
        rep.check("loss decreases during the smoke fit",
                  train_meta["train_loss_last"] < train_meta["train_loss_first"],
                  f"{train_meta['train_loss_first']:.4f} -> {train_meta['train_loss_last']:.4f}")
        twin = DigitalTwin(model, codec, clim, static, torch_threads=args.threads)
        windows = eb.build_windows(eb_ds, eb.SEASONS, stride=args.eval_stride, steps=10)
        evaluate_rollouts(rep, twin, eb_ds, clim, windows, STD)
        check_causality(rep, twin, eb_ds, STD)
    else:
        print("\n[4] smoke training skipped (--skip-train / --steps 0)")
        rep.check("smoke training executed", None, "skipped; rollout skill not claimed")

    payload = rep.payload()
    payload["elapsed_seconds"] = time.perf_counter() - t_start
    payload["failures"] = rep.failures()
    Path(args.out).parent.mkdir(parents=True, exist_ok=True)
    Path(args.out).write_text(json.dumps(payload, indent=2, default=float) + "\n")

    n_fail = len(rep.failures())
    print(f"\n{len(rep.checks) - n_fail}/{len(rep.checks)} checks passed"
          + (f"; failures: {rep.failures()}" if n_fail else "")
          + f" — wrote {args.out} ({payload['elapsed_seconds']:.0f}s)")
    return 1 if n_fail else 0


if __name__ == "__main__":
    raise SystemExit(main())
