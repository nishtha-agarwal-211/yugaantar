# Multiverse of Monsoon: Climate Digital Twin of India

> Spatiotemporal Earth-System modeling, autoregressive rollout forecasting, and counterfactual climate shock simulation for India (32 × 36 grid, 1° resolution). Inspired by ISRO's Bharatiya Antariksh Hackathon PS-5.

---

## 1. Project Overview

This repository houses the **Multiverse of Monsoon Climate Digital Twin**, designed for:
1. **Timeline Prime (Autoregressive Rollout)**: Given 3 consecutive days of observed atmospheric state over India across 6 dynamic physical channels, predict the daily weather forward day-by-day for 10+ days without exploding, drifting, or collapsing into blurry averages.
2. **Multiverse (Causal Shock Simulation)**: Inject localized initial perturbations (e.g., Bay of Bengal SST warming, Arabian Sea wind stall) and simulate how atmospheric waves and anomalies ripple across India obeying physical locality and speed limits.

---

## 2. Atmospheric Domain & Channels

- **Grid Resolution**: 32 × 36 cells of 1° × 1° spatial resolution.
  - Latitude: $6.5^\circ\text{N} \dots 37.5^\circ\text{N}$ (South to North)
  - Longitude: $65.5^\circ\text{E} \dots 100.5^\circ\text{E}$ (West to East)
- **Dynamic Channels (Physical Units)**:
  1. `t2m`: 2-meter air temperature ($\text{K}$)
  2. `tp`: Daily total precipitation ($\text{mm/day}$, non-negative)
  3. `u10`: 10-meter eastward wind ($\text{m/s}$)
  4. `v10`: 10-meter northward wind ($\text{m/s}$)
  5. `msl`: Mean sea-level pressure ($\text{Pa}$)
  6. `tcwv`: Total column water vapour ($\text{kg/m}^2$)
- **Static Channels**: `land_sea_mask` (0/1) and `orography_m` (surface elevation in meters).

---

## 3. Repository Structure

```
.
├── API_CONTRACT.md              # Exact Round 1 wire format & scoring rules
├── PRD Multiverse of Monsoon... # Full Product Requirements Document
├── problem statement.md         # Official track mission & constraints
├── stats.json                   # Normalization parameters (internal use only)
├── evaluate_baselines.py        # Phase 1: offline benchmark & scoring harness
├── baseline_scores.json         # Evaluated metrics for Persistence vs Climatology
├── climatology.py               # Phase 2: compact 2-frequency harmonic climatology
├── dataset.py                   # Phase 2: state codec, static bundle, sliding-window dataset
├── model.py                     # Phase 2: local-only delta network + DigitalTwin bundle
├── check_phase2.py              # Phase 2: end-to-end verification harness
├── artifacts/
│   ├── climatology_coefs.npz    # 77 KB, 34,560 harmonic coefficients (2005-2019 fit)
│   ├── static_codec.npz         # 9 KB, land/sea + orography/coords planes + codec constants
│   └── phase2_check.json        # Latest verification report
├── serve.py                     # Offline FastAPI inference server (entrypoint)
├── dashboard/                   # Phase 5: interactive multiverse dashboard
│   ├── app.py                   # FastAPI UI host + bootstrap API (port 8001)
│   ├── state.py                 # Synthetic 3-day analysis windows (32x36 grid)
│   └── static/                  # Canvas map UI (vanilla JS), coastline, styles
└── README.md                    # Project documentation
```

---

## 4. Phase 1: Reference Baselines & Scoring Harness

We implement and evaluate two reference baselines following the exact metric formulas in `monsoon_r1.scoring`:
1. **Persistence**: Repeats `init_states[-1]` across all lead days.
2. **Harmonic Climatology**: 2-frequency annual Fourier harmonic decomposition fitted on 2005–2019 data and evaluated at future day-of-year phases ($\sin, \cos$).

To run the offline benchmark:
```bash
python evaluate_baselines.py
```

---

## 5. Phase 2: Data Pipeline, Harmonic Climatology & Model Architecture

The twin never forecasts an absolute field. It forecasts the **one-day increment
of the state**, with a compact harmonic climatology as the reference climate:

```
state[t+1] = state[t] + Network(state[t-2 : t+1], anomaly[t], static, doy[t], doy[t+1])
```

### 5.1 Compact harmonic climatology — [climatology.py](climatology.py)

Fit by least squares on **2005-01-01 … 2019-12-31 only**, per grid cell and channel:

```
clim(doy) = mean + a1 sin(2π doy/365) + b1 cos(2π doy/365)
                 + a2 sin(4π doy/365) + b2 cos(4π doy/365)
```

* Five numbers per cell and channel → a `[5, 6, 32, 36]` tensor.
* Saved as float32 annual mean + float16 wave coefficients: **77 KB** on disk
  (135 KB in float32). There is no 365-day lookup table of past weather anywhere
  in the repo; the climatology is a 5-parameter-per-cell seasonal model.
* The API only sends `doy_sincos` of the last input day. The second harmonic is
  recovered with the double-angle identities, so `predict_sincos(sin, cos)` and
  the calendar path agree to 0.0 — a rollout never needs a date.
* Validation (2020-2021) beats a constant annual mean on every channel:
  t2m 1.84 K, tcwv 6.44, msl 292 Pa, tp 6.62 mm/day; domain-mean t2m seasonal
  cycle correlation 0.987.

```bash
python climatology.py fit --out artifacts/climatology_coefs.npz
python climatology.py show --coefs artifacts/climatology_coefs.npz
```

### 5.2 Data pipeline — [dataset.py](dataset.py)

`ClimateWindows` turns the 2005-2021 daily record into sliding windows: three
consecutive observed days in, up to 10 autoregressive targets out, with every
input and target day strictly inside the requested split (no leakage).

| item | shape | meaning |
|---|---|---|
| `hist` | `[3, 6, 32, 36]` | latent states of the last three days |
| `anom` / `anom_seq` | `[6,32,36]` / `[K+1, 6, 32, 36]` | state minus harmonic climatology of the same day |
| `static` | `[4, 32, 36]` | `land_sea_mask`, normalised `orography_m`, lat, lon |
| `doy` | `[8, K]` | `[sin, cos, sin2, cos2]` of the source day and of each forecast day |
| `target` / `delta` | `[K, 6, 32, 36]` | latent absolute states and one-day increments |

`StateCodec` normalises with the shipped `stats.json` constants for the five
linear channels. Precipitation is standardised in `log1p` space and decoded with
`expm1(clip(x, 0, ·))`, so **rain is mathematically ≥ 0 mm/day**. Decoding also
clamps t2m, msl, tcwv and the wind components to physical ranges; on real
observations the guards never fire (verified), they only make the validity gate
impossible to fail. The static bundle additionally carries normalised latitude
and longitude planes, because a purely local convolution otherwise has no way to
know where it is on the globe.

```bash
python dataset.py build --data india_climate_train.nc --stats stats.json \
    --climatology artifacts/climatology_coefs.npz --out artifacts/static_codec.npz
```

### 5.3 Local-only architecture — [model.py](model.py)

Six ConvNeXt-style blocks, each a 3×3 depthwise convolution + 1×1 pointwise
expansion/projection with GroupNorm, GELU, LayerScale and a residual. **No
attention, no transformer, no linear layer, no global pooling** — a one-day
response can reach at most `1 + n_blocks = 7` cells, and that radius grows by 7
cells per rolled-out day. `LocalDeltaNet.assert_local_ops()` enforces this.

* **262,374 parameters** (budget 3 M).
* **19.5 ms** for a 10-step rollout on 4 CPU threads here (budget 500 ms); the
  inference-time batch is always one frame.
* The output head is **zero-initialised**, so an untrained twin is exactly
  persistence (measured: head output 0, rollout within 1.5e-5 of repeat-last-day,
  which is codec round-off). Training only learns the correction.
* Each latent increment is clamped to ±4 standard deviations per step as a
  runaway guard; the clamp is elementwise and cannot create a non-local response.
* The whole system (weights + climatology + statics + codec) round-trips through
  a single 1.2 MB `artifacts/model_bundle.pt` file, loadable with
  `torch.load(..., weights_only=True)`.

### 5.4 Verification — [check_phase2.py](check_phase2.py)

```bash
python check_phase2.py --steps 300 --threads 4
```

33 checks, all passing on this machine, including: compact bundle, float16
losslessness (`max |err|/std < 1e-3`), `[sin, cos]` ≡ calendar path, window
alignment `target = init_last + k`, zero split leakage, delta/cumsum identity,
encode/decode round-trip, no attention, parameter budget, timing, determinism,
null-shock bit-identity, and 14-step stability. A 300-step smoke fit reaches
mean RMSE skill **+0.17 at day 1 and +0.06 at day 4** against the better of
persistence and climatology, with anti-blur 10.0/10 and heavy-rain CSI above
persistence at every lead; the day-1 shock response is 100% localised.

Measured on the smoke fit, the open item for Phase 3 is **propagation**: the
`propagates` sub-check (response energy outside the box+3 neighbourhood growing
by days 4-8) fires in 1 of 4 cases, giving 12.2/15 causality points. A real
training run with a multi-step rollout loss is expected to fix this by keeping
the anomaly amplitude alive instead of damping it towards climatology.

## 6. Serving & Inference

The offline inference server runs via FastAPI/Uvicorn on `0.0.0.0:8000`:
```bash
python serve.py --port 8000
```

Verify compliance using the official schema validator:
```bash
python -m monsoon_r1.schema_validator http://localhost:8000
```

---

## 6. Phase 5: Interactive Multiverse Dashboard

The desk-review dashboard is a separate process in `dashboard/`; the graded
entrypoint stays `serve.py` at the repo root.

```bash
python serve.py --port 8000            # model server (contract unchanged)
python dashboard/app.py --port 8001    # dashboard at http://127.0.0.1:8001
```

- **Live, never pre-rendered.** Each panel is a `POST http://localhost:8000/simulate`
  issued by the browser itself. One 10-step call per run; the timeline slider only
  indexes into the returned leads, so scrubbing Day 1–10 costs no extra requests.
  A local round trip is ~130 ms for the 1.25 MB response.
- **Map.** 32×36 cells, lat 6.5–37.5, lon 65.5–100.5, equirectangular projection with
  the bundled simplified coastline (`dashboard/static/geo/india.json`).
- **Channels.** `t2m` (K→°C), `tp`, `u10`/`v10` (animated streamlines), `msl` (Pa→hPa),
  `tcwv`, each with its own colour ramp and units.
- **Shock sandbox.** Presets (Bay of Bengal SST warming +3 K, Arabian Sea wind stall
  −40%, Himalayan cold surge −5 K, null control), or click the map to move the centre;
  radius and intensity sliders build a 6×32×36 perturbation delta with a cosine taper
  in physical units, which is exactly what `/simulate` expects.
- **Dual / difference view.** Timeline Prime, Multiverse and Difference
  (perturbed − baseline) side by side, on a symmetric anomaly scale taken from the
  largest response across all 10 leads.
- **Analysis windows.** The benchmark's windows are hidden, so `dashboard/state.py`
  builds three deterministic synthetic monsoon windows from the shipped statistics and
  coastline. The UI labels them as synthetic. Point `MODEL_API` (or `?api=`) at another
  host to drive a different server.
- `serve.py` allows loopback CORS origins so the dashboard, served from its own port,
  can call `/simulate` directly.

If `model_weights.pt` is absent the server answers with its persistence fallback and the
dashboard shows a banner: the difference map then shows the injected shock but no
wave propagation.
