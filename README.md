# Multiverse of Monsoon — Climate Digital Twin of India

A spatiotemporal earth-system twin for the 1° India grid (32 × 36 cells, 6.5–37.5° N,
65.5–100.5° E) across six atmospheric channels. It does two things:

1. **Timeline Prime** — given three consecutive observed days, roll the atmospheric state
   forward day by day for 10 days without drifting, exploding, or collapsing into a
   climatological blur.
2. **Multiverse** — inject a localised shock (Bay of Bengal warming, Arabian Sea wind stall,
   Himalayan cold surge), and watch how far and how fast it ripples across India.

Everything runs offline on CPU: no weather API, no network call, no calendar lookup.

| # | channel | unit | notes |
|---|---|---|---|
| 0 | `t2m` | K | 2 m temperature |
| 1 | `tp` | mm/day | daily total precipitation, never negative |
| 2 | `u10` | m/s | 10 m eastward wind |
| 3 | `v10` | m/s | 10 m northward wind |
| 4 | `msl` | Pa | mean sea-level pressure |
| 5 | `tcwv` | kg/m² | total column water vapour |

---

## Quick start

```bash
python -m venv .venv && source .venv/bin/activate
pip install torch numpy scipy xarray netCDF4 fastapi uvicorn

python serve.py --port 8000          # model API (graded entrypoint)
python dashboard/app.py --port 8001  # dashboard at http://127.0.0.1:8001
```

`serve.py` needs `model_weights.pt` next to it. **This workspace does not contain that
file yet** (see [Submission status](#submission-status)); without it the server answers
from a contract-valid persistence fallback and `/health` says so.

---

## Repository layout

```
serve.py               FastAPI model server: POST /health, POST /simulate (graded entrypoint)
train.py               physics-informed training loop -> model_weights.pt
model.py               local ConvNeXt delta twin (LocalDeltaNet + DigitalTwin bundle)
dataset.py             state codec, static bundle, sliding-window dataset
climatology.py         2-frequency harmonic climatology (fit / evaluate / save)
evaluate_baselines.py  persistence and climatology reference scores
check_phase2.py        end-to-end verification harness (33 checks)
audit_repo.py          repository integrity scan (anti-cheat, code health, submission checks)
brief.pdf              two-page submission brief
stats.json             per-channel mean/std used by the codec
artifacts/             climatology coefficients, static bundle, latest verification report
dashboard/             multiverse UI (FastAPI host + canvas map, vanilla JS)
API_CONTRACT.md        exact wire format and scoring rules
```

---

## Architecture

The twin never predicts an absolute field. It predicts the **one-day increment** of the
state, relative to a compact harmonic climatology:

```
state[t+1] = state[t] + f(state[t-2 : t+1], anomaly[t], static, doy[t], doy[t+1])
```

* **Local ConvNeXt core** (`model.py`). Six blocks, each `3×3` depthwise convolution →
  GroupNorm → `1×1` expansion/projection → GELU, with LayerScale `1e-2` and a residual.
  **262,374 parameters.** There is no attention, no transformer, no global pooling and no
  linear layer — `LocalDeltaNet.assert_local_ops()` fails loudly if one is added. One day's
  response can reach at most **7 cells**, and that radius grows by 7 cells per rolled-out
  day, so a shock propagates instead of teleporting.
* **Served U-Net** (`serve.py`). The graded server wraps the same local blocks in a
  three-stage decoder (775,078 parameters, width 32/64/128, two max-pool stages) with a
  tanh-bounded increment of at most 2σ per step. It shares the block design but not the
  strict locality budget: measured day-1 sensitivity already spans the whole domain, which
  is the open item F3 in `brief.pdf`. Serving the local core is the compliant fix.
* **Static conditioning.** Land–sea mask, normalised orography, and normalised latitude and
  longitude planes are model inputs, so a purely convolutional model still knows where it is.
* **One bundle.** Weights, climatology, statics and the codec fit in a single ~1.2 MB
  checkpoint loadable with `torch.load(..., weights_only=True)`; a 10-step rollout costs
  ~20 ms on four CPU threads.

### Physics and stability

* **Harmonic climatology, not a date table.** `clim(doy) = mean + a₁sin t + b₁cos t +
  a₂sin 2t + b₂cos 2t` with `t = 2π·doy/365`, fitted by least squares on 2005–2019 only:
  34,560 coefficients, **77 KB**. The API only sends `doy_sincos` of the last input day; the
  second harmonic is recovered with double-angle identities, so inference never touches a
  calendar and the `[sin, cos]` and calendar paths agree exactly.
* **Rain cannot go negative.** `tp` is standardised in `log1p` space and decoded with
  `expm1(clip(x, 0, ·))` — a smooth, softplus-family positive map. The validity gate then
  floors `tp`/`tcwv` at 0 and clamps `t2m` to 233.16–333.14 K (−40…+60 °C) in float64.
* **Anti-drift.** Delta parameterisation keeps persistence and variance; the output head is
  zero-initialised so an untrained twin is bit-exactly persistence; increments are clamped
  per step; the model is deterministic, so a zero perturbation is bit-identical to the
  unperturbed call.
* **Anti-blur.** Training adds Sobel gradient-magnitude matching and a spatial-variance band
  (0.55–1.8) on the final unrolled step, tighter than the scored 0.40–2.50 band.

---

## Training

Training needs `india_climate_train.nc`: `data[time, channel, lat, lon]` in contract channel
order plus a `static[2, lat, lon]` field, 2005–2021. It is **not** committed (multi-year
daily archives are out of scope for the repo) and is not present in this workspace.

```bash
# 1. fit the harmonic climatology (2005-2019 only)
python climatology.py fit --data india_climate_train.nc \
    --out artifacts/climatology_coefs.npz --rain-space linear

# 2. build the static bundle + codec
python dataset.py build --data india_climate_train.nc --stats stats.json \
    --climatology artifacts/climatology_coefs.npz --out artifacts/static_codec.npz

# 3. train -> model_weights.pt (the file serve.py loads)
python train.py --data india_climate_train.nc --out model_weights.pt \
    --epochs 16 --batch-size 8 --max-unroll 4 --curriculum 0.25 0.50 0.75

# quick smoke run first
python train.py --data india_climate_train.nc --epochs 2 --max-train-batches 8
```

`train.py` uses a multi-step autoregressive curriculum (loss backpropagated through 1→4
unrolled days), a latitude-weighted MSE + L1 compound loss, 4× weight on cells above
10 mm/day plus a differentiable soft-CSI term, the anti-blur terms above, and perturbation
augmentation in 20 % of batches with locality/energy/boundedness regularisers. Checkpoints
are selected on 2020–2021 validation: `skill_day4 + 0.5·mean_CSI(1,4,8) − 0.5·anti-blur
failures`, and the run records whether persistence is beaten at day 4.

```bash
python evaluate_baselines.py     # persistence vs climatology reference scores
python check_phase2.py --steps 300 --threads 4   # 33 end-to-end checks
```

`check_phase2.py` needs both the NetCDF and the organiser's `monsoon_r1` team kit
(`pip install monsoon_r1_kit-1.0.0-py3-none-any.whl`); neither ships in this repo, and the
harness exits with an explicit message when they are missing.
```

---

## Serving (`serve.py`)

```bash
python serve.py --port 8000        # binds 0.0.0.0, as the judge runs it
```

* `POST /health` → `{"status": "ready", "model_name": "..."}` once weights are loaded.
* `POST /simulate` with `{"init_states": [3,6,32,36], "doy_sincos": [sin, cos], "steps": 10,
  "perturbation": null | [6,32,36]}` → `{"rollout": [steps,6,32,36], "inference_ms": ...}`,
  plain nested JSON lists, physical units in and out, `rollout[k]` = day `k+1` after the last
  input day. See `API_CONTRACT.md` for the exact contract.
* The perturbation is added to `init_states[-1]` before the first step, exactly as specified.
* Loading happens at import, `/health` reports not-ready on failure, and CORS is restricted to
  loopback origins so the dashboard (a different port) can call it from the browser.
* No checkpoint → clean persistence fallback: contract-valid output, `model_name` says
  `fallback-persistence (model_weights.pt MISSING)`.

---

## Dashboard (`dashboard/`)

```bash
python dashboard/app.py --port 8001        # UI on http://127.0.0.1:8001
MODEL_API=http://127.0.0.1:9000 python dashboard/app.py --port 8001   # other host
```

* **Live, never pre-rendered.** Every run is one browser `POST /simulate` (~130 ms, 1.25 MB);
  the day slider only indexes the returned leads.
* **Flow.** Pick an analysis window (monsoon peak, break phase, pre-monsoon heat — synthetic,
  and labelled as such) → pick a shock preset (BoB +3 K/+15 % tcwv, Arabian Sea stall −40 %,
  Himalayan cold surge −5 K, null control) or click the map to move the centre and set
  radius/intensity → Timeline Prime, Multiverse and Difference side by side on a symmetric
  anomaly scale → scrub day 1–10 per channel → read the crop-belt panel.
* **Map.** Equirectangular 32 × 36 grid with the bundled simplified coastline, per-channel
  colour ramps and units, animated streamlines for wind.
* **Crop belt.** 112 land cells over lat 18–26° N, lon 73–88° E with lat-weighted rain, wind
  and heat-degree-day anomalies — the agronomic reading of a multiverse scenario.
* A server-side `/api/health` probe shows which model answered and banners the fallback.

---

## Repository integrity scan

```bash
python audit_repo.py          # human readable, exit 1 on a critical flag
python audit_repo.py --json   # machine readable
```

The organiser's own command is
`python -c "import monsoon_r1.repo_scan as rs; from pathlib import Path; print(rs.scan(Path('.')))"`,
which needs the `monsoon_r1` team kit; the wheel is not part of the repository and is not on
PyPI, so `audit_repo.py` re-implements the same checks offline.

Seven checks, mirroring `monsoon_r1.repo_scan`: external endpoints in
inference code, hardcoded dates in inference code, committed multi-year archives (> 400
daily slices), checkpoint presence and size (< 2 GB), a real training pipeline, a clean
server/UI split, and that every module parses on the scoring interpreter.

Latest run on this workspace: **6 of 7 checks pass, 1 critical** — no checkpoint is
present, so `serve.py` is serving its persistence fallback. Everything else is clean:
0 non-loopback URLs, no calendar in the served path, largest committed file 111 KB,
10 modules parse, training pipeline and server/UI separation intact.

## Submission status

| item | state |
|---|---|
| `brief.pdf` (2 pages) | complete — architecture, physics, failure modes, dashboard, impact, multiverse results |
| repo integrity scan | 6/7 pass; `weights/present` is the single open flag |
| checkpoint (`model_weights.pt`) | **missing** — run `python train.py --data india_climate_train.nc` |
| served-model day-1 locality | **open** (U-Net F3) — the local ConvNeXt core is the compliant path |

Both open items are fixable from this repo alone: the checkpoint is one training command
once the NetCDF is available, and the locality item is a serving-architecture switch.
