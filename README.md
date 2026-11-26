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
├── serve.py                     # Offline FastAPI inference server (entrypoint)
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

## 5. Serving & Inference

The offline inference server runs via FastAPI/Uvicorn on `0.0.0.0:8000`:
```bash
python serve.py --port 8000
```

Verify compliance using the official schema validator:
```bash
python -m monsoon_r1.schema_validator http://localhost:8000
```
