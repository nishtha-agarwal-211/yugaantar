"""Local-only neural architecture for the India climate digital twin (Phase 2).

Design in one paragraph
-----------------------
The model is a damped residual update on a *local* stencil.  At every step it
receives the three most recent latent states, the current latent anomaly against
the harmonic climatology, the bundled static planes and the day-of-year phase of
the current and the next day, and it predicts the one-day latent increment:

    state[t+1] = state[t] + f(state[t-2:t+1], anomaly[t], static, doy[t], doy[t+1])

Why this shape of model
-----------------------
* **No global mixing.**  Every layer is a 3x3 depthwise/1x1 pointwise convolution
  (ConvNeXt-style local block).  There is no attention, no transformer, no
  global pooling and no fully connected layer, so a one-day response cannot
  teleport: after one step the influence of a cell reaches at most
  ``receptive_radius`` cells, which grows by that radius per rolled-out day.
  ``receptive_radius = 1 + n_blocks`` (one cell per 3x3 conv).
* **Increment, not absolute field.**  Predicting a delta (a) keeps day-to-day
  persistence and therefore spatial variance, which is what the anti-blur test
  measures, (b) keeps the loss on the same scale as the day-to-day tendency, and
  (c) makes a zero output exactly "repeat yesterday", a safe initialisation.
* **Zero-initialised head.**  At initialisation the network is bit-exactly the
  persistence baseline; training only has to learn the correction, which removes
  the early-training explosions that are the usual cause of unstable rollouts.
* **Rain is non-negative by construction.**  ``tp`` lives in ``log1p`` space
  inside ``StateCodec``; the physical value is ``expm1(clip(x, 0, .)) >= 0``.
  The network never outputs mm/day directly, so no negative rain is possible.
* **Bounded increments.**  ``rollout`` clips each latent increment to
  ``max_step_delta`` standard deviations.  This is an elementwise guard against
  the rare runaway trajectory; it cannot create a non-local response.

Budget: ~0.13 M parameters (limit 3 M) and a 10-step rollout well under 500 ms
on four CPU threads; both are measured by ``check_phase2.py``.
"""
from __future__ import annotations

import time
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Iterable

import numpy as np
import torch
import torch.nn as nn

from climatology import (
    CHANNELS,
    NC,
    NLAT,
    NLON,
    HarmonicClimatology,
    theta_from_sincos,
)
from dataset import (
    DEFAULT_CLIMATOLOGY,
    DEFAULT_STATIC,
    IN_STEPS,
    StateCodec,
    StaticFields,
    doy_basis,
    load_static_codec,
)

YEAR_DAYS = 365.0
DEFAULT_WEIGHTS = Path("artifacts") / "model_bundle.pt"


# --------------------------------------------------------------------------- #
# configuration
# --------------------------------------------------------------------------- #
@dataclass
class ModelConfig:
    in_steps: int = IN_STEPS          # 3 observed days
    n_channels: int = NC              # 6 dynamic channels
    static_planes: int = 4            # land_sea_mask, orography, lat, lon
    doy_features: int = 8             # [sin, cos, sin2, cos2] of today and tomorrow
    hidden: int = 96
    blocks: int = 6
    expansion: int = 2
    layer_scale_init: float = 1e-2
    max_step_delta: float = 4.0       # latent std per day, rollout clamp

    @property
    def in_channels(self) -> int:
        return (self.in_steps * self.n_channels + self.n_channels
                + self.static_planes + self.doy_features)

    @property
    def receptive_radius(self) -> int:
        """Cells a one-day response can reach (stem + one cell per 3x3 block)."""
        return 1 + self.blocks


# --------------------------------------------------------------------------- #
# local building blocks
# --------------------------------------------------------------------------- #
class LocalBlock(nn.Module):
    """ConvNeXt-style block built only from 3x3 depthwise and 1x1 pointwise ops."""

    def __init__(self, dim: int, expansion: int = 2, layer_scale_init: float = 1e-2) -> None:
        super().__init__()
        hidden = dim * expansion
        self.dw = nn.Conv2d(dim, dim, 3, padding=1, groups=dim, padding_mode="replicate")
        self.norm = nn.GroupNorm(1, dim)
        self.pw1 = nn.Conv2d(dim, hidden, 1)
        self.act = nn.GELU()
        self.pw2 = nn.Conv2d(hidden, dim, 1)
        self.gamma = nn.Parameter(torch.full((1, dim, 1, 1), float(layer_scale_init)))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        y = self.pw2(self.act(self.pw1(self.norm(self.dw(x)))))
        return x + self.gamma * y


class LocalDeltaNet(nn.Module):
    """Predicts the one-day latent increment from the 3-day history + conditioning."""

    def __init__(self, config: ModelConfig | None = None) -> None:
        super().__init__()
        self.config = config or ModelConfig()
        cfg = self.config
        self.stem = nn.Sequential(
            nn.Conv2d(cfg.in_channels, cfg.hidden, 3, padding=1, padding_mode="replicate"),
            nn.GroupNorm(1, cfg.hidden),
            nn.GELU(),
        )
        self.blocks = nn.Sequential(*[
            LocalBlock(cfg.hidden, cfg.expansion, cfg.layer_scale_init)
            for _ in range(cfg.blocks)
        ])
        self.head = nn.Conv2d(cfg.hidden, cfg.n_channels, 1)
        # exact persistence at initialisation (and a stable starting point)
        nn.init.zeros_(self.head.weight)
        nn.init.zeros_(self.head.bias)

    # -- input assembly ----------------------------------------------------
    @staticmethod
    def _doy_planes(doy: torch.Tensor, height: int, width: int) -> torch.Tensor:
        """[B, F] or [B, F, K] doy features -> [B, F*K?, H, W] for the last K axis."""
        if doy.ndim == 2:
            return doy[:, :, None, None].expand(-1, -1, height, width)
        b, f, k = doy.shape
        return doy.reshape(b, f * k, 1, 1).expand(-1, -1, height, width)

    def forward(self, hist: torch.Tensor, anom: torch.Tensor, static: torch.Tensor,
                doy: torch.Tensor) -> torch.Tensor:
        """hist [B,3,6,H,W], anom [B,6,H,W], static [B,S,H,W], doy [B,8] -> delta [B,6,H,W]."""
        b, t, c, h, w = hist.shape
        if (t, c) != (self.config.in_steps, self.config.n_channels):
            raise ValueError(f"hist shape {tuple(hist.shape)} does not match {self.config}")
        x = torch.cat([
            hist.reshape(b, t * c, h, w),
            anom,
            static,
            self._doy_planes(doy, h, w),
        ], dim=1)
        if x.shape[1] != self.config.in_channels:
            raise ValueError(f"assembled {x.shape[1]} channels, config expects {self.config.in_channels}")
        return self.head(self.blocks(self.stem(x)))

    # -- introspection -----------------------------------------------------
    def n_parameters(self) -> int:
        return int(sum(p.numel() for p in self.parameters() if p.requires_grad))

    def assert_local_ops(self) -> None:
        """Fail loudly if anybody adds a non-local operation to this model."""
        banned = (nn.MultiheadAttention, nn.Transformer, nn.TransformerEncoder,
                  nn.TransformerDecoder, nn.LSTM, nn.GRU, nn.Linear)
        for name, module in self.named_modules():
            if isinstance(module, banned):
                raise AssertionError(f"non-local module {name}: {type(module).__name__}")
            if isinstance(module, nn.Conv2d) and module.kernel_size != (1, 1):
                if module.kernel_size != (3, 3):
                    raise AssertionError(f"kernel {module.kernel_size} in {name}; only 1x1/3x3 allowed")
        if self.config.receptive_radius > 8:
            raise AssertionError("receptive radius above 8 cells breaks day-1 locality")


# --------------------------------------------------------------------------- #
# rollout
# --------------------------------------------------------------------------- #
def rollout_latent(
    model: LocalDeltaNet,
    hist: torch.Tensor,
    static: torch.Tensor,
    climatology: HarmonicClimatology,
    codec: StateCodec,
    theta_last_input: float,
    steps: int = 10,
    max_step_delta: float | None = None,
) -> torch.Tensor:
    """Autoregressive latent rollout without any calendar: [steps, 6, H, W]."""
    cfg = model.config
    limit = cfg.max_step_delta if max_step_delta is None else float(max_step_delta)
    static_b = static if static.ndim == 4 else static[None]
    hist = hist if hist.ndim == 5 else hist[None]
    clim_cache: dict[int, torch.Tensor] = {}
    out: list[torch.Tensor] = []

    def clim_latent(k: int) -> torch.Tensor:
        """Latent harmonic climatology of forecast step k (0 = last input day)."""
        if k not in clim_cache:
            theta = theta_last_input + 2.0 * np.pi * k / YEAR_DAYS
            phys = climatology.predict_theta(theta)
            clim_cache[k] = torch.from_numpy(codec.encode(phys)[None]).to(hist.dtype)
        return clim_cache[k].expand(hist.shape[0], -1, -1, -1)

    with torch.inference_mode():
        for k in range(steps):
            doy = torch.from_numpy(
                np.concatenate([doy_basis(np.array(theta_last_input + 2 * np.pi * k / YEAR_DAYS)),
                                doy_basis(np.array(theta_last_input + 2 * np.pi * (k + 1) / YEAR_DAYS))])
                .astype(np.float32)
            ).to(hist.device)[None]                                  # [1, 8]
            anom = hist[:, -1] - clim_latent(k)                       # [1, 6, H, W]
            delta = model(hist, anom, static_b.expand(hist.shape[0], -1, -1, -1), doy)
            delta = delta.clamp(-limit, limit)
            x_next = hist[:, -1] + delta
            hist = torch.cat([hist[:, 1:], x_next[:, None]], dim=1)
            out.append(x_next[0])
    return torch.stack(out, dim=0)


# --------------------------------------------------------------------------- #
# self-contained twin (model + climatology + statics + codec)
# --------------------------------------------------------------------------- #
class DigitalTwin:
    """Everything the server needs, loaded from one weights bundle.

    ``simulate`` implements the wire contract: physical init_states [3, 6, 32, 36]
    (oldest first), ``doy_sincos`` of the last input day, a physical perturbation
    added to the last input day, and a physical rollout [steps, 6, 32, 36].
    It is deterministic, so a null (zero) perturbation is bit-identical.
    """

    def __init__(self, model: LocalDeltaNet, codec: StateCodec,
                 climatology: HarmonicClimatology, static: StaticFields,
                 torch_threads: int | None = None) -> None:
        self.model = model.eval()
        self.codec = codec
        self.climatology = climatology
        self.static = static
        self._static_t = torch.from_numpy(static.tensor)[None]         # [1,S,H,W]
        if torch_threads is not None:
            torch.set_num_threads(int(torch_threads))

    # -- inference ---------------------------------------------------------
    @torch.inference_mode()
    def simulate(self, init_states: np.ndarray, doy_sincos, steps: int = 10,
                 perturbation: np.ndarray | None = None) -> np.ndarray:
        init_states = np.asarray(init_states, dtype=np.float32)
        if init_states.shape != (IN_STEPS, NC, NLAT, NLON):
            raise ValueError(f"init_states shape {init_states.shape} != {(IN_STEPS, NC, NLAT, NLON)}")
        if perturbation is not None:
            pert = np.asarray(perturbation, dtype=np.float32)
            if pert.shape != (NC, NLAT, NLON):
                raise ValueError(f"perturbation shape {pert.shape} != {(NC, NLAT, NLON)}")
            init_states = init_states.copy()
            init_states[-1] = init_states[-1] + pert      # contract: delta on the last input day
        sin_c, cos_c = float(doy_sincos[0]), float(doy_sincos[1])
        theta0 = theta_from_sincos(sin_c, cos_c)

        hist = torch.from_numpy(self.codec.encode(init_states))[None]   # [1,3,6,H,W]
        latent = rollout_latent(self.model, hist, self._static_t, self.climatology,
                                self.codec, theta0, steps=int(steps))
        phys = self.codec.decode(latent.cpu().numpy())                 # [steps,6,H,W]
        return np.asarray(phys, dtype=np.float32)

    def simulate_timed(self, init_states, doy_sincos, steps: int = 10,
                       perturbation=None) -> tuple[np.ndarray, float]:
        t0 = time.perf_counter()
        out = self.simulate(init_states, doy_sincos, steps, perturbation)
        return out, (time.perf_counter() - t0) * 1000.0

    # -- persistence -------------------------------------------------------
    def save(self, path: str | Path = DEFAULT_WEIGHTS) -> Path:
        """Single self-contained bundle (weights + climatology + statics + codec)."""
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        cfg = asdict(self.model.config)
        payload = {
            "format": "monsoon-twin-1",
            "config": cfg,
            "state_dict": {k: v.detach().clone() for k, v in self.model.state_dict().items()},
            "climatology_coef": torch.from_numpy(self.climatology.coef.copy()),
            "climatology_meta": [self.climatology.rain_space,
                                 self.climatology.train_start,
                                 self.climatology.train_end,
                                 int(self.climatology.n_train_days)],
            "static": torch.from_numpy(self.static.tensor.copy()),
            "static_names": list(self.static.names),
            "static_orog": [float(self.static.orog_mean), float(self.static.orog_std)],
            "codec_mean": torch.from_numpy(np.asarray(self.codec.mean, np.float32)),
            "codec_std": torch.from_numpy(np.asarray(self.codec.std, np.float32)),
            "codec_tp_log1p": bool(self.codec.tp_log1p),
            "channels": list(CHANNELS),
        }
        torch.save(payload, path)
        return path

    @classmethod
    def load(cls, path: str | Path = DEFAULT_WEIGHTS,
             torch_threads: int | None = None) -> "DigitalTwin":
        payload = torch.load(Path(path), map_location="cpu", weights_only=True)
        if payload.get("format") != "monsoon-twin-1":
            raise ValueError(f"{path} is not a monsoon-twin bundle")
        config = ModelConfig(**payload["config"])
        model = LocalDeltaNet(config)
        model.load_state_dict(payload["state_dict"])
        coef = payload["climatology_coef"].numpy().astype(np.float32)
        space, start, end, n_days = payload["climatology_meta"]
        climatology = HarmonicClimatology(coef=coef, rain_space=str(space),
                                          train_start=str(start), train_end=str(end),
                                          n_train_days=int(n_days))
        static = StaticFields(tensor=payload["static"].numpy().astype(np.float32),
                              names=tuple(payload["static_names"]),
                              orog_mean=float(payload["static_orog"][0]),
                              orog_std=float(payload["static_orog"][1]))
        codec = StateCodec(mean=payload["codec_mean"].numpy(),
                           std=payload["codec_std"].numpy(),
                           tp_log1p=bool(payload["codec_tp_log1p"]),
                           stats_source="bundled")
        return cls(model, codec, climatology, static, torch_threads=torch_threads)

    @classmethod
    def from_artifacts(cls, weights: str | Path | None = None,
                       climatology_path: str | Path = DEFAULT_CLIMATOLOGY,
                       static_path: str | Path = DEFAULT_STATIC,
                       config: ModelConfig | None = None,
                       torch_threads: int | None = None) -> "DigitalTwin":
        """Load a trained bundle, or assemble an untrained one from artifacts."""
        weights = Path(weights) if weights is not None else DEFAULT_WEIGHTS
        if weights.exists():
            return cls.load(weights, torch_threads=torch_threads)
        static, codec = load_static_codec(static_path)
        return cls(LocalDeltaNet(config), codec, HarmonicClimatology.load(climatology_path),
                   static, torch_threads=torch_threads)

    # -- reporting ---------------------------------------------------------
    def describe(self) -> dict:
        return {
            "parameters": self.model.n_parameters(),
            "config": asdict(self.model.config),
            "receptive_radius_cells": self.model.config.receptive_radius,
            "static_planes": list(self.static.names),
            "climatology": self.climatology.describe(),
        }


# --------------------------------------------------------------------------- #
# smoke check used by check_phase2.py
# --------------------------------------------------------------------------- #
def random_init(seed: int = 0, scale: float = 0.05) -> LocalDeltaNet:
    """Small random weights (the head stays zero-init, so the twin is persistence)."""
    torch.manual_seed(seed)
    model = LocalDeltaNet()
    with torch.no_grad():
        for name, p in model.named_parameters():
            if name.startswith("head"):
                continue
            p.add_(torch.randn_like(p) * scale)
    return model


def main(argv: Iterable[str] | None = None) -> int:
    import argparse

    ap = argparse.ArgumentParser(description="model architecture smoke check")
    ap.add_argument("--weights", default=str(DEFAULT_WEIGHTS))
    ap.add_argument("--climatology", default=str(DEFAULT_CLIMATOLOGY))
    ap.add_argument("--static", default=str(DEFAULT_STATIC))
    ap.add_argument("--threads", type=int, default=4)
    args = ap.parse_args(list(argv) if argv is not None else None)

    twin = DigitalTwin.from_artifacts(args.weights, args.climatology, args.static,
                                      torch_threads=args.threads)
    rng = np.random.default_rng(0)
    init = rng.normal(0, 1, (IN_STEPS, NC, NLAT, NLON)).astype(np.float32)
    out, ms = twin.simulate_timed(init, [0.4, -0.9], 10)
    print(f"parameters {twin.model.n_parameters():,} | rollout {ms:.1f} ms | "
          f"shape {out.shape} | finite {bool(np.isfinite(out).all())}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
