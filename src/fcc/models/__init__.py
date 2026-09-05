"""Model registry: build forecasters by short name from experiment config."""

from __future__ import annotations

from ..config import DEFAULTS
from .base import Forecaster
from .chronos_model import ChronosForecaster
from .classical import CLASSICAL_MODELS, StatsForecastForecaster
from .timesfm_model import TimesFMForecaster

CHRONOS_CHECKPOINTS = {
    "chronos-t5-tiny": "amazon/chronos-t5-tiny",
    "chronos-t5-small": "amazon/chronos-t5-small",
    "chronos-t5-base": "amazon/chronos-t5-base",
    "chronos-t5-large": "amazon/chronos-t5-large",
    "chronos-bolt-mini": "amazon/chronos-bolt-mini",
    "chronos-bolt-small": "amazon/chronos-bolt-small",
    "chronos-bolt-base": "amazon/chronos-bolt-base",
}

TIMESFM_CHECKPOINTS = {
    "timesfm-2.5-200m": "google/timesfm-2.5-200m-pytorch",
}

# CPU sdpa attention collapses superlinearly above an effective batch that
# shrinks as the model grows (measured on M5 Pro: t5-small fine at 32,
# t5-base at 16, t5-large at 8; bolt models are decoder-free and cheap).
CHRONOS_BATCH_CAPS = {
    "chronos-t5-tiny": 32,
    "chronos-t5-small": 32,
    "chronos-t5-base": 16,
    "chronos-t5-large": 4,
    "chronos-bolt-mini": 64,
    "chronos-bolt-small": 64,
    "chronos-bolt-base": 64,
}


def build_model(name: str, cfg: dict) -> Forecaster:
    if name in CLASSICAL_MODELS:
        return StatsForecastForecaster(
            name, CLASSICAL_MODELS[name], context_len=cfg.get("context_len", 512)
        )
    if name in CHRONOS_CHECKPOINTS:
        return ChronosForecaster(
            name=name,
            checkpoint=CHRONOS_CHECKPOINTS[name],
            device=cfg.get("device", "cpu"),
            batch_size=min(cfg.get("batch_size", 32), CHRONOS_BATCH_CAPS.get(name, 32)),
            context_len=cfg.get("context_len", 512),
        )
    if name in TIMESFM_CHECKPOINTS:
        return TimesFMForecaster(
            name=name,
            checkpoint=TIMESFM_CHECKPOINTS[name],
            batch_size=cfg.get("timesfm_batch", 8),
            context_len=cfg.get("context_len", 512),
        )
    raise ValueError(
        f"unknown model {name!r}; available: "
        f"{sorted(CLASSICAL_MODELS) + sorted(CHRONOS_CHECKPOINTS) + sorted(TIMESFM_CHECKPOINTS)}"
    )
