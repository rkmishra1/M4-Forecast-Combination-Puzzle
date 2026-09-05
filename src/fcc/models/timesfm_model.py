"""Google TimesFM 2.5 time-series foundation model (200M, PyTorch backend)."""

from __future__ import annotations

import gc

import numpy as np

from .base import Forecaster


class TimesFMForecaster(Forecaster):
    def __init__(
        self,
        name: str = "timesfm-2.5-200m",
        checkpoint: str = "google/timesfm-2.5-200m-pytorch",
        batch_size: int = 8,
        context_len: int = 512,
        chunk: int = 64,
    ) -> None:
        self.name = name
        self.kind = "foundation"
        self.checkpoint = checkpoint
        self.batch_size = batch_size  # per_core_batch_size, low for 8 GB RAM
        self.context_len = context_len
        self.chunk = chunk
        self._model = None
        self._tfm = None
        self._compiled_h = None

    def _load(self):
        if self._model is None:
            import timesfm
            import torch

            torch.set_float32_matmul_precision("high")
            self._tfm = timesfm
            self._model = timesfm.TimesFM_2p5_200M_torch.from_pretrained(self.checkpoint)

    def forecast(self, series: dict[str, np.ndarray], h: int, m: int) -> dict[str, np.ndarray]:
        self._load()
        if self._compiled_h != h:
            self._model.compile(
                self._tfm.ForecastConfig(
                    max_context=self.context_len,
                    max_horizon=h,
                    normalize_inputs=True,
                    use_continuous_quantile_head=True,
                    force_flip_invariance=True,
                    infer_is_positive=True,
                    fix_quantile_crossing=True,
                    per_core_batch_size=self.batch_size,
                )
            )
            self._compiled_h = h
        ids = list(series)
        preds: dict[str, np.ndarray] = {}
        for i in range(0, len(ids), self.chunk):
            chunk = ids[i : i + self.chunk]
            inputs = [
                np.asarray(series[u][-self.context_len :], dtype=np.float32) for u in chunk
            ]
            point, _quantiles = self._model.forecast(horizon=h, inputs=inputs)
            for u, row in zip(chunk, point):
                preds[u] = np.asarray(row, dtype=float)
            gc.collect()  # bound memory on small-RAM machines
        return preds
