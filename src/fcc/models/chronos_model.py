"""Amazon Chronos time-series foundation models (chronos-forecasting)."""

from __future__ import annotations

import numpy as np

from .base import Forecaster


def _median_from_forecast(fcst) -> np.ndarray:
    """Extract the median forecast [batch, horizon] from a chronos predict result.

    chronos-forecasting 2.x legacy T5 pipelines return a plain tensor of
    sampled paths [batch, n_samples, horizon]; newer forecast objects expose
    a .quantile(q) method; very old versions returned (mean, quantiles).
    """
    import torch

    # Plain tensor: [batch, n_samples_or_quantiles, horizon] -> median over axis 1
    if isinstance(fcst, torch.Tensor):
        arr = fcst.detach().cpu().numpy()
        if arr.ndim == 3:
            arr = np.median(arr, axis=1)
        return arr

    # Forecast object with a real .quantile() method (not torch.Tensor.quantile)
    if hasattr(fcst, "quantile") and callable(fcst.quantile):
        try:
            q = fcst.quantile(0.5)
        except TypeError:
            q = fcst.quantile(torch.tensor(0.5))
        arr = q.detach().cpu().numpy() if isinstance(q, torch.Tensor) else np.asarray(q)
        if arr.ndim == 3:  # [batch, 1, horizon]
            arr = arr[:, 0, :]
        return arr

    # Legacy tuple (mean, quantiles [batch, nq, horizon])
    if isinstance(fcst, tuple) and len(fcst) == 2:
        arr = fcst[1]
        arr = arr.detach().cpu().numpy() if isinstance(arr, torch.Tensor) else np.asarray(arr)
        return arr[:, arr.shape[1] // 2, :]

    arr = np.asarray(fcst)
    if arr.ndim == 3:
        arr = np.median(arr, axis=1)
    return arr


class ChronosForecaster(Forecaster):
    def __init__(
        self,
        name: str,
        checkpoint: str,
        device: str = "cpu",
        batch_size: int = 32,
        context_len: int = 512,
        seed: int = 0,
    ) -> None:
        self.name = name
        self.kind = "foundation"
        self.checkpoint = checkpoint
        self.device = device
        self.batch_size = batch_size
        self.context_len = context_len
        self.seed = seed
        self._pipe = None

    def _load(self):
        if self._pipe is None:
            import torch
            from chronos import BaseChronosPipeline

            self._torch = torch
            self._pipe = BaseChronosPipeline.from_pretrained(
                self.checkpoint,
                device_map=self.device,
                torch_dtype=torch.float32,
            )

    def forecast(self, series: dict[str, np.ndarray], h: int, m: int) -> dict[str, np.ndarray]:
        self._load()
        torch = self._torch
        torch.manual_seed(self.seed)
        ids = list(series)
        preds: dict[str, np.ndarray] = {}
        for i in range(0, len(ids), self.batch_size):
            chunk = ids[i : i + self.batch_size]
            ctx = [
                torch.tensor(
                    np.asarray(series[u][-self.context_len :], dtype=np.float32)
                )
                for u in chunk
            ]
            fcst = self._pipe.predict(inputs=ctx, prediction_length=h)
            med = _median_from_forecast(fcst)
            for u, row in zip(chunk, med):
                preds[u] = np.asarray(row, dtype=float)
        return preds
