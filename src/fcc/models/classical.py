"""Classical statistical baselines via Nixtla's statsforecast."""

from __future__ import annotations

import warnings

import numpy as np
import pandas as pd
from statsforecast import StatsForecast
from statsforecast.models import AutoARIMA, AutoETS, AutoTheta, SeasonalNaive

from .base import Forecaster


def _long_df(series: dict[str, np.ndarray]) -> pd.DataFrame:
    ids = list(series.keys())
    lens = [len(series[u]) for u in ids]
    return pd.DataFrame(
        {
            "unique_id": np.repeat(np.asarray(ids, dtype=object), lens),
            "ds": np.concatenate([np.arange(n) for n in lens]),
            "y": np.concatenate([np.asarray(series[u], dtype=float) for u in ids]),
        }
    )


def _extract_preds(out: pd.DataFrame, model_name: str, ids: list[str]) -> dict[str, np.ndarray]:
    """Pull per-series forecast arrays out of a statsforecast output frame.

    Output layout: flat DataFrame with columns unique_id, ds, <model_name>.
    """
    col = model_name if model_name in out.columns else [
        c for c in out.columns if c not in ("unique_id", "ds")
    ][0]
    preds: dict[str, np.ndarray] = {}
    for uid, grp in out.groupby("unique_id", sort=False):
        preds[str(uid)] = grp.sort_values("ds")[col].to_numpy(dtype=float)
    return preds


class StatsForecastForecaster(Forecaster):
    def __init__(self, name: str, factory, context_len: int = 512) -> None:
        self.name = name
        self.kind = "classical"
        self._factory = factory
        self.context_len = context_len

    def forecast(self, series: dict[str, np.ndarray], h: int, m: int) -> dict[str, np.ndarray]:
        # Cap context to the same window the foundation models receive: keeps
        # the pool comparable and bounds fit time on long M4 series.
        series = {u: arr[-self.context_len :] for u, arr in series.items()}
        df = _long_df(series)
        model = self._factory(m)
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            sf = StatsForecast(models=[model], freq=1, n_jobs=8)
            out = sf.forecast(h=h, df=df)
        preds = _extract_preds(out, type(model).__name__, list(series))
        return {u: preds[u] for u in series}


def make_naive2(m: int) -> SeasonalNaive:
    """Naive2 = seasonal naive (equals plain naive when m == 1)."""
    return SeasonalNaive(season_length=max(m, 1))


def make_theta(m: int) -> AutoTheta:
    return AutoTheta(season_length=max(m, 1))


def make_ets(m: int) -> AutoETS:
    return AutoETS(season_length=max(m, 1))


def make_arima(m: int) -> AutoARIMA:
    return AutoARIMA(season_length=max(m, 1))


CLASSICAL_MODELS = {
    "naive2": make_naive2,
    "autotheta": make_theta,
    "autoets": make_ets,
    "autoarima": make_arima,
}
