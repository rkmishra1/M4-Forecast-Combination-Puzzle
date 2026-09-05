"""Forecast accuracy metrics following the official M4/M5 definitions.

- sMAPE: symmetric MAPE, M4 variant 200*|e|/(|y|+|yhat|) (reported as fraction).
- MASE:  mean absolute error scaled by the in-sample MAE of the seasonal-naive
         benchmark computed on the training portion (M4 definition).
- RMSSE: root mean squared error scaled by the in-sample MSE of the seasonal-
         naive benchmark (M5 definition).
- OWA:   Overall Weighted Average, the M4 summary metric, relative to Naive2.
"""

from __future__ import annotations

import numpy as np


def smape(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    y_true = np.asarray(y_true, dtype=float)
    y_pred = np.asarray(y_pred, dtype=float)
    denom = np.abs(y_true) + np.abs(y_pred)
    diff = np.abs(y_true - y_pred)
    terms = np.where(denom > 0, 2.0 * diff / np.maximum(denom, 1e-12), 0.0)
    return float(np.mean(terms))


def mase_scale(train: np.ndarray, m: int) -> float:
    """In-sample MAE of the seasonal-naive benchmark (MASE denominator)."""
    train = np.asarray(train, dtype=float)
    if len(train) <= m:
        return np.nan
    return float(np.mean(np.abs(train[m:] - train[:-m])))


def mse_scale(train: np.ndarray, m: int) -> float:
    """In-sample MSE of the seasonal-naive benchmark (RMSSE denominator)."""
    train = np.asarray(train, dtype=float)
    if len(train) <= m:
        return np.nan
    return float(np.mean((train[m:] - train[:-m]) ** 2))


def mase(y_true: np.ndarray, y_pred: np.ndarray, scale: float) -> float:
    if not np.isfinite(scale) or scale <= 1e-12:
        return np.nan
    y_true = np.asarray(y_true, dtype=float)
    y_pred = np.asarray(y_pred, dtype=float)
    return float(np.mean(np.abs(y_true - y_pred)) / scale)


def rmsse(y_true: np.ndarray, y_pred: np.ndarray, scale_sq: float) -> float:
    if not np.isfinite(scale_sq) or scale_sq <= 1e-12:
        return np.nan
    y_true = np.asarray(y_true, dtype=float)
    y_pred = np.asarray(y_pred, dtype=float)
    return float(np.sqrt(np.mean((y_true - y_pred) ** 2) / scale_sq))


def owa(model_smape: float, model_mase: float, bench_smape: float, bench_mase: float) -> float:
    """Overall Weighted Average relative to the Naive2 benchmark (M4 metric)."""
    return 0.5 * (model_smape / bench_smape + model_mase / bench_mase)
