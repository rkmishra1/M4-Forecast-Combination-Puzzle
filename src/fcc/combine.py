"""Forecast combination rules.

Every rule receives the same inputs:
- model_names: ordered list of pool models
- ids: series ids to combine
- test_forecasts / val_forecasts: dict[model][uid] -> point forecast, shape (h,)
- val_targets: dict[uid] -> actuals of the validation window, shape (h,)
- val_errors: dict[model][uid] -> validation MASE (lower is better)

and returns dict[rule][uid] -> combined test forecast, plus tidy weight rows
for the weight analysis (rule, unique_id, model, weight).

This deliberately revisits the classical "forecast combination puzzle"
(Simple & Weighted averages vs. estimated weights) in the setting where the
pool consists of time-series foundation models.
"""

from __future__ import annotations

import numpy as np
from scipy.optimize import nnls

EPS = 1e-6


def _weights_to_rows(rule: str, uid: str, model_names, w: np.ndarray) -> list[dict]:
    return [
        {"rule": rule, "unique_id": uid, "model": mname, "weight": float(wi)}
        for mname, wi in zip(model_names, w)
    ]


def _weighted(test_stack: np.ndarray, w: np.ndarray) -> np.ndarray:
    return w @ test_stack  # [h,]


def combine_all(
    rules: list[str],
    model_names: list[str],
    ids: list[str],
    test_forecasts: dict[str, dict[str, np.ndarray]],
    val_forecasts: dict[str, dict[str, np.ndarray]],
    val_targets: dict[str, np.ndarray],
    val_errors: dict[str, dict[str, float]],
) -> tuple[dict[str, dict[str, np.ndarray]], list[dict]]:
    k = len(model_names)
    combos: dict[str, dict[str, np.ndarray]] = {}
    weight_rows: list[dict] = []
    unknown = [r for r in rules if r not in RULE_FUNCS and r != "oracle_best"]
    if unknown:
        raise ValueError(f"unknown combination rules: {unknown}")

    # Pre-stack forecasts per series
    test_stack = {
        uid: np.vstack([test_forecasts[mn][uid] for mn in model_names]) for uid in ids
    }
    val_stack = {
        uid: np.vstack([val_forecasts[mn][uid] for mn in model_names]) for uid in ids
    }
    err_mat = np.array(
        [[val_errors[mn].get(uid, np.nan) for uid in ids] for mn in model_names]
    )  # [k, n]

    for rule in rules:
        if rule == "oracle_best":
            continue  # handled by the pipeline (needs test errors)
        combos[rule] = {}
        for j, uid in enumerate(ids):
            ts = test_stack[uid]
            w = np.full(k, 1.0 / k)
            if rule == "equal":
                comb = ts.mean(axis=0)
            elif rule == "median":
                comb = np.median(ts, axis=0)
            elif rule == "trim1_equal":
                e = err_mat[:, j]
                worst = int(np.nanargmax(e))
                mask = np.ones(k, dtype=bool)
                mask[worst] = False
                w = mask / mask.sum()
                comb = _weighted(ts, w)
            elif rule == "inv_error_global":
                mean_err = np.nanmean(err_mat, axis=1)
                w = 1.0 / (mean_err + EPS)
                w = w / w.sum()
                comb = _weighted(ts, w)
            elif rule == "inv_error_series":
                e = err_mat[:, j]
                w = 1.0 / (e + EPS)
                w = w / w.sum()
                comb = _weighted(ts, w)
            elif rule == "inv_rank_global":
                mean_err = np.nanmean(err_mat, axis=1)
                ranks = np.argsort(np.argsort(mean_err)) + 1.0  # 1 = best
                w = 1.0 / ranks
                w = w / w.sum()
                comb = _weighted(ts, w)
            elif rule == "inv_rank_series":
                e = err_mat[:, j]
                ranks = np.argsort(np.argsort(e)) + 1.0
                w = 1.0 / ranks
                w = w / w.sum()
                comb = _weighted(ts, w)
            elif rule == "nnls_series":
                A = val_stack[uid].T  # [h, k]
                b = val_targets[uid]
                try:
                    w_nnls, _ = nnls(A, b)
                    if w_nnls.sum() <= EPS:  # degenerate all-zero solution
                        w_nnls = np.full(k, 1.0 / k)
                    else:
                        w_nnls = w_nnls / w_nnls.sum()
                except Exception:
                    w_nnls = np.full(k, 1.0 / k)
                w = w_nnls
                comb = _weighted(ts, w)
            elif rule == "best_single_val":
                e = err_mat[:, j]
                best = int(np.nanargmin(e))
                w = np.zeros(k)
                w[best] = 1.0
                comb = ts[best].copy()
            else:  # pragma: no cover
                raise ValueError(rule)
            combos[rule][uid] = comb
            if rule != "median":
                weight_rows.extend(_weights_to_rows(rule, uid, model_names, w))
    return combos, weight_rows


RULE_FUNCS = {
    "equal",
    "median",
    "trim1_equal",
    "inv_error_global",
    "inv_error_series",
    "inv_rank_global",
    "inv_rank_series",
    "nnls_series",
    "best_single_val",
}
