"""Smoke tests: metrics, data, each model family, and combination math.

Usage:
    .venv/bin/python scripts/smoke_test.py            # classical + metrics + combine
    .venv/bin/python scripts/smoke_test.py --fm       # also Chronos + TimesFM
"""

from __future__ import annotations

import argparse
from pathlib import Path
import sys

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from fcc.combine import combine_all
from fcc.data import load_m4
from fcc.metrics import mase, mase_scale, mse_scale, rmsse, smape
from fcc.models import build_model


def check(name, cond):
    status = "PASS" if cond else "FAIL"
    print(f"[{status}] {name}")
    if not cond:
        raise SystemExit(f"smoke test failed: {name}")


def test_metrics():
    train = np.arange(1.0, 11.0)  # 1..10, unit diffs
    check("mase_scale m=1", abs(mase_scale(train, 1) - 1.0) < 1e-12)
    check("mse_scale m=1", abs(mse_scale(train, 1) - 1.0) < 1e-12)
    check("mase known value", abs(mase(np.array([11.0]), np.array([10.0]), 1.0) - 1.0) < 1e-12)
    check("rmsse known value", abs(rmsse(np.array([11.0]), np.array([9.0]), 1.0) - 2.0) < 1e-12)
    check("smape known value", abs(smape(np.array([10.0]), np.array([11.0])) - 2.0 / 21.0) < 1e-12)


def test_combine():
    ids = ["s1", "s2"]
    models = ["a", "b"]
    val_t = {"s1": np.array([2.0, 2.0]), "s2": np.array([1.0, 1.0])}
    val_f = {
        "a": {"s1": np.array([1.0, 1.0]), "s2": np.array([3.0, 3.0])},
        "b": {"s1": np.array([3.0, 3.0]), "s2": np.array([1.0, 1.0])},
    }
    test_f = {
        "a": {"s1": np.array([10.0, 10.0]), "s2": np.array([30.0, 30.0])},
        "b": {"s1": np.array([30.0, 30.0]), "s2": np.array([10.0, 10.0])},
    }
    errs = {"a": {"s1": 1.0, "s2": 2.0}, "b": {"s1": 1.0, "s2": 2.0}}
    combos, wrows = combine_all(
        rules=["equal", "median", "inv_error_series", "nnls_series", "best_single_val"],
        model_names=models,
        ids=ids,
        test_forecasts=test_f,
        val_forecasts=val_f,
        val_targets=val_t,
        val_errors=errs,
    )
    check("equal == mean", np.allclose(combos["equal"]["s1"], [20.0, 20.0]))
    check("median", np.allclose(combos["median"]["s2"], [20.0, 20.0]))
    check("inv_error_series weights sum 1", all(
        abs(sum(r["weight"] for r in wrows if r["rule"] == "inv_error_series" and r["unique_id"] == u) - 1) < 1e-9
        for u in ids
    ))
    check("best_single_val picks argmin", np.allclose(combos["best_single_val"]["s2"], [30.0, 30.0]))


def _sample():
    data = load_m4("weekly", n_series=5, seed=1)
    contexts = {u: data.splits(u)["val_context"] for u in data.series}
    return data, contexts


def test_classical():
    data, contexts = _sample()
    for name in ["naive2", "autotheta", "autoets", "autoarima"]:
        model = build_model(name, {"device": "cpu", "batch_size": 16, "context_len": 512})
        preds = model.forecast(contexts, data.h, data.m)
        ok = set(preds) == set(contexts) and all(
            preds[u].shape == (data.h,) and np.all(np.isfinite(preds[u])) for u in preds
        )
        check(f"classical {name}", ok)


def test_foundation():
    data, contexts = _sample()
    cfg = {"device": "cpu", "batch_size": 16, "context_len": 512}
    for name in ["chronos-t5-small", "timesfm-2.5-200m"]:
        model = build_model(name, cfg)
        preds = model.forecast(contexts, data.h, data.m)
        ok = set(preds) == set(contexts) and all(
            preds[u].shape == (data.h,) and np.all(np.isfinite(preds[u])) for u in preds
        )
        check(f"foundation {name}", ok)


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--fm", action="store_true", help="include foundation models")
    args = ap.parse_args()
    test_metrics()
    test_combine()
    test_classical()
    if args.fm:
        test_foundation()
    print("\nAll smoke tests passed.")
