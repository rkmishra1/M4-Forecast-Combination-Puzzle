"""Experiment pipeline.

Per frequency:
1. load M4 data, filter series (min context length, valid metric scales)
2. produce validation and test forecasts for every pool model
   (cached to parquet so runs are resumable and models are loaded one at a
   time -- important on small-RAM machines)
3. estimate combination weights from validation errors
4. evaluate every individual model and combination rule on the test window

Outputs (under results_dir):
- metrics_per_series.csv  long table: freq, unique_id, method, mase, smape, rmsse
- val_errors.csv          validation MASE per model per series
- runtimes.csv            wall-clock seconds per (freq, model, split)
- weights.csv             tidy combination weights for weighted rules
- summary.csv / summary.md / figures via fcc.analyze
"""

from __future__ import annotations

import gc
import time
from pathlib import Path

import numpy as np
import pandas as pd

from .analyze import analyze
from .combine import combine_all
from .data import load_m4
from .metrics import mase, mase_scale, mse_scale, rmsse, smape
from .models import build_model


def _cache_load(path: Path) -> dict[str, np.ndarray]:
    df = pd.read_parquet(path)
    preds = {}
    for uid, grp in df.groupby("unique_id", sort=False):
        preds[str(uid)] = grp.sort_values("step")["y_pred"].to_numpy(dtype=float)
    return preds


def _cache_save(path: Path, preds: dict[str, np.ndarray]) -> None:
    rows = [(uid, i, float(v)) for uid, arr in preds.items() for i, v in enumerate(arr)]
    pd.DataFrame(rows, columns=["unique_id", "step", "y_pred"]).to_parquet(path, index=False)


def _sanitize_inplace(preds: dict[str, np.ndarray], h: int, contexts: dict) -> int:
    """Replace non-finite forecast values with the last observed context value."""
    n_fixed = 0
    for uid, arr in list(preds.items()):
        arr = np.asarray(arr, dtype=float).ravel()
        if arr.shape[0] != h:
            fill = arr[-1] if len(arr) else contexts[uid][-1]
            arr = np.pad(arr, (0, h - len(arr)), constant_values=fill)[:h]
        bad = ~np.isfinite(arr)
        if bad.any():
            arr[bad] = contexts[uid][-1]
            n_fixed += int(bad.sum())
        preds[uid] = arr
    return n_fixed


def run(cfg: dict) -> None:
    results_dir = Path(cfg["results_dir"])
    cache_dir = results_dir / "cache"
    cache_dir.mkdir(parents=True, exist_ok=True)

    runtime_rows: list[dict] = []
    valerr_rows: list[dict] = []
    metric_rows: list[dict] = []
    weight_rows: list[dict] = []

    for freq in cfg["frequencies"]:
        data = load_m4(freq, cfg["data_dir"], cfg["n_series"], cfg["seed"])
        h, m = data.h, data.m

        keep: dict[str, dict] = {}
        n_short = n_scale = 0
        for uid in data.series:
            sp = data.splits(uid)
            if len(sp["val_context"]) < cfg["min_context"]:
                n_short += 1
                continue
            s_test, s_test_sq = mase_scale(sp["train"], m), mse_scale(sp["train"], m)
            s_val, s_val_sq = mase_scale(sp["val_context"], m), mse_scale(sp["val_context"], m)
            if min(s_test, s_val, s_test_sq, s_val_sq) <= 1e-12 or not all(
                np.isfinite(x) for x in (s_test, s_val, s_test_sq, s_val_sq)
            ):
                n_scale += 1
                continue
            keep[uid] = {
                "sp": sp,
                "s_test": s_test,
                "s_test_sq": s_test_sq,
                "s_val": s_val,
            }
        print(
            f"[{freq}] h={h} m={m}: kept {len(keep)} series "
            f"(dropped {n_short} short, {n_scale} degenerate)",
            flush=True,
        )
        if not keep:
            continue
        ids = list(keep)
        contexts = {
            split: {uid: keep[uid]["sp"][f"{split}_context"] for uid in ids}
            for split in ("val", "test")
        }
        val_targets = {uid: keep[uid]["sp"]["val_target"] for uid in ids}
        test_targets = {uid: keep[uid]["sp"]["test_target"] for uid in ids}

        # ---- 1. individual model forecasts (cached, one model in memory at a time)
        val_preds: dict[str, dict[str, np.ndarray]] = {}
        test_preds: dict[str, dict[str, np.ndarray]] = {}
        for name in cfg["models"]:
            cpaths = {
                split: cache_dir / f"{freq}__{name}__{split}.parquet"
                for split in ("val", "test")
            }
            if all(p.exists() for p in cpaths.values()):
                val_preds[name] = _cache_load(cpaths["val"])
                test_preds[name] = _cache_load(cpaths["test"])
                print(f"[{freq}] {name}: loaded from cache", flush=True)
                continue
            model = build_model(name, cfg)
            for split in ("val", "test"):
                if cpaths[split].exists():
                    preds = _cache_load(cpaths[split])
                else:
                    t0 = time.perf_counter()
                    preds = model.forecast(contexts[split], h, m)
                    n_fixed = _sanitize_inplace(preds, h, contexts[split])
                    dt = time.perf_counter() - t0
                    _cache_save(cpaths[split], preds)
                    runtime_rows.append(
                        dict(
                            freq=freq,
                            model=name,
                            split=split,
                            seconds=round(dt, 2),
                            n_nonfinite_fixed=n_fixed,
                            n_series=len(preds),
                        )
                    )
                    print(
                        f"[{freq}] {name} {split}: {dt:.1f}s "
                        f"({n_fixed} non-finite values patched)",
                        flush=True,
                    )
                (val_preds if split == "val" else test_preds)[name] = preds
            del model
            gc.collect()

        # ---- 2. validation errors (weight estimation input)
        val_errors: dict[str, dict[str, float]] = {}
        for name in cfg["models"]:
            val_errors[name] = {}
            for uid in ids:
                e = mase(val_targets[uid], val_preds[name][uid], keep[uid]["s_val"])
                val_errors[name][uid] = e
                valerr_rows.append(
                    dict(freq=freq, unique_id=uid, model=name, val_mase=e)
                )

        # ---- 3. combinations
        combos, wrows = combine_all(
            rules=cfg["combinations"],
            model_names=cfg["models"],
            ids=ids,
            test_forecasts=test_preds,
            val_forecasts=val_preds,
            val_targets=val_targets,
            val_errors=val_errors,
        )
        for r in wrows:
            r["freq"] = freq
        weight_rows.extend(wrows)

        # ---- 4. test metrics for individual models
        test_mase: dict[str, dict[str, float]] = {}
        for name in cfg["models"]:
            test_mase[name] = {}
            for uid in ids:
                e = mase(test_targets[uid], test_preds[name][uid], keep[uid]["s_test"])
                test_mase[name][uid] = e
                metric_rows.append(
                    dict(
                        freq=freq,
                        unique_id=uid,
                        method=name,
                        kind="model",
                        mase=e,
                        smape=smape(test_targets[uid], test_preds[name][uid]),
                        rmsse=rmsse(
                            test_targets[uid], test_preds[name][uid], keep[uid]["s_test_sq"]
                        ),
                    )
                )

        # oracle_best: per series the best test-MASE model (upper bound)
        if "oracle_best" in combos or "oracle_best" in cfg["combinations"]:
            for uid in ids:
                best = min(cfg["models"], key=lambda mn: test_mase[mn][uid])
                metric_rows.append(
                    dict(
                        freq=freq,
                        unique_id=uid,
                        method="combo/oracle_best",
                        kind="oracle",
                        mase=test_mase[best][uid],
                        smape=smape(test_targets[uid], test_preds[best][uid]),
                        rmsse=rmsse(
                            test_targets[uid],
                            test_preds[best][uid],
                            keep[uid]["s_test_sq"],
                        ),
                    )
                )

        # ---- 5. test metrics for combinations
        for rule, preds in combos.items():
            for uid in ids:
                metric_rows.append(
                    dict(
                        freq=freq,
                        unique_id=uid,
                        method=f"combo/{rule}",
                        kind="combo",
                        mase=mase(test_targets[uid], preds[uid], keep[uid]["s_test"]),
                        smape=smape(test_targets[uid], preds[uid]),
                        rmsse=rmsse(
                            test_targets[uid], preds[uid], keep[uid]["s_test_sq"]
                        ),
                    )
                )

    pd.DataFrame(metric_rows).to_csv(results_dir / "metrics_per_series.csv", index=False)
    pd.DataFrame(valerr_rows).to_csv(results_dir / "val_errors.csv", index=False)
    if runtime_rows:
        pd.DataFrame(runtime_rows).to_csv(results_dir / "runtimes.csv", index=False)
    if weight_rows:
        pd.DataFrame(weight_rows).to_csv(results_dir / "weights.csv", index=False)
    print(f"\nSaved outputs under {results_dir}", flush=True)
    analyze(results_dir)
