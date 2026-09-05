"""Experiment configuration."""

from __future__ import annotations

from pathlib import Path

import yaml

DEFAULTS = {
    "run_id": "run",
    "frequencies": ["hourly", "daily", "weekly", "monthly", "quarterly", "yearly"],
    "n_series": None,          # per frequency; None = use all
    "seed": 42,
    "min_context": 32,         # require len(val context) >= this (TimesFM minimum)
    "device": "cpu",
    "batch_size": 32,          # chronos inference batch size
    "context_len": 512,        # max context fed to foundation models
    "models": [
        "naive2",
        "autotheta",
        "autoets",
        "autoarima",
        "chronos-t5-small",
        "chronos-t5-base",
        "timesfm-2.5-200m",
    ],
    "combinations": [
        "equal",
        "median",
        "trim1_equal",
        "inv_error_global",
        "inv_error_series",
        "inv_rank_series",
        "nnls_series",
        "best_single_val",
        "oracle_best",
    ],
    "data_dir": "data",
    "results_dir": None,       # defaults to results/<run_id>
}


def load_config(path: str | Path) -> dict:
    with open(path) as f:
        user = yaml.safe_load(f) or {}
    cfg = dict(DEFAULTS)
    cfg.update(user)
    if cfg["results_dir"] is None:
        cfg["results_dir"] = str(Path("results") / cfg["run_id"])
    return cfg
