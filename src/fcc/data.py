"""M4 competition dataset loading.

M4 conventions (github.com/Mcompetitions/M4-methods):
- Train CSV: the official training portion of each series (ragged rows).
- Test CSV:  ONLY the h holdout values per series (the future to forecast).

The full series is reconstructed as train ++ test. Derived splits, for a
series with official train length T and M4 horizon h:

    test context = y[:T]      test target  = y[T:T+h]    (official M4 eval)
    val  context = y[:T-h]    val  target  = y[T-h:T]    (combination-weight estimation)
"""

from __future__ import annotations

import urllib.request
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd

M4_BASE_URL = "https://raw.githubusercontent.com/Mcompetitions/M4-methods/master/Dataset"

# M4 forecast horizon (h) and seasonality (m) per frequency.
# m follows the official M4 evaluation: Daily and Weekly use m=1 (no seasonality).
FREQ_META: dict[str, dict[str, int]] = {
    "yearly": {"h": 6, "m": 1},
    "quarterly": {"h": 8, "m": 4},
    "monthly": {"h": 18, "m": 12},
    "weekly": {"h": 13, "m": 1},
    "daily": {"h": 14, "m": 1},
    "hourly": {"h": 48, "m": 24},  # h=48 per official M4-info.csv
}


@dataclass
class M4FreqData:
    freq: str
    h: int
    m: int
    series: dict[str, np.ndarray]  # full series (train + holdout)
    train_len: dict[str, int]      # official train length T per series

    def splits(self, uid: str) -> dict[str, np.ndarray]:
        y = self.series[uid]
        T = self.train_len[uid]
        h = self.h
        return {
            "train": y[:T],
            "val_context": y[: T - h],
            "val_target": y[T - h : T],
            "test_context": y[:T],
            "test_target": y[T : T + h],
        }


def download_m4(data_dir: str | Path) -> None:
    out = Path(data_dir) / "raw" / "m4"
    out.mkdir(parents=True, exist_ok=True)
    for freq in FREQ_META:
        title = freq.capitalize()
        for split in ("train", "test"):
            dest = out / f"{title}-{split}.csv"
            if dest.exists() and dest.stat().st_size > 0:
                continue
            url = f"{M4_BASE_URL}/{split.capitalize()}/{title}-{split}.csv"
            print(f"downloading {url}")
            urllib.request.urlretrieve(url, dest)


def _read_m4_csv(path: Path) -> dict[str, np.ndarray]:
    df = pd.read_csv(path, index_col=0)
    return {str(idx): row.dropna().to_numpy(dtype=np.float64) for idx, row in df.iterrows()}


def load_m4(
    freq: str,
    data_dir: str | Path = "data",
    n_series: int | None = None,
    seed: int = 42,
) -> M4FreqData:
    """Load one M4 frequency, optionally subsampling series (deterministic)."""
    if freq not in FREQ_META:
        raise ValueError(f"unknown frequency {freq!r}; expected one of {sorted(FREQ_META)}")
    h, m = FREQ_META[freq]["h"], FREQ_META[freq]["m"]
    raw = Path(data_dir) / "raw" / "m4"
    title = freq.capitalize()
    train_path = raw / f"{title}-train.csv"
    test_path = raw / f"{title}-test.csv"
    if not (train_path.exists() and test_path.exists()):
        download_m4(data_dir)
    train = _read_m4_csv(train_path)
    test = _read_m4_csv(test_path)

    ids = sorted(set(train) & set(test))
    if n_series is not None and n_series < len(ids):
        rng = np.random.default_rng(seed)
        idx = rng.choice(len(ids), size=n_series, replace=False)
        ids = [ids[i] for i in sorted(idx)]

    series: dict[str, np.ndarray] = {}
    train_len: dict[str, int] = {}
    n_malformed = 0
    for uid in ids:
        tr, te = train[uid], test[uid]
        T = len(tr)
        if len(te) != h or T <= h:
            n_malformed += 1
            continue
        series[uid] = np.concatenate([tr, te])
        train_len[uid] = T
    if n_malformed:
        print(f"[{freq}] skipped {n_malformed} malformed series (test len != h or train <= h)")
    return M4FreqData(freq=freq, h=h, m=m, series=series, train_len=train_len)
