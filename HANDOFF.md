# Handoff: run on a powerful machine

## What this project is

Empirical forecasting study for an International Journal of Forecasting (IJF)
submission: **"Forecast combination with time-series foundation models"** —
does combining zero-shot foundation models (Amazon Chronos, Google TimesFM)
with each other and with classical statistical methods (Theta, ETS, ARIMA)
beat the best individual model, and how close do simple rules (equal/median
averages) get to estimated-weight combinations (inverse-error, NNLS stacking)?
This revisits the classical "forecast combination puzzle" in the modern
foundation-model setting.

- **Data**: M4 competition (100k series, 6 frequencies, horizons per official
  M4-info.csv: yearly 6, quarterly 8, monthly 18, weekly 13, daily 14, hourly 48).
- **Evaluation**: official M4 test window. A held-out validation window (last h
  points of official training data) supplies combination weights — no test
  leakage. Metrics: MASE, sMAPE, RMSSE (official M4/M5 definitions), pooled
  OWA vs Naive2, mean per-series rank.
- **All models see the same context**: the most recent 512 observations
  (classical models included, for comparability and speed).

## Code state (what is verified)

- All smoke tests PASS on the dev machine (macOS, 8 GB RAM, CPU-only):
  metrics math, combination rules, naive2/theta/ets/arima, Chronos-T5-small,
  TimesFM-2.5-200M (both downloaded from HuggingFace and produced valid
  forecasts).
- M4 data auto-downloads from the M4-methods GitHub repo on first use
  (~260 MB) into `data/raw/m4/`.
- Forecasts are cached per (freq, model, split) as parquet under
  `results/<run_id>/cache/` — interrupted runs resume; safe to kill/restart.
- A partial pilot (150 series/freq) validated the pipeline end-to-end on
  daily data before being stopped for speed; the full run belongs on better
  hardware.

## Setup on the new machine

```bash
# Python 3.10-3.12 required (timesfm needs >=3.10)
cd IJF
uv venv --python 3.12 .venv          # or: python3.12 -m venv .venv
uv pip install --python .venv/bin/python -r requirements.txt
uv pip install --python .venv/bin/python -e . --no-deps
# (without uv: .venv/bin/pip install -r requirements.txt && .venv/bin/pip install -e . --no-deps)

# quick validation (downloads ~1 GB of HF checkpoints on first run)
.venv/bin/python scripts/smoke_test.py --fm
```

If CUDA GPU: `uv pip install torch --index-url https://download.pytorch.org/whl/cu121`
before chronos/timesfm, and set `device: cuda` in the config.

## Run order

```bash
# 1) pilot (~30-60 min on a big machine): validates everything at scale of 150 series/freq
.venv/bin/python scripts/run.py --config configs/pilot.yaml

# 2) FULL study: all ~100k M4 series, 9-model pool
.venv/bin/python scripts/run.py --config configs/full.yaml
```

Tune in `configs/full.yaml`: `device` (cuda/mps/cpu), `batch_size`,
`timesfm_batch`, `context_len`. On a GPU box set device: cuda, batch_size: 256,
timesfm_batch: 64. Classical models (statsforecast) run on CPU regardless;
AutoARIMA is the slowest classical piece even at 512 context — on a many-core
box you may raise `n_jobs` in `src/fcc/models/classical.py` (was pinned to 1
for sandbox safety).

Outputs land in `results/<run_id>/`: `summary.csv`, `summary.md`, `ranks.csv`,
`metrics_per_series.csv`, `val_errors.csv`, `weights.csv`, `runtimes.csv`,
`figs/mase_heatmap.png`, `figs/mean_ranks.png`.

## What to bring back to the dev machine

The `results/<run_id>/` directory (CSVs + figs, small once cache/ is excluded)
is all that's needed for analysis and paper drafting.

## Full-study roadmap (after full1)

1. Statistical inference: Diebold-Mariano tests w/ Harvey correction +
   multiple-comparison control; per-frequency breakdowns.
2. Weight analysis: distance of NNLS/inv-error weights from uniform; weight
   stability across frequencies.
3. Ablations: classical-only vs foundation-only vs mixed pools; pool size
   k = 2..9; validation-window length sensitivity.
4. Optional extra pools if hardware allows: Moirai, Lag-Llama, Tiny Time
   Mixers (would require new wrappers in `src/fcc/models/`).
5. Writing: IJF framing — large-scale, rigorous empirical comparison.
