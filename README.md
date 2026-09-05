# Forecast Combination of Time-Series Foundation Models

Empirical study for a planned *International Journal of Forecasting* submission:
**does combining time-series foundation models (Chronos, TimesFM) with each other
and with classical statistical methods beat the best individual model, and how
close do simple rules get to estimated-weight combinations?**

This revisits the classical *forecast combination puzzle* (Bates & Granger 1969;
Stock & Watson 2004; Timmermann 2006) — where simple averages are notoriously
hard to beat — in the modern setting where the pool includes zero-shot
foundation models whose relative performance varies widely across frequencies
and series.

## Study design

- **Data**: M4 competition (100k series across 6 frequencies). The pilot
  subsamples 150 series per frequency (deterministic, seed 42).
- **Splits**: official M4 test horizon is the evaluation window. The final h
  observations of the official training data serve as a validation window for
  estimating combination weights (no test leakage: weights use validation
  errors only; the oracle uses test errors and is reported solely as an upper
  bound).
- **Model pool** (7):
  - `naive2` — seasonal naive benchmark
  - `autotheta`, `autoets`, `autoarima` — classical statistical (statsforecast)
  - `chronos-t5-small`, `chronos-t5-base` — Amazon Chronos (zero-shot)
  - `timesfm-2.5-200m` — Google TimesFM 2.5 (zero-shot)
- **Combination rules** (9): equal, median, trim-best-of-k equal, inverse
  validation error (global / per-series), inverse rank (global / per-series),
  NNLS per-series stacking, best-single-on-validation, and the test oracle.
- **Metrics**: MASE, sMAPE, RMSSE (official M4/M5 definitions, scaled on the
  training portion), pooled OWA relative to Naive2, and mean per-series rank.

## Layout

```
configs/pilot.yaml       experiment configuration
scripts/run.py           run experiment from a config
scripts/smoke_test.py    fast checks (+ --fm for foundation models)
src/fcc/
  data.py                M4 loading, train/val/test splits
  metrics.py             MASE / sMAPE / RMSSE / OWA
  models/                unified Forecaster interface + wrappers
  combine.py             combination rules
  pipeline.py            orchestration (forecast caching, eval)
  analyze.py             tables + figures
results/pilot1/          pilot outputs (summary.csv/.md, figs/, cache/)
data/raw/m4/             M4 CSVs (downloaded, not committed)
```

## Usage

```bash
uv venv --python 3.12 .venv
uv pip install --python .venv/bin/python -r requirements.txt
uv pip install --python .venv/bin/python -e . --no-deps
.venv/bin/python scripts/smoke_test.py --fm
.venv/bin/python scripts/run.py --config configs/pilot.yaml
```

Individual model forecasts are cached as parquet under `results/<run>/cache/`,
so interrupted runs resume and pool composition can be changed without
recomputing everything. Forecasts are sanitized (non-finite values replaced by
last observation) before metrics.

## Pilot → full study roadmap

1. **Pilot (this repo, CPU)**: 150 series/freq × 6 freqs × 7 models × 9 rules.
   Purpose: validate the pipeline, get first effect sizes and runtimes.
2. **Full M4**: all series, add `chronos-bolt-base`, `chronos-t5-large`,
   Moirai/Lag-Llama if feasible on target hardware; M4-style OWA table.
3. **Statistical inference**: Diebold-Mariano tests with multiple-comparison
   correction (Harvey-Leybourne); per-frequency breakdowns; weight analysis
   (how far NNLS weights drift from uniform).
4. **Ablations**: classical-only vs foundation-only vs mixed pools (does the
   puzzle behave differently within foundation models?); pool size k=2..7;
   validation-window sensitivity.
5. **Writing**: frame around combination puzzle + zero-shot foundation models;
   IJF values large-scale rigorous empirical work.

## Hardware notes

Developed on Apple M1 / 8 GB RAM: foundation models run on CPU with small
batches; forecasts are computed one model at a time with explicit GC between
models. TimesFM 2.5 (200M) and Chronos-T5 small/base fit comfortably; larger
checkpoints (Chronos-T5-large, TimesFM-500m) need ≥16 GB or a GPU machine.
