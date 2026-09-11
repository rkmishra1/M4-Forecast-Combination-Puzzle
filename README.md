<div align="center">

# Forecast Combination of Time-Series Foundation Models

**Do Chronos + TimesFM combinations beat the best single model — and can simple rules keep up with estimated weights?**

[![Python 3.12](https://img.shields.io/badge/python-3.12-3776AB.svg?logo=python&logoColor=white)](https://www.python.org/)
[![Dataset: M4](https://img.shields.io/badge/dataset-M4%20competition-8A2BE2.svg)](https://github.com/Mcompetitions/M4-methods)
[![Status: pilot](https://img.shields.io/badge/status-pilot%20%2F%20pre--submission-orange.svg)](#pilot--full-study-roadmap)

</div>

---

Empirical study for a planned *International Journal of Forecasting* submission:
**does combining time-series foundation models (Chronos, TimesFM) with each other
and with classical statistical methods beat the best individual model, and how
close do simple rules get to estimated-weight combinations?**

> This revisits the classical **forecast combination puzzle** (Bates & Granger 1969;
> Stock & Watson 2004; Timmermann 2006) — where simple averages are notoriously hard
> to beat — in the modern setting where the pool includes zero-shot foundation
> models whose relative performance varies widely across frequencies and series.

## Contents

- [Study design](#study-design)
- [Results (pilot)](#results-pilot)
- [Repository layout](#repository-layout)
- [Usage](#usage)
- [Pilot → full study roadmap](#pilot--full-study-roadmap)
- [Hardware notes](#hardware-notes)
- [References](#references)

## Study design

| Aspect | Choice |
| --- | --- |
| **Data** | M4 competition (100k series, 6 frequencies). Pilot subsamples **150 series/frequency**, deterministic (seed 42). |
| **Evaluation window** | Official M4 test horizon. |
| **Validation window** | Final *h* observations of the official training data — used to estimate combination weights. |
| **Leakage control** | Weights use validation errors only. The oracle uses test errors and is reported **solely as an upper bound**. |

### Model pool (7)

| Model | Family | Notes |
| --- | --- | --- |
| `naive2` | benchmark | seasonal naive |
| `autotheta`, `autoets`, `autoarima` | classical statistical | `statsforecast` |
| `chronos-t5-small`, `chronos-t5-base` | foundation (zero-shot) | Amazon Chronos |
| `timesfm-2.5-200m` | foundation (zero-shot) | Google TimesFM 2.5 |

### Combination rules (9)

`equal` · `median` · `trim-best-of-k equal` · inverse validation error (`global` / `per-series`) ·
inverse rank (`global` / `per-series`) · `NNLS per-series stacking` · `best-single-on-validation` ·
`test oracle` (upper bound).

### Metrics

**MASE**, **sMAPE**, **RMSSE** (official M4/M5 definitions, scaled on the training portion),
pooled **OWA** relative to Naive2, and **mean per-series rank**.

## Results (pilot)

Pooled across all frequencies (150 series/freq). Lower is better; **OWA** is relative to Naive2.

| Method | MASE | sMAPE | RMSSE | OWA |
| --- | ---: | ---: | ---: | ---: |
| naive2 | 2.2008 | 0.1081 | 1.7439 | 1.0000 |
| autoarima | 1.8952 | 0.0967 | 1.4791 | 0.8777 |
| autoets | 2.0443 | 0.1039 | 1.6096 | 0.9448 |
| autotheta | 2.0352 | 0.1001 | 1.6119 | 0.9255 |
| chronos-t5-base | 1.8722 | 0.0843 | 1.4647 | 0.8154 |
| chronos-t5-small | 1.8255 | 0.0831 | 1.4430 | 0.7989 |
| timesfm-2.5-200m | 1.8293 | 0.0833 | 1.4348 | 0.8007 |
| combo/best_single_val | 1.8224 | 0.0881 | 1.4309 | 0.8216 |
| combo/equal | 1.8071 | 0.0842 | 1.4196 | 0.7998 |
| combo/median | 1.8131 | 0.0840 | 1.4245 | 0.8003 |
| combo/trim1_equal | 1.7795 | 0.0829 | 1.3986 | 0.7878 |
| combo/inv_error_global | 1.7834 | 0.0822 | 1.3997 | 0.7855 |
| **combo/inv_error_series** | **1.7584** | **0.0810** | **1.3785** | **0.7743** |
| combo/inv_rank_series | 1.7681 | 0.0823 | 1.3877 | 0.7823 |
| combo/nnls_series | 1.8258 | 0.0850 | 1.4286 | 0.8078 |
| _combo/oracle_best_ | _1.3665_ | _0.0623_ | _1.0988_ | _0.5988_ |

**Early read.** Equal weighting (OWA 0.7998) essentially ties the best single model
(`chronos-t5-small`, 0.7989) — the puzzle holds. The best *feasible* rule,
per-series inverse validation error (0.7743), beats the best single model by ~3%,
while per-series NNLS stacking underperforms it. The oracle (0.5988) shows large
remaining headroom. Per-frequency and per-rank tables are in
[`results/pilot1/summary.md`](results/pilot1/summary.md).

## Repository layout

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

Individual model forecasts are cached as parquet under `results/<run>/cache/`, so
interrupted runs resume and pool composition can be changed without recomputing
everything. Forecasts are sanitized (non-finite values replaced by the last
observation) before metrics.

## Pilot → full study roadmap

| # | Stage | Scope |
| --- | --- | --- |
| 1 | **Pilot** (this repo, CPU) | 150 series/freq × 6 freqs × 7 models × 9 rules — validate the pipeline, get first effect sizes and runtimes. |
| 2 | **Full M4** | All series; add `chronos-bolt-base`, `chronos-t5-large`, Moirai / Lag-Llama if hardware allows; M4-style OWA table. |
| 3 | **Statistical inference** | Diebold–Mariano tests with multiple-comparison correction (Harvey–Leybourne); per-frequency breakdowns; weight analysis (drift of NNLS weights from uniform). |
| 4 | **Ablations** | classical-only vs foundation-only vs mixed pools (does the puzzle behave differently *within* foundation models?); pool size *k* = 2…7; validation-window sensitivity. |
| 5 | **Writing** | Frame around the combination puzzle + zero-shot foundation models; IJF values large-scale, rigorous empirical work. |

## Hardware notes

Developed on Apple M5 Pro/ 24GB RAM: foundation models run on CPU with small batches;
forecasts are computed one model at a time with explicit GC between models.
TimesFM 2.5 (200M) and Chronos-T5 small/base fit comfortably; larger checkpoints
(Chronos-T5-large, TimesFM-500m) need ≥ 16 GB or a GPU machine.

## References

- Bates, J. M., & Granger, C. W. J. (1969). The combination of forecasts. *OR*, 20(4), 451–468.
- Stock, J. H., & Watson, M. W. (2004). Combination forecasts of output growth in a seven-country data set. *Journal of Forecasting*, 23(6), 405–430.
- Timmermann, A. (2006). Forecast combinations. In *Handbook of Economic Forecasting* (Vol. 1, pp. 135–196). Elsevier.
- Makridakis, S., Spiliotis, E., & Assimakopoulos, V. (2020). The M4 Competition: 100,000 time series and 61 forecasting methods. *International Journal of Forecasting*, 36(1), 54–74.
