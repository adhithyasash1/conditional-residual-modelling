# Conditional Residual Modelling

Conditional residual modelling for calibrated lower-tail uncertainty on next-day S&P 500 log returns.

The repo asks one question:

Can a conditional residual model put more uncertainty into stressed market states than a static empirical band, while still staying calibrated on the 5% lower tail?

This is an uncertainty project, not an alpha system.

## What the Project Keeps

The retained production path is:

1. Fit an XGBoost point forecast for next-day `spx_logret`
2. Model residual uncertainty with a feedforward conditional quantile regressor
3. Compare it against regime-conditional empirical residual quantiles
4. Judge both with lower-tail calibration, especially pooled walk-forward Kupiec tests

The quantile model predicts this ordered set directly:

`(0.025, 0.05, 0.25, 0.5, 0.75, 0.95, 0.975)`

The head is monotonic by construction:

- one learned median anchor
- non-negative softplus gaps
- cumulative sums into strictly ordered quantiles

That gives non-crossing quantiles without assuming a Gaussian or Student-t residual family.

## Why This Repo Exists

The original cVAE branch capped the tail too hard in stress. That was the wrong inductive bias for the problem.

This repo keeps the useful project structure from the reference implementation, but the active uncertainty model is now direct quantile regression on residuals. The lower 5% quantile is then calibrated with a one-sided validation conformal shift before test evaluation. That extra calibration step turned out to matter in pooled walk-forward coverage.

## Current Live Result

From the latest saved run in `data/processed/training_results.json`:

- pooled walk-forward unconditional lower-tail breach rate: `5.73%`, `p = 0.1253`
- pooled walk-forward stressed-only breach rate: `5.38%`, `p = 0.7282`
- acceptance bar passed: `True`

Single-split results are also saved, but the walk-forward result is the real bar here because it exposes drift.

## Important Caveat

The current uncertainty pipeline clears the lower-tail acceptance bar, but the XGBoost point forecast does **not** beat the better naive benchmark on the single split and on one walk-forward fold in the current run.

That is left visible on purpose. The repo does not hide tradeoffs.

## Data

### Yahoo Finance

- `^GSPC` for SPX
- `^VIX`
- `^SKEW`
- `^VVIX`
- `SPY` OHLCV

### FRED

- `DGS2`
- `DGS10`
- `FEDFUNDS`

### Alignment Rules

- The daily index is anchored on actual SPX trading dates
- The target is never forward-filled
- Non-target daily series are forward-filled with a 10-day cap
- `FEDFUNDS` is forward-filled without a cap because it is stepwise by design

## Features

### Standard SPX Block

- SPX log returns
- Lags `[1, 2, 5]` of `spx`, `spx_logret`, `vix`, `dgs10`
- Rolling volatility over `(5, 20, 60)`
- Rolling moving averages over `(5, 20, 60)`
- Running-max drawdown

### Alt-Data Block

This block is load-bearing. It is not cosmetic.

- `skew` level
- `skew_pctchg5`
- `vvix` level
- `vvix_pctchg5`
- `spy_intraday_range = (High - Low) / Close`
- `spy_intraday_range_mean20`
- `spy_intraday_range_z20`
- `spy_log_volume_z20`

### Quantile Conditioning Row

The uncertainty model conditions on:

- `vix`
- `dgs2`
- `dgs10`
- `fed_funds`
- `spx_vol5`
- `spx_vol20`
- `spx_vol60`
- `spx_drawdown`
- `skew`
- `skew_pctchg5`
- `vvix`
- `vvix_pctchg5`
- `spy_intraday_range`
- `spy_intraday_range_mean20`
- `spy_intraday_range_z20`
- `spy_log_volume_z20`

It also uses a residual history window of length `10`.

## Regimes

Regimes are fit on the training slice only.

The regime score is the train-only z-score blend of:

- `spx_vol20`
- `vix`
- `-spx_drawdown`

Train-slice 33rd and 67th percentiles define:

- `calm`
- `normal`
- `stressed`

Those thresholds are then applied forward to validation and test.

The quantile regressor up-weights stressed observations through the pinball loss.

## Evaluation

Every run reports:

- baseline MAE / RMSE / directional accuracy
- naive comparators: zero and train-mean
- residual error by regime
- mean band width by regime
- Spearman correlation of width vs VIX
- interval coverage by regime
- Winkler interval score by regime
- Kupiec proportion-of-failures tests at the lower 5% tail
- pooled walk-forward Kupiec across 3 expanding folds

The acceptance bar is the pooled walk-forward lower-tail calibration:

- unconditional Kupiec must pass
- stressed-only Kupiec must pass

## Repo Layout

```text
conditional-residuals-modelling/
├── data/
│   ├── raw/
│   ├── interim/
│   └── processed/
├── models/
├── reports/
│   ├── figures/
│   └── RESULTS.md
├── scripts/
│   ├── download_data.py
│   ├── train.py
│   └── validate_run.py
├── src/conditional_residual_modelling/
│   ├── __init__.py
│   ├── config.py
│   ├── data.py
│   ├── features.py
│   ├── pipeline.py
│   ├── evaluation/
│   └── models/
└── tests/
```

## Quickstart

```bash
python3 -m venv .venv
source .venv/bin/activate
python -m pip install -r requirements.txt
python -m pip install -e .
cp .env.example .env
```

If you have a FRED API key, put it in `.env`. If you do not, the downloader falls back to the public FRED CSV endpoint.

## Standard Workflow

### 1. Download and rebuild data

```bash
source .venv/bin/activate
python scripts/download_data.py
```

This writes:

- `data/raw/market_macro_panel.csv`
- `data/processed/master_daily.csv`
- `data/processed/features.csv`

### 2. Train the pipeline

```bash
source .venv/bin/activate
python scripts/train.py
```

Useful flag:

```bash
python scripts/train.py --skip-plots
```

### 3. Validate saved outputs

```bash
source .venv/bin/activate
python scripts/validate_run.py
```

This script:

- rebuilds processed inputs
- retrains the models
- reloads `test_predictions.csv`
- recomputes Kupiec and band-width diagnostics
- raises if recomputed values disagree with the saved outputs

## Main Artifacts

After a normal run, the files that matter most are:

- `data/processed/master_daily.csv`
- `data/processed/features.csv`
- `data/processed/test_predictions.csv`
- `data/processed/training_results.json`
- `models/baseline.joblib`
- `models/quantile_regressor.pt`
- `reports/figures/quantile_training.png`
- `reports/figures/band_width_vs_vix.png`
- `reports/RESULTS.md`

## How to Read the Outputs

### `training_results.json`

This is the main machine-readable report. It includes:

- data summary
- single-split evaluation
- walk-forward fold details
- pooled Kupiec results
- calibration metadata for the lower conformal shift
- acceptance status

### `test_predictions.csv`

This is the easiest file to inspect by hand. It includes:

- realized next-day log return
- point forecast
- quantile columns
- final lower and upper model bands
- empirical comparator bands

### `RESULTS.md`

This is the shortest summary of whether the run passed the lower-tail bar.

## Reproducibility and Stability

- All randomness is seeded from `RANDOM_SEED`
- BLAS / OpenMP / Torch threads are capped to `1` in `src/conditional_residual_modelling/__init__.py`
- The package sets writable temp cache locations for matplotlib on locked-down macOS environments
- Tests cover alignment, risk metrics, and a synthetic end-to-end smoke run

## Current Data Window

In the latest saved run, the usable feature frame spans:

- start: `2007-01-10`
- end: `2026-04-20`
- rows: `4,849`

That start date is driven by the available overlap once the full alt-data block and rolling features are enforced.
