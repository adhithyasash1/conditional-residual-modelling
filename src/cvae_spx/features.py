"""Feature engineering and split construction for SPX uncertainty modeling."""
from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd

from .config import PROCESSED_DIR, REQUIRED_ALT_FEATURE_COLS, SplitConfig, TARGET_COL, get_logger

logger = get_logger(__name__)


def add_spx_return_features(df: pd.DataFrame) -> pd.DataFrame:
    out = df.copy()
    out["spx_ret"] = out[TARGET_COL].pct_change()
    out["spx_logret"] = np.log(out[TARGET_COL]).diff()
    return out


def add_lag_features(df: pd.DataFrame, columns: list[str], lags: list[int]) -> pd.DataFrame:
    out = df.copy()
    for col in columns:
        if col not in out.columns:
            raise KeyError(f"Required lag feature column missing: {col}")
        for lag in lags:
            out[f"{col}_lag{lag}"] = out[col].shift(lag)
    return out


def add_rolling_spx_features(
    df: pd.DataFrame,
    windows: tuple[int, ...] = (5, 20, 60),
) -> pd.DataFrame:
    out = df.copy()
    for window in windows:
        out[f"spx_vol{window}"] = out["spx_logret"].rolling(window).std() * np.sqrt(252.0)
        out[f"spx_ma{window}"] = out[TARGET_COL].rolling(window).mean()
    running_max = out[TARGET_COL].cummax()
    out["spx_drawdown"] = out[TARGET_COL] / running_max - 1.0
    return out


def add_alt_data_features(df: pd.DataFrame) -> pd.DataFrame:
    out = df.copy()
    required_cols = {"skew", "vvix", "spy_high", "spy_low", "spy_close", "spy_volume"}
    missing = sorted(required_cols.difference(out.columns))
    if missing:
        raise KeyError(f"Missing required alt-data source columns: {missing}")

    out["skew_pctchg5"] = out["skew"].pct_change(5)
    out["vvix_pctchg5"] = out["vvix"].pct_change(5)
    out["spy_intraday_range"] = (out["spy_high"] - out["spy_low"]) / out["spy_close"]
    out["spy_intraday_range_mean20"] = out["spy_intraday_range"].rolling(20).mean()

    intraday_std20 = out["spy_intraday_range"].rolling(20).std()
    out["spy_intraday_range_z20"] = (
        (out["spy_intraday_range"] - out["spy_intraday_range_mean20"]) / intraday_std20.replace(0.0, np.nan)
    )

    out["spy_log_volume"] = np.log(out["spy_volume"].replace(0.0, np.nan))
    log_vol_mean20 = out["spy_log_volume"].rolling(20).mean()
    log_vol_std20 = out["spy_log_volume"].rolling(20).std()
    out["spy_log_volume_z20"] = (
        (out["spy_log_volume"] - log_vol_mean20) / log_vol_std20.replace(0.0, np.nan)
    )
    return out


def fit_regime_proxy(train_df: pd.DataFrame) -> dict[str, object]:
    """Fit train-only z-score blend thresholds for calm/normal/stressed buckets."""
    components: list[dict[str, float | str]] = []
    z_parts: list[pd.Series] = []
    for col, sign in (("spx_vol20", 1.0), ("vix", 1.0), ("spx_drawdown", -1.0)):
        if col not in train_df.columns:
            raise KeyError(f"Regime component column missing: {col}")
        series = train_df[col].astype(float)
        mean = float(series.mean())
        std = float(series.std(ddof=0) + 1e-12)
        components.append({"column": col, "mean": mean, "std": std, "sign": sign})
        z_parts.append(sign * ((series - mean) / std))

    score = pd.concat(z_parts, axis=1).mean(axis=1)
    return {
        "components": components,
        "low_threshold": float(score.quantile(0.33)),
        "high_threshold": float(score.quantile(0.67)),
        "method": "train_only_zscore_blend_with_train_quantile_buckets",
    }


def apply_regime_proxy(df: pd.DataFrame, stats: dict[str, object]) -> pd.DataFrame:
    out = df.copy()
    z_parts: list[pd.Series] = []
    for spec in stats["components"]:
        col = str(spec["column"])
        mean = float(spec["mean"])
        std = float(spec["std"])
        sign = float(spec["sign"])
        z_parts.append(sign * ((out[col].astype(float) - mean) / (std + 1e-12)))

    score = pd.concat(z_parts, axis=1).mean(axis=1)
    labels = pd.Series("normal", index=out.index, dtype=object)
    labels[score <= float(stats["low_threshold"])] = "calm"
    labels[score >= float(stats["high_threshold"])] = "stressed"
    out["regime_score"] = score
    out["regime"] = labels
    return out


def apply_train_only_regimes(splits: dict[str, pd.DataFrame]) -> tuple[dict[str, pd.DataFrame], dict[str, object]]:
    clean = {
        name: split.drop(columns=["regime", "regime_score"], errors="ignore").copy()
        for name, split in splits.items()
    }
    stats = fit_regime_proxy(clean["train"])
    return {name: apply_regime_proxy(split, stats) for name, split in clean.items()}, stats


def build_feature_frame(df: pd.DataFrame) -> pd.DataFrame:
    out = df.copy()
    out = add_spx_return_features(out)
    out = add_rolling_spx_features(out)
    out = add_lag_features(out, ["spx", "spx_logret", "vix", "dgs10"], [1, 2, 5])
    out = add_alt_data_features(out)
    out["target_next_logret"] = out["spx_logret"].shift(-1)
    out = out.dropna().copy()

    missing_alt = [col for col in REQUIRED_ALT_FEATURE_COLS if col not in out.columns]
    if missing_alt:
        raise KeyError(f"Required alt-data features missing after feature build: {missing_alt}")

    logger.info(
        "Feature frame shape %s spanning %s to %s",
        out.shape,
        out.index.min(),
        out.index.max(),
    )
    return out


def time_split(df: pd.DataFrame, cfg: SplitConfig = SplitConfig()) -> dict[str, pd.DataFrame]:
    n = len(df)
    n_train = int(n * cfg.train_frac)
    n_val = int(n * cfg.val_frac)
    n_test = n - n_train - n_val
    if min(n_train, n_val, n_test) <= 0:
        raise ValueError("Split fractions produced an empty split.")
    return {
        "train": df.iloc[:n_train].copy(),
        "val": df.iloc[n_train : n_train + n_val].copy(),
        "test": df.iloc[n_train + n_val :].copy(),
    }


def walk_forward_splits(
    df: pd.DataFrame,
    cfg: SplitConfig = SplitConfig(),
    n_folds: int = 3,
) -> list[dict[str, pd.DataFrame]]:
    if n_folds < 1:
        raise ValueError("n_folds must be at least 1.")

    n = len(df)
    n_val = int(n * cfg.val_frac)
    n_test = int(n * cfg.test_frac)
    initial_train = n - n_val - n_folds * n_test
    if min(initial_train, n_val, n_test) <= 0:
        raise ValueError("Not enough rows for the requested walk-forward setup.")

    folds: list[dict[str, pd.DataFrame]] = []
    for fold in range(n_folds):
        train_end = initial_train + fold * n_test
        val_end = train_end + n_val
        test_end = val_end + n_test
        folds.append(
            {
                "train": df.iloc[:train_end].copy(),
                "val": df.iloc[train_end:val_end].copy(),
                "test": df.iloc[val_end:test_end].copy(),
            }
        )
    return folds


def save_features(df: pd.DataFrame, path: Path | None = None) -> Path:
    out_path = path or (PROCESSED_DIR / "features.csv")
    df.to_csv(out_path)
    logger.info("Saved features to %s", out_path)
    return out_path
