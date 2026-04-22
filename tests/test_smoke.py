from __future__ import annotations

import numpy as np
import pandas as pd

from cvae_spx.config import BaselineConfig, EvaluationConfig, QuantileConfig, SplitConfig, TrainingConfig
from cvae_spx.features import apply_train_only_regimes, build_feature_frame, time_split, walk_forward_splits
from cvae_spx.models.quantile_regressor import ConditionalQuantileRegressor
from cvae_spx.pipeline import run_training_from_features


def _synthetic_master_panel(n: int = 520, seed: int = 7) -> pd.DataFrame:
    rng = np.random.default_rng(seed)
    dates = pd.bdate_range("2015-01-01", periods=n)

    market_shock = rng.normal(0.0, 0.009, size=n)
    spx = 2000.0 * np.exp(np.cumsum(market_shock))
    vix = np.clip(18.0 + np.abs(rng.normal(0.0, 4.0, size=n)).cumsum() / np.sqrt(np.arange(1, n + 1)), 8.0, None)
    skew = 120.0 + rng.normal(0.0, 2.5, size=n).cumsum() / np.sqrt(np.arange(1, n + 1))
    vvix = 90.0 + np.abs(rng.normal(0.0, 3.0, size=n)).cumsum() / np.sqrt(np.arange(1, n + 1))
    spy_close = 0.1 * spx + rng.normal(0.0, 2.0, size=n)
    intraday_scale = np.clip(np.abs(rng.normal(0.015, 0.005, size=n)), 0.002, 0.05)
    spy_high = spy_close * (1.0 + intraday_scale)
    spy_low = spy_close * (1.0 - intraday_scale)
    spy_open = spy_close * (1.0 + rng.normal(0.0, 0.002, size=n))
    spy_volume = np.exp(rng.normal(18.5, 0.25, size=n)).astype(float)
    dgs2 = 1.5 + rng.normal(0.0, 0.01, size=n).cumsum()
    dgs10 = 2.5 + rng.normal(0.0, 0.01, size=n).cumsum()
    fed_funds = np.repeat([0.25, 1.0, 2.5, 4.0], repeats=n // 4 + 1)[:n]

    return pd.DataFrame(
        {
            "spx": spx,
            "vix": vix,
            "skew": skew,
            "vvix": vvix,
            "spy_open": spy_open,
            "spy_high": spy_high,
            "spy_low": spy_low,
            "spy_close": spy_close,
            "spy_volume": spy_volume,
            "dgs2": dgs2,
            "dgs10": dgs10,
            "fed_funds": fed_funds,
        },
        index=dates,
    )


def test_walk_forward_splits_are_chronological() -> None:
    features = build_feature_frame(_synthetic_master_panel())
    folds = walk_forward_splits(features, SplitConfig(), n_folds=3)

    assert len(folds) == 3
    for fold in folds:
        assert fold["train"].index.max() < fold["val"].index.min()
        assert fold["val"].index.max() < fold["test"].index.min()


def test_train_only_regime_fit_applies_forward() -> None:
    features = build_feature_frame(_synthetic_master_panel())
    splits = time_split(features, SplitConfig())
    labeled, stats = apply_train_only_regimes(splits)

    assert stats["method"] == "train_only_zscore_blend_with_train_quantile_buckets"
    assert set(labeled["test"]["regime"].unique()).issubset({"calm", "normal", "stressed"})


def test_quantile_regressor_requires_median_anchor() -> None:
    try:
        ConditionalQuantileRegressor(context_size=10, cond_dim=4, quantiles=(0.05, 0.95))
    except ValueError as exc:
        assert "0.5" in str(exc)
    else:  # pragma: no cover
        raise AssertionError("Expected missing-median quantiles to raise ValueError.")


def test_pipeline_smoke() -> None:
    features = build_feature_frame(_synthetic_master_panel())
    config = TrainingConfig(
        split=SplitConfig(),
        baseline=BaselineConfig(
            xgb_n_estimators=40,
            xgb_max_depth=3,
            xgb_learning_rate=0.05,
            xgb_subsample=0.9,
            xgb_colsample_bytree=0.9,
            xgb_reg_lambda=1.0,
            xgb_min_child_weight=1.0,
            xgb_early_stopping_rounds=5,
        ),
        quantile=QuantileConfig(
            window_size=10,
            hidden_sizes=(16, 8),
            dropout=0.0,
            learning_rate=1e-3,
            batch_size=64,
            max_epochs=3,
            patience=3,
            weight_decay=0.0,
            gradient_clip_norm=1.0,
            stressed_loss_weight=2.0,
        ),
        evaluation=EvaluationConfig(
            interval_alpha=0.10,
            lower_tail_prob=0.05,
            kupiec_significance=0.05,
            empirical_min_obs=25,
        ),
    )

    results = run_training_from_features(
        features=features,
        training_config=config,
        seed=0,
        walk_forward_folds=2,
        save_outputs=False,
        make_plots=False,
    )

    assert results["model_name"] == "xgboost_quantile"
    assert "single_split" in results
    assert "walk_forward" in results
    assert results["single_split"]["evaluation"]["prediction_alignment"]["n_scored_rows"] == results["single_split"]["test_rows"]
    assert set(results["single_split"]["evaluation"]["interval_metrics"]["model"].keys()) == {"calm", "normal", "stressed"}
