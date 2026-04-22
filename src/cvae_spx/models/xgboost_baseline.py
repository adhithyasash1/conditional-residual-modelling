"""XGBoost point forecast baseline and naive comparators."""
from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
from xgboost import XGBRegressor

from ..config import MODELS_DIR, TARGET_COL, TARGET_LOGRET_COL, BaselineConfig, get_logger

logger = get_logger(__name__)

FEATURE_EXCLUDE = {
    TARGET_COL,
    TARGET_LOGRET_COL,
    "regime",
    "regime_score",
}


def select_feature_columns(df: pd.DataFrame) -> list[str]:
    return [
        col
        for col in df.columns
        if col not in FEATURE_EXCLUDE and df[col].dtype.kind in {"f", "i"}
    ]


def mae(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    return float(np.mean(np.abs(y_true - y_pred)))


def rmse(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    return float(np.sqrt(np.mean((y_true - y_pred) ** 2)))


def directional_accuracy(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    return float(np.mean(np.sign(y_true) == np.sign(y_pred)))


def forecast_metrics(y_true: np.ndarray, y_pred: np.ndarray) -> dict[str, float]:
    return {
        "mae": mae(y_true, y_pred),
        "rmse": rmse(y_true, y_pred),
        "directional_accuracy": directional_accuracy(y_true, y_pred),
    }


@dataclass
class BaselineArtifacts:
    model: XGBRegressor
    feature_cols: list[str]
    metrics: dict[str, dict[str, float]]
    predictions: dict[str, pd.Series]
    residuals: dict[str, pd.Series]
    naive_metrics: dict[str, dict[str, dict[str, float]]]
    train_target_mean: float


def _naive_benchmarks(y_train: np.ndarray, y_test: np.ndarray) -> dict[str, dict[str, float]]:
    train_mean = float(np.mean(y_train))
    zero_pred = np.zeros_like(y_test, dtype=float)
    mean_pred = np.full_like(y_test, fill_value=train_mean, dtype=float)
    return {
        "zero": forecast_metrics(y_test, zero_pred),
        "train_mean": forecast_metrics(y_test, mean_pred),
    }


def train_xgboost_baseline(
    splits: dict[str, pd.DataFrame],
    cfg: BaselineConfig = BaselineConfig(),
    seed: int = 42,
) -> BaselineArtifacts:
    feature_cols = select_feature_columns(splits["train"])
    logger.info("XGBoost baseline using %d feature columns", len(feature_cols))

    X_train = splits["train"][feature_cols].astype(float).values
    y_train = splits["train"][TARGET_LOGRET_COL].astype(float).values
    X_val = splits["val"][feature_cols].astype(float).values
    y_val = splits["val"][TARGET_LOGRET_COL].astype(float).values

    model = XGBRegressor(
        n_estimators=cfg.xgb_n_estimators,
        max_depth=cfg.xgb_max_depth,
        learning_rate=cfg.xgb_learning_rate,
        subsample=cfg.xgb_subsample,
        colsample_bytree=cfg.xgb_colsample_bytree,
        reg_lambda=cfg.xgb_reg_lambda,
        min_child_weight=cfg.xgb_min_child_weight,
        objective="reg:squarederror",
        tree_method="hist",
        n_jobs=1,
        random_state=seed,
        early_stopping_rounds=cfg.xgb_early_stopping_rounds,
    )
    model.fit(X_train, y_train, eval_set=[(X_val, y_val)], verbose=False)

    metrics: dict[str, dict[str, float]] = {}
    predictions: dict[str, pd.Series] = {}
    residuals: dict[str, pd.Series] = {}
    naive_metrics: dict[str, dict[str, dict[str, float]]] = {}

    for split_name, split_df in splits.items():
        X = split_df[feature_cols].astype(float).values
        y = split_df[TARGET_LOGRET_COL].astype(float).values
        y_hat = model.predict(X)
        metrics[split_name] = forecast_metrics(y, y_hat)
        predictions[split_name] = pd.Series(y_hat, index=split_df.index, name="y_hat")
        residuals[split_name] = pd.Series(y - y_hat, index=split_df.index, name="residual")
        naive_metrics[split_name] = _naive_benchmarks(y_train, y)
        logger.info("Baseline %s metrics: %s", split_name, metrics[split_name])

    return BaselineArtifacts(
        model=model,
        feature_cols=feature_cols,
        metrics=metrics,
        predictions=predictions,
        residuals=residuals,
        naive_metrics=naive_metrics,
        train_target_mean=float(np.mean(y_train)),
    )


def save_baseline_artifacts(
    artifacts: BaselineArtifacts,
    out_dir: Path = MODELS_DIR,
) -> Path:
    out_path = out_dir / "baseline.joblib"
    joblib.dump(
        {
            "model": artifacts.model,
            "feature_cols": artifacts.feature_cols,
            "metrics": artifacts.metrics,
            "naive_metrics": artifacts.naive_metrics,
            "train_target_mean": artifacts.train_target_mean,
        },
        out_path,
    )
    logger.info("Saved XGBoost artifact to %s", out_path)
    return out_path
