"""End-to-end training and evaluation pipeline."""
from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from .config import (
    PROCESSED_DIR,
    QUANTILE_CONDITION_COLS,
    RANDOM_SEED,
    REPORTS_DIR,
    TARGET_LOGRET_COL,
    TrainingConfig,
    get_logger,
    set_seed,
)
from .data import build_master_dataset, download_all
from .evaluation.evaluate import (
    band_width_diagnostics,
    fit_regime_conditional_empirical_quantiles,
    interval_metrics_by_regime,
    kupiec_passes,
    maybe_kupiec,
    plot_band_width_vs_vix,
    plot_training_curves,
    pool_kupiec_records,
    predict_regime_conditional_empirical_quantiles,
    regime_error_table,
)
from .features import apply_train_only_regimes, build_feature_frame, save_features, time_split, walk_forward_splits
from .models.quantile_regressor import (
    QuantileArtifacts,
    predict_quantiles,
    save_quantile_artifacts,
    train_quantile_regressor,
)
from .models.xgboost_baseline import BaselineArtifacts, save_baseline_artifacts, train_xgboost_baseline

logger = get_logger(__name__)

MODEL_NAME = "xgboost_quantile"
MODEL_LABEL = "XGBoost + Conditional Quantiles"
COMPARATOR_NAME = "regime_conditional_empirical_quantiles"


def _sanitize(obj: Any) -> Any:
    if isinstance(obj, dict):
        return {str(key): _sanitize(value) for key, value in obj.items()}
    if isinstance(obj, list):
        return [_sanitize(value) for value in obj]
    if isinstance(obj, tuple):
        return [_sanitize(value) for value in obj]
    if isinstance(obj, (np.floating,)):
        return float(obj)
    if isinstance(obj, (np.integer,)):
        return int(obj)
    if isinstance(obj, (np.bool_,)):
        return bool(obj)
    if isinstance(obj, pd.Timestamp):
        return obj.isoformat()
    return obj


def _quantile_column(q: float) -> str:
    return f"q_{q:.3f}"


def _safe_nanmean(values: list[float]) -> float:
    arr = np.asarray(values, dtype=float)
    if arr.size == 0 or np.isnan(arr).all():
        return float("nan")
    return float(np.nanmean(arr))


def run_data_download() -> pd.DataFrame:
    return download_all()


def rebuild_processed_inputs(download: bool = False) -> tuple[pd.DataFrame, pd.DataFrame]:
    if download:
        run_data_download()
    master = build_master_dataset(download_if_missing=download)
    features = build_feature_frame(master)
    save_features(features)
    return master, features


def _load_or_build_features() -> pd.DataFrame:
    feature_path = PROCESSED_DIR / "features.csv"
    if feature_path.exists():
        return pd.read_csv(feature_path, parse_dates=["date"]).set_index("date").sort_index()
    master = build_master_dataset(download_if_missing=False)
    features = build_feature_frame(master)
    save_features(features)
    return features


def _select_quantile_indices(quantiles: tuple[float, ...], lower_tail_prob: float) -> tuple[int, int]:
    lower_idx = next((i for i, q in enumerate(quantiles) if np.isclose(q, lower_tail_prob)), None)
    upper_idx = next((i for i, q in enumerate(quantiles) if np.isclose(q, 1.0 - lower_tail_prob)), None)
    if lower_idx is None or upper_idx is None:
        raise ValueError(
            f"Quantile set must include {lower_tail_prob} and {1.0 - lower_tail_prob}. Got {quantiles}."
        )
    return lower_idx, upper_idx


def _resolve_quantile_condition_cols(df: pd.DataFrame) -> list[str]:
    missing = [col for col in QUANTILE_CONDITION_COLS if col not in df.columns]
    if missing:
        raise KeyError(f"Quantile conditioning columns missing from feature frame: {missing}")
    return list(QUANTILE_CONDITION_COLS)


def _score_quantiles_on_split(
    splits: dict[str, pd.DataFrame],
    baseline: BaselineArtifacts,
    quantile_artifacts: QuantileArtifacts,
    split_name: str,
    lower_tail_prob: float,
) -> tuple[pd.DataFrame, dict[str, int]]:
    if split_name not in {"val", "test"}:
        raise ValueError("split_name must be 'val' or 'test'.")

    if split_name == "val":
        combined = pd.concat([splits["train"], splits["val"]], axis=0)
        combined_residuals = np.concatenate(
            [
                baseline.residuals["train"].astype(float).values,
                baseline.residuals["val"].astype(float).values,
            ]
        )
        split_start = len(splits["train"])
        split_stop = split_start + len(splits["val"])
        target_frame = splits["val"]
        split_predictions = baseline.predictions["val"]
        split_residuals = baseline.residuals["val"]
    else:
        combined = pd.concat([splits["train"], splits["val"], splits["test"]], axis=0)
        combined_residuals = np.concatenate(
            [
                baseline.residuals["train"].astype(float).values,
                baseline.residuals["val"].astype(float).values,
                baseline.residuals["test"].astype(float).values,
            ]
        )
        split_start = len(splits["train"]) + len(splits["val"])
        split_stop = split_start + len(splits["test"])
        target_frame = splits["test"]
        split_predictions = baseline.predictions["test"]
        split_residuals = baseline.residuals["test"]

    combined_conditions = combined[quantile_artifacts.cond_cols].astype(float).values
    quantile_preds, offsets, quantiles = predict_quantiles(
        quantile_artifacts,
        combined_residuals,
        combined_conditions,
    )

    mask = (offsets >= split_start) & (offsets < split_stop)
    quantile_preds = quantile_preds[mask]
    offsets = offsets[mask]
    split_rows = offsets - split_start

    scored = target_frame.iloc[split_rows].copy()
    scored["y_hat"] = split_predictions.iloc[split_rows].astype(float).values
    scored["residual"] = split_residuals.iloc[split_rows].astype(float).values

    for idx, q in enumerate(quantiles):
        scored[_quantile_column(q)] = scored["y_hat"].values + quantile_preds[:, idx]

    lower_idx, upper_idx = _select_quantile_indices(quantiles, lower_tail_prob)
    scored["band_lower"] = scored[_quantile_column(quantiles[lower_idx])]
    scored["band_upper"] = scored[_quantile_column(quantiles[upper_idx])]

    alignment = {
        "split_name": split_name,
        "n_split_rows": int(len(target_frame)),
        "n_scored_rows": int(len(scored)),
        "context_rows_needed": int(quantile_artifacts.config.window_size),
        "split_start_offset": int(split_start),
    }
    return scored, alignment


def _fit_lower_conformal_shift(
    scored_val: pd.DataFrame,
    lower_tail_prob: float,
) -> dict[str, object]:
    raw_lower = scored_val["band_lower"].astype(float).values
    y_true = scored_val[TARGET_LOGRET_COL].astype(float).values
    scores = raw_lower - y_true
    scores = np.asarray(scores, dtype=float)
    n = len(scores)
    if n == 0:
        return {"shift": 0.0, "n_calibration_rows": 0}
    rank = min(n - 1, int(np.ceil((n + 1) * (1.0 - lower_tail_prob))) - 1)
    shift = float(max(0.0, np.sort(scores)[rank]))
    return {"shift": shift, "n_calibration_rows": int(n)}


def _apply_lower_shift(
    scored: pd.DataFrame,
    shift: float,
    quantiles: tuple[float, ...],
) -> pd.DataFrame:
    adjusted = scored.copy()
    for q in quantiles:
        if q <= 0.05 + 1e-12:
            adjusted[_quantile_column(q)] = adjusted[_quantile_column(q)] - shift
    adjusted["band_lower"] = adjusted["band_lower"] - shift
    return adjusted


def _build_empirical_predictions(
    splits: dict[str, pd.DataFrame],
    baseline: BaselineArtifacts,
    quantiles: tuple[float, ...],
    min_obs: int,
) -> tuple[dict[str, object], np.ndarray]:
    lookup = fit_regime_conditional_empirical_quantiles(
        residuals=baseline.residuals["train"].astype(float).values,
        regime=splits["train"]["regime"].astype(str),
        quantiles=quantiles,
        min_obs=min_obs,
    )
    empirical_preds, _ = predict_regime_conditional_empirical_quantiles(
        y_hat=baseline.predictions["test"].astype(float).values,
        regime=splits["test"]["regime"].astype(str),
        lookup=lookup,
    )
    return lookup, empirical_preds


def _split_point_forecast_summary(baseline: BaselineArtifacts, split_name: str) -> dict[str, object]:
    baseline_metrics = baseline.metrics[split_name]
    naive_metrics = baseline.naive_metrics[split_name]
    better_naive_mae = min(metric["mae"] for metric in naive_metrics.values())
    return {
        "baseline": baseline_metrics,
        "naive": naive_metrics,
        "better_naive_mae": float(better_naive_mae),
        "beats_better_naive": bool(baseline_metrics["mae"] < better_naive_mae),
    }


def evaluate_split(
    splits: dict[str, pd.DataFrame],
    baseline: BaselineArtifacts,
    quantile_artifacts: QuantileArtifacts,
    training_config: TrainingConfig,
    save_predictions_to: Path | None = None,
    make_plots: bool = True,
) -> dict[str, Any]:
    eval_cfg = training_config.evaluation
    quantile_quantiles = quantile_artifacts.quantiles
    lower_idx, upper_idx = _select_quantile_indices(quantile_quantiles, eval_cfg.lower_tail_prob)

    scored_test, alignment = _score_quantiles_on_split(
        splits=splits,
        baseline=baseline,
        quantile_artifacts=quantile_artifacts,
        split_name="test",
        lower_tail_prob=eval_cfg.lower_tail_prob,
    )
    if len(scored_test) != len(splits["test"]):
        raise ValueError(
            "Test scoring did not cover the full test slice. "
            "This usually means the history window was not seeded correctly."
        )

    calibration_info = {"shift": 0.0, "n_calibration_rows": 0, "applied": False}
    if eval_cfg.apply_lower_conformal_shift:
        scored_val, _ = _score_quantiles_on_split(
            splits=splits,
            baseline=baseline,
            quantile_artifacts=quantile_artifacts,
            split_name="val",
            lower_tail_prob=eval_cfg.lower_tail_prob,
        )
        calibration_info = _fit_lower_conformal_shift(scored_val, eval_cfg.lower_tail_prob)
        calibration_info["applied"] = bool(calibration_info["shift"] > 0.0)
        scored_test = _apply_lower_shift(scored_test, calibration_info["shift"], quantile_quantiles)

    empirical_lookup, empirical_preds = _build_empirical_predictions(
        splits=splits,
        baseline=baseline,
        quantiles=quantile_quantiles,
        min_obs=eval_cfg.empirical_min_obs,
    )

    empirical_lower = empirical_preds[:, lower_idx]
    empirical_upper = empirical_preds[:, upper_idx]
    scored_test["empirical_band_lower"] = empirical_lower
    scored_test["empirical_band_upper"] = empirical_upper

    y_true = scored_test[TARGET_LOGRET_COL].astype(float).values
    y_hat = scored_test["y_hat"].astype(float).values
    regime = scored_test["regime"].astype(str)
    vix = scored_test["vix"].astype(float)
    model_lower = scored_test["band_lower"].astype(float).values
    model_upper = scored_test["band_upper"].astype(float).values

    point_summary = _split_point_forecast_summary(baseline, "test")
    residual_table = regime_error_table(
        residuals=pd.Series(y_true - y_hat, index=scored_test.index),
        regime=regime,
    )
    model_interval_metrics = interval_metrics_by_regime(
        y_true,
        model_lower,
        model_upper,
        regime,
        alpha=eval_cfg.interval_alpha,
        regime_order=eval_cfg.regime_labels,
    )
    empirical_interval_metrics = interval_metrics_by_regime(
        y_true,
        empirical_lower,
        empirical_upper,
        regime,
        alpha=eval_cfg.interval_alpha,
        regime_order=eval_cfg.regime_labels,
    )
    model_width = band_width_diagnostics(model_lower, model_upper, regime, vix, regime_order=eval_cfg.regime_labels)
    empirical_width = band_width_diagnostics(
        empirical_lower,
        empirical_upper,
        regime,
        vix,
        regime_order=eval_cfg.regime_labels,
    )

    model_breaches = y_true < model_lower
    empirical_breaches = y_true < empirical_lower
    stressed_mask = regime.eq("stressed").to_numpy()

    kupiec = {
        "unconditional": {
            "model": maybe_kupiec(model_breaches, eval_cfg.lower_tail_prob),
            "empirical": maybe_kupiec(empirical_breaches, eval_cfg.lower_tail_prob),
        },
        "stressed_only": {
            "model": maybe_kupiec(model_breaches[stressed_mask], eval_cfg.lower_tail_prob),
            "empirical": maybe_kupiec(empirical_breaches[stressed_mask], eval_cfg.lower_tail_prob),
        },
    }

    if save_predictions_to is not None:
        save_frame = scored_test.copy()
        save_frame.index.name = "date"
        save_frame.to_csv(save_predictions_to)

    figure_paths: dict[str, str] = {}
    if make_plots:
        figure_paths["training_curve"] = str(plot_training_curves(quantile_artifacts.history))
        figure_paths["band_width_vs_vix"] = str(plot_band_width_vs_vix(model_lower, model_upper, vix))

    return {
        "prediction_alignment": alignment,
        "calibration": {
            "lower_conformal_shift": calibration_info["shift"],
            "n_calibration_rows": calibration_info["n_calibration_rows"],
            "applied": calibration_info["applied"],
        },
        "point_forecast": point_summary,
        "residual_errors_by_regime": residual_table.to_dict(orient="index"),
        "band_width": {
            "model": model_width,
            "empirical": empirical_width,
        },
        "interval_metrics": {
            "model": model_interval_metrics,
            "empirical": empirical_interval_metrics,
        },
        "kupiec": kupiec,
        "quantiles": [float(q) for q in quantile_quantiles],
        "empirical_lookup": empirical_lookup,
        "figure_paths": figure_paths,
    }


def _train_single_split(
    splits: dict[str, pd.DataFrame],
    training_config: TrainingConfig,
    seed: int,
) -> tuple[dict[str, pd.DataFrame], BaselineArtifacts, QuantileArtifacts, dict[str, Any]]:
    labeled_splits, regime_stats = apply_train_only_regimes(splits)
    baseline = train_xgboost_baseline(labeled_splits, cfg=training_config.baseline, seed=seed)

    cond_cols = _resolve_quantile_condition_cols(labeled_splits["train"])
    quantile_artifacts = train_quantile_regressor(
        residuals_train=baseline.residuals["train"].astype(float).values,
        conditions_train=labeled_splits["train"][cond_cols].astype(float).values,
        residuals_val=baseline.residuals["val"].astype(float).values,
        conditions_val=labeled_splits["val"][cond_cols].astype(float).values,
        cond_cols=cond_cols,
        regime_train=labeled_splits["train"]["regime"].astype(str).values,
        regime_val=labeled_splits["val"]["regime"].astype(str).values,
        cfg=training_config.quantile,
        seed=seed,
    )
    return labeled_splits, baseline, quantile_artifacts, regime_stats


def run_training_from_features(
    features: pd.DataFrame,
    training_config: TrainingConfig | None = None,
    seed: int = RANDOM_SEED,
    walk_forward_folds: int = 3,
    save_outputs: bool = True,
    make_plots: bool = True,
) -> dict[str, Any]:
    set_seed(seed)
    training_config = training_config or TrainingConfig()

    single_splits = time_split(features, training_config.split)
    labeled_single, baseline_single, quantile_single, regime_stats = _train_single_split(
        single_splits,
        training_config,
        seed=seed,
    )

    prediction_path = PROCESSED_DIR / "test_predictions.csv" if save_outputs else None
    single_eval = evaluate_split(
        splits=labeled_single,
        baseline=baseline_single,
        quantile_artifacts=quantile_single,
        training_config=training_config,
        save_predictions_to=prediction_path,
        make_plots=make_plots,
    )

    walk_forward_records: list[dict[str, Any]] = []
    for fold_number, fold_splits in enumerate(
        walk_forward_splits(features, training_config.split, n_folds=walk_forward_folds),
        start=1,
    ):
        labeled_fold, baseline_fold, quantile_fold, _ = _train_single_split(
            fold_splits,
            training_config,
            seed=seed + fold_number,
        )
        fold_eval = evaluate_split(
            splits=labeled_fold,
            baseline=baseline_fold,
            quantile_artifacts=quantile_fold,
            training_config=training_config,
            save_predictions_to=None,
            make_plots=False,
        )
        walk_forward_records.append(
            {
                "fold": fold_number,
                "train_rows": int(len(fold_splits["train"])),
                "val_rows": int(len(fold_splits["val"])),
                "test_rows": int(len(fold_splits["test"])),
                "evaluation": fold_eval,
            }
        )

    pooled_kupiec = {
        "unconditional": {
            "model": pool_kupiec_records(
                [record["evaluation"]["kupiec"]["unconditional"]["model"] for record in walk_forward_records]
            ),
            "empirical": pool_kupiec_records(
                [record["evaluation"]["kupiec"]["unconditional"]["empirical"] for record in walk_forward_records]
            ),
        },
        "stressed_only": {
            "model": pool_kupiec_records(
                [record["evaluation"]["kupiec"]["stressed_only"]["model"] for record in walk_forward_records]
            ),
            "empirical": pool_kupiec_records(
                [record["evaluation"]["kupiec"]["stressed_only"]["empirical"] for record in walk_forward_records]
            ),
        },
    }

    walk_forward_summary = {
        "n_folds": int(walk_forward_folds),
        "pooled_kupiec": pooled_kupiec,
        "mean_point_forecast_mae": float(
            np.mean([record["evaluation"]["point_forecast"]["baseline"]["mae"] for record in walk_forward_records])
        ),
        "mean_model_width_by_regime": {
            label: _safe_nanmean(
                [record["evaluation"]["band_width"]["model"]["mean_width_by_regime"][label] for record in walk_forward_records]
            )
            for label in training_config.evaluation.regime_labels
        },
    }

    point_forecast_checks = {
        "single_split_beats_better_naive": bool(single_eval["point_forecast"]["beats_better_naive"]),
        "walk_forward_all_folds_beat_better_naive": bool(
            all(record["evaluation"]["point_forecast"]["beats_better_naive"] for record in walk_forward_records)
        ),
    }

    acceptance = {
        "pooled_unconditional_kupiec_pass": kupiec_passes(
            pooled_kupiec["unconditional"]["model"],
            significance=training_config.evaluation.kupiec_significance,
        ),
        "pooled_stressed_kupiec_pass": kupiec_passes(
            pooled_kupiec["stressed_only"]["model"],
            significance=training_config.evaluation.kupiec_significance,
        ),
    }
    acceptance["acceptance_bar_passed"] = bool(
        acceptance["pooled_unconditional_kupiec_pass"]
        and acceptance["pooled_stressed_kupiec_pass"]
    )

    artifacts = {}
    if save_outputs:
        artifacts = {
            "baseline_model": str(save_baseline_artifacts(baseline_single)),
            "quantile_model": str(save_quantile_artifacts(quantile_single)),
        }

    results: dict[str, Any] = {
        "model_name": MODEL_NAME,
        "model_label": MODEL_LABEL,
        "comparator_name": COMPARATOR_NAME,
        "config": training_config.to_dict(),
        "data_summary": {
            "rows": int(len(features)),
            "columns": list(features.columns),
            "start": str(features.index.min().date()),
            "end": str(features.index.max().date()),
        },
        "single_split": {
            "train_rows": int(len(single_splits["train"])),
            "val_rows": int(len(single_splits["val"])),
            "test_rows": int(len(single_splits["test"])),
            "regime_fit": regime_stats,
            "evaluation": single_eval,
        },
        "walk_forward": {
            "folds": walk_forward_records,
            "summary": walk_forward_summary,
        },
        "point_forecast_checks": point_forecast_checks,
        "acceptance": acceptance,
        "artifacts": artifacts,
    }
    results = _sanitize(results)

    if save_outputs:
        results_path = PROCESSED_DIR / "training_results.json"
        results_path.write_text(json.dumps(results, indent=2))
        _write_results_markdown(results)
        logger.info("Saved training results to %s", results_path)

    return results


def _write_results_markdown(results: dict[str, Any], out_path: Path = REPORTS_DIR / "RESULTS.md") -> Path:
    pooled = results["walk_forward"]["summary"]["pooled_kupiec"]
    acceptance = results["acceptance"]
    single = results["single_split"]["evaluation"]
    lines = [
        "# Results",
        "",
        "## Question",
        "",
        "Can a conditional residual model widen bands in stressed regimes while staying calibrated at the 5% lower tail?",
        "",
        "## Single Split",
        "",
        f"- Point forecast beats better naive: `{single['point_forecast']['beats_better_naive']}`",
        f"- Unconditional lower-tail breach rate: `{single['kupiec']['unconditional']['model']['breach_rate']:.4f}`",
        f"- Unconditional Kupiec p-value: `{single['kupiec']['unconditional']['model']['p_value']:.4f}`",
        f"- Stressed-only lower-tail breach rate: `{single['kupiec']['stressed_only']['model']['breach_rate']:.4f}`",
        f"- Stressed-only Kupiec p-value: `{single['kupiec']['stressed_only']['model']['p_value']:.4f}`",
        "",
        "## Walk Forward",
        "",
        f"- Pooled unconditional breach rate: `{pooled['unconditional']['model']['breach_rate']:.4f}`",
        f"- Pooled unconditional Kupiec p-value: `{pooled['unconditional']['model']['p_value']:.4f}`",
        f"- Pooled stressed breach rate: `{pooled['stressed_only']['model']['breach_rate']:.4f}`",
        f"- Pooled stressed Kupiec p-value: `{pooled['stressed_only']['model']['p_value']:.4f}`",
        "",
        "## Acceptance",
        "",
        f"- Pooled unconditional Kupiec passes: `{acceptance['pooled_unconditional_kupiec_pass']}`",
        f"- Pooled stressed Kupiec passes: `{acceptance['pooled_stressed_kupiec_pass']}`",
        f"- Acceptance bar passed: `{acceptance['acceptance_bar_passed']}`",
    ]
    out_path.write_text("\n".join(lines) + "\n")
    return out_path


def run_training(
    training_config: TrainingConfig | None = None,
    seed: int = RANDOM_SEED,
    walk_forward_folds: int = 3,
    make_plots: bool = True,
) -> dict[str, Any]:
    features = _load_or_build_features()
    return run_training_from_features(
        features=features,
        training_config=training_config,
        seed=seed,
        walk_forward_folds=walk_forward_folds,
        save_outputs=True,
        make_plots=make_plots,
    )
