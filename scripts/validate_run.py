#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import math
from pathlib import Path
import sys

import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from cvae_spx.config import PROCESSED_DIR, RANDOM_SEED, SplitConfig
from cvae_spx.evaluation.evaluate import band_width_diagnostics, maybe_kupiec
from cvae_spx.features import apply_train_only_regimes, time_split
from cvae_spx.pipeline import rebuild_processed_inputs, run_training


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Rebuild processed inputs, retrain the models, reload saved predictions, "
            "recompute Kupiec and band-width diagnostics, and raise on mismatches."
        )
    )
    parser.add_argument("--download", action="store_true", help="Refresh raw data before validation.")
    parser.add_argument("--seed", type=int, default=RANDOM_SEED, help="Random seed for retraining.")
    parser.add_argument("--walk-forward-folds", type=int, default=3, help="Number of walk-forward folds.")
    parser.add_argument("--tolerance", type=float, default=1e-10, help="Absolute tolerance for numeric checks.")
    parser.add_argument("--print-full-json", action="store_true", help="Print the full results payload.")
    return parser.parse_args()


def assert_close(name: str, actual: float, expected: float, tol: float) -> None:
    if pd.isna(actual) and pd.isna(expected):
        return
    if not math.isclose(float(actual), float(expected), rel_tol=0.0, abs_tol=tol):
        raise AssertionError(f"{name} mismatch: actual={actual!r} expected={expected!r} tol={tol}")


def load_predictions(path: Path) -> pd.DataFrame:
    frame = pd.read_csv(path, parse_dates=["date"]).set_index("date").sort_index()
    if "band_lower" not in frame.columns or "band_upper" not in frame.columns:
        raise ValueError(f"Saved predictions at {path} are missing required band columns.")
    return frame


def validate_saved_outputs(results: dict, tolerance: float) -> dict[str, object]:
    results_path = PROCESSED_DIR / "training_results.json"
    predictions_path = PROCESSED_DIR / "test_predictions.csv"
    features_path = PROCESSED_DIR / "features.csv"

    saved_results = json.loads(results_path.read_text())
    if saved_results != results:
        raise AssertionError("Saved training_results.json does not match the in-memory training output.")

    features = pd.read_csv(features_path, parse_dates=["date"]).set_index("date").sort_index()
    predictions = load_predictions(predictions_path)

    labeled_splits, _ = apply_train_only_regimes(time_split(features, SplitConfig()))
    test = labeled_splits["test"]
    aligned = test.loc[predictions.index].copy()
    if len(aligned) != len(predictions):
        raise AssertionError("Saved predictions do not align one-for-one with the labeled test split.")

    y_true = predictions["target_next_logret"].astype(float).values
    y_hat = predictions["y_hat"].astype(float).values
    band_lower = predictions["band_lower"].astype(float).values
    band_upper = predictions["band_upper"].astype(float).values
    regimes = aligned["regime"].astype(str)
    vix = aligned["vix"].astype(float)

    model_hits = y_true < band_lower
    stressed_mask = regimes.eq("stressed").to_numpy()
    unconditional_kupiec = maybe_kupiec(model_hits, expected_prob=0.05)
    stressed_kupiec = maybe_kupiec(model_hits[stressed_mask], expected_prob=0.05)
    width_diag = band_width_diagnostics(band_lower, band_upper, regimes, vix)

    saved_single = results["single_split"]["evaluation"]
    saved_kupiec_unconditional = saved_single["kupiec"]["unconditional"]["model"]
    saved_kupiec_stressed = saved_single["kupiec"]["stressed_only"]["model"]
    saved_width = saved_single["band_width"]["model"]

    for key in ("n_obs", "n_breaches", "expected_prob", "breach_rate", "lr_uc", "p_value"):
        assert_close(
            f"kupiec.unconditional.{key}",
            unconditional_kupiec[key],
            saved_kupiec_unconditional[key],
            tolerance,
        )
        assert_close(
            f"kupiec.stressed_only.{key}",
            stressed_kupiec[key],
            saved_kupiec_stressed[key],
            tolerance,
        )

    assert_close(
        "point_forecast.mae",
        float(pd.Series(y_true - y_hat).abs().mean()),
        float(saved_single["point_forecast"]["baseline"]["mae"]),
        tolerance,
    )
    assert_close(
        "band_width.spearman_corr_with_vix",
        width_diag["spearman_corr_with_vix"],
        saved_width["spearman_corr_with_vix"],
        tolerance,
    )
    for regime, actual in width_diag["mean_width_by_regime"].items():
        assert_close(
            f"band_width.mean_width_by_regime.{regime}",
            actual,
            saved_width["mean_width_by_regime"][regime],
            tolerance,
        )

    return {
        "checks_passed": True,
        "validated_metrics": {
            "kupiec_unconditional": unconditional_kupiec,
            "kupiec_stressed_only": stressed_kupiec,
            "band_width": width_diag,
            "prediction_rows": int(len(predictions)),
        },
    }


def main() -> None:
    args = parse_args()
    master, features = rebuild_processed_inputs(download=args.download)
    results = run_training(
        seed=args.seed,
        walk_forward_folds=args.walk_forward_folds,
        make_plots=True,
    )
    validation = validate_saved_outputs(results, tolerance=args.tolerance)

    summary = {
        "master_rows": len(master),
        "feature_rows": len(features),
        "acceptance": results["acceptance"],
        "validation": validation,
    }
    print(json.dumps(summary, indent=2))
    if args.print_full_json:
        print(json.dumps(results, indent=2))


if __name__ == "__main__":
    main()
