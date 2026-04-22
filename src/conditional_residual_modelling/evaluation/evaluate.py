"""Metrics, empirical comparators, and plotting helpers."""
from __future__ import annotations

from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402
from scipy.stats import chi2  # noqa: E402

from ..config import FIGURES_DIR


def mae(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    return float(np.mean(np.abs(y_true - y_pred)))


def rmse(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    return float(np.sqrt(np.mean((y_true - y_pred) ** 2)))


def directional_accuracy(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    return float(np.mean(np.sign(y_true) == np.sign(y_pred)))


def interval_coverage(y_true: np.ndarray, lower: np.ndarray, upper: np.ndarray) -> float:
    return float(np.mean((y_true >= lower) & (y_true <= upper)))


def interval_width(lower: np.ndarray, upper: np.ndarray) -> float:
    return float(np.mean(np.asarray(upper) - np.asarray(lower)))


def winkler_interval_score(
    y_true: np.ndarray,
    lower: np.ndarray,
    upper: np.ndarray,
    alpha: float,
) -> float:
    width = np.asarray(upper) - np.asarray(lower)
    below = np.maximum(np.asarray(lower) - np.asarray(y_true), 0.0)
    above = np.maximum(np.asarray(y_true) - np.asarray(upper), 0.0)
    score = width + (2.0 / alpha) * (below + above)
    return float(np.mean(score))


def kupiec_from_counts(
    n_breaches: int,
    n_obs: int,
    expected_prob: float,
) -> dict[str, float]:
    if not 0.0 < expected_prob < 1.0:
        raise ValueError("expected_prob must lie strictly between 0 and 1.")
    if n_obs <= 0:
        raise ValueError("Need at least one observation for the Kupiec test.")

    observed_prob = n_breaches / n_obs

    def _log_like(prob: float) -> float:
        if prob <= 0.0:
            return 0.0 if n_breaches == 0 else float("-inf")
        if prob >= 1.0:
            return 0.0 if n_breaches == n_obs else float("-inf")
        return (n_obs - n_breaches) * np.log(1.0 - prob) + n_breaches * np.log(prob)

    lr_uc = float(max(-2.0 * (_log_like(expected_prob) - _log_like(observed_prob)), 0.0))
    return {
        "n_obs": float(n_obs),
        "n_breaches": float(n_breaches),
        "expected_prob": float(expected_prob),
        "breach_rate": float(observed_prob),
        "lr_uc": lr_uc,
        "p_value": float(chi2.sf(lr_uc, df=1)),
    }


def kupiec_pof_test(breaches: np.ndarray | pd.Series, expected_prob: float) -> dict[str, float]:
    hits = np.asarray(breaches, dtype=bool)
    return kupiec_from_counts(int(hits.sum()), int(hits.size), expected_prob)


def maybe_kupiec(breaches: np.ndarray | pd.Series, expected_prob: float) -> dict[str, float]:
    hits = np.asarray(breaches, dtype=bool)
    if hits.size == 0:
        return {
            "n_obs": 0.0,
            "n_breaches": 0.0,
            "expected_prob": float(expected_prob),
            "breach_rate": float("nan"),
            "lr_uc": float("nan"),
            "p_value": float("nan"),
        }
    return kupiec_pof_test(hits, expected_prob)


def regime_error_table(residuals: pd.Series, regime: pd.Series) -> pd.DataFrame:
    frame = pd.DataFrame({"residual": residuals, "regime": regime}).dropna()
    grouped = frame.groupby("regime")["residual"]
    return pd.DataFrame(
        {
            "count": grouped.size(),
            "mae": grouped.apply(lambda s: float(np.mean(np.abs(s)))),
            "rmse": grouped.apply(lambda s: float(np.sqrt(np.mean(s**2)))),
            "std": grouped.std(ddof=0),
        }
    ).sort_index()


def interval_metrics_by_regime(
    y_true: np.ndarray,
    lower: np.ndarray,
    upper: np.ndarray,
    regime: pd.Series,
    alpha: float,
    regime_order: tuple[str, ...] = ("calm", "normal", "stressed"),
) -> dict[str, dict[str, float]]:
    frame = pd.DataFrame(
        {
            "y_true": y_true,
            "lower": lower,
            "upper": upper,
            "regime": regime.astype(str).values,
        },
        index=regime.index,
    ).dropna()

    out: dict[str, dict[str, float]] = {}
    for label in regime_order:
        subset = frame[frame["regime"] == label]
        if subset.empty:
            out[label] = {
                "count": 0.0,
                "coverage": float("nan"),
                "mean_width": float("nan"),
                "winkler_score": float("nan"),
            }
            continue
        out[label] = {
            "count": float(len(subset)),
            "coverage": interval_coverage(subset["y_true"].values, subset["lower"].values, subset["upper"].values),
            "mean_width": interval_width(subset["lower"].values, subset["upper"].values),
            "winkler_score": winkler_interval_score(
                subset["y_true"].values,
                subset["lower"].values,
                subset["upper"].values,
                alpha=alpha,
            ),
        }
    return out


def band_width_diagnostics(
    lower: np.ndarray,
    upper: np.ndarray,
    regime: pd.Series,
    vix: pd.Series,
    regime_order: tuple[str, ...] = ("calm", "normal", "stressed"),
) -> dict[str, object]:
    width = pd.Series(np.asarray(upper) - np.asarray(lower), index=regime.index)
    width_by_regime = {
        label: float(width[regime.astype(str) == label].mean()) if (regime.astype(str) == label).any() else float("nan")
        for label in regime_order
    }
    width_frame = pd.DataFrame({"width": width, "vix": vix.astype(float)}).dropna()
    if len(width_frame) >= 2 and width_frame["width"].nunique() > 1 and width_frame["vix"].nunique() > 1:
        spearman = float(width_frame["width"].corr(width_frame["vix"], method="spearman"))
    else:
        spearman = float("nan")
    return {
        "mean_width_by_regime": width_by_regime,
        "spearman_corr_with_vix": spearman,
    }


def fit_regime_conditional_empirical_quantiles(
    residuals: np.ndarray | pd.Series,
    regime: pd.Series,
    quantiles: tuple[float, ...],
    min_obs: int = 25,
) -> dict[str, object]:
    frame = pd.DataFrame({"residual": residuals, "regime": regime}).dropna()
    if frame.empty:
        raise ValueError("Need at least one training residual to fit empirical quantiles.")

    global_quantiles = {str(q): float(np.quantile(frame["residual"], q)) for q in quantiles}
    lookup: dict[str, object] = {
        "quantiles": [float(q) for q in quantiles],
        "min_obs": int(min_obs),
        "global": {"count": int(len(frame)), "values": global_quantiles},
        "regimes": {},
    }

    for label, subset in frame.groupby("regime"):
        use_global = len(subset) < min_obs
        values = global_quantiles if use_global else {str(q): float(np.quantile(subset["residual"], q)) for q in quantiles}
        lookup["regimes"][str(label)] = {
            "count": int(len(subset)),
            "fallback_to_global": bool(use_global),
            "values": values,
        }
    return lookup


def predict_regime_conditional_empirical_quantiles(
    y_hat: np.ndarray,
    regime: pd.Series,
    lookup: dict[str, object],
) -> tuple[np.ndarray, tuple[float, ...]]:
    quantiles = tuple(float(q) for q in lookup["quantiles"])
    out = np.zeros((len(y_hat), len(quantiles)), dtype=float)
    global_values = lookup["global"]["values"]
    regime_map = lookup["regimes"]
    for idx, label in enumerate(regime.astype(str)):
        values = regime_map.get(label, {"values": global_values})["values"]
        for jdx, q in enumerate(quantiles):
            out[idx, jdx] = y_hat[idx] + float(values[str(q)])
    return out, quantiles


def pool_kupiec_records(records: list[dict[str, float]]) -> dict[str, float]:
    valid = [record for record in records if record["n_obs"] > 0]
    if not valid:
        return {
            "n_obs": 0.0,
            "n_breaches": 0.0,
            "expected_prob": float("nan"),
            "breach_rate": float("nan"),
            "lr_uc": float("nan"),
            "p_value": float("nan"),
        }
    expected_prob = float(valid[0]["expected_prob"])
    n_obs = int(sum(record["n_obs"] for record in valid))
    n_breaches = int(sum(record["n_breaches"] for record in valid))
    return kupiec_from_counts(n_breaches, n_obs, expected_prob)


def kupiec_passes(result: dict[str, float], significance: float = 0.05) -> bool:
    p_value = result.get("p_value", float("nan"))
    return bool(np.isfinite(p_value) and p_value > significance)


def _save_figure(fig: plt.Figure, path: Path) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    return path


def plot_training_curves(history: dict[str, list[float]], out_path: Path = FIGURES_DIR / "quantile_training.png") -> Path:
    fig, ax = plt.subplots(figsize=(8, 4.5))
    ax.plot(history.get("train_loss", []), label="train")
    ax.plot(history.get("val_loss", []), label="val")
    ax.set_title("Quantile Regressor Pinball Loss")
    ax.set_xlabel("Epoch")
    ax.set_ylabel("Loss")
    ax.grid(alpha=0.3)
    ax.legend()
    return _save_figure(fig, out_path)


def plot_band_width_vs_vix(
    lower: np.ndarray,
    upper: np.ndarray,
    vix: pd.Series,
    out_path: Path = FIGURES_DIR / "band_width_vs_vix.png",
) -> Path:
    width = pd.Series(np.asarray(upper) - np.asarray(lower), index=vix.index)
    frame = pd.DataFrame({"width": width, "vix": vix.astype(float)}).dropna()
    fig, axes = plt.subplots(2, 1, figsize=(10, 7), height_ratios=[2, 1])

    axes[0].plot(frame.index, frame["width"], label="band width", color="tab:blue", linewidth=1.0)
    axes[0].set_ylabel("Band width")
    axes[0].set_title("Band Width Over Time")
    axes[0].grid(alpha=0.3)
    twin = axes[0].twinx()
    twin.plot(frame.index, frame["vix"], label="VIX", color="tab:red", linewidth=1.0, alpha=0.7)
    twin.set_ylabel("VIX")

    handles, labels = axes[0].get_legend_handles_labels()
    handles2, labels2 = twin.get_legend_handles_labels()
    axes[0].legend(handles + handles2, labels + labels2, loc="best")

    axes[1].scatter(frame["vix"], frame["width"], s=12, alpha=0.6, color="tab:purple")
    axes[1].set_xlabel("VIX")
    axes[1].set_ylabel("Band width")
    axes[1].set_title("Band Width vs VIX")
    axes[1].grid(alpha=0.3)

    fig.tight_layout()
    return _save_figure(fig, out_path)
