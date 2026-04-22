"""Central project configuration and reproducibility helpers."""
from __future__ import annotations

import logging
import os
import random
from dataclasses import asdict, dataclass, field
from pathlib import Path

import numpy as np
from dotenv import load_dotenv

load_dotenv()

PROJECT_ROOT = Path(__file__).resolve().parents[2]
DATA_DIR = PROJECT_ROOT / "data"
RAW_DIR = DATA_DIR / "raw"
INTERIM_DIR = DATA_DIR / "interim"
PROCESSED_DIR = DATA_DIR / "processed"
MODELS_DIR = PROJECT_ROOT / "models"
REPORTS_DIR = PROJECT_ROOT / "reports"
FIGURES_DIR = REPORTS_DIR / "figures"

for _path in (RAW_DIR, INTERIM_DIR, PROCESSED_DIR, MODELS_DIR, FIGURES_DIR):
    _path.mkdir(parents=True, exist_ok=True)

RANDOM_SEED = int(os.getenv("RANDOM_SEED", "42"))
DATA_START_DATE = os.getenv("DATA_START_DATE", "1990-01-01")
DATA_END_DATE = os.getenv("DATA_END_DATE", "") or None
FRED_API_KEY = os.getenv("FRED_API_KEY", "")

TARGET_COL = "spx"
TARGET_LOGRET_COL = "target_next_logret"
PRIMARY_INTERVAL_ALPHA = 0.10
LOWER_TAIL_PROB = PRIMARY_INTERVAL_ALPHA / 2.0

YAHOO_CLOSE_SERIES: dict[str, str] = {
    "spx": "^GSPC",
    "vix": "^VIX",
    "skew": "^SKEW",
    "vvix": "^VVIX",
}
YAHOO_OHLCV_SERIES: dict[str, str] = {
    "spy": "SPY",
}
FRED_SERIES: dict[str, str] = {
    "dgs2": "DGS2",
    "dgs10": "DGS10",
    "fed_funds": "FEDFUNDS",
}

REQUIRED_ALT_FEATURE_COLS: tuple[str, ...] = (
    "skew",
    "skew_pctchg5",
    "vvix",
    "vvix_pctchg5",
    "spy_intraday_range",
    "spy_intraday_range_mean20",
    "spy_intraday_range_z20",
    "spy_log_volume_z20",
)

QUANTILE_CONDITION_COLS: tuple[str, ...] = (
    "vix",
    "dgs2",
    "dgs10",
    "fed_funds",
    "spx_vol5",
    "spx_vol20",
    "spx_vol60",
    "spx_drawdown",
    "skew",
    "skew_pctchg5",
    "vvix",
    "vvix_pctchg5",
    "spy_intraday_range",
    "spy_intraday_range_mean20",
    "spy_intraday_range_z20",
    "spy_log_volume_z20",
)


@dataclass(frozen=True)
class SplitConfig:
    train_frac: float = 0.70
    val_frac: float = 0.15
    test_frac: float = 0.15


@dataclass(frozen=True)
class BaselineConfig:
    xgb_n_estimators: int = 600
    xgb_max_depth: int = 3
    xgb_learning_rate: float = 0.03
    xgb_subsample: float = 0.8
    xgb_colsample_bytree: float = 0.8
    xgb_reg_lambda: float = 1.0
    xgb_min_child_weight: float = 5.0
    xgb_early_stopping_rounds: int = 50


@dataclass(frozen=True)
class QuantileConfig:
    window_size: int = 10
    quantiles: tuple[float, ...] = (0.025, 0.05, 0.25, 0.5, 0.75, 0.95, 0.975)
    hidden_sizes: tuple[int, ...] = (64, 32)
    dropout: float = 0.0
    pre_layernorm: bool = False
    learning_rate: float = 1e-3
    batch_size: int = 128
    max_epochs: int = 250
    patience: int = 30
    weight_decay: float = 1e-5
    gradient_clip_norm: float = 1.0
    stressed_loss_weight: float = 2.0


@dataclass(frozen=True)
class EvaluationConfig:
    interval_alpha: float = PRIMARY_INTERVAL_ALPHA
    lower_tail_prob: float = LOWER_TAIL_PROB
    kupiec_significance: float = 0.05
    empirical_min_obs: int = 25
    regime_labels: tuple[str, ...] = ("calm", "normal", "stressed")
    apply_lower_conformal_shift: bool = True


@dataclass(frozen=True)
class TrainingConfig:
    split: SplitConfig = field(default_factory=SplitConfig)
    baseline: BaselineConfig = field(default_factory=BaselineConfig)
    quantile: QuantileConfig = field(default_factory=QuantileConfig)
    evaluation: EvaluationConfig = field(default_factory=EvaluationConfig)

    def to_dict(self) -> dict[str, object]:
        return asdict(self)


def get_logger(name: str) -> logging.Logger:
    logger = logging.getLogger(name)
    if not logger.handlers:
        handler = logging.StreamHandler()
        handler.setFormatter(
            logging.Formatter("[%(asctime)s] %(levelname)s %(name)s: %(message)s", "%H:%M:%S")
        )
        logger.addHandler(handler)
        logger.setLevel(logging.INFO)
    return logger


def set_seed(seed: int = RANDOM_SEED) -> None:
    os.environ["PYTHONHASHSEED"] = str(seed)
    random.seed(seed)
    np.random.seed(seed)
    try:
        import torch

        torch.manual_seed(seed)
        if torch.cuda.is_available():
            torch.cuda.manual_seed_all(seed)
    except ImportError:  # pragma: no cover
        pass
