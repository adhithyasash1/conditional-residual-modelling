"""Direct conditional quantile regression for residual uncertainty."""
from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader

from ..config import MODELS_DIR, QuantileConfig, get_logger, set_seed
from .shared import ResidualWindowDataset, build_mlp, compute_stats, make_regime_sample_weights

logger = get_logger(__name__)


class ConditionalQuantileRegressor(nn.Module):
    """Predict ordered residual quantiles with a median anchor and softplus gaps."""

    def __init__(
        self,
        context_size: int,
        cond_dim: int,
        quantiles: tuple[float, ...],
        hidden_sizes: tuple[int, ...] = (64, 32),
        dropout: float = 0.1,
        pre_layernorm: bool = False,
    ) -> None:
        super().__init__()
        ordered = tuple(sorted(quantiles))
        if not ordered:
            raise ValueError("quantiles must be non-empty.")
        if any(q <= 0.0 or q >= 1.0 for q in ordered):
            raise ValueError("quantiles must lie strictly between 0 and 1.")
        if 0.5 not in ordered:
            raise ValueError("quantiles must include 0.5 for the median anchor.")

        self.quantiles = ordered
        self.median_index = ordered.index(0.5)
        input_dim = context_size + cond_dim
        if hidden_sizes:
            self.backbone = build_mlp(
                [input_dim, *hidden_sizes],
                dropout=dropout,
                pre_layernorm=pre_layernorm,
            )
            last_dim = hidden_sizes[-1]
        else:
            self.backbone = nn.Identity()
            last_dim = input_dim
        self.anchor_head = nn.Linear(last_dim, 1)
        self.gap_head = nn.Linear(last_dim, len(ordered) - 1)
        self.register_buffer("quantile_levels", torch.tensor(ordered, dtype=torch.float32))

    def forward(self, context: torch.Tensor, cond: torch.Tensor) -> torch.Tensor:
        features = torch.cat([context, cond], dim=-1)
        hidden = self.backbone(features)
        anchor = self.anchor_head(hidden)
        gaps = torch.nn.functional.softplus(self.gap_head(hidden)) + 1e-4

        left_gaps = gaps[:, : self.median_index]
        right_gaps = gaps[:, self.median_index :]

        if left_gaps.shape[1]:
            lower = anchor - torch.flip(torch.cumsum(torch.flip(left_gaps, dims=[1]), dim=1), dims=[1])
        else:
            lower = torch.empty((context.shape[0], 0), device=context.device, dtype=context.dtype)

        if right_gaps.shape[1]:
            upper = anchor + torch.cumsum(right_gaps, dim=1)
        else:
            upper = torch.empty((context.shape[0], 0), device=context.device, dtype=context.dtype)

        return torch.cat([lower, anchor, upper], dim=1)


def pinball_loss(
    y_true: torch.Tensor,
    preds: torch.Tensor,
    quantiles: torch.Tensor,
    sample_weight: torch.Tensor | None = None,
) -> torch.Tensor:
    diff = y_true.unsqueeze(-1) - preds
    loss = torch.maximum(quantiles * diff, (quantiles - 1.0) * diff).mean(dim=-1)
    if sample_weight is not None:
        weights = sample_weight.reshape(-1)
        weights = weights / weights.mean().clamp_min(1e-8)
        return (loss * weights).mean()
    return loss.mean()


@dataclass
class QuantileArtifacts:
    model: ConditionalQuantileRegressor
    config: QuantileConfig
    cond_cols: list[str]
    residual_stats: dict[str, float]
    cond_stats: dict[str, np.ndarray]
    history: dict[str, list[float]]
    quantiles: tuple[float, ...]


def train_quantile_regressor(
    residuals_train: np.ndarray,
    conditions_train: np.ndarray,
    residuals_val: np.ndarray,
    conditions_val: np.ndarray,
    cond_cols: list[str],
    regime_train: np.ndarray | list[str] | None = None,
    regime_val: np.ndarray | list[str] | None = None,
    cfg: QuantileConfig = QuantileConfig(),
    seed: int = 42,
    device: str | None = None,
) -> QuantileArtifacts:
    set_seed(seed)
    device = device or ("cuda" if torch.cuda.is_available() else "cpu")

    residual_mean = float(residuals_train.mean())
    residual_std = float(residuals_train.std() + 1e-8)
    cond_mean, cond_std = compute_stats(conditions_train)

    def _standardize(residuals: np.ndarray, conditions: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        scaled_residuals = ((residuals - residual_mean) / residual_std).astype(np.float32)
        scaled_conditions = ((conditions - cond_mean) / cond_std).astype(np.float32)
        return scaled_residuals, scaled_conditions

    train_res, train_cond = _standardize(residuals_train, conditions_train)
    val_res, val_cond = _standardize(residuals_val, conditions_val)

    train_weights = make_regime_sample_weights(regime_train, cfg.stressed_loss_weight)
    val_weights = make_regime_sample_weights(regime_val, cfg.stressed_loss_weight)

    train_ds = ResidualWindowDataset(train_res, train_cond, cfg.window_size, sample_weights=train_weights)
    val_ds = ResidualWindowDataset(val_res, val_cond, cfg.window_size, sample_weights=val_weights)

    train_loader = DataLoader(train_ds, batch_size=cfg.batch_size, shuffle=True)
    val_loader = DataLoader(val_ds, batch_size=cfg.batch_size, shuffle=False)

    model = ConditionalQuantileRegressor(
        context_size=cfg.window_size,
        cond_dim=conditions_train.shape[1],
        quantiles=cfg.quantiles,
        hidden_sizes=cfg.hidden_sizes,
        dropout=cfg.dropout,
        pre_layernorm=cfg.pre_layernorm,
    ).to(device)
    quantile_levels = model.quantile_levels.to(device)

    optimizer = torch.optim.Adam(
        model.parameters(),
        lr=cfg.learning_rate,
        weight_decay=cfg.weight_decay,
    )

    history: dict[str, list[float]] = {"train_loss": [], "val_loss": []}
    best_state: dict[str, torch.Tensor] | None = None
    best_val = float("inf")
    patience_counter = 0

    for epoch in range(1, cfg.max_epochs + 1):
        model.train()
        train_loss = 0.0
        train_rows = 0
        for context, cond, target, weight in train_loader:
            context = context.to(device)
            cond = cond.to(device)
            target = target.to(device)
            weight = weight.to(device)
            preds = model(context, cond)
            loss = pinball_loss(target, preds, quantile_levels, sample_weight=weight)
            optimizer.zero_grad()
            loss.backward()
            nn.utils.clip_grad_norm_(model.parameters(), cfg.gradient_clip_norm)
            optimizer.step()
            train_loss += loss.item() * context.shape[0]
            train_rows += context.shape[0]

        model.eval()
        val_loss = 0.0
        val_rows = 0
        with torch.no_grad():
            for context, cond, target, weight in val_loader:
                context = context.to(device)
                cond = cond.to(device)
                target = target.to(device)
                weight = weight.to(device)
                preds = model(context, cond)
                loss = pinball_loss(target, preds, quantile_levels, sample_weight=weight)
                val_loss += loss.item() * context.shape[0]
                val_rows += context.shape[0]

        history["train_loss"].append(train_loss / max(train_rows, 1))
        history["val_loss"].append(val_loss / max(val_rows, 1))

        if epoch == 1 or epoch % 10 == 0:
            logger.info(
                "[quantile] epoch=%d train=%.5f val=%.5f",
                epoch,
                history["train_loss"][-1],
                history["val_loss"][-1],
            )

        current_val = history["val_loss"][-1]
        if current_val < best_val - 1e-6:
            best_val = current_val
            best_state = {key: value.detach().cpu().clone() for key, value in model.state_dict().items()}
            patience_counter = 0
        else:
            patience_counter += 1
            if patience_counter >= cfg.patience:
                logger.info("[quantile] early stopping at epoch %d", epoch)
                break

    if best_state is not None:
        model.load_state_dict(best_state)

    return QuantileArtifacts(
        model=model,
        config=cfg,
        cond_cols=cond_cols,
        residual_stats={"mean": residual_mean, "std": residual_std},
        cond_stats={"mean": cond_mean, "std": cond_std},
        history=history,
        quantiles=cfg.quantiles,
    )


def save_quantile_artifacts(artifacts: QuantileArtifacts, out_dir: Path = MODELS_DIR) -> Path:
    out_path = out_dir / "quantile_regressor.pt"
    torch.save(
        {
            "state_dict": artifacts.model.state_dict(),
            "config": artifacts.config.__dict__,
            "cond_cols": artifacts.cond_cols,
            "residual_stats": artifacts.residual_stats,
            "cond_stats": {key: value.tolist() for key, value in artifacts.cond_stats.items()},
            "history": artifacts.history,
            "quantiles": list(artifacts.quantiles),
        },
        out_path,
    )
    logger.info("Saved quantile regressor artifact to %s", out_path)
    return out_path


@torch.no_grad()
def predict_quantiles(
    artifacts: QuantileArtifacts,
    residuals: np.ndarray,
    conditions: np.ndarray,
    device: str | None = None,
) -> tuple[np.ndarray, np.ndarray, tuple[float, ...]]:
    device = device or ("cuda" if torch.cuda.is_available() else "cpu")
    model = artifacts.model.to(device).eval()

    residuals_scaled = (
        (residuals - artifacts.residual_stats["mean"]) / (artifacts.residual_stats["std"] + 1e-8)
    ).astype(np.float32)
    conditions_scaled = (
        (conditions - artifacts.cond_stats["mean"]) / (artifacts.cond_stats["std"] + 1e-8)
    ).astype(np.float32)

    dataset = ResidualWindowDataset(residuals_scaled, conditions_scaled, artifacts.config.window_size)
    loader = DataLoader(dataset, batch_size=512, shuffle=False)

    batches: list[np.ndarray] = []
    for context, cond, _, _ in loader:
        preds = model(context.to(device), cond.to(device))
        batches.append(preds.cpu().numpy())

    preds_scaled = np.concatenate(batches, axis=0)
    preds = preds_scaled * artifacts.residual_stats["std"] + artifacts.residual_stats["mean"]
    offsets = np.arange(artifacts.config.window_size, artifacts.config.window_size + len(preds))
    return preds, offsets, artifacts.quantiles
