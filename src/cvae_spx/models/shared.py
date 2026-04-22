"""Shared utilities for residual-window models."""
from __future__ import annotations

from collections.abc import Iterable

import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import Dataset


class ResidualWindowDataset(Dataset):
    """Use `window_size` lagged residuals to predict the next residual."""

    def __init__(
        self,
        residuals: np.ndarray,
        conditions: np.ndarray,
        window_size: int,
        sample_weights: np.ndarray | None = None,
    ) -> None:
        if residuals.ndim != 1:
            raise ValueError("residuals must be a 1-D array.")
        if len(residuals) != len(conditions):
            raise ValueError("residuals and conditions must have the same number of rows.")
        if len(residuals) <= window_size:
            raise ValueError(
                f"Need more than {window_size} residual rows, got {len(residuals)}."
            )
        self.residuals = residuals.astype(np.float32)
        self.conditions = conditions.astype(np.float32)
        if sample_weights is None:
            self.sample_weights = np.ones(len(residuals), dtype=np.float32)
        else:
            if len(sample_weights) != len(residuals):
                raise ValueError("sample_weights and residuals must have the same length.")
            self.sample_weights = sample_weights.astype(np.float32)
        self.window_size = int(window_size)

    def __len__(self) -> int:
        return len(self.residuals) - self.window_size

    def __getitem__(self, idx: int) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        stop = idx + self.window_size
        context = self.residuals[idx:stop]
        target = self.residuals[stop]
        cond = self.conditions[stop]
        weight = self.sample_weights[stop]
        return (
            torch.from_numpy(context),
            torch.from_numpy(cond),
            torch.tensor(target, dtype=torch.float32),
            torch.tensor(weight, dtype=torch.float32),
        )


def build_mlp(
    sizes: Iterable[int],
    dropout: float = 0.0,
    pre_layernorm: bool = False,
) -> nn.Sequential:
    dims = list(sizes)
    layers: list[nn.Module] = []
    for in_dim, out_dim in zip(dims[:-1], dims[1:]):
        if pre_layernorm:
            layers.append(nn.LayerNorm(in_dim))
        layers.append(nn.Linear(in_dim, out_dim))
        layers.append(nn.ReLU())
        if dropout > 0:
            layers.append(nn.Dropout(dropout))
    return nn.Sequential(*layers)


def compute_stats(values: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    mean = values.mean(axis=0)
    std = values.std(axis=0)
    return mean, np.where(std < 1e-8, 1.0, std)


def make_regime_sample_weights(
    regime_labels: np.ndarray | list[str] | None,
    stressed_loss_weight: float,
) -> np.ndarray | None:
    if regime_labels is None:
        return None
    labels = np.asarray(regime_labels, dtype=object)
    weights = np.ones(len(labels), dtype=np.float32)
    weights[labels == "stressed"] = float(stressed_loss_weight)
    return weights / max(float(weights.mean()), 1e-8)
