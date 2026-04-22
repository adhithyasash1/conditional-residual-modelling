from __future__ import annotations

import pandas as pd

from conditional_residual_modelling.data import align_to_spx_trading_dates


def test_align_to_spx_trading_dates_anchors_on_observed_spx_rows() -> None:
    index = pd.to_datetime(
        [
            "2024-01-02",
            "2024-01-03",
            "2024-01-04",
            "2024-01-05",
            "2024-01-08",
        ]
    )
    panel = pd.DataFrame(
        {
            "spx": [4700.0, 4710.0, 4725.0, None, None],
            "vix": [12.0, 12.5, 13.0, 13.5, 14.0],
            "fed_funds": [5.25, None, None, None, 5.25],
        },
        index=index,
    )

    aligned = align_to_spx_trading_dates(panel)

    expected_index = pd.to_datetime(["2024-01-02", "2024-01-03", "2024-01-04"])
    assert aligned.index.equals(expected_index)
    assert aligned["spx"].tolist() == [4700.0, 4710.0, 4725.0]
    assert aligned["vix"].tolist() == [12.0, 12.5, 13.0]
    assert aligned["fed_funds"].tolist() == [5.25, 5.25, 5.25]


def test_align_to_spx_trading_dates_preserves_true_target_gaps() -> None:
    index = pd.to_datetime(["2024-01-02", "2024-01-03", "2024-01-04"])
    panel = pd.DataFrame(
        {
            "spx": [4700.0, None, 4725.0],
            "vix": [12.0, 12.5, 13.0],
        },
        index=index,
    )

    aligned = align_to_spx_trading_dates(panel)

    expected_index = pd.to_datetime(["2024-01-02", "2024-01-04"])
    assert aligned.index.equals(expected_index)
    assert aligned["spx"].tolist() == [4700.0, 4725.0]
