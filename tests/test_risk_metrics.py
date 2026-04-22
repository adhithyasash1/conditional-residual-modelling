from __future__ import annotations

import numpy as np
import pandas as pd

from cvae_spx.evaluation.evaluate import (
    fit_regime_conditional_empirical_quantiles,
    kupiec_pof_test,
    predict_regime_conditional_empirical_quantiles,
)


def test_kupiec_matches_expected_hit_rate() -> None:
    breaches = np.array([True, False] * 10)
    result = kupiec_pof_test(breaches, expected_prob=0.5)

    assert result["n_obs"] == 20.0
    assert result["n_breaches"] == 10.0
    assert result["breach_rate"] == 0.5
    assert result["p_value"] > 0.9


def test_empirical_quantiles_fallback_to_global_when_regime_too_small() -> None:
    residuals = np.array([-2.0, -1.0, -0.5, 0.1, 0.2, 0.3], dtype=float)
    regime = pd.Series(["calm", "calm", "normal", "normal", "normal", "stressed"])
    lookup = fit_regime_conditional_empirical_quantiles(
        residuals=residuals,
        regime=regime,
        quantiles=(0.05, 0.5, 0.95),
        min_obs=2,
    )

    assert lookup["regimes"]["stressed"]["fallback_to_global"] is True

    preds, quantiles = predict_regime_conditional_empirical_quantiles(
        y_hat=np.array([0.0, 0.0]),
        regime=pd.Series(["stressed", "calm"]),
        lookup=lookup,
    )

    assert quantiles == (0.05, 0.5, 0.95)
    assert preds.shape == (2, 3)
    assert np.isclose(preds[0, 0], float(lookup["global"]["values"]["0.05"]))
