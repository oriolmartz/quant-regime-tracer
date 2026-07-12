from __future__ import annotations

import numpy as np
import pandas as pd

from app.services.annualization import infer_annualization_profile
from app.services.data_loader import generate_sample_market_data
from app.services.features import engineer_features
from app.services.risk_metrics import expected_persistence_days, max_drawdown, transition_matrix


def test_engineer_features_returns_clean_model_matrix() -> None:
    raw = generate_sample_market_data("QQQ", None, None)
    features, columns, warnings = engineer_features(raw)

    assert len(features) >= 180
    assert "rolling_volatility" in columns
    assert "drawdown" in columns
    assert features[columns].isna().sum().sum() == 0
    assert features["date"].is_monotonic_increasing
    assert isinstance(warnings, list)


def test_transition_matrix_normalizes_observed_rows() -> None:
    states = np.array([0, 0, 1, 1, 2, 2, 2, 0])
    matrix = transition_matrix(states, 3)

    assert matrix.shape == (3, 3)
    assert np.allclose(matrix.sum(axis=1), 1.0)
    assert matrix[0, 0] > 0
    assert matrix[2, 2] > 0


def test_expected_persistence_and_max_drawdown() -> None:
    assert expected_persistence_days(0.8) == 5.000000000000001
    assert expected_persistence_days(1.0) is None
    assert max_drawdown(pd.Series([100, 120, 90, 95])) == -0.25


def test_annualization_profile_distinguishes_exchange_and_seven_day_calendars() -> None:
    exchange_dates = pd.bdate_range("2024-01-01", "2025-12-31")
    continuous_dates = pd.date_range("2024-01-01", "2025-12-31", freq="D")

    exchange = infer_annualization_profile(exchange_dates)
    continuous = infer_annualization_profile(continuous_dates)

    assert exchange["periods_per_year"] == 252.0
    assert exchange["calendar_type"] == "exchange_calendar"
    assert continuous["periods_per_year"] == 365.0
    assert continuous["calendar_type"] == "seven_day_calendar"


def test_feature_annualization_uses_observed_calendar() -> None:
    dates = pd.date_range("2024-01-01", periods=400, freq="D")
    close = 100 * np.exp(np.linspace(0, 0.25, len(dates)))
    frame = pd.DataFrame({"date": dates, "close": close, "volume": 1_000})

    features, _, _ = engineer_features(frame)

    assert features.attrs["annualization_factor"] == 365.0
    assert features.attrs["annualization_profile"]["calendar_type"] == "seven_day_calendar"
