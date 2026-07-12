from __future__ import annotations

from datetime import date
from io import StringIO

import pandas as pd
import pytest

from app.services.data_loader import generate_sample_market_data, parse_uploaded_csv, resolve_window


def test_resolve_window_presets_are_stable() -> None:
    start, end, interval = resolve_window(None, date(2026, 6, 30), "1Y")
    assert interval == "1Y"
    assert end == date(2026, 6, 30)
    assert 360 <= (end - start).days <= 366



def test_resolve_two_year_window_is_available() -> None:
    start, end, interval = resolve_window(None, date(2026, 6, 30), "2Y")
    assert interval == "2Y"
    assert 725 <= (end - start).days <= 731


def test_sample_data_is_deterministic_for_asset() -> None:
    first = generate_sample_market_data("SPY", date(2024, 1, 1), date(2025, 1, 1))
    second = generate_sample_market_data("SPY", date(2024, 1, 1), date(2025, 1, 1))
    pd.testing.assert_series_equal(first["close"].head(20), second["close"].head(20), check_names=False)
    assert {"date", "close", "volume", "synthetic_state"}.issubset(first.columns)
    assert len(first) >= 180


def test_uploaded_csv_parses_and_deduplicates_dates() -> None:
    dates = pd.bdate_range("2024-01-01", periods=220)
    frame = pd.DataFrame({"date": dates, "close": range(100, 320), "volume": 1_000})
    duplicate = frame.iloc[[0]].copy()
    raw = pd.concat([duplicate, frame], ignore_index=True).to_csv(index=False).encode()

    parsed = parse_uploaded_csv(raw)

    assert len(parsed) == 220
    assert parsed["date"].is_monotonic_increasing
    assert parsed["date"].duplicated().sum() == 0


def test_uploaded_csv_rejects_short_or_missing_required_columns() -> None:
    short = pd.DataFrame({"date": pd.bdate_range("2024-01-01", periods=20), "close": 100}).to_csv(index=False).encode()
    with pytest.raises(ValueError, match="at least 180"):
        parse_uploaded_csv(short)

    missing_close = b"date,price\n2024-01-01,100\n"
    with pytest.raises(ValueError, match="date.*close"):
        parse_uploaded_csv(missing_close)


def test_sample_mode_uses_offline_source() -> None:
    from app.services.data_loader import load_market_data

    loaded = load_market_data("SPY", start=date(2024, 1, 1), end=date(2025, 1, 1), data_mode="sample")

    assert loaded.source == "sample"
    assert not loaded.is_real_data
    assert len(loaded.frame) >= 180


def test_legacy_prefer_live_false_maps_to_sample() -> None:
    from app.services.data_loader import load_market_data

    loaded = load_market_data("SPY", start=date(2024, 1, 1), end=date(2025, 1, 1), data_mode=None, prefer_live_data=False)

    assert loaded.source == "sample"
    assert not loaded.is_real_data
