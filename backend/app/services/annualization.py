from __future__ import annotations

from typing import Any, Iterable

import pandas as pd

CALENDAR_DAYS_PER_YEAR = 365.2425
DEFAULT_MARKET_PERIODS_PER_YEAR = 252.0


def infer_annualization_profile(dates: Iterable[Any]) -> dict[str, Any]:
    """Infer an annualization factor from the timestamps actually supplied.

    The engine works with daily observations, but not every asset follows the same
    calendar. Exchange-traded assets commonly contribute about 252 observations per
    year, seven-day markets about 365, and uploaded datasets may follow another cadence.
    The profile is therefore inferred from the observed date spacing rather than from
    a ticker whitelist.
    """
    parsed = pd.Series(pd.to_datetime(list(dates), errors="coerce")).dropna()
    if parsed.empty:
        return {
            "periods_per_year": DEFAULT_MARKET_PERIODS_PER_YEAR,
            "calendar_type": "default_exchange_calendar",
            "method": "fallback_default",
            "observed_periods_per_year": None,
            "weekend_observation_share": 0.0,
        }

    parsed = parsed.dt.normalize().drop_duplicates().sort_values().reset_index(drop=True)
    if len(parsed) < 2:
        return {
            "periods_per_year": DEFAULT_MARKET_PERIODS_PER_YEAR,
            "calendar_type": "default_exchange_calendar",
            "method": "fallback_default",
            "observed_periods_per_year": None,
            "weekend_observation_share": float((parsed.dt.dayofweek >= 5).mean()) if len(parsed) else 0.0,
        }

    span_days = max(int((parsed.iloc[-1] - parsed.iloc[0]).days), 1)
    observed = float((len(parsed) - 1) * CALENDAR_DAYS_PER_YEAR / span_days)
    weekend_share = float((parsed.dt.dayofweek >= 5).mean())

    # Snap the two most common daily calendars to stable, interpretable factors.
    # Other cadences remain empirical so six-day, FX-like, weekly or custom CSV
    # datasets are not silently forced into an equity or crypto convention.
    if weekend_share >= 0.05 and observed >= 330.0:
        factor = 365.0
        calendar_type = "seven_day_calendar"
        method = "inferred_seven_day_calendar"
    elif weekend_share < 0.02 and 240.0 <= observed <= 270.0:
        factor = DEFAULT_MARKET_PERIODS_PER_YEAR
        calendar_type = "exchange_calendar"
        method = "inferred_exchange_calendar"
    else:
        factor = max(1.0, min(366.0, round(observed, 1)))
        calendar_type = "empirical_calendar"
        method = "observed_timestamp_frequency"

    return {
        "periods_per_year": float(factor),
        "calendar_type": calendar_type,
        "method": method,
        "observed_periods_per_year": float(observed),
        "weekend_observation_share": weekend_share,
    }


def annualization_factor_from_frame(frame: pd.DataFrame) -> float:
    profile = frame.attrs.get("annualization_profile") if hasattr(frame, "attrs") else None
    if isinstance(profile, dict) and profile.get("periods_per_year") is not None:
        return float(profile["periods_per_year"])
    if hasattr(frame, "attrs") and frame.attrs.get("annualization_factor") is not None:
        return float(frame.attrs["annualization_factor"])
    if "date" in frame.columns:
        return float(infer_annualization_profile(frame["date"])["periods_per_year"])
    return DEFAULT_MARKET_PERIODS_PER_YEAR
