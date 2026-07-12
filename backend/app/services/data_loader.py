from __future__ import annotations

from dataclasses import dataclass
from datetime import date, timedelta
from hashlib import sha256
from pathlib import Path
from typing import Literal, Optional

import numpy as np
import pandas as pd

SUPPORTED_ASSETS = [
    "SPY",
    "QQQ",
    "BTC-USD",
    "ETH-USD",
    "AAPL",
    "MSFT",
    "NVDA",
    "META",
    "IWM",
    "DIA",
    "GLD",
    "TLT",
]
WINDOW_PRESETS = ["6M", "1Y", "2Y", "3Y", "5Y", "MAX"]
DATA_MODES = ["real", "auto", "sample"]
DEFAULT_START = date(2021, 1, 1)
CACHE_DIR = Path(__file__).resolve().parents[3] / ".cache" / "market_data"
CACHE_DIR.mkdir(parents=True, exist_ok=True)

DataMode = Literal["real", "auto", "sample"]


class MarketDataError(RuntimeError):
    """Raised when real-market-data mode cannot produce a real dataset."""


@dataclass
class LoadedData:
    frame: pd.DataFrame
    source: str
    warning: Optional[str] = None
    cache_hit: bool = False
    provider: Optional[str] = None
    requested_start: Optional[date] = None
    requested_end: Optional[date] = None

    @property
    def is_real_data(self) -> bool:
        return self.source in {"yfinance", "cache:yfinance"}


def resolve_window(start: Optional[date], end: Optional[date], interval: Optional[str]) -> tuple[date, date, str]:
    """Resolve explicit dates or a preset window into a real-data analysis window.

    The normal operating mode is market-data first, so the implicit end date is the
    current date. Deterministic sample data remains available through `data_mode='sample'`
    for offline tests and reproducible examples.
    """
    normalized = (interval or "5Y").upper()
    end_date = end or date.today()
    if start:
        return start, end_date, normalized
    if normalized == "6M":
        start_date = end_date - timedelta(days=183)
    elif normalized == "1Y":
        start_date = end_date - timedelta(days=365)
    elif normalized == "2Y":
        start_date = end_date - timedelta(days=365 * 2)
    elif normalized == "3Y":
        start_date = end_date - timedelta(days=365 * 3)
    elif normalized == "MAX":
        start_date = date(2016, 1, 1)
    else:
        normalized = "5Y"
        start_date = end_date - timedelta(days=365 * 5)
    return start_date, end_date, normalized


def _date_range(start: Optional[date], end: Optional[date], continuous: bool = False) -> pd.DatetimeIndex:
    start_ts = pd.Timestamp(start or DEFAULT_START)
    end_ts = pd.Timestamp(end or date.today())
    if end_ts <= start_ts:
        end_ts = start_ts + pd.Timedelta(days=365)
    return pd.date_range(start_ts, end_ts, freq="D") if continuous else pd.bdate_range(start_ts, end_ts)


def _asset_seed(asset: str) -> int:
    # Python's built-in hash is salted per process; sha256 keeps sample data reproducible.
    return int(sha256(asset.upper().encode("utf-8")).hexdigest()[:8], 16)


def _safe_asset_name(asset: str) -> str:
    return asset.upper().replace("^", "IDX_").replace("-", "_").replace("/", "_")


def _cache_path(asset: str, start: date, end: date) -> Path:
    return CACHE_DIR / f"{_safe_asset_name(asset)}_{start.isoformat()}_{end.isoformat()}.csv"


def _cache_glob(asset: str) -> list[Path]:
    return sorted(CACHE_DIR.glob(f"{_safe_asset_name(asset)}_*.csv"), key=lambda p: p.stat().st_mtime, reverse=True)


def _read_cache_file(path: Path) -> pd.DataFrame:
    cached = pd.read_csv(path)
    cached["date"] = pd.to_datetime(cached["date"], errors="coerce")
    cached["close"] = pd.to_numeric(cached["close"], errors="coerce")
    if "volume" in cached.columns:
        cached["volume"] = pd.to_numeric(cached["volume"], errors="coerce")
    else:
        cached["volume"] = np.nan
    return cached.dropna(subset=["date", "close"]).sort_values("date").drop_duplicates("date")


def _cache_edge_tolerance_days(frame: pd.DataFrame) -> int:
    """Allow weekends/holidays without accepting a materially shorter cached window."""
    gaps = frame["date"].diff().dropna().dt.total_seconds().div(86_400)
    median_gap = float(gaps.median()) if not gaps.empty else 1.0
    return max(7, min(31, int(np.ceil(median_gap * 3))))


def _cache_covers_request(frame: pd.DataFrame, start: date, end: date) -> bool:
    if frame.empty:
        return False
    tolerance = pd.Timedelta(days=_cache_edge_tolerance_days(frame))
    first = frame["date"].min()
    last = frame["date"].max()
    return first <= pd.Timestamp(start) + tolerance and last >= pd.Timestamp(end) - tolerance


def _slice_cached_frame(frame: pd.DataFrame, start: date, end: date) -> pd.DataFrame:
    mask = (frame["date"] >= pd.Timestamp(start)) & (frame["date"] <= pd.Timestamp(end))
    return frame.loc[mask, ["date", "close", "volume"]].copy()


def _normalize_downloaded_frame(raw: pd.DataFrame) -> pd.DataFrame:
    if isinstance(raw.columns, pd.MultiIndex):
        raw.columns = [c[0] if isinstance(c, tuple) else c for c in raw.columns]
    raw = raw.reset_index()
    date_col = "Date" if "Date" in raw.columns else raw.columns[0]
    close_col = "Close" if "Close" in raw.columns else "Adj Close" if "Adj Close" in raw.columns else None
    if close_col is None:
        raise ValueError("Downloaded data did not contain a Close or Adj Close column.")
    out = pd.DataFrame(
        {
            "date": pd.to_datetime(raw[date_col], errors="coerce"),
            "close": pd.to_numeric(raw[close_col], errors="coerce"),
            "volume": pd.to_numeric(raw["Volume"], errors="coerce") if "Volume" in raw.columns else np.nan,
        }
    ).dropna(subset=["date", "close"])
    out = out.sort_values("date").drop_duplicates("date")
    return out


def generate_sample_market_data(asset: str, start: Optional[date], end: Optional[date]) -> pd.DataFrame:
    """Generate deterministic regime-switching sample data.

    This is intentionally not the default data path. It exists for offline tests,
    reproducible examples and graceful fallback when `data_mode='auto'` is selected.
    """
    rng = np.random.default_rng(_asset_seed(asset))
    continuous = asset.upper() in {"BTC-USD", "ETH-USD"}
    dates = _date_range(start, end, continuous=continuous)
    n = len(dates)
    if n < 180:
        dates = pd.date_range(dates.min(), periods=360, freq="D") if continuous else pd.bdate_range(dates.min(), periods=360)
        n = len(dates)

    transition = np.array(
        [
            [0.945, 0.045, 0.010],
            [0.075, 0.850, 0.075],
            [0.035, 0.115, 0.850],
        ]
    )
    # State 0: expansion, state 1: transition/sideways, state 2: stress.
    drifts = np.array([0.00058, 0.00004, -0.00092])
    vols = np.array([0.0062, 0.0120, 0.0250])

    states = np.zeros(n, dtype=int)
    for i in range(1, n):
        states[i] = rng.choice([0, 1, 2], p=transition[states[i - 1]])

    shock = rng.normal(0, 1, size=n)
    cluster = pd.Series(np.abs(shock)).rolling(8, min_periods=1).mean().to_numpy()
    returns = drifts[states] + vols[states] * shock * (0.75 + 0.25 * cluster)

    base_price = {
        "SPY": 475,
        "QQQ": 420,
        "BTC-USD": 68000,
        "ETH-USD": 3400,
        "AAPL": 205,
        "MSFT": 430,
        "NVDA": 125,
        "META": 520,
        "IWM": 210,
        "DIA": 390,
        "GLD": 230,
        "TLT": 92,
    }.get(asset.upper(), 100)
    close = base_price * np.exp(np.cumsum(returns))
    volume = rng.lognormal(mean=16, sigma=0.35, size=n).astype(int)

    return pd.DataFrame(
        {
            "date": dates,
            "close": close,
            "volume": volume,
            "synthetic_state": states,
        }
    )


def _load_from_cache(asset: str, start: date, end: date) -> Optional[LoadedData]:
    exact_path = _cache_path(asset, start, end)
    candidates = [exact_path] if exact_path.exists() else []
    candidates.extend([p for p in _cache_glob(asset) if p != exact_path])

    eligible: list[tuple[int, float, Path, pd.DataFrame]] = []
    requested_span = max(1, (end - start).days)
    for path in candidates:
        try:
            full_frame = _read_cache_file(path)
        except Exception:
            continue
        if not _cache_covers_request(full_frame, start, end):
            continue
        cached = _slice_cached_frame(full_frame, start, end)
        if len(cached) < 180:
            continue

        covered_span = max(1, (full_frame["date"].max() - full_frame["date"].min()).days)
        excess_span = max(0, covered_span - requested_span)
        eligible.append((excess_span, -path.stat().st_mtime, path, cached))

    if not eligible:
        return None

    # Prefer the narrowest cache that fully covers the requested dates; use recency as tie-breaker.
    _, _, _, cached = min(eligible, key=lambda item: (item[0], item[1]))
    return LoadedData(
        frame=cached,
        source="cache:yfinance",
        cache_hit=True,
        provider="yfinance",
        requested_start=start,
        requested_end=end,
    )


def _download_yfinance(asset: str, start: date, end: date) -> LoadedData:
    try:
        import yfinance as yf
    except Exception as exc:  # pragma: no cover - environment dependent
        raise MarketDataError(f"yfinance is not installed or cannot be imported: {exc}") from exc

    try:
        # yfinance end is exclusive; add one day so explicit end dates are included when available.
        raw = yf.download(
            asset,
            start=str(start),
            end=str(end + timedelta(days=1)),
            progress=False,
            auto_adjust=True,
            group_by="column",
            threads=False,
        )
        if raw is None or raw.empty:
            raise ValueError("provider returned an empty frame")
        out = _normalize_downloaded_frame(raw)
        if len(out) < 180:
            raise ValueError(f"provider returned too few valid observations ({len(out)} < 180)")
        cache_path = _cache_path(asset, start, end)
        out.to_csv(cache_path, index=False)
        return LoadedData(
            frame=out,
            source="yfinance",
            cache_hit=False,
            provider="yfinance",
            requested_start=start,
            requested_end=end,
        )
    except Exception as exc:  # pragma: no cover - external API best effort
        raise MarketDataError(f"Could not load real market data for {asset} from yfinance: {exc}") from exc


def load_real_market_data(asset: str, start: date, end: date, force_refresh: bool = False) -> LoadedData:
    """Load real market data from cache/yfinance, never deterministic sample data."""
    if not force_refresh:
        cached = _load_from_cache(asset, start, end)
        if cached is not None:
            return cached
    return _download_yfinance(asset, start, end)


def _resolve_mode(data_mode: Optional[str], prefer_live_data: Optional[bool]) -> DataMode:
    """Resolve the new data-mode API while preserving the old prefer_live_data flag."""
    if prefer_live_data is not None and data_mode is None:
        return "auto" if prefer_live_data else "sample"
    normalized = (data_mode or "real").lower()
    if normalized not in DATA_MODES:
        raise ValueError(f"Unsupported data_mode '{data_mode}'. Expected one of: {', '.join(DATA_MODES)}")
    return normalized  # type: ignore[return-value]


def load_market_data(
    asset: str,
    start: Optional[date] = None,
    end: Optional[date] = None,
    interval: Optional[str] = "5Y",
    data_mode: Optional[str] = "real",
    prefer_live_data: Optional[bool] = None,
    force_refresh: bool = False,
) -> LoadedData:
    """Load market data according to an explicit data-source policy.

    Modes:
    - real: require cache/yfinance data; raise `MarketDataError` if unavailable.
    - auto: try cache/yfinance first; fallback to deterministic sample data with warning.
    - sample: deterministic sample data only.

    `prefer_live_data` is kept for backward compatibility. New callers should use
    `data_mode` because it is explicit about whether sample fallback is allowed.
    """
    asset = asset.upper()
    resolved_start, resolved_end, _ = resolve_window(start, end, interval)
    mode = _resolve_mode(data_mode, prefer_live_data)

    if mode == "sample":
        return LoadedData(
            frame=generate_sample_market_data(asset, resolved_start, resolved_end),
            source="sample",
            provider="deterministic_sample",
            requested_start=resolved_start,
            requested_end=resolved_end,
        )

    try:
        return load_real_market_data(asset, resolved_start, resolved_end, force_refresh=force_refresh)
    except MarketDataError as exc:
        if mode == "real":
            raise
        return LoadedData(
            frame=generate_sample_market_data(asset, resolved_start, resolved_end),
            source="sample",
            warning=f"Real market data failed; using deterministic sample data because data_mode='auto'. Reason: {exc}",
            provider="deterministic_sample",
            requested_start=resolved_start,
            requested_end=resolved_end,
        )


def parse_uploaded_csv(contents: bytes) -> pd.DataFrame:
    from io import BytesIO

    frame = pd.read_csv(BytesIO(contents))
    frame.columns = [c.strip().lower() for c in frame.columns]
    if "date" not in frame.columns or "close" not in frame.columns:
        raise ValueError("CSV must contain at least 'date' and 'close' columns.")
    out = pd.DataFrame(
        {
            "date": pd.to_datetime(frame["date"], errors="coerce"),
            "close": pd.to_numeric(frame["close"], errors="coerce"),
            "volume": pd.to_numeric(frame.get("volume", np.nan), errors="coerce"),
        }
    )
    out = out.dropna(subset=["date", "close"]).sort_values("date")
    out = out.drop_duplicates("date")
    if len(out) < 180:
        raise ValueError("CSV must contain at least 180 valid observations for regime analysis.")
    return out
