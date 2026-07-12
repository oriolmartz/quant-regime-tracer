from __future__ import annotations

from datetime import date
from typing import Any, Dict, List, Optional, Literal

from pydantic import BaseModel, Field


class AnalyzeRequest(BaseModel):
    asset: str = Field(default="SPY", description="Ticker symbol or supported market asset")
    start: Optional[date] = Field(default=None)
    end: Optional[date] = Field(default=None)
    interval: Optional[str] = Field(
        default="5Y",
        description="Preset analysis window when explicit start/end are not provided. Supported: 6M, 1Y, 2Y, 3Y, 5Y, MAX.",
    )
    n_regimes: int = Field(default=3, ge=2, le=5)
    data_mode: Literal['real', 'auto', 'sample'] = Field(
        default='real',
        description="Data policy: 'real' requires cache/yfinance, 'auto' allows deterministic sample fallback, 'sample' uses offline deterministic data.",
    )
    prefer_live_data: Optional[bool] = Field(
        default=None,
        description="Deprecated compatibility flag. Use data_mode instead.",
    )
    force_refresh: bool = Field(default=False, description="Bypass local cache and request fresh provider data when data_mode uses real data.")
    language: Literal['en', 'es'] = Field(default='en', description='Output language for memo/report fields.')


class TimePoint(BaseModel):
    date: str
    close: float
    log_return: Optional[float] = None
    rolling_volatility: Optional[float] = None
    drawdown: Optional[float] = None
    momentum_20: Optional[float] = None
    rsi: Optional[float] = None
    regime: int
    regime_label: str
    regime_probability: Optional[float] = None
    posterior_entropy: Optional[float] = None
    state_probability_0: Optional[float] = None
    state_probability_1: Optional[float] = None
    state_probability_2: Optional[float] = None
    state_probability_3: Optional[float] = None
    state_probability_4: Optional[float] = None


class RegimeStat(BaseModel):
    regime: int
    label: str
    observations: int
    mean_return: float
    annualized_volatility: float
    mean_drawdown: float
    mean_momentum: float
    latest_probability: float


class RegimeSegment(BaseModel):
    start: str
    end: str
    regime: int
    label: str
    observations: int
    segment_return: float
    max_drawdown: float


class ModelDiagnostics(BaseModel):
    model_type: str
    n_observations: int
    n_features: int
    n_regimes: int
    feature_columns: List[str]
    data_start: str
    data_end: str
    confidence_status: str
    separability_score: Optional[float] = None
    notes: List[str] = []




class CompareRequest(BaseModel):
    assets: List[str] = Field(default_factory=lambda: ["SPY", "QQQ", "BTC-USD"])
    start: Optional[date] = Field(default=None)
    end: Optional[date] = Field(default=None)
    interval: Optional[str] = Field(default="5Y")
    n_regimes: int = Field(default=3, ge=2, le=5)
    data_mode: Literal['real', 'auto', 'sample'] = Field(default='real')
    prefer_live_data: Optional[bool] = Field(default=None, description="Deprecated compatibility flag. Use data_mode instead.")
    force_refresh: bool = Field(default=False)
    language: Literal['en', 'es'] = Field(default='en')


class CompareAssetSummary(BaseModel):
    asset: str
    source: str
    current_regime: str
    confidence: float
    assignment_type: Optional[str] = None
    evidence_strength: Optional[float] = None
    risk_score: float
    risk_band: str
    stress_transition_probability: float
    stay_probability: float
    latest_drawdown: float
    annualized_volatility: float
    annualization_factor: float
    annualization_calendar: str
    baseline_agreement: float
    baseline_verdict: str
    data_quality_status: str
    warnings: List[str]


class CompareResponse(BaseModel):
    api_version: str
    interval: Optional[str]
    assets_requested: List[str]
    summaries: List[CompareAssetSummary]
    highest_risk_asset: Optional[str]
    lowest_risk_asset: Optional[str]
    average_risk_score: float
    portfolio_memo: Dict[str, Any]


class AnalyzeResponse(BaseModel):
    asset: str
    source: str
    interval: Optional[str]
    start: Optional[str]
    end: Optional[str]
    api_version: str
    warnings: List[str]
    data_quality: Dict[str, Any]
    source_report: Dict[str, Any]
    current_regime: Dict[str, Any]
    risk_metrics: Dict[str, Any]
    regime_stats: List[RegimeStat]
    transition_matrix: List[List[float]]
    transition_labels: List[str]
    time_series: List[TimePoint]
    regime_segments: List[RegimeSegment]
    baseline: Dict[str, Any]
    stability: Dict[str, Any]
    model_card: Dict[str, Any]
    diagnostics: ModelDiagnostics
    memo: Dict[str, Any]
    model_evaluation: Dict[str, Any]
    traceback: Dict[str, Any] = Field(default_factory=dict)
    current_traceback: Optional[Dict[str, Any]] = None
    traceback_points: List[Dict[str, Any]] = Field(default_factory=list)
    report_markdown: str
