from __future__ import annotations

from collections import Counter
from typing import Any

import numpy as np
import pandas as pd

from app.services.annualization import annualization_factor_from_frame, infer_annualization_profile
from app.services.risk_metrics import max_drawdown, transition_matrix


def regime_segments(frame: pd.DataFrame, max_segments: int = 12) -> list[dict[str, Any]]:
    """Compress the full regime path into contiguous episodes for UI and reporting."""
    if frame.empty:
        return []

    segments: list[dict[str, Any]] = []
    start_idx = 0
    current_regime = int(frame["regime"].iloc[0])
    current_label = str(frame["regime_label"].iloc[0])

    for i in range(1, len(frame)):
        state = int(frame["regime"].iloc[i])
        if state != current_regime:
            segments.append(_build_segment(frame.iloc[start_idx:i], current_regime, current_label))
            start_idx = i
            current_regime = state
            current_label = str(frame["regime_label"].iloc[i])
    segments.append(_build_segment(frame.iloc[start_idx:], current_regime, current_label))

    # Preserve the most recent episodes first because current regimes matter most.
    return segments[-max_segments:]


def _build_segment(part: pd.DataFrame, regime: int, label: str) -> dict[str, Any]:
    start_price = float(part["close"].iloc[0])
    end_price = float(part["close"].iloc[-1])
    segment_return = end_price / start_price - 1 if start_price else 0.0
    return {
        "start": part["date"].iloc[0].strftime("%Y-%m-%d"),
        "end": part["date"].iloc[-1].strftime("%Y-%m-%d"),
        "regime": int(regime),
        "label": label,
        "observations": int(len(part)),
        "segment_return": float(segment_return),
        "max_drawdown": float(max_drawdown(part["close"])),
    }


def _disagreement_segments(df: pd.DataFrame, max_segments: int = 8) -> list[dict[str, Any]]:
    if df.empty or "disagreement" not in df.columns:
        return []
    segments: list[dict[str, Any]] = []
    in_segment = False
    start_idx = 0
    for idx, flag in enumerate(df["disagreement"].tolist()):
        if flag and not in_segment:
            start_idx = idx
            in_segment = True
        elif not flag and in_segment:
            part = df.iloc[start_idx:idx]
            segments.append(_build_disagreement_segment(part))
            in_segment = False
    if in_segment:
        segments.append(_build_disagreement_segment(df.iloc[start_idx:]))
    return segments[-max_segments:]


def _build_disagreement_segment(part: pd.DataFrame) -> dict[str, Any]:
    return {
        "start": part["date"].iloc[0].strftime("%Y-%m-%d"),
        "end": part["date"].iloc[-1].strftime("%Y-%m-%d"),
        "observations": int(len(part)),
        "hmm_stress_share": float(part["hmm_stress"].mean()),
        "baseline_stress_share": float(part["baseline_stress"].mean()),
    }


def _baseline_verdict(stress_agreement: float) -> str:
    if stress_agreement >= 0.72:
        return "aligned"
    if stress_agreement >= 0.55:
        return "mixed"
    return "divergent"


def _stress_agreement_report(df: pd.DataFrame, stress_flag: pd.Series, name: str, description: str, latest_label: str) -> dict[str, Any]:
    review = pd.DataFrame({
        "date": df["date"],
        "hmm_stress": df["regime_label"].astype(str).str.lower().str.contains("stress"),
        "baseline_stress": stress_flag.astype(bool),
    })
    review["disagreement"] = review["hmm_stress"] != review["baseline_stress"]
    stress_agreement = float((~review["disagreement"]).mean()) if len(review) else 0.0
    return {
        "name": name,
        "description": description,
        "latest_label": latest_label,
        "stress_agreement": stress_agreement,
        "disagreement_rate": 1.0 - stress_agreement,
        "verdict": _baseline_verdict(stress_agreement),
        "disagreement_segments": _disagreement_segments(review),
    }


def _ewma_volatility_baseline(frame: pd.DataFrame, annualization_factor: float | None = None) -> dict[str, Any]:
    df = frame.copy()
    periods_per_year = float(annualization_factor or annualization_factor_from_frame(frame))
    ewma_vol = df["log_return"].ewm(span=30, adjust=False).std().bfill().fillna(0) * np.sqrt(periods_per_year)
    threshold = float(ewma_vol.quantile(0.80)) if len(ewma_vol) else 0.0
    stress = ewma_vol >= threshold
    latest_label = "EWMA volatility stress" if bool(stress.iloc[-1]) else "EWMA non-stress"
    report = _stress_agreement_report(
        df,
        stress,
        "EWMA volatility stress baseline",
        "Baseline that uses exponentially weighted volatility, so recent shocks matter more than old observations.",
        latest_label,
    )
    report.update({"threshold": threshold, "latest_value": float(ewma_vol.iloc[-1]) if len(ewma_vol) else None})
    return report


def _drawdown_stress_baseline(frame: pd.DataFrame) -> dict[str, Any]:
    df = frame.copy()
    drawdown = pd.to_numeric(df["drawdown"], errors="coerce").fillna(0.0)
    threshold = float(drawdown.quantile(0.20)) if len(drawdown) else 0.0
    stress = drawdown <= threshold
    latest_label = "Drawdown stress" if bool(stress.iloc[-1]) else "Drawdown contained"
    report = _stress_agreement_report(
        df,
        stress,
        "Drawdown stress baseline",
        "Baseline that marks stress when the asset sits in the worst drawdown quintile of the analysis window.",
        latest_label,
    )
    report.update({"threshold": threshold, "latest_value": float(drawdown.iloc[-1]) if len(drawdown) else None})
    return report


def baseline_volatility_regimes(
    frame: pd.DataFrame,
    n_regimes: int,
    annualization_factor: float | None = None,
) -> dict[str, Any]:
    """Transparent baseline suite for skeptical model review.

    The primary baseline buckets rolling volatility into quantile regimes. Two additional
    baselines use different assumptions: EWMA volatility reacts faster to recent shocks,
    while drawdown stress checks whether the HMM stress state maps to realised damage.
    """
    df = frame.copy()
    n_bins = min(max(2, n_regimes), max(2, df["rolling_volatility"].nunique()))
    try:
        df["baseline_state"] = pd.qcut(df["rolling_volatility"], q=n_bins, labels=False, duplicates="drop")
    except ValueError:
        df["baseline_state"] = 0

    max_bucket = int(df["baseline_state"].max()) if df["baseline_state"].notna().any() else 0
    primary_stress = df["baseline_state"] == max_bucket
    df["baseline_label"] = np.where(primary_stress, "Volatility stress", "Non-stress")
    df["hmm_stress"] = df["regime_label"].astype(str).str.lower().str.contains("stress")
    df["baseline_stress"] = primary_stress
    df["disagreement"] = df["hmm_stress"] != df["baseline_stress"]
    stress_agreement = float((~df["disagreement"]).mean()) if len(df) else 0.0
    disagreement_rate = 1.0 - stress_agreement

    latest_state = int(df["baseline_state"].iloc[-1]) if len(df) else 0
    counts = Counter(df["baseline_state"].dropna().astype(int).tolist())
    distribution = [
        {"bucket": int(bucket), "observations": int(count), "share": float(count / len(df))}
        for bucket, count in sorted(counts.items())
    ]

    baselines = [
        {
            "name": "Rolling-volatility quantile baseline",
            "description": "Primary transparent baseline that buckets observations by rolling volatility only.",
            "latest_label": str(df["baseline_label"].iloc[-1]) if len(df) else "—",
            "stress_agreement": stress_agreement,
            "disagreement_rate": disagreement_rate,
            "verdict": _baseline_verdict(stress_agreement),
        },
        _ewma_volatility_baseline(frame, annualization_factor=annualization_factor),
        _drawdown_stress_baseline(frame),
    ]
    suite_mean_agreement = float(np.mean([b["stress_agreement"] for b in baselines])) if baselines else 0.0

    return {
        "name": "Rolling-volatility quantile baseline",
        "description": "Primary transparent baseline that buckets observations by rolling volatility only.",
        "latest_state": latest_state,
        "latest_label": str(df["baseline_label"].iloc[-1]) if len(df) else "—",
        "stress_agreement": stress_agreement,
        "disagreement_rate": disagreement_rate,
        "verdict": _baseline_verdict(stress_agreement),
        "distribution": distribution,
        "disagreement_segments": _disagreement_segments(df),
        "baseline_suite": baselines,
        "suite_mean_agreement": suite_mean_agreement,
        "suite_verdict": _baseline_verdict(suite_mean_agreement),
        "interpretation": (
            "The primary baseline checks HMM stress against rolling-volatility buckets. The suite also compares EWMA volatility "
            "and drawdown stress, so the HMM is not benchmarked only against a single weak sanity check."
        ),
    }


def transition_stability(frame: pd.DataFrame, n_regimes: int) -> dict[str, Any]:
    """Compare first-half and second-half transition behavior."""
    if len(frame) < 240:
        return {
            "status": "insufficient_window",
            "message": "At least 240 engineered observations are recommended for transition stability review.",
            "frobenius_distance": None,
        }

    mid = len(frame) // 2
    first = transition_matrix(frame["regime"].iloc[:mid].to_numpy(), n_regimes)
    second = transition_matrix(frame["regime"].iloc[mid:].to_numpy(), n_regimes)
    distance = float(np.linalg.norm(first - second))
    if distance < 0.45:
        status = "stable"
    elif distance < 0.85:
        status = "moderate_drift"
    else:
        status = "unstable"
    return {
        "status": status,
        "message": "First-half and second-half transition matrices were compared for drift.",
        "frobenius_distance": distance,
    }


def data_quality_report(frame: pd.DataFrame, source: str, cache_hit: bool = False) -> dict[str, Any]:
    dates = pd.to_datetime(frame["date"], errors="coerce")
    duplicate_dates = int(dates.duplicated().sum())
    missing_close = int(pd.to_numeric(frame.get("close"), errors="coerce").isna().sum())
    missing_volume = int(pd.to_numeric(frame.get("volume", pd.Series(dtype=float)), errors="coerce").isna().sum()) if "volume" in frame.columns else None
    date_span_days = int((dates.max() - dates.min()).days) if len(dates.dropna()) else 0
    gaps = dates.sort_values().diff().dt.days.dropna()
    largest_gap_days = int(gaps.max()) if len(gaps) else 0
    observations = int(len(frame))

    status = "strong"
    notes = []
    if observations < 240:
        status = "fragile"
        notes.append("Short analysis window; transition estimates can be unstable.")
    if largest_gap_days > 10:
        status = "review"
        notes.append("Detected a large date gap; check calendar continuity or data source quality.")
    if missing_close > 0:
        status = "review"
        notes.append("Close prices contained missing values before cleaning.")
    if source == "sample":
        notes.append("Deterministic sample data is used; this run is not backed by real market data.")
    if cache_hit:
        notes.append("Loaded from local cache to avoid repeated live-provider calls.")

    annualization_profile = infer_annualization_profile(frame["date"]) if "date" in frame.columns else {}

    return {
        "status": status,
        "source": source,
        "cache_hit": bool(cache_hit),
        "observations": observations,
        "date_start": dates.min().strftime("%Y-%m-%d") if len(dates.dropna()) else None,
        "date_end": dates.max().strftime("%Y-%m-%d") if len(dates.dropna()) else None,
        "date_span_days": date_span_days,
        "largest_gap_days": largest_gap_days,
        "duplicate_dates": duplicate_dates,
        "missing_close": missing_close,
        "missing_volume": missing_volume,
        "annualization_factor": annualization_profile.get("periods_per_year"),
        "annualization_calendar": annualization_profile.get("calendar_type"),
        "annualization_method": annualization_profile.get("method"),
        "notes": notes,
    }


def build_model_card(model_type: str, n_regimes: int, feature_cols: list[str]) -> dict[str, Any]:
    return {
        "model_name": "QuantRegimeTracer HMM Regime Engine",
        "version": "0.10.0",
        "model_type": model_type,
        "intended_use": "Decision-support context for regime review, risk monitoring and portfolio discussion.",
        "not_intended_for": [
            "Directional buy/sell recommendations",
            "Autonomous trading execution",
            "Financial advice",
            "Ground-truth classification of market states",
        ],
        "inputs": feature_cols,
        "outputs": [
            f"{n_regimes} inferred regimes",
            "Regime labels derived from state statistics",
            "Empirical Markov transition matrix",
            "Baseline comparison against volatility-only regimes",
            "Point-level Regime Traceback evidence",
            "Guarded executive risk memo",
        ],
        "assumptions": [
            "Recent historical structure contains useful information about regime behavior.",
            "Feature engineering captures enough volatility, trend and drawdown context for unsupervised clustering.",
            "Regime labels are interpretability aids, not true labels.",
        ],
        "failure_modes": [
            "Small samples can produce unstable regimes.",
            "Non-stationary markets can invalidate transition estimates quickly.",
            "HMM states may rotate between runs or assets; labels must be derived, not hard-coded.",
            "Real-data outages are explicit; fallback only occurs in data_mode='auto'.",
        ],
        "guardrails": [
            "Memo generator avoids buy/sell instructions.",
            "Output is framed as risk review, not prediction certainty.",
            "Warnings, data-quality checks, traceback evidence and assignment and uncertainty diagnostics are surfaced in the UI.",
        ],
    }
