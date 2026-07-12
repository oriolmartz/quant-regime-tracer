from __future__ import annotations

from typing import Any

import numpy as np
import pandas as pd

from app.services.annualization import annualization_factor_from_frame


def _safe_float(value: Any, default: float | None = None) -> float | None:
    try:
        number = float(value)
    except (TypeError, ValueError):
        return default
    if not np.isfinite(number):
        return default
    return number


def _pct_rank(series: pd.Series) -> pd.Series:
    numeric = pd.to_numeric(series, errors="coerce")
    if numeric.notna().sum() <= 1:
        return pd.Series(0.5, index=series.index)
    return numeric.rank(pct=True).fillna(0.5)


def _format_date(value: Any) -> str:
    try:
        return pd.to_datetime(value).strftime("%Y-%m-%d")
    except Exception:
        return str(value)


def _feature_signal(feature: str, value: float | None, percentile: float | None) -> tuple[str, str]:
    pct = percentile or 0.5
    if feature == "rolling_volatility":
        if pct >= 0.80:
            return "stress evidence", "Volatility sits in the upper tail of this analysis window."
        if pct <= 0.25:
            return "expansion evidence", "Volatility is low relative to the window."
        return "neutral", "Volatility is near its historical middle range."
    if feature == "drawdown":
        if value is not None and value <= -0.12:
            return "stress evidence", "Drawdown is materially negative, which supports defensive regime interpretation."
        if value is not None and value >= -0.03:
            return "expansion evidence", "Drawdown is shallow, which weakens stress interpretation."
        return "transition evidence", "Drawdown is present but not extreme."
    if feature == "momentum_20":
        if value is not None and value > 0.04:
            return "expansion evidence", "20-period momentum is positive."
        if value is not None and value < -0.04:
            return "stress evidence", "20-period momentum is negative."
        return "sideways evidence", "20-period momentum is muted."
    if feature == "rsi":
        if value is not None and value >= 70:
            return "upper-tail trend", "RSI is elevated; regime review should check for extended trend conditions."
        if value is not None and value <= 35:
            return "stress / oversold evidence", "RSI is low, which can coincide with stress or exhaustion."
        return "neutral", "RSI is inside a middle range."
    if feature == "log_return":
        if pct >= 0.90:
            return "positive shock", "Return sits in the upper tail of the sample."
        if pct <= 0.10:
            return "negative shock", "Return sits in the lower tail of the sample."
        return "normal range", "One-period return is not an extreme observation."
    return "context", "Feature contributes to the standardized HMM input vector."


def _baseline_votes(frame: pd.DataFrame, idx: int, hmm_is_stress: bool) -> list[dict[str, Any]]:
    rolling_vol = pd.to_numeric(frame["rolling_volatility"], errors="coerce").fillna(0.0)
    drawdown = pd.to_numeric(frame["drawdown"], errors="coerce").fillna(0.0)
    log_return = pd.to_numeric(frame["log_return"], errors="coerce").fillna(0.0)

    vol_threshold = float(rolling_vol.quantile(0.80)) if len(rolling_vol) else 0.0
    periods_per_year = annualization_factor_from_frame(frame)
    ewma_vol = log_return.ewm(span=30, adjust=False).std().bfill().fillna(0.0) * np.sqrt(periods_per_year)
    ewma_threshold = float(ewma_vol.quantile(0.80)) if len(ewma_vol) else 0.0
    drawdown_threshold = float(drawdown.quantile(0.20)) if len(drawdown) else 0.0

    raw_votes = [
        (
            "Rolling-volatility quantile",
            bool(rolling_vol.iloc[idx] >= vol_threshold),
            _safe_float(rolling_vol.iloc[idx]),
            vol_threshold,
            "Stress if rolling volatility is in the top quintile of the selected window.",
        ),
        (
            "EWMA volatility stress",
            bool(ewma_vol.iloc[idx] >= ewma_threshold),
            _safe_float(ewma_vol.iloc[idx]),
            ewma_threshold,
            "Stress if exponentially weighted volatility is in the top quintile.",
        ),
        (
            "Drawdown stress",
            bool(drawdown.iloc[idx] <= drawdown_threshold),
            _safe_float(drawdown.iloc[idx]),
            drawdown_threshold,
            "Stress if drawdown is in the worst quintile of the selected window.",
        ),
    ]
    return [
        {
            "name": name,
            "stress_vote": stress_vote,
            "agrees_with_hmm_stress": bool(stress_vote == hmm_is_stress),
            "latest_value": value,
            "threshold": threshold,
            "rule": rule,
        }
        for name, stress_vote, value, threshold, rule in raw_votes
    ]


def _state_probabilities(row: pd.Series, n_regimes: int) -> list[dict[str, Any]]:
    probs = []
    for state in range(n_regimes):
        value = _safe_float(row.get(f"state_probability_{state}"), 0.0) or 0.0
        probs.append({"state": state, "probability": value})
    return sorted(probs, key=lambda item: item["probability"], reverse=True)



def _assignment_type(confidence: float | None, entropy: float | None) -> str:
    conf = float(confidence or 0.0)
    ent = float(entropy or 0.0)
    if conf >= 0.995 and ent <= 0.03:
        return "Near one-hot"
    if conf >= 0.80 and ent <= 0.18:
        return "Strong assignment"
    if conf >= 0.55:
        return "Probabilistic split"
    return "Ambiguous posterior"


def _trace_evidence_strength(posterior_entropy: float | None, baseline_agreement: float, feature_evidence: list[dict[str, Any]]) -> float:
    entropy_component = 1.0 - max(0.0, min(1.0, float(posterior_entropy or 0.0)))
    feature_extremity = 0.0
    if feature_evidence:
        extremities = []
        for item in feature_evidence:
            pct = item.get("percentile")
            if pct is not None:
                extremities.append(abs(float(pct) - 0.5) * 2.0)
        if extremities:
            feature_extremity = max(0.0, min(1.0, float(np.mean(sorted(extremities, reverse=True)[:3]))))
    score = 0.42 * entropy_component + 0.35 * max(0.0, min(1.0, baseline_agreement)) + 0.23 * feature_extremity
    return float(min(0.99, max(0.0, score)))

def _build_point(frame: pd.DataFrame, idx: int, transition_matrix: np.ndarray, transition_labels: list[str]) -> dict[str, Any]:
    row = frame.iloc[idx]
    state = int(row["regime"])
    label = str(row["regime_label"])
    n_regimes = len(transition_labels) if transition_labels else max(1, int(frame["regime"].max()) + 1)

    previous_state = int(frame["regime"].iloc[idx - 1]) if idx > 0 else None
    previous_label = str(frame["regime_label"].iloc[idx - 1]) if idx > 0 else None
    if previous_state is not None and transition_matrix.size:
        transition_prior = _safe_float(transition_matrix[previous_state, state], 0.0)
    else:
        transition_prior = None

    ranks = {
        "log_return": _pct_rank(frame["log_return"]),
        "rolling_volatility": _pct_rank(frame["rolling_volatility"]),
        "drawdown": _pct_rank(frame["drawdown"]),
        "momentum_20": _pct_rank(frame["momentum_20"]),
        "rsi": _pct_rank(frame["rsi"]),
    }
    feature_names = ["rolling_volatility", "drawdown", "momentum_20", "rsi", "log_return"]
    feature_evidence = []
    for feature in feature_names:
        value = _safe_float(row.get(feature))
        percentile = _safe_float(ranks[feature].iloc[idx], 0.5)
        signal, rationale = _feature_signal(feature, value, percentile)
        feature_evidence.append(
            {
                "feature": feature,
                "value": value,
                "percentile": percentile,
                "signal": signal,
                "rationale": rationale,
            }
        )

    hmm_is_stress = "stress" in label.lower()
    votes = _baseline_votes(frame, idx, hmm_is_stress)
    agreement_count = int(sum(1 for vote in votes if vote["agrees_with_hmm_stress"]))
    agreement_share = float(agreement_count / len(votes)) if votes else 0.0
    event_tags: list[str] = []
    if idx == len(frame) - 1:
        event_tags.append("latest")
    if idx > 0 and previous_state != state:
        event_tags.append("regime_shift")
    if (_safe_float(row.get("posterior_entropy"), 0.0) or 0.0) >= float(pd.to_numeric(frame["posterior_entropy"], errors="coerce").quantile(0.85)):
        event_tags.append("high_uncertainty")
    if (_safe_float(row.get("drawdown"), 0.0) or 0.0) <= float(pd.to_numeric(frame["drawdown"], errors="coerce").quantile(0.10)):
        event_tags.append("drawdown_tail")

    dominant = feature_evidence[0]
    stress_votes = sum(1 for vote in votes if vote["stress_vote"])
    posterior_confidence = _safe_float(row.get('regime_probability'), 0.0) or 0.0
    posterior_entropy = _safe_float(row.get('posterior_entropy'), 0.0) or 0.0
    assignment_type = _assignment_type(posterior_confidence, posterior_entropy)
    evidence_strength = _trace_evidence_strength(posterior_entropy, agreement_share, feature_evidence)
    interpretation = (
        f"The selected observation maps to {label} as a {assignment_type.lower()} latent-state assignment "
        f"(γ={posterior_confidence:.1%}, H(γ)={posterior_entropy:.1%}). "
        f"The strongest visible feature cue is {dominant['feature']} ({dominant['signal']}). "
        f"{agreement_count}/{len(votes)} transparent baselines agree with the HMM stress/non-stress interpretation. "
        "This is an explanation of state assignment, not a directional market forecast."
    )
    if previous_state is not None:
        interpretation += f" The previous-state transition prior into this state was {(transition_prior or 0.0):.1%}."

    return {
        "date": _format_date(row["date"]),
        "close": _safe_float(row.get("close")),
        "assigned_state": state,
        "semantic_label": label,
        "previous_state": previous_state,
        "previous_label": previous_label,
        "posterior_confidence": posterior_confidence,
        "posterior_entropy": posterior_entropy,
        "assignment_type": assignment_type,
        "evidence_strength": evidence_strength,
        "transition_prior": transition_prior,
        "baseline_agreement_count": agreement_count,
        "baseline_total": len(votes),
        "baseline_agreement_share": agreement_share,
        "baseline_stress_votes": stress_votes,
        "state_probabilities": _state_probabilities(row, n_regimes),
        "feature_evidence": feature_evidence,
        "baseline_votes": votes,
        "event_tags": event_tags,
        "inference_path": [
            "market observation",
            "engineered risk features",
            "HMM posterior gamma",
            "empirical Markov transition prior",
            "baseline stress cross-check",
            "semantic regime interpretation",
        ],
        "interpretation": interpretation,
    }


def build_regime_traceback(
    frame: pd.DataFrame,
    transition_matrix: np.ndarray,
    transition_labels: list[str],
    max_points: int = 120,
) -> dict[str, Any]:
    """Build auditable point-level evidence for regime assignments.

    The output is designed for the UI Traceback mode: users can inspect a sampled date
    and see the path from raw market observation to features, posterior probabilities,
    transition prior, baseline votes and final semantic label.
    """
    if frame.empty:
        return {
            "summary": "No traceback available for an empty time series.",
            "current": None,
            "points": [],
            "methodology": [],
        }

    n = len(frame)
    base_indices = set(np.linspace(0, n - 1, min(max_points, n)).astype(int).tolist())
    latest_idx = n - 1
    entropy = pd.to_numeric(frame.get("posterior_entropy"), errors="coerce").fillna(0.0)
    drawdown = pd.to_numeric(frame.get("drawdown"), errors="coerce").fillna(0.0)
    base_indices.add(latest_idx)
    base_indices.add(int(entropy.idxmax()))
    base_indices.add(int(drawdown.idxmin()))

    regimes = frame["regime"].astype(int).tolist()
    shift_indices = [idx for idx in range(1, n) if regimes[idx] != regimes[idx - 1]]
    for idx in shift_indices[-40:]:
        base_indices.add(idx)

    indices = sorted(idx for idx in base_indices if 0 <= idx < n)
    points = [_build_point(frame, idx, transition_matrix, transition_labels) for idx in indices]
    current = _build_point(frame, latest_idx, transition_matrix, transition_labels)

    return {
        "name": "Regime Traceback",
        "summary": "Most regime tools label the market; QuantRegimeTracer traces the evidence path behind the label.",
        "current": current,
        "points": points,
        "methodology": [
            "Select a date from the inferred state path.",
            "Inspect engineered volatility, drawdown, momentum, RSI and return features with within-window percentiles.",
            "Separate posterior state assignment strength gamma from uncertainty H(gamma) and overall evidence strength.",
            "Compare the assigned state with the empirical Markov transition prior from the previous state.",
            "Check whether transparent volatility/drawdown baselines agree with the HMM stress/non-stress interpretation.",
        ],
    }
