from __future__ import annotations

from typing import Any

import numpy as np
import pandas as pd
from fastapi import FastAPI, File, HTTPException, UploadFile
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import PlainTextResponse

from app.graphs.memo_graph import generate_memo_with_graph
from app.models.schemas import AnalyzeRequest, AnalyzeResponse, CompareRequest, CompareResponse
from app.services.data_loader import (
    DATA_MODES,
    SUPPORTED_ASSETS,
    WINDOW_PRESETS,
    MarketDataError,
    generate_sample_market_data,
    load_market_data,
    parse_uploaded_csv,
    resolve_window,
)
from app.services.annualization import infer_annualization_profile
from app.services.features import engineer_features
from app.services.model_evaluation import temporal_hmm_evaluation
from app.services.regime_model import run_regime_model
from app.services.risk_metrics import annualized_volatility, max_drawdown
from app.services.traceback import build_regime_traceback
from app.services.validation import (
    baseline_volatility_regimes,
    build_model_card,
    data_quality_report,
    regime_segments,
    transition_stability,
)
from app.utils.serialization import clean_number

API_VERSION = "0.10.0"

app = FastAPI(
    title="QuantRegimeTracer API",
    description="Auditable market-regime inference with HMMs, Markov transitions, real-market-data ingestion, validation diagnostics and point-level traceback.",
    version=API_VERSION,
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["http://localhost:5173", "http://127.0.0.1:5173", "*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


@app.get("/health")
def health() -> dict[str, str]:
    return {"status": "ok", "service": "QuantRegimeTracer API", "version": API_VERSION}


@app.get("/assets")
def assets() -> dict[str, list[str]]:
    return {"assets": SUPPORTED_ASSETS}


@app.get("/time-windows")
def time_windows() -> dict[str, list[str]]:
    return {"windows": WINDOW_PRESETS}


@app.get("/data-sources")
def data_sources() -> dict[str, Any]:
    return {
        "default_mode": "real",
        "modes": DATA_MODES,
        "providers": ["yfinance", "local cache", "uploaded_csv", "deterministic sample"],
        "policy": {
            "real": "Require real market data from local cache or yfinance. No sample fallback.",
            "auto": "Try cache/yfinance first, then fall back to deterministic sample data with an explicit warning.",
            "sample": "Use deterministic offline sample data for tests and reproducibility only.",
        },
    }


@app.get("/project-card")
def project_card() -> dict[str, Any]:
    """Technical project metadata exposed for downstream documentation or portfolio pages."""
    return {
        "name": "QuantRegimeTracer",
        "tagline": "Auditable market-regime inference and transition-risk tracing.",
        "description": (
            "A full-stack quantitative research workbench that detects hidden market regimes with HMMs, "
            "estimates Markov transition probabilities, validates against a baseline suite, recommends a regime count with BIC, "
            "checks multi-seed assignment stability, uses real-market-data ingestion and traces point-level evidence paths behind regime labels."
        ),
        "stack": ["FastAPI", "React", "TypeScript", "Tailwind", "HMM", "Markov Chains", "LangGraph", "Recharts", "Docker"],
        "positioning": "Auditable regime-inference workbench for risk review, not a trading bot.",
        "version": API_VERSION,
        "release_highlights": [
            "Cross-asset regime comparison",
            "Bilingual EN/ES interface and memo generation",
            "TypeScript frontend with typed API contracts",
            "Validation panels for baseline-suite agreement, transition stability and data quality",
            "Temporal HMM model selection with BIC/AIC and held-out log-likelihood",
            "Multi-seed assignment stability review for HMM robustness",
            "Regime Traceback mode for point-level evidence behind each label",
            "Pytest suite, smoke test and lockfile-based frontend Docker build",
        ],
    }


@app.get("/case-study")
def case_study() -> dict[str, Any]:
    return {
        "title": "QuantRegimeTracer — Auditable Market-Regime Inference Workbench",
        "problem": "Market data is noisy, regime shifts are hard to explain, and naive dashboards often hide model uncertainty.",
        "approach": [
            "Engineer volatility, drawdown, trend and momentum features from price history.",
            "Infer hidden regimes with a Gaussian HMM, using KMeans only as an explicit offline/failure fallback.",
            "Estimate empirical Markov transition probabilities over the inferred state path.",
            "Validate stress regimes against a baseline suite: rolling-volatility quantiles, EWMA volatility stress and drawdown stress.",
            "Score HMM candidates on a chronological train/test split with AIC/BIC and held-out log-likelihood diagnostics.",
            "Recommend a regime count via BIC and check whether the user-selected count matches the evidence.",
            "Refit the selected HMM across multiple seeds and report assignment stability with Adjusted Rand Index.",
            "Generate a guarded executive memo through a LangGraph-style analysis workflow.",
            "Expose Regime Traceback: feature evidence, posterior state mass, Markov transition prior and baseline agreement for selected dates.",
        ],
        "architecture": [
            "FastAPI analysis backend",
            "Feature engineering and HMM regime engine",
            "Validation layer: baseline suite + model selection + multi-seed stability + transition stability + data quality",
            "Traceback layer: date-level evidence path from raw observation to semantic regime interpretation",
            "React/TypeScript/Tailwind dashboard",
            "Markdown/JSON export for review",
            "Cross-asset comparison endpoint for portfolio-level regime review",
        ],
        "limitations": [
            "Regimes are inferred, not ground-truth labels.",
            "Transition probabilities are historical estimates and can decay quickly under structural change.",
            "The app is for technical risk review and model diagnostics, not financial advice.",
        ],
    }


@app.get("/quickstart")
def quickstart() -> dict[str, Any]:
    return {
        "version": API_VERSION,
        "backend": [
            "cd backend",
            "python -m venv .venv",
            r".\.venv\Scripts\activate  # Windows PowerShell",
            "pip install -r requirements.txt",
            "uvicorn app.main:app --reload --port 8000",
        ],
        "frontend": [
            "cd frontend",
            "npm ci",
            "npm run doctor",
            "npm run dev",
        ],
        "common_windows_fixes": [
            "If npm is not recognized, install Node.js or add portable Node to PATH.",
            "If npm install fails with ENOSPC, free disk space and run npm cache clean --force.",
            "Tailwind is pinned to v3.4.17 to avoid the Tailwind v4 PostCSS plugin breaking change.",
        ],
    }


@app.get("/model-card")
def model_card() -> dict[str, Any]:
    return build_model_card("GaussianHMM / KMeans fallback", 3, [
        "log_return",
        "rolling_volatility",
        "rolling_mean_return",
        "drawdown",
        "ma_distance",
        "momentum_20",
        "rsi",
        "optional_volume_change",
    ])


@app.get("/sample-csv", response_class=PlainTextResponse)
def sample_csv() -> str:
    frame = generate_sample_market_data("SPY", None, None)[["date", "close", "volume"]].head(260)
    return frame.to_csv(index=False)


def _chart_frame(out: pd.DataFrame, max_points: int = 800) -> pd.DataFrame:
    if len(out) > max_points:
        return out.iloc[np.linspace(0, len(out) - 1, max_points).astype(int)].copy()
    return out.copy()


def _risk_band(score: float) -> str:
    if score >= 0.66:
        return "elevated"
    if score >= 0.38:
        return "moderate"
    return "contained"



def _bounded(value: Any, default: float = 0.0) -> float:
    try:
        if value is None:
            return default
        number = float(value)
        if np.isnan(number):
            return default
        return float(min(1.0, max(0.0, number)))
    except Exception:
        return default


def _assignment_label(confidence: float, entropy: float, model_type: str | None = None) -> str:
    model = str(model_type or '').lower()
    if 'kmeans' in model:
        return 'Hard cluster proxy'
    if confidence >= 0.995 and entropy <= 0.03:
        return 'Near one-hot'
    if confidence >= 0.80 and entropy <= 0.18:
        return 'Strong assignment'
    if confidence >= 0.55:
        return 'Probabilistic split'
    return 'Ambiguous posterior'


def _stability_component(model_evaluation: dict[str, Any]) -> float:
    stability = model_evaluation.get('assignment_stability') or {}
    verdict = str(stability.get('verdict') or stability.get('status') or '').lower()
    if verdict == 'stable':
        return 0.92
    if verdict == 'moderate':
        return 0.68
    if verdict == 'unstable':
        return 0.35
    if verdict in {'ok', 'healthy'}:
        return 0.80
    if verdict in {'unavailable', 'insufficient_window'}:
        return 0.55
    return 0.60


def _data_quality_component(data_quality: dict[str, Any]) -> float:
    status = str(data_quality.get('status') or '').lower()
    if status in {'ok', 'good', 'passed'}:
        return 0.95
    if 'warn' in status or 'caveat' in status or 'usable' in status:
        return 0.72
    if status in {'failed', 'error'}:
        return 0.35
    return 0.70


def _augment_current_regime(
    current: dict[str, Any],
    baseline: dict[str, Any],
    model_evaluation: dict[str, Any],
    data_quality: dict[str, Any],
    diagnostics: dict[str, Any],
) -> dict[str, Any]:
    augmented = dict(current)
    confidence = _bounded(augmented.get('confidence'))
    posterior_entropy = _bounded(augmented.get('posterior_entropy'))
    transition_entropy = _bounded(augmented.get('transition_entropy'))
    baseline_agreement = _bounded(
        baseline.get('suite_mean_agreement', baseline.get('stress_agreement', baseline.get('agreement', None))),
        default=0.50,
    )
    stability_score = _stability_component(model_evaluation)
    selection_score = 1.0 if model_evaluation.get('selected_matches_recommendation', True) else 0.68
    quality_score = _data_quality_component(data_quality)

    # This is intentionally not the same as HMM posterior mass. It is a traceability
    # score combining assignment sharpness with independent checks and robustness signals.
    assignment_component = 0.65 * (1.0 - posterior_entropy) + 0.35 * confidence
    evidence_strength = (
        0.26 * assignment_component
        + 0.24 * baseline_agreement
        + 0.20 * stability_score
        + 0.15 * selection_score
        + 0.15 * quality_score
    )

    augmented.update({
        'assignment_strength': confidence,
        'assignment_type': _assignment_label(confidence, posterior_entropy, diagnostics.get('model_type')),
        'evidence_strength': float(min(0.99, max(0.0, evidence_strength))),
        'evidence_components': {
            'assignment_sharpness': float(assignment_component),
            'baseline_agreement': float(baseline_agreement),
            'multi_seed_stability': float(stability_score),
            'model_selection_alignment': float(selection_score),
            'data_quality': float(quality_score),
            'transition_uncertainty': float(transition_entropy),
        },
        'confidence_interpretation': 'HMM posterior mass is a latent-state assignment signal, not directional market certainty.',
    })
    return augmented

def _build_report_markdown(
    asset: str,
    source: str,
    interval: str | None,
    result,
    risk_metrics: dict[str, Any],
    memo: dict[str, Any],
    baseline: dict[str, Any],
    stability: dict[str, Any],
    data_quality: dict[str, Any],
    model_evaluation: dict[str, Any] | None = None,
    traceback: dict[str, Any] | None = None,
    language: str = "en",
) -> str:
    current = result.current_regime
    model_evaluation = model_evaluation or {}
    traceback = traceback or {}
    current_traceback = traceback.get("current") or {}
    evidence = memo.get("evidence_observed", [])
    limitations = memo.get("model_limitations", [])
    considerations = memo.get("portfolio_review_considerations", [])
    if language == "es":
        labels = {
            "version": "Versión", "source": "Fuente", "interval": "Intervalo", "window": "Ventana",
            "model": "Modelo", "current": "Régimen actual", "label": "Etiqueta", "confidence": "Fuerza de asignación",
            "risk_band": "Banda de riesgo", "stay": "Probabilidad de permanencia", "stress": "Probabilidad de transición a estrés",
            "evidence": "Evidencia observada", "baseline": "Baseline, estabilidad y calidad de datos",
            "review": "Consideraciones de revisión", "limits": "Limitaciones",
            "footer": "Este informe se genera para revisión técnica de riesgo. No es asesoramiento financiero y no proporciona instrucciones de compra/venta.",
            "custom": "personalizado", "observations": "observaciones", "largest_gap": "gap máximo",
            "traceback": "Trazabilidad de régimen", "posterior": "MAP posterior gamma", "entropy": "Entropía posterior",
            "transition_prior": "Prior de transición", "baseline_agreement": "Acuerdo con baselines",
            "annualization": "Factor de anualización"
        }
        title = f"# Informe QuantRegimeTracer — {asset}"
    else:
        labels = {
            "version": "Version", "source": "Source", "interval": "Interval", "window": "Window",
            "model": "Model", "current": "Current regime", "label": "Label", "confidence": "Assignment strength",
            "risk_band": "Risk band", "stay": "Stay probability", "stress": "Stress transition probability",
            "evidence": "Evidence observed", "baseline": "Baseline, stability and data quality",
            "review": "Review considerations", "limits": "Limitations",
            "footer": "This report is generated for technical risk review. It is not financial advice and does not provide buy/sell instructions.",
            "custom": "custom", "observations": "observations", "largest_gap": "largest gap",
            "traceback": "Regime Traceback", "posterior": "MAP posterior gamma", "entropy": "Posterior entropy",
            "transition_prior": "Transition prior", "baseline_agreement": "Baseline agreement",
            "annualization": "Annualization factor"
        }
        title = f"# QuantRegimeTracer Report — {asset}"

    lines = [
        title,
        "",
        f"**{labels['version']}:** {API_VERSION}",
        f"**{labels['source']}:** {source}",
        f"**{labels['interval']}:** {interval or labels['custom']}",
        f"**{labels['window']}:** {result.diagnostics.get('data_start')} → {result.diagnostics.get('data_end')}",
        f"**{labels['model']}:** {result.diagnostics.get('model_type')} · {result.diagnostics.get('n_regimes')} regimes",
        f"**{labels['annualization']}:** {risk_metrics.get('annualization_factor', 252):g} periods/year · {risk_metrics.get('annualization_calendar', 'inferred calendar')}",
        "",
        f"## {labels['current']}",
        f"- {labels['label']}: **{current.get('label')}**",
        f"- Evidence strength: {current.get('evidence_strength', 0):.1%}",
        f"- Assignment type: {current.get('assignment_type', '—')}",
        f"- {labels['confidence']}: {current.get('assignment_strength', current.get('confidence', 0)):.1%}",
        f"- {labels['risk_band']}: {_risk_band(current.get('risk_score', 0))}",
        f"- {labels['stay']}: {current.get('stay_probability', 0):.1%}",
        f"- {labels['stress']}: {current.get('stress_transition_probability', 0):.1%}",
        "",
        f"## {labels['evidence']}",
    ]
    lines.extend([f"- {item}" for item in evidence])
    lines.extend([
        "",
        f"## {labels['baseline']}",
        f"- Baseline: {baseline.get('name')}",
        f"- Primary baseline stress agreement: {baseline.get('stress_agreement', 0):.1%}",
        f"- Baseline-suite mean agreement: {baseline.get('suite_mean_agreement', 0):.1%}",
        f"- Baseline-suite verdict: {baseline.get('suite_verdict') or baseline.get('verdict')}",
        f"- Transition stability: {stability.get('status')} ({stability.get('message')})",
        f"- Data quality: {data_quality.get('status')} · {data_quality.get('observations')} {labels['observations']} · {labels['largest_gap']} {data_quality.get('largest_gap_days')} days",
        "",
        "## HMM temporal holdout",
        f"- Status: {model_evaluation.get('status', 'unavailable')}",
        f"- Selected regimes: {model_evaluation.get('selected_regimes', '—')}",
        f"- Recommended regimes: {model_evaluation.get('recommended_n_regimes', model_evaluation.get('best_bic_regimes', '—'))}",
        f"- Selected matches recommendation: {model_evaluation.get('selected_matches_recommendation', '—')}",
        f"- Best BIC regimes: {model_evaluation.get('best_bic_regimes', '—')}",
        f"- Selected train/test log-likelihood gap: {model_evaluation.get('selected_train_test_loglik_gap', '—')}",
        f"- Multi-seed stability: {(model_evaluation.get('assignment_stability') or {}).get('verdict', (model_evaluation.get('assignment_stability') or {}).get('status', '—'))}",
        "- Interpretation: chronological train/test log-likelihood, AIC/BIC and multi-seed ARI are diagnostics, not a trading backtest.",
        "",
        f"## {labels['traceback']}",
        f"- {labels['posterior']}: {current_traceback.get('posterior_confidence', 0):.1%}",
        f"- {labels['entropy']}: {current_traceback.get('posterior_entropy', 0):.1%}",
        f"- {labels['transition_prior']}: {current_traceback.get('transition_prior'):.1%}" if current_traceback.get('transition_prior') is not None else f"- {labels['transition_prior']}: —",
        f"- {labels['baseline_agreement']}: {current_traceback.get('baseline_agreement_count', '—')}/{current_traceback.get('baseline_total', '—')}",
        f"- Interpretation: {current_traceback.get('interpretation', 'No traceback interpretation available.')}",
        "",
        f"## {labels['review']}",
    ])
    lines.extend([f"- {item}" for item in considerations])
    lines.extend(["", f"## {labels['limits']}"])
    lines.extend([f"- {item}" for item in limitations])
    lines.extend([
        "",
        "---",
        labels["footer"],
    ])
    return "\n".join(lines)


def _clean_mapping(obj: dict[str, Any]) -> dict[str, Any]:
    return {k: clean_number(v) for k, v in obj.items()}


def _deep_clean(obj: Any) -> Any:
    if isinstance(obj, dict):
        return {k: _deep_clean(v) for k, v in obj.items()}
    if isinstance(obj, list):
        return [_deep_clean(v) for v in obj]
    return clean_number(obj)


def _build_source_report(
    source: str,
    data_mode: str | None,
    cache_hit: bool,
    frame: pd.DataFrame,
    requested_start: str | None = None,
    requested_end: str | None = None,
) -> dict[str, Any]:
    dates = pd.to_datetime(frame.get("date"), errors="coerce") if "date" in frame else pd.Series(dtype="datetime64[ns]")
    annualization_profile = infer_annualization_profile(dates)
    return {
        "mode": data_mode or "custom",
        "source": source,
        "is_real_data": source in {"yfinance", "cache:yfinance"},
        "provider": "yfinance" if source in {"yfinance", "cache:yfinance"} else source,
        "cache_hit": bool(cache_hit),
        "requested_start": requested_start,
        "requested_end": requested_end,
        "actual_start": dates.min().strftime("%Y-%m-%d") if len(dates.dropna()) else None,
        "actual_end": dates.max().strftime("%Y-%m-%d") if len(dates.dropna()) else None,
        "observations": int(len(frame)),
        "annualization_factor": annualization_profile.get("periods_per_year"),
        "annualization_calendar": annualization_profile.get("calendar_type"),
        "annualization_method": annualization_profile.get("method"),
        "policy": (
            "real data required" if data_mode == "real" else
            "real data first, sample fallback allowed" if data_mode == "auto" else
            "deterministic offline sample" if data_mode == "sample" else
            "user uploaded data"
        ),
    }


def _build_response(asset: str, source: str, frame: pd.DataFrame, warnings: list[str], n_regimes: int, interval: str | None, cache_hit: bool = False, language: str = "en", data_mode: str | None = None, source_report: dict[str, Any] | None = None) -> AnalyzeResponse:
    data_quality = data_quality_report(frame, source, cache_hit=cache_hit)
    warnings.extend(data_quality.get("notes", []))

    features, feature_cols, feature_warnings = engineer_features(frame)
    warnings.extend(feature_warnings)
    annualization_profile = dict(features.attrs.get("annualization_profile", {}))
    annualization_factor = float(features.attrs.get("annualization_factor", annualization_profile.get("periods_per_year", 252.0)))

    model_evaluation = temporal_hmm_evaluation(features, feature_cols, selected_regimes=n_regimes)
    if model_evaluation.get("status") in {"unavailable", "failed", "insufficient_window"}:
        warnings.append(str(model_evaluation.get("message", "Temporal HMM evaluation was not available.")))
    elif not model_evaluation.get("selected_matches_recommendation", True):
        warnings.append(
            f"Model-selection note: BIC recommends {model_evaluation.get('recommended_n_regimes')} regimes, "
            f"while this run uses {n_regimes} for the selected analysis view. Review the validation tab before interpreting labels."
        )
    assignment_stability = model_evaluation.get("assignment_stability") or {}
    if assignment_stability.get("status") not in {None, "ok"}:
        warnings.append(str(assignment_stability.get("message", "Multi-seed HMM stability review was not available.")))

    result = run_regime_model(features, feature_cols, n_regimes=n_regimes, annualization_factor=annualization_factor)
    warnings.extend(result.warnings)
    out = result.frame

    risk_metrics = {
        "latest_close": clean_number(out["close"].iloc[-1]),
        "annualized_volatility": clean_number(annualized_volatility(out["log_return"], annualization_factor)),
        "annualization_factor": clean_number(annualization_factor),
        "annualization_calendar": annualization_profile.get("calendar_type"),
        "annualization_method": annualization_profile.get("method"),
        "max_drawdown": clean_number(max_drawdown(out["close"])),
        "latest_drawdown": clean_number(out["drawdown"].iloc[-1]),
        "latest_momentum_20": clean_number(out["momentum_20"].iloc[-1]),
        "latest_rsi": clean_number(out["rsi"].iloc[-1]),
        "observations": int(len(out)),
    }

    segments = regime_segments(out)
    baseline = baseline_volatility_regimes(out, n_regimes, annualization_factor=annualization_factor)
    stability = transition_stability(out, n_regimes)
    model_card = build_model_card(result.diagnostics["model_type"], n_regimes, feature_cols)
    result.current_regime = _augment_current_regime(result.current_regime, baseline, model_evaluation, data_quality, result.diagnostics)
    traceback = build_regime_traceback(out, result.transition_matrix, result.transition_labels)

    memo = generate_memo_with_graph(
        {
            "asset": asset,
            "source": source,
            "warnings": warnings,
            "current_regime": result.current_regime,
            "regime_stats": result.regime_stats,
            "risk_metrics": risk_metrics,
            "transition_labels": result.transition_labels,
            "diagnostics": result.diagnostics,
            "baseline": baseline,
            "stability": stability,
            "data_quality": data_quality,
            "model_evaluation": model_evaluation,
            "regime_segments": segments,
            "traceback": traceback,
            "language": language,
        }
    )

    report_markdown = _build_report_markdown(asset, source, interval, result, risk_metrics, memo, baseline, stability, data_quality, model_evaluation=model_evaluation, traceback=traceback, language=language)
    source_report = source_report or _build_source_report(source, data_mode, cache_hit, frame)

    out_for_chart = _chart_frame(out)
    time_series = []
    for _, row in out_for_chart.iterrows():
        time_series.append(
            {
                "date": row["date"].strftime("%Y-%m-%d"),
                "close": clean_number(row["close"]),
                "log_return": clean_number(row.get("log_return")),
                "rolling_volatility": clean_number(row.get("rolling_volatility")),
                "drawdown": clean_number(row.get("drawdown")),
                "momentum_20": clean_number(row.get("momentum_20")),
                "rsi": clean_number(row.get("rsi")),
                "regime": int(row["regime"]),
                "regime_label": str(row["regime_label"]),
                "regime_probability": clean_number(row.get("regime_probability")),
                "posterior_entropy": clean_number(row.get("posterior_entropy")),
                "state_probability_0": clean_number(row.get("state_probability_0")),
                "state_probability_1": clean_number(row.get("state_probability_1")),
                "state_probability_2": clean_number(row.get("state_probability_2")),
                "state_probability_3": clean_number(row.get("state_probability_3")),
                "state_probability_4": clean_number(row.get("state_probability_4")),
            }
        )

    return AnalyzeResponse(
        asset=asset,
        source=source,
        interval=interval,
        start=result.diagnostics["data_start"],
        end=result.diagnostics["data_end"],
        api_version=API_VERSION,
        warnings=list(dict.fromkeys([w for w in warnings if w])),
        data_quality=_deep_clean(data_quality),
        source_report=_deep_clean(source_report),
        current_regime=_clean_mapping(result.current_regime),
        risk_metrics=risk_metrics,
        regime_stats=[_clean_mapping(stat) for stat in result.regime_stats],
        transition_matrix=[[clean_number(v) for v in row] for row in result.transition_matrix.tolist()],
        transition_labels=result.transition_labels,
        time_series=time_series,
        regime_segments=[_clean_mapping(seg) for seg in segments],
        baseline=_deep_clean(baseline),
        stability=_deep_clean(stability),
        model_card=model_card,
        diagnostics=_deep_clean(result.diagnostics),
        model_evaluation=_deep_clean(model_evaluation),
        traceback=_deep_clean(traceback),
        current_traceback=_deep_clean(traceback.get("current")),
        traceback_points=_deep_clean(traceback.get("points", [])),
        memo=memo,
        report_markdown=report_markdown,
    )


@app.post("/analyze", response_model=AnalyzeResponse)
def analyze(request: AnalyzeRequest) -> AnalyzeResponse:
    warnings: list[str] = []
    resolved_start, resolved_end, normalized_interval = resolve_window(request.start, request.end, request.interval)
    try:
        loaded = load_market_data(
            asset=request.asset,
            start=resolved_start,
            end=resolved_end,
            interval=normalized_interval,
            data_mode=request.data_mode,
            prefer_live_data=request.prefer_live_data,
            force_refresh=request.force_refresh,
        )
    except MarketDataError as exc:
        raise HTTPException(
            status_code=502,
            detail={
                "message": "Real market data could not be loaded. Switch data_mode to 'auto' for explicit sample fallback, or 'sample' for offline analysis.",
                "reason": str(exc),
            },
        ) from exc
    if loaded.warning:
        warnings.append(loaded.warning)
    if loaded.source == "sample":
        warnings.append("Using deterministic sample data; this run is not real-market-data backed.")
    if loaded.cache_hit:
        warnings.append("Loaded real market data from local cache.")
    source_report = _build_source_report(
        loaded.source,
        request.data_mode if request.prefer_live_data is None else ("auto" if request.prefer_live_data else "sample"),
        loaded.cache_hit,
        loaded.frame,
        str(loaded.requested_start or resolved_start),
        str(loaded.requested_end or resolved_end),
    )
    return _build_response(request.asset.upper(), loaded.source, loaded.frame, warnings, request.n_regimes, normalized_interval, cache_hit=loaded.cache_hit, language=request.language, data_mode=source_report["mode"], source_report=source_report)


@app.post("/compare", response_model=CompareResponse)
def compare_assets(request: CompareRequest) -> CompareResponse:
    """Cross-asset regime comparison for technical risk review.

    This endpoint intentionally summarizes only review-relevant outputs. It is not a portfolio
    optimizer and does not issue buy/sell instructions.
    """
    if not request.assets:
        raise HTTPException(status_code=400, detail="At least one asset is required.")
    clean_assets = []
    for asset in request.assets:
        normalized = asset.strip().upper()
        if normalized and normalized not in clean_assets:
            clean_assets.append(normalized)
    if len(clean_assets) > 8:
        raise HTTPException(status_code=400, detail="Comparison is limited to 8 assets for interactive responsiveness.")

    resolved_start, resolved_end, normalized_interval = resolve_window(request.start, request.end, request.interval)
    summaries: list[dict[str, Any]] = []
    errors: list[str] = []

    for asset in clean_assets:
        try:
            loaded = load_market_data(
                asset=asset,
                start=resolved_start,
                end=resolved_end,
                interval=normalized_interval,
                data_mode=request.data_mode,
                prefer_live_data=request.prefer_live_data,
                force_refresh=request.force_refresh,
            )
            warnings: list[str] = []
            if loaded.warning:
                warnings.append(loaded.warning)
            if loaded.source == "sample":
                warnings.append("Using deterministic sample data; this run is not real-market-data backed.")
            if loaded.cache_hit:
                warnings.append("Loaded real market data from local cache.")
            response = _build_response(asset, loaded.source, loaded.frame, warnings, request.n_regimes, normalized_interval, cache_hit=loaded.cache_hit, language=request.language, data_mode=request.data_mode if request.prefer_live_data is None else ("auto" if request.prefer_live_data else "sample"))
            score = float(response.current_regime.get("risk_score", 0.0))
            summaries.append({
                "asset": asset,
                "source": response.source,
                "current_regime": str(response.current_regime.get("label", "—")),
                "confidence": float(response.current_regime.get("confidence", 0.0)),
                "assignment_type": str(response.current_regime.get("assignment_type", "—")),
                "evidence_strength": float(response.current_regime.get("evidence_strength", 0.0)),
                "risk_score": score,
                "risk_band": _risk_band(score),
                "stress_transition_probability": float(response.current_regime.get("stress_transition_probability", 0.0)),
                "stay_probability": float(response.current_regime.get("stay_probability", 0.0)),
                "latest_drawdown": float(response.risk_metrics.get("latest_drawdown", 0.0)),
                "annualized_volatility": float(response.risk_metrics.get("annualized_volatility", 0.0)),
                "annualization_factor": float(response.risk_metrics.get("annualization_factor", 252.0)),
                "annualization_calendar": str(response.risk_metrics.get("annualization_calendar", "inferred_calendar")),
                "baseline_agreement": float(response.baseline.get("stress_agreement", 0.0)),
                "baseline_verdict": str(response.baseline.get("verdict", "—")),
                "data_quality_status": str(response.data_quality.get("status", "—")),
                "warnings": response.warnings[:4],
            })
        except Exception as exc:
            errors.append(f"{asset}: {exc}")

    if not summaries:
        raise HTTPException(status_code=400, detail={"message": "No assets could be analyzed.", "errors": errors})

    summaries_sorted = sorted(summaries, key=lambda row: row["risk_score"], reverse=True)
    average_risk = float(np.mean([row["risk_score"] for row in summaries_sorted]))
    highest = summaries_sorted[0]["asset"] if summaries_sorted else None
    lowest = summaries_sorted[-1]["asset"] if summaries_sorted else None
    if request.language == "es":
        memo = {
            "summary": f"{len(summaries_sorted)} activos analizados. Mayor prioridad actual de revisión: {highest}. Menor score de riesgo actual: {lowest}.",
            "highest_risk_asset": highest,
            "lowest_risk_asset": lowest,
            "average_risk_score": average_risk,
            "review_notes": [
                "Usa la comparación como capa de priorización antes de abrir informes individuales por activo.",
                "Las diferencias grandes en score de riesgo deben contrastarse con calidad de datos y acuerdo con baseline.",
                "Esto es revisión de riesgo multi-activo, no optimización de portfolio ni recomendación de trading.",
            ],
            "errors": errors,
        }
    else:
        memo = {
            "summary": f"{len(summaries_sorted)} assets analyzed. Highest current review priority: {highest}. Lowest current risk score: {lowest}.",
            "highest_risk_asset": highest,
            "lowest_risk_asset": lowest,
            "average_risk_score": average_risk,
            "review_notes": [
                "Use the comparison as a prioritization layer before opening individual asset reports.",
                "Large differences in risk score should be checked against data quality and baseline agreement.",
                "This is cross-asset risk review, not a portfolio optimization or trading recommendation.",
            ],
            "errors": errors,
        }

    return CompareResponse(
        api_version=API_VERSION,
        interval=normalized_interval,
        assets_requested=clean_assets,
        summaries=summaries_sorted,
        highest_risk_asset=highest,
        lowest_risk_asset=lowest,
        average_risk_score=average_risk,
        portfolio_memo=memo,
    )


@app.post("/upload-csv", response_model=AnalyzeResponse)
async def upload_csv(file: UploadFile = File(...), n_regimes: int = 3, language: str = "en") -> AnalyzeResponse:
    if not file.filename.lower().endswith(".csv"):
        raise HTTPException(status_code=400, detail="Only CSV files are supported.")
    try:
        contents = await file.read()
        frame = parse_uploaded_csv(contents)
    except Exception as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc
    return _build_response("CUSTOM_CSV", "uploaded_csv", frame, [], n_regimes, "custom", language=language if language in {"en", "es"} else "en", data_mode="uploaded_csv")
