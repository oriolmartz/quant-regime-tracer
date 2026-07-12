# QuantRegimeTracer API Contract

QuantRegimeTracer is real-market-data first. The API default is `data_mode='real'`, which requires local cache or yfinance data. Deterministic sample fallback occurs only when callers explicitly use `data_mode='auto'` or `data_mode='sample'`.

## Metadata

| Method | Endpoint | Description |
|---|---|---|
| `GET` | `/health` | Status and API version. |
| `GET` | `/assets` | Supported ticker list. |
| `GET` | `/time-windows` | Window presets. |
| `GET` | `/data-sources` | Data-mode policy and provider details. |
| `GET` | `/model-card` | Model assumptions, outputs and guardrails. |
| `GET` | `/case-study` | Architecture and methodology summary. |
| `GET` | `/sample-csv` | Offline sample CSV for parser testing. |

## Analyze

`POST /analyze`

```json
{
  "asset": "SPY",
  "interval": "5Y",
  "n_regimes": 3,
  "data_mode": "real",
  "force_refresh": false,
  "language": "en"
}
```

### Request fields

| Field | Type | Default | Notes |
|---|---:|---:|---|
| `asset` | string | `SPY` | Ticker symbol. |
| `start` | date/null | null | Explicit start override. |
| `end` | date/null | null | Explicit end override. |
| `interval` | string | `5Y` | `6M`, `1Y`, `2Y`, `3Y`, `5Y`, `MAX`. |
| `n_regimes` | int | `3` | 2 to 5. |
| `data_mode` | string | `real` | `real`, `auto`, `sample`. |
| `force_refresh` | bool | false | Bypass cache in real/auto mode. |
| `language` | string | `en` | `en` or `es`. |
| `prefer_live_data` | bool/null | null | Deprecated compatibility flag. Use `data_mode`. |

### Data modes

| Mode | Meaning |
|---|---|
| `real` | Require real data from cache/yfinance. Returns HTTP 502 if unavailable. |
| `auto` | Try cache/yfinance, then explicit deterministic sample fallback with warning. |
| `sample` | Deterministic offline sample data only. |

### Key response fields

```json
{
  "asset": "SPY",
  "source": "yfinance",
  "source_report": {
    "mode": "real",
    "source": "yfinance",
    "is_real_data": true,
    "provider": "yfinance",
    "cache_hit": false,
    "policy": "real data required"
  },
  "current_regime": {
    "label": "Drawdown transition",
    "assignment_type": "Near one-hot",
    "assignment_strength": 0.998,
    "evidence_strength": 0.73,
    "posterior_entropy": 0.01
  },
  "time_series": [],
  "regime_stats": [],
  "transition_matrix": [],
  "baseline": {},
  "model_evaluation": {},
  "current_traceback": {
    "posterior_confidence": 0.91,
    "assignment_type": "Strong assignment",
    "evidence_strength": 0.76,
    "posterior_entropy": 0.12,
    "transition_prior": 0.18,
    "baseline_agreement_count": 2,
    "baseline_total": 3
  },
  "traceback_points": [],
  "report_markdown": "..."
}
```

### Regime Traceback fields

`current_traceback` and `traceback_points` expose the point-level inference path behind regime assignments. Each traceback point includes:

- assigned state and semantic label
- posterior state mass `γ_t`, assignment type, evidence strength and posterior entropy `H(γ_t)`
- previous-state Markov transition prior
- feature evidence with within-window percentiles
- baseline votes from rolling volatility, EWMA volatility and drawdown stress rules
- a compact interpretation of why the point maps to the selected regime

This is the project-specific auditability layer: the API does not only return a regime label; it returns why the label was assigned and how much uncertainty surrounded the assignment and how strong the evidence path is after baseline checks.

Treat a response as real-data backed only if:

```text
source_report.is_real_data == true
source in {"yfinance", "cache:yfinance"}
```

## Compare

`POST /compare`

```json
{
  "assets": ["SPY", "QQQ", "BTC-USD", "GLD", "TLT"],
  "interval": "3Y",
  "n_regimes": 3,
  "data_mode": "real"
}
```

The endpoint deduplicates assets, caps comparison at eight assets and returns sorted risk summaries. It does not optimize a portfolio or issue trading instructions.

## Upload CSV

`POST /upload-csv?n_regimes=3&language=en`

Required CSV columns:

```text
date,close
```

Optional:

```text
volume
```

At least 180 valid observations are required.

## Failure behavior

When `data_mode='real'` and neither cache nor yfinance can provide enough observations, `/analyze` returns HTTP 502:

```json
{
  "detail": {
    "message": "Real market data could not be loaded. Switch data_mode to 'auto' for explicit sample fallback, or 'sample' for offline analysis.",
    "reason": "..."
  }
}
```


## Calendar-aware annualization

The analysis response exposes `risk_metrics.annualization_factor`, `risk_metrics.annualization_calendar` and `risk_metrics.annualization_method`. These values are inferred from the actual timestamps and are used consistently across annualized features, regime statistics, baselines and exported reports.
