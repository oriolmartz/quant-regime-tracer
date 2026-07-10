# QuantRegimeTracer Validation Protocol

This project should be validated as a quantitative research workbench, not as a trading system. Correctness does not mean that the model predicts price direction. Correctness means that data provenance, feature construction, regime inference, uncertainty reporting, validation diagnostics, API contracts and UI behavior are internally consistent and reproducible.

## 1. Environment and dependency validation

Run the backend and frontend gates from a clean environment.

```bash
cd backend
python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
pytest -q
python scripts/smoke_test.py
```

```bash
cd frontend
npm ci
npm run doctor
npm run typecheck
npm run smoke
npm run build
npm audit --omit=dev
```

Expected outcome:

- all backend tests pass;
- the smoke test can call the API and produce an analysis response;
- TypeScript and production build pass;
- the frontend smoke script confirms expected source files and package metadata;
- `npm audit --omit=dev` reports no runtime vulnerabilities.

## 2. Data-source validation

Validate that real-data mode is actually real-data backed.

```bash
python -c "import yfinance as yf; print(yf.download('SPY', period='1y', progress=False, auto_adjust=True, threads=False).tail())"
```

Then run the API/UI in `Real only` mode and inspect `source_report`.

A run is real-data backed only if:

```text
source_report.is_real_data == true
source_report.source in {"yfinance", "cache:yfinance"}
```

A `502` in `Real only` mode is acceptable when the provider fails. The system should not silently fall back to synthetic data in that mode. Use `Real + fallback` only for degraded local runs and demos where provider failure should not block UI inspection.

## 3. Feature-engineering validation

Check the feature layer independently of the model:

- no invalid dates after parsing;
- at least 180 valid observations;
- `close` is positive and finite;
- `log_return` is finite after initial lag removal;
- drawdown is non-positive and starts near zero;
- rolling volatility is non-negative;
- RSI stays within a bounded range;
- missing volume does not break the feature matrix.

Feature correctness matters because every downstream regime, posterior, transition and traceback explanation depends on these values.

## 4. Model-output validation

For each analyzed asset/window, review the validation tab and exported JSON.

Minimum checks:

- `diagnostics.model_type` should be `GaussianHMM` for a full probabilistic run;
- if `KMeans fallback` appears, the UI must label assignments as hard-cluster proxies;
- `n_observations` and `n_features` should match the analyzed window and feature pipeline;
- state labels should be assigned from statistics, not hard-coded IDs;
- transition matrix rows should sum approximately to one;
- posterior entropy should rise near ambiguous state boundaries and remain low during sharply separated states;
- MAP posterior mass near 1.0 should be described as state-assignment strength, not market certainty.

## 5. Regime-count validation

Use the model-evaluation panel and real-data validation report.

Review:

- BIC/AIC by candidate regime count;
- train/test log-likelihood per observation;
- train/test likelihood gap;
- recommended `k`;
- whether the selected `k` matches the recommended `k`.

A selected `k` different from BIC recommendation is not automatically wrong. It should be treated as a methodological choice for interpretability, and the report should make that visible.

## 6. Stability validation

Review the multi-seed stability result.

- high mean ARI: assignments are stable across HMM initializations;
- moderate ARI: use the model, but discuss uncertainty;
- low ARI: do not over-interpret regime labels; inspect baselines and traceback.

This check protects the project from presenting one random HMM fit as if it were uniquely determined.

## 7. Baseline validation

Compare the HMM stress/non-stress interpretation against transparent baselines:

- rolling-volatility quantile baseline;
- EWMA-volatility stress baseline;
- drawdown stress baseline.

Good validation is not necessarily perfect agreement. Useful outcomes include:

- high agreement: HMM is aligned with simple risk rules;
- mixed agreement: HMM may be adding structure or may be unstable;
- divergence: inspect disagreement segments and avoid overclaiming.

## 8. Traceback validation

Open the Regime Traceback tab and inspect multiple dates:

- latest observation;
- a known transition boundary;
- a stable expansion segment;
- a stress/drawdown segment.

For each selected point, verify that the explanation is consistent with:

- assigned state and semantic label;
- MAP posterior state mass;
- posterior entropy;
- Markov transition prior from previous state;
- volatility, drawdown, momentum, RSI and return evidence;
- baseline votes and agreement share;
- final interpretation.

Traceback correctness means that the explanation follows from the model outputs and feature values. It does not mean the explanation is causal or predictive.

## 9. Product validation

Validate the product layer separately from model correctness.

Manual UI checks:

- `Real only` fails loudly if the provider is unavailable;
- `Real + fallback` exposes sample fallback with a warning;
- `Sample only` is deterministic and useful offline;
- CSV upload rejects invalid schemas and short files;
- custom start date changes the actual window;
- changing `n_regimes` updates model-selection diagnostics;
- exported Markdown and JSON include source, model, validation and traceback fields;
- language toggle does not break key technical labels.

## 10. What would make it incorrect

The project should be considered incorrect or misleading if any of the following happen:

- synthetic sample data is presented as real market data;
- KMeans fallback is described as HMM posterior inference;
- `100%` posterior state mass is described as market prediction confidence;
- transition probabilities are described as guaranteed forecasts;
- baseline agreement is treated as proof of trading edge;
- UI labels hide provider failures or model instability;
- exported reports omit limitations and data source information.

## 11. Evidence to keep before publishing

Before sharing the project, keep a small validation bundle:

```text
pytest output
frontend build output
one real-data JSON export for SPY or QQQ
one real-data validation report across several assets
one screenshot of Dashboard
one screenshot of Regime Traceback
one screenshot of Validation
```

This makes the project reviewable without asking a recruiter or interviewer to clone and run the full stack immediately.


## 12. Committed validation evidence

The repository includes a generated real-data validation bundle:

```text
reports/real_data_validation.md
reports/real_data_validation.md
reports/real_data_validation.json
```

This bundle is not a claim of trading performance. It is evidence that the real-data ingestion, HMM inference, model-selection diagnostics, baseline comparison and stability review execute on actual market data.

The bundle should be reviewed with the same guardrails as the product UI: selected-vs-recommended `k` mismatches, overfit-risk verdicts and moderate stability are first-class review signals, not errors to hide.
