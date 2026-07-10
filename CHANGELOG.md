
## Documentation and attribution polish

- Added discreet UI attribution: `QuantRegimeTracer · Built by Oriol Martínez`.
- Added an author note to the Case Study panel.
- Rebalanced README diagrams: compact data-ingestion flow and clearer system-structure architecture.
- Re-rendered core formulas using GitHub Markdown math.
- Added a concrete screenshot capture plan with exact file names and README placements.

# Changelog

## Real-data evidence documentation pass

- Added generated real-data validation artifacts under `reports/`.
- Documented the SPY/QQQ/BTC-USD/GLD/TLT validation bundle in the README.
- Added explicit interpretation of the systematic `k=3` selected versus `k=5` BIC-recommended mismatch.
- Documented GLD as an overfit-risk case where the validation layer correctly warns against over-interpreting the regime path.
- Documented QQQ as a moderate-stability case where multi-seed HMM assignments should be interpreted cautiously.
- Updated the case study to frame validation disagreement as a core product feature rather than a defect to hide.

## QuantRegimeTracer traceback release

- Renamed the project identity to QuantRegimeTracer.
- Added Regime Traceback backend service for point-level evidence reconstruction.
- Added `current_traceback`, `traceback` and `traceback_points` to analysis responses.
- Added Traceback UI tab with feature evidence, posterior state mass, posterior entropy, Markov transition prior and baseline votes.
- Updated README/API docs around auditable regime inference rather than generic dashboard positioning.
- Added backend tests for traceback evidence and transition-prior consistency.

## Real-data-first release

- Changed default data policy to `data_mode='real'`.
- Added explicit data modes: `real`, `auto`, `sample`.
- Added `/data-sources` endpoint documenting data-source behavior.
- Added `source_report` to analysis responses.
- Added real-only failure behavior: HTTP 502 when cache/yfinance cannot produce enough observations.
- Kept deterministic sample mode for tests and offline reproducibility.
- Updated frontend controls with data-mode selector and force-refresh option.
- Updated README and API docs to emphasize real market data, model diagnostics and technical risk review.

## Prior validation release

- Added temporal HMM diagnostics with chronological holdout, AIC/BIC and held-out log-likelihood.
- Added multi-seed HMM assignment stability review with Adjusted Rand Index.
- Added baseline suite: rolling volatility, EWMA volatility and drawdown stress.
- Added posterior entropy and richer posterior state mass visualization.

## Traceability UI correctness pass

- Replaced visible `100%` posterior badge in the latest-assignment panel with assignment-shape language and MAP-state-mass detail.
- Clarified that posterior state mass is not forecast confidence or market certainty.
- Aligned the custom start date control with the rest of the analysis controls.
- Added `docs/VALIDATION_PROTOCOL.md` to document how to validate model correctness, data provenance, traceback behavior and product behavior before publishing or review.

## Documentation polish

- Fixed case-sensitive report references to `reports/real_data_validation.md`.
- Updated the README repository tree to match the actual backend package layout.
- Clarified that HMM fitting uses `hmmlearn.GaussianHMM`; the project contribution is the feature, validation, traceback, reporting and UI layer around it.
