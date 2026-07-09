# QuantRegimeTracer — technical case study

QuantRegimeTracer is a real-market-data-first regime inference workbench. It converts market price histories into inferred latent regimes, Markov transition diagnostics, validation evidence, point-level traceback explanations and guarded review memos.

## Problem

A raw price chart does not expose regime uncertainty, transition risk, model stability, data-source quality or the evidence path behind a regime label. QuantRegimeTracer builds a review layer that makes those assumptions inspectable.

## Architecture

- FastAPI analysis backend.
- Real data loader with yfinance, local cache, explicit sample fallback mode and CSV uploads.
- Feature engineering for returns, volatility, drawdown, trend and momentum.
- Gaussian HMM regime engine with explicit KMeans fallback if HMM fitting is unavailable.
- Post-fit semantic labeling to avoid hard-coded HMM state IDs.
- Empirical Markov transition matrix and persistence metrics.
- Validation layer with baseline suite, temporal holdout, BIC/AIC and multi-seed ARI stability.
- Regime Traceback layer that reconstructs feature evidence, posterior uncertainty, transition prior and baseline agreement for selected dates.
- React/TypeScript UI for source-aware analysis, traceback inspection and export.

## Guardrails

- No buy/sell instructions.
- No autonomous execution.
- Source report identifies whether the run is real-data backed.
- Sample fallback is explicit and visible.
- Model diagnostics are separated from trading validation.


## Real-data validation evidence

A validation bundle was generated with `data_mode=real` across SPY, QQQ, BTC-USD, GLD and TLT. The run used yfinance-backed market data, `GaussianHMM` inference, baseline-suite comparison, chronological train/test diagnostics and multi-seed stability review.

The most important result is not that every asset looked clean. The useful result is that the validation layer surfaced review points:

- all five assets were real-backed and successfully analyzed;
- the default `k=3` interpretability setting differed from the BIC recommendation of `k=5` across the evaluated assets;
- QQQ showed only moderate assignment stability across HMM seeds;
- GLD was flagged as an overfit-risk case in temporal holdout diagnostics.

This makes the project more credible: the system is allowed to say "review this" instead of forcing every output into a confident regime story.

## Case study: when validation detects model risk

The GLD run is a useful example of why the validation layer exists.

The HMM produced a latent-state assignment, but the temporal holdout diagnostics flagged `overfit_risk`: train likelihood was much stronger than held-out likelihood, suggesting that the model fit the training window far better than the test window.

A simpler dashboard might still show a regime label and move on. QuantRegimeTracer keeps the label, but attaches the warning: the assignment may be mathematically available, yet the model's temporal generalization is weak for that asset/window.

The system therefore separates:

- state assignment strength;
- posterior concentration;
- baseline agreement;
- multi-seed stability;
- temporal generalization;
- and final interpretability.

This is why the project avoids presenting HMM output as a trading signal. The validation layer can disagree with the visual regime path, and that disagreement is part of the output.

## Limitations

- HMM regimes are latent, not ground truth.
- yfinance availability is external.
- No PnL backtest, slippage model or execution assumptions are included.
- Markov transition estimates can decay under structural change.
