# Git-ready commands

Recommended commit:

```bash
git add .
git commit -m "feat: add auditable regime traceback workflow"
```

Validation commands before pushing:

```bash
cd backend
pytest -q
python scripts/smoke_test.py

cd ../frontend
npm ci
npm run typecheck
npm run smoke
npm run build
npm audit --omit=dev
```

Suggested GitHub repo description:

> Full-stack quantitative workbench for auditable market-regime inference with HMMs, Markov transitions, Regime Traceback, temporal validation diagnostics and real market data.
