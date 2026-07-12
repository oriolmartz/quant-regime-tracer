from __future__ import annotations

import numpy as np
import pandas as pd


def transition_matrix(states: np.ndarray, n_states: int) -> np.ndarray:
    mat = np.zeros((n_states, n_states), dtype=float)
    for a, b in zip(states[:-1], states[1:]):
        mat[int(a), int(b)] += 1
    row_sums = mat.sum(axis=1, keepdims=True)
    with np.errstate(divide="ignore", invalid="ignore"):
        mat = np.divide(mat, row_sums, where=row_sums != 0)
    mat[np.isnan(mat)] = 0.0
    return mat


def expected_persistence_days(stay_probability: float) -> float | None:
    if stay_probability >= 0.999 or stay_probability <= 0:
        return None
    return 1.0 / (1.0 - stay_probability)


def max_drawdown(close: pd.Series) -> float:
    running_max = close.cummax()
    drawdown = close / running_max - 1
    return float(drawdown.min())


def annualized_volatility(log_returns: pd.Series, periods_per_year: float = 252.0) -> float:
    return float(log_returns.std() * np.sqrt(float(periods_per_year)))
