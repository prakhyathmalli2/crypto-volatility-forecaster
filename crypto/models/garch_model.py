"""GARCH(1,1) volatility baseline."""
from __future__ import annotations

import joblib
import numpy as np
import pandas as pd
from arch import arch_model


def fit_garch(returns: pd.Series, p: int = 1, q: int = 1):
    clean = pd.Series(returns).dropna().astype(float) * 100
    return arch_model(clean, vol="GARCH", p=p, q=q, dist="t", mean="Zero").fit(disp="off")


def forecast_volatility(result, horizons=(1, 6, 24)) -> dict[int, float]:
    max_h = max(horizons)
    fc = result.forecast(horizon=max_h, reindex=False)
    variance = np.asarray(fc.variance.iloc[-1], dtype=float)
    return {h: float(np.sqrt(np.sum(variance[:h])) / 100) for h in horizons}


def save_result(result, path):
    joblib.dump(result, path)
