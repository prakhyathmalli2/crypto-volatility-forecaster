"""Forecast and trading metrics."""
from __future__ import annotations

import numpy as np


def rmse(y_true, y_pred):
    return float(np.sqrt(np.mean((np.asarray(y_true) - np.asarray(y_pred)) ** 2)))


def mae(y_true, y_pred):
    return float(np.mean(np.abs(np.asarray(y_true) - np.asarray(y_pred))))


def directional_accuracy(y_true, y_pred):
    yt, yp = np.asarray(y_true), np.asarray(y_pred)
    return float(np.mean((yt > 0) == (yp > 0)))


def sharpe_ratio(returns, periods_per_year=8760):
    r = np.asarray(returns, dtype=float)
    if len(r) == 0 or np.std(r) == 0:
        return 0.0
    return float(np.sqrt(periods_per_year) * np.mean(r) / np.std(r, ddof=1))


def sortino_ratio(returns, periods_per_year=8760):
    r = np.asarray(returns, dtype=float)
    downside = r[r < 0]
    if len(r) == 0 or len(downside) == 0 or np.std(downside, ddof=1) == 0:
        return 0.0
    return float(np.sqrt(periods_per_year) * np.mean(r) / np.std(downside, ddof=1))


def max_drawdown(equity_curve):
    curve = np.asarray(equity_curve, dtype=float)
    peak = np.maximum.accumulate(curve)
    return float(np.min((curve - peak) / peak))
