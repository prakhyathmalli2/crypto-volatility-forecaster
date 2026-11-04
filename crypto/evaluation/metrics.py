"""Quantitative metrics for volatility forecasting and trading strategies.

Calculates:
- Volatility error metrics: RMSE, MAE, MSE, MAPE, Pearson Correlation
- Directional classification metrics: Directional Accuracy, Brier Score
- Portfolio performance metrics: Sharpe Ratio, Sortino Ratio, Maximum Drawdown,
  Win Rate, Profit Factor, and Net Return.
"""
from __future__ import annotations

import numpy as np


def rmse(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    """Root Mean Squared Error."""
    yt, yp = np.asarray(y_true, dtype=float), np.asarray(y_pred, dtype=float)
    return float(np.sqrt(np.mean((yt - yp) ** 2)))


def mae(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    """Mean Absolute Error."""
    yt, yp = np.asarray(y_true, dtype=float), np.asarray(y_pred, dtype=float)
    return float(np.mean(np.abs(yt - yp)))


def mse(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    """Mean Squared Error."""
    yt, yp = np.asarray(y_true, dtype=float), np.asarray(y_pred, dtype=float)
    return float(np.mean((yt - yp) ** 2))


def mape(y_true: np.ndarray, y_pred: np.ndarray, eps: float = 1e-6) -> float:
    """Mean Absolute Percentage Error (%)."""
    yt, yp = np.asarray(y_true, dtype=float), np.asarray(y_pred, dtype=float)
    return float(np.mean(np.abs((yt - yp) / np.maximum(np.abs(yt), eps))) * 100.0)


def correlation_score(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    """Pearson correlation coefficient between true and predicted values."""
    yt, yp = np.asarray(y_true, dtype=float), np.asarray(y_pred, dtype=float)
    if np.std(yt) == 0 or np.std(yp) == 0:
        return 0.0
    corr = np.corrcoef(yt, yp)[0, 1]
    return float(corr) if not np.isnan(corr) else 0.0


def directional_accuracy(y_true: np.ndarray, y_pred_prob: np.ndarray) -> float:
    """Directional hit rate comparing predicted upward probability (>0.5) to actual outcome (>0)."""
    yt = np.asarray(y_true, dtype=float)
    yp = np.asarray(y_pred_prob, dtype=float)
    actual_up = yt > 0.0
    pred_up = yp >= 0.5
    return float(np.mean(actual_up == pred_up))


def sharpe_ratio(
    returns: np.ndarray, periods_per_year: int = 8760, risk_free_rate: float = 0.0
) -> float:
    """Annualized Sharpe ratio for hourly (8760) return stream."""
    r = np.asarray(returns, dtype=float)
    if len(r) == 0 or np.std(r, ddof=1) <= 1e-9:
        return 0.0
    excess_mean = float(np.mean(r) - (risk_free_rate / periods_per_year))
    return float(np.sqrt(periods_per_year) * (excess_mean / np.std(r, ddof=1)))


def sortino_ratio(
    returns: np.ndarray, periods_per_year: int = 8760, risk_free_rate: float = 0.0
) -> float:
    """Annualized Sortino ratio penalizing only downside volatility."""
    r = np.asarray(returns, dtype=float)
    downside = r[r < 0.0]
    if len(r) == 0 or len(downside) == 0 or np.std(downside, ddof=1) <= 1e-9:
        return 0.0
    excess_mean = float(np.mean(r) - (risk_free_rate / periods_per_year))
    return float(np.sqrt(periods_per_year) * (excess_mean / np.std(downside, ddof=1)))


def max_drawdown(equity_curve: np.ndarray) -> float:
    """Maximum peak-to-trough equity percentage decline."""
    curve = np.asarray(equity_curve, dtype=float)
    if len(curve) == 0:
        return 0.0
    peak = np.maximum.accumulate(curve)
    drawdowns = (curve - peak) / np.maximum(peak, 1e-9)
    return float(np.min(drawdowns))


def win_rate(trade_returns: np.ndarray) -> float:
    """Percentage of completed trades with positive net returns."""
    trades = np.asarray(trade_returns, dtype=float)
    if len(trades) == 0:
        return 0.0
    return float(np.mean(trades > 0.0) * 100.0)


def profit_factor(trade_returns: np.ndarray) -> float:
    """Gross profits divided by gross losses."""
    trades = np.asarray(trade_returns, dtype=float)
    gains = trades[trades > 0.0].sum()
    losses = np.abs(trades[trades < 0.0].sum())
    if losses <= 1e-9:
        return 10.0 if gains > 0 else 1.0
    return float(gains / losses)


def precision_recall_f1(
    y_true: np.ndarray, y_pred_prob: np.ndarray, threshold: float = 0.5
) -> dict[str, float]:
    """Calculate Precision, Recall, and F1 for binary directional classification."""
    yt = (np.asarray(y_true, dtype=float) > 0.0).astype(int)
    yp = (np.asarray(y_pred_prob, dtype=float) >= threshold).astype(int)

    tp = int(np.sum((yt == 1) & (yp == 1)))
    fp = int(np.sum((yt == 0) & (yp == 1)))
    fn = int(np.sum((yt == 1) & (yp == 0)))
    tn = int(np.sum((yt == 0) & (yp == 0)))

    prec = float(tp / (tp + fp)) if (tp + fp) > 0 else 0.0
    rec = float(tp / (tp + fn)) if (tp + fn) > 0 else 0.0
    f1 = float(2 * (prec * rec) / (prec + rec)) if (prec + rec) > 0 else 0.0

    return {
        "precision": round(prec, 4),
        "recall": round(rec, 4),
        "f1": round(f1, 4),
        "true_positives": tp,
        "false_positives": fp,
        "true_negatives": tn,
        "false_negatives": fn,
    }


def brier_score(y_true: np.ndarray, y_pred_prob: np.ndarray) -> float:
    """Mean squared difference between predicted probabilities and actual outcomes."""
    yt = (np.asarray(y_true, dtype=float) > 0.0).astype(float)
    yp = np.clip(np.asarray(y_pred_prob, dtype=float), 0.0, 1.0)
    return float(np.mean((yp - yt) ** 2))


def average_trade_metrics(trade_returns: np.ndarray) -> dict[str, float]:
    """Calculate average gain per winning trade and average loss per losing trade."""
    trades = np.asarray(trade_returns, dtype=float)
    gains = trades[trades > 0.0]
    losses = trades[trades < 0.0]

    avg_gain = float(np.mean(gains) * 100.0) if len(gains) > 0 else 0.0
    avg_loss = float(np.mean(losses) * 100.0) if len(losses) > 0 else 0.0

    return {
        "avg_win_return_pct": round(avg_gain, 3),
        "avg_loss_return_pct": round(avg_loss, 3),
    }
