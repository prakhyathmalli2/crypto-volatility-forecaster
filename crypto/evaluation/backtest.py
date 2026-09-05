"""Leak-safe vectorized paper backtest driven by direction + volatility forecasts."""
from __future__ import annotations

import numpy as np
import pandas as pd

from evaluation.metrics import max_drawdown, sharpe_ratio, sortino_ratio


class Backtester:
    def __init__(self, df: pd.DataFrame, initial_capital=10000.0, transaction_cost=0.001):
        self.df = df.reset_index(drop=True).copy()
        self.initial_capital = float(initial_capital)
        self.cost = float(transaction_cost)

    def run(self, predicted_direction, predicted_volatility):
        pred_dir = np.asarray(predicted_direction, dtype=float)
        pred_vol = np.asarray(predicted_volatility, dtype=float)
        close = self.df["close"].to_numpy(dtype=float)
        n = min(len(close) - 1, len(pred_dir), len(pred_vol))
        close = close[:n + 1]
        pred_dir, pred_vol = pred_dir[:n], pred_vol[:n]

        # Signal at t is applied to return t -> t+1.
        vol_rank = pd.Series(pred_vol).rolling(168, min_periods=30).rank(pct=True).fillna(0.5).to_numpy()
        position = np.where((pred_dir >= 0.55) & (vol_rank < 0.75), 1.0,
                    np.where((pred_dir <= 0.45) | (vol_rank >= 0.90), 0.0, 0.5))
        returns = close[1:] / close[:-1] - 1
        changes = np.abs(np.diff(np.r_[0.0, position]))
        strategy_returns = position * returns - changes * self.cost
        equity = self.initial_capital * np.cumprod(1 + strategy_returns)
        results = {
            "final_equity": float(equity[-1]),
            "total_return_pct": float((equity[-1] / self.initial_capital - 1) * 100),
            "buy_and_hold_pct": float((close[-1] / close[0] - 1) * 100),
            "sharpe": sharpe_ratio(strategy_returns),
            "sortino": sortino_ratio(strategy_returns),
            "max_drawdown_pct": float(max_drawdown(equity) * 100),
            "num_trades": int(np.count_nonzero(changes > 0)),
        }
        return results, equity, position
