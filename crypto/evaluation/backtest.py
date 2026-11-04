"""Leak-safe historical simulation and model evaluation engine.

Evaluates trading performance on held-out test predictions with:
- Transaction fee accounting (0.1% per trade) + execution slippage (0.05%)
- Volatility-adaptive position sizing
- Buy & Hold benchmark comparison
- Quantitative comparative model table: Naive vs GARCH vs LSTM vs Transformer vs Ensemble
- Exports evaluation report JSON and equity curve CSV to data/processed/
"""
from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from config import (
    INITIAL_CAPITAL,
    PROCESSED_DIR,
    SLIPPAGE,
    TRANSACTION_COST,
)
from evaluation.metrics import (
    correlation_score,
    directional_accuracy,
    mae,
    max_drawdown,
    profit_factor,
    rmse,
    sharpe_ratio,
    sortino_ratio,
    win_rate,
)


class Backtester:
    """Vectorized walk-forward trading simulator with fees and slippage."""

    def __init__(
        self,
        df: pd.DataFrame,
        initial_capital: float = INITIAL_CAPITAL,
        transaction_cost: float = TRANSACTION_COST,
        slippage: float = SLIPPAGE,
    ):
        self.df = df.reset_index(drop=True).copy()
        self.initial_capital = float(initial_capital)
        self.cost = float(transaction_cost + slippage)

    def run(
        self,
        predicted_direction: np.ndarray,
        predicted_volatility: np.ndarray,
    ) -> tuple[dict[str, Any], np.ndarray, np.ndarray]:
        """
        Simulate strategy returns on held-out test period.
        Signal at time t is applied to market return from t to t+1.
        """
        close = self.df["close"].to_numpy(dtype=float)
        pred_dir = np.asarray(predicted_direction, dtype=float)
        pred_vol = np.asarray(predicted_volatility, dtype=float)

        n = min(len(close) - 1, len(pred_dir), len(pred_vol))
        close = close[: n + 1]
        pred_dir = pred_dir[:n]
        pred_vol = pred_vol[:n]

        # Calculate volatility percentile rank over recent history
        vol_series = pd.Series(pred_vol)
        vol_rank = (
            vol_series.rolling(window=min(len(vol_series), 72), min_periods=5)
            .rank(pct=True)
            .fillna(0.5)
            .to_numpy()
        )

        # Position Sizing Rules:
        # High confidence upward direction + acceptable volatility -> Long 100%
        # Moderate upward direction -> Long 50%
        # Downward direction OR volatility spike (>= 90th percentile) -> Cash / De-risk (0%)
        position = np.where(
            (pred_dir >= 0.55) & (vol_rank < 0.80),
            1.0,
            np.where(
                (pred_dir >= 0.50) & (vol_rank < 0.70),
                0.5,
                0.0,
            ),
        )

        # Asset returns from t to t+1
        market_returns = (close[1:] / close[:-1]) - 1.0

        # Position turnover changes for transaction fees
        position_changes = np.abs(np.diff(np.r_[0.0, position]))
        fee_drag = position_changes * self.cost

        # Net strategy return
        strategy_returns = (position * market_returns) - fee_drag
        equity_curve = self.initial_capital * np.cumprod(1.0 + strategy_returns)

        # Trade returns for win rate and profit factor
        trade_indices = np.where(position_changes > 0)[0]
        trade_returns = strategy_returns[trade_indices] if len(trade_indices) > 0 else strategy_returns

        buy_and_hold_returns = market_returns
        buy_and_hold_equity = self.initial_capital * np.cumprod(1.0 + buy_and_hold_returns)

        results = {
            "initial_capital": self.initial_capital,
            "final_equity": float(equity_curve[-1]),
            "total_net_return_pct": float((equity_curve[-1] / self.initial_capital - 1.0) * 100.0),
            "buy_and_hold_return_pct": float((close[-1] / close[0] - 1.0) * 100.0),
            "sharpe_ratio": sharpe_ratio(strategy_returns),
            "sortino_ratio": sortino_ratio(strategy_returns),
            "max_drawdown_pct": float(max_drawdown(equity_curve) * 100.0),
            "buy_and_hold_max_drawdown_pct": float(max_drawdown(buy_and_hold_equity) * 100.0),
            "total_trades": int(np.count_nonzero(position_changes > 0)),
            "win_rate_pct": win_rate(trade_returns),
            "profit_factor": profit_factor(trade_returns),
            "transaction_cost_pct": float(self.cost * 100.0),
        }

        return results, equity_curve, position


def generate_comparative_evaluation_report(
    test_df: pd.DataFrame,
    yv_test: np.ndarray,
    yd_test: np.ndarray,
    naive_vol: np.ndarray,
    garch_vol: np.ndarray,
    lstm_vol: np.ndarray,
    lstm_dir: np.ndarray,
    transformer_vol: np.ndarray,
    transformer_dir: np.ndarray,
) -> dict[str, Any]:
    """
    Produce academic comparative evaluation table across all models:
    Naive Volatility vs GARCH(1,1) vs LSTM vs Transformer vs Ensemble.
    """
    # 1h forecast horizon is index 0
    yv_true = yv_test[:, 0]
    yd_true = yd_test[:, 0]

    # Model predictions for 1h horizon
    p_naive = naive_vol[:, 0]
    p_garch = garch_vol[:, 0]
    p_lstm_v = lstm_vol[:, 0]
    p_trans_v = transformer_vol[:, 0]

    p_lstm_d = lstm_dir[:, 0]
    p_trans_d = transformer_dir[:, 0]

    # Weighted Ensemble
    p_ens_v = 0.35 * p_garch + 0.35 * p_lstm_v + 0.30 * p_trans_v
    p_ens_d = 0.50 * p_lstm_d + 0.50 * p_trans_d

    bt = Backtester(test_df)
    bt_naive, eq_naive, _ = bt.run(np.full(len(yv_true), 0.55), p_naive)
    bt_garch, eq_garch, _ = bt.run(np.full(len(yv_true), 0.55), p_garch)
    bt_lstm, eq_lstm, _ = bt.run(p_lstm_d, p_lstm_v)
    bt_trans, eq_trans, _ = bt.run(p_trans_d, p_trans_v)
    bt_ens, eq_ens, pos_ens = bt.run(p_ens_d, p_ens_v)

    comparison_table = [
        {
            "Model": "Naive Baseline (Rolling Std)",
            "RMSE (1h)": round(rmse(yv_true, p_naive), 5),
            "MAE (1h)": round(mae(yv_true, p_naive), 5),
            "Correlation": round(correlation_score(yv_true, p_naive), 4),
            "Directional Accuracy": "N/A (Vol only)",
            "Sharpe Ratio": round(bt_naive["sharpe_ratio"], 2),
            "Max Drawdown (%)": round(bt_naive["max_drawdown_pct"], 2),
            "Net Return (%)": round(bt_naive["total_net_return_pct"], 2),
        },
        {
            "Model": "GARCH(1,1) Econometric",
            "RMSE (1h)": round(rmse(yv_true, p_garch), 5),
            "MAE (1h)": round(mae(yv_true, p_garch), 5),
            "Correlation": round(correlation_score(yv_true, p_garch), 4),
            "Directional Accuracy": "N/A (Vol only)",
            "Sharpe Ratio": round(bt_garch["sharpe_ratio"], 2),
            "Max Drawdown (%)": round(bt_garch["max_drawdown_pct"], 2),
            "Net Return (%)": round(bt_garch["total_net_return_pct"], 2),
        },
        {
            "Model": "LSTM Multi-Task Sequential",
            "RMSE (1h)": round(rmse(yv_true, p_lstm_v), 5),
            "MAE (1h)": round(mae(yv_true, p_lstm_v), 5),
            "Correlation": round(correlation_score(yv_true, p_lstm_v), 4),
            "Directional Accuracy": f"{round(directional_accuracy(yd_true, p_lstm_d) * 100, 1)}%",
            "Sharpe Ratio": round(bt_lstm["sharpe_ratio"], 2),
            "Max Drawdown (%)": round(bt_lstm["max_drawdown_pct"], 2),
            "Net Return (%)": round(bt_lstm["total_net_return_pct"], 2),
        },
        {
            "Model": "Transformer Multi-Head Attention",
            "RMSE (1h)": round(rmse(yv_true, p_trans_v), 5),
            "MAE (1h)": round(mae(yv_true, p_trans_v), 5),
            "Correlation": round(correlation_score(yv_true, p_trans_v), 4),
            "Directional Accuracy": f"{round(directional_accuracy(yd_true, p_trans_d) * 100, 1)}%",
            "Sharpe Ratio": round(bt_trans["sharpe_ratio"], 2),
            "Max Drawdown (%)": round(bt_trans["max_drawdown_pct"], 2),
            "Net Return (%)": round(bt_trans["total_net_return_pct"], 2),
        },
        {
            "Model": "Ensemble (GARCH + LSTM + Transformer)",
            "RMSE (1h)": round(rmse(yv_true, p_ens_v), 5),
            "MAE (1h)": round(mae(yv_true, p_ens_v), 5),
            "Correlation": round(correlation_score(yv_true, p_ens_v), 4),
            "Directional Accuracy": f"{round(directional_accuracy(yd_true, p_ens_d) * 100, 1)}%",
            "Sharpe Ratio": round(bt_ens["sharpe_ratio"], 2),
            "Max Drawdown (%)": round(bt_ens["max_drawdown_pct"], 2),
            "Net Return (%)": round(bt_ens["total_net_return_pct"], 2),
        },
    ]

    report = {
        "summary": "Held-out test set comparative evaluation across econometric and deep sequential models",
        "comparison_table": comparison_table,
        "ensemble_backtest": bt_ens,
    }

    # Save to data/processed/
    (PROCESSED_DIR / "evaluation_report.json").write_text(
        json.dumps(report, indent=2), encoding="utf-8"
    )

    # Save backtest equity curve
    equity_df = pd.DataFrame(
        {
            "open_time": test_df["open_time"].iloc[: len(eq_ens)],
            "close": test_df["close"].iloc[: len(eq_ens)],
            "strategy_equity": eq_ens,
            "position": pos_ens,
        }
    )
    equity_df.to_csv(PROCESSED_DIR / "backtest_curve.csv", index=False)

    return report
