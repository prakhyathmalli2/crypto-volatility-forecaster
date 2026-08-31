"""
Quantitative Backtesting & Performance Evaluation Engine
Simulates trading strategy execution with transaction costs and volatility-adjusted position sizing.
Extracts trade logs and computes industry-standard financial metrics (Sharpe, Sortino, Max Drawdown, Win Rate).
"""

import pandas as pd
import numpy as np
import logging

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


def extract_trade_records(bt_df: pd.DataFrame) -> pd.DataFrame:
    """
    Extract individual completed trade cycles from backtested position series.
    """
    trades = []
    in_trade = False
    entry_date = None
    entry_price = 0.0
    entry_pos_size = 1.0

    for date, row in bt_df.iterrows():
        pos = row['Position']
        close_price = row['Close']
        pos_size = row.get('Position_Size', 1.0)

        # Entry condition: Was flat, now long
        if not in_trade and pos > 0:
            in_trade = True
            entry_date = date
            entry_price = close_price
            entry_pos_size = pos_size

        # Exit condition: Was long, now flat
        elif in_trade and pos == 0:
            in_trade = False
            exit_date = date
            exit_price = close_price
            raw_pnl_pct = ((exit_price - entry_price) / (entry_price + 1e-9)) * 100.0
            net_pnl_pct = raw_pnl_pct * entry_pos_size - 0.2  # 0.2% round-trip fee
            duration_days = (exit_date - entry_date).days if hasattr(exit_date - entry_date, 'days') else 1

            trades.append({
                "Entry Date": entry_date.strftime('%Y-%m-%d') if hasattr(entry_date, 'strftime') else str(entry_date),
                "Exit Date": exit_date.strftime('%Y-%m-%d') if hasattr(exit_date, 'strftime') else str(exit_date),
                "Entry Price ($)": round(entry_price, 2),
                "Exit Price ($)": round(exit_price, 2),
                "Duration (Days)": max(duration_days, 1),
                "Position Sizing": f"{round(entry_pos_size * 100)}%",
                "Net PnL (%)": round(net_pnl_pct, 2),
                "Outcome": "WIN 🟢" if net_pnl_pct > 0 else "LOSS 🔴"
            })

    if not trades:
        return pd.DataFrame(columns=[
            "Entry Date", "Exit Date", "Entry Price ($)", "Exit Price ($)",
            "Duration (Days)", "Position Sizing", "Net PnL (%)", "Outcome"
        ])

    return pd.DataFrame(trades)


def run_strategy_backtest(
    df: pd.DataFrame,
    signals_series: pd.Series,
    initial_capital: float = 10000.0,
    fee_pct: float = 0.001,
    volatility_sizing: bool = True,
    garch_vol_series: pd.Series = None
) -> dict:
    """
    Run vectorized/event backtest on market price series and model signals.
    """
    common_idx = df.index.intersection(signals_series.index)
    bt_df = df.loc[common_idx].copy()
    bt_df['Signal'] = signals_series.loc[common_idx].astype(int)

    # Shift signal by 1 day to execute at next open/close (no lookahead execution)
    bt_df['Position'] = bt_df['Signal'].shift(1).fillna(0)

    # Volatility-Adjusted Smart Position Sizing
    if volatility_sizing and (garch_vol_series is not None):
        vol_aligned = garch_vol_series.reindex(common_idx).fillna(garch_vol_series.median())
        median_vol = float(vol_aligned.median())
        size_multiplier = (median_vol / (vol_aligned + 1e-9)).clip(lower=0.4, upper=1.0)
        bt_df['Position_Size'] = bt_df['Position'] * size_multiplier
    else:
        bt_df['Position_Size'] = bt_df['Position']

    # Identify trade transitions (Buy/Sell actions)
    bt_df['Trade_Action'] = bt_df['Position'].diff().abs().fillna(0)
    bt_df['Transaction_Costs'] = bt_df['Trade_Action'] * fee_pct

    # Daily strategy returns
    daily_price_return = bt_df['Return'].fillna(0)
    bt_df['Gross_Strategy_Return'] = bt_df['Position_Size'] * daily_price_return
    bt_df['Net_Strategy_Return'] = bt_df['Gross_Strategy_Return'] - bt_df['Transaction_Costs']

    # Cumulative equity curves
    bt_df['Strategy_Equity'] = initial_capital * (1 + bt_df['Net_Strategy_Return']).cumprod()
    bt_df['Benchmark_Equity'] = initial_capital * (1 + daily_price_return).cumprod()

    # Drawdowns
    rolling_peak = bt_df['Strategy_Equity'].cummax()
    bt_df['Drawdown'] = (bt_df['Strategy_Equity'] - rolling_peak) / (rolling_peak + 1e-9)
    max_drawdown_pct = float(bt_df['Drawdown'].min() * 100.0)

    bm_peak = bt_df['Benchmark_Equity'].cummax()
    bt_df['Benchmark_Drawdown'] = (bt_df['Benchmark_Equity'] - bm_peak) / (bm_peak + 1e-9)
    bm_max_drawdown_pct = float(bt_df['Benchmark_Drawdown'].min() * 100.0)

    # Key Performance Statistics
    total_strategy_return_pct = float(((bt_df['Strategy_Equity'].iloc[-1] - initial_capital) / initial_capital) * 100.0)
    total_benchmark_return_pct = float(((bt_df['Benchmark_Equity'].iloc[-1] - initial_capital) / initial_capital) * 100.0)

    num_days = len(bt_df)
    cagr_strategy = float(((bt_df['Strategy_Equity'].iloc[-1] / initial_capital) ** (365.0 / max(num_days, 1)) - 1.0) * 100.0)
    cagr_benchmark = float(((bt_df['Benchmark_Equity'].iloc[-1] / initial_capital) ** (365.0 / max(num_days, 1)) - 1.0) * 100.0)

    # Sharpe Ratio
    rf_daily = 0.02 / 365.0
    excess_returns = bt_df['Net_Strategy_Return'] - rf_daily
    daily_std = bt_df['Net_Strategy_Return'].std()
    sharpe_ratio = float((excess_returns.mean() / (daily_std + 1e-9)) * np.sqrt(365.0))

    # Sortino Ratio
    downside_returns = bt_df['Net_Strategy_Return'][bt_df['Net_Strategy_Return'] < 0]
    downside_std = downside_returns.std()
    sortino_ratio = float((excess_returns.mean() / (downside_std + 1e-9)) * np.sqrt(365.0))

    # Trade Log extraction
    trade_log_df = extract_trade_records(bt_df)
    
    # Win rate calculation
    if not trade_log_df.empty:
        total_trades = len(trade_log_df)
        winning_trades = len(trade_log_df[trade_log_df['Net PnL (%)'] > 0])
        win_rate_pct = float((winning_trades / max(total_trades, 1)) * 100.0)
    else:
        active_days = bt_df[bt_df['Position'] > 0]
        win_rate_pct = float((len(active_days[active_days['Net_Strategy_Return'] > 0]) / max(len(active_days), 1)) * 100.0)
        total_trades = int((bt_df['Trade_Action'] > 0).sum() // 2)

    gross_gains = bt_df.loc[bt_df['Net_Strategy_Return'] > 0, 'Net_Strategy_Return'].sum()
    gross_losses = abs(bt_df.loc[bt_df['Net_Strategy_Return'] < 0, 'Net_Strategy_Return'].sum())
    profit_factor = float(gross_gains / (gross_losses + 1e-9))

    metrics = {
        "initial_capital": initial_capital,
        "final_strategy_equity": round(float(bt_df['Strategy_Equity'].iloc[-1]), 2),
        "final_benchmark_equity": round(float(bt_df['Benchmark_Equity'].iloc[-1]), 2),
        "total_strategy_return_pct": round(total_strategy_return_pct, 2),
        "total_benchmark_return_pct": round(total_benchmark_return_pct, 2),
        "outperformance_pct": round(total_strategy_return_pct - total_benchmark_return_pct, 2),
        "cagr_strategy_pct": round(cagr_strategy, 2),
        "cagr_benchmark_pct": round(cagr_benchmark, 2),
        "sharpe_ratio": round(sharpe_ratio, 2),
        "sortino_ratio": round(sortino_ratio, 2),
        "max_drawdown_pct": round(max_drawdown_pct, 2),
        "benchmark_max_drawdown_pct": round(bm_max_drawdown_pct, 2),
        "win_rate_pct": round(win_rate_pct, 1),
        "profit_factor": round(profit_factor, 2),
        "total_completed_trades": total_trades,
        "total_days_evaluated": num_days
    }

    return {
        "metrics": metrics,
        "equity_df": bt_df,
        "trade_log_df": trade_log_df
    }
