"""High-performance technical indicators calculation for live TradingView-style charts.

Computes:
- EMAs (9, 20, 50, 200)
- RSI (14)
- MACD (12, 26, 9)
- ATR (14)
- Bollinger Bands (20, 2.0)
- VWAP
- Volume Moving Average (20)
"""
from __future__ import annotations

from typing import Any
import numpy as np
import pandas as pd


def calculate_indicators(candles_data: list[dict[str, Any]] | pd.DataFrame) -> dict[str, Any]:
    """
    Calculate comprehensive technical indicators from raw candle history.

    Returns:
    - time_series: dict of aligned indicator series for charting
    - current: latest scalar values for header cards and signal engines
    """
    if isinstance(candles_data, list):
        if not candles_data:
            return {"time_series": {}, "current": {}}
        df = pd.DataFrame(candles_data)
    else:
        df = candles_data.copy()

    if len(df) < 5:
        return {"time_series": {}, "current": {}}

    for col in ["open", "high", "low", "close", "volume"]:
        if col in df.columns:
            df[col] = pd.to_numeric(df[col], errors="coerce")

    # 1. Exponential Moving Averages (EMA)
    close = df["close"]
    ema9 = close.ewm(span=9, adjust=False).mean()
    ema20 = close.ewm(span=20, adjust=False).mean()
    ema50 = close.ewm(span=50, adjust=False).mean()
    ema200 = close.ewm(span=min(200, max(len(df), 10)), adjust=False).mean()

    # 2. Relative Strength Index (RSI 14)
    delta = close.diff()
    gain = delta.clip(lower=0.0)
    loss = -delta.clip(upper=0.0)
    avg_gain = gain.ewm(alpha=1.0 / 14, min_periods=14, adjust=False).mean()
    avg_loss = loss.ewm(alpha=1.0 / 14, min_periods=14, adjust=False).mean()
    rs = avg_gain / avg_loss.replace(0, np.nan)
    rsi = 100.0 - (100.0 / (1.0 + rs))
    rsi = rsi.fillna(50.0)

    # 3. Moving Average Convergence Divergence (MACD 12, 26, 9)
    ema12 = close.ewm(span=12, adjust=False).mean()
    ema26 = close.ewm(span=26, adjust=False).mean()
    macd_line = ema12 - ema26
    macd_signal = macd_line.ewm(span=9, adjust=False).mean()
    macd_hist = macd_line - macd_signal

    # 4. Average True Range (ATR 14)
    high = df["high"]
    low = df["low"]
    prev_close = close.shift(1)
    tr = pd.concat(
        [
            high - low,
            (high - prev_close).abs(),
            (low - prev_close).abs(),
        ],
        axis=1,
    ).max(axis=1)
    atr = tr.ewm(span=14, adjust=False).mean()

    # 5. Bollinger Bands (20, 2.0) & SMAs
    sma20 = close.rolling(20, min_periods=5).mean()
    sma50 = close.rolling(min(50, len(df)), min_periods=5).mean()
    sma200 = close.rolling(min(200, len(df)), min_periods=5).mean()
    std20 = close.rolling(20, min_periods=5).std()
    bb_upper = sma20 + (2.0 * std20)
    bb_lower = sma20 - (2.0 * std20)
    bb_pct = (close - bb_lower) / (bb_upper - bb_lower).replace(0, np.nan)

    # 6. Volume Weighted Average Price (VWAP)
    typical_price = (high + low + close) / 3.0
    vol = df["volume"].replace(0, 1e-6)
    cum_vol_price = (typical_price * vol).cumsum()
    cum_vol = vol.cumsum()
    vwap = cum_vol_price / cum_vol

    # 7. Volume SMA
    vol_sma20 = vol.rolling(20, min_periods=5).mean()

    # 8. Stochastic Oscillator (%K 14, %D 3)
    low14 = low.rolling(14, min_periods=5).min()
    high14 = high.rolling(14, min_periods=5).max()
    denom = (high14 - low14).replace(0, np.nan)
    stoch_raw = 100.0 * ((close - low14) / denom)
    stoch_k = stoch_raw.rolling(3, min_periods=1).mean().fillna(50.0)
    stoch_d = stoch_k.rolling(3, min_periods=1).mean().fillna(50.0)

    # 9. Average Directional Index (ADX 14 with +DI, -DI)
    up_move = high - high.shift(1)
    down_move = low.shift(1) - low
    plus_dm = np.where((up_move > down_move) & (up_move > 0), up_move, 0.0)
    minus_dm = np.where((down_move > up_move) & (down_move > 0), down_move, 0.0)

    alpha_14 = 1.0 / 14.0
    smooth_tr = tr.ewm(alpha=alpha_14, adjust=False).mean()
    smooth_plus = pd.Series(plus_dm, index=df.index).ewm(alpha=alpha_14, adjust=False).mean()
    smooth_minus = pd.Series(minus_dm, index=df.index).ewm(alpha=alpha_14, adjust=False).mean()

    plus_di = 100.0 * (smooth_plus / smooth_tr.replace(0, np.nan)).fillna(0.0)
    minus_di = 100.0 * (smooth_minus / smooth_tr.replace(0, np.nan)).fillna(0.0)
    dx_denom = (plus_di + minus_di).replace(0, np.nan)
    dx = 100.0 * ((plus_di - minus_di).abs() / dx_denom).fillna(0.0)
    adx = dx.ewm(alpha=alpha_14, adjust=False).mean().fillna(20.0)

    # 10. Price Momentum & Rate of Change (ROC 10)
    mom_lookback = min(10, max(len(df) - 1, 1))
    shifted_close = close.shift(mom_lookback)
    momentum = (close - shifted_close).fillna(0.0)
    roc = (((close - shifted_close) / shifted_close.replace(0, np.nan)) * 100.0).fillna(0.0)

    # 11. Historical Realized Volatility (24-period)
    log_ret = np.log(close / close.shift(1).replace(0, np.nan)).fillna(0.0)
    hist_vol_24 = (log_ret.rolling(min(24, len(df)), min_periods=3).std() * np.sqrt(365 * 24) * 100.0).fillna(0.0)

    # Format output series aligned with timestamps
    ts_list = (
        df["timestamp"].tolist()
        if "timestamp" in df.columns
        else (df["open_time"].astype(str).tolist() if "open_time" in df.columns else list(range(len(df))))
    )

    def _sanitize(series: pd.Series) -> list[float | None]:
        return [None if np.isnan(v) else round(float(v), 2) for v in series]

    time_series = {
        "timestamps": ts_list,
        "ema9": _sanitize(ema9),
        "ema20": _sanitize(ema20),
        "ema50": _sanitize(ema50),
        "ema200": _sanitize(ema200),
        "sma20": _sanitize(sma20),
        "sma50": _sanitize(sma50),
        "sma200": _sanitize(sma200),
        "rsi": _sanitize(rsi),
        "macd_line": _sanitize(macd_line),
        "macd_signal": _sanitize(macd_signal),
        "macd_hist": _sanitize(macd_hist),
        "stoch_k": _sanitize(stoch_k),
        "stoch_d": _sanitize(stoch_d),
        "adx": _sanitize(adx),
        "plus_di": _sanitize(plus_di),
        "minus_di": _sanitize(minus_di),
        "momentum": _sanitize(momentum),
        "roc": _sanitize(roc),
        "hist_vol": _sanitize(hist_vol_24),
        "atr": _sanitize(atr),
        "bb_upper": _sanitize(bb_upper),
        "bb_middle": _sanitize(sma20),
        "bb_lower": _sanitize(bb_lower),
        "vwap": _sanitize(vwap),
        "vol_sma20": _sanitize(vol_sma20),
    }

    # Extract latest values
    last_idx = -1
    current = {
        "price": round(float(close.iloc[last_idx]), 2),
        "ema9": round(float(ema9.iloc[last_idx]), 2),
        "ema20": round(float(ema20.iloc[last_idx]), 2),
        "ema50": round(float(ema50.iloc[last_idx]), 2),
        "sma20": round(float(sma20.iloc[last_idx]), 2),
        "sma50": round(float(sma50.iloc[last_idx]), 2),
        "rsi": round(float(rsi.iloc[last_idx]), 2),
        "macd_line": round(float(macd_line.iloc[last_idx]), 2),
        "macd_signal": round(float(macd_signal.iloc[last_idx]), 2),
        "macd_hist": round(float(macd_hist.iloc[last_idx]), 2),
        "stoch_k": round(float(stoch_k.iloc[last_idx]), 2),
        "stoch_d": round(float(stoch_d.iloc[last_idx]), 2),
        "adx": round(float(adx.iloc[last_idx]), 2),
        "plus_di": round(float(plus_di.iloc[last_idx]), 2),
        "minus_di": round(float(minus_di.iloc[last_idx]), 2),
        "trend_strength": "STRONG" if adx.iloc[last_idx] >= 25 else "WEAK/RANGING",
        "momentum": round(float(momentum.iloc[last_idx]), 2),
        "roc": round(float(roc.iloc[last_idx]), 2),
        "hist_vol": round(float(hist_vol_24.iloc[last_idx]), 2),
        "atr": round(float(atr.iloc[last_idx]), 2),
        "bb_upper": round(float(bb_upper.iloc[last_idx]), 2),
        "bb_lower": round(float(bb_lower.iloc[last_idx]), 2),
        "vwap": round(float(vwap.iloc[last_idx]), 2),
        "trend_regime": (
            "BULLISH"
            if ema20.iloc[last_idx] > ema50.iloc[last_idx] and close.iloc[last_idx] > ema20.iloc[last_idx]
            else ("BEARISH" if ema20.iloc[last_idx] < ema50.iloc[last_idx] and close.iloc[last_idx] < ema20.iloc[last_idx] else "NEUTRAL")
        ),
    }

    return {"time_series": time_series, "current": current}
