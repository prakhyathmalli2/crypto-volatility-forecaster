"""
Feature Engineering Module
Calculates quantitative technical indicators (RSI, MACD, Bollinger Bands, ATR, Stochastic, EMAs)
and merges them with statistical GARCH volatility estimates and Parkinson extreme-value volatility.
"""

import pandas as pd
import numpy as np
import logging

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


def calculate_rsi(series: pd.Series, period: int = 14) -> pd.Series:
    """Calculate Relative Strength Index (RSI)."""
    delta = series.diff()
    gain = delta.clip(lower=0)
    loss = -delta.clip(upper=0)

    avg_gain = gain.ewm(alpha=1/period, min_periods=period, adjust=False).mean()
    avg_loss = loss.ewm(alpha=1/period, min_periods=period, adjust=False).mean()

    rs = avg_gain / (avg_loss + 1e-9)
    rsi = 100 - (100 / (1 + rs))
    rsi.name = f"RSI_{period}"
    return rsi


def calculate_stochastic_oscillator(
    high: pd.Series,
    low: pd.Series,
    close: pd.Series,
    k_period: int = 14,
    d_period: int = 3
) -> pd.DataFrame:
    """Calculate Stochastic Oscillator (%K and %D)."""
    lowest_low = low.rolling(window=k_period).min()
    highest_high = high.rolling(window=k_period).max()
    
    stoch_k = 100.0 * ((close - lowest_low) / ((highest_high - lowest_low) + 1e-9))
    stoch_d = stoch_k.rolling(window=d_period).mean()

    return pd.DataFrame({
        f"Stoch_K_{k_period}": stoch_k,
        f"Stoch_D_{d_period}": stoch_d
    }, index=close.index)


def calculate_macd(
    series: pd.Series,
    fast: int = 12,
    slow: int = 26,
    signal: int = 9
) -> pd.DataFrame:
    """Calculate Moving Average Convergence Divergence (MACD)."""
    ema_fast = series.ewm(span=fast, adjust=False).mean()
    ema_slow = series.ewm(span=slow, adjust=False).mean()
    macd_line = ema_fast - ema_slow
    signal_line = macd_line.ewm(span=signal, adjust=False).mean()
    histogram = macd_line - signal_line

    return pd.DataFrame({
        "MACD_Line": macd_line,
        "MACD_Signal": signal_line,
        "MACD_Hist": histogram
    }, index=series.index)


def calculate_bollinger_bands(
    series: pd.Series,
    window: int = 20,
    num_std: float = 2.0
) -> pd.DataFrame:
    """Calculate Bollinger Bands, Bandwidth, and %B."""
    sma = series.rolling(window=window).mean()
    std = series.rolling(window=window).std()

    upper_band = sma + (std * num_std)
    lower_band = sma - (std * num_std)
    bandwidth = (upper_band - lower_band) / (sma + 1e-9)
    pct_b = (series - lower_band) / ((upper_band - lower_band) + 1e-9)

    return pd.DataFrame({
        "BB_Upper": upper_band,
        "BB_Middle": sma,
        "BB_Lower": lower_band,
        "BB_Bandwidth": bandwidth,
        "BB_PctB": pct_b
    }, index=series.index)


def calculate_atr(
    high: pd.Series,
    low: pd.Series,
    close: pd.Series,
    period: int = 14
) -> pd.DataFrame:
    """Calculate Average True Range (ATR) and Normalized ATR %."""
    prev_close = close.shift(1)
    tr1 = high - low
    tr2 = (high - prev_close).abs()
    tr3 = (low - prev_close).abs()
    tr = pd.concat([tr1, tr2, tr3], axis=1).max(axis=1)

    atr = tr.ewm(alpha=1/period, min_periods=period, adjust=False).mean()
    natr_pct = (atr / (close + 1e-9)) * 100.0

    return pd.DataFrame({
        f"ATR_{period}": atr,
        f"NATR_Pct_{period}": natr_pct
    }, index=close.index)


def build_full_feature_matrix(
    df: pd.DataFrame,
    garch_vol_series: pd.Series = None
) -> pd.DataFrame:
    """
    Construct complete feature matrix combining price action, momentum, trend,
    volatility indicators, and GARCH conditional volatility.
    """
    logger.info("Building comprehensive quantitative feature matrix...")
    features = pd.DataFrame(index=df.index)

    # 1. Price Momentum & Oscillators
    features['RSI_14'] = calculate_rsi(df['Close'], period=14)
    features['RSI_7'] = calculate_rsi(df['Close'], period=7)

    stoch_df = calculate_stochastic_oscillator(df['High'], df['Low'], df['Close'], k_period=14, d_period=3)
    features['Stoch_K'] = stoch_df['Stoch_K_14']
    features['Stoch_D'] = stoch_df['Stoch_D_3']

    macd_df = calculate_macd(df['Close'])
    features['MACD_Line'] = macd_df['MACD_Line']
    features['MACD_Signal'] = macd_df['MACD_Signal']
    features['MACD_Hist'] = macd_df['MACD_Hist']

    # 2. Volatility Bands & Metrics
    bb_df = calculate_bollinger_bands(df['Close'], window=20, num_std=2.0)
    features['BB_Bandwidth'] = bb_df['BB_Bandwidth']
    features['BB_PctB'] = bb_df['BB_PctB']

    atr_df = calculate_atr(df['High'], df['Low'], df['Close'], period=14)
    features['ATR_14'] = atr_df['ATR_14']
    features['NATR_Pct_14'] = atr_df['NATR_Pct_14']

    # Rolling Realized Volatilities (annualized %)
    features['Realized_Vol_7d'] = df['Log_Return'].rolling(7).std() * np.sqrt(365.0) * 100.0
    features['Realized_Vol_14d'] = df['Log_Return'].rolling(14).std() * np.sqrt(365.0) * 100.0
    features['Realized_Vol_30d'] = df['Log_Return'].rolling(30).std() * np.sqrt(365.0) * 100.0

    # 3. Moving Average Ratios & Trend
    ema_9 = df['Close'].ewm(span=9, adjust=False).mean()
    ema_21 = df['Close'].ewm(span=21, adjust=False).mean()
    sma_50 = df['Close'].rolling(50).mean()

    features['EMA_Ratio_9_21'] = (ema_9 / (ema_21 + 1e-9)) - 1.0
    features['Price_to_SMA50'] = (df['Close'] / (sma_50 + 1e-9)) - 1.0

    # 4. Volume Features
    vol_sma20 = df['Volume'].rolling(20).mean()
    features['Volume_Ratio_20'] = df['Volume'] / (vol_sma20 + 1e-9)

    # 5. Lagged Log Returns (Autoregressive features)
    features['Return_Lag_1'] = df['Log_Return'].shift(1)
    features['Return_Lag_2'] = df['Log_Return'].shift(2)
    features['Return_Lag_3'] = df['Log_Return'].shift(3)

    # 6. GARCH Conditional Volatility Integration
    if garch_vol_series is not None:
        aligned_vol = garch_vol_series.reindex(df.index)
        features['GARCH_Cond_Vol'] = aligned_vol
        features['GARCH_Vol_Change_1d'] = features['GARCH_Cond_Vol'].pct_change()
        features['Vol_Ratio_GARCH_to_Realized'] = features['GARCH_Cond_Vol'] / (features['Realized_Vol_14d'] + 1e-9)

    # Clean infinities and NaNs from rolling calculations
    features.replace([np.inf, -np.inf], np.nan, inplace=True)
    features.dropna(inplace=True)
    logger.info(f"Feature matrix built with {features.shape[1]} features and {len(features)} rows.")

    return features
