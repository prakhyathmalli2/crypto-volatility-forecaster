"""Comprehensive technical, volatility, and multi-source feature engineering.

Generates:
- Parkinson Volatility estimator
- Garman-Klass Volatility estimator
- Rolling log-return standard deviation
- Bollinger Bands bandwidth & position
- RSI (Relative Strength Index)
- MACD (MACD line, signal line, histogram)
- ATR (Average True Range) and ATR percentage
- Exponential Moving Average distances (EMA 12, 26, 50, 200)
- Volume Z-scores & candle geometry features
- On-chain and sentiment indicators
- Serializes feature schema and standard scaler to artifacts/
"""
from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import joblib
import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler

from config import (
    ARTIFACTS_DIR,
    ATR_PERIOD,
    BOLLINGER_PERIOD,
    BOLLINGER_STD,
    FEATURES_DIR,
    MACD_FAST,
    MACD_SIGNAL,
    MACD_SLOW,
    RSI_PERIOD,
    VOLATILITY_WINDOWS,
)


def compute_parkinson_volatility(df: pd.DataFrame, window: int = 24) -> pd.Series:
    """
    Parkinson volatility estimator based on High and Low prices:
    sigma_P = sqrt( (1 / (4 * ln(2))) * (1 / N) * sum( (ln(High / Low))^2 ) )
    """
    log_hl_ratio = np.log(df["high"] / np.maximum(df["low"], 1e-9))
    factor = 1.0 / (4.0 * np.log(2.0))
    rolling_sq = (log_hl_ratio ** 2).rolling(window=window, min_periods=window).mean()
    return np.sqrt(factor * rolling_sq)


def compute_garman_klass_volatility(df: pd.DataFrame, window: int = 24) -> pd.Series:
    """
    Garman-Klass volatility estimator incorporating Open, High, Low, and Close prices:
    sigma_GK = sqrt( (1 / N) * sum( 0.5 * (ln(H/L))^2 - (2*ln(2) - 1) * (ln(C/O))^2 ) )
    """
    log_hl = np.log(df["high"] / np.maximum(df["low"], 1e-9))
    log_co = np.log(df["close"] / np.maximum(df["open"], 1e-9))
    term1 = 0.5 * (log_hl ** 2)
    term2 = (2.0 * np.log(2.0) - 1.0) * (log_co ** 2)
    rolling_val = (term1 - term2).rolling(window=window, min_periods=window).mean()
    # Clip negative values resulting from numerical precision
    return np.sqrt(np.maximum(rolling_val, 0.0))


def compute_atr(df: pd.DataFrame, period: int = ATR_PERIOD) -> pd.DataFrame:
    """Average True Range (ATR) and normalized ATR percentage."""
    out = df.copy()
    prev_close = out["close"].shift(1)
    tr1 = out["high"] - out["low"]
    tr2 = (out["high"] - prev_close).abs()
    tr3 = (out["low"] - prev_close).abs()
    tr = pd.concat([tr1, tr2, tr3], axis=1).max(axis=1)
    out["atr"] = tr.rolling(window=period, min_periods=period).mean()
    out["atr_pct"] = out["atr"] / out["close"]
    return out


def compute_rsi(df: pd.DataFrame, period: int = RSI_PERIOD) -> pd.DataFrame:
    """Wilder's Exponential Relative Strength Index (RSI)."""
    out = df.copy()
    delta = out["close"].diff()
    gain = delta.clip(lower=0.0).ewm(alpha=1.0 / period, adjust=False).mean()
    loss = (-delta.clip(upper=0.0)).ewm(alpha=1.0 / period, adjust=False).mean()
    rs = gain / np.maximum(loss, 1e-9)
    out["rsi"] = 100.0 - (100.0 / (1.0 + rs))
    return out


def compute_macd(
    df: pd.DataFrame,
    fast: int = MACD_FAST,
    slow: int = MACD_SLOW,
    signal: int = MACD_SIGNAL,
) -> pd.DataFrame:
    """Moving Average Convergence Divergence (MACD)."""
    out = df.copy()
    ema_fast = out["close"].ewm(span=fast, adjust=False).mean()
    ema_slow = out["close"].ewm(span=slow, adjust=False).mean()
    out["macd_line"] = ema_fast - ema_slow
    out["macd_signal"] = out["macd_line"].ewm(span=signal, adjust=False).mean()
    out["macd_hist"] = out["macd_line"] - out["macd_signal"]
    return out


def compute_bollinger_bands(
    df: pd.DataFrame, period: int = BOLLINGER_PERIOD, num_std: float = BOLLINGER_STD
) -> pd.DataFrame:
    """Bollinger Bands bandwidth and position (%B)."""
    out = df.copy()
    sma = out["close"].rolling(window=period, min_periods=period).mean()
    std = out["close"].rolling(window=period, min_periods=period).std(ddof=1)
    upper = sma + num_std * std
    lower = sma - num_std * std
    out["bb_width"] = (upper - lower) / np.maximum(sma, 1e-9)
    out["bb_position"] = (out["close"] - lower) / np.maximum(upper - lower, 1e-9)
    return out


def compute_moving_averages(df: pd.DataFrame) -> pd.DataFrame:
    """EMA distance features (trend & momentum divergence)."""
    out = df.copy()
    for span in (12, 26, 50, 200):
        ema = out["close"].ewm(span=span, adjust=False).mean()
        out[f"ema_{span}_dist"] = (out["close"] / np.maximum(ema, 1e-9)) - 1.0
    return out


def engineer_features(df: pd.DataFrame) -> pd.DataFrame:
    """Execute complete feature engineering transformations on aligned dataset."""
    out = df.copy()

    if "log_return" not in out.columns:
        out["log_return"] = np.log(out["close"] / out["close"].shift(1))

    # 1. Advanced volatility estimators
    out["parkinson_vol_24"] = compute_parkinson_volatility(out, window=24)
    out["garman_klass_vol_24"] = compute_garman_klass_volatility(out, window=24)

    for w in VOLATILITY_WINDOWS:
        col = f"hist_vol_{w}"
        if col not in out.columns:
            out[col] = out["log_return"].rolling(window=w, min_periods=w).std(ddof=1) * np.sqrt(w)

    # 2. Technical indicators
    out = compute_atr(out)
    out = compute_rsi(out)
    out = compute_macd(out)
    out = compute_bollinger_bands(out)
    out = compute_moving_averages(out)

    # 3. Volume and Candle dynamics
    vol_mean = out["volume"].rolling(window=72, min_periods=24).mean()
    vol_std = out["volume"].rolling(window=72, min_periods=24).std(ddof=1)
    out["volume_z"] = (out["volume"] - vol_mean) / np.maximum(vol_std, 1e-9)
    out["range_pct"] = (out["high"] - out["low"]) / out["close"]
    out["body_pct"] = (out["close"] - out["open"]) / out["open"]

    # 4. Clean infinities and NaNs
    out = out.replace([np.inf, -np.inf], np.nan)
    return out.dropna().reset_index(drop=True)


# Official ordered list of ML model input features
CANONICAL_FEATURES = [
    "log_return",
    "parkinson_vol_24",
    "garman_klass_vol_24",
    "hist_vol_6",
    "hist_vol_12",
    "hist_vol_24",
    "hist_vol_72",
    "atr_pct",
    "rsi",
    "macd_line",
    "macd_signal",
    "macd_hist",
    "bb_width",
    "bb_position",
    "ema_12_dist",
    "ema_26_dist",
    "ema_50_dist",
    "ema_200_dist",
    "volume_z",
    "range_pct",
    "body_pct",
    "sentiment_score",
    "news_count",
    "fear_greed_value",
    "active_addresses",
    "tx_count",
    "hash_rate",
    "difficulty",
]


def fit_and_save_feature_store(
    train_df: pd.DataFrame,
    valid_df: pd.DataFrame,
    test_df: pd.DataFrame,
) -> tuple[StandardScaler, list[str]]:
    """Fit standard scaler on training split ONLY and save feature schema and matrices."""
    # Ensure all canonical features exist in all frames
    for col in CANONICAL_FEATURES:
        for frame in (train_df, valid_df, test_df):
            if col not in frame.columns:
                frame[col] = 0.0

    scaler = StandardScaler()
    scaler.fit(train_df[CANONICAL_FEATURES])

    # Save scaler and schema into artifacts/
    scaler_path = ARTIFACTS_DIR / "scaler.pkl"
    joblib.dump(scaler, scaler_path)

    schema_path = ARTIFACTS_DIR / "feature_schema.json"
    schema_info = {
        "features": CANONICAL_FEATURES,
        "n_features": len(CANONICAL_FEATURES),
        "scaler_means": {col: float(m) for col, m in zip(CANONICAL_FEATURES, scaler.mean_)},
        "scaler_scales": {col: float(s) for col, s in zip(CANONICAL_FEATURES, scaler.scale_)},
    }
    schema_path.write_text(json.dumps(schema_info, indent=2), encoding="utf-8")

    # Export scaled features matrices to data/features/
    train_scaled = train_df.copy()
    valid_scaled = valid_df.copy()
    test_scaled = test_df.copy()

    train_scaled[CANONICAL_FEATURES] = scaler.transform(train_df[CANONICAL_FEATURES])
    valid_scaled[CANONICAL_FEATURES] = scaler.transform(valid_df[CANONICAL_FEATURES])
    test_scaled[CANONICAL_FEATURES] = scaler.transform(test_df[CANONICAL_FEATURES])

    train_scaled.to_csv(FEATURES_DIR / "train_features.csv", index=False)
    valid_scaled.to_csv(FEATURES_DIR / "valid_features.csv", index=False)
    test_scaled.to_csv(FEATURES_DIR / "test_features.csv", index=False)

    return scaler, CANONICAL_FEATURES
