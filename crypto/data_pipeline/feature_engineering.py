"""Technical, volatility, momentum, volume, and exogenous features."""
from __future__ import annotations

import numpy as np
import pandas as pd

from config import ATR_PERIOD, RSI_PERIOD


def compute_atr(df: pd.DataFrame, period: int = ATR_PERIOD) -> pd.DataFrame:
    out = df.copy()
    prev_close = out["close"].shift(1)
    tr = pd.concat([(out["high"] - out["low"]), (out["high"] - prev_close).abs(), (out["low"] - prev_close).abs()], axis=1).max(axis=1)
    out["atr"] = tr.rolling(period).mean()
    out["atr_pct"] = out["atr"] / out["close"]
    return out


def compute_rsi(df: pd.DataFrame, period: int = RSI_PERIOD) -> pd.DataFrame:
    out = df.copy()
    delta = out["close"].diff()
    gain = delta.clip(lower=0).ewm(alpha=1 / period, adjust=False).mean()
    loss = (-delta.clip(upper=0)).ewm(alpha=1 / period, adjust=False).mean()
    rs = gain / (loss + 1e-9)
    out["rsi"] = 100 - (100 / (1 + rs))
    return out


def compute_bollinger(df: pd.DataFrame, period: int = 20, num_std: float = 2.0) -> pd.DataFrame:
    out = df.copy()
    ma = out["close"].rolling(period).mean()
    std = out["close"].rolling(period).std()
    out["bb_upper"] = ma + num_std * std
    out["bb_lower"] = ma - num_std * std
    out["bb_width"] = (out["bb_upper"] - out["bb_lower"]) / (ma + 1e-9)
    out["bb_position"] = (out["close"] - out["bb_lower"]) / (out["bb_upper"] - out["bb_lower"] + 1e-9)
    return out


def compute_moving_averages(df: pd.DataFrame) -> pd.DataFrame:
    out = df.copy()
    for span in (12, 26, 50, 200):
        out[f"ema_{span}"] = out["close"].ewm(span=span, adjust=False).mean()
        out[f"ema_{span}_dist"] = out["close"] / out[f"ema_{span}"] - 1
    return out


def engineer_features(df: pd.DataFrame) -> pd.DataFrame:
    out = df.copy()
    if "log_return" not in out.columns:
        out["log_return"] = np.log(out["close"] / out["close"].shift(1))
    for w in (6, 12, 24, 72):
        col = f"hist_vol_{w}"
        if col not in out.columns:
            out[col] = out["log_return"].rolling(w).std() * np.sqrt(w)
    out = compute_atr(out)
    out = compute_rsi(out)
    out = compute_bollinger(out)
    out = compute_moving_averages(out)
    vol_mean = out["volume"].rolling(72).mean()
    vol_std = out["volume"].rolling(72).std()
    out["volume_z"] = (out["volume"] - vol_mean) / (vol_std + 1e-9)
    out["range_pct"] = (out["high"] - out["low"]) / out["close"]
    out["body_pct"] = (out["close"] - out["open"]) / out["open"]
    out = out.replace([np.inf, -np.inf], np.nan)
    return out.dropna().reset_index(drop=True)
