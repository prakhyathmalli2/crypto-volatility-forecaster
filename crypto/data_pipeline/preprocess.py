"""Data preprocessing, leak-safe timestamp alignment, and target generation.

Merges market data with on-chain and sentiment indicators using backward as-of
joins, strictly avoiding lookahead bias. Computes forward-looking realized
volatility targets across multiple forecasting horizons (1h, 6h, 24h).
"""
from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from config import (
    FEATURES_DIR,
    FORECAST_HORIZONS,
    PROCESSED_DIR,
    SEQ_LEN,
    TRAIN_RATIO,
    VALID_RATIO,
    VOLATILITY_WINDOWS,
)


def clean_ohlcv(df: pd.DataFrame) -> pd.DataFrame:
    """Validate and clean raw OHLCV market candles."""
    out = df.copy()
    out["open_time"] = pd.to_datetime(out["open_time"], utc=True)
    out = out.drop_duplicates(subset=["open_time"]).sort_values("open_time")

    numeric_cols = ["open", "high", "low", "close", "volume"]
    for col in numeric_cols:
        out[col] = pd.to_numeric(out[col], errors="coerce")

    # Drop any null or non-positive price rows
    out = out.dropna(subset=numeric_cols)
    out = out[(out["open"] > 0) & (out["high"] > 0) & (out["low"] > 0) & (out["close"] > 0)]
    return out.reset_index(drop=True)


def merge_sources_leak_free(
    ohlcv_df: pd.DataFrame,
    onchain_df: pd.DataFrame | None = None,
    sentiment_df: pd.DataFrame | None = None,
) -> pd.DataFrame:
    """
    Merge market, on-chain, and sentiment series using UTC nanosecond timestamps
    and backward as-of joins so information at time t is only ever matched with
    exogenous data published at or before time t.
    """
    merged = ohlcv_df.copy()
    merged["open_time"] = pd.to_datetime(merged["open_time"], utc=True).dt.as_unit("ns")
    merged = merged.sort_values("open_time").reset_index(drop=True)

    # 1. Backward as-of join with on-chain metrics
    if onchain_df is not None and not onchain_df.empty:
        oc = onchain_df.copy()
        ts_col = "timestamp" if "timestamp" in oc.columns else "open_time"
        oc["open_time"] = pd.to_datetime(oc[ts_col], utc=True).dt.as_unit("ns")
        oc = oc.drop(columns=["timestamp"], errors="ignore")
        oc = oc.sort_values("open_time").drop_duplicates("open_time", keep="last").reset_index(drop=True)

        merged = pd.merge_asof(
            merged,
            oc,
            on="open_time",
            direction="backward",
        )

    # 2. Backward as-of join with sentiment metrics
    if sentiment_df is not None and not sentiment_df.empty:
        se = sentiment_df.copy()
        se["open_time"] = pd.to_datetime(se["open_time"], utc=True).dt.as_unit("ns")
        se = se.sort_values("open_time").drop_duplicates("open_time", keep="last").reset_index(drop=True)

        merged = pd.merge_asof(
            merged,
            se,
            on="open_time",
            direction="backward",
        )

    # Clean fill any missing exogenous columns with forward fill only, then neutral fallback (no lookahead bias)
    exogenous_defaults = {
        "active_addresses": 800000.0,
        "tx_count": 300000.0,
        "hash_rate": 500.0,
        "difficulty": 80.0,
        "miners_revenue": 30000000.0,
        "mempool_size": 15000000.0,
        "fear_greed_value": 50.0,
        "sentiment_score": 0.0,
        "news_count": 0.0,
    }

    for col, default_val in exogenous_defaults.items():
        if col in merged.columns:
            merged[col] = pd.to_numeric(merged[col], errors="coerce").ffill().fillna(default_val)
        else:
            merged[col] = default_val

    # Ensure any other joined exogenous columns only forward-fill and default to 0.0 for leading missing values
    exog_cols: set[str] = set()
    if onchain_df is not None:
        exog_cols.update(c for c in onchain_df.columns if c not in ("open_time", "timestamp"))
    if sentiment_df is not None:
        exog_cols.update(c for c in sentiment_df.columns if c not in ("open_time", "timestamp"))
    for col in exog_cols:
        if col in merged.columns and col not in exogenous_defaults:
            merged[col] = pd.to_numeric(merged[col], errors="coerce").ffill().fillna(0.0)

    return merged.reset_index(drop=True)


def compute_realized_volatility_targets(
    df: pd.DataFrame, horizons: tuple[int, ...] = (1, 6, 24)
) -> pd.DataFrame:
    """
    Calculate future realized volatility and directional return targets.

    Mathematical formula for Realized Volatility over horizon h:
    sigma_{t+1:t+h} = sqrt( (1/h) * sum_{i=1}^h (r_{t+i} - mean(r))^2 ) * sqrt(h)

    Directional target:
    direction_{h} = 1 if (close_{t+h} / close_t - 1) > 0 else 0
    """
    out = df.copy()
    out["log_return"] = np.log(out["close"] / out["close"].shift(1))

    for h in horizons:
        # Shift future returns backward so row t receives returns from t+1 to t+h
        future_log_returns = out["log_return"].shift(-1)
        if h == 1:
            # For 1-hour horizon, realized volatility is the absolute next-step return
            out[f"realized_vol_{h}"] = future_log_returns.abs()
        else:
            # Rolling standard deviation over window h, shifted so it looks strictly forward
            rolling_vol = (
                future_log_returns.iloc[::-1]
                .rolling(window=h, min_periods=h)
                .std(ddof=1)
                .iloc[::-1]
                * np.sqrt(h)
            )
            out[f"realized_vol_{h}"] = rolling_vol

        # Forward price return over horizon h
        forward_price = out["close"].shift(-h)
        out[f"future_return_{h}"] = forward_price / out["close"] - 1.0
        out[f"direction_{h}"] = (out[f"future_return_{h}"] > 0.0).astype(int)

    # Historical backward-looking volatility known at time t (features, not targets)
    for w in VOLATILITY_WINDOWS:
        out[f"hist_vol_{w}"] = out["log_return"].rolling(window=w, min_periods=w).std(ddof=1) * np.sqrt(w)

    return out.dropna().reset_index(drop=True)


# Compatibility alias
add_targets = compute_realized_volatility_targets



def split_chronological_datasets(
    df: pd.DataFrame,
    train_ratio: float = TRAIN_RATIO,
    valid_ratio: float = VALID_RATIO,
    purge_horizon: int = 24,
) -> dict[str, pd.DataFrame]:
    """
    Split dataset chronologically (never randomly) to prevent data leakage in time-series forecasting.
    Purges forward-label rows at split boundaries so targets never use future prices from subsequent splits.
    Train (0 .. train_end - purge_horizon) | Validation (train_end .. valid_end - purge_horizon) | Test (valid_end .. end)
    """
    n = len(df)
    train_end = int(n * train_ratio)
    valid_end = int(n * (train_ratio + valid_ratio))

    if train_end <= SEQ_LEN or valid_end >= n:
        raise ValueError(
            f"Dataset length {n} is too small for chronological split with sequence length {SEQ_LEN}"
        )

    train_slice_end = max(0, train_end - purge_horizon) if purge_horizon > 0 else train_end
    valid_slice_end = max(train_end, valid_end - purge_horizon) if purge_horizon > 0 else valid_end

    train_df = df.iloc[:train_slice_end].copy().reset_index(drop=True)
    valid_df = df.iloc[train_end:valid_slice_end].copy().reset_index(drop=True)
    test_df = df.iloc[valid_end:].copy().reset_index(drop=True)

    # Save to data/processed/ and data/features/
    train_df.to_csv(PROCESSED_DIR / "train.csv", index=False)
    valid_df.to_csv(PROCESSED_DIR / "valid.csv", index=False)
    test_df.to_csv(PROCESSED_DIR / "test.csv", index=False)

    split_meta = {
        "total_rows": n,
        "train_rows": len(train_df),
        "valid_rows": len(valid_df),
        "test_rows": len(test_df),
        "train_date_range": [str(train_df["open_time"].iloc[0]), str(train_df["open_time"].iloc[-1])],
        "valid_date_range": [str(valid_df["open_time"].iloc[0]), str(valid_df["open_time"].iloc[-1])],
        "test_date_range": [str(test_df["open_time"].iloc[0]), str(test_df["open_time"].iloc[-1])],
    }
    (PROCESSED_DIR / "split_metadata.json").write_text(json.dumps(split_meta, indent=2), encoding="utf-8")

    return {"train": train_df, "valid": valid_df, "test": test_df}
