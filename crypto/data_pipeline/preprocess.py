"""Data quality, source alignment, and leak-safe forecast targets."""
from __future__ import annotations

import numpy as np
import pandas as pd


def clean_ohlcv(df: pd.DataFrame, timeframe: str = "1h") -> pd.DataFrame:
    out = df.copy()
    out["open_time"] = pd.to_datetime(out["open_time"], utc=True)
    out = out.drop_duplicates("open_time").sort_values("open_time").set_index("open_time")
    # We do not interpolate market candles: missing candles are data-quality issues, not synthetic prices.
    out = out[~out.index.duplicated(keep="last")]
    out = out.dropna(subset=["open", "high", "low", "close", "volume"])
    return out.reset_index()


def merge_sources(
    ohlcv: pd.DataFrame,
    onchain: pd.DataFrame | None = None,
    sentiment: pd.DataFrame | None = None,
) -> pd.DataFrame:
    """
    Merge market, on-chain, and sentiment data using UTC nanosecond-resolution
    timestamps so pandas merge_asof receives identical datetime dtypes.
    """
    out = ohlcv.copy()

    # Normalize market timestamps to one exact dtype.
    out["open_time"] = (
        pd.to_datetime(out["open_time"], utc=True)
        .dt.as_unit("ns")
    )

    out = out.sort_values("open_time").reset_index(drop=True)

    # -----------------------------
    # Merge on-chain data
    # -----------------------------
    if onchain is not None and not onchain.empty:
        oc = onchain.copy()

        oc["timestamp"] = (
            pd.to_datetime(oc["timestamp"], utc=True)
            .dt.as_unit("ns")
        )

        oc = (
            oc.sort_values("timestamp")
            .drop_duplicates("timestamp", keep="last")
            .reset_index(drop=True)
        )

        out = pd.merge_asof(
            out,
            oc,
            left_on="open_time",
            right_on="timestamp",
            direction="backward",
        )

        out = out.drop(columns=["timestamp"], errors="ignore")

    # -----------------------------
    # Merge sentiment data
    # -----------------------------
    if sentiment is not None and not sentiment.empty:
        se = sentiment.copy()

        se["open_time"] = (
            pd.to_datetime(se["open_time"], utc=True)
            .dt.as_unit("ns")
        )

        se = (
            se.sort_values("open_time")
            .drop_duplicates("open_time", keep="last")
            .reset_index(drop=True)
        )

        out = pd.merge_asof(
            out,
            se,
            on="open_time",
            direction="backward",
        )

    # -----------------------------
    # Handle missing exogenous data
    # -----------------------------
    neutral_cols = [
        "active_addresses",
        "tx_count",
        "hash_rate",
        "difficulty",
        "miners_revenue",
        "sentiment_score",
        "news_count",
    ]

    for col in neutral_cols:
        if col in out.columns:
            out[col] = (
                pd.to_numeric(out[col], errors="coerce")
                .ffill()
                .bfill()
                .fillna(0.0)
            )

    return out


def add_targets(df: pd.DataFrame) -> pd.DataFrame:
    out = df.copy()
    out["log_return"] = np.log(out["close"] / out["close"].shift(1))
    for h in (1, 6, 24):
        # Forward-looking target: future realized volatility based on returns after t.
        future_returns = out["log_return"].shift(-1)
        if h == 1:
            out[f"realized_vol_{h}"] = future_returns.abs()
        else:
            out[f"realized_vol_{h}"] = future_returns.rolling(h).std().shift(-(h - 1)) * np.sqrt(h)
        out[f"future_return_{h}"] = out["close"].shift(-h) / out["close"] - 1.0
        out[f"direction_{h}"] = (out[f"future_return_{h}"] > 0).astype(int)
    # Historical volatility features known at time t.
    for w in (6, 12, 24, 72):
        out[f"hist_vol_{w}"] = out["log_return"].rolling(w).std() * np.sqrt(w)
    return out.dropna().reset_index(drop=True)
