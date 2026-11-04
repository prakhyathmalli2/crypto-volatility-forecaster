"""Unit tests for data pipeline, feature engineering, and leak-free alignment."""
from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from data_pipeline.feature_engineering import (
    compute_garman_klass_volatility,
    compute_parkinson_volatility,
    engineer_features,
)
from data_pipeline.preprocess import (
    clean_ohlcv,
    compute_realized_volatility_targets,
    merge_sources_leak_free,
    split_chronological_datasets,
)


@pytest.fixture
def sample_ohlcv() -> pd.DataFrame:
    """Generate 300 hourly candles for testing."""
    times = pd.date_range("2024-01-01", periods=300, freq="1h", tz="UTC")
    np.random.seed(42)
    prices = 50000.0 * np.cumprod(1.0 + np.random.normal(0, 0.005, 300))
    df = pd.DataFrame(
        {
            "open_time": times,
            "open": prices * 0.999,
            "high": prices * 1.008,
            "low": prices * 0.992,
            "close": prices,
            "volume": np.random.uniform(50, 200, 300),
            "source_exchange": "binance",
        }
    )
    return df


def test_clean_ohlcv(sample_ohlcv):
    # Insert duplicate and negative price to verify cleaning
    bad_row = sample_ohlcv.iloc[0:1].copy()
    bad_row["open"] = -100.0
    dirty = pd.concat([sample_ohlcv, bad_row, sample_ohlcv.iloc[10:11]])
    cleaned = clean_ohlcv(dirty)
    assert len(cleaned) == len(sample_ohlcv)
    assert (cleaned["open"] > 0).all()
    assert (cleaned["close"] > 0).all()


def test_leak_free_backward_merge(sample_ohlcv):
    """Verify on-chain data published at 12:00 is NEVER merged with 11:00 market candle."""
    onchain = pd.DataFrame(
        {
            "timestamp": pd.date_range("2024-01-01", periods=10, freq="12h", tz="UTC"),
            "active_addresses": [1000 + i * 100 for i in range(10)],
        }
    )
    merged = merge_sources_leak_free(sample_ohlcv, onchain_df=onchain)
    # The first candle is at 00:00, second at 01:00. Both should match 00:00 onchain value (1000)
    assert merged["active_addresses"].iloc[0] == 1000
    assert merged["active_addresses"].iloc[1] == 1000
    # Candle at 11:00 should STILL have 1000, not 1100 (which wasn't available until 12:00)
    assert merged["active_addresses"].iloc[11] == 1000
    assert merged["active_addresses"].iloc[12] == 1100


def test_parkinson_and_garman_klass_formulas(sample_ohlcv):
    p_vol = compute_parkinson_volatility(sample_ohlcv, window=24)
    gk_vol = compute_garman_klass_volatility(sample_ohlcv, window=24)

    # Values must be non-negative and finite
    assert (p_vol.dropna() >= 0.0).all()
    assert (gk_vol.dropna() >= 0.0).all()
    assert not p_vol.dropna().isna().any()
    assert not gk_vol.dropna().isna().any()


def test_realized_volatility_targets(sample_ohlcv):
    targeted = compute_realized_volatility_targets(sample_ohlcv, horizons=(1, 6, 24))
    for h in (1, 6, 24):
        assert f"realized_vol_{h}" in targeted.columns
        assert f"direction_{h}" in targeted.columns
        assert (targeted[f"realized_vol_{h}"] >= 0.0).all()
        assert targeted[f"direction_{h}"].isin([0, 1]).all()


def test_chronological_splitting(sample_ohlcv):
    targeted = compute_realized_volatility_targets(sample_ohlcv)
    feats = engineer_features(targeted)
    splits = split_chronological_datasets(feats, train_ratio=0.70, valid_ratio=0.15)

    train = splits["train"]
    valid = splits["valid"]
    test = splits["test"]

    # Verify chronological ordering: train < valid < test
    assert train["open_time"].max() < valid["open_time"].min()
    assert valid["open_time"].max() < test["open_time"].min()


def test_no_future_leakage_in_merge(sample_ohlcv):
    """Verify external data available only at t=12h never leaks into candles from 0h to 11h."""
    start_time = sample_ohlcv["open_time"].iloc[12]
    onchain = pd.DataFrame(
        {
            "timestamp": pd.date_range(start_time, periods=5, freq="12h", tz="UTC"),
            "active_addresses": [999999.0, 888888.0, 777777.0, 666666.0, 555555.0],
        }
    )
    merged = merge_sources_leak_free(sample_ohlcv, onchain_df=onchain)

    # Pre-observation candles (indices 0..11) must use neutral default, NEVER the future 999999.0
    default_val = 800000.0
    for i in range(12):
        assert merged["active_addresses"].iloc[i] == default_val
        assert merged["active_addresses"].iloc[i] != 999999.0

    # Observation at 12h matches
    assert merged["active_addresses"].iloc[12] == 999999.0
    # Forward fill carries into 13h
    assert merged["active_addresses"].iloc[13] == 999999.0


def test_split_boundary_label_purging(sample_ohlcv):
    """Verify 24-step forward targets in train/valid never use price data from the subsequent split."""
    targeted = compute_realized_volatility_targets(sample_ohlcv, horizons=(1, 6, 24))
    feats = engineer_features(targeted)
    splits = split_chronological_datasets(feats, train_ratio=0.70, valid_ratio=0.15, purge_horizon=24)

    train = splits["train"]
    valid = splits["valid"]
    test = splits["test"]

    max_horizon_delta = pd.Timedelta(hours=24)

    # Train's last target price timestamp must be <= valid's first candle
    train_last_target_time = train["open_time"].iloc[-1] + max_horizon_delta
    valid_first_time = valid["open_time"].iloc[0]
    assert train_last_target_time <= valid_first_time, (
        f"Train target timestamp ({train_last_target_time}) crosses into valid start ({valid_first_time})"
    )

    # Valid's last target price timestamp must be <= test's first candle
    valid_last_target_time = valid["open_time"].iloc[-1] + max_horizon_delta
    test_first_time = test["open_time"].iloc[0]
    assert valid_last_target_time <= test_first_time, (
        f"Valid target timestamp ({valid_last_target_time}) crosses into test start ({test_first_time})"
    )
