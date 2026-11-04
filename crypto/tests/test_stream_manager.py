"""
Unit tests for RealTimeStreamManager and live candle streaming.
"""
from __future__ import annotations

import pytest
import time
from market_data.stream_manager import (
    RealTimeStreamManager,
    BINANCE_TIMEFRAME_MAP,
    TIMEFRAME_SECONDS,
    normalize_crypto_symbol_for_binance,
)
from market_data.provider import detect_asset_type, get_market_status


def test_timeframe_mappings():
    """Verify canonical timeframe to seconds and Binance interval mapping."""
    assert BINANCE_TIMEFRAME_MAP["1m"] == "1m"
    assert BINANCE_TIMEFRAME_MAP["5m"] == "5m"
    assert BINANCE_TIMEFRAME_MAP["15m"] == "15m"
    assert BINANCE_TIMEFRAME_MAP["1h"] == "1h"
    assert BINANCE_TIMEFRAME_MAP["4h"] == "4h"
    assert BINANCE_TIMEFRAME_MAP["1d"] == "1d"
    assert BINANCE_TIMEFRAME_MAP["1w"] == "1w"

    assert TIMEFRAME_SECONDS["1m"] == 60
    assert TIMEFRAME_SECONDS["5m"] == 300
    assert TIMEFRAME_SECONDS["15m"] == 900
    assert TIMEFRAME_SECONDS["1h"] == 3600
    assert TIMEFRAME_SECONDS["1d"] == 86400


def test_crypto_symbol_normalization():
    """Verify symbol normalization for Binance WebSocket topics."""
    assert normalize_crypto_symbol_for_binance("BTC/USDT") == "btcusdt"
    assert normalize_crypto_symbol_for_binance("ETH/USDT") == "ethusdt"
    assert normalize_crypto_symbol_for_binance("SOL-USDT") == "solusdt"


def test_asset_type_detection():
    """Verify crypto vs non-crypto classification."""
    assert detect_asset_type("BTC/USDT") == "crypto"
    assert detect_asset_type("ETH/USDT") == "crypto"
    assert detect_asset_type("^NSEI") in ["index", "indian_stock", "stock"]
    assert detect_asset_type("RELIANCE.NS") == "indian_stock"
    assert detect_asset_type("AAPL") in ["stock", "global_stock"]
    assert detect_asset_type("USDINR=X") == "forex"


def test_market_status_non_crypto():
    """Verify market status correctly flags market hours for Indian stocks."""
    status = get_market_status("^NSEI")
    assert status in ["OPEN", "CLOSED", "WEEKEND"]


def test_channel_key_generation():
    """Verify stream manager channel keys are normalized."""
    manager = RealTimeStreamManager()
    assert manager._get_channel_key("btc/usdt", "1m") == "BTC/USDT:1m"
    assert manager._get_channel_key("BTC/USDT", "5m") == "BTC/USDT:5m"
    assert manager._get_channel_key("^NSEI", "1h") == "^NSEI:1h"
    assert manager._get_channel_key("AAPL", "invalid") == "AAPL:1h"
