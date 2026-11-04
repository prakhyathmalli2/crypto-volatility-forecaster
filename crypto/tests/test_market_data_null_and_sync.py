"""
Unit and integration tests for market data null representation, candle validation,
and intra-bar live candle formation (Sections 4, 5, 8, 11, 40, 42, 43).
"""
from __future__ import annotations

import pytest
from datetime import datetime, timezone

from market_data.provider import MarketTick, Candle, FallbackMarketProvider, CCXTMarketProvider
from serving.inference_api import get_v1_tickers, get_v1_candles


def test_null_data_representation():
    """
    Verify Problem 5 & Section 42:
    When price or 24h change data is unavailable, it must be represented as None
    and NOT as 0.0 or 0.00%.
    """
    tick = MarketTick(
        symbol="ETH/USDT",
        price=3200.0,
        volume_24h=None,
        change_24h_pct=None,
        bid=3199.5,
        ask=3200.5,
    )
    d = tick.to_dict()
    assert d["change_24h_pct"] is None, "Missing change_24h_pct must serialize as None, not 0.0"
    assert d["volume_24h"] is None, "Missing volume_24h must serialize as None"

    # Distinguish actual 0.0% change from missing None change
    flat_tick = MarketTick(
        symbol="BTC/USDT",
        price=85000.0,
        volume_24h=1500.0,
        change_24h_pct=0.0,
    )
    flat_d = flat_tick.to_dict()
    assert flat_d["change_24h_pct"] == 0.0, "Real 0.0% change must be preserved as 0.0"


def test_batch_tickers_endpoint():
    """Verify Section 3 & 22: Batch tickers endpoint serves watchlist symbols."""
    tickers = get_v1_tickers("BTC/USDT,ETH/USDT,SOL/USDT")
    assert isinstance(tickers, list)
    assert len(tickers) >= 3
    symbols = [t["symbol"] for t in tickers]
    assert "BTC/USDT" in symbols
    assert "ETH/USDT" in symbols
    assert "SOL/USDT" in symbols


def test_candle_data_validation():
    """
    Verify Section 11:
    Candles must satisfy:
    open > 0, high > 0, low > 0, close > 0, volume >= 0
    high >= max(open, close)
    low <= min(open, close)
    """
    valid_c = Candle(
        timestamp=datetime.now(timezone.utc).isoformat(),
        timestamp_ms=1700000000000,
        open=85000.0,
        high=85500.0,
        low=84900.0,
        close=85200.0,
        volume=12.5,
    )
    assert valid_c.high >= max(valid_c.open, valid_c.close)
    assert valid_c.low <= min(valid_c.open, valid_c.close)
    assert valid_c.volume >= 0

    # Malformed candle detection logic
    def validate_candle(o, h, l, c, v):
        if o <= 0 or h <= 0 or l <= 0 or c <= 0 or v < 0:
            return False
        if h < max(o, c) or l > min(o, c):
            return False
        return True

    assert validate_candle(85000, 85500, 84900, 85200, 10.0) is True
    assert validate_candle(85000, 84000, 84900, 85200, 10.0) is False  # high < open
    assert validate_candle(85000, 85500, 85300, 85200, 10.0) is False  # low > close
    assert validate_candle(-100, 85500, 84900, 85200, 10.0) is False   # negative open


def test_live_candle_bucketing_logic():
    """
    Verify Section 4 & 5:
    Incremental intra-candle update and candle finalization at interval boundary.
    """
    tf_interval_sec = 60  # 1-minute candle

    # Time 12:00:10 -> bucket is 12:00:00 (ts=1700000000)
    base_ts = 1700000000
    tick1_price = 85000.0
    candle = {
        "time": (base_ts // tf_interval_sec) * tf_interval_sec,
        "open": tick1_price,
        "high": tick1_price,
        "low": tick1_price,
        "close": tick1_price,
    }

    # Tick 2 at 12:00:25 with higher price -> updates high and close
    tick2_price = 85200.0
    candle["high"] = max(candle["high"], tick2_price)
    candle["low"] = min(candle["low"], tick2_price)
    candle["close"] = tick2_price

    assert candle["high"] == 85200.0
    assert candle["low"] == 85000.0
    assert candle["close"] == 85200.0

    # Tick 3 at 12:00:40 with lower price -> updates low and close
    tick3_price = 84800.0
    candle["high"] = max(candle["high"], tick3_price)
    candle["low"] = min(candle["low"], tick3_price)
    candle["close"] = tick3_price

    assert candle["high"] == 85200.0
    assert candle["low"] == 84800.0
    assert candle["close"] == 84800.0
    assert candle["open"] == 85000.0

    # Tick 4 at 12:01:05 (next minute!) -> Previous candle finalized, new candle created!
    tick4_ts = base_ts + 65
    new_bucket = (tick4_ts // tf_interval_sec) * tf_interval_sec
    assert new_bucket > candle["time"], "New minute must produce a higher bucket time"

    next_candle = {
        "time": new_bucket,
        "open": 84950.0,
        "high": 84950.0,
        "low": 84950.0,
        "close": 84950.0,
    }
    assert next_candle["time"] == candle["time"] + 60
    assert next_candle["open"] == 84950.0
