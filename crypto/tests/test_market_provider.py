"""Unit tests for MarketDataProvider and Technical Indicators Engine."""
from __future__ import annotations

import pytest
from market_data.provider import FallbackMarketProvider, MarketTick, Candle
from market_data.indicators import calculate_indicators


def test_market_provider_ticker():
    provider = FallbackMarketProvider()
    tick = provider.get_ticker("BTC/USDT")
    assert isinstance(tick, MarketTick)
    assert tick.price > 0
    assert tick.symbol == "BTC/USDT"
    assert isinstance(tick.change_24h_pct, float)


def test_market_provider_candles():
    provider = FallbackMarketProvider()
    candles = provider.get_ohlcv("BTC/USDT", timeframe="1h", limit=20)
    assert len(candles) > 0
    first = candles[0]
    assert isinstance(first, Candle)
    assert first.close > 0
    assert first.timestamp != ""


def test_technical_indicators_calculation():
    # Synthetic candle data
    sample_candles = [
        {"open": 60000 + i * 10, "high": 60100 + i * 10, "low": 59900 + i * 10, "close": 60050 + i * 10, "volume": 100 + i}
        for i in range(40)
    ]
    ind = calculate_indicators(sample_candles)
    assert "current" in ind
    assert "time_series" in ind

    curr = ind["current"]
    assert "rsi" in curr
    assert "atr" in curr
    assert "ema20" in curr
    assert curr["trend_regime"] in ("BULLISH", "BEARISH", "NEUTRAL")
