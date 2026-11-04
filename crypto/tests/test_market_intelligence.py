"""Tests for the Market Intelligence Platform:
- Multi-Asset Search
- Financial Fundamentals
- New Technical Indicators (SMA, Stochastic)
- News & Events Chart Markers & Impact
- AI Assistant with Fundamentals & Quick Actions
- Multi-Asset Support (Equities, Forex, Indices, Crypto)
"""
from __future__ import annotations

import pytest

from serving.inference_api import (
    search_market_assets,
    get_v1_fundamentals,
    get_v1_ticker,
    get_v1_candles,
)
from market_data.fundamentals import get_asset_fundamentals
from market_data.indicators import calculate_indicators
from market_data.provider import search_instruments, detect_asset_type, get_market_status
from news.news_service import get_news_service
from events.events_service import get_events_service
from llm.inference.tools import get_tool_registry
from llm.inference.llm_router import get_ai_router


def test_asset_detection_and_market_hours():
    # Asset type detection
    assert detect_asset_type("BTC/USDT") == "crypto"
    assert detect_asset_type("RELIANCE.NS") == "indian_stock"
    assert detect_asset_type("TCS.BO") == "indian_stock"
    assert detect_asset_type("^NSEI") == "index"
    assert detect_asset_type("^GSPC") == "index"
    assert detect_asset_type("USDINR=X") == "forex"
    assert detect_asset_type("AAPL") == "global_stock"

    # Market status checks
    crypto_status = get_market_status("BTC/USDT")
    assert crypto_status == "OPEN"


def test_search_instruments():
    # Search for Reliance
    rel_results = search_instruments("reliance")
    assert len(rel_results) > 0
    assert any("RELIANCE.NS" in r["symbol"] for r in rel_results)

    # Search for Nifty
    nifty_results = search_instruments("nifty")
    assert len(nifty_results) > 0
    assert any("^NSEI" in r["symbol"] for r in nifty_results)

    # Search for Bitcoin
    btc_results = search_instruments("btc")
    assert len(btc_results) > 0
    assert any("BTC/USDT" in r["symbol"] for r in btc_results)

    # API Endpoint search test
    data = search_market_assets(q="tata")
    assert isinstance(data, list)
    assert any("TCS" in r["name"] or "TATAMOTORS" in r["symbol"] for r in data)


def test_financial_fundamentals():
    # Test fundamentals provider for stock
    fund = get_asset_fundamentals("AAPL")
    assert fund["symbol"] == "AAPL"
    assert "metrics" in fund
    assert "company_name" in fund

    # Test fundamentals for crypto (fallback gracefully without crashing)
    btc_fund = get_asset_fundamentals("BTC/USDT")
    assert btc_fund["symbol"] == "BTC/USDT"
    assert "metrics" in btc_fund

    # Test API endpoint
    api_data = get_v1_fundamentals(symbol="AAPL")
    assert api_data["symbol"] == "AAPL"
    assert "metrics" in api_data


def test_sma_and_stochastic_indicators():
    import numpy as np
    n = 60
    candles = []
    for i in range(n):
        p = 100.0 + float(i) * 0.5 + np.sin(i) * 2.0
        candles.append({
            "timestamp": 1700000000 + i * 3600,
            "open": p - 0.2,
            "high": p + 1.0,
            "low": p - 1.0,
            "close": p,
            "volume": 1000.0 + float(i) * 10.0,
        })

    result = calculate_indicators(candles)
    curr = result["current"]
    ts = result["time_series"]

    # Check SMA
    assert "sma20" in curr
    assert "sma50" in curr
    assert "sma20" in ts

    # Check Stochastic
    assert "stoch_k" in curr
    assert "stoch_d" in curr
    assert 0.0 <= curr["stoch_k"] <= 100.0
    assert 0.0 <= curr["stoch_d"] <= 100.0


def test_news_and_events_chart_markers():
    news_svc = get_news_service()
    articles = news_svc.fetch_articles()
    assert len(articles) > 0
    for art in articles:
        assert hasattr(art, "chart_time")
        assert hasattr(art, "impact")
        assert art.chart_time is not None
        assert art.impact in ["Potentially positive", "Potentially negative", "Neutral", "Neutral/Informational"]

    events_svc = get_events_service()
    events = events_svc.get_upcoming_events()
    assert len(events) > 0
    for ev in events:
        assert "chart_time" in ev
        assert ev["chart_time"] is not None
        assert ev["importance"] in ["HIGH", "MEDIUM", "LOW", "HIGH IMPACT EVENT", "CRITICAL"]


def test_ai_chatbot_fundamentals_and_quick_actions():
    router = get_ai_router()

    # Quick action: Why is this moving?
    resp1 = router.answer_query("Why is this moving?", symbol="RELIANCE.NS")
    assert "RELIANCE.NS" in resp1["response"] or "Reliance" in resp1["response"] or "LIVE TELEMETRY" in resp1["response"]

    # Quick action: Explain RSI
    resp2 = router.answer_query("Explain RSI", symbol="BTC/USDT")
    assert "RSI" in resp2["response"]

    # Quick action: Explain the ML prediction
    resp3 = router.answer_query("Explain the ML prediction", symbol="BTC/USDT")
    assert "MODEL OUTPUT" in resp3["response"] or "prediction" in resp3["response"].lower()

    # Tool registry fundamentals test
    tools = get_tool_registry()
    fund = tools.get_fundamentals("AAPL")
    assert "symbol" in fund
    assert fund["symbol"] == "AAPL"


def test_api_multi_asset_ticker_and_candles():
    # Test BTC ticker
    btc_data = get_v1_ticker(symbol="BTC/USDT")
    assert btc_data["symbol"] == "BTC/USDT"
    assert "market_status" in btc_data

    # Test BTC candles
    candles = get_v1_candles(symbol="BTC/USDT", timeframe="1h", limit=10)
    assert len(candles) > 0
    assert "close" in candles[0]
