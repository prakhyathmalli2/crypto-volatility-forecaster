"""Unit tests for AI Assistant tool calling registry and deterministic router."""
from __future__ import annotations

import pytest
from llm.inference.tools import MarketToolRegistry
from llm.inference.llm_router import AIRouter


def test_tool_registry_executions():
    tools = MarketToolRegistry()

    # 1. Live price
    tick = tools.get_live_price("BTC/USDT")
    assert tick["price"] > 0

    # 2. Indicators
    ind = tools.get_indicators("BTC/USDT")
    assert "rsi" in ind

    # 3. News
    news = tools.get_latest_news("BTC/USDT", limit=2)
    assert len(news) > 0

    # 4. Events
    events = tools.get_market_events("BTC/USDT")
    assert len(events) > 0

    # 5. Signal
    sig = tools.get_current_signal("BTC/USDT")
    assert "signal" in sig
    assert "entry" in sig

    # 6. Risk metrics
    risk = tools.get_risk_metrics("BTC/USDT")
    assert "stop_loss" in risk


def test_ai_router_deterministic_query():
    router = AIRouter()

    # Test "Why is BTC moving?"
    res1 = router.answer_query("Why is BTC moving right now?", symbol="BTC/USDT")
    assert "BTC/USDT" in res1["symbol"]
    assert "LIVE TELEMETRY" in res1["response"]
    assert "MODEL OUTPUT" in res1["response"]
    assert res1["provider"] == "deterministic_reasoning_engine"

    # Test Signal Inquiry
    res2 = router.answer_query("What is the active trading signal?", symbol="BTC/USDT")
    assert "Trading Signal" in res2["response"]
    assert "Stop-Loss" in res2["response"]
