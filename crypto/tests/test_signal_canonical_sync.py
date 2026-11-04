"""
Unit and integration tests for canonical signal synchronization (Sections 12, 13, 14, 16, 17, 41).
Verifies that:
1. Signal engine creates a structured canonical signal object with signal_id, data_timestamp, setup_type, etc.
2. Serving layer and AI Assistant tools consume the identical canonical signal.
3. Chatbot router reports the exact same signal_id and action as the serving engine (no BUY/SELL contradictions).
4. Expired signals are properly flagged.
"""
from __future__ import annotations

import time
from datetime import datetime, timezone, timedelta
import pytest

from trading.signal_generator import SignalIntelligenceEngine, StructuredSignal
from serving.inference_api import get_v1_signal
from llm.inference.tools import MarketToolRegistry
from llm.inference.llm_router import AIRouter


def test_canonical_signal_structure():
    """Verify canonical signal object contains all required fields (Section 12)."""
    engine = SignalIntelligenceEngine()
    sig = engine.generate(
        predicted_direction=0.44,
        predicted_volatility=0.012,
        current_price=85844.98,
        atr=515.07,
        symbol="BTC/USDT",
        lstm_direction=0.42,
        transformer_direction=0.45,
        technical_regime="BEARISH",
        sentiment_score=-0.10,
        check_event_risk=False,
    )

    d = sig.to_dict()
    assert "signal_id" in d and len(d["signal_id"]) > 0
    assert d["symbol"] == "BTC/USDT"
    assert d["signal"] in ["SELL", "STRONG_SELL"]
    assert d["setup_type"] == "SHORT_SETUP"
    assert d["execution_status"] in ["WAITING_FOR_ENTRY", "ACTIVE_IN_ZONE"]
    assert d["risk_reward"] == 2.0
    assert d["take_profit"] < d["entry"] < d["stop_loss"]
    assert "data_timestamp" in d
    assert "generated_at" in d
    assert "expires_at" in d
    assert d["is_expired"] is False


def test_serving_and_ai_tools_canonical_sync():
    """
    Verify Problem 2 & Section 41:
    Serving layer get_v1_signal() and LLM tools get_current_signal() must share the
    EXACT SAME canonical signal_id and signal action.
    """
    # Fetch signal from serving layer
    api_signal = get_v1_signal(symbol="BTC/USDT", timeframe="1h")
    assert "signal_id" in api_signal
    assert "signal" in api_signal

    # Fetch signal through AI assistant tool registry
    tools = MarketToolRegistry()
    tool_signal = tools.get_current_signal(symbol="BTC/USDT", timeframe="1h")

    # Canonical equality assertion: Identical signal ID and identical action
    assert tool_signal["signal_id"] == api_signal["signal_id"], (
        f"DESYNC DETECTED: tool signal_id {tool_signal['signal_id']} != api signal_id {api_signal['signal_id']}"
    )
    assert tool_signal["signal"] == api_signal["signal"], (
        f"CONTRADICTION DETECTED: tool signal {tool_signal['signal']} != api signal {api_signal['signal']}"
    )
    assert tool_signal["confidence"] == api_signal["confidence"]
    assert tool_signal["entry"] == api_signal["entry"]
    assert tool_signal["stop_loss"] == api_signal["stop_loss"]
    assert tool_signal["take_profit"] == api_signal["take_profit"]


def test_ai_chatbot_explanation_uses_canonical_signal():
    """
    Verify Section 16 & 17:
    The AI Chatbot explains the canonical signal and NEVER invents a conflicting BUY/SELL.
    """
    router = AIRouter()
    res = router.answer_query("What is the active trading signal?", symbol="BTC/USDT")
    response_text = res["response"]

    api_signal = get_v1_signal(symbol="BTC/USDT", timeframe="1h")
    canonical_action = api_signal["signal"]
    canonical_id = api_signal["signal_id"]

    # Chatbot response must reference the canonical signal action
    assert canonical_action in response_text, (
        f"Chatbot did not report canonical signal {canonical_action}! Response: {response_text}"
    )
    assert canonical_id[:8] in response_text, (
        f"Chatbot did not report canonical signal_id prefix {canonical_id[:8]}! Response: {response_text}"
    )

    # Chatbot must not contradict itself by saying BUY if canonical is SELL, or SELL if canonical is BUY
    if canonical_action in ["SELL", "STRONG_SELL"]:
        assert "Active Trading Signal: BUY" not in response_text
    elif canonical_action in ["BUY", "STRONG_BUY"]:
        assert "Active Trading Signal: SELL" not in response_text


def test_expired_signal_handling():
    """Verify Section 15: Expired signals are not treated as active."""
    engine = SignalIntelligenceEngine()
    sig = engine.generate(
        predicted_direction=0.60,
        predicted_volatility=0.01,
        current_price=65000.0,
        symbol="BTC/USDT",
        check_event_risk=False,
    )
    # Manually backdate expiration
    past_time = datetime.now(timezone.utc) - timedelta(minutes=10)
    sig.expires_at = past_time.isoformat()

    d = sig.to_dict()
    assert d["is_expired"] is True
