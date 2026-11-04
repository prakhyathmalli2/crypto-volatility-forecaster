"""Unit tests for SignalIntelligenceEngine and model agreement logic."""
from __future__ import annotations

import pytest
from trading.signal_generator import SignalIntelligenceEngine, StructuredSignal


def test_signal_strong_buy():
    engine = SignalIntelligenceEngine()
    sig = engine.generate(
        predicted_direction=0.68,
        predicted_volatility=0.0090,
        reference_volatility=0.0090,  # ratio = 1.0 (calm)
        current_price=65000.0,
        atr=800.0,
        lstm_direction=0.66,
        transformer_direction=0.69,
        technical_regime="BULLISH",
        sentiment_score=0.25,
        check_event_risk=False,
    )
    assert isinstance(sig, StructuredSignal)
    assert sig.signal == "STRONG_BUY"
    assert sig.confidence >= 0.70
    assert sig.model_agreement >= 0.75
    assert sig.take_profit > sig.entry > sig.stop_loss
    assert sig.risk_reward == 2.0


def test_signal_reduce_on_volatility_shock():
    engine = SignalIntelligenceEngine()
    sig = engine.generate(
        predicted_direction=0.65,
        predicted_volatility=0.0350,
        reference_volatility=0.0100,  # ratio = 3.5x baseline (extreme shock!)
        current_price=65000.0,
        check_event_risk=False,
    )
    assert sig.signal == "REDUCE"
    assert sig.risk_level == "HIGH"
    assert "volatility" in sig.reason.lower()


def test_signal_hold_in_neutral_regime():
    engine = SignalIntelligenceEngine()
    sig = engine.generate(
        predicted_direction=0.50,
        predicted_volatility=0.0100,
        reference_volatility=0.0100,
        current_price=65000.0,
        check_event_risk=False,
    )
    assert sig.signal == "HOLD"
    assert sig.confidence <= 0.60


def test_signal_strong_sell():
    engine = SignalIntelligenceEngine()
    sig = engine.generate(
        predicted_direction=0.32,
        predicted_volatility=0.0120,
        reference_volatility=0.0100,
        current_price=65000.0,
        lstm_direction=0.30,
        transformer_direction=0.33,
        technical_regime="BEARISH",
        sentiment_score=-0.20,
        check_event_risk=False,
    )
    assert sig.signal == "STRONG_SELL"
    assert sig.model_agreement >= 0.75
    assert sig.take_profit < sig.entry < sig.stop_loss
