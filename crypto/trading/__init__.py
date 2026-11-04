"""Quantitative trading and risk management engine package."""
from trading.execution import PaperBroker
from trading.principles import get_contextual_principle
from trading.risk_manager import RiskManager, TradePlan
from trading.signal_generator import (
    Signal,
    SignalGenerator,
    SignalIntelligenceEngine,
    StructuredSignal,
)

__all__ = [
    "PaperBroker",
    "RiskManager",
    "TradePlan",
    "Signal",
    "SignalGenerator",
    "SignalIntelligenceEngine",
    "StructuredSignal",
    "get_contextual_principle",
]
