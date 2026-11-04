"""Unit tests for signal generator, risk manager, and paper broker."""
from __future__ import annotations

import pytest

from trading.execution import PaperBroker
from trading.risk_manager import RiskManager
from trading.signal_generator import SignalGenerator


def test_signal_generator_rules():
    gen = SignalGenerator()

    # Rule 1: Extreme Volatility Shock -> REDUCE
    sig_shock = gen.generate(predicted_direction=0.65, predicted_volatility=0.08, reference_volatility=0.03)
    assert sig_shock.action == "REDUCE"
    assert sig_shock.risk_level == "HIGH"

    # Rule 2: High direction prob + acceptable volatility -> BUY
    sig_buy = gen.generate(predicted_direction=0.60, predicted_volatility=0.02, reference_volatility=0.02)
    assert sig_buy.action == "BUY"

    # Rule 3: Downward direction prob -> SELL
    sig_sell = gen.generate(predicted_direction=0.40, predicted_volatility=0.02, reference_volatility=0.02)
    assert sig_sell.action == "SELL"

    # Rule 4: Neutral direction prob -> HOLD
    sig_hold = gen.generate(predicted_direction=0.50, predicted_volatility=0.02, reference_volatility=0.02)
    assert sig_hold.action == "HOLD"


def test_risk_manager_sizing_and_circuit_breaker():
    rm = RiskManager(capital=10000.0)

    # 1. Plan trade: 2% risk on 10000 = $200 max loss
    plan = rm.plan_trade(entry_price=50000.0, atr=1000.0, side="BUY")
    assert plan.quantity > 0
    assert plan.stop_loss < 50000.0
    assert plan.take_profit > 50000.0
    # Stop distance = 2.0 * ATR = 2000. Max risk = $200. Qty = 200 / 2000 = 0.1 BTC
    assert abs(plan.risk_amount_usd - 200.0) < 1.0

    # 2. Circuit breaker triggers on 20% drawdown
    active = rm.update_portfolio_state(current_equity=7900.0)  # 21% loss from 10000
    assert not active
    assert not rm.trading_enabled


def test_paper_broker_ledger(tmp_path):
    ledger_path = tmp_path / "test_account.json"
    broker = PaperBroker(initial_capital=10000.0, path=ledger_path)

    # Execute BUY
    summary = broker.execute_market_order("BUY", quantity=0.1, execution_price=50000.0)
    assert summary["crypto_position"] == 0.1
    assert summary["cash"] < 5000.0  # $5000 + fee deducted
    assert summary["avg_entry_price"] == 50000.0

    # Execute profitable SELL
    summary2 = broker.execute_market_order("SELL", quantity=0.1, execution_price=55000.0)
    assert summary2["crypto_position"] == 0.0
    assert summary2["realized_pnl"] > 0.0
    assert summary2["cash"] > 10000.0  # Profit recorded
