"""Quantitative risk management and capital preservation controls.

Implements:
- Volatility / ATR-based stop loss placement
- 2:1 Reward-to-Risk (2R) take-profit targets
- Dynamic position sizing respecting the 2% maximum portfolio risk rule
- Hard drawdown circuit breaker (halts trading if drawdown >= 20%)
"""
from __future__ import annotations

from dataclasses import asdict, dataclass

from config import (
    INITIAL_CAPITAL,
    MAX_DRAWDOWN_LIMIT,
    MAX_POSITION_RISK_PCT,
    STOP_LOSS_ATR_MULTIPLIER,
    TAKE_PROFIT_RR,
)


@dataclass
class TradePlan:
    side: str
    entry_price: float
    quantity: float
    stop_loss: float
    take_profit: float
    risk_amount_usd: float
    reward_amount_usd: float
    risk_reward_ratio: float
    max_position_notional: float

    def to_dict(self) -> dict:
        return asdict(self)


class RiskManager:
    """Risk control system enforcing position sizing limits and circuit breakers."""

    def __init__(self, capital: float = INITIAL_CAPITAL):
        self.capital = float(capital)
        self.peak_capital = float(capital)
        self.trading_enabled = True
        self.current_drawdown = 0.0

    def calculate_position_size(self, entry_price: float, stop_distance: float) -> float:
        """Size position such that a stop-out loses at most MAX_POSITION_RISK_PCT of capital."""
        if stop_distance <= 0 or not self.trading_enabled:
            return 0.0
        max_dollar_risk = self.capital * MAX_POSITION_RISK_PCT
        raw_quantity = max_dollar_risk / stop_distance
        # Ensure notional position value does not exceed available cash
        max_notional_qty = (self.capital * 0.95) / entry_price
        return float(min(raw_quantity, max_notional_qty))

    def plan_trade(self, entry_price: float, atr: float, side: str = "BUY") -> TradePlan:
        """Create structured execution trade plan with strict stop and target."""
        price = float(entry_price)
        # Minimum stop distance of 0.5% price to avoid microscopic stops on low ATR
        distance = max(float(atr) * STOP_LOSS_ATR_MULTIPLIER, price * 0.005)

        if side == "BUY":
            stop = price - distance
            target = price + (TAKE_PROFIT_RR * distance)
        else:
            stop = price + distance
            target = price - (TAKE_PROFIT_RR * distance)

        quantity = self.calculate_position_size(price, distance)
        risk_usd = quantity * distance
        reward_usd = quantity * (distance * TAKE_PROFIT_RR)

        return TradePlan(
            side=side,
            entry_price=round(price, 2),
            quantity=round(quantity, 6),
            stop_loss=round(stop, 2),
            take_profit=round(target, 2),
            risk_amount_usd=round(risk_usd, 2),
            reward_amount_usd=round(reward_usd, 2),
            risk_reward_ratio=float(TAKE_PROFIT_RR),
            max_position_notional=round(quantity * price, 2),
        )

    def update_portfolio_state(self, current_equity: float) -> bool:
        """
        Track drawdowns and trigger circuit breaker if threshold is breached.
        Returns True if trading is active, False if circuit breaker tripped.
        """
        self.capital = float(current_equity)
        if self.capital > self.peak_capital:
            self.peak_capital = self.capital

        self.current_drawdown = (self.peak_capital - self.capital) / max(self.peak_capital, 1.0)
        if self.current_drawdown >= MAX_DRAWDOWN_LIMIT:
            self.trading_enabled = False  # Trip circuit breaker
        return self.trading_enabled
