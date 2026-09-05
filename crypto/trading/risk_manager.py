"""Paper-trading risk controls: ATR stop, 2R target, position sizing and kill switch."""
from __future__ import annotations

from dataclasses import dataclass

from config import MAX_DRAWDOWN_LIMIT, MAX_POSITION_RISK_PCT, STOP_LOSS_ATR_MULTIPLIER, TAKE_PROFIT_RR


@dataclass
class TradePlan:
    side: str
    entry_price: float
    quantity: float
    stop_price: float
    take_profit: float
    risk_amount: float


class RiskManager:
    def __init__(self, capital: float):
        self.capital = float(capital)
        self.peak_capital = float(capital)
        self.trading_enabled = True

    def position_size(self, entry_price: float, stop_price: float) -> float:
        distance = abs(entry_price - stop_price)
        if distance <= 0 or not self.trading_enabled:
            return 0.0
        return (self.capital * MAX_POSITION_RISK_PCT) / distance

    def plan(self, entry_price: float, atr: float, side: str = "long") -> TradePlan:
        distance = max(float(atr) * STOP_LOSS_ATR_MULTIPLIER, entry_price * 0.005)
        stop = entry_price - distance if side == "long" else entry_price + distance
        target = entry_price + TAKE_PROFIT_RR * distance if side == "long" else entry_price - TAKE_PROFIT_RR * distance
        qty = self.position_size(entry_price, stop)
        return TradePlan(side, entry_price, qty, stop, target, qty * distance)

    def update_capital(self, new_capital: float) -> bool:
        self.capital = float(new_capital)
        self.peak_capital = max(self.peak_capital, self.capital)
        drawdown = (self.peak_capital - self.capital) / max(self.peak_capital, 1e-9)
        if drawdown >= MAX_DRAWDOWN_LIMIT:
            self.trading_enabled = False
        return self.trading_enabled
