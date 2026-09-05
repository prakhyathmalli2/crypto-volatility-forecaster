"""Deterministic paper-trading engine. Never sends a live order."""
from __future__ import annotations

import json
from datetime import datetime, timezone
from pathlib import Path

from config import PAPER_DIR, TRANSACTION_COST


class PaperBroker:
    def __init__(self, capital: float, path: Path | None = None):
        self.starting_cash = float(capital)
        self.cash = float(capital)
        self.position = 0.0
        self.avg_entry = 0.0
        self.realized_pnl = 0.0
        self.path = path or (PAPER_DIR / "paper_account.json")
        if self.path.exists():
            self._load()

    def _load(self):
        data = json.loads(self.path.read_text(encoding="utf-8"))
        self.starting_cash = data.get("starting_cash", self.starting_cash)
        self.cash = data.get("cash", self.cash)
        self.position = data.get("position", self.position)
        self.avg_entry = data.get("avg_entry", self.avg_entry)
        self.realized_pnl = data.get("realized_pnl", self.realized_pnl)

    def _save(self):
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self.path.write_text(json.dumps(self.snapshot(), indent=2), encoding="utf-8")

    def snapshot(self, mark_price: float | None = None):
        equity = self.cash + (self.position * mark_price if mark_price is not None else 0.0)
        return {"starting_cash": self.starting_cash, "cash": self.cash, "position": self.position,
                "avg_entry": self.avg_entry, "realized_pnl": self.realized_pnl, "equity": equity}

    def market_order(self, side: str, quantity: float, price: float):
        quantity = float(quantity)
        price = float(price)
        fee = abs(quantity * price) * TRANSACTION_COST
        if side == "BUY":
            cost = quantity * price + fee
            if cost > self.cash:
                raise ValueError("Insufficient paper cash")
            new_pos = self.position + quantity
            self.avg_entry = ((self.position * self.avg_entry) + quantity * price) / max(new_pos, 1e-9)
            self.position = new_pos
            self.cash -= cost
        elif side in ("SELL", "REDUCE"):
            quantity = min(quantity, self.position)
            proceeds = quantity * price - fee
            self.realized_pnl += quantity * (price - self.avg_entry) - fee
            self.position -= quantity
            self.cash += proceeds
            if self.position <= 1e-12:
                self.position, self.avg_entry = 0.0, 0.0
        else:
            raise ValueError("side must be BUY or SELL/REDUCE")
        self._save()
        return self.snapshot(price)
