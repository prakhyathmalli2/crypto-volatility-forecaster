"""Deterministic simulated paper-trading execution broker.

Maintains:
- Multi-asset portfolio accounting (crypto, equities, forex)
- Position ledger with average entry price calculation
- Exact realized and unrealized P&L accounting
- Transaction fee drag (0.1% cost per trade)
- Strict order validation (prevents overselling and overdrafts)
- Audit log stored in data/paper_trading/paper_account.json
NEVER sends live exchange orders.
"""
from __future__ import annotations

import json
import os
import threading
from dataclasses import asdict, dataclass
from datetime import datetime, timedelta, timezone
from pathlib import Path
from typing import Any

from config import PAPER_DIR, TRANSACTION_COST


def to_ist_string(dt_or_iso: Any) -> str:
    """Format datetime or ISO timestamp into Indian Standard Time (IST)."""
    if not dt_or_iso:
        dt = datetime.now(timezone.utc)
    elif isinstance(dt_or_iso, str):
        try:
            if dt_or_iso.endswith("Z"):
                dt = datetime.fromisoformat(dt_or_iso.replace("Z", "+00:00"))
            else:
                dt = datetime.fromisoformat(dt_or_iso)
        except Exception:
            dt = datetime.now(timezone.utc)
    else:
        dt = dt_or_iso
    if dt.tzinfo is None:
        dt = dt.replace(tzinfo=timezone.utc)
    ist_tz = timezone(timedelta(hours=5, minutes=30))
    return dt.astimezone(ist_tz).strftime("%d %b %Y, %H:%M:%S IST")


_paper_ledger_lock = threading.Lock()


class PaperBroker:
    """Simulated multi-asset market order execution engine and portfolio ledger."""

    def __init__(self, initial_capital: float = 10000.0, path: Path | None = None):
        self.starting_cash = float(initial_capital)
        self.cash = float(initial_capital)
        self.positions: dict[str, dict[str, Any]] = {}
        self.realized_pnl = 0.0
        self.trade_history: list[dict[str, Any]] = []
        self.path = path or (PAPER_DIR / "paper_account.json")

        if self.path.exists():
            self._load_state()

    def reset(self, initial_capital: float | None = None) -> dict[str, Any]:
        """Reset the simulated paper trading ledger to initial cash balance."""
        if initial_capital is not None:
            self.starting_cash = float(initial_capital)
        self.cash = self.starting_cash
        self.positions = {}
        self.realized_pnl = 0.0
        self.trade_history = []
        self._save_state()
        return self.get_portfolio_summary()

    # Legacy compatibility properties for single-asset consumers
    @property
    def position(self) -> float:
        """Returns quantity of primary default asset (BTC/USDT) or first holding."""
        if "BTC/USDT" in self.positions:
            return float(self.positions["BTC/USDT"].get("quantity", 0.0))
        if self.positions:
            first_key = next(iter(self.positions))
            return float(self.positions[first_key].get("quantity", 0.0))
        return 0.0

    @position.setter
    def position(self, val: float):
        if "BTC/USDT" not in self.positions:
            self.positions["BTC/USDT"] = {"symbol": "BTC/USDT", "quantity": float(val), "avg_entry_price": 60000.0}
        else:
            self.positions["BTC/USDT"]["quantity"] = float(val)

    @property
    def avg_entry_price(self) -> float:
        if "BTC/USDT" in self.positions:
            return float(self.positions["BTC/USDT"].get("avg_entry_price", 0.0))
        if self.positions:
            first_key = next(iter(self.positions))
            return float(self.positions[first_key].get("avg_entry_price", 0.0))
        return 0.0

    @avg_entry_price.setter
    def avg_entry_price(self, val: float):
        if "BTC/USDT" in self.positions:
            self.positions["BTC/USDT"]["avg_entry_price"] = float(val)

    def _load_state(self):
        try:
            data = json.loads(self.path.read_text(encoding="utf-8"))
            self.starting_cash = float(data.get("starting_cash", self.starting_cash))
            self.cash = float(data.get("cash", self.cash))
            self.realized_pnl = float(data.get("realized_pnl", 0.0))
            self.trade_history = data.get("trade_history", [])

            # Load multi-asset positions dictionary
            raw_pos = data.get("positions", {})
            if isinstance(raw_pos, dict):
                self.positions = raw_pos
            elif isinstance(raw_pos, list):
                self.positions = {p["symbol"]: p for p in raw_pos if "symbol" in p}
            else:
                self.positions = {}

            # Fallback legacy scalar position migration
            if not self.positions and float(data.get("crypto_position", 0.0)) > 0:
                self.positions["BTC/USDT"] = {
                    "symbol": "BTC/USDT",
                    "quantity": float(data.get("crypto_position", 0.0)),
                    "avg_entry_price": float(data.get("avg_entry_price", 0.0)),
                }

            if self.cash <= 1.0 and not self.positions:
                self.cash = self.starting_cash
        except Exception as exc:
            print(f"[PaperBroker Warning] Could not load ledger state: {exc}")

    def _save_state(self):
        with _paper_ledger_lock:
            self.path.parent.mkdir(parents=True, exist_ok=True)
            state = self.get_portfolio_summary()
            state["trade_history"] = self.trade_history[-100:]  # Retain last 100 trades
            tmp_path = self.path.with_suffix(".tmp")
            tmp_path.write_text(json.dumps(state, indent=2), encoding="utf-8")
            os.replace(tmp_path, self.path)

    def reset_account(self, initial_capital: float | None = None) -> dict[str, Any]:
        """Reset paper trading account back to clean initial cash with zero positions."""
        init_cap = float(initial_capital) if initial_capital and initial_capital > 0 else self.starting_cash
        self.starting_cash = init_cap
        self.cash = init_cap
        self.positions = {}
        self.realized_pnl = 0.0
        self.trade_history = []
        self._save_state()
        return self.get_portfolio_summary()

    def get_portfolio_summary(
        self,
        mark_prices: dict[str, float] | float | None = None,
        mark_price: float | None = None,
    ) -> dict[str, Any]:
        """
        Compute live equity, unrealized PnL, and returns using single source of truth:
        Portfolio Value = Cash + sum(Quantity * Mark Price)
        """
        price_dict: dict[str, float] = {}
        if mark_price is not None:
            price_dict["BTC/USDT"] = float(mark_price)
        elif isinstance(mark_prices, (int, float)):
            price_dict["BTC/USDT"] = float(mark_prices)
        elif isinstance(mark_prices, dict):
            price_dict = dict(mark_prices)

        total_positions_value = 0.0
        total_unrealized_pnl = 0.0
        positions_list: list[dict[str, Any]] = []

        for sym, pos in list(self.positions.items()):
            qty = float(pos.get("quantity", 0.0))
            if qty <= 1e-8:
                continue
            entry_p = float(pos.get("avg_entry_price", 0.0))
            mark_p = float(price_dict.get(sym, entry_p)) if (sym in price_dict and price_dict[sym] > 0) else entry_p
            pos_val = qty * mark_p
            u_pnl = (mark_p - entry_p) * qty

            total_positions_value += pos_val
            total_unrealized_pnl += u_pnl

            ret_pct = ((mark_p - entry_p) / entry_p) * 100.0 if entry_p > 0 else 0.0

            positions_list.append({
                "symbol": sym,
                "quantity": round(qty, 6),
                "avg_entry_price": round(entry_p, 2),
                "mark_price": round(mark_p, 2),
                "position_value": round(pos_val, 2),
                "unrealized_pnl": round(u_pnl, 2),
                "return_pct": round(ret_pct, 2),
            })

        total_equity = self.cash + total_positions_value
        total_return_pct = ((total_equity / self.starting_cash) - 1.0) * 100.0 if self.starting_cash > 0 else 0.0

        # Legacy fields for backward compatibility
        primary_pos = self.positions.get("BTC/USDT", {})
        crypto_qty = float(primary_pos.get("quantity", 0.0))
        avg_entry = float(primary_pos.get("avg_entry_price", 0.0))
        mark_price_single = float(price_dict.get("BTC/USDT", avg_entry if avg_entry > 0 else 50000.0))

        return {
            "starting_cash": round(self.starting_cash, 2),
            "cash": round(self.cash, 2),
            "total_positions_value": round(total_positions_value, 2),
            "position_value": round(total_positions_value, 2),  # Compatibility alias
            "total_equity": round(total_equity, 2),
            "portfolio_value": round(total_equity, 2),
            "realized_pnl": round(self.realized_pnl, 2),
            "unrealized_pnl": round(total_unrealized_pnl, 2),
            "total_return_pct": round(total_return_pct, 2),
            "positions": positions_list,
            # Legacy fields
            "crypto_position": round(crypto_qty, 6),
            "avg_entry_price": round(avg_entry, 2),
            "mark_price": round(mark_price_single, 2),
        }

    def execute_market_order(
        self,
        side: str,
        quantity: float,
        execution_price: float,
        symbol: str = "BTC/USDT",
    ) -> dict[str, Any]:
        """
        Execute simulated market order with strict validations:
        - Rejects BUY if total cost exceeds available cash (NO silent downsizing).
        - Rejects SELL if requested quantity exceeds owned holdings (NO silent truncation).
        - Computes true realized PnL and updates average entry price.
        """
        qty = float(quantity)
        price = float(execution_price)
        side = side.upper().strip()
        sym = symbol.strip() if symbol else "BTC/USDT"

        if qty <= 0:
            raise ValueError(f"Order rejected: quantity must be strictly positive (got {qty})")
        if price <= 0:
            raise ValueError(f"Order rejected: execution price must be strictly positive (got {price})")

        fee = qty * price * TRANSACTION_COST
        now_dt = datetime.now(timezone.utc)
        now_utc = now_dt.isoformat()
        now_ist = to_ist_string(now_dt)

        currency_symbol = "₹" if (".NS" in sym or ".BO" in sym) else "$"

        if side == "BUY":
            total_required = (qty * price) + fee
            if total_required > self.cash:
                raise ValueError(
                    f"Insufficient cash: order requires {currency_symbol}{total_required:,.2f} "
                    f"({qty:g} @ {currency_symbol}{price:,.2f} + {currency_symbol}{fee:,.2f} fee), "
                    f"but available cash is only {currency_symbol}{self.cash:,.2f}."
                )

            # Update position
            existing = self.positions.get(sym, {"symbol": sym, "quantity": 0.0, "avg_entry_price": 0.0})
            old_qty = float(existing.get("quantity", 0.0))
            old_avg = float(existing.get("avg_entry_price", 0.0))

            new_qty = old_qty + qty
            new_avg = ((old_qty * old_avg) + (qty * price)) / new_qty

            self.positions[sym] = {
                "symbol": sym,
                "quantity": new_qty,
                "avg_entry_price": new_avg,
            }
            self.cash -= total_required

            trade_pnl = 0.0
            order_record = {
                "timestamp": now_utc,
                "timestamp_ist": now_ist,
                "symbol": sym,
                "side": "BUY",
                "quantity": round(qty, 6),
                "price": round(price, 2),
                "order_value": round(qty * price, 2),
                "fee": round(fee, 2),
                "realized_pnl": 0.0,
                "resulting_cash": round(self.cash, 2),
                "resulting_position": round(new_qty, 6),
            }

        elif side in ("SELL", "REDUCE"):
            existing = self.positions.get(sym)
            if not existing or float(existing.get("quantity", 0.0)) <= 1e-8:
                raise ValueError(f"Cannot sell {sym}: you do not hold any open position in this asset.")

            owned_qty = float(existing.get("quantity", 0.0))
            if qty > (owned_qty + 1e-7):
                raise ValueError(
                    f"Order rejected: cannot sell {qty:g} units of {sym}. "
                    f"You only hold {owned_qty:g} units."
                )

            # Cap tiny floating differences
            qty_to_sell = min(qty, owned_qty)
            avg_entry = float(existing.get("avg_entry_price", price))

            proceeds = (qty_to_sell * price) - fee
            trade_pnl = ((price - avg_entry) * qty_to_sell) - fee

            self.realized_pnl += trade_pnl
            self.cash += proceeds

            remaining_qty = owned_qty - qty_to_sell
            if remaining_qty <= 1e-8:
                del self.positions[sym]
                resulting_qty = 0.0
            else:
                self.positions[sym]["quantity"] = remaining_qty
                resulting_qty = remaining_qty

            order_record = {
                "timestamp": now_utc,
                "timestamp_ist": now_ist,
                "symbol": sym,
                "side": side,
                "quantity": round(qty_to_sell, 6),
                "price": round(price, 2),
                "order_value": round(qty_to_sell * price, 2),
                "fee": round(fee, 2),
                "realized_pnl": round(trade_pnl, 2),
                "resulting_cash": round(self.cash, 2),
                "resulting_position": round(resulting_qty, 6),
            }
        else:
            raise ValueError(f"Unsupported order side '{side}'. Must be BUY or SELL.")

        self.trade_history.append(order_record)
        self._save_state()
        summary = self.get_portfolio_summary(mark_prices={sym: price})
        summary["status"] = "EXECUTED"
        summary["last_order"] = order_record
        summary["execution_summary"] = (
            f"Executed {order_record['side']} {order_record['quantity']} {sym} @ {order_record['price']:,.2f} "
            f"(Fee: ${order_record['fee']:.2f}, Resulting Cash: ${order_record['resulting_cash']:,.2f})"
        )
        return summary
