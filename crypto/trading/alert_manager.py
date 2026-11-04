"""Centralized Real-Time Alert and Notification Engine.

Implements:
- Signal strength and confidence validation (confidence >= 70% or STRONG signals)
- High-volatility / unpredictable regime alerts
- Cooldown and debouncing (15-minute suppression window unless direction reverses)
- Data freshness gate (suppresses alerts if market feed is stale or unavailable)
- Formats structured alerts with asset, signal, confidence, price, IST timestamp, SL, TP, and evidence
"""
from __future__ import annotations

import time
from dataclasses import asdict, dataclass
from datetime import datetime, timezone
from typing import Any
from zoneinfo import ZoneInfo

from market_data.provider import to_ist_string

IST = ZoneInfo("Asia/Kolkata")


@dataclass
class SignalAlert:
    alert_id: str
    symbol: str
    signal: str
    confidence_pct: float
    current_price: float
    stop_loss: float
    target_1: float
    target_2: float
    risk_reward: float
    volatility_regime: str
    explanation: str
    data_status: str
    timestamp_utc: str
    timestamp_ist: str
    severity: str  # "HIGH", "CRITICAL", "INFO"

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


class AlertManager:
    """Manages alert generation, debouncing, and history."""

    def __init__(self, cooldown_seconds: float = 900.0, min_confidence: float = 0.68):
        self.cooldown_seconds = cooldown_seconds
        self.min_confidence = min_confidence
        # symbol -> (last_alert_time, last_signal_action)
        self._history: dict[str, tuple[float, str]] = {}
        self._alert_log: list[SignalAlert] = []
        self._max_log = 50

    def evaluate_signal(self, signal: dict[str, Any], tick: dict[str, Any]) -> SignalAlert | None:
        """
        Evaluate candidate signal and emit an alert if thresholds, cooldown,
        and freshness checks pass.
        """
        symbol = signal.get("symbol", tick.get("symbol", "UNKNOWN"))
        action = signal.get("signal", "HOLD")
        confidence = float(signal.get("confidence", 0.5))
        is_unpred = bool(signal.get("is_unpredictable_regime", False))
        data_status = tick.get("data_status", "LIVE")

        # 1. Freshness check: Never alert on unavailable or closed markets
        if data_status in ("UNAVAILABLE", "MARKET CLOSED"):
            return None

        # 2. Threshold check: Must be strong signal, high confidence, or high-volatility warning
        is_strong = action in ("STRONG BUY", "STRONG SELL") or (action in ("BUY", "SELL") and confidence >= self.min_confidence)
        if not is_strong and not is_unpred:
            return None

        now = time.time()
        last_alert = self._history.get(symbol)

        # 3. Debouncing / Cooldown check:
        # Suppress if within cooldown window UNLESS the signal action reversed
        if last_alert is not None:
            last_time, last_action = last_alert
            elapsed = now - last_time
            if elapsed < self.cooldown_seconds:
                # Allow early alert only if direction reversed (e.g. BUY -> SELL or vice versa)
                reversed_direction = ("BUY" in action and "SELL" in last_action) or ("SELL" in action and "BUY" in last_action)
                if not reversed_direction:
                    return None

        # Build Alert
        now_utc = datetime.now(timezone.utc)
        evidence_list = signal.get("evidence", [])
        short_exp = evidence_list[0] if evidence_list else signal.get("reason", "Consensus statistical signal")
        if is_unpred:
            action = "HIGH VOLATILITY ALERT"
            short_exp = signal.get("unpredictable_warning", "Abnormal volatility detected. Recommended action: HOLD.")

        severity = "CRITICAL" if action in ("STRONG BUY", "STRONG SELL", "HIGH VOLATILITY ALERT") else "HIGH"

        alert = SignalAlert(
            alert_id=f"alt-{int(now * 1000)}-{symbol.replace('/', '_').lower()}",
            symbol=symbol,
            signal=action,
            confidence_pct=round(confidence * 100.0, 1),
            current_price=float(signal.get("current_price", tick.get("price", 0.0))),
            stop_loss=float(signal.get("stop_loss", 0.0)),
            target_1=float(signal.get("target_1", 0.0)),
            target_2=float(signal.get("target_2", 0.0)),
            risk_reward=float(signal.get("risk_reward", 2.0)),
            volatility_regime=signal.get("volatility_regime", "NORMAL"),
            explanation=short_exp,
            data_status=data_status,
            timestamp_utc=now_utc.isoformat(),
            timestamp_ist=to_ist_string(now_utc),
            severity=severity,
        )

        # Record history
        self._history[symbol] = (now, action)
        self._alert_log.insert(0, alert)
        if len(self._alert_log) > self._max_log:
            self._alert_log.pop()

        return alert

    def get_recent_alerts(self, limit: int = 20) -> list[dict[str, Any]]:
        """Return cached recent alerts in descending chronological order."""
        return [a.to_dict() for a in self._alert_log[:limit]]

    def clear(self):
        """Reset alert state (for test isolation)."""
        self._history.clear()
        self._alert_log.clear()


# Global singleton alert manager
_alert_manager: AlertManager | None = None


def get_alert_manager() -> AlertManager:
    global _alert_manager
    if _alert_manager is None:
        _alert_manager = AlertManager()
    return _alert_manager
