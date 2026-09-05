"""Explainable signal engine using direction + volatility + confidence."""
from __future__ import annotations

from dataclasses import dataclass
import numpy as np

from config import SIGNAL_LOOKBACK


@dataclass
class Signal:
    action: str
    confidence: float
    risk_level: str
    reason: str


class SignalGenerator:
    def __init__(self, lookback: int = SIGNAL_LOOKBACK):
        self.vol_history: list[float] = []
        self.lookback = lookback

    def generate(self, predicted_direction: float, predicted_volatility: float, reference_volatility: float | None = None) -> Signal:
        self.vol_history.append(float(predicted_volatility))
        hist = self.vol_history[-self.lookback:]
        ref = reference_volatility if reference_volatility is not None else float(np.median(hist))
        vol_ratio = predicted_volatility / max(ref, 1e-9)
        direction_score = abs(predicted_direction - 0.5) * 2
        confidence = float(np.clip(0.55 * direction_score + 0.45 * min(abs(vol_ratio - 1) + 0.35, 1), 0, 1))

        if vol_ratio >= 1.8:
            action, risk = "REDUCE", "HIGH"
            reason = "Forecast volatility is materially above its recent baseline."
        elif predicted_direction >= 0.55 and vol_ratio <= 1.5:
            action, risk = "BUY", "LOW" if vol_ratio < 1.0 else "MEDIUM"
            reason = "Positive direction probability with acceptable forecast risk."
        elif predicted_direction <= 0.45:
            action, risk = "SELL", "MEDIUM" if vol_ratio < 1.5 else "HIGH"
            reason = "Negative direction probability dominates."
        else:
            action, risk = "HOLD", "MEDIUM"
            reason = "Direction edge is weak relative to model uncertainty."
        return Signal(action, confidence, risk, reason)
