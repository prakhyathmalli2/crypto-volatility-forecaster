"""Explainable quantitative trading signal intelligence engine.

Integrates multi-model forecasts (GARCH, LSTM, Transformer), technical trend regimes,
sentiment polarity, and macro event risk into structured, auditable trade decisions.
Supports STRONG BUY, BUY, HOLD / NO TRADE, SELL, STRONG SELL, and REDUCE regimes.
"""
from __future__ import annotations

import uuid
from dataclasses import asdict, dataclass, field
from datetime import datetime, timedelta, timezone
from typing import Any

import numpy as np


class SignalAction(str):
    """Canonical signal action token representing trading decisions."""
    def __eq__(self, other: object) -> bool:
        if not isinstance(other, str):
            return False
        s_self = str(self).strip().replace(" ", "_").upper()
        s_other = str(other).strip().replace(" ", "_").upper()
        return s_self == s_other

    def __hash__(self) -> int:
        return hash(str(self).strip().replace(" ", "_").upper())


@dataclass
class StructuredSignal:
    symbol: str
    signal: str | SignalAction  # STRONG BUY | BUY | HOLD / NO TRADE | SELL | STRONG SELL
    action: str | SignalAction  # Compatibility alias
    confidence: float  # [0.0, 1.0] Calibrated model evidence confidence
    direction_probability: float  # [0.0, 1.0] Long direction probability
    buy_probability: float = 0.0  # Normalized BUY probability [0.0, 1.0]
    sell_probability: float = 0.0  # Normalized SELL probability [0.0, 1.0]
    hold_probability: float = 0.0  # Normalized HOLD probability [0.0, 1.0]
    volatility_forecast: float = 0.01
    volatility_ratio: float = 1.0
    market_regime: str = "NORMAL_CONSOLIDATION"  # NORMAL_BULLISH, ELEVATED_BEARISH, VOL_SHOCK, etc.
    volatility_regime: str = "NORMAL"  # LOW | NORMAL | ELEVATED | EXTREME
    risk_level: str = "MEDIUM"  # LOW | MEDIUM | HIGH | EXTREME
    model_agreement: float = 0.60  # [0.0, 1.0] e.g. 0.80 (4/5)
    model_agreement_fraction: str = "3/5"  # e.g. "4/5"
    entry: float = 0.0
    stop_loss: float = 0.0
    take_profit: float = 0.0
    target_1: float = 0.0
    target_2: float = 0.0
    risk_reward: float = 2.0
    uncertainty_interval: dict[str, float] = field(default_factory=dict)
    evidence: list[str] = field(default_factory=list)
    risks: list[str] = field(default_factory=list)
    reasons: list[str] = field(default_factory=list)
    reason: str = ""  # Primary explanation
    timestamp: str = ""
    model_version: str = "ensemble-v2.0"
    event_risk_high: bool = False
    is_unpredictable_regime: bool = False
    unpredictable_warning: str = ""
    signal_id: str = ""
    data_timestamp: str = ""
    generated_at: str = ""
    expires_at: str = ""
    setup_type: str = "NEUTRAL"
    execution_status: str = "IDLE"
    swing_high: float = 0.0
    swing_low: float = 0.0
    confidence_breakdown: dict[str, Any] = field(default_factory=dict)

    @property
    def is_expired(self) -> bool:
        if not self.expires_at:
            return False
        try:
            exp = datetime.fromisoformat(self.expires_at)
            if exp.tzinfo is None:
                exp = exp.replace(tzinfo=timezone.utc)
            return datetime.now(timezone.utc) > exp
        except Exception:
            return False

    def to_dict(self) -> dict[str, Any]:
        d = asdict(self)
        d["is_expired"] = self.is_expired
        return d


from config import (
    BUY_DIR_THRESHOLD,
    BUY_VOL_RATIO_MAX,
    EVENT_RISK_HOURS_WINDOW,
    MODEL_AGREEMENT_MIN_STRONG,
    SELL_DIR_THRESHOLD,
    SIGNAL_LOOKBACK,
    STOP_LOSS_ATR_MULTIPLIER,
    STRONG_BUY_CONFIDENCE,
    STRONG_BUY_DIR_THRESHOLD,
    STRONG_BUY_VOL_RATIO_MAX,
    STRONG_SELL_CONFIDENCE,
    STRONG_SELL_DIR_THRESHOLD,
    TAKE_PROFIT_RR,
    VOL_SHOCK_RATIO_THRESHOLD,
)
from events.events_service import get_events_service


# Compatibility alias for legacy consumers
Signal = StructuredSignal


class SignalIntelligenceEngine:
    """Quantitative decision engine evaluating multi-modal signals and risk constraints."""

    def __init__(self, lookback: int = SIGNAL_LOOKBACK):
        self.lookback = lookback
        self.vol_history: list[float] = []
        self.symbol_vol_history: dict[str, list[float]] = {}
        self.events_service = get_events_service()

    def generate(
        self,
        predicted_direction: float,
        predicted_volatility: float,
        current_price: float = 65000.0,
        atr: float | None = None,
        reference_volatility: float | None = None,
        uncertainty_quantile: float = 0.005,
        symbol: str = "BTC/USDT",
        lstm_direction: float | None = None,
        transformer_direction: float | None = None,
        technical_regime: str = "NEUTRAL",  # BULLISH | BEARISH | NEUTRAL
        sentiment_score: float = 0.0,  # [-1.0, 1.0]
        rsi: float | None = None,
        macd_line: float | None = None,
        macd_signal: float | None = None,
        adx: float | None = None,
        momentum: float | None = None,
        swing_high: float | None = None,
        swing_low: float | None = None,
        indicators: dict[str, Any] | None = None,
        check_event_risk: bool = True,
        data_timestamp: str | None = None,
        signal_id: str | None = None,
        ttl_seconds: int = 300,
        **kwargs: Any,
    ) -> StructuredSignal:
        if indicators:
            if rsi is None and "rsi" in indicators:
                rsi = float(indicators["rsi"])
            if macd_line is None and "macd_line" in indicators:
                macd_line = float(indicators["macd_line"])
            if macd_signal is None and "macd_signal" in indicators:
                macd_signal = float(indicators["macd_signal"])
            if adx is None and "adx" in indicators:
                adx = float(indicators["adx"])
            if momentum is None:
                momentum = float(indicators.get("momentum_10", indicators.get("momentum", 0.0)))
            if atr is None and "atr" in indicators:
                atr = float(indicators["atr"])

        sym_key = symbol.upper().strip()
        if sym_key not in self.symbol_vol_history:
            self.symbol_vol_history[sym_key] = []
        self.symbol_vol_history[sym_key].append(float(predicted_volatility))
        recent_vol = self.symbol_vol_history[sym_key][-self.lookback :]

        self.vol_history.append(float(predicted_volatility))

        ref_vol = (
            reference_volatility
            if reference_volatility is not None and reference_volatility > 0
            else float(np.median(recent_vol))
        )
        vol_ratio = float(predicted_volatility / max(ref_vol, 1e-6))

        # 1. Volatility Regime Classification
        if vol_ratio >= VOL_SHOCK_RATIO_THRESHOLD:
            vol_regime = "EXTREME"
        elif vol_ratio >= 1.35:
            vol_regime = "ELEVATED"
        elif vol_ratio >= 0.80:
            vol_regime = "NORMAL"
        else:
            vol_regime = "LOW"

        # 2. Conformal Uncertainty Bounds
        lower_bound = max(predicted_volatility - uncertainty_quantile, 0.0)
        upper_bound = predicted_volatility + uncertainty_quantile
        uncertainty = {"lower": round(float(lower_bound), 5), "upper": round(float(upper_bound), 5)}

        # 3. Model Agreement Evaluation across 5 independent factors
        factors_agree = 0
        total_factors = 5
        agreement_details: list[str] = []

        lstm_dir = lstm_direction if lstm_direction is not None else predicted_direction
        trans_dir = transformer_direction if transformer_direction is not None else predicted_direction

        is_bullish_test = predicted_direction >= 0.50

        # Detailed Evidence and Risk lists
        evidence: list[str] = []
        risks: list[str] = []

        if is_bullish_test:
            if lstm_dir >= 0.52:
                factors_agree += 1
                agreement_details.append(f"LSTM indicates upward continuation ({lstm_dir*100:.1f}%)")
                evidence.append(f"+ LSTM indicates upward momentum ({lstm_dir*100:.1f}%)")
            else:
                risks.append(f"- LSTM neutral or divergent ({lstm_dir*100:.1f}%)")

            if trans_dir >= 0.52:
                factors_agree += 1
                agreement_details.append(f"Transformer attention detects positive drift ({trans_dir*100:.1f}%)")
                evidence.append(f"+ Transformer attention detects positive drift ({trans_dir*100:.1f}%)")
            else:
                risks.append(f"- Transformer shows muted upward conviction ({trans_dir*100:.1f}%)")

            if vol_ratio <= 1.35:
                factors_agree += 1
                agreement_details.append(f"Econometric variance within calm bounds ({vol_ratio:.2f}x)")
                evidence.append(f"+ Econometric variance within calm boundaries ({vol_ratio:.2f}x baseline)")
            else:
                risks.append(f"- Elevated volatility regime ({vol_ratio:.2f}x baseline)")

            if technical_regime == "BULLISH":
                factors_agree += 1
                agreement_details.append("Technical trend confirmed (EMA20 > EMA50, Price > EMA20)")
                evidence.append("+ Technical trend aligned bullish (EMA20 > EMA50)")
            else:
                risks.append(f"- Technical trend not confirming ({technical_regime})")

            if sentiment_score >= 0.08:
                factors_agree += 1
                agreement_details.append(f"Market news sentiment is positive ({sentiment_score:+.2f})")
                evidence.append(f"+ Positive financial news sentiment ({sentiment_score:+.2f})")
            elif sentiment_score <= -0.15:
                risks.append(f"- Negative news sentiment headwind ({sentiment_score:+.2f})")
        else:
            if lstm_dir <= 0.48:
                factors_agree += 1
                agreement_details.append(f"LSTM indicates downward persistence ({(1-lstm_dir)*100:.1f}% short)")
                evidence.append(f"+ LSTM indicates downward persistence ({(1-lstm_dir)*100:.1f}%)")
            else:
                risks.append(f"- LSTM not confirming downward thesis ({lstm_dir*100:.1f}%)")

            if trans_dir <= 0.48:
                factors_agree += 1
                agreement_details.append(f"Transformer attention detects negative drift ({(1-trans_dir)*100:.1f}% short)")
                evidence.append(f"+ Transformer detects negative drift ({(1-trans_dir)*100:.1f}%)")
            else:
                risks.append(f"- Transformer attention divergent ({trans_dir*100:.1f}%)")

            if vol_ratio >= 1.15:
                factors_agree += 1
                agreement_details.append(f"Elevated volatility confirms distribution regime ({vol_ratio:.2f}x)")
                evidence.append(f"+ Volatility expansion confirms distribution regime ({vol_ratio:.2f}x)")
            else:
                risks.append(f"- Low volatility distribution lack of sell conviction ({vol_ratio:.2f}x)")

            if technical_regime == "BEARISH":
                factors_agree += 1
                agreement_details.append("Technical trend confirmed bearish (EMA20 < EMA50)")
                evidence.append("+ Technical trend aligned bearish (EMA20 < EMA50)")
            else:
                risks.append(f"- Technical trend not confirming ({technical_regime})")

            if sentiment_score <= -0.08:
                factors_agree += 1
                agreement_details.append(f"Market sentiment is fearful/negative ({sentiment_score:+.2f})")
                evidence.append(f"+ Negative market news sentiment ({sentiment_score:+.2f})")
            elif sentiment_score >= 0.15:
                risks.append(f"- Positive news sentiment resisting downward move ({sentiment_score:+.2f})")

        # Additional indicator evidence
        if rsi is not None:
            if rsi < 32:
                evidence.append(f"+ RSI oversold condition ({rsi:.1f})")
            elif rsi > 68:
                risks.append(f"- RSI overbought condition ({rsi:.1f})")
            elif 45 <= rsi <= 55:
                evidence.append(f"~ RSI in equilibrium channel ({rsi:.1f})")

        if macd_line is not None and macd_signal is not None:
            if macd_line > macd_signal:
                evidence.append(f"+ MACD above signal line ({macd_line:.2f} > {macd_signal:.2f})")
            else:
                risks.append(f"- MACD below signal line ({macd_line:.2f} < {macd_signal:.2f})")

        if adx is not None:
            if adx >= 25:
                evidence.append(f"+ ADX confirms established trend strength ({adx:.1f})")
            else:
                risks.append(f"- ADX indicates choppy / low-trend environment ({adx:.1f})")

        agreement_score = round(factors_agree / total_factors, 2)
        agreement_fraction = f"{factors_agree}/{total_factors}"

        # 4. Check Macro Event Risk
        event_assessment = self.events_service.assess_event_risk(symbol=symbol) if check_event_risk else {"event_risk_high": False, "risk_note": ""}
        event_risk_high = event_assessment.get("event_risk_high", False)
        if event_risk_high:
            risks.append(f"- Macro high-impact event imminent: {event_assessment.get('risk_note')}")

        # 5. Volatility Shock & Unpredictable Regime Detection
        is_vol_shock = vol_ratio >= VOL_SHOCK_RATIO_THRESHOLD
        has_conflicting_indicators = (
            (is_bullish_test and technical_regime == "BEARISH" and predicted_direction > 0.58)
            or (not is_bullish_test and technical_regime == "BULLISH" and predicted_direction < 0.42)
            or (is_bullish_test and sentiment_score <= -0.35 and predicted_direction > 0.60)
            or (rsi is not None and is_bullish_test and rsi > 78)
            or (rsi is not None and not is_bullish_test and rsi < 22)
        )

        is_unpredictable = is_vol_shock or has_conflicting_indicators or (event_risk_high and vol_ratio >= 1.4)

        # 6. Direction Probability Separation
        # Calculate distinct BUY, SELL, HOLD probabilities that sum to 1.0
        p_dir = float(np.clip(predicted_direction, 0.01, 0.99))
        dist_from_neutral = abs(p_dir - 0.5) * 2.0  # 0.0 when 0.5, 1.0 when 0 or 1

        if is_unpredictable:
            # During unpredictable regimes, HOLD probability absorbs the mass
            hold_prob = float(np.clip(0.55 + 0.35 * (1.0 - dist_from_neutral), 0.50, 0.85))
            rem = 1.0 - hold_prob
            buy_prob = rem * p_dir
            sell_prob = rem * (1.0 - p_dir)
        elif is_bullish_test:
            hold_prob = float(np.clip(0.40 * (1.0 - dist_from_neutral), 0.08, 0.40))
            rem = 1.0 - hold_prob
            buy_prob = rem * p_dir
            sell_prob = rem * (1.0 - p_dir)
        else:
            hold_prob = float(np.clip(0.40 * (1.0 - dist_from_neutral), 0.08, 0.40))
            rem = 1.0 - hold_prob
            buy_prob = rem * p_dir
            sell_prob = rem * (1.0 - p_dir)

        total_prob = buy_prob + sell_prob + hold_prob
        buy_prob = round(buy_prob / total_prob, 4)
        sell_prob = round(sell_prob / total_prob, 4)
        hold_prob = round(hold_prob / total_prob, 4)

        # 7. Model Confidence Calibration
        # Confidence reflects statistical quality of evidence, agreement, and volatility calm
        # Never produce fixed 47%
        conviction_component = dist_from_neutral * 0.40
        agreement_component = agreement_score * 0.35
        stability_component = 0.25 * (1.0 / max(vol_ratio, 1.0))
        raw_confidence = 0.25 + conviction_component + agreement_component + stability_component

        if is_unpredictable:
            calibrated_confidence = float(np.clip(raw_confidence * 0.65, 0.22, 0.55))
        elif factors_agree >= 4:
            calibrated_confidence = float(np.clip(raw_confidence, 0.72, 0.93))
        elif factors_agree >= 3:
            calibrated_confidence = float(np.clip(raw_confidence, 0.58, 0.78))
        else:
            calibrated_confidence = float(np.clip(raw_confidence, 0.32, 0.55))

        if event_risk_high:
            calibrated_confidence = max(calibrated_confidence * 0.85, 0.20)

        # 8. Dynamic Stop Loss and Targets (ATR + Swing High/Low)
        price = float(current_price)
        calc_atr = float(atr) if (atr and atr > 0) else max(price * 0.018, 5.0)

        s_high = float(swing_high) if (swing_high and swing_high > price) else (price + 2.5 * calc_atr)
        s_low = float(swing_low) if (swing_low and 0 < swing_low < price) else (price - 2.5 * calc_atr)

        reasons: list[str] = []
        unpredictable_msg = ""

        # 9. Decision Engine Logic
        if is_vol_shock:
            signal_name = "REDUCE"
            risk_level = "HIGH"
            market_regime = "VOLATILITY_SHOCK"
            is_unpredictable_regime = True
            unpredictable_msg = (
                f"HIGH VOLATILITY DETECTED ({vol_ratio:.2f}x normal baseline). "
                "Prediction confidence is unstable due to abnormal volatility expansion. "
                "Recommended action: REDUCE / DE-RISK POSITIONS."
            )
            reasons.append(unpredictable_msg)
            stop_loss = round(s_low * 0.995, 2)
            target_1 = round(price + (1.5 * calc_atr), 2)
            target_2 = round(s_high, 2)
            risk_reward = 1.0

        elif has_conflicting_indicators:
            signal_name = "HOLD"
            risk_level = "MEDIUM"
            market_regime = "CONFLICTING_SIGNALS"
            is_unpredictable_regime = True
            unpredictable_msg = (
                "CONFLICTING INDICATORS DETECTED: Technical trend and statistical model signals are divergent. "
                "Recommended action: HOLD / NO TRADE until consensus is established."
            )
            reasons.append(unpredictable_msg)
            stop_loss = round(s_low, 2)
            target_1 = round(s_high, 2)
            target_2 = round(price + (2.0 * calc_atr), 2)
            risk_reward = 1.0

        elif (
            predicted_direction >= STRONG_BUY_DIR_THRESHOLD
            and calibrated_confidence >= STRONG_BUY_CONFIDENCE
            and vol_ratio <= STRONG_BUY_VOL_RATIO_MAX
            and factors_agree >= 4
            and not event_risk_high
        ):
            signal_name = "STRONG BUY"
            risk_level = "LOW"
            market_regime = f"{vol_regime}_BULLISH"
            reasons.append(f"Strong upward consensus ({buy_prob*100:.1f}% BUY prob) with {agreement_fraction} model agreement.")
            reasons.append(f"Controlled volatility ({vol_ratio:.2f}x) supports sustainable trend extension.")
            reasons.extend(agreement_details)

            # Volatility-aware Stop Loss below recent swing low and 1.5 ATR
            stop_loss = round(min(s_low * 0.998, price - (1.5 * calc_atr)), 2)
            stop_dist = max(price - stop_loss, price * 0.005)
            target_1 = round(price + (2.0 * stop_dist), 2)
            target_2 = round(max(s_high, price + (3.0 * stop_dist)), 2)
            risk_reward = round((target_1 - price) / max(stop_dist, 1e-4), 2)

        elif predicted_direction >= BUY_DIR_THRESHOLD and vol_ratio <= BUY_VOL_RATIO_MAX:
            signal_name = "BUY"
            risk_level = "MEDIUM" if vol_ratio > 1.15 or event_risk_high else "LOW"
            market_regime = f"{vol_regime}_BULLISH"
            reasons.append(f"Upward directional momentum favorable ({buy_prob*100:.1f}% BUY prob) in {vol_regime} volatility regime.")
            if event_risk_high:
                reasons.append(event_assessment["risk_note"])

            stop_loss = round(min(s_low * 0.998, price - (1.6 * calc_atr)), 2)
            stop_dist = max(price - stop_loss, price * 0.005)
            target_1 = round(price + (2.0 * stop_dist), 2)
            target_2 = round(max(s_high, price + (2.8 * stop_dist)), 2)
            risk_reward = round((target_1 - price) / max(stop_dist, 1e-4), 2)

        elif (
            predicted_direction <= STRONG_SELL_DIR_THRESHOLD
            and calibrated_confidence >= STRONG_SELL_CONFIDENCE
            and factors_agree >= 4
        ):
            signal_name = "STRONG SELL"
            risk_level = "HIGH"
            market_regime = f"{vol_regime}_BEARISH"
            reasons.append(f"Strong downward consensus ({sell_prob*100:.1f}% SELL prob) with {agreement_fraction} model agreement.")
            reasons.extend(agreement_details)

            # Volatility-aware Stop Loss above recent swing high and 1.5 ATR
            stop_loss = round(max(s_high * 1.002, price + (1.5 * calc_atr)), 2)
            stop_dist = max(stop_loss - price, price * 0.005)
            target_1 = round(price - (2.0 * stop_dist), 2)
            target_2 = round(min(s_low, price - (3.0 * stop_dist)), 2)
            risk_reward = round((price - target_1) / max(stop_dist, 1e-4), 2)

        elif predicted_direction <= SELL_DIR_THRESHOLD:
            signal_name = "SELL"
            risk_level = "HIGH" if vol_ratio >= 1.30 else "MEDIUM"
            market_regime = f"{vol_regime}_BEARISH"
            reasons.append(f"Downward momentum dominant ({sell_prob*100:.1f}% SELL prob).")

            stop_loss = round(max(s_high * 1.002, price + (1.6 * calc_atr)), 2)
            stop_dist = max(stop_loss - price, price * 0.005)
            target_1 = round(price - (2.0 * stop_dist), 2)
            target_2 = round(min(s_low, price - (2.8 * stop_dist)), 2)
            risk_reward = round((price - target_1) / max(stop_dist, 1e-4), 2)

        else:
            signal_name = "HOLD"
            risk_level = "LOW" if vol_ratio < 1.0 else "MEDIUM"
            market_regime = f"{vol_regime}_CONSOLIDATION"
            reasons.append(f"Market in consolidation channel (BUY: {buy_prob*100:.0f}%, SELL: {sell_prob*100:.0f}%, HOLD: {hold_prob*100:.0f}%).")
            reasons.append("Directional edge insufficient to overcome trading fees and execution slippage.")
            if event_risk_high:
                reasons.append(event_assessment["risk_note"])

            stop_loss = round(min(s_low, price - calc_atr), 2)
            target_1 = round(max(s_high, price + calc_atr), 2)
            target_2 = round(price + (2.0 * calc_atr), 2)
            risk_reward = 1.0

        primary_reason = reasons[0] if reasons else "Neutral market conditions."

        now = datetime.now(timezone.utc)
        gen_at = now.isoformat()
        exp_at = (now + timedelta(seconds=ttl_seconds)).isoformat()
        data_ts = data_timestamp or gen_at
        sig_id = signal_id or f"sig-{uuid.uuid4().hex[:8]}"

        if signal_name in ("STRONG BUY", "STRONG_BUY", "BUY"):
            setup = "LONG_SETUP"
            exec_stat = "WAITING_FOR_ENTRY"
        elif signal_name in ("STRONG SELL", "STRONG_SELL", "SELL"):
            setup = "SHORT_SETUP"
            exec_stat = "WAITING_FOR_ENTRY"
        elif signal_name == "REDUCE":
            setup = "REDUCE_EXPOSURE"
            exec_stat = "REDUCE_POSITION"
        else:
            setup = "NEUTRAL"
            exec_stat = "IDLE"

        action_val = SignalAction(signal_name)
        return StructuredSignal(
            symbol=symbol,
            signal=action_val,
            action=action_val,
            confidence=round(calibrated_confidence, 2),
            direction_probability=round(predicted_direction, 4),
            buy_probability=buy_prob,
            sell_probability=sell_prob,
            hold_probability=hold_prob,
            volatility_forecast=round(predicted_volatility, 5),
            volatility_ratio=round(vol_ratio, 2),
            market_regime=market_regime,
            volatility_regime=vol_regime,
            risk_level=risk_level,
            model_agreement=agreement_score,
            model_agreement_fraction=agreement_fraction,
            entry=round(price, 2),
            stop_loss=round(stop_loss, 2),
            take_profit=round(target_1, 2),
            target_1=round(target_1, 2),
            target_2=round(target_2, 2),
            risk_reward=round(risk_reward, 2),
            uncertainty_interval=uncertainty,
            evidence=evidence,
            risks=risks,
            reasons=reasons,
            reason=primary_reason,
            timestamp=gen_at,
            model_version="ensemble-v2.0",
            event_risk_high=event_risk_high,
            is_unpredictable_regime=is_unpredictable,
            unpredictable_warning=unpredictable_msg,
            signal_id=sig_id,
            data_timestamp=data_ts,
            generated_at=gen_at,
            expires_at=exp_at,
            setup_type=setup,
            execution_status=exec_stat,
            swing_high=round(s_high, 2),
            swing_low=round(s_low, 2),
            confidence_breakdown={
                "conviction_score": round(dist_from_neutral, 2),
                "model_agreement": agreement_score,
                "volatility_penalty": round(max(0.0, vol_ratio - 1.0), 2),
                "event_risk_discount": 0.15 if event_risk_high else 0.0,
            },
        )


# Legacy wrapper alias
class SignalGenerator(SignalIntelligenceEngine):
    pass
