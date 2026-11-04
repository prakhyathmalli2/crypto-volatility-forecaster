"""Educational quantitative trading principles and market wisdom database.

Provides contextual risk and discipline guidelines grounded in publicly documented
methodologies from classical market practitioners (Livermore, Graham, Buffett,
Paul Tudor Jones, Druckenmiller, Minervini, Dalio).
"""
from __future__ import annotations

from dataclasses import asdict, dataclass
from typing import Any


@dataclass
class TradingPrinciple:
    id: str
    concept: str
    source_attribution: str  # Historical attribution
    context_trigger: str  # EXTREME_VOL | TREND | RANGE | EVENT_RISK | GENERAL
    rule_summary: str
    detailed_explanation: str

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


PRINCIPLES_DATABASE = [
    TradingPrinciple(
        id="risk_first_ptj",
        concept="Asymmetric Risk/Reward & Capital Defense",
        source_attribution="Principle associated with Paul Tudor Jones",
        context_trigger="EXTREME_VOL",
        rule_summary="Don't focus on making money; focus on protecting what you have.",
        detailed_explanation="When volatility expands rapidly, position sizes must shrink proportionally to ensure maximum portfolio heat never exceeds your predetermined risk threshold.",
    ),
    TradingPrinciple(
        id="vol_vs_dir_livermore",
        concept="Separating Volatility Expansion from Directional Conviction",
        source_attribution="Classical market observation (Reminiscences of a Stock Operator / Jesse Livermore)",
        context_trigger="EXTREME_VOL",
        rule_summary="High volatility signals risk, not direction.",
        detailed_explanation="A violent price swing does not imply a bullish trend; price surges accompanied by surging volatility often represent liquidity sweeps or exhaustion moves rather than sustainable trend continuations.",
    ),
    TradingPrinciple(
        id="macro_uncertainty_druckenmiller",
        concept="Preserving Liquidity Ahead of Binary Catalysts",
        source_attribution="Methodology associated with Stanley Druckenmiller",
        context_trigger="EVENT_RISK",
        rule_summary="Never gamble on binary economic releases when volatility spreads widen.",
        detailed_explanation="If a major economic event (such as an FOMC rate decision or CPI release) is scheduled within hours, forecast variance spikes. Favorable positioning waits for post-event price acceptance.",
    ),
    TradingPrinciple(
        id="invalidation_minervini",
        concept="Pre-Trade Invalidation & Stop-Loss Discipline",
        source_attribution="Risk framework associated with Mark Minervini",
        context_trigger="TREND",
        rule_summary="Always establish your invalidation stop before placing your entry.",
        detailed_explanation="A professional trader determines where an idea is statistically wrong before calculating potential profits. If ATR widens, widen your stop and decrease notional position quantity.",
    ),
    TradingPrinciple(
        id="overtrading_buffett",
        concept="Patient Selective Execution in Quiet Regimes",
        source_attribution="Principle associated with Warren Buffett",
        context_trigger="RANGE",
        rule_summary="The stock market is a no-called-strike game. You don't have to swing at every pitch.",
        detailed_explanation="When technical indicators show conflicting signals and model agreement falls below threshold, HOLD is the statistically superior strategy to avoid trading fee erosion.",
    ),
    TradingPrinciple(
        id="dalio_all_weather",
        concept="Regime Diversification & Model Agreement",
        source_attribution="Systematic framework associated with Ray Dalio",
        context_trigger="GENERAL",
        rule_summary="Never rely on a single predictor; require multi-modal evidence.",
        detailed_explanation="Our system combines econometric variance (GARCH), sequential memory (LSTM), and multi-head attention (Transformer) to prevent blind spots inherent in any single mathematical structure.",
    ),
]


def get_contextual_principle(
    volatility_regime: str = "NORMAL",
    signal_action: str = "HOLD",
    event_risk_high: bool = False,
) -> dict[str, Any]:
    """Select the most relevant educational principle for current market state."""
    if event_risk_high:
        p = next(x for x in PRINCIPLES_DATABASE if x.id == "macro_uncertainty_druckenmiller")
    elif volatility_regime in ("EXTREME", "ELEVATED") or signal_action == "REDUCE":
        p = next(x for x in PRINCIPLES_DATABASE if x.id == "risk_first_ptj")
    elif signal_action in ("STRONG_BUY", "STRONG_SELL", "BUY", "SELL"):
        p = next(x for x in PRINCIPLES_DATABASE if x.id == "invalidation_minervini")
    elif signal_action == "HOLD":
        p = next(x for x in PRINCIPLES_DATABASE if x.id == "overtrading_buffett")
    else:
        p = next(x for x in PRINCIPLES_DATABASE if x.id == "dalio_all_weather")

    return p.to_dict()
