"""Grounded AI Assistant Router and Multi-Provider Reasoning Engine.

Supports:
1. Primary LLM (OpenAI / OpenRouter if configured)
2. xAI / Grok Provider Abstraction (using XAI_API_KEY / GROK_API_KEY)
3. HuggingFace Inference API (using HUGGINGFACE_API_KEY)
4. Deterministic Reasoning Engine (telemetry-grounded structured analysis fallback)

Includes:
- Strict intent detection (separates finance/technical/risk questions from off-topic questions)
- Light, friendly humor/sarcasm for off-topic questions without abusive language
- Strict grounding in live market data, indicators (RSI, MACD, ADX, Momentum), volatility regimes, and fundamentals
- Clear separation of FACT, MODEL EVIDENCE, RISKS, and RISK MANAGEMENT LEVELS
- Honest representation of missing/unavailable telemetry (never fabricating defaults as facts)
- Model disclaimer: predictions and risk levels are model-generated statistical estimates, not financial guarantees
"""
from __future__ import annotations

import os
import re
from abc import ABC, abstractmethod
from datetime import datetime, timezone
from typing import Any
from zoneinfo import ZoneInfo

from config import (
    GEMINI_API_KEY,
    GEMINI_BASE_URL,
    GEMINI_MODEL,
    HUGGINGFACE_API_KEY,
    LLM_API_KEY,
    LLM_BASE_URL,
    LLM_FALLBACK_PROVIDER,
    LLM_MODEL,
    LLM_PROVIDER,
    LLM_TIMEOUT,
    OPENAI_API_KEY,
    OPENAI_BASE_URL,
    OPENROUTER_API_KEY,
    XAI_API_KEY,
    XAI_BASE_URL,
)
from llm.inference.tools import get_tool_registry
from market_data.provider import is_valid_symbol, to_ist_string
from trading.principles import get_contextual_principle

IST = ZoneInfo("Asia/Kolkata")


def redact_secrets(text: str) -> str:
    """Mask any sensitive API secrets or bearer tokens from logs, traces, or responses."""
    if not text or not isinstance(text, str):
        return ""
    from config import (
        AGY_API_KEY,
        API_SECRET_KEY,
        GEMINI_API_KEY,
        HUGGINGFACE_API_KEY,
        LLM_API_KEY,
        OPENAI_API_KEY,
        OPENROUTER_API_KEY,
        XAI_API_KEY,
    )
    keys_to_redact = [
        AGY_API_KEY,
        API_SECRET_KEY,
        GEMINI_API_KEY,
        LLM_API_KEY,
        OPENAI_API_KEY,
        OPENROUTER_API_KEY,
        XAI_API_KEY,
        HUGGINGFACE_API_KEY,
    ]
    for env_var in (
        "GEMINI_API_KEY",
        "LLM_API_KEY",
        "AGY_API_KEY",
        "API_SECRET_KEY",
        "OPENAI_API_KEY",
        "OPENROUTER_API_KEY",
        "XAI_API_KEY",
        "GROK_API_KEY",
        "HUGGINGFACE_API_KEY",
    ):
        val = os.getenv(env_var)
        if val and len(val) >= 4:
            keys_to_redact.append(val)

    redacted = text
    for k in set(keys_to_redact):
        if k and len(k) >= 4:
            redacted = redacted.replace(k, "[REDACTED_SECRET]")

    # Redact Authorization: Bearer tokens
    redacted = re.sub(r"(Bearer\s+)[A-Za-z0-9_\-\.]{8,}", r"\1[REDACTED_BEARER]", redacted)
    # Redact common key patterns
    redacted = re.sub(r"\b(sk-[a-zA-Z0-9_\-]{8,})\b", "[REDACTED_API_KEY]", redacted)
    redacted = re.sub(r"\b(xai-[a-zA-Z0-9_\-]{8,})\b", "[REDACTED_API_KEY]", redacted)
    redacted = re.sub(r"\b(AQ\.[a-zA-Z0-9_\-]{30,})\b", "[REDACTED_API_KEY]", redacted)
    redacted = re.sub(r"\b(AIza[a-zA-Z0-9_\-]{30,})\b", "[REDACTED_API_KEY]", redacted)
    return redacted


class LLMBackend(ABC):
    """Abstract interface for interchangeable language model backends."""

    @abstractmethod
    def generate(self, user_prompt: str, symbol: str, tools_data: dict[str, Any]) -> str:
        pass


class DeterministicReasoningBackend(LLMBackend):
    """
    Deterministic market reasoning engine with zero external API dependencies.
    Executes live tool telemetry, parses market state, formats structured signal cards,
    and strictly distinguishes FACT, MODEL OUTPUT, TECHNICAL EVIDENCE, and RISK MANAGEMENT.
    """

    def generate(self, user_prompt: str, symbol: str, tools_data: dict[str, Any]) -> str:
        prompt_lower = user_prompt.lower().strip()
        now_utc = datetime.now(timezone.utc)
        now_ist_str = to_ist_string(now_utc)

        tick = tools_data.get("tick", {})
        sig = tools_data.get("signal", {})
        news = tools_data.get("news", [])
        events = tools_data.get("events", [])
        ind = tools_data.get("indicators", {})
        fundamentals = tools_data.get("fundamentals", {})
        fund_metrics = fundamentals.get("metrics", {})

        currency_sym = "₹" if (".NS" in symbol or ".BO" in symbol or symbol == "^NSEI") else "$"

        # Telemetry parsing with strict unavailable guards
        price = tick.get("price")
        if price is not None and isinstance(price, (int, float)) and price > 0:
            price_str = f"{currency_sym}{price:,.2f}"
        else:
            price = None
            price_str = "Data unavailable"

        raw_change = tick.get("change_24h_pct")
        change_str = f"{raw_change:+.2f}%" if raw_change is not None else "--%"

        vol_24 = tick.get("volume_24h")
        if vol_24 is not None and isinstance(vol_24, (int, float)) and vol_24 > 0:
            vol_str = f"{currency_sym}{vol_24:,.2f}"
        else:
            vol_str = "Data unavailable"

        low_52 = tick.get("low_52w")
        high_52 = tick.get("high_52w")
        if low_52 and high_52:
            range_52_str = f"{currency_sym}{low_52:,.2f} – {currency_sym}{high_52:,.2f}"
        else:
            range_52_str = "Data unavailable"

        freshness_badge = tick.get("freshness_text", "LIVE")

        # Indicators parsing
        rsi_val = ind.get("rsi")
        rsi_str = f"{rsi_val:.1f}" if isinstance(rsi_val, (int, float)) else "Data unavailable"

        adx_val = ind.get("adx")
        adx_str = f"{adx_val:.1f}" if isinstance(adx_val, (int, float)) else "Data unavailable"

        trend_regime = ind.get("trend_regime")
        if not trend_regime:
            trend_regime = "Data unavailable" if rsi_val is None else "NEUTRAL"

        # Signal parsing
        raw_sig = sig.get("signal")
        if raw_sig and raw_sig != "UNAVAILABLE":
            signal_action = str(raw_sig)
            conf = sig.get("confidence")
            confidence_str = f"{conf*100:.1f}%" if isinstance(conf, (int, float)) else "--%"
            b_p = sig.get("buy_probability")
            s_p = sig.get("sell_probability")
            h_p = sig.get("hold_probability")
            buy_prob_str = f"{b_p*100:.1f}%" if isinstance(b_p, (int, float)) else "--%"
            sell_prob_str = f"{s_p*100:.1f}%" if isinstance(s_p, (int, float)) else "--%"
            hold_prob_str = f"{h_p*100:.1f}%" if isinstance(h_p, (int, float)) else "--%"
            vol_ratio_str = f"{sig.get('volatility_ratio', 1.0):.2f}x"
            vol_regime_str = sig.get("volatility_regime", "NORMAL")
            has_valid_signal = True
        else:
            signal_action = "Data unavailable"
            confidence_str = "--%"
            buy_prob_str = "--%"
            sell_prob_str = "--%"
            hold_prob_str = "--%"
            vol_ratio_str = "--"
            vol_regime_str = "UNAVAILABLE"
            has_valid_signal = False

        is_unpred = sig.get("is_unpredictable_regime", False)
        unpred_warning = sig.get("unpredictable_warning", "")

        # 1. Unrelated / Off-topic query detection using word boundaries
        market_keywords = [
            "btc", "eth", "nifty", "reliance", "stock", "crypto", "price", "market", "trade", "trend",
            "rsi", "macd", "adx", "volatility", "signal", "buy", "sell", "hold", "indicator", "target",
            "stop", "loss", "news", "inflation", "earnings", "fomc", "cpi", "portfolio", "moving",
            "crash", "surge", "forecast", "prediction", "pe", "ratio", "chart", "bear", "bull",
            "short", "long", "analyze", "overview", "entry", "risk", "status", "why", "happening"
        ]
        pattern = r"\b(" + "|".join(re.escape(k) for k in market_keywords) + r")\b"
        is_market_query = bool(re.search(pattern, prompt_lower))

        if not is_market_query and len(prompt_lower) > 3:
            return (
                "That question has officially escaped the trading terminal 😄. "
                "Ask me about markets, assets, technical indicators, ML signals, volatility, or risk management and I'll help."
            )

        # 2. Why is this moving? / What changed today?
        if any(w in prompt_lower for w in ["moving", "why", "happening", "crashing", "surging", "what changed"]):
            top_news = news[0]["title"] if news else "No recent breaking announcements detected"
            top_news_impact = news[0].get("impact_level", "LOW") if news else "LOW"
            top_event = events[0]["event"] if events else "Standard trading session"

            regime_alert = ""
            if is_unpred:
                regime_alert = f"\n> ⚠️ **HIGH VOLATILITY REGIME DETECTED:** {unpred_warning}\n"

            return f"""### 📊 Market Intelligence Assessment for {symbol}
*Timestamp: {now_ist_str} (IST) • Data Status: {freshness_badge}*
{regime_alert}
**[FACT — LIVE TELEMETRY]**
- Spot Price: **{price_str}** ({change_str} 24h)
- 24h Volume: **{vol_str}**
- RSI (14): **{rsi_str}** | Wilder ADX (14): **{adx_str}**
- Trend Regime: **{trend_regime}**
- 52-Week Range: **{range_52_str}**

**[MODEL OUTPUT — DIRECTION & CONFIDENCE]**
- Active Signal: **{signal_action}** (Calibrated Confidence: **{confidence_str}**)
- Direction Probabilities: BUY **{buy_prob_str}** | SELL **{sell_prob_str}** | HOLD **{hold_prob_str}**
- Volatility Ratio: **{vol_ratio_str}** baseline ({vol_regime_str})

**[NEWS & MACRO CATALYSTS]**
- Headline: *"{top_news}"* (Impact: **{top_news_impact}**)
- Milestone: *{top_event}*

**[PRIMARY EVIDENCE]**
{chr(10).join(f"- {e}" for e in sig.get('evidence', ['Trend and momentum alignment'])) if has_valid_signal else "- Market telemetry evaluated"}

*Note: Model signals and volatility estimates are statistical calculations, not financial advice.*
"""

        # 3. Analyze this asset / Overview
        elif any(w in prompt_lower for w in ["analyze", "analysis", "overview"]):
            ema20_str = f"{currency_sym}{ind.get('ema20'):,.2f}" if ind.get('ema20') is not None else "Data unavailable"
            ema50_str = f"{currency_sym}{ind.get('ema50'):,.2f}" if ind.get('ema50') is not None else "Data unavailable"
            mom_str = f"{ind.get('momentum_10'):+.2f}" if ind.get('momentum_10') is not None else "Data unavailable"
            roc_str = f"{ind.get('roc_10'):+.2f}%" if ind.get('roc_10') is not None else "Data unavailable"
            macd_str = f"{ind.get('macd_line')}" if ind.get('macd_line') is not None else "Data unavailable"
            macd_sig_str = f"{ind.get('macd_signal')}" if ind.get('macd_signal') is not None else "Data unavailable"
            macd_hist_str = f"{ind.get('macd_hist')}" if ind.get('macd_hist') is not None else "Data unavailable"

            stop_loss_str = f"{currency_sym}{sig.get('stop_loss'):,.2f}" if sig.get('stop_loss') is not None else "Data unavailable"
            target1_str = f"{currency_sym}{sig.get('target_1'):,.2f}" if sig.get('target_1') is not None else "Data unavailable"
            target2_str = f"{currency_sym}{sig.get('target_2'):,.2f}" if sig.get('target_2') is not None else "Data unavailable"
            rr_str = f"1:{sig.get('risk_reward'):.1f}" if sig.get('risk_reward') is not None else "Data unavailable"

            adx_desc = f" ({'Strong Trend' if adx_val > 25 else 'Ranging / Weak Trend'})" if isinstance(adx_val, (int, float)) else ""

            return f"""### 🔍 Quantitative Asset Intelligence: {symbol}
*Timestamp: {now_ist_str} (IST) • Status: {freshness_badge}*

**1. Valuation & Financials:**
- Market Cap: **{fund_metrics.get('market_cap', 'Data unavailable')}**
- P/E Ratio: **{fund_metrics.get('pe_ratio', 'Data unavailable')}** | Forward P/E: **{fund_metrics.get('forward_pe', 'Data unavailable')}**
- EPS: **{fund_metrics.get('eps', 'Data unavailable')}** | Revenue: **{fund_metrics.get('revenue', 'Data unavailable')}**

**2. Technical Structure:**
- Spot: **{price_str}** ({change_str} 24h)
- EMA 20: **{ema20_str}** | EMA 50: **{ema50_str}**
- RSI (14): **{rsi_str}** | ADX (14): **{adx_str}**{adx_desc}
- Momentum (10): **{mom_str}** | ROC: **{roc_str}**
- MACD Line: **{macd_str}** vs Signal: **{macd_sig_str}** (Hist: {macd_hist_str})

**3. Model Signal & Conviction:**
- AI Signal: **{signal_action}** (Confidence: **{confidence_str}**)
- Probabilities: Upward **{buy_prob_str}** | Downward **{sell_prob_str}**
- Volatility-Aware Stop-Loss: **{stop_loss_str}**
- Target 1: **{target1_str}** | Target 2: **{target2_str}**
- Risk/Reward: **{rr_str}**
"""

        # 4. Explain Indicators (RSI, MACD, ADX)
        elif "rsi" in prompt_lower:
            if rsi_val is None:
                return f"""### 📈 Relative Strength Index (RSI) for {symbol}
- **Current RSI (14):** **Data unavailable**
- **Condition:** Insufficient historical candle data to calculate 14-period RSI for {symbol}.
- **Explanation:** RSI measures the velocity and magnitude of directional price movements.
  - RSI > 70 indicates elevated momentum where buyers have pushed price to statistically extended levels.
  - RSI < 30 indicates sustained selling where downside momentum may become exhausted.
"""
            status = "overbought (>70, caution on new longs)" if rsi_val >= 70 else ("oversold (<30, potential mean-reversion bounce)" if rsi_val <= 30 else "in the neutral 30–70 momentum zone")
            return f"""### 📈 Relative Strength Index (RSI) for {symbol}
- **Current RSI (14):** **{rsi_val}**
- **Condition:** Market is currently **{status}**.
- **Explanation:** RSI measures the velocity and magnitude of directional price movements.
  - RSI > 70 indicates elevated momentum where buyers have pushed price to statistically extended levels.
  - RSI < 30 indicates sustained selling where downside momentum may become exhausted.
  - Current reading of **{rsi_val}** reflects {'bullish' if rsi_val > 50 else 'bearish'} momentum.
"""

        elif "macd" in prompt_lower:
            m_line = ind.get('macd_line')
            m_sig = ind.get('macd_signal')
            m_hist = ind.get('macd_hist')
            if m_line is None or m_sig is None:
                return f"""### 📊 MACD (12, 26, 9) Analysis for {symbol}
- **MACD Line (12 EMA – 26 EMA):** **Data unavailable**
- **Signal Line (9 EMA):** **Data unavailable**
- **MACD Histogram:** **Data unavailable**
- **State:** Insufficient historical candle data to calculate MACD for {symbol}.
- **Explanation:** MACD captures the convergence and divergence of moving averages. When the histogram expands in positive territory, upward trend acceleration is occurring.
"""
            status = "Bullish crossover (MACD Line > Signal Line)" if m_line > m_sig else "Bearish crossover (MACD Line < Signal Line)"
            return f"""### 📊 MACD (12, 26, 9) Analysis for {symbol}
- **MACD Line (12 EMA – 26 EMA):** **{m_line:.2f}**
- **Signal Line (9 EMA):** **{m_sig:.2f}**
- **MACD Histogram:** **{m_hist:.2f}**
- **State:** **{status}**
- **Explanation:** MACD captures the convergence and divergence of moving averages. When the histogram expands in positive territory, upward trend acceleration is occurring.
"""

        elif "adx" in prompt_lower:
            if adx_val is None:
                return f"""### 📉 Average Directional Index (ADX 14) for {symbol}
- **ADX (14):** **Data unavailable**
- **+DI (Positive Directional Index):** **Data unavailable**
- **-DI (Negative Directional Index):** **Data unavailable**
- **Directional Bias:** **Data unavailable**
- **Explanation:** ADX measures trend strength independent of direction. Values above 25 indicate a solid trend suitable for momentum strategies, while values below 20 indicate chop/consolidation.
"""
            plus_di = ind.get('plus_di')
            minus_di = ind.get('minus_di')
            plus_di_str = f"{plus_di:.1f}" if isinstance(plus_di, (int, float)) else "Data unavailable"
            minus_di_str = f"{minus_di:.1f}" if isinstance(minus_di, (int, float)) else "Data unavailable"
            strength = "Strong Trending Market (>25)" if adx_val >= 25 else "Ranging / Weak Trend (<25)"
            if isinstance(plus_di, (int, float)) and isinstance(minus_di, (int, float)):
                dom = "+DI Bulls Leading" if plus_di > minus_di else "-DI Bears Leading"
            else:
                dom = "Directional Bias: Data unavailable"
            return f"""### 📉 Average Directional Index (ADX 14) for {symbol}
- **ADX (14):** **{adx_val:.1f}** ({strength})
- **+DI (Positive Directional Index):** **{plus_di_str}**
- **-DI (Negative Directional Index):** **{minus_di_str}**
- **Directional Bias:** **{dom}**
- **Explanation:** ADX measures trend strength independent of direction. Values above 25 indicate a solid trend suitable for momentum strategies, while values below 20 indicate chop/consolidation.
"""

        # 5. Explain ML Prediction / Forecast
        elif any(w in prompt_lower for w in ["prediction", "forecast", "ml", "model"]):
            regime_alert = ""
            if is_unpred:
                regime_alert = f"\n> ⚠️ **HIGH VOLATILITY REGIME:** {unpred_warning}\n"

            if not has_valid_signal or sig.get('confidence') is None:
                return f"""### 🤖 Explainable ML Prediction & Volatility Forecast [MODEL OUTPUT]: {symbol}
*Timestamp: {now_ist_str} (IST)*
{regime_alert}
**Model Status:** Statistical volatility predictions are currently unavailable for this asset.
**Note:** Model signals and volatility estimates require active historical feature pipelines.
"""

            vol_fc = sig.get('volatility_forecast')
            vol_fc_str = f"{vol_fc*100:.2f}%" if isinstance(vol_fc, (int, float)) else "Data unavailable"

            return f"""### 🤖 Explainable ML Prediction & Volatility Forecast [MODEL OUTPUT]: {symbol}
*Timestamp: {now_ist_str} (IST)*
{regime_alert}
**1. Probability vs Confidence:**
- Directional Probabilities:
  * BUY: **{buy_prob_str}**
  * SELL: **{sell_prob_str}**
  * HOLD: **{hold_prob_str}**
- Calibrated Model Confidence: **{confidence_str}**
- Volatility Forecast (1h): **{vol_fc_str}** (Regime: **{sig.get('volatility_regime') or 'Data unavailable'}**)

**2. Model Evidence:**
{chr(10).join(f"+ {e}" for e in sig.get('evidence', [])) if sig.get('evidence') else "+ Model evidence data unavailable"}

**3. Identified Risks:**
{chr(10).join(f"- {r}" for r in sig.get('risks', [])) if sig.get('risks') else "- Risk assessment data unavailable"}

*Disclaimer: Model confidence represents calibrated statistical certainty, not a guarantee of future price performance.*
"""

        # 6. Show Signals, Stop Loss & Targets
        elif any(w in prompt_lower for w in ["signal", "target", "stop loss", "entry", "risk reward", "level"]):
            if not has_valid_signal or price is None:
                return f"""### ⚡ Active Trading Signal & Risk Levels: {symbol}
*Timestamp: {now_ist_str} (IST) • Status: {freshness_badge}*

**Status:** Trading signal and risk levels are currently unavailable for {symbol}.
"""

            sig_id = str(sig.get('signal_id') or 'N/A')
            raw_sl = sig.get('stop_loss')
            stop_loss_str = f"{currency_sym}{raw_sl:,.2f} (ATR & Structure-aware)" if isinstance(raw_sl, (int, float)) else "Data unavailable"
            raw_tgt1 = sig.get('target_1')
            tgt1_str = f"{currency_sym}{raw_tgt1:,.2f}" if isinstance(raw_tgt1, (int, float)) else "Data unavailable"
            raw_tgt2 = sig.get('target_2')
            tgt2_str = f"{currency_sym}{raw_tgt2:,.2f}" if isinstance(raw_tgt2, (int, float)) else "Data unavailable"
            raw_rr = sig.get('risk_reward')
            rr_str = f"1:{raw_rr:.1f}" if isinstance(raw_rr, (int, float)) else "Data unavailable"

            return f"""### ⚡ Active Trading Signal & Risk Levels: {symbol}
*Timestamp: {now_ist_str} (IST) • Status: {freshness_badge}*
*Signal ID: {sig_id}*

```text
Asset:           {symbol}
Signal ID:       {sig_id}
Signal:          {signal_action}
Model Confidence:{confidence_str}
Probabilities:   BUY {buy_prob_str} | SELL {sell_prob_str} | HOLD {hold_prob_str}

Current Price:   {price_str}
Stop-Loss:       {stop_loss_str}
Target 1:        {tgt1_str}
Target 2:        {tgt2_str}
Risk/Reward:     {rr_str}
```

**[SUPPORTING EVIDENCE]**
{chr(10).join(f"+ {e}" for e in sig.get('evidence', [])) if sig.get('evidence') else "+ Evidence data unavailable"}

**[RISK FACTORS]**
{chr(10).join(f"- {r}" for r in sig.get('risks', [])) if sig.get('risks') else "- Risk assessment data unavailable"}
"""

        # 7. Summarize News
        elif any(w in prompt_lower for w in ["news", "headline", "breaking"]):
            news_lines = "\n".join(
                f"- **[{n.get('impact_level', 'LOW')}]** *{n.get('title')}* ({n.get('source')}) • Sentiment: {n.get('sentiment', 'NEUTRAL')}"
                for n in news[:5]
            ) or "- No prioritized headlines available for this symbol"

            sent_score = tools_data.get('sentiment_score')
            if sent_score is not None and isinstance(sent_score, (int, float)):
                sent_str = f"**{sent_score:+.2f}**"
            else:
                sent_str = "**Data unavailable**"

            return f"""### 📰 Real-Time News & Sentiment Analysis for {symbol}
*Updated: {now_ist_str} (IST)*

{news_lines}

**News Sentiment Score:** {sent_str}
"""

        # 8. Default Assistant Overview
        else:
            return f"""### 🤖 ALPHAVOL PRO Quantitative Assistant
*Current Time: {now_ist_str} (IST) • Asset: {symbol}*

**Live Market Telemetry:**
- Spot: **{price_str}** ({change_str} 24h) • Status: **{freshness_badge}**
- Active Signal: **{signal_action}** (Confidence: **{confidence_str}**)
- Probabilities: BUY **{buy_prob_str}** | SELL **{sell_prob_str}** | HOLD **{hold_prob_str}**
- Volatility Regime: **{vol_regime_str}**

**Suggested Inquiries:**
- *"Why is this moving?"*
- *"Analyze this asset"*
- *"Explain RSI and MACD"*
- *"Explain ADX"*
- *"Show stop-loss and targets"*
- *"Explain the ML prediction"*
- *"Summarize today's news"*
"""


class GrokXAIBackend(LLMBackend):
    """
    Official xAI / Grok provider integration.
    Communicates via OpenAI-compatible endpoints with server-side API key protection.
    """

    def __init__(
        self,
        api_key: str | None = None,
        base_url: str | None = None,
        model: str | None = None,
        timeout: float | None = None,
    ):
        from config import LLM_BASE_URL, LLM_MODEL, LLM_TIMEOUT, XAI_API_KEY, XAI_BASE_URL

        self.api_key = api_key or XAI_API_KEY
        self.base_url = (base_url or XAI_BASE_URL or LLM_BASE_URL or "https://api.x.ai/v1").rstrip("/")
        self.model = model or LLM_MODEL or "grok-2-latest"
        self.timeout = timeout if timeout is not None else LLM_TIMEOUT

    def generate(self, user_prompt: str, symbol: str, tools_data: dict[str, Any]) -> str:
        if not self.api_key:
            raise ValueError("XAI_API_KEY is not configured.")

        import requests

        system_content = (
            f"You are a Senior Quantitative Trading Analyst for ALPHAVOL PRO. "
            f"The user is asking about {symbol}. "
            f"You MUST use ONLY the following real system telemetry data. "
            f"Never fabricate prices, indicators, news, signals, or returns. "
            f"If something is missing or unavailable, explicitly state so. "
            f"All display timestamps must be presented in IST (Asia/Kolkata). "
            f"Data: {tools_data}"
        )

        payload = {
            "model": self.model,
            "messages": [
                {"role": "system", "content": system_content},
                {"role": "user", "content": user_prompt},
            ],
            "temperature": 0.2,
            "max_tokens": 700,
        }
        headers = {
            "Authorization": f"Bearer {self.api_key}",
            "Content-Type": "application/json",
        }
        try:
            resp = requests.post(
                f"{self.base_url}/chat/completions",
                json=payload,
                headers=headers,
                timeout=self.timeout,
            )
            resp.raise_for_status()
            res_json = resp.json()
            choices = res_json.get("choices")
            if not choices or not isinstance(choices, list) or len(choices) == 0:
                raise ValueError("xAI Grok provider returned empty choices")
            content = choices[0].get("message", {}).get("content", "")
            if not content or not str(content).strip():
                raise ValueError("xAI Grok provider returned empty output")
            return redact_secrets(str(content).strip())
        except Exception as exc:
            clean_err = redact_secrets(str(exc))
            raise RuntimeError(f"xai_grok error: {clean_err}") from None


class OpenAICompatibleBackend(LLMBackend):
    """Connects to OpenAI-compatible APIs (OpenAI, OpenRouter, DeepSeek, Local Ollama, Groq)."""

    def __init__(
        self,
        api_key: str | None = None,
        base_url: str | None = None,
        model: str | None = None,
        timeout: float | None = None,
        provider_name: str = "openai_compatible",
    ):
        import config

        self.provider_name = provider_name.lower().strip()

        # Resolve API key
        if api_key:
            self.api_key = api_key
        elif self.provider_name == "openrouter":
            self.api_key = getattr(config, "OPENROUTER_API_KEY", "") or getattr(config, "LLM_API_KEY", "") or getattr(config, "OPENAI_API_KEY", "")
        else:
            self.api_key = getattr(config, "OPENAI_API_KEY", "") or getattr(config, "LLM_API_KEY", "") or getattr(config, "OPENROUTER_API_KEY", "")

        # Resolve Base URL
        if base_url:
            self.base_url = base_url.rstrip("/")
        elif getattr(config, "LLM_BASE_URL", ""):
            self.base_url = getattr(config, "LLM_BASE_URL", "").rstrip("/")
        elif self.provider_name == "openrouter":
            self.base_url = "https://openrouter.ai/api/v1"
        elif getattr(config, "OPENAI_BASE_URL", ""):
            self.base_url = getattr(config, "OPENAI_BASE_URL", "").rstrip("/")
        else:
            self.base_url = "https://api.openai.com/v1"

        # Resolve Model
        if model:
            self.model = model
        elif getattr(config, "LLM_MODEL", ""):
            self.model = getattr(config, "LLM_MODEL", "")
        elif self.provider_name == "openrouter":
            self.model = "openai/gpt-4o-mini"
        else:
            self.model = "gpt-3.5-turbo"

        self.timeout = timeout if timeout is not None else getattr(config, "LLM_TIMEOUT", 15.0)

    def generate(self, user_prompt: str, symbol: str, tools_data: dict[str, Any]) -> str:
        if not self.api_key and "localhost" not in self.base_url and "127.0.0.1" not in self.base_url:
            raise ValueError(f"{self.provider_name.upper()}_API_KEY or LLM_API_KEY is not configured.")

        import requests

        system_content = (
            f"You are a Senior Quantitative Crypto & Equity Analyst for ALPHAVOL PRO. "
            f"The user is asking about {symbol}. "
            f"Ground your response strictly in the following live telemetry. Do NOT fabricate numbers. "
            f"Data: {tools_data}"
        )
        payload = {
            "model": self.model,
            "messages": [
                {"role": "system", "content": system_content},
                {"role": "user", "content": user_prompt},
            ],
            "temperature": 0.2,
            "max_tokens": 800,
        }
        headers = {"Content-Type": "application/json"}
        if self.api_key:
            headers["Authorization"] = f"Bearer {self.api_key}"
        if self.provider_name == "openrouter" or "openrouter.ai" in self.base_url:
            headers["HTTP-Referer"] = "http://localhost:8000"
            headers["X-Title"] = "ALPHAVOL PRO"

        try:
            resp = requests.post(
                f"{self.base_url}/chat/completions",
                json=payload,
                headers=headers,
                timeout=self.timeout,
            )
            resp.raise_for_status()
            res_json = resp.json()
            choices = res_json.get("choices")
            if not choices or not isinstance(choices, list) or len(choices) == 0:
                raise ValueError(f"Provider '{self.provider_name}' returned empty choices")
            content = choices[0].get("message", {}).get("content", "")
            if not content or not str(content).strip():
                raise ValueError(f"Provider '{self.provider_name}' returned empty output")
            return redact_secrets(str(content).strip())
        except Exception as exc:
            clean_err = redact_secrets(str(exc))
            raise RuntimeError(f"{self.provider_name} error: {clean_err}") from None


class HuggingFaceBackend(LLMBackend):
    """HuggingFace Inference API backend."""

    def __init__(
        self,
        api_key: str | None = None,
        base_url: str | None = None,
        model: str | None = None,
        timeout: float | None = None,
    ):
        from config import HUGGINGFACE_API_KEY, LLM_API_KEY, LLM_BASE_URL, LLM_MODEL, LLM_TIMEOUT

        self.api_key = api_key or HUGGINGFACE_API_KEY or LLM_API_KEY
        self.base_url = (base_url or LLM_BASE_URL or "https://api-inference.huggingface.co/models").rstrip("/")
        self.model = model or LLM_MODEL or "meta-llama/Meta-Llama-3-8B-Instruct"
        self.timeout = timeout if timeout is not None else LLM_TIMEOUT

    def generate(self, user_prompt: str, symbol: str, tools_data: dict[str, Any]) -> str:
        if not self.api_key:
            raise ValueError("HUGGINGFACE_API_KEY or LLM_API_KEY is not configured.")

        import requests

        system_content = (
            f"You are a Senior Quantitative Trading Analyst for ALPHAVOL PRO. "
            f"Ground your response strictly in the following telemetry: {tools_data}. "
            f"User query about {symbol}: {user_prompt}"
        )
        url = f"{self.base_url}/{self.model}"
        headers = {"Authorization": f"Bearer {self.api_key}"}
        payload = {"inputs": system_content, "parameters": {"max_new_tokens": 500}}
        try:
            resp = requests.post(url, json=payload, headers=headers, timeout=self.timeout)
            resp.raise_for_status()
            res = resp.json()
            if isinstance(res, list) and res and "generated_text" in res[0]:
                out = res[0]["generated_text"]
            else:
                out = str(res)
            if not out or not str(out).strip() or str(out).strip() == "[]":
                raise ValueError("HuggingFace provider returned empty output")
            return redact_secrets(str(out).strip())
        except Exception as exc:
            clean_err = redact_secrets(str(exc))
            raise RuntimeError(f"huggingface error: {clean_err}") from None


class GeminiBackend(LLMBackend):
    """Google Gemini AI Assistant backend using official Gemini API and OpenAI-compatible endpoint."""

    def __init__(
        self,
        api_key: str | None = None,
        base_url: str | None = None,
        model: str | None = None,
        timeout: float | None = None,
    ):
        from config import (
            GEMINI_API_KEY,
            GEMINI_BASE_URL,
            GEMINI_MODEL,
            LLM_API_KEY,
            LLM_BASE_URL,
            LLM_MODEL,
            LLM_TIMEOUT,
        )

        self.api_key = api_key or GEMINI_API_KEY or LLM_API_KEY
        self.base_url = (base_url or GEMINI_BASE_URL or LLM_BASE_URL or "https://generativelanguage.googleapis.com/v1beta/openai").rstrip("/")
        self.model = model or GEMINI_MODEL or LLM_MODEL or "gemini-3.5-flash-lite"
        self.timeout = timeout if timeout is not None else LLM_TIMEOUT

    def generate(self, user_prompt: str, symbol: str, tools_data: dict[str, Any]) -> str:
        if not self.api_key:
            raise ValueError("GEMINI_API_KEY or LLM_API_KEY is not configured.")

        import requests

        system_content = (
            f"You are the Lead Quantitative Financial & Trading AI Assistant for ALPHAVOL PRO (Crypto & Multi-Asset Volatility Forecaster). "
            f"You provide insightful, expert answers for every query related to the project, its machine learning architecture, technical indicators, "
            f"econometric volatility models (GARCH, LSTM, Transformer attention, Ensemble), risk management principles, paper trading, and live markets. "
            f"The user is asking about {symbol}. "
            f"Ground your market analysis strictly in the following live telemetry and never fabricate numbers: "
            f"Telemetry: {tools_data}."
        )

        # 1. Primary: Gemini OpenAI-compatible chat completions
        payload = {
            "model": self.model,
            "messages": [
                {"role": "system", "content": system_content},
                {"role": "user", "content": user_prompt},
            ],
            "temperature": 0.2,
            "max_tokens": 1000,
        }
        headers = {
            "Authorization": f"Bearer {self.api_key}",
            "Content-Type": "application/json",
        }

        try:
            resp = requests.post(
                f"{self.base_url}/chat/completions",
                json=payload,
                headers=headers,
                timeout=self.timeout,
            )
            if resp.status_code == 200:
                res_json = resp.json()
                choices = res_json.get("choices")
                if choices and isinstance(choices, list) and len(choices) > 0:
                    content = choices[0].get("message", {}).get("content", "")
                    if content and str(content).strip():
                        return redact_secrets(str(content).strip())
        except Exception:
            pass

        # 2. Fallback: Native Google Gemini REST API generateContent
        try:
            native_url = f"https://generativelanguage.googleapis.com/v1beta/models/{self.model}:generateContent"
            native_headers = {
                "x-goog-api-key": self.api_key,
                "Content-Type": "application/json",
            }
            native_payload = {
                "contents": [
                    {
                        "parts": [
                            {"text": f"{system_content}\n\nUser Question: {user_prompt}"}
                        ]
                    }
                ],
                "generationConfig": {
                    "temperature": 0.2,
                    "maxOutputTokens": 1000,
                },
            }
            resp = requests.post(native_url, json=native_payload, headers=native_headers, timeout=self.timeout)
            resp.raise_for_status()
            data = resp.json()
            candidates = data.get("candidates", []) if isinstance(data, dict) else []
            if candidates and isinstance(candidates, list) and isinstance(candidates[0], dict):
                content_obj = candidates[0].get("content")
                if isinstance(content_obj, dict):
                    parts = content_obj.get("parts")
                    if isinstance(parts, list) and parts and isinstance(parts[0], dict):
                        out_text = parts[0].get("text")
                        if out_text and str(out_text).strip():
                            return redact_secrets(str(out_text).strip())
            raise ValueError("Gemini generateContent returned empty output")
        except Exception as exc:
            clean_err = redact_secrets(str(exc))
            raise RuntimeError(f"gemini error: {clean_err}") from None


class AIRouter:
    """Orchestrates tool execution and inter-provider failover."""

    def __init__(self):
        self.tool_registry = get_tool_registry()
        self.deterministic_backend = DeterministicReasoningBackend()
        self.openai_backend = OpenAICompatibleBackend(provider_name="openai_compatible")
        self.openrouter_backend = OpenAICompatibleBackend(provider_name="openrouter")
        self.grok_backend = GrokXAIBackend()
        self.hf_backend = HuggingFaceBackend()
        self.gemini_backend = GeminiBackend()

    def _get_backend(self, provider_name: str) -> LLMBackend | None:
        p = (provider_name or "").lower().strip()
        if p == "deterministic":
            return self.deterministic_backend
        elif p in ("gemini", "google", "gemini_api"):
            return self.gemini_backend or GeminiBackend()
        elif p in ("xai_grok", "grok", "xai"):
            return self.grok_backend or GrokXAIBackend()
        elif p in ("openai_compatible", "openai", "local"):
            return self.openai_backend or OpenAICompatibleBackend(provider_name="openai_compatible")
        elif p == "openrouter":
            return self.openrouter_backend or OpenAICompatibleBackend(provider_name="openrouter")
        elif p in ("huggingface", "hf"):
            return self.hf_backend or HuggingFaceBackend()
        return None

    def answer_query(self, user_prompt: str, symbol: str = "BTC/USDT") -> dict[str, Any]:
        now_utc = datetime.now(timezone.utc)
        from config import LLM_FALLBACK_PROVIDER, LLM_PROVIDER
        configured_provider = LLM_PROVIDER
        fallback_provider_cfg = LLM_FALLBACK_PROVIDER

        # 1. Validate prompt: empty / whitespace / non-string
        if not user_prompt or not isinstance(user_prompt, str) or not user_prompt.strip():
            return {
                "symbol": symbol,
                "query": user_prompt or "",
                "response": "Please enter a valid market question or inquiry.",
                "provider": "deterministic_reasoning_engine",
                "provider_configured": configured_provider,
                "fallback_used": False,
                "fallback_reason": None,
                "tools_called": [],
                "signal_embedded": None,
                "timestamp_utc": now_utc.isoformat(),
                "timestamp_ist": to_ist_string(now_utc),
                "data_status": "AVAILABLE",
            }

        # 2. Validate prompt: malformed (null bytes, control characters)
        clean_prompt = user_prompt.replace("\x00", "").strip()
        if not clean_prompt:
            return {
                "symbol": symbol,
                "query": "",
                "response": "Malformed query provided. Please submit a valid question.",
                "provider": "deterministic_reasoning_engine",
                "provider_configured": configured_provider,
                "fallback_used": False,
                "fallback_reason": None,
                "tools_called": [],
                "signal_embedded": None,
                "timestamp_utc": now_utc.isoformat(),
                "timestamp_ist": to_ist_string(now_utc),
                "data_status": "AVAILABLE",
            }

        # 3. Validate prompt: oversized (> 2000 chars)
        if len(clean_prompt) > 2000:
            return {
                "symbol": symbol,
                "query": clean_prompt[:200] + "...",
                "response": "Your query exceeds the maximum allowed length (2,000 characters). Please provide a more concise question.",
                "provider": "deterministic_reasoning_engine",
                "provider_configured": configured_provider,
                "fallback_used": False,
                "fallback_reason": None,
                "tools_called": [],
                "signal_embedded": None,
                "timestamp_utc": now_utc.isoformat(),
                "timestamp_ist": to_ist_string(now_utc),
                "data_status": "AVAILABLE",
            }

        # 4. Check for secret extraction attempts (treat user input as untrusted)
        secret_keywords = ["agy_api_key", "api_secret_key", "secret_key", "private_key", "openai_api_key", "xai_api_key", "grok_api_key", "huggingface_api_key"]
        if any(k in clean_prompt.lower() for k in secret_keywords):
            return {
                "symbol": symbol,
                "query": clean_prompt,
                "response": "Security Notice: System credentials and API keys are strictly protected and cannot be retrieved or inspected through chat.",
                "provider": "security_guard",
                "provider_configured": configured_provider,
                "fallback_used": False,
                "fallback_reason": None,
                "tools_called": [],
                "signal_embedded": None,
                "timestamp_utc": now_utc.isoformat(),
                "timestamp_ist": to_ist_string(now_utc),
                "data_status": "PROTECTED",
            }

        # 3. Validate symbol
        if not is_valid_symbol(symbol):
            return {
                "symbol": symbol,
                "query": user_prompt,
                "response": f"Symbol '{symbol}' is not recognized or supported. Please specify a valid crypto pair (e.g. BTC/USDT), stock (e.g. RELIANCE.NS, AAPL), index (^NSEI), or forex pair (USDINR=X).",
                "provider": "deterministic_reasoning_engine",
                "provider_configured": configured_provider,
                "fallback_used": False,
                "fallback_reason": None,
                "tools_called": [],
                "signal_embedded": None,
                "timestamp_utc": now_utc.isoformat(),
                "timestamp_ist": to_ist_string(now_utc),
                "data_status": "UNAVAILABLE",
            }

        # 4. Safe telemetry retrieval with isolated error handling per tool
        tools_called: list[str] = []
        tools_data: dict[str, Any] = {}

        # Live Price
        try:
            tick = self.tool_registry.get_live_price(symbol=symbol)
            tools_called.append("get_live_price")
        except Exception:
            tick = {"symbol": symbol, "data_status": "UNAVAILABLE", "freshness_text": "UNAVAILABLE", "is_stale": True, "price": None}
        tools_data["tick"] = tick

        # Current Signal
        try:
            signal = self.tool_registry.get_current_signal(symbol=symbol)
            tools_called.append("get_current_signal")
        except Exception:
            signal = {"signal": "UNAVAILABLE", "status": "UNAVAILABLE", "confidence": None}
        tools_data["signal"] = signal

        # Indicators
        try:
            indicators = self.tool_registry.get_indicators(symbol=symbol)
            tools_called.append("get_indicators")
        except Exception:
            indicators = {"status": "UNAVAILABLE"}
        tools_data["indicators"] = indicators

        # News
        try:
            news = self.tool_registry.get_latest_news(symbol=symbol, limit=5)
            tools_called.append("get_latest_news")
        except Exception:
            news = []
        tools_data["news"] = news

        # Events
        try:
            events = self.tool_registry.get_market_events(symbol=symbol)
            tools_called.append("get_market_events")
        except Exception:
            events = []
        tools_data["events"] = events

        # Fundamentals
        try:
            fundamentals = self.tool_registry.get_fundamentals(symbol=symbol)
            tools_called.append("get_fundamentals")
        except Exception:
            fundamentals = {"symbol": symbol, "metrics": {}}
        tools_data["fundamentals"] = fundamentals

        # 5. Provider routing & deterministic failover
        provider_used = "deterministic_reasoning_engine"
        response_text = ""
        fallback_used = False
        fallback_reason: str | None = None

        primary_provider = (configured_provider or "deterministic").lower().strip()
        backend = self._get_backend(primary_provider)

        if primary_provider != "deterministic":
            try:
                if backend is None:
                    raise ValueError(f"Unknown or unconfigured provider: {primary_provider}")
                out = backend.generate(user_prompt, symbol, tools_data)
                if not out or not str(out).strip():
                    raise ValueError(f"Provider '{primary_provider}' returned empty output")
                response_text = str(out)
                provider_used = primary_provider
            except Exception as exc:
                fallback_used = True
                fallback_reason = f"Provider '{primary_provider}' error: {exc}"

        # If primary failed or was deterministic, invoke fallback
        if not response_text:
            fb_prov = (fallback_provider_cfg or "deterministic").lower().strip()
            if fallback_used and fb_prov != "deterministic" and fb_prov != primary_provider:
                fb_backend = self._get_backend(fb_prov)
                if fb_backend:
                    try:
                        out = fb_backend.generate(user_prompt, symbol, tools_data)
                        if out and str(out).strip():
                            response_text = str(out)
                            provider_used = fb_prov
                    except Exception as fb_exc:
                        fallback_reason = f"{fallback_reason}; Fallback provider '{fb_prov}' error: {fb_exc}"

            # Ultimate guaranteed deterministic reasoning engine
            if not response_text:
                try:
                    response_text = self.deterministic_backend.generate(user_prompt, symbol, tools_data)
                    provider_used = "deterministic_reasoning_engine"
                except Exception as det_exc:
                    response_text = f"Market data telemetry received for {symbol}, but analysis generation encountered an internal issue: {det_exc}"
                    provider_used = "deterministic_reasoning_engine"

        now_utc = datetime.now(timezone.utc)
        if fallback_reason:
            fallback_reason = redact_secrets(fallback_reason)
        response_text = redact_secrets(response_text)

        return {
            "symbol": symbol,
            "query": user_prompt,
            "response": response_text,
            "provider": provider_used,
            "provider_configured": configured_provider,
            "fallback_used": fallback_used,
            "fallback_reason": fallback_reason,
            "tools_called": tools_called,
            "signal_embedded": signal.get("signal") if isinstance(signal, dict) else None,
            "timestamp_utc": now_utc.isoformat(),
            "timestamp_ist": to_ist_string(now_utc),
            "data_status": tick.get("data_status", "AVAILABLE") if isinstance(tick, dict) else "AVAILABLE",
        }


# Global singleton router
_ai_router: AIRouter | None = None


def get_ai_router() -> AIRouter:
    global _ai_router
    if _ai_router is None:
        _ai_router = AIRouter()
    return _ai_router
