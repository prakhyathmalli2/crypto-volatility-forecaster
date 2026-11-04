"""Deterministic tool registry for the AI Trading Assistant.

Provides 12 real-time market tools allowing the AI to query live prices,
indicators, news, events, ML volatility predictions, paper portfolios, and backtests.
"""
from __future__ import annotations

from typing import Any
from market_data.provider import get_market_provider
from market_data.indicators import calculate_indicators
from news.news_service import get_news_service
from events.events_service import get_events_service
from trading.principles import get_contextual_principle
from trading.execution import PaperBroker
from config import INITIAL_CAPITAL


class MarketToolRegistry:
    """Provides structured execution of real-time tools for AI reasoning."""

    def __init__(self):
        self.market_provider = get_market_provider()
        self.news_service = get_news_service()
        self.events_service = get_events_service()
        self.paper_broker = PaperBroker(initial_capital=INITIAL_CAPITAL)

    def get_live_price(self, symbol: str = "BTC/USDT") -> dict[str, Any]:
        """Tool: Retrieve current live market tick and 24h metrics."""
        tick = self.market_provider.get_ticker(symbol)
        return tick.to_dict()

    def get_market_data(self, symbol: str = "BTC/USDT", timeframe: str = "1h", limit: int = 50) -> dict[str, Any]:
        """Tool: Retrieve historical OHLCV candles."""
        candles = self.market_provider.get_ohlcv(symbol, timeframe=timeframe, limit=limit)
        return {"symbol": symbol, "timeframe": timeframe, "count": len(candles), "candles": [c.to_dict() for c in candles[-10:]]}

    def get_chart_data(self, symbol: str = "BTC/USDT", timeframe: str = "1h") -> dict[str, Any]:
        """Tool: Retrieve recent price action summary."""
        candles = self.market_provider.get_ohlcv(symbol, timeframe=timeframe, limit=24)
        if not candles:
            return {"symbol": symbol, "error": "No candles available"}
        closes = [c.close for c in candles]
        return {
            "symbol": symbol,
            "period_open": closes[0],
            "period_close": closes[-1],
            "high": max(c.high for c in candles),
            "low": min(c.low for c in candles),
            "change_pct": round(((closes[-1] - closes[0]) / closes[0]) * 100, 2),
        }

    def get_indicators(self, symbol: str = "BTC/USDT", timeframe: str = "1h") -> dict[str, Any]:
        """Tool: Calculate current technical indicators (EMA, RSI, MACD, ATR, Bollinger, VWAP)."""
        candles = self.market_provider.get_ohlcv(symbol, timeframe=timeframe, limit=100)
        ind = calculate_indicators([c.to_dict() for c in candles])
        return ind.get("current", {})

    def get_latest_news(self, symbol: str = "BTC/USDT", limit: int = 5) -> list[dict[str, Any]]:
        """Tool: Fetch top prioritized news headlines for asset."""
        return self.news_service.get_prioritized_news(symbol=symbol, limit=limit)

    def get_market_events(self, symbol: str = "BTC/USDT") -> list[dict[str, Any]]:
        """Tool: Fetch upcoming economic and crypto milestones."""
        return self.events_service.get_upcoming_events(symbol=symbol, max_hours=72.0)

    def get_sentiment(self) -> dict[str, Any]:
        """Tool: Retrieve Crypto Fear & Greed Index and aggregate sentiment."""
        from data_pipeline.ingest_sentiment_data import fetch_fear_and_greed_index
        try:
            df = fetch_fear_and_greed_index()
            val = int(df["fear_greed_value"].iloc[-1])
            score = float(df["sentiment_score"].iloc[-1])
            classification = "Extreme Greed" if val > 75 else ("Greed" if val > 55 else ("Fear" if val < 45 else ("Extreme Fear" if val < 25 else "Neutral")))
            # Canonical keys: fear_and_greed_value, classification
            # TODO: Deprecate fear_and_greed_score and sentiment_label in next API revision
            return {
                "status": "AVAILABLE",
                "fear_and_greed_value": val,
                "fear_and_greed_score": val,  # Temporary alias for frontend backward compatibility
                "sentiment_score": score,
                "classification": classification,
                "sentiment_label": classification,  # Temporary alias for frontend backward compatibility
            }
        except Exception as exc:
            return {
                "status": "UNAVAILABLE",
                "fear_and_greed_value": None,
                "fear_and_greed_score": None,
                "sentiment_score": None,
                "classification": "Data unavailable",
                "sentiment_label": "Data unavailable",
                "error": str(exc),
            }

    def get_ml_forecast(self, symbol: str = "BTC/USDT") -> dict[str, Any]:
        """Tool: Retrieve multi-horizon GARCH + LSTM + Transformer ensemble forecast."""
        try:
            from serving.inference_api import predict_volatility, PredictRequest
            req = PredictRequest(symbol=symbol, timeframe="1h")
            pred = predict_volatility(req)
            h = pred.get("ensemble_forecasts", {})
            h1 = h.get("1h", {})
            h6 = h.get("6h", {})
            h24 = h.get("24h", {})
            return {
                "status": "AVAILABLE",
                "symbol": symbol,
                "1h_volatility": h1.get("volatility"),
                "6h_volatility": h6.get("volatility"),
                "24h_volatility": h24.get("volatility"),
                "1h_direction_prob": h1.get("directional_probability"),
                "model_weights": {"garch": 0.35, "lstm": 0.35, "transformer": 0.30},
                "inference_mode": pred.get("inference_mode", "ml_ensemble"),
                "fallback_used": pred.get("fallback_used", False),
            }
        except Exception as exc:
            return {
                "status": "UNAVAILABLE",
                "symbol": symbol,
                "1h_volatility": None,
                "6h_volatility": None,
                "24h_volatility": None,
                "1h_direction_prob": None,
                "model_weights": None,
                "error": str(exc),
            }

    def get_current_signal(self, symbol: str = "BTC/USDT", timeframe: str = "1h") -> dict[str, Any]:
        """Tool: Query the canonical ML Signal Intelligence Engine."""
        try:
            from serving.inference_api import get_v1_signal
            return get_v1_signal(symbol=symbol, timeframe=timeframe)
        except Exception as exc:
            return {
                "status": "UNAVAILABLE",
                "signal": "UNAVAILABLE",
                "symbol": symbol,
                "confidence": None,
                "error": str(exc),
            }

    def get_risk_metrics(self, symbol: str = "BTC/USDT") -> dict[str, Any]:
        """Tool: Query current risk limits, ATR stop distances, and circuit breakers."""
        try:
            from trading.risk_manager import RiskManager
            rm = RiskManager(capital=INITIAL_CAPITAL)
            tick = self.market_provider.get_ticker(symbol)
            ind = self.get_indicators(symbol=symbol)
            atr = ind.get("atr")
            if atr is None or atr <= 0:
                return {"status": "UNAVAILABLE", "symbol": symbol, "error": "ATR unavailable"}
            plan = rm.plan_trade(entry_price=tick.price, atr=atr, side="BUY")
            return plan.to_dict()
        except Exception as exc:
            return {"status": "UNAVAILABLE", "symbol": symbol, "error": str(exc)}

    def get_paper_positions(self) -> dict[str, Any]:
        """Tool: Retrieve simulated paper account portfolio status."""
        try:
            return self.paper_broker.get_portfolio_summary()
        except Exception as exc:
            return {"status": "UNAVAILABLE", "positions": [], "error": str(exc)}

    def get_backtest_results(self) -> dict[str, Any]:
        """Tool: Fetch academic held-out comparative test set performance metrics."""
        from config import PROCESSED_DIR
        import json
        p = PROCESSED_DIR / "evaluation_report.json"
        if p.exists():
            try:
                data = json.loads(p.read_text(encoding="utf-8"))
                data["status"] = "AVAILABLE"
                return data
            except Exception:
                pass
        return {
            "status": "UNAVAILABLE",
            "summary": "No evaluation report available",
            "metrics": None,
        }

    def get_fundamentals(self, symbol: str = "BTC/USDT") -> dict[str, Any]:
        """Tool: Retrieve company and asset fundamentals (P/E, Market Cap, EPS, Revenue, 52w Range)."""
        try:
            from market_data.fundamentals import get_asset_fundamentals
            return get_asset_fundamentals(symbol=symbol)
        except Exception as exc:
            return {"symbol": symbol, "status": "UNAVAILABLE", "metrics": {}, "error": str(exc)}


# Global singleton
_tool_registry: MarketToolRegistry | None = None


def get_tool_registry() -> MarketToolRegistry:
    global _tool_registry
    if _tool_registry is None:
        _tool_registry = MarketToolRegistry()
    return _tool_registry
