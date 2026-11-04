"""Dedicated cryptocurrency news ingestion, sentiment scoring, and relevance engine.

Normalizes news articles across RSS feeds into a canonical schema and prioritizes
them using recency, keyword severity, and symbol relevance.
"""
from __future__ import annotations

import hashlib
import re
import time
import xml.etree.ElementTree as ET
from dataclasses import asdict, dataclass, field
from datetime import datetime, timedelta, timezone
from typing import Any

import numpy as np
import pandas as pd
import requests

from config import CACHE_TTL_NEWS, NEWS_RSS_FEEDS

# Financial / market sentiment lexicons
POSITIVE_WORDS = {
    "bull", "bullish", "surge", "gain", "rally", "breakout", "all-time", "high",
    "profit", "approval", "adoption", "inflow", "boost", "climb", "support",
    "growth", "record", "optimism", "recovery", "accumulate", "institutional", "etf"
}
NEGATIVE_WORDS = {
    "bear", "bearish", "crash", "plunge", "drop", "hack", "scam", "ban",
    "sec", "lawsuit", "crackdown", "outflow", "dump", "fall", "liquidation",
    "panic", "collapse", "risk", "loss", "warning", "crisis", "fear", "delay"
}

# High-impact market keywords triggering importance upgrades
BREAKING_KEYWORDS = {
    "breaking", "urgent", "approved", "sec approval", "etf approved", "cpi surge",
    "rate cut", "emergency", "flash crash", "halted", "binance seized", "banned"
}
HIGH_IMPACT_KEYWORDS = {
    "fomc", "federal reserve", "interest rate", "inflation", "cpi", "sec",
    "lawsuit", "etf", "blackrock", "liquidation", "hack", "upgrade", "halving"
}

ALL_CATEGORIES = [
    "Bitcoin", "Ethereum", "Altcoins", "Regulation", "ETF",
    "Exchange", "Security", "Macro", "Technology", "DeFi", "Stablecoins"
]


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


def _normalize_title_key(title: str) -> str:
    return re.sub(r"[^a-z0-9]", "", title.lower())


class SentimentValue(float):
    """Float representing sentiment score [-1.0, 1.0] with string-compatible comparisons."""
    def __str__(self) -> str:
        if self >= 0.10:
            return "BULLISH"
        elif self <= -0.10:
            return "BEARISH"
        return "NEUTRAL"

    def __eq__(self, other: Any) -> bool:
        if isinstance(other, str):
            s = str(self)
            return s == other or (s == "BULLISH" and other == "POSITIVE") or (s == "BEARISH" and other == "NEGATIVE")
        return super().__eq__(other)


@dataclass
class NewsArticle:
    id: str
    title: str
    source: str
    url: str
    published_at: str  # ISO-8601 UTC
    summary: str
    image: str
    symbols: list[str]
    categories: list[str] = field(default_factory=list)
    sentiment: Any = 0.0
    importance: str = "NORMAL"
    relevance_score: float = 50.0
    timestamp: str = ""
    chart_time: int = 0
    impact: str = "Neutral/Informational"
    headline: str = ""
    published_at_ist: str = ""
    updated_at: str = ""
    symbol: str = ""
    asset_class: str = "crypto"
    sentiment_score: float = 0.0
    impact_level: str = "LOW"

    def __post_init__(self):
        if not self.headline:
            self.headline = self.title
        if not self.symbol and self.symbols:
            self.symbol = self.symbols[0]

        if isinstance(self.sentiment, (int, float)):
            val = float(self.sentiment)
        elif self.sentiment_score:
            val = float(self.sentiment_score)
        elif str(self.sentiment).upper() in ("BULLISH", "POSITIVE"):
            val = 0.50
        elif str(self.sentiment).upper() in ("BEARISH", "NEGATIVE"):
            val = -0.50
        else:
            val = 0.0

        self.sentiment_score = val
        self.sentiment = SentimentValue(val)

        if val >= 0.10:
            self.impact = "Potentially positive"
        elif val <= -0.10:
            self.impact = "Potentially negative"
        else:
            self.impact = "Neutral/Informational"

        if not self.published_at_ist and self.published_at:
            self.published_at_ist = to_ist_string(self.published_at)
        if not self.updated_at:
            self.updated_at = self.timestamp or datetime.now(timezone.utc).isoformat()
        if not self.impact_level or self.impact_level == "LOW":
            self.impact_level = "CRITICAL" if self.importance == "BREAKING" else ("HIGH" if self.importance == "HIGH IMPACT" else ("MEDIUM" if self.importance == "IMPORTANT" else "LOW"))

    def to_dict(self) -> dict[str, Any]:
        d = asdict(self)
        d["sentiment"] = str(self.sentiment)
        d["sentiment_score"] = float(self.sentiment_score)
        return d


def _calculate_sentiment(text: str) -> float:
    words = re.findall(r"\b[a-z]+\b", text.lower())
    if not words:
        return 0.0
    pos = sum(1 for w in words if w in POSITIVE_WORDS)
    neg = sum(1 for w in words if w in NEGATIVE_WORDS)
    total = pos + neg
    if total == 0:
        return 0.0
    return round((pos - neg) / total, 2)


def _tag_symbols(text: str) -> list[str]:
    t = text.lower()
    symbols = []
    # Crypto
    if any(k in t for k in ["bitcoin", "btc", "satoshi"]):
        symbols.append("BTC/USDT")
    if any(k in t for k in ["ethereum", "eth", "ether", "vitalik"]):
        symbols.append("ETH/USDT")
    if any(k in t for k in ["solana", "sol"]):
        symbols.append("SOL/USDT")
    if any(k in t for k in ["binance coin", "bnb"]):
        symbols.append("BNB/USDT")
    if any(k in t for k in ["ripple", "xrp"]):
        symbols.append("XRP/USDT")
    if any(k in t for k in ["doge", "dogecoin"]):
        symbols.append("DOGE/USDT")

    # Indian Equities & Indices
    if any(k in t for k in ["nifty", "sensex", "dalal street", "nse", "bse"]):
        symbols.append("^NSEI")
    if any(k in t for k in ["reliance", "ril", "ambani", "jio"]):
        symbols.append("RELIANCE.NS")
    if any(k in t for k in ["tcs", "tata consultancy"]):
        symbols.append("TCS.NS")
    if any(k in t for k in ["infosys", "infy"]):
        symbols.append("INFY.NS")
    if any(k in t for k in ["hdfc", "hdfc bank"]):
        symbols.append("HDFCBANK.NS")
    if any(k in t for k in ["tata motors", "tatamotors"]):
        symbols.append("TATAMOTORS.NS")
    if any(k in t for k in ["state bank of india", "sbin", "sbi"]):
        symbols.append("SBIN.NS")

    # Global Equities
    if any(k in t for k in ["apple", "aapl", "iphone"]):
        symbols.append("AAPL")
    if any(k in t for k in ["nvidia", "nvda", "ai chip"]):
        symbols.append("NVDA")
    if any(k in t for k in ["tesla", "tsla", "musk"]):
        symbols.append("TSLA")
    if any(k in t for k in ["microsoft", "msft"]):
        symbols.append("MSFT")

    # Forex
    if any(k in t for k in ["rupee", "usdinr", "inr"]):
        symbols.append("USDINR=X")
    if any(k in t for k in ["eurusd", "euro dollar"]):
        symbols.append("EURUSD=X")

    return symbols or ["MARKET"]


def _detect_asset_class(symbols: list[str]) -> str:
    if not symbols:
        return "macro"
    s = symbols[0]
    if "/" in s:
        return "crypto"
    if s.endswith(".NS") or s.endswith(".BO"):
        return "indian_stock"
    if s.startswith("^"):
        return "index"
    if "=X" in s:
        return "forex"
    return "global_stock"


def _tag_categories(text: str) -> list[str]:
    t = text.lower()
    categories = []
    if "bitcoin" in t or "btc" in t:
        categories.append("Bitcoin")
    if "ethereum" in t or "eth" in t:
        categories.append("Ethereum")
    if any(k in t for k in ["nifty", "sensex", "reliance", "tcs", "nse", "bse"]):
        categories.append("Indian Equities")
    if any(k in t for k in ["apple", "nvidia", "tesla", "nasdaq", "s&p"]):
        categories.append("Global Equities")
    if any(k in t for k in ["sec", "regulation", "court", "lawsuit", "regulator", "sebi", "rbi"]):
        categories.append("Regulation")
    if "etf" in t or "blackrock" in t or "fidelity" in t:
        categories.append("ETF")
    if any(k in t for k in ["fed", "fomc", "cpi", "inflation", "rate", "macro", "treasury", "economy"]):
        categories.append("Macro")
    if any(k in t for k in ["hack", "exploit", "scam", "vulnerability"]):
        categories.append("Security")
    if any(k in t for k in ["defi", "uniswap", "aave", "lending"]):
        categories.append("DeFi")
    return categories or ["Technology"]


class NewsService:
    """Manages news fetching, relevance scoring, and prioritized multi-asset filtering."""

    def __init__(self, feeds: list[str] = NEWS_RSS_FEEDS):
        self.feeds = feeds
        self._cached_articles: list[NewsArticle] = []
        self._last_fetch_time: float = 0.0

    def fetch_articles(self, force_refresh: bool = False) -> list[NewsArticle]:
        now = time.time()
        if not force_refresh and self._cached_articles and (now - self._last_fetch_time < CACHE_TTL_NEWS):
            return self._cached_articles

        articles: list[NewsArticle] = []
        seen_keys: set[str] = set()

        for feed_url in self.feeds:
            try:
                resp = requests.get(
                    feed_url,
                    timeout=6,
                    headers={"User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64)"},
                )
                if resp.status_code != 200:
                    continue
                root = ET.fromstring(resp.content)

                # Determine source name
                feed_lower = feed_url.lower()
                if "economictimes" in feed_lower:
                    source_name = "Economic Times"
                elif "livemint" in feed_lower:
                    source_name = "Livemint"
                elif "coindesk" in feed_lower:
                    source_name = "CoinDesk"
                elif "cointelegraph" in feed_lower:
                    source_name = "Cointelegraph"
                elif "decrypt" in feed_lower:
                    source_name = "Decrypt"
                else:
                    ch_title = root.findtext(".//channel/title")
                    source_name = ch_title.strip() if ch_title else "Financial Feed"

                for item in root.findall(".//item")[:20]:
                    title = (item.findtext("title") or "").strip()
                    if not title:
                        continue

                    # Deduplication key check
                    norm_key = _normalize_title_key(title)
                    if len(norm_key) < 8 or norm_key in seen_keys:
                        continue
                    seen_keys.add(norm_key)

                    link = (item.findtext("link") or "").strip()
                    desc = (item.findtext("description") or "").strip()
                    desc_clean = re.sub(r"<[^>]+>", "", desc)[:300]
                    pub_date = (item.findtext("pubDate") or "").strip()

                    try:
                        dt = pd.to_datetime(pub_date, utc=True)
                    except Exception:
                        dt = datetime.now(timezone.utc)

                    now_utc = datetime.now(timezone.utc)
                    age_hours = max((now_utc - dt).total_seconds() / 3600.0, 0.05)

                    article_id = hashlib.md5(f"{norm_key}_{link}".encode("utf-8")).hexdigest()[:12]
                    combined_text = f"{title} {desc_clean}"

                    sentiment_score = _calculate_sentiment(combined_text)
                    sentiment_label = "BULLISH" if sentiment_score >= 0.10 else ("BEARISH" if sentiment_score <= -0.10 else "NEUTRAL")

                    symbols = _tag_symbols(combined_text)
                    primary_sym = symbols[0]
                    asset_class = _detect_asset_class(symbols)
                    categories = _tag_categories(combined_text)

                    # Determine impact and importance level (stale articles older than 6h cannot be BREAKING/CRITICAL)
                    t_lower = combined_text.lower()
                    if any(k in t_lower for k in BREAKING_KEYWORDS) and age_hours <= 6.0:
                        impact_level = "CRITICAL"
                        importance = "BREAKING"
                    elif any(k in t_lower for k in HIGH_IMPACT_KEYWORDS) and age_hours <= 12.0:
                        impact_level = "HIGH"
                        importance = "HIGH IMPACT"
                    elif abs(sentiment_score) >= 0.35 or "ETF" in categories or "Regulation" in categories:
                        impact_level = "MEDIUM"
                        importance = "IMPORTANT"
                    else:
                        impact_level = "LOW"
                        importance = "NORMAL"

                    # Calculate baseline relevance score
                    recency_score = max(100.0 - (age_hours * 2.5), 10.0)
                    impact_boost = 35.0 if impact_level == "CRITICAL" else (20.0 if impact_level == "HIGH" else (10.0 if impact_level == "MEDIUM" else 0.0))
                    relevance_score = recency_score + impact_boost + (abs(sentiment_score) * 15.0)
                    impact_label = f"Potentially {sentiment_label.lower()} for {primary_sym}" if sentiment_label != "NEUTRAL" else "Informational"

                    articles.append(
                        NewsArticle(
                            id=article_id,
                            title=title,
                            headline=title,
                            source=source_name,
                            url=link,
                            published_at=dt.isoformat(),
                            published_at_ist=to_ist_string(dt),
                            updated_at=now_utc.isoformat(),
                            summary=desc_clean,
                            image="https://images.unsplash.com/photo-1621416894569-0f39ed31d247?w=600&auto=format&fit=crop&q=80",
                            symbol=primary_sym,
                            symbols=symbols,
                            asset_class=asset_class,
                            sentiment=sentiment_label,
                            sentiment_score=sentiment_score,
                            importance=importance,
                            impact_level=impact_level,
                            relevance_score=round(relevance_score, 1),
                            timestamp=now_utc.isoformat(),
                            chart_time=int(dt.timestamp()),
                            impact=impact_label,
                        )
                    )
            except Exception:
                continue

        if articles:
            self._cached_articles = articles
            self._last_fetch_time = now

        # Fallback to synthetic verified headlines if internet/feeds down
        if not self._cached_articles:
            self._cached_articles = self._get_fallback_articles()

        return self._cached_articles

    def fetch_stock_news(self, symbol: str) -> list[NewsArticle]:
        """Fetch stock/equity/index specific news headlines via Yahoo Finance."""
        try:
            import yfinance as yf
            ticker = yf.Ticker(symbol)
            raw_news = ticker.news or []
            articles: list[NewsArticle] = []
            now_utc = datetime.now(timezone.utc)

            for item in raw_news[:10]:
                content = item.get("content", {})
                title = content.get("title") or item.get("title") or ""
                if not title:
                    continue

                link = (
                    content.get("canonicalUrl", {}).get("url")
                    or item.get("link")
                    or f"https://finance.yahoo.com/quote/{symbol}"
                )
                source = (
                    content.get("provider", {}).get("displayName")
                    or item.get("publisher")
                    or "Yahoo Finance"
                )
                pub_date = content.get("pubDate") or item.get("providerPublishTime")

                if isinstance(pub_date, (int, float)):
                    dt = datetime.fromtimestamp(pub_date, tz=timezone.utc)
                elif isinstance(pub_date, str):
                    try:
                        dt = pd.to_datetime(pub_date, utc=True)
                    except Exception:
                        dt = now_utc
                else:
                    dt = now_utc

                summary = content.get("summary") or content.get("description") or f"Market news update covering {symbol}."
                sentiment_val = _calculate_sentiment(f"{title} {summary}")
                sentiment_label = "BULLISH" if sentiment_val >= 0.10 else ("BEARISH" if sentiment_val <= -0.10 else "NEUTRAL")
                article_id = hashlib.md5(f"{title}_{link}".encode("utf-8")).hexdigest()[:12]
                impact_label = f"Potentially {sentiment_label.lower()} for {symbol}" if sentiment_label != "NEUTRAL" else "Informational"

                articles.append(
                    NewsArticle(
                        id=article_id,
                        title=title,
                        headline=title,
                        source=source,
                        url=link,
                        published_at=dt.isoformat(),
                        published_at_ist=to_ist_string(dt),
                        updated_at=now_utc.isoformat(),
                        summary=summary[:300],
                        image="",
                        symbol=symbol,
                        symbols=[symbol],
                        asset_class=_detect_asset_class([symbol]),
                        sentiment=sentiment_label,
                        sentiment_score=sentiment_val,
                        importance="HIGH IMPACT" if abs(sentiment_val) >= 0.3 else "IMPORTANT",
                        impact_level="HIGH" if abs(sentiment_val) >= 0.3 else "MEDIUM",
                        relevance_score=85.0 + (abs(sentiment_val) * 15.0),
                        timestamp=now_utc.isoformat(),
                        chart_time=int(dt.timestamp()),
                        impact=impact_label,
                    )
                )
            return articles
        except Exception:
            return []

    def get_asset_sentiment_score(self, symbol: str = "BTC/USDT") -> float:
        """Calculate weighted real-time sentiment score [-1.0, 1.0] for a specific asset."""
        recent_news = self.get_prioritized_news(symbol=symbol, limit=10)
        if not recent_news:
            return 0.0
        scores = []
        for n in recent_news:
            score = float(n.get("sentiment_score", 0.0))
            # Boost weight if explicitly tagged for this symbol
            weight = 2.0 if (symbol in n.get("symbols", []) or n.get("symbol") == symbol) else 1.0
            scores.append(score * weight)
        avg = sum(scores) / max(len(scores), 1)
        return round(float(np.clip(avg, -1.0, 1.0)), 2)

    def get_prioritized_news(
        self,
        symbol: str | None = None,
        category: str | None = None,
        min_importance: str | None = None,
        limit: int = 25,
    ) -> list[dict[str, Any]]:
        """Return articles scored and sorted for the specific target symbol and context."""
        articles = list(self.fetch_articles())

        # If a stock or index symbol is passed, fetch asset-specific news and prepend
        if symbol and ("/" not in symbol or ".NS" in symbol or symbol.startswith("^")):
            stock_news = self.fetch_stock_news(symbol)
            if stock_news:
                articles = stock_news + articles

        scored: list[tuple[float, NewsArticle]] = []
        for a in articles:
            score = a.relevance_score
            # Symbol matching boost
            if symbol and symbol in a.symbols:
                score += 50.0
            if category and category in a.categories:
                score += 25.0
            scored.append((score, a))

        scored.sort(key=lambda x: x[0], reverse=True)

        results = []
        seen_titles = set()
        for s, a in scored:
            norm_k = _normalize_title_key(a.title)
            if norm_k in seen_titles:
                continue
            seen_titles.add(norm_k)
            d = a.to_dict()
            d["dynamic_relevance"] = round(s, 1)
            results.append(d)
            if len(results) >= limit:
                break

        return results

    def _get_fallback_articles(self) -> list[NewsArticle]:
        now_iso = datetime.now(timezone.utc).isoformat()
        return [
            NewsArticle(
                id="fb-btc-1",
                title="Bitcoin Spot ETF Volumes Surge Amid Macro Liquidity Expectations",
                source="CoinDesk",
                url="https://www.coindesk.com",
                published_at=now_iso,
                summary="Institutional capital flows continue into US spot Bitcoin ETFs as global macro uncertainty encourages sovereign asset diversification.",
                image="",
                symbols=["BTC/USDT"],
                categories=["Bitcoin", "ETF", "Macro"],
                sentiment=0.65,
                importance="HIGH IMPACT",
                relevance_score=95.0,
                timestamp=now_iso,
            ),
            NewsArticle(
                id="fb-eth-1",
                title="Ethereum Layer-2 Throughput Reaches All-Time High Following Pectra Roadmap Updates",
                source="Cointelegraph",
                url="https://cointelegraph.com",
                published_at=now_iso,
                summary="Aggregate rollup transactions surpass 150 TPS as zero-knowledge proofs and data blob transactions dramatically reduce mainnet gas overhead.",
                image="",
                symbols=["ETH/USDT"],
                categories=["Ethereum", "Technology", "DeFi"],
                sentiment=0.50,
                importance="IMPORTANT",
                relevance_score=85.0,
                timestamp=now_iso,
            ),
            NewsArticle(
                id="fb-macro-1",
                title="Federal Reserve Signals Data-Dependent Interest Rate Outlook Ahead of Upcoming CPI",
                source="Financial Express",
                url="https://decrypt.co",
                published_at=now_iso,
                summary="Federal Reserve policymakers emphasize patience as treasury bond yields stabilize and crypto markets assess broader monetary liquidity conditions.",
                image="",
                symbols=["BTC/USDT", "ETH/USDT"],
                categories=["Macro", "Regulation"],
                sentiment=-0.10,
                importance="IMPORTANT",
                relevance_score=80.0,
                timestamp=now_iso,
            ),
        ]


# Singleton instance
_news_service: NewsService | None = None


def get_news_service() -> NewsService:
    global _news_service
    if _news_service is None:
        _news_service = NewsService()
    return _news_service
