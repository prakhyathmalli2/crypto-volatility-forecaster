"""Cryptocurrency sentiment data ingestion pipeline.

Collects:
1. Crypto Fear & Greed Index multi-year historical series from Alternative.me.
2. News headlines from CoinDesk and Cointelegraph RSS feeds with polarity scoring.
Saves aggregated sentiment time-series to data/raw/sentiment/.
"""
from __future__ import annotations

import json
import re
import xml.etree.ElementTree as ET
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import pandas as pd
import requests
from tenacity import retry, retry_if_exception_type, stop_after_attempt, wait_exponential

from config import FEAR_GREED_API_URL, NEWS_RSS_FEEDS, SENTIMENT_RAW_DIR

# Rule-based sentiment lexicon for financial/crypto text
POSITIVE_WORDS = {
    "bull", "bullish", "surge", "gain", "rally", "breakout", "high", "all-time",
    "profit", "approval", "adoption", "inflow", "boost", "climb", "support",
    "growth", "record", "optimism", "recovery", "accumulate", "institutional"
}
NEGATIVE_WORDS = {
    "bear", "bearish", "crash", "plunge", "drop", "hack", "scam", "ban",
    "sec", "lawsuit", "crackdown", "outflow", "dump", "fall", "liquidation",
    "panic", "collapse", "risk", "loss", "warning", "crisis", "fear"
}


def _lexical_sentiment(text: str) -> float:
    """Fast lexical sentiment score between -1.0 (extremely negative) and +1.0 (extremely positive)."""
    words = re.findall(r"\b[a-z]+\b", text.lower())
    if not words:
        return 0.0
    pos_count = sum(1 for w in words if w in POSITIVE_WORDS)
    neg_count = sum(1 for w in words if w in NEGATIVE_WORDS)
    total = pos_count + neg_count
    if total == 0:
        return 0.0
    return float((pos_count - neg_count) / total)


@retry(
    stop=stop_after_attempt(3),
    wait=wait_exponential(multiplier=1, min=2, max=6),
    retry=retry_if_exception_type((requests.RequestException,)),
    reraise=True,
)
def fetch_fear_and_greed_index() -> pd.DataFrame:
    """Fetch complete historical Crypto Fear & Greed Index series."""
    resp = requests.get(
        FEAR_GREED_API_URL,
        timeout=20,
        headers={"User-Agent": "crypto-volatility-research/2.0"},
    )
    resp.raise_for_status()
    payload = resp.json()
    data = payload.get("data", [])
    if not data:
        return pd.DataFrame(columns=["open_time", "fear_greed_value", "sentiment_score"])

    df = pd.DataFrame(data)
    df["open_time"] = pd.to_datetime(df["timestamp"].astype(int), unit="s", utc=True)
    df["fear_greed_value"] = pd.to_numeric(df["value"], errors="coerce")
    # Normalize 0..100 index to -1.0 .. +1.0 scale: (value - 50) / 50
    df["sentiment_score"] = (df["fear_greed_value"] - 50.0) / 50.0
    return df[["open_time", "fear_greed_value", "sentiment_score"]].drop_duplicates("open_time").sort_values("open_time").reset_index(drop=True)


def _fetch_rss_items(feed_url: str) -> list[dict[str, Any]]:
    """Parse standard RSS 2.0 or Atom XML feed without third-party C dependencies."""
    items: list[dict[str, Any]] = []
    try:
        resp = requests.get(
            feed_url,
            timeout=15,
            headers={"User-Agent": "crypto-volatility-research/2.0"},
        )
        resp.raise_for_status()
        root = ET.fromstring(resp.content)

        # Standard RSS channel/item
        for item in root.findall(".//item"):
            title = item.findtext("title", "").strip()
            pub_date = item.findtext("pubDate", "").strip()
            if title and pub_date:
                try:
                    dt = pd.to_datetime(pub_date, utc=True)
                    items.append({"open_time": dt, "title": title, "source": feed_url})
                except Exception:
                    continue
    except Exception as exc:
        print(f"[RSS Ingest Warning] Failed to parse feed {feed_url}: {exc}")
    return items


def fetch_news_sentiment() -> pd.DataFrame:
    """Collect latest news headlines across feeds and calculate polarity."""
    all_items: list[dict[str, Any]] = []
    for feed in NEWS_RSS_FEEDS:
        all_items.extend(_fetch_rss_items(feed))

    if not all_items:
        return pd.DataFrame(columns=["open_time", "sentiment_score", "news_count"])

    df = pd.DataFrame(all_items).drop_duplicates(subset=["open_time", "title"])
    df["score"] = df["title"].apply(_lexical_sentiment)

    # Aggregate to 1h bars
    hourly = (
        df.set_index("open_time")
        .resample("1h")
        .agg(sentiment_score=("score", "mean"), news_count=("title", "count"))
        .dropna(subset=["sentiment_score"])
        .reset_index()
    )
    return hourly


def build_sentiment_dataset() -> tuple[pd.DataFrame, dict[str, Any]]:
    """Combine Fear & Greed historical series with latest news polarity."""
    fg_df = pd.DataFrame()
    news_df = pd.DataFrame()

    try:
        fg_df = fetch_fear_and_greed_index()
    except Exception as exc:
        print(f"[Sentiment Warning] Fear & Greed API failed: {exc}")

    try:
        news_df = fetch_news_sentiment()
    except Exception as exc:
        print(f"[Sentiment Warning] News RSS failed: {exc}")

    if not fg_df.empty and not news_df.empty:
        merged = pd.merge(fg_df, news_df, on="open_time", how="outer")
        merged["sentiment_score"] = merged["sentiment_score_x"].combine_first(merged["sentiment_score_y"])
        merged = merged.drop(columns=["sentiment_score_x", "sentiment_score_y"], errors="ignore")
    elif not fg_df.empty:
        merged = fg_df
        merged["news_count"] = 0
    elif not news_df.empty:
        merged = news_df
        merged["fear_greed_value"] = 50.0
    else:
        # Fallback neutral series
        now = datetime.now(timezone.utc)
        timestamps = pd.date_range(end=now, periods=730, freq="D", tz="UTC")
        merged = pd.DataFrame(
            {
                "open_time": timestamps,
                "fear_greed_value": 50.0,
                "sentiment_score": 0.0,
                "news_count": 0,
            }
        )

    merged = merged.sort_values("open_time").ffill().bfill().reset_index(drop=True)

    metadata = {
        "sources": ["alternative.me/fng", "rss_crypto_feeds"],
        "download_time_utc": datetime.now(timezone.utc).isoformat(),
        "number_of_rows": len(merged),
        "date_range": [
            str(merged["open_time"].iloc[0]),
            str(merged["open_time"].iloc[-1]),
        ],
    }
    return merged, metadata


def save_sentiment_dataset() -> dict[str, Path]:
    """Save sentiment data and metadata."""
    df, meta = build_sentiment_dataset()
    csv_path = SENTIMENT_RAW_DIR / "btc_sentiment.csv"
    meta_path = SENTIMENT_RAW_DIR / "btc_sentiment_meta.json"

    df.to_csv(csv_path, index=False)
    meta_path.write_text(json.dumps(meta, indent=2), encoding="utf-8")
    return {"csv": csv_path, "meta": meta_path}


if __name__ == "__main__":
    paths = save_sentiment_dataset()
    print(f"Saved sentiment data to {paths['csv']}")
