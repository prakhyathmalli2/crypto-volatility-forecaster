"""Keyless crypto-news sentiment ingestion from RSS feeds."""
from __future__ import annotations

import re
import requests
import pandas as pd
import feedparser
from vaderSentiment.vaderSentiment import SentimentIntensityAnalyzer

from config import NEWS_RSS_FEEDS, RAW_DIR

SIA = SentimentIntensityAnalyzer()


def _is_btc_related(text: str) -> bool:
    t = text.lower()
    return bool(re.search(r"\b(bitcoin|btc|crypto|cryptocurrency|digital asset)\b", t))


def fetch_news(feeds: list[str] | None = None) -> pd.DataFrame:
    rows: list[dict] = []
    for feed_url in feeds or NEWS_RSS_FEEDS:
        try:
            response = requests.get(feed_url, timeout=15, headers={"User-Agent": "crypto-volatility-research/1.0"})
            response.raise_for_status()
            feed = feedparser.parse(response.content)
            for item in feed.entries:
                title = str(item.get("title", "")).strip()
                published = item.get("published", item.get("updated"))
                if not title or not published or not _is_btc_related(title):
                    continue
                rows.append({"published_at": pd.to_datetime(published, utc=True), "title": title, "source": feed_url})
        except Exception:
            continue
    if not rows:
        return pd.DataFrame(columns=["published_at", "title", "source", "sentiment"])
    df = pd.DataFrame(rows).drop_duplicates(subset=["published_at", "title"])
    df["sentiment"] = df["title"].map(lambda t: SIA.polarity_scores(t)["compound"])
    return df.sort_values("published_at")


def hourly_sentiment_index(df: pd.DataFrame) -> pd.DataFrame:
    if df.empty:
        return pd.DataFrame(columns=["open_time", "sentiment_score", "news_count"])
    hourly = (
        df.set_index("published_at")
        .resample("1h")
        .agg(sentiment_score=("sentiment", "mean"), news_count=("title", "count"))
        .reset_index()
        .rename(columns={"published_at": "open_time"})
    )
    return hourly


def save_sentiment():
    news = fetch_news()
    hourly = hourly_sentiment_index(news)
    out = RAW_DIR / "sentiment_btc.csv"
    hourly.to_csv(out, index=False)
    return out


if __name__ == "__main__":
    out = save_sentiment()
    print(f"Saved {out}")
