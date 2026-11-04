"""Unit tests for News and Macro Events engines."""
from __future__ import annotations

import pytest
from news.news_service import NewsService, NewsArticle
from events.events_service import EventsService, MarketEvent


def test_news_service_fetching():
    service = NewsService()
    articles = service.fetch_articles()
    assert len(articles) > 0
    first = articles[0]
    assert isinstance(first, NewsArticle)
    assert first.title != ""
    assert first.importance in ("BREAKING", "HIGH IMPACT", "IMPORTANT", "NORMAL")
    assert -1.0 <= first.sentiment <= 1.0


def test_news_prioritization_by_symbol():
    service = NewsService()
    prioritized = service.get_prioritized_news(symbol="BTC/USDT", limit=5)
    assert len(prioritized) > 0
    assert "dynamic_relevance" in prioritized[0]


def test_events_service_calendar():
    service = EventsService()
    events = service.get_upcoming_events()
    assert len(events) > 0
    first = events[0]
    assert "event" in first
    assert "hours_until" in first
    assert first["hours_until"] >= 0


def test_event_risk_assessment():
    service = EventsService()
    assessment = service.assess_event_risk(symbol="BTC/USDT", hours_window=80.0)
    assert "event_risk_high" in assessment
    assert isinstance(assessment["event_risk_high"], bool)
    assert "risk_note" in assessment
