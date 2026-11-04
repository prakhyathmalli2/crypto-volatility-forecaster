"""
Unit and integration tests for real-time WebSocket market and signal streams (Section 3, 6, 43).
"""
from __future__ import annotations

import asyncio
from typing import Any
import pytest
from market_data.provider import MarketTick


class MockWebSocket:
    def __init__(self):
        self.sent_messages: list[dict[str, Any]] = []
        self.is_closed = False

    async def accept(self):
        pass

    async def send_json(self, data: dict[str, Any]):
        if self.is_closed:
            raise RuntimeError("WebSocket closed")
        self.sent_messages.append(data)

    async def receive_text(self) -> str:
        await asyncio.sleep(0.01)
        return "ping"

    async def close(self):
        self.is_closed = True


@pytest.mark.anyio
async def test_websocket_market_stream_event_structure(monkeypatch):
    """Verify Section 3 & 4: /ws/market/{symbol} broadcasts normalized market tick events."""
    import serving.inference_api as inf

    # Mock provider to guarantee deterministic zero-latency tick
    def mock_get_ticker(symbol):
        return MarketTick(
            symbol=symbol,
            price=85844.98,
            bid=85840.0,
            ask=85845.0,
            volume_24h=12500.0,
            change_24h_pct=5.2,
            timestamp="2026-09-21T18:00:00Z",
            exchange="binance",
        )

    monkeypatch.setattr(inf._market_provider, "get_ticker", mock_get_ticker)

    mock_ws = MockWebSocket()
    task = asyncio.create_task(inf.ws_market_stream(mock_ws, symbol="BTC/USDT"))
    await asyncio.sleep(0.5)
    task.cancel()
    try:
        await task
    except (asyncio.CancelledError, Exception):
        pass

    assert len(mock_ws.sent_messages) > 0, "WebSocket feed must send ticks"
    first_msg = mock_ws.sent_messages[0]

    assert first_msg["type"] == "tick"
    assert "data" in first_msg
    tick = first_msg["data"]
    assert tick["symbol"] == "BTC/USDT"
    assert tick["price"] == 85844.98
    assert "timestamp" in tick


@pytest.mark.anyio
async def test_websocket_signal_stream_canonical_sync(monkeypatch):
    """Verify Section 14 & 41: /ws/signals/{symbol} broadcasts canonical signal events."""
    import serving.inference_api as inf

    def mock_get_signal(symbol, timeframe="1h", force_refresh=False):
        return {
            "signal_id": "test-sig-uuid-1234",
            "symbol": symbol,
            "signal": "SELL",
            "confidence": 0.47,
            "setup_type": "SHORT_SETUP",
            "execution_status": "WAITING_FOR_ENTRY",
            "entry": 85844.98,
            "stop_loss": 86360.05,
            "take_profit": 84814.84,
            "risk_reward": 2.0,
        }

    monkeypatch.setattr(inf, "get_v1_signal", mock_get_signal)

    mock_ws = MockWebSocket()
    task = asyncio.create_task(inf.ws_signals_stream(mock_ws, symbol="BTC/USDT"))
    await asyncio.sleep(0.5)
    task.cancel()
    try:
        await task
    except (asyncio.CancelledError, Exception):
        pass

    assert len(mock_ws.sent_messages) > 0, "Signal WebSocket feed must broadcast signals"
    first_msg = mock_ws.sent_messages[0]

    assert first_msg["type"] in ["signal", "heartbeat"]
    if first_msg["type"] == "signal":
        assert "data" in first_msg
        sig = first_msg["data"]
        assert sig["symbol"] == "BTC/USDT"
        assert sig["signal_id"] == "test-sig-uuid-1234"
        assert sig["signal"] == "SELL"
        assert sig["setup_type"] == "SHORT_SETUP"
        assert sig["execution_status"] == "WAITING_FOR_ENTRY"
