"""Real-time market streaming manager and candle aggregator for ALPHAVOL PRO.

Architectural role:
- Connects directly to exchange WebSocket feeds (e.g. Binance raw WebSocket streams).
- Aggregates incoming trades / klines into canonical timeframes (1m, 5m, 15m, 1h, 4h, 1d, 1w).
- Normalizes candles and ticks into standard UTC schema with IST presentation timestamps.
- Manages client WebSocket fan-out subscriptions, heartbeats, and automatic reconnection.
- Handles non-crypto assets with honest market status (DELAYED or MARKET CLOSED).
"""
from __future__ import annotations

import asyncio
import json
import math
import time
from datetime import datetime, timezone
from typing import Any, Callable

import websockets
from market_data.provider import (
    Candle,
    MarketTick,
    detect_asset_type,
    get_market_provider,
    get_market_status,
    to_ist_string,
)

# Canonical timeframe conversion to seconds
TIMEFRAME_SECONDS: dict[str, int] = {
    "1m": 60,
    "5m": 300,
    "15m": 900,
    "1h": 3600,
    "4h": 14400,
    "1d": 86400,
    "1w": 604800,
}

# Binance timeframe mapping
BINANCE_TIMEFRAME_MAP: dict[str, str] = {
    "1m": "1m",
    "5m": "5m",
    "15m": "15m",
    "1h": "1h",
    "4h": "4h",
    "1d": "1d",
    "1w": "1w",
}


def normalize_crypto_symbol_for_binance(symbol: str) -> str:
    """Convert BTC/USDT to btcusdt for Binance streams."""
    return symbol.replace("/", "").replace("-", "").lower()


class RealTimeStreamManager:
    """Central singleton managing live exchange streams and client fanout."""

    def __init__(self):
        self._active_connections: dict[str, set[Any]] = {}  # channel -> set of client websockets
        self._binance_tasks: dict[str, asyncio.Task] = {}     # channel -> background task
        self._last_candles: dict[str, dict[str, Any]] = {}   # channel -> latest normalized candle
        self._last_ticks: dict[str, MarketTick] = {}          # symbol -> latest tick
        self._market_provider = get_market_provider()
        self._lock = asyncio.Lock()
        self._running = True

    def _get_channel_key(self, symbol: str, timeframe: Any) -> str:
        tf_str = timeframe if isinstance(timeframe, str) else getattr(timeframe, "default", "1h")
        if not isinstance(tf_str, str):
            tf_str = "1h"
        canonical_tf = tf_str.lower() if tf_str.lower() in TIMEFRAME_SECONDS else "1h"
        return f"{symbol.upper()}:{canonical_tf}"

    async def register_client(self, websocket: Any, symbol: str, timeframe: str = "1h"):
        """Register a client websocket for a specific symbol & timeframe channel."""
        channel = self._get_channel_key(symbol, timeframe)
        async with self._lock:
            if channel not in self._active_connections:
                self._active_connections[channel] = set()
            self._active_connections[channel].add(websocket)

            # Ensure upstream stream is running for this channel
            if channel not in self._binance_tasks or self._binance_tasks[channel].done():
                self._binance_tasks[channel] = asyncio.create_task(
                    self._run_upstream_stream(symbol, timeframe, channel)
                )

        # Send initial tick snapshot immediately on connection for zero-latency client hydration
        try:
            initial_tick = await asyncio.to_thread(self._market_provider.get_ticker, symbol)
            await websocket.send_json({
                "type": "tick",
                "data": initial_tick.to_dict(),
            })
        except Exception:
            pass

        # Send cached latest candle immediately if available
        if channel in self._last_candles:
            try:
                await websocket.send_json({
                    "type": "candle",
                    "data": self._last_candles[channel],
                })
            except Exception:
                pass

    async def unregister_client(self, websocket: Any, symbol: str, timeframe: str = "1h"):
        """Unregister a client websocket."""
        channel = self._get_channel_key(symbol, timeframe)
        async with self._lock:
            if channel in self._active_connections:
                self._active_connections[channel].discard(websocket)
                if not self._active_connections[channel]:
                    # Cancel upstream stream if no clients are listening
                    task = self._binance_tasks.pop(channel, None)
                    if task and not task.done():
                        task.cancel()

    async def _broadcast_to_channel(self, channel: str, message: dict[str, Any]):
        """Broadcast a message to all connected clients on a channel."""
        conns = list(self._active_connections.get(channel, []))
        if not conns:
            return

        dead_conns = []
        for ws in conns:
            try:
                await ws.send_json(message)
            except Exception:
                dead_conns.append(ws)

        if dead_conns:
            async with self._lock:
                for ws in dead_conns:
                    self._active_connections.get(channel, set()).discard(ws)
                if not self._active_connections.get(channel):
                    task = self._binance_tasks.pop(channel, None)
                    if task and not task.done():
                        task.cancel()

    async def _run_upstream_stream(self, symbol: str, timeframe: str, channel: str):
        """Worker maintaining upstream real-time data stream."""
        asset_type = detect_asset_type(symbol)

        if asset_type == "crypto":
            await self._run_binance_stream(symbol, timeframe, channel)
        else:
            await self._run_non_crypto_stream(symbol, timeframe, channel)

    async def _run_binance_stream(self, symbol: str, timeframe: str, channel: str):
        """Stream genuine live klines directly from Binance WebSocket."""
        binance_sym = normalize_crypto_symbol_for_binance(symbol)
        interval = BINANCE_TIMEFRAME_MAP.get(timeframe.lower(), "1h")
        url = f"wss://stream.binance.com:9443/ws/{binance_sym}@kline_{interval}"

        backoff = 1.0
        while self._running:
            try:
                async with websockets.connect(
                    url,
                    ping_interval=20,
                    ping_timeout=10,
                    close_timeout=5,
                ) as ws:
                    backoff = 1.0
                    while self._running:
                        raw = await ws.recv()
                        data = json.loads(raw)
                        k = data.get("k", {})
                        if not k:
                            continue

                        # Extract exact candle fields
                        start_time_ms = int(k.get("t", 0))
                        start_time_sec = start_time_ms // 1000
                        o = float(k.get("o", 0.0))
                        h = float(k.get("h", 0.0))
                        l = float(k.get("l", 0.0))
                        c = float(k.get("c", 0.0))
                        v = float(k.get("v", 0.0))
                        is_closed = bool(k.get("x", False))

                        now_utc = datetime.now(timezone.utc)
                        iso_ts = datetime.fromtimestamp(start_time_sec, timezone.utc).isoformat()
                        ist_str = to_ist_string(iso_ts)

                        candle_payload = {
                            "symbol": symbol,
                            "timeframe": timeframe,
                            "time": start_time_sec,
                            "timestamp": iso_ts,
                            "timestamp_ms": start_time_ms,
                            "timestamp_ist": ist_str,
                            "open": o,
                            "high": h,
                            "low": l,
                            "close": c,
                            "volume": round(v, 4),
                            "is_closed": is_closed,
                            "data_status": "LIVE",
                            "freshness_text": f"LIVE • Updated {datetime.now(timezone.utc).strftime('%H:%M:%S')} UTC",
                            "provider": "binance_ws",
                        }

                        self._last_candles[channel] = candle_payload

                        # Also construct matching tick payload
                        tick_payload = {
                            "symbol": symbol,
                            "price": c,
                            "bid": c,
                            "ask": c,
                            "high_24h": h,
                            "low_24h": l,
                            "volume_24h": v,
                            "timestamp": now_utc.isoformat(),
                            "timestamp_ist": to_ist_string(now_utc.isoformat()),
                            "data_status": "LIVE",
                            "freshness_text": "LIVE • Updated sub-second",
                            "provider": "binance_ws",
                        }

                        # Broadcast to clients
                        await self._broadcast_to_channel(channel, {
                            "type": "candle",
                            "data": candle_payload,
                        })
                        await self._broadcast_to_channel(channel, {
                            "type": "tick",
                            "data": tick_payload,
                        })

            except asyncio.CancelledError:
                break
            except Exception:
                # WebSocket stream disconnected or unavailable.
                # Fallback to provider polling at <= 3.0s cadence so connected clients continue receiving updates
                try:
                    tick = await asyncio.to_thread(self._market_provider.get_ticker, symbol)
                    now_utc = datetime.now(timezone.utc)
                    now_sec = int(now_utc.timestamp())
                    interval_sec = TIMEFRAME_SECONDS.get(timeframe.lower(), 3600)
                    bar_time_sec = (now_sec // interval_sec) * interval_sec
                    iso_ts = datetime.fromtimestamp(bar_time_sec, timezone.utc).isoformat()

                    prev = self._last_candles.get(channel)
                    if prev and prev.get("time") == bar_time_sec:
                        o = prev["open"]
                        h = max(prev["high"], tick.price)
                        l = min(prev["low"], tick.price)
                        v = prev["volume"]
                    else:
                        o = tick.price
                        h = tick.high_24h or tick.price
                        l = tick.low_24h or tick.price
                        v = tick.volume_24h or 0.0

                    candle_payload = {
                        "symbol": symbol,
                        "timeframe": timeframe,
                        "time": bar_time_sec,
                        "timestamp": iso_ts,
                        "timestamp_ms": bar_time_sec * 1000,
                        "timestamp_ist": to_ist_string(iso_ts),
                        "open": round(o, 4),
                        "high": round(h, 4),
                        "low": round(l, 4),
                        "close": round(tick.price, 4),
                        "volume": round(v, 4),
                        "is_closed": False,
                        "data_status": tick.data_status,
                        "freshness_text": tick.freshness_text,
                        "provider": f"{tick.provider}_fallback_poll",
                    }
                    self._last_candles[channel] = candle_payload

                    await self._broadcast_to_channel(channel, {
                        "type": "candle",
                        "data": candle_payload,
                    })
                    await self._broadcast_to_channel(channel, {
                        "type": "tick",
                        "data": tick.to_dict(),
                    })
                except Exception:
                    pass
                await asyncio.sleep(2.5)

    async def _run_non_crypto_stream(self, symbol: str, timeframe: str, channel: str):
        """Streaming worker for equities, indices, and forex with accurate market states."""
        interval_sec = TIMEFRAME_SECONDS.get(timeframe.lower(), 3600)

        while self._running:
            try:
                # Fetch latest tick from provider
                tick = await asyncio.to_thread(self._market_provider.get_ticker, symbol)
                now_utc = datetime.now(timezone.utc)
                now_sec = int(now_utc.timestamp())

                # Compute current candle boundary in seconds
                bar_time_sec = (now_sec // interval_sec) * interval_sec
                iso_ts = datetime.fromtimestamp(bar_time_sec, timezone.utc).isoformat()

                # Get existing candle or start new one
                prev = self._last_candles.get(channel)
                if prev and prev.get("time") == bar_time_sec:
                    o = prev["open"]
                    h = max(prev["high"], tick.price)
                    l = min(prev["low"], tick.price)
                    v = prev["volume"]
                else:
                    o = tick.price
                    h = tick.price
                    l = tick.price
                    v = tick.volume_24h or 0.0

                c = tick.price
                status = tick.data_status or get_market_status(symbol)

                candle_payload = {
                    "symbol": symbol,
                    "timeframe": timeframe,
                    "time": bar_time_sec,
                    "timestamp": iso_ts,
                    "timestamp_ms": bar_time_sec * 1000,
                    "timestamp_ist": to_ist_string(iso_ts),
                    "open": round(o, 4),
                    "high": round(h, 4),
                    "low": round(l, 4),
                    "close": round(c, 4),
                    "volume": round(v, 4),
                    "is_closed": False,
                    "data_status": status,
                    "freshness_text": tick.freshness_text or f"{status} • {to_ist_string(tick.timestamp)}",
                    "provider": tick.provider or "yfinance",
                }

                self._last_candles[channel] = candle_payload

                await self._broadcast_to_channel(channel, {
                    "type": "candle",
                    "data": candle_payload,
                })
                await self._broadcast_to_channel(channel, {
                    "type": "tick",
                    "data": tick.to_dict(),
                })

            except asyncio.CancelledError:
                break
            except Exception:
                pass

            # Polling cadence: <= 3.0s (2.5s) regardless of open or closed market,
            # while preserving honest data_status (MARKET CLOSED / WEEKEND / STALE)
            await asyncio.sleep(2.5)


# Singleton instance
_stream_manager = RealTimeStreamManager()

def get_stream_manager() -> RealTimeStreamManager:
    return _stream_manager
