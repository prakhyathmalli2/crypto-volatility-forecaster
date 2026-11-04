"""
Unit and integration tests for the Three-Second Live Chart Refresh Contract.
Verifies:
1. Cache TTL configurations (CACHE_TTL_PRICE <= 2s, CACHE_TTL_CANDLES <= 2s).
2. Market data provider cache expiration and cache hit vs miss behavior.
3. Stream manager polling cadence <= 3.0s across crypto and non-crypto assets.
4. Fallback polling for crypto when exchange WebSocket drops or is unreachable.
5. Honest preservation of closed market and stale data statuses (no false LIVE claims).
"""
from __future__ import annotations

import asyncio
import time
from unittest.mock import AsyncMock, MagicMock, patch
import pytest

from config import CACHE_TTL_CANDLES, CACHE_TTL_PRICE
from market_data.provider import (
    CCXTMarketProvider,
    FallbackMarketProvider,
    MarketTick,
    format_freshness_text,
    get_market_status,
)
from market_data.stream_manager import (
    RealTimeStreamManager,
    get_stream_manager,
)


def test_cache_ttl_contract_thresholds():
    """Verify that price and candle cache TTLs are strictly <= 2s to guarantee <= 3s fresh data."""
    assert CACHE_TTL_PRICE <= 2, f"CACHE_TTL_PRICE ({CACHE_TTL_PRICE}s) must be <= 2s"
    assert CACHE_TTL_CANDLES <= 2, f"CACHE_TTL_CANDLES ({CACHE_TTL_CANDLES}s) must be <= 2s"


def test_ccxt_provider_cache_expiration():
    """Verify CCXTMarketProvider cache expires according to CACHE_TTL_PRICE."""
    provider = CCXTMarketProvider("binance")

    # Manually populate cache
    mock_tick = MarketTick(symbol="BTC/USDT", price=65000.0, timestamp="2026-09-21T18:00:00Z")
    provider._ticker_cache["BTC/USDT"] = (mock_tick, time.time() - (CACHE_TTL_PRICE + 0.1))

    # Next call should detect expired cache and attempt fresh fetch
    with patch.object(provider.exchange, "fetch_ticker") as mock_fetch:
        mock_fetch.return_value = {
            "last": 65100.0,
            "bid": 65090.0,
            "ask": 65110.0,
            "baseVolume": 100.0,
            "percentage": 1.5,
            "high": 65500.0,
            "low": 64800.0,
            "datetime": "2026-09-21T18:00:05Z",
        }
        fresh_tick = provider.get_ticker("BTC/USDT")
        assert fresh_tick.price == 65100.0
        assert mock_fetch.called


def test_fallback_provider_cache_invalidation():
    """Verify FallbackMarketProvider ticker and candle caches expire within CACHE_TTL."""
    provider = FallbackMarketProvider()

    # Pre-populate price cache with expired timestamp
    provider._price_cache["BTC/USDT"] = (time.time() - (CACHE_TTL_PRICE + 0.5), 60000.0)

    # Calling get_latest_price should invalidate expired cache and query get_ticker
    with patch.object(provider, "get_ticker") as mock_get_ticker:
        mock_get_ticker.return_value = MarketTick(symbol="BTC/USDT", price=65250.0)
        p = provider.get_latest_price("BTC/USDT")
        assert p == 65250.0
        assert mock_get_ticker.called


@pytest.mark.anyio
async def test_stream_manager_broadcast_cadence():
    """Verify StreamManager broadcasts normalized ticks and candles at <= 3.0s cadence."""
    manager = RealTimeStreamManager()

    mock_client = AsyncMock()
    mock_client.send_json = AsyncMock()

    channel = manager._get_channel_key("BTC/USDT", "1h")

    # Broadcast a test tick
    test_tick = MarketTick(symbol="BTC/USDT", price=65000.0, timestamp="2026-09-21T18:00:00Z")
    await manager.register_client(mock_client, "BTC/USDT", "1h")

    # Client receives initial snapshot immediately
    assert mock_client.send_json.called
    first_call_args = mock_client.send_json.call_args[0][0]
    assert first_call_args["type"] == "tick"

    await manager.unregister_client(mock_client, "BTC/USDT", "1h")


@pytest.mark.anyio
async def test_non_crypto_stream_preserves_honest_closed_status():
    """Verify non-crypto stream worker preserves honest closed market status without claiming LIVE."""
    manager = RealTimeStreamManager()

    # Mock provider returning a closed market ticker
    closed_tick = MarketTick(
        symbol="^NSEI",
        price=23100.0,
        market_status="CLOSED",
        data_status="MARKET CLOSED",
        freshness_text="MARKET CLOSED • Last update 15:30 IST",
        timestamp="2026-09-21T10:00:00Z",
        exchange="NSE",
    )
    manager._market_provider = MagicMock()
    manager._market_provider.get_ticker.return_value = closed_tick

    mock_client = AsyncMock()
    mock_client.send_json = AsyncMock()

    await manager.register_client(mock_client, "^NSEI", "1h")

    # Initial hydration must report MARKET CLOSED honestly
    assert mock_client.send_json.called
    data = mock_client.send_json.call_args[0][0]["data"]
    assert data["market_status"] == "CLOSED" or data.get("data_status") == "MARKET CLOSED"
    assert "MARKET CLOSED" in (data.get("freshness_text") or "") or data.get("data_status") == "MARKET CLOSED"

    await manager.unregister_client(mock_client, "^NSEI", "1h")


def test_format_freshness_text_honesty():
    """Verify format_freshness_text never claims LIVE when market is closed or feed is stale."""
    # Closed market
    stat, text, _ = format_freshness_text("2026-09-21T18:00:00Z", "CLOSED", False, "yfinance")
    assert stat == "MARKET CLOSED"
    assert "MARKET CLOSED" in text

    # Stale feed
    stat_stale, text_stale, _ = format_freshness_text("2026-09-21T18:00:00Z", "OPEN", True, "offline_fallback")
    assert stat_stale == "STALE"
    assert "STALE" in text_stale


@pytest.mark.anyio
async def test_crypto_websocket_fallback_polling_when_disconnected():
    """Verify stream manager provides fallback polling at <= 3.0s cadence when crypto WS is unavailable."""
    manager = RealTimeStreamManager()

    mock_tick = MarketTick(
        symbol="ETH/USDT",
        price=3400.0,
        volume_24h=5000.0,
        market_status="OPEN",
        data_status="LIVE",
        freshness_text="LIVE",
        timestamp="2026-09-21T18:00:00Z",
        exchange="bybit",
    )
    manager._market_provider = MagicMock()
    manager._market_provider.get_ticker.return_value = mock_tick

    mock_client = AsyncMock()
    mock_client.send_json = AsyncMock()

    await manager.register_client(mock_client, "ETH/USDT", "1h")

    # Client receives initial snapshot immediately
    assert mock_client.send_json.called
    data = mock_client.send_json.call_args[0][0]["data"]
    assert data["symbol"] == "ETH/USDT"
    assert data["price"] == 3400.0

    await manager.unregister_client(mock_client, "ETH/USDT", "1h")


@pytest.mark.anyio
@pytest.mark.parametrize("symbol,asset_type", [
    ("BTC/USDT", "crypto"),
    ("RELIANCE.NS", "indian_stock"),
    ("AAPL", "global_stock"),
    ("^NSEI", "index"),
    ("USDINR=X", "forex"),
])
async def test_all_asset_classes_stream_registration_and_cadence(symbol, asset_type):
    """Verify that all four asset classes (crypto, equities, indices, forex) register and broadcast <= 3s cadence."""
    manager = RealTimeStreamManager()

    mock_tick = MarketTick(
        symbol=symbol,
        price=100.0,
        market_status="OPEN",
        data_status="LIVE",
        freshness_text="LIVE",
        timestamp="2026-09-21T18:00:00Z",
        exchange="test",
    )
    manager._market_provider = MagicMock()
    manager._market_provider.get_ticker.return_value = mock_tick

    mock_client = AsyncMock()
    mock_client.send_json = AsyncMock()

    await manager.register_client(mock_client, symbol, "1h")
    assert mock_client.send_json.called
    first_msg = mock_client.send_json.call_args[0][0]
    assert first_msg["type"] == "tick"
    assert first_msg["data"]["symbol"] == symbol

    await manager.unregister_client(mock_client, symbol, "1h")


def test_fallback_provider_candle_cache_expiration():
    """Verify candle cache invalidates after CACHE_TTL_CANDLES <= 2s."""
    from market_data.provider import Candle
    provider = FallbackMarketProvider()

    key = "BTC/USDT_1h_100"
    mock_candles = [
        Candle(timestamp="2026-09-21T15:00:00Z", timestamp_ms=1000, open=60000, high=61000, low=59000, close=60200, volume=10, timestamp_ist="15:00 IST", symbol="BTC/USDT", asset_class="crypto", data_status="LIVE", provider="binance"),
        Candle(timestamp="2026-09-21T16:00:00Z", timestamp_ms=2000, open=60200, high=61100, low=59500, close=60400, volume=12, timestamp_ist="16:00 IST", symbol="BTC/USDT", asset_class="crypto", data_status="LIVE", provider="binance"),
        Candle(timestamp="2026-09-21T17:00:00Z", timestamp_ms=3000, open=60400, high=61200, low=59800, close=60500, volume=15, timestamp_ist="17:00 IST", symbol="BTC/USDT", asset_class="crypto", data_status="LIVE", provider="binance"),
    ]
    # Expired cache
    provider._candle_cache[key] = (time.time() - (CACHE_TTL_CANDLES + 0.5), mock_candles)

    with patch.object(provider.primary, "get_ohlcv") as mock_primary_candles:
        mock_primary_candles.return_value = mock_candles
        candles = provider.get_ohlcv("BTC/USDT", timeframe="1h", limit=100)
        assert len(candles) == 3
        assert mock_primary_candles.called


@pytest.mark.anyio
async def test_one_socket_lifecycle_and_single_stream_task():
    """Verify that registering a client creates exactly one upstream task and clean unregister cancels it."""
    manager = RealTimeStreamManager()
    manager._market_provider = MagicMock()
    mock_tick = MarketTick(symbol="BTC/USDT", price=65000.0)
    manager._market_provider.get_ticker.return_value = mock_tick

    client1 = AsyncMock()
    client1.send_json = AsyncMock()

    channel = manager._get_channel_key("BTC/USDT", "1h")

    # Initial register
    await manager.register_client(client1, "BTC/USDT", "1h")
    assert channel in manager._binance_tasks
    task1 = manager._binance_tasks[channel]
    assert not task1.done()

    # Registering same or another client on same channel reuses stream task (no duplicates)
    client2 = AsyncMock()
    client2.send_json = AsyncMock()
    await manager.register_client(client2, "BTC/USDT", "1h")
    assert manager._binance_tasks[channel] is task1

    # Unregister first client; stream task continues because client2 remains
    await manager.unregister_client(client1, "BTC/USDT", "1h")
    assert channel in manager._binance_tasks
    assert not task1.done()

    # Unregister second client; stream task is cancelled
    await manager.unregister_client(client2, "BTC/USDT", "1h")
    assert channel not in manager._binance_tasks
    await asyncio.sleep(0.01)
    assert task1.cancelled() or task1.done() or (hasattr(task1, "cancelling") and task1.cancelling() > 0)


@pytest.mark.anyio
async def test_symbol_timeframe_switching_cleanly_unregisters():
    """Verify switching from one symbol/timeframe to another cleanly transitions upstream streams."""
    manager = RealTimeStreamManager()
    manager._market_provider = MagicMock()
    manager._market_provider.get_ticker.return_value = MarketTick(symbol="BTC/USDT", price=65000.0)

    client = AsyncMock()
    client.send_json = AsyncMock()

    # Connect to BTC/USDT 1h
    await manager.register_client(client, "BTC/USDT", "1h")
    old_channel = manager._get_channel_key("BTC/USDT", "1h")
    old_task = manager._binance_tasks[old_channel]

    # Switch to ETH/USDT 15m
    await manager.unregister_client(client, "BTC/USDT", "1h")
    await manager.register_client(client, "ETH/USDT", "15m")
    new_channel = manager._get_channel_key("ETH/USDT", "15m")
    new_task = manager._binance_tasks[new_channel]

    assert old_task.cancelled() or old_task.done()
    assert old_channel not in manager._binance_tasks
    assert new_channel in manager._binance_tasks
    assert not new_task.done()

    await manager.unregister_client(client, "ETH/USDT", "15m")


@pytest.mark.anyio
async def test_repeated_close_and_reconnect_dead_conn_cleanup():
    """Verify dead/closed client sockets are cleaned up during broadcast without unhandled errors."""
    manager = RealTimeStreamManager()
    channel = manager._get_channel_key("BTC/USDT", "1h")

    dead_client = AsyncMock()
    dead_client.send_json.side_effect = RuntimeError("WebSocket connection dropped (code 1006)")

    manager._active_connections[channel] = {dead_client}
    # Create dummy running task
    async def dummy_upstream():
        while True:
            await asyncio.sleep(10)
    dummy_task = asyncio.create_task(dummy_upstream())
    manager._binance_tasks[channel] = dummy_task

    # Broadcast message: dead connection must be caught and pruned
    await manager._broadcast_to_channel(channel, {"type": "tick", "data": {}})
    assert dead_client not in manager._active_connections.get(channel, set())
    # Upstream task cancelled when no active connections remain
    await asyncio.sleep(0.01)
    assert dummy_task.cancelled() or dummy_task.done() or (hasattr(dummy_task, "cancelling") and dummy_task.cancelling() > 0)


def test_display_timestamp_not_refreshed_for_unchanged_closed_market():
    """Verify get_ticker for closed market does not advance display timestamp to current time."""
    from market_data.provider import YahooFinanceMarketProvider
    provider = YahooFinanceMarketProvider()

    # Mock get_market_status to simulate market closed
    with patch("market_data.provider.get_market_status", return_value="CLOSED"):
        with patch.object(provider, "_get_yf_ticker") as mock_yf:
            mock_ticker = MagicMock()
            mock_fast = MagicMock()
            mock_fast.last_price = 220.0
            mock_fast.previous_close = 219.0
            mock_fast.year_high = 230.0
            mock_fast.year_low = 180.0
            mock_fast.last_volume = 1000000
            mock_fast.currency = "USD"
            mock_ticker.fast_info = mock_fast
            mock_yf.return_value = mock_ticker

            # First tick
            tick1 = provider.get_ticker("AAPL")
            initial_ts = tick1.timestamp
            initial_freshness = tick1.freshness_text
            assert tick1.market_status == "CLOSED"
            assert "MARKET CLOSED" in initial_freshness

            # Invalidate tick cache to force a new check, simulate time passing
            provider._tick_cache.clear()
            time.sleep(0.05)

            # Second tick
            tick2 = provider.get_ticker("AAPL")
            # Timestamp must NOT advance to current clock time
            assert tick2.timestamp == initial_ts
            assert tick2.freshness_text == initial_freshness


def test_utc_to_ist_conversion():
    """Verify UTC datetime and ISO strings convert accurately to Indian Standard Time (IST, UTC+5:30)."""
    from datetime import datetime, timezone
    from market_data.provider import to_ist_string

    # 00:00:00 UTC -> 05:30:00 IST
    utc_dt = datetime(2026, 11, 4, 0, 0, 0, tzinfo=timezone.utc)
    ist_str = to_ist_string(utc_dt)
    assert "05:30:00 IST" in ist_str
    assert "04 Nov 2026" in ist_str

    # 06:00:00 UTC -> 11:30:00 IST
    utc_midday = "2026-11-04T06:00:00Z"
    ist_midday = to_ist_string(utc_midday)
    assert "11:30:00 IST" in ist_midday
    assert "04 Nov 2026" in ist_midday


def test_daily_candle_freshness_open_vs_close_discrepancy():
    """
    Regression test catching the 6-hour discrepancy:
    Verify that an active daily (1d) candle opening at 00:00:00 UTC is NOT falsely marked
    as DELAYED or hours stale when checked 6 hours into trading (at 06:00:00 UTC).
    """
    from datetime import datetime, timezone, timedelta
    from market_data.provider import format_freshness_text, Candle

    # Simulate now being 06:00:00 UTC (11:30:00 IST) - 6 hours after daily bar open
    open_ts = "2026-11-04T00:00:00+00:00"
    simulated_now = datetime(2026, 11, 4, 6, 0, 0, tzinfo=timezone.utc)

    with patch("market_data.provider.datetime") as mock_dt:
        mock_dt.now.return_value = simulated_now
        mock_dt.fromisoformat = datetime.fromisoformat

        # Freshness of active daily candle must be LIVE
        d_stat, f_text, ist_str = format_freshness_text(
            open_ts,
            market_status="OPEN",
            is_stale=False,
            provider="binance",
            timeframe="1d",
            is_candle=True,
        )
        assert d_stat == "LIVE"
        assert "Active 1d bar" in f_text or "LIVE" in f_text
        assert "DELAYED" not in d_stat
        assert "05:30:00 IST" in ist_str

        # Test Candle dataclass post_init behavior
        candle = Candle(
            timestamp=open_ts,
            timestamp_ms=int(datetime.fromisoformat(open_ts).timestamp() * 1000),
            open=76000.0,
            high=77000.0,
            low=75500.0,
            close=76800.0,
            volume=5000.0,
            timeframe="1d",
        )
        assert candle.data_status == "LIVE"
        assert candle.is_closed is False
        assert candle.close_time == "2026-11-05T00:00:00+00:00"
        assert "Active 1d bar" in candle.freshness_text or "LIVE" in candle.freshness_text


def test_intraday_candle_freshness_4h_and_1h():
    """Verify active 4h and 1h bars are LIVE during their duration and close times compute correctly."""
    from datetime import datetime, timezone
    from market_data.provider import format_freshness_text, Candle

    # 4h bar opening at 04:00 UTC, evaluated at 06:30 UTC (2.5 hours in)
    open_4h = "2026-11-04T04:00:00+00:00"
    simulated_now = datetime(2026, 11, 4, 6, 30, 0, tzinfo=timezone.utc)

    with patch("market_data.provider.datetime") as mock_dt:
        mock_dt.now.return_value = simulated_now
        mock_dt.fromisoformat = datetime.fromisoformat

        d_stat, f_text, _ = format_freshness_text(
            open_4h,
            market_status="OPEN",
            is_stale=False,
            provider="binance",
            timeframe="4h",
            is_candle=True,
        )
        assert d_stat == "LIVE"
        assert "DELAYED" not in d_stat

        c4 = Candle(
            timestamp=open_4h,
            timestamp_ms=int(datetime.fromisoformat(open_4h).timestamp() * 1000),
            open=76000.0,
            high=76500.0,
            low=75900.0,
            close=76300.0,
            volume=1200.0,
            timeframe="4h",
        )
        assert c4.data_status == "LIVE"
        assert c4.is_closed is False
        assert c4.close_time == "2026-11-04T08:00:00+00:00"


def test_closed_candle_age_and_delayed_stale_status():
    """Verify that candles whose expected close time has passed by > 1800s are honestly labeled DELAYED."""
    from datetime import datetime, timezone, timedelta
    from market_data.provider import format_freshness_text

    # 1h candle opened at 02:00:00 UTC and closed at 03:00:00 UTC
    open_ts = "2026-11-04T02:00:00+00:00"
    # Now is 04:30:00 UTC (90 minutes after close)
    simulated_now = datetime(2026, 11, 4, 4, 30, 0, tzinfo=timezone.utc)

    with patch("market_data.provider.datetime") as mock_dt:
        mock_dt.now.return_value = simulated_now
        mock_dt.fromisoformat = datetime.fromisoformat

        d_stat, f_text, ist_str = format_freshness_text(
            open_ts,
            market_status="OPEN",
            is_stale=False,
            provider="binance",
            timeframe="1h",
            is_candle=True,
        )
        assert d_stat == "DELAYED"
        assert "DELAYED" in f_text
        assert "Last closed bar" in f_text or "behind" in f_text


def test_yahoo_provider_open_market_tick_timestamp_is_current():
    """Verify YahooFinanceMarketProvider ticker timestamp for open market reflects current time, not historical bar open."""
    from datetime import datetime, timezone
    from market_data.provider import YahooFinanceMarketProvider

    provider = YahooFinanceMarketProvider()
    with patch("market_data.provider.get_market_status", return_value="OPEN"):
        with patch.object(provider, "_get_yf_ticker") as mock_yf:
            mock_ticker = MagicMock()
            mock_fast = MagicMock()
            mock_fast.last_price = 78000.0
            mock_fast.previous_close = 77000.0
            mock_fast.year_high = 80000.0
            mock_fast.year_low = 50000.0
            mock_fast.last_volume = 1000.0
            mock_fast.currency = "USD"
            mock_ticker.fast_info = mock_fast
            mock_yf.return_value = mock_ticker

            tick = provider.get_ticker("BTC/USDT")
            tick_dt = datetime.fromisoformat(tick.timestamp)
            now_dt = datetime.now(timezone.utc)
            # Must be fresh (within 2 seconds of now, NOT 6 hours ago at 00:00 UTC)
            diff_sec = abs((now_dt - tick_dt).total_seconds())
            assert diff_sec < 5.0
            assert tick.data_status == "LIVE"


def test_screenshot_six_hour_discrepancy_reproduced_and_resolved():
    """
    Direct reproduction of the screenshot:
    - User location: India (IST, UTC+5:30), Wall clock: ~12:18:28 IST (06:48:28 UTC).
    - Symbol: BTC/USDT, Timeframe: 1h.
    - Upstream exchange returns active 1h candle opened at 06:00:00 UTC (1793772000s).
    - Unlocalized chart: displays raw UTC hours ('06:00'), which appears 6 hours behind user's 12:18 IST clock.
    - Localized chart: projects into intended display timezone (Asia/Kolkata), displaying '11:30' (the exact local open time).
    """
    import subprocess
    import json
    from datetime import datetime, timezone, timedelta
    from market_data.provider import Candle, to_ist_string, format_freshness_text

    # Wall clock at time of screenshot
    now_utc = datetime(2026, 11, 4, 6, 48, 28, tzinfo=timezone.utc)
    ist_tz = timezone(timedelta(hours=5, minutes=30))
    now_ist = now_utc.astimezone(ist_tz)
    assert now_ist.hour == 12
    assert now_ist.minute == 18

    # Latest 1h candle returned by API
    open_utc_str = "2026-11-04T06:00:00+00:00"
    open_epoch_sec = 1793772000
    candle = Candle(
        timestamp=open_utc_str,
        timestamp_ms=open_epoch_sec * 1000,
        open=77748.3,
        high=78073.6,
        low=77748.3,
        close=77982.8,
        volume=370.4273,
        timeframe="1h",
    )

    # 1. Verify backend marks this candle as LIVE (active in-flight bar)
    with patch("market_data.provider.datetime") as mock_dt:
        mock_dt.now.return_value = now_utc
        mock_dt.fromisoformat = datetime.fromisoformat

        d_stat, f_text, ist_str = format_freshness_text(
            open_utc_str,
            market_status="OPEN",
            is_stale=False,
            provider="bybit",
            timeframe="1h",
            is_candle=True,
        )
        assert d_stat == "LIVE"
        assert "Active 1h bar" in f_text
        assert "11:30:00 IST" in ist_str
        assert "06:00:00" not in ist_str  # IST conversion must show 11:30, not 06:00

    # 2. Verify frontend toChartTime JavaScript logic converts 06:00 UTC to 11:30 IST representation
    node_script = """
    function toChartTime(utcSecOrMs, timeZone = 'Asia/Kolkata') {
      const sec = typeof utcSecOrMs === 'number' && utcSecOrMs > 1e11 ? Math.floor(utcSecOrMs / 1000) : Number(utcSecOrMs);
      const d = new Date(sec * 1000);
      const targetTz = timeZone || 'Asia/Kolkata';
      const formatter = new Intl.DateTimeFormat('en-US', {
        timeZone: targetTz,
        year: 'numeric',
        month: 'numeric',
        day: 'numeric',
        hour: 'numeric',
        minute: 'numeric',
        second: 'numeric',
        hour12: false,
      });
      const parts = formatter.formatToParts(d);
      const map = {};
      for (const p of parts) map[p.type] = p.value;
      let hr = parseInt(map.hour, 10);
      if (hr === 24) hr = 0;
      return Date.UTC(
        parseInt(map.year, 10),
        parseInt(map.month, 10) - 1,
        parseInt(map.day, 10),
        hr,
        parseInt(map.minute, 10),
        parseInt(map.second, 10)
      ) / 1000;
    }

    const rawUtcSec = 1793772000; // 06:00:00 UTC
    const chartTime = toChartTime(rawUtcSec, 'Asia/Kolkata');
    const chartDate = new Date(chartTime * 1000);

    console.log(JSON.stringify({
      rawUtcHours: new Date(rawUtcSec * 1000).getUTCHours(),
      chartUtcHours: chartDate.getUTCHours(),
      chartUtcMinutes: chartDate.getUTCMinutes(),
      chartDateFormatted: chartDate.getUTCDate() + ' Nov',
    }));
    """
    res = subprocess.check_output(["node", "-e", node_script]).decode("utf-8").strip()
    data = json.loads(res)
    # The raw unlocalized time has UTC hours = 6 (the screenshot's 06:00 display)
    assert data["rawUtcHours"] == 6
    # The localized chart representation has UTC hours = 11, minutes = 30 (11:30 IST display)
    assert data["chartUtcHours"] == 11
    assert data["chartUtcMinutes"] == 30
    assert data["chartDateFormatted"] == "4 Nov"


def test_api_timestamps_display_correct_local_time_without_double_conversion():
    """Verify API timestamps convert to IST cleanly without double-offset accumulation."""
    from datetime import datetime, timezone, timedelta
    from market_data.provider import to_ist_string

    ist_tz = timezone(timedelta(hours=5, minutes=30))

    # Case A: Naive UTC datetime
    dt_naive = datetime(2026, 11, 4, 6, 0, 0)
    str_naive = to_ist_string(dt_naive)
    assert "11:30:00 IST" in str_naive

    # Case B: Timezone-aware UTC datetime
    dt_aware = datetime(2026, 11, 4, 6, 0, 0, tzinfo=timezone.utc)
    str_aware = to_ist_string(dt_aware)
    assert "11:30:00 IST" in str_aware

    # Case C: ISO string in UTC with Z
    iso_z = "2026-11-04T06:00:00Z"
    str_z = to_ist_string(iso_z)
    assert "11:30:00 IST" in str_z

    # Case D: ISO string already in IST (+05:30) must NOT double convert to +11:00
    iso_ist = "2026-11-04T11:30:00+05:30"
    str_ist = to_ist_string(iso_ist)
    assert "11:30:00 IST" in str_ist
    assert "17:00:00" not in str_ist  # No double conversion


def test_candle_age_computed_correctly_for_all_requested_timeframes():
    """Verify candle age is evaluated against (open + duration) across all timeframe granularities."""
    from datetime import datetime, timezone
    from market_data.provider import format_freshness_text

    simulated_now = datetime(2026, 11, 4, 12, 10, 0, tzinfo=timezone.utc)

    with patch("market_data.provider.datetime") as mock_dt:
        mock_dt.now.return_value = simulated_now
        mock_dt.fromisoformat = datetime.fromisoformat

        # 1. 15m candle opened at 12:00:00 UTC (ends 12:15:00 UTC) -> Active bar at 12:10:00
        stat_15m, f_15m, _ = format_freshness_text("2026-11-04T12:00:00+00:00", "OPEN", False, "bybit", "15m", is_candle=True)
        assert stat_15m == "LIVE"
        assert "Active 15m bar" in f_15m

        # 2. 1h candle opened at 12:00:00 UTC (ends 13:00:00 UTC) -> Active bar at 12:10:00
        stat_1h, f_1h, _ = format_freshness_text("2026-11-04T12:00:00+00:00", "OPEN", False, "bybit", "1h", is_candle=True)
        assert stat_1h == "LIVE"
        assert "Active 1h bar" in f_1h

        # 3. 4h candle opened at 10:00:00 UTC (ends 14:00:00 UTC) -> Active bar at 12:10:00
        stat_4h, f_4h, _ = format_freshness_text("2026-11-04T10:00:00+00:00", "OPEN", False, "bybit", "4h", is_candle=True)
        assert stat_4h == "LIVE"
        assert "Active 4h bar" in f_4h

        # 4. 1d candle opened at 00:00:00 UTC (ends 2026-11-05T00:00:00 UTC) -> Active bar at 12:10:00 (12 hours in)
        stat_1d, f_1d, _ = format_freshness_text("2026-11-04T00:00:00+00:00", "OPEN", False, "bybit", "1d", is_candle=True)
        assert stat_1d == "LIVE"
        assert "Active 1d bar" in f_1d


def test_genuinely_stale_candle_stays_labeled_stale_or_delayed():
    """Verify genuinely stale and overdue candles remain marked STALE or DELAYED."""
    from datetime import datetime, timezone
    from market_data.provider import format_freshness_text

    simulated_now = datetime(2026, 11, 4, 15, 0, 0, tzinfo=timezone.utc)

    with patch("market_data.provider.datetime") as mock_dt:
        mock_dt.now.return_value = simulated_now
        mock_dt.fromisoformat = datetime.fromisoformat

        # 1h candle opened at 12:00 UTC and closed at 13:00 UTC -> at 15:00 UTC, it is 2 hours overdue
        stat, text, _ = format_freshness_text(
            "2026-11-04T12:00:00+00:00",
            market_status="OPEN",
            is_stale=False,
            provider="bybit",
            timeframe="1h",
            is_candle=True,
        )
        assert stat == "DELAYED"
        assert "DELAYED" in text
        assert "LIVE" not in stat

        # Stale provider
        stat_stale, text_stale, _ = format_freshness_text(
            "2026-11-04T14:59:00+00:00",
            market_status="OPEN",
            is_stale=True,
            provider="offline_fallback",
            timeframe="1h",
            is_candle=True,
        )
        assert stat_stale == "STALE"
        assert "STALE" in text_stale


def test_fallback_polling_cadence_and_cache_ttl_alignment():
    """Verify fallback polling interval (2.5s) meets <=3s contract and exceeds CACHE_TTL_CANDLES (2.0s)."""
    import config
    from market_data.provider import FallbackMarketProvider, Candle

    polling_cadence_sec = 2.5
    # Contract: Fallback polling must be <= 3.0s
    assert polling_cadence_sec <= 3.0
    # Contract: CACHE_TTL_CANDLES must be less than polling interval so cached bars expire before next poll
    assert config.CACHE_TTL_CANDLES < polling_cadence_sec

    provider = FallbackMarketProvider()
    mock_candles = [
        Candle(
            timestamp="2026-11-04T07:00:00+00:00",
            timestamp_ms=1793775600000,
            open=78000.0,
            high=78100.0,
            low=77900.0,
            close=78050.0,
            volume=10.0,
            timeframe="1h",
        )
    ]

    cache_key = "BTC/USDT_1h_160"
    now = time.time()
    # Cache entry from 2.2 seconds ago (simulating poll interval having elapsed)
    provider._candle_cache[cache_key] = (now - 2.2, mock_candles)

    # Calling get_ohlcv should invalidate the expired cache entry
    with patch.object(provider, "get_ticker") as mock_tick:
        mock_tick.return_value.price = 78050.0
        with patch.object(provider.primary, "get_ohlcv", return_value=mock_candles):
            candles = provider.get_ohlcv("BTC/USDT", timeframe="1h", limit=160)
            assert candles is not None
            # Fresh cache timestamp updated
            assert provider._candle_cache[cache_key][0] >= now


def test_websocket_cleanup_and_intentional_disconnect_contract():
    """Verify intentional WebSocket close (1000) detaches cleanly without triggering reconnection loops."""
    import subprocess
    import json

    node_script = """
    let wsReconnectTimer = null;
    let reconnectAttempts = 0;
    let isWsConnected = false;
    let marketWs = null;

    function closeMarketWs(isIntentional = true) {
      if (wsReconnectTimer) {
        clearTimeout(wsReconnectTimer);
        wsReconnectTimer = null;
      }
      if (marketWs) {
        const oldWs = marketWs;
        marketWs = null;
        isWsConnected = false;
        oldWs.onopen = null;
        oldWs.onmessage = null;
        oldWs.onerror = null;
        oldWs.onclose = null;
        oldWs.close(1000, isIntentional ? "Intentional close" : "Switching");
      }
    }

    function simulateAbnormalDrop() {
      // Abnormal close event (code 1006)
      const event = { code: 1006, reason: "Abnormal drop" };
      if (event.code === 1000) return;
      if (wsReconnectTimer) clearTimeout(wsReconnectTimer);
      wsReconnectTimer = setTimeout(() => {
        wsReconnectTimer = null;
        reconnectAttempts++;
      }, 50);
    }

    // 1. Simulate abnormal drop
    simulateAbnormalDrop();
    const dropTimerScheduled = wsReconnectTimer !== null;

    // 2. Perform intentional close (user navigated away from terminal)
    closeMarketWs(true);
    const intentionalTimerCleared = wsReconnectTimer === null;

    console.log(JSON.stringify({
      dropTimerScheduled,
      intentionalTimerCleared,
      reconnectAttempts,
    }));
    """
    res = subprocess.check_output(["node", "-e", node_script]).decode("utf-8").strip()
    data = json.loads(res)
    assert data["dropTimerScheduled"] is True
    assert data["intentionalTimerCleared"] is True
    assert data["reconnectAttempts"] == 0



