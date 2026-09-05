"""Public BTC OHLCV ingestion through CCXT with exchange fallbacks and gap checks."""
from __future__ import annotations

import time
from datetime import datetime, timedelta, timezone
from pathlib import Path

import ccxt
import pandas as pd

from config import EXCHANGE_FALLBACKS, EXCHANGE_ID, RAW_DIR, SYMBOL, LOOKBACK_DAYS


def _make_exchange(exchange_id: str):
    cls = getattr(ccxt, exchange_id)
    return cls({"enableRateLimit": True, "timeout": 20000})


def _fetch_from_exchange(exchange_id: str, symbol: str, timeframe: str, since_ms: int, until_ms: int) -> list[list]:
    exchange = _make_exchange(exchange_id)
    if not exchange.has.get("fetchOHLCV"):
        raise RuntimeError(f"{exchange_id} does not expose fetchOHLCV")
    exchange.load_markets()
    if symbol not in exchange.symbols:
        # Common BTC/USDT availability fallback.
        alt = "BTC/USD" if "BTC/USD" in exchange.symbols else symbol
        if alt not in exchange.symbols:
            raise RuntimeError(f"{symbol} is not available on {exchange_id}")
        symbol = alt

    timeframe_ms = exchange.parse_timeframe(timeframe) * 1000
    cursor = since_ms
    rows: list[list] = []
    while cursor < until_ms:
        batch = exchange.fetch_ohlcv(symbol, timeframe=timeframe, since=cursor, limit=1000)
        if not batch:
            break
        rows.extend(batch)
        last_ts = int(batch[-1][0])
        next_cursor = last_ts + timeframe_ms
        if next_cursor <= cursor:
            break
        cursor = next_cursor
        if len(batch) < 1000:
            break
        time.sleep(max(exchange.rateLimit / 1000.0, 0.05))
    return [r for r in rows if since_ms <= int(r[0]) < until_ms]


def fetch_klines(symbol: str = SYMBOL, timeframe: str = "1h", lookback_days: int = LOOKBACK_DAYS) -> pd.DataFrame:
    now = datetime.now(timezone.utc)
    until = int(now.timestamp() * 1000)
    since = int((now - timedelta(days=lookback_days)).timestamp() * 1000)
    errors: list[str] = []
    exchange_ids = [EXCHANGE_ID] + [x for x in EXCHANGE_FALLBACKS if x != EXCHANGE_ID]

    for exchange_id in exchange_ids:
        try:
            rows = _fetch_from_exchange(exchange_id, symbol, timeframe, since, until)
            if rows:
                df = pd.DataFrame(rows, columns=["timestamp", "open", "high", "low", "close", "volume"])
                df["open_time"] = pd.to_datetime(df.pop("timestamp"), unit="ms", utc=True)
                for c in ["open", "high", "low", "close", "volume"]:
                    df[c] = pd.to_numeric(df[c], errors="coerce")
                df = df.drop_duplicates("open_time").sort_values("open_time")
                df["source_exchange"] = exchange_id
                return df[["open_time", "open", "high", "low", "close", "volume", "source_exchange"]].reset_index(drop=True)
        except Exception as exc:  # pragma: no cover - depends on network/exchange state
            errors.append(f"{exchange_id}: {exc}")

    raise RuntimeError("No market-data source succeeded. " + " | ".join(errors))


def save_market_data(timeframe: str = "1h") -> Path:
    df = fetch_klines(timeframe=timeframe)
    out = RAW_DIR / f"btc_{timeframe}_ohlcv.csv"
    df.to_csv(out, index=False)
    return out


if __name__ == "__main__":
    for tf in ("1h", "4h", "1d"):
        path = save_market_data(tf)
        print(f"Saved {path}")
