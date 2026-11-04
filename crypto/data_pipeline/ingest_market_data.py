"""Multi-source cryptocurrency market data ingestion pipeline.

Fetches multi-year historical OHLCV data for BTC and ETH from Binance via CCXT
with automatic rate-limit retries, pagination, and a reliable yfinance fallback.
Data is saved as partitioned Parquet and CSV files under data/raw/market/ with
metadata versioning.
"""
from __future__ import annotations

import json
import time
from datetime import datetime, timedelta, timezone
from pathlib import Path
from typing import Any

import ccxt
import pandas as pd
from tenacity import retry, retry_if_exception_type, stop_after_attempt, wait_exponential

from config import (
    ASSETS,
    DEFAULT_TIMEFRAME,
    EXCHANGE_FALLBACKS,
    EXCHANGE_ID,
    LOOKBACK_DAYS,
    MARKET_RAW_DIR,
)


def _make_exchange(exchange_id: str) -> ccxt.Exchange:
    """Instantiate CCXT exchange with rate-limiting enabled."""
    cls = getattr(ccxt, exchange_id, None)
    if cls is None:
        raise ValueError(f"Unknown exchange: {exchange_id}")
    return cls({"enableRateLimit": True, "timeout": 25000})


@retry(
    stop=stop_after_attempt(4),
    wait=wait_exponential(multiplier=1, min=2, max=10),
    retry=retry_if_exception_type((ccxt.NetworkError, ccxt.RateLimitExceeded)),
    reraise=True,
)
def _fetch_ohlcv_batch(
    exchange: ccxt.Exchange, symbol: str, timeframe: str, since_ms: int, limit: int = 1000
) -> list[list]:
    """Fetch single batch of OHLCV candles with exponential backoff on network/rate-limit errors."""
    return exchange.fetch_ohlcv(symbol, timeframe=timeframe, since=since_ms, limit=limit)


def _fetch_from_exchange(
    exchange_id: str, symbol: str, timeframe: str, since_ms: int, until_ms: int
) -> list[list]:
    """Fetch full historical window by paginating through exchange OHLCV."""
    exchange = _make_exchange(exchange_id)
    if not exchange.has.get("fetchOHLCV"):
        raise RuntimeError(f"{exchange_id} does not support fetchOHLCV")

    exchange.load_markets()
    # Resolve symbol or fallback (e.g. BTC/USD if BTC/USDT not listed)
    resolved_symbol = symbol
    if symbol not in exchange.symbols:
        base, quote = symbol.split("/")
        alt_symbol = f"{base}/USD"
        if alt_symbol in exchange.symbols:
            resolved_symbol = alt_symbol
        else:
            raise RuntimeError(f"Symbol {symbol} not available on {exchange_id}")

    timeframe_ms = exchange.parse_timeframe(timeframe) * 1000
    cursor = since_ms
    rows: list[list] = []

    while cursor < until_ms:
        batch = _fetch_ohlcv_batch(
            exchange, resolved_symbol, timeframe, since_ms=cursor, limit=1000
        )
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


def _fetch_from_yfinance(
    symbol: str, timeframe: str, lookback_days: int
) -> pd.DataFrame:
    """Fallback fetcher using yfinance if exchange endpoints are unavailable."""
    import yfinance as yf

    # Map CCXT symbols to Yahoo tickers
    ticker_map = {
        "BTC/USDT": "BTC-USD",
        "ETH/USDT": "ETH-USD",
        "BTC/USD": "BTC-USD",
        "ETH/USD": "ETH-USD",
    }
    ticker = ticker_map.get(symbol, symbol.replace("/", "-"))

    # Map timeframes to yfinance intervals
    interval_map = {"1h": "1h", "4h": "1h", "1d": "1d"}
    interval = interval_map.get(timeframe, "1h")

    # yfinance 1h data is restricted to the last 730 days
    days = min(lookback_days, 725) if interval == "1h" else lookback_days
    end_dt = datetime.now(timezone.utc)
    start_dt = end_dt - timedelta(days=days)

    df = yf.download(
        ticker,
        start=start_dt.strftime("%Y-%m-%d"),
        end=end_dt.strftime("%Y-%m-%d"),
        interval=interval,
        progress=False,
        auto_adjust=False,
    )

    if df.empty:
        raise RuntimeError(f"yfinance returned empty data for {ticker}")

    # Flatten MultiIndex columns if present
    if isinstance(df.columns, pd.MultiIndex):
        df.columns = [c[0].lower() for c in df.columns]
    else:
        df.columns = [c.lower() for c in df.columns]

    df = df.reset_index()
    time_col = "Datetime" if "Datetime" in df.columns else "Date"
    df["open_time"] = pd.to_datetime(df[time_col], utc=True)

    # If 4h was requested and we downloaded 1h, resample to 4h
    if timeframe == "4h" and interval == "1h":
        df = (
            df.set_index("open_time")
            .resample("4h")
            .agg(
                {
                    "open": "first",
                    "high": "max",
                    "low": "min",
                    "close": "last",
                    "volume": "sum",
                }
            )
            .dropna()
            .reset_index()
        )

    for col in ["open", "high", "low", "close", "volume"]:
        df[col] = pd.to_numeric(df[col], errors="coerce")

    df["source_exchange"] = "yfinance"
    return df[
        ["open_time", "open", "high", "low", "close", "volume", "source_exchange"]
    ].dropna().reset_index(drop=True)


def fetch_historical_ohlcv(
    symbol: str = "BTC/USDT",
    timeframe: str = DEFAULT_TIMEFRAME,
    lookback_days: int = LOOKBACK_DAYS,
) -> tuple[pd.DataFrame, dict[str, Any]]:
    """Fetch historical OHLCV data with exchange prioritization and yfinance fallback."""
    now = datetime.now(timezone.utc)
    until_ms = int(now.timestamp() * 1000)
    since_ms = int((now - timedelta(days=lookback_days)).timestamp() * 1000)

    exchange_candidates = [EXCHANGE_ID] + [
        x for x in EXCHANGE_FALLBACKS if x != EXCHANGE_ID
    ]
    errors: list[str] = []

    # 1. Try exchange endpoints via CCXT
    for exc_id in exchange_candidates:
        try:
            rows = _fetch_from_exchange(exc_id, symbol, timeframe, since_ms, until_ms)
            if rows and len(rows) >= 50:
                df = pd.DataFrame(
                    rows,
                    columns=["timestamp", "open", "high", "low", "close", "volume"],
                )
                df["open_time"] = pd.to_datetime(df.pop("timestamp"), unit="ms", utc=True)
                for c in ["open", "high", "low", "close", "volume"]:
                    df[c] = pd.to_numeric(df[c], errors="coerce")
                df = df.drop_duplicates("open_time").sort_values("open_time")
                df["source_exchange"] = exc_id
                clean_df = df[
                    ["open_time", "open", "high", "low", "close", "volume", "source_exchange"]
                ].reset_index(drop=True)

                metadata = {
                    "symbol": symbol,
                    "timeframe": timeframe,
                    "source": exc_id,
                    "download_time_utc": now.isoformat(),
                    "date_range": [
                        str(clean_df["open_time"].iloc[0]),
                        str(clean_df["open_time"].iloc[-1]),
                    ],
                    "number_of_rows": len(clean_df),
                }
                return clean_df, metadata
        except Exception as exc:
            errors.append(f"{exc_id}: {exc}")

    # 2. Fallback to yfinance
    try:
        df = _fetch_from_yfinance(symbol, timeframe, lookback_days)
        metadata = {
            "symbol": symbol,
            "timeframe": timeframe,
            "source": "yfinance",
            "download_time_utc": now.isoformat(),
            "date_range": [
                str(df["open_time"].iloc[0]),
                str(df["open_time"].iloc[-1]),
            ],
            "number_of_rows": len(df),
        }
        return df, metadata
    except Exception as exc:
        errors.append(f"yfinance: {exc}")

    raise RuntimeError(
        f"Failed to fetch market data for {symbol} ({timeframe}). Errors: "
        + " | ".join(errors)
    )


def save_market_dataset(
    symbol: str = "BTC/USDT",
    timeframe: str = "1h",
    lookback_days: int = LOOKBACK_DAYS,
) -> dict[str, Path]:
    """Ingest and save market data in both Parquet and CSV formats with metadata."""
    df, metadata = fetch_historical_ohlcv(symbol, timeframe, lookback_days)
    base_name = f"{symbol.replace('/', '_').lower()}_{timeframe}"

    # Partition / save paths under data/raw/market/
    csv_path = MARKET_RAW_DIR / f"{base_name}.csv"
    parquet_path = MARKET_RAW_DIR / f"{base_name}.parquet"
    meta_path = MARKET_RAW_DIR / f"{base_name}_meta.json"

    df.to_csv(csv_path, index=False)
    try:
        df.to_parquet(parquet_path, index=False)
    except Exception:
        pass  # Parquet optional if pyarrow dll blocked

    meta_path.write_text(json.dumps(metadata, indent=2), encoding="utf-8")
    return {"csv": csv_path, "parquet": parquet_path, "meta": meta_path}


def ingest_all_market_data():
    """Ingest standard BTC and ETH pairs across target timeframes."""
    results = {}
    for sym in ASSETS:
        for tf in ("1h", "4h", "1d"):
            print(f"[Ingest Market] Fetching {sym} ({tf})...")
            paths = save_market_dataset(sym, tf)
            results[f"{sym}_{tf}"] = paths
            print(f"  -> Saved {paths['csv']}")
    return results


if __name__ == "__main__":
    ingest_all_market_data()
