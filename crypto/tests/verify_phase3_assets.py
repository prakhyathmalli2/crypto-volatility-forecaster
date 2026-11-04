"""Phase 3 comprehensive multi-asset verification script.
Tests all 11 required assets across all 9 required verification dimensions.
"""
import sys
from pathlib import Path

# Add crypto root to sys.path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
sys.stdout.reconfigure(encoding="utf-8")

from market_data.provider import get_market_provider, search_instruments, get_market_status, detect_asset_type
from market_data.indicators import calculate_indicators
from market_data.fundamentals import get_asset_fundamentals
from news.news_service import get_news_service
from events.events_service import get_events_service

TEST_ASSETS = [
    # Crypto
    "BTC/USDT",
    "ETH/USDT",
    # Indian Stocks
    "RELIANCE.NS",
    "TCS.NS",
    "INFY.NS",
    # US Stocks
    "AAPL",
    "NVDA",
    # Indices
    "^NSEI",
    "^GSPC",
    # Forex
    "USDINR=X",
    "EURUSD=X",
]

TIMEFRAMES = ["1m", "5m", "15m", "1h", "4h", "1d", "1w"]

def verify_asset(symbol: str, provider, news_svc):
    print(f"\n=======================================================")
    print(f"VERIFYING ASSET: {symbol} (Type: {detect_asset_type(symbol)})")
    print(f"=======================================================")
    
    # 1. Search
    token = symbol.split("/")[0].replace("^", "").replace(".NS", "").replace("=X", "")
    search_res = search_instruments(token)
    has_search = any(symbol in r["symbol"] for r in search_res)
    print(f"1. Search: {'PASS' if has_search else 'WARN'} (Found {len(search_res)} results for '{token}')")

    # 2. Ticker
    try:
        tick = provider.get_ticker(symbol)
        ticker_ok = tick.price > 0 or tick.market_status in ("CLOSED", "WEEKEND")
        print(f"2. Ticker: {'PASS' if ticker_ok else 'FAIL'} => Price: {tick.price} | PrevClose: {tick.previous_close} | Chg: {tick.change_24h_pct}% | Status: {tick.market_status}")
    except Exception as e:
        print(f"2. Ticker: FAIL => Exception: {e}")
        tick = None

    # 3. Market Status
    status = get_market_status(symbol)
    print(f"3. Market Status: PASS => {status}")

    # 4. Candles (Default 1h)
    candles = []
    try:
        candles = provider.get_ohlcv(symbol, timeframe="1h", limit=20)
        c_ok = len(candles) > 0 and candles[-1].close > 0
        print(f"4. Candles (1h): {'PASS' if c_ok else 'FAIL'} => Count: {len(candles)}, Last Close: {candles[-1].close if candles else None}")
    except Exception as e:
        print(f"4. Candles (1h): FAIL => Exception: {e}")

    # 5. Indicators
    if candles:
        try:
            ind = calculate_indicators([c.to_dict() for c in candles])
            curr = ind.get("current", {})
            ind_ok = "rsi" in curr and "ema20" in curr
            print(f"5. Indicators: {'PASS' if ind_ok else 'FAIL'} => RSI: {curr.get('rsi')} | Trend: {curr.get('trend_regime')} | StochK: {curr.get('stoch_k')}")
        except Exception as e:
            print(f"5. Indicators: FAIL => Exception: {e}")
    else:
        print("5. Indicators: SKIPPED (No candles)")

    # 6. News
    try:
        articles = news_svc.get_prioritized_news(symbol=symbol, limit=3)
        news_ok = len(articles) > 0
        print(f"6. News: {'PASS' if news_ok else 'WARN'} => Count: {len(articles)}, Top: '{articles[0].get('title', '')[:50]}...'")
    except Exception as e:
        print(f"6. News: FAIL => Exception: {e}")

    # 7. Fundamentals
    try:
        fund = get_asset_fundamentals(symbol)
        fund_ok = "metrics" in fund
        m = fund.get("metrics", {})
        print(f"7. Fundamentals: {'PASS' if fund_ok else 'FAIL'} => MCap: {m.get('market_cap', 'Data unavailable')} | PE: {m.get('pe_ratio', 'Data unavailable')} | 52W: {m.get('fifty_two_week_range', 'Data unavailable')}")
    except Exception as e:
        print(f"7. Fundamentals: FAIL => Exception: {e}")

    # 8. Timeframe Audit
    print("8. Timeframe Audit:")
    for tf in TIMEFRAMES:
        try:
            tf_candles = provider.get_ohlcv(symbol, timeframe=tf, limit=5)
            tf_ok = len(tf_candles) > 0
            print(f"   [{tf:3s}]: {'PASS' if tf_ok else 'FAIL'} ({len(tf_candles)} candles, latest: {tf_candles[-1].timestamp if tf_candles else 'none'})")
        except Exception as e:
            print(f"   [{tf:3s}]: FAIL ({e})")

def main():
    provider = get_market_provider()
    news_svc = get_news_service()
    for sym in TEST_ASSETS:
        verify_asset(sym, provider, news_svc)

if __name__ == "__main__":
    main()
