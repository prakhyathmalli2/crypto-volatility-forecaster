"""Phase 2 & Phase 3 comprehensive backend verification script.
Tests all endpoints over real HTTP, measures latencies, and verifies WebSockets with real client.
"""
import asyncio
import json
import time
import urllib.request
import urllib.parse
import sys

# Ensure UTF-8 stdout
sys.stdout.reconfigure(encoding="utf-8")

BASE_URL = "http://127.0.0.1:8000"

def test_http_endpoint(method: str, path: str, body: dict | None = None) -> tuple[int, float, dict | list]:
    url = f"{BASE_URL}{path}"
    headers = {"Content-Type": "application/json"}
    data = json.dumps(body).encode("utf-8") if body else None
    req = urllib.request.Request(url, data=data, headers=headers, method=method)
    
    t0 = time.perf_counter()
    try:
        with urllib.request.urlopen(req, timeout=15) as resp:
            elapsed_ms = (time.perf_counter() - t0) * 1000.0
            content = resp.read().decode("utf-8")
            parsed = json.loads(content)
            return resp.status, elapsed_ms, parsed
    except urllib.error.HTTPError as e:
        elapsed_ms = (time.perf_counter() - t0) * 1000.0
        return e.code, elapsed_ms, {"error": e.reason}
    except Exception as e:
        elapsed_ms = (time.perf_counter() - t0) * 1000.0
        return 500, elapsed_ms, {"error": str(e)}

async def test_websocket_client(path: str, timeout: float = 5.0) -> dict:
    """Connect to WebSocket endpoint and read first message."""
    try:
        import websockets
        ws_url = f"ws://127.0.0.1:8000{path}"
        t0 = time.perf_counter()
        async with websockets.connect(ws_url, close_timeout=2) as ws:
            raw = await asyncio.wait_for(ws.recv(), timeout=timeout)
            elapsed_ms = (time.perf_counter() - t0) * 1000.0
            return {"status": "ok", "latency_ms": round(elapsed_ms, 2), "data": json.loads(raw)}
    except ImportError:
        return {"status": "skipped", "error": "websockets package not installed in python env"}
    except Exception as e:
        return {"status": "error", "error": str(e)}

def run_phase2_verification():
    endpoints = [
        ("GET", "/health", None),
        ("GET", "/api/v1/market/search?q=reliance", None),
        ("GET", "/api/v1/market/BTC%2FUSDT/ticker", None),
        ("GET", "/api/v1/market/RELIANCE.NS/ticker", None),
        ("GET", "/api/v1/market/BTC%2FUSDT/candles?timeframe=1h&limit=10", None),
        ("GET", "/api/v1/market/RELIANCE.NS/candles?timeframe=1h&limit=10", None),
        ("GET", "/api/v1/market/BTC%2FUSDT/indicators?timeframe=1h", None),
        ("GET", "/api/v1/market/RELIANCE.NS/indicators?timeframe=1h", None),
        ("GET", "/api/v1/fundamentals/RELIANCE.NS", None),
        ("GET", "/api/v1/signal/BTC%2FUSDT", None),
        ("GET", "/api/v1/signal/RELIANCE.NS", None),
        ("GET", "/api/v1/news?symbol=BTC%2FUSDT", None),
        ("GET", "/api/v1/news?symbol=RELIANCE.NS", None),
        ("GET", "/api/v1/events", None),
        ("POST", "/api/v1/chat", {"query": "Why is this moving?", "symbol": "BTC/USDT"}),
    ]

    print("=================================================================")
    print("PHASE 2: REAL HTTP ENDPOINT VERIFICATION (LIVE SERVER)")
    print("=================================================================")
    all_ok = True
    for method, path, body in endpoints:
        status, ms, data = test_http_endpoint(method, path, body)
        is_ok = (status == 200)
        if not is_ok:
            all_ok = False
        sample = str(data)[:70] + "..." if len(str(data)) > 70 else str(data)
        print(f"[{'PASS' if is_ok else 'FAIL'}] {status} ({ms:6.1f}ms) | {method} {path} => {sample}")

    print("\n=================================================================")
    print("WEBSOCKET STREAMING VERIFICATION")
    print("=================================================================")
    ws_market = asyncio.run(test_websocket_client("/ws/market/BTC%2FUSDT"))
    print("WS /ws/market/BTC%2FUSDT =>", ws_market)
    ws_signals = asyncio.run(test_websocket_client("/ws/signals/BTC%2FUSDT"))
    print("WS /ws/signals/BTC%2FUSDT =>", ws_signals)
    ws_notif = asyncio.run(test_websocket_client("/ws/notifications"))
    print("WS /ws/notifications =>", ws_notif)

if __name__ == "__main__":
    run_phase2_verification()
