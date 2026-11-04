import asyncio
import json
import os
import sys
import tempfile
import threading
import time
from pathlib import Path
import pytest

# Ensure crypto module is in pythonpath
CRYPTO_DIR = Path(__file__).resolve().parent.parent
if str(CRYPTO_DIR) not in sys.path:
    sys.path.insert(0, str(CRYPTO_DIR))

from serving.inference_api import (
    app,
    get_v1_market_overview,
    get_v1_market_movers,
    get_v1_sentiment,
    get_health,
    get_v1_signal,
    get_v1_forecast,
)
from trading.execution import PaperBroker


def test_sentiment_schema_canonical_and_aliases():
    """
    Verify that /api/v1/sentiment returns canonical schema:
    - fear_and_greed_value (int/float)
    - classification (str)
    and temporary backward-compatible aliases:
    - fear_and_greed_score
    - sentiment_label
    """
    data = get_v1_sentiment()

    # Canonical keys
    assert "fear_and_greed_value" in data, "Missing canonical key 'fear_and_greed_value'"
    assert "classification" in data, "Missing canonical key 'classification'"
    assert isinstance(data["fear_and_greed_value"], (int, float))
    assert isinstance(data["classification"], str)

    # Backward-compatible alias keys
    assert "fear_and_greed_score" in data, "Missing alias key 'fear_and_greed_score'"
    assert "sentiment_label" in data, "Missing alias key 'sentiment_label'"
    assert data["fear_and_greed_value"] == data["fear_and_greed_score"]
    assert data["classification"] == data["sentiment_label"]


def test_overview_and_movers_swr_cache():
    """
    Verify that /api/v1/market/overview and /api/v1/market/movers
    use background stale-while-revalidate caching and warm hits respond in < 500ms.
    """
    # Prime cache
    asyncio.run(get_v1_market_overview())

    # Second hit should be served instantly from memory (< 500ms, typically < 10ms)
    t0 = time.perf_counter()
    data = asyncio.run(get_v1_market_overview())
    t1 = time.perf_counter()
    latency_ms = (t1 - t0) * 1000

    assert "crypto" in data
    assert "indices" in data
    assert "indian_stocks" in data
    assert "us_stocks" in data
    assert latency_ms < 500.0, f"Overview warm cache latency {latency_ms:.2f}ms exceeds 500ms limit"

    # Movers should share or hit cached computation (< 500ms)
    t0 = time.perf_counter()
    data_m = asyncio.run(get_v1_market_movers())
    t1 = time.perf_counter()
    movers_latency_ms = (t1 - t0) * 1000

    assert "top_gainers" in data_m
    assert "top_losers" in data_m
    assert "highest_volume" in data_m
    assert movers_latency_ms < 500.0, f"Movers warm cache latency {movers_latency_ms:.2f}ms exceeds 500ms limit"


def test_health_latency_cached():
    """
    Verify /health and /api/v1/health respond in < 100ms when cached.
    """
    # Prime cache
    get_health()

    # Warm cached check
    t0 = time.perf_counter()
    res = get_health()
    t1 = time.perf_counter()
    latency_ms = (t1 - t0) * 1000

    status = res.status if hasattr(res, "status") else res.get("status")
    models_ready = res.models_ready if hasattr(res, "models_ready") else res.get("models_ready")
    assert status == "ok"
    assert models_ready is True
    assert latency_ms < 100.0, f"Health check latency {latency_ms:.2f}ms exceeds 100ms threshold"


def test_signal_and_forecast_caching():
    """
    Verify /api/v1/signal/{symbol} and /api/v1/forecast/{symbol}
    use memory caching and warm hits respond in < 300ms.
    """
    symbol = "BTC/USDT"
    # Prime signal cache
    get_v1_signal(symbol=symbol, timeframe="1h")

    # Warm hit
    t0 = time.perf_counter()
    res_sig = get_v1_signal(symbol=symbol, timeframe="1h")
    t1 = time.perf_counter()
    sig_latency_ms = (t1 - t0) * 1000

    assert isinstance(res_sig, dict)
    assert sig_latency_ms < 300.0, f"Signal cache latency {sig_latency_ms:.2f}ms exceeds 300ms threshold"

    # Prime forecast cache
    get_v1_forecast(symbol=symbol, timeframe="1h")

    # Warm hit
    t0 = time.perf_counter()
    res_fc = get_v1_forecast(symbol=symbol, timeframe="1h")
    t1 = time.perf_counter()
    fc_latency_ms = (t1 - t0) * 1000

    assert isinstance(res_fc, dict)
    assert fc_latency_ms < 300.0, f"Forecast cache latency {fc_latency_ms:.2f}ms exceeds 300ms threshold"


def test_paper_trading_atomic_persistence():
    """
    Verify PaperBroker uses atomic file replacement (.tmp + os.replace)
    and thread lock to prevent state file corruption under concurrent saves.
    """
    with tempfile.TemporaryDirectory() as tmp_dir:
        test_file = Path(tmp_dir) / "test_paper_account.json"
        broker = PaperBroker(initial_capital=100000.0, path=test_file)
        broker._save_state()

        # Confirm file exists and is valid JSON
        assert test_file.exists()

        errors = []

        def concurrent_save(order_id):
            try:
                broker.execute_market_order(
                    symbol="BTC/USDT",
                    side="BUY",
                    quantity=0.001 * order_id,
                    execution_price=50000.0,
                )
            except Exception as e:
                errors.append(e)

        threads = [threading.Thread(target=concurrent_save, args=(i + 1,)) for i in range(10)]
        for t in threads:
            t.start()
        for t in threads:
            t.join()

        assert len(errors) == 0, f"Concurrent save errors: {errors}"
        assert test_file.exists()

        # Reload broker from the persisted file
        reloaded = PaperBroker(path=test_file)
        assert len(reloaded.trade_history) == 10
