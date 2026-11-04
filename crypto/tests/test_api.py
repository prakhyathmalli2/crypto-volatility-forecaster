"""Unit tests for FastAPI serving endpoints and response schemas."""
from __future__ import annotations

import asyncio
import json
import os
from pathlib import Path
import pytest

from config import PAPER_DIR
from serving import inference_api
from serving.inference_api import (
    PredictRequest,
    TradeRequest,
    app,
    execute_paper_trade,
    get_health,
    get_latest_predict,
    get_paper_account_status,
    get_v1_forecast,
    get_v1_signal,
    predict_volatility,
)
from trading.execution import PaperBroker


async def _make_asgi_request(app_callable, method: str, path: str, headers: dict | None = None, body: dict | None = None):
    headers_list = [(b"host", b"testserver")]
    body_bytes = b""
    if body is not None:
        body_bytes = json.dumps(body).encode("utf-8")
        headers_list.append((b"content-type", b"application/json"))
        headers_list.append((b"content-length", str(len(body_bytes)).encode("latin1")))
    if headers:
        for k, v in headers.items():
            headers_list.append((k.lower().encode("latin1"), str(v).encode("latin1")))

    scope = {
        "type": "http",
        "asgi": {"version": "3.0"},
        "http_version": "1.1",
        "method": method.upper(),
        "path": path,
        "raw_path": path.encode("ascii"),
        "query_string": b"",
        "headers": headers_list,
        "server": ("testserver", 80),
        "client": ("testclient", 50000),
    }
    status_code = None
    response_body = []
    sent = False

    async def receive():
        nonlocal sent
        if not sent:
            sent = True
            return {"type": "http.request", "body": body_bytes, "more_body": False}
        return {"type": "http.request", "body": b"", "more_body": False}

    async def send(message):
        nonlocal status_code
        if message["type"] == "http.response.start":
            status_code = message["status"]
        elif message["type"] == "http.response.body":
            response_body.append(message.get("body", b""))

    await app_callable(scope, receive, send)
    raw = b"".join(response_body).decode("utf-8", errors="replace")
    try:
        data = json.loads(raw)
    except Exception:
        data = raw
    return status_code, data


def asgi_client(method: str, path: str, headers: dict | None = None, body: dict | None = None):
    return asyncio.run(_make_asgi_request(app, method, path, headers, body))


def test_api_health_endpoint():
    health = get_health()
    assert health.status == "ok"
    assert isinstance(health.models_ready, bool)
    assert isinstance(health.data_drift, dict)


def test_api_predict_endpoint():
    req = PredictRequest(symbol="BTC/USDT", timeframe="1h")
    resp = predict_volatility(req)

    assert resp.symbol == "BTC/USDT"
    assert resp.current_price > 0
    assert "1h" in resp.ensemble_forecasts
    assert "6h" in resp.ensemble_forecasts
    assert "24h" in resp.ensemble_forecasts

    f1 = resp.ensemble_forecasts["1h"]
    assert f1.volatility > 0
    assert f1.uncertainty_lower <= f1.volatility <= f1.uncertainty_upper
    assert 0.0 <= f1.directional_probability <= 1.0
    assert resp.trading_signal["action"] in ["BUY", "SELL", "HOLD", "REDUCE"]


def test_api_predict_latest_get():
    resp = get_latest_predict(symbol="BTC/USDT", timeframe="1h")
    assert "predictions" in resp
    assert "1h" in resp["predictions"]
    assert "signal" in resp
    assert resp["current_price"] > 0


def test_api_paper_trading_read_only_unauthenticated(tmp_path, monkeypatch):
    test_broker = PaperBroker(initial_capital=10000.0, path=tmp_path / "temp_account.json")
    monkeypatch.setattr(inference_api, "_paper_broker", test_broker)

    # Public read-only endpoints require no API key
    status_code, data = asgi_client("GET", "/api/v1/paper/account")
    assert status_code == 200
    assert data["paper_trading_enabled"]
    assert "portfolio" in data

    status_code, data = asgi_client("GET", "/api/v1/paper/portfolio")
    assert status_code == 200
    assert "portfolio" in data

    status_code, data = asgi_client("GET", "/paper/status")
    assert status_code == 200
    assert "portfolio" in data


def test_api_paper_mutations_auth_enforcement(tmp_path, monkeypatch):
    persistent_account_path = PAPER_DIR / "paper_account.json"
    persistent_content_before = persistent_account_path.read_text(encoding="utf-8") if persistent_account_path.exists() else None

    test_ledger = tmp_path / "temp_paper_account.json"
    test_broker = PaperBroker(initial_capital=10000.0, path=test_ledger)
    monkeypatch.setattr(inference_api, "_paper_broker", test_broker)

    valid_key = "secure-test-key-555"
    monkeypatch.setenv("AGY_API_KEY", valid_key)
    monkeypatch.setattr(inference_api, "API_SECRET_KEY", valid_key)

    # 1. Missing key is rejected on all mutation endpoints
    status_code, err = asgi_client("POST", "/api/v1/paper/order", body={"side": "BUY", "quantity": 0.01, "price": 60000.0, "symbol": "BTC/USDT"})
    assert status_code == 401
    assert "X-API-Key" in err.get("detail", "")
    assert test_broker.cash == 10000.0  # Broker state unaltered

    status_code, err = asgi_client("POST", "/api/v1/paper/reset")
    assert status_code == 401
    assert "X-API-Key" in err.get("detail", "")

    status_code, err = asgi_client("POST", "/paper/trade", body={"side": "BUY", "quantity": 0.01, "price": 60000.0, "symbol": "BTC/USDT"})
    assert status_code == 401
    assert "X-API-Key" in err.get("detail", "")
    assert test_broker.cash == 10000.0  # Broker state unaltered

    # 2. Invalid key is rejected on all mutation endpoints
    bad_headers = {"X-API-Key": "invalid-wrong-key"}
    status_code, err = asgi_client("POST", "/api/v1/paper/order", headers=bad_headers, body={"side": "BUY", "quantity": 0.01, "price": 60000.0, "symbol": "BTC/USDT"})
    assert status_code == 401
    assert test_broker.cash == 10000.0  # Broker state unaltered

    status_code, err = asgi_client("POST", "/api/v1/paper/reset", headers=bad_headers)
    assert status_code == 401

    status_code, err = asgi_client("POST", "/paper/trade", headers=bad_headers, body={"side": "BUY", "quantity": 0.01, "price": 60000.0, "symbol": "BTC/USDT"})
    assert status_code == 401
    assert test_broker.cash == 10000.0  # Broker state unaltered

    # 3. Valid key authorizes mutations
    good_headers = {"X-API-Key": valid_key}
    status_code, res = asgi_client("POST", "/api/v1/paper/order", headers=good_headers, body={"side": "BUY", "quantity": 0.01, "price": 60000.0, "symbol": "BTC/USDT"})
    assert status_code == 200
    assert test_broker.cash < 10000.0  # Cash reduced by trade cost
    assert "BTC/USDT" in test_broker.positions

    status_code, res = asgi_client("POST", "/paper/trade", headers=good_headers, body={"side": "BUY", "quantity": 0.005, "price": 60000.0, "symbol": "BTC/USDT"})
    assert status_code == 200

    status_code, res = asgi_client("POST", "/api/v1/paper/reset", headers=good_headers)
    assert status_code == 200
    assert test_broker.cash == 10000.0  # Cash reset
    assert not test_broker.positions

    # 4. Fail closed when key is unconfigured on server
    monkeypatch.delenv("AGY_API_KEY", raising=False)
    monkeypatch.delenv("API_SECRET_KEY", raising=False)
    monkeypatch.setattr(inference_api, "API_SECRET_KEY", "")

    status_code, err = asgi_client("POST", "/api/v1/paper/order", headers=good_headers, body={"side": "BUY", "quantity": 0.01, "price": 60000.0, "symbol": "BTC/USDT"})
    assert status_code == 401
    assert "not configured" in err.get("detail", "").lower()

    # 5. Assert the persistent account file remained completely untouched
    if persistent_content_before is not None:
        assert persistent_account_path.read_text(encoding="utf-8") == persistent_content_before


def test_api_health_degraded_when_scaler_invalid(monkeypatch):
    class BadScaler:
        def transform(self, X):
            raise ValueError("Corrupt feature vector")

    saved_models = dict(inference_api._models)
    fake_models = dict(saved_models)
    fake_models["scaler"] = BadScaler()
    monkeypatch.setattr(inference_api, "_models", fake_models)

    health = get_health()
    assert health.status == "degraded"
    assert health.models_ready is False
    assert health.feature_scaler_valid is False
    assert "scaler_error" in health.model_status or health.model_status.get("scaler") != "valid"


def test_api_predict_and_forecast_fallback_truthfulness(monkeypatch):
    # 1. Normal execution
    req = PredictRequest(symbol="BTC/USDT", timeframe="1h")
    resp = predict_volatility(req)
    assert resp.inference_mode in ("ml_ensemble", "empirical_fallback")
    assert isinstance(resp.fallback_used, bool)

    fc = get_v1_forecast("BTC/USDT", "1h")
    assert "inference_mode" in fc
    assert "fallback_used" in fc
    assert fc["fallback_used"] == resp.fallback_used

    sig = get_v1_signal("BTC/USDT", "1h", force_refresh=True)
    assert "inference_mode" in sig
    assert "fallback_used" in sig

    # 2. Simulated failure triggers empirical fallback with truthful reporting
    class BrokenModel:
        def predict(self, X):
            raise RuntimeError("CUDA out of memory simulation")

    saved_models = dict(inference_api._models)
    broken_models = dict(saved_models)
    if "btc_usdt" in broken_models and isinstance(broken_models["btc_usdt"], dict):
        broken_asset = dict(broken_models["btc_usdt"])
        broken_asset["lstm"] = BrokenModel()
        broken_models["btc_usdt"] = broken_asset
    broken_models["lstm"] = BrokenModel()
    monkeypatch.setattr(inference_api, "_models", broken_models)

    fallback_resp = predict_volatility(req)
    assert fallback_resp.fallback_used is True
    assert fallback_resp.inference_mode == "empirical_fallback"
    assert "Model execution failed" in (fallback_resp.fallback_reason or "")

    fc_fallback = get_v1_forecast("BTC/USDT", "1h")
    assert fc_fallback["fallback_used"] is True
    assert fc_fallback["inference_mode"] == "empirical_fallback"

