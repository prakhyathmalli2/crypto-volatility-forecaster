"""FastAPI serving layer for live BTC forecasts and paper-trading controls."""
from __future__ import annotations

import threading
from datetime import datetime, timezone

import joblib
import numpy as np
import pandas as pd
from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field
import ccxt

from config import DEFAULT_TIMEFRAME, MODEL_DIR, PAPER_TRADING_ENABLED, SEQ_LEN, SYMBOL, INITIAL_CAPITAL
from data_pipeline.feature_engineering import engineer_features
from data_pipeline.ingest_market_data import fetch_klines
from data_pipeline.preprocess import add_targets, merge_sources
from data_pipeline.ingest_onchain_data import build_onchain_dataset
from data_pipeline.ingest_sentiment_data import fetch_news, hourly_sentiment_index
from models.train import FEATURE_COLS
from models.garch_model import fit_garch, forecast_volatility
from trading.risk_manager import RiskManager
from trading.signal_generator import SignalGenerator
from trading.execution import PaperBroker

app = FastAPI(title="Crypto Volatility Forecast & Smart Signal Generator", version="2.0.0")
app.add_middleware(CORSMiddleware, allow_origins=["*"], allow_methods=["*"], allow_headers=["*"])

_model_lock = threading.Lock()
_models = None
_signal = SignalGenerator()
_broker = PaperBroker(INITIAL_CAPITAL)


class TradeRequest(BaseModel):
    side: str = Field(pattern="^(BUY|SELL|REDUCE)$")
    quantity: float = Field(gt=0)
    price: float = Field(gt=0)


def _load_models():
    global _models
    with _model_lock:
        if _models is None:
            paths = [MODEL_DIR / "feature_scaler.pkl", MODEL_DIR / "lstm_model.keras", MODEL_DIR / "transformer_model.keras", MODEL_DIR / "garch_btc.pkl"]
            missing = [str(p) for p in paths if not p.exists()]
            if missing:
                raise FileNotFoundError("Model artifacts missing: " + ", ".join(missing))
            import tensorflow as tf
            _models = {
                "scaler": joblib.load(paths[0]),
                "lstm": tf.keras.models.load_model(paths[1]),
                "transformer": tf.keras.models.load_model(paths[2]),
                "garch": joblib.load(paths[3]),
            }
        return _models


def _feature_frame(timeframe: str = DEFAULT_TIMEFRAME, limit_days: int = 45):
    # fetch_klines is currently configured in days; 45d is enough for live feature context across all requested timeframes.
    raw = fetch_klines(SYMBOL, timeframe=timeframe, lookback_days=limit_days)
    onchain = None
    sentiment = None
    try:
        onchain = build_onchain_dataset("90days")
    except Exception:
        pass
    try:
        sentiment = hourly_sentiment_index(fetch_news())
    except Exception:
        pass
    merged = merge_sources(raw, onchain, sentiment)
    engineered = engineer_features(merged)
    for c in FEATURE_COLS:
        if c not in engineered.columns:
            engineered[c] = 0.0
    return engineered


def _predict(df: pd.DataFrame):
    models = _load_models()
    tail = df.iloc[-SEQ_LEN:]
    scaled = models["scaler"].transform(tail[FEATURE_COLS])
    X = scaled.reshape(1, SEQ_LEN, len(FEATURE_COLS))
    lstm_pred = models["lstm"].predict(X, verbose=0)
    trans_pred = models["transformer"].predict(X, verbose=0)

    lvol = np.asarray(lstm_pred["volatility"])[0]
    tvol = np.asarray(trans_pred["volatility"])[0]
    ldir = np.asarray(lstm_pred["direction"])[0]
    tdir = np.asarray(trans_pred["direction"])[0]

    garch = forecast_volatility(models["garch"], (1, 6, 24))
    neural_vol = (lvol + tvol) / 2
    garch_vol = np.array([garch[h] for h in (1, 6, 24)], dtype=float)
    vol = 0.65 * neural_vol + 0.35 * garch_vol
    direction = (ldir + tdir) / 2
    # Direction/volatility are intentionally separate: volatility does not imply bullishness.
    ref_vol = float(df["hist_vol_24"].iloc[-1]) if "hist_vol_24" in df else float(np.median(vol))
    sig = _signal.generate(float(direction[1]), float(vol[1]), ref_vol)
    return {
        "timestamp": datetime.now(timezone.utc).isoformat(),
        "symbol": SYMBOL,
        "timeframe": DEFAULT_TIMEFRAME,
        "current_price": float(df["close"].iloc[-1]),
        "atr": float(df["atr"].iloc[-1]),
        "predictions": {
            "1h": {"volatility": float(vol[0]), "up_probability": float(direction[0])},
            "6h": {"volatility": float(vol[1]), "up_probability": float(direction[1])},
            "24h": {"volatility": float(vol[2]), "up_probability": float(direction[2])},
        },
        "signal": sig.__dict__,
    }


@app.get("/health")
def health():
    return {"status": "ok", "models_ready": all((MODEL_DIR / x).exists() for x in ["feature_scaler.pkl", "lstm_model.keras", "transformer_model.keras", "garch_btc.pkl"])}


@app.get("/market")
def market(timeframe: str = DEFAULT_TIMEFRAME, days: int = 7):
    try:
        df = fetch_klines(SYMBOL, timeframe=timeframe, lookback_days=max(days, 2))
        df = df.tail(500).copy()
        df["open_time"] = df["open_time"].astype(str)
        return df.to_dict(orient="records")
    except Exception as exc:
        raise HTTPException(503, f"Market data unavailable: {exc}")


@app.get("/predict/latest")
def latest_predict(timeframe: str = DEFAULT_TIMEFRAME):
    try:
        # The trained sequence models are calibrated on 1h bars. 4h/1d remain available
        # for the live chart until separate timeframe-specific models are trained.
        model_timeframe = "1h"
        df = _feature_frame(timeframe=model_timeframe)
        result = _predict(df)
        result["timeframe"] = model_timeframe
        result["requested_chart_timeframe"] = timeframe
        return result
    except FileNotFoundError as exc:
        raise HTTPException(503, str(exc))
    except Exception as exc:
        raise HTTPException(503, f"Prediction unavailable: {exc}")


@app.get("/paper/status")
def paper_status(price: float | None = None):
    return {"enabled": PAPER_TRADING_ENABLED, **_broker.snapshot(price)}


@app.post("/paper/trade")
def paper_trade(request: TradeRequest):
    if not PAPER_TRADING_ENABLED:
        raise HTTPException(403, "Paper trading is disabled")
    try:
        result = _broker.market_order(request.side, request.quantity, request.price)
        return result
    except Exception as exc:
        raise HTTPException(400, str(exc))
