"""Real-time live streaming candle fetcher and in-memory feature transformer.

Fetches the latest N candles directly from Binance/CCXT in memory, executes the
exact canonical feature transformations, applies the saved scaler from
artifacts/scaler.pkl according to artifacts/feature_schema.json, and feeds the
resulting (1, seq_len, n_features) tensor directly into model inference.
"""
from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import ccxt
import joblib
import numpy as np
import pandas as pd

from config import ARTIFACTS_DIR, EXCHANGE_ID, SEQ_LEN
from data_pipeline.feature_engineering import CANONICAL_FEATURES, engineer_features
from data_pipeline.preprocess import clean_ohlcv, merge_sources_leak_free


class LiveStreamBuffer:
    """In-memory live market fetcher and feature pipeline."""

    def __init__(self, exchange_id: str = EXCHANGE_ID):
        self.exchange_id = exchange_id
        cls = getattr(ccxt, exchange_id, ccxt.binance)
        self.exchange = cls({"enableRateLimit": True, "timeout": 15000})

        # Load persisted schema and scaler
        self.schema_path = ARTIFACTS_DIR / "feature_schema.json"
        self.scaler_path = ARTIFACTS_DIR / "scaler.pkl"
        self._schema = None
        self._scaler = None

    def _ensure_artifacts(self):
        if self._schema is None and self.schema_path.exists():
            self._schema = json.loads(self.schema_path.read_text(encoding="utf-8"))
        if self._scaler is None and self.scaler_path.exists():
            self._scaler = joblib.load(self.scaler_path)

    def fetch_live_candles(self, symbol: str = "BTC/USDT", timeframe: str = "1h", limit: int = 120) -> pd.DataFrame:
        """Fetch latest candles directly from exchange with fallbacks and local cache fallback."""
        from market_data.provider import detect_asset_type, get_market_provider

        atype = detect_asset_type(symbol)

        # 1. Non-crypto symbols (Indian stocks, US stocks, indices, forex) route directly to market provider
        if atype != "crypto":
            try:
                provider = get_market_provider()
                candles = provider.get_ohlcv(symbol, timeframe=timeframe, limit=limit)
                if candles and len(candles) >= 15:
                    rows = []
                    for c in candles:
                        rows.append({
                            "open_time": pd.to_datetime(c.timestamp, utc=True),
                            "open": c.open,
                            "high": c.high,
                            "low": c.low,
                            "close": c.close,
                            "volume": c.volume or 0.0,
                        })
                    df = pd.DataFrame(rows)
                    return df.dropna().sort_values("open_time").reset_index(drop=True)
            except Exception:
                pass

        # 2. Crypto symbols query CCXT exchanges
        if atype == "crypto":
            exchange_candidates = [self.exchange_id, "bybit", "kraken", "coinbase"]
            for exc_id in exchange_candidates:
                try:
                    cls = getattr(ccxt, exc_id, None)
                    if cls is None:
                        continue
                    exc = cls({"enableRateLimit": True, "timeout": 4000})
                    exc.load_markets()
                    resolved_sym = symbol if symbol in exc.symbols else f"{symbol.split('/')[0]}/USD"
                    if resolved_sym in exc.symbols:
                        raw_candles = exc.fetch_ohlcv(resolved_sym, timeframe=timeframe, limit=limit)
                        if raw_candles and len(raw_candles) >= 30:
                            df = pd.DataFrame(raw_candles, columns=["timestamp", "open", "high", "low", "close", "volume"])
                            df["open_time"] = pd.to_datetime(df.pop("timestamp"), unit="ms", utc=True)
                            for col in ["open", "high", "low", "close", "volume"]:
                                df[col] = pd.to_numeric(df[col], errors="coerce")
                            return df.dropna().sort_values("open_time").reset_index(drop=True)
                except Exception:
                    continue

        # Fallback to centralized market provider (supports NSE, US Stocks, Indices, Forex, CCXT, YahooFinance)
        try:
            from market_data.provider import get_market_provider
            provider = get_market_provider()
            candles = provider.get_ohlcv(symbol, timeframe=timeframe, limit=limit)
            if candles and len(candles) >= 15:
                rows = []
                for c in candles:
                    rows.append({
                        "open_time": pd.to_datetime(c.timestamp, utc=True),
                        "open": c.open,
                        "high": c.high,
                        "low": c.low,
                        "close": c.close,
                        "volume": c.volume or 0.0,
                    })
                df = pd.DataFrame(rows)
                return df.dropna().sort_values("open_time").reset_index(drop=True)
        except Exception:
            pass

        # Ultimate fallback: local market cache if internet/exchange connection drops
        from config import MARKET_RAW_DIR
        local_cache = MARKET_RAW_DIR / f"{symbol.replace('/', '_').lower()}_{timeframe}.csv"
        if local_cache.exists():
            df = pd.read_csv(local_cache, parse_dates=["open_time"])
            return df.tail(limit).reset_index(drop=True)

        raise RuntimeError(f"Could not retrieve live candles for {symbol} across exchanges or local cache")

    def _get_asset_scaler(self, symbol: str):
        slug = "eth_usdt" if "ETH" in symbol.upper() else ("nsei" if ("NSEI" in symbol.upper() or "NIFTY" in symbol.upper()) else "btc_usdt")
        asset_scaler_path = ARTIFACTS_DIR / slug / "scaler.pkl"
        if asset_scaler_path.exists():
            return joblib.load(asset_scaler_path)
        if self._scaler is not None:
            return self._scaler
        default_scaler_path = ARTIFACTS_DIR / "scaler.pkl"
        if default_scaler_path.exists():
            return joblib.load(default_scaler_path)
        return None

    def _get_asset_schema(self, symbol: str) -> list[str]:
        slug = "eth_usdt" if "ETH" in symbol.upper() else ("nsei" if ("NSEI" in symbol.upper() or "NIFTY" in symbol.upper()) else "btc_usdt")
        asset_schema_path = ARTIFACTS_DIR / slug / "feature_schema.json"
        if asset_schema_path.exists():
            return json.loads(asset_schema_path.read_text(encoding="utf-8"))
        if self._schema is not None:
            return self._schema
        default_schema_path = ARTIFACTS_DIR / "feature_schema.json"
        if default_schema_path.exists():
            return json.loads(default_schema_path.read_text(encoding="utf-8"))
        return CANONICAL_FEATURES

    def prepare_inference_tensor(
        self,
        symbol: str = "BTC/USDT",
        timeframe: str = "1h",
        onchain_df: pd.DataFrame | None = None,
        sentiment_df: pd.DataFrame | None = None,
    ) -> tuple[np.ndarray, pd.DataFrame]:
        """
        Produce normalized 3D tensor: shape (1, seq_len, n_features)
        guaranteeing exact schema feature ordering and scaling.
        """
        self._ensure_artifacts()
        scaler = self._get_asset_scaler(symbol)
        if scaler is None:
            raise FileNotFoundError(f"Scaler not found in artifacts/ for {symbol}")

        schema = self._get_asset_schema(symbol)
        if hasattr(scaler, "n_features_in_") and len(schema) != scaler.n_features_in_:
            raise ValueError(
                f"Feature schema dimension mismatch: schema has {len(schema)} features, "
                f"but scaler expected {scaler.n_features_in_} features."
            )
        if hasattr(scaler, "feature_names_in_") and list(scaler.feature_names_in_) != list(schema):
            raise ValueError(
                f"Feature schema name ordering mismatch between schema and fitted scaler: "
                f"{list(schema)} vs {list(scaler.feature_names_in_)}"
            )

        # 1. Fetch live candles
        raw_df = self.fetch_live_candles(symbol=symbol, timeframe=timeframe, limit=120)
        cleaned = clean_ohlcv(raw_df)

        # 2. Merge exogenous indicators backward in time
        merged = merge_sources_leak_free(cleaned, onchain_df=onchain_df, sentiment_df=sentiment_df)

        # 3. Engineer features
        feat_df = engineer_features(merged)

        # Ensure all schema features exist
        for col in schema:
            if col not in feat_df.columns:
                feat_df[col] = 0.0

        if len(feat_df) < 8:
            raise ValueError(f"Insufficient candles ({len(feat_df)}) for feature inference")

        target_seq_len = min(SEQ_LEN, len(feat_df))

        # Extract sequence tail
        tail_df = feat_df.iloc[-target_seq_len:].copy()

        # Scale using fitted training scaler in exact schema order
        scaled_matrix = scaler.transform(tail_df[schema])
        tensor = scaled_matrix[np.newaxis, :, :]  # (1, seq_len, n_features)

        return tensor, tail_df
