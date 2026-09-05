"""Central configuration for Crypto Volatility Forecast & Smart Signal Generator."""
from __future__ import annotations

import os
from pathlib import Path

ROOT_DIR = Path(__file__).resolve().parent
DATA_DIR = ROOT_DIR / "data"
RAW_DIR = DATA_DIR / "raw"
PROCESSED_DIR = DATA_DIR / "processed"
MODEL_DIR = ROOT_DIR / "saved_models"
LOG_DIR = ROOT_DIR / "logs"
PAPER_DIR = ROOT_DIR / "paper_trading"

for _d in (RAW_DIR, PROCESSED_DIR, MODEL_DIR, LOG_DIR, PAPER_DIR):
    _d.mkdir(parents=True, exist_ok=True)

# Market data: public endpoints through CCXT. No API key is needed for OHLCV/ticker reads.
SYMBOL = os.getenv("SYMBOL", "BTC/USDT")
EXCHANGE_ID = os.getenv("EXCHANGE_ID", "binance")
EXCHANGE_FALLBACKS = [x.strip() for x in os.getenv("EXCHANGE_FALLBACKS", "kraken,bybit,coinbase").split(",") if x.strip()]
TIMEFRAMES = ["1h", "4h", "1d"]
DEFAULT_TIMEFRAME = os.getenv("DEFAULT_TIMEFRAME", "1h")
LOOKBACK_DAYS = int(os.getenv("LOOKBACK_DAYS", "730"))

# Optional GitHub research datasets. They are used as seeds/fallbacks, never as a hard dependency.
OHLCV_DATASET_REPO = os.getenv("OHLCV_DATASET_REPO", "Speirsy11/crypto-dataset")
NEWS_DATASET_REPO = os.getenv("NEWS_DATASET_REPO", "Speirsy11/crypto-news-dataset")

# Free public on-chain source: Blockchain.com charts API.
BLOCKCHAIN_CHARTS_BASE = os.getenv("BLOCKCHAIN_CHARTS_BASE", "https://api.blockchain.info/charts")

# RSS-based news sources require no account/API key.
NEWS_RSS_FEEDS = [
    x.strip() for x in os.getenv(
        "NEWS_RSS_FEEDS",
        "https://www.coindesk.com/arc/outboundfeeds/rss/,https://cointelegraph.com/rss"
    ).split(",") if x.strip()
]

# Modeling / feature engineering.
SEQ_LEN = int(os.getenv("SEQ_LEN", "48"))
VOLATILITY_WINDOWS = [6, 12, 24, 72]
FORECAST_HORIZONS = [1, 6, 24]
RSI_PERIOD = 14
ATR_PERIOD = 14
TRAIN_RATIO = float(os.getenv("TRAIN_RATIO", "0.70"))
VALID_RATIO = float(os.getenv("VALID_RATIO", "0.15"))
RANDOM_SEED = 42

# Risk / paper trading.
INITIAL_CAPITAL = float(os.getenv("INITIAL_CAPITAL", "10000"))
MAX_POSITION_RISK_PCT = float(os.getenv("MAX_POSITION_RISK_PCT", "0.02"))
STOP_LOSS_ATR_MULTIPLIER = float(os.getenv("STOP_LOSS_ATR_MULTIPLIER", "2.0"))
MAX_DRAWDOWN_LIMIT = float(os.getenv("MAX_DRAWDOWN_LIMIT", "0.20"))
TAKE_PROFIT_RR = float(os.getenv("TAKE_PROFIT_RR", "2.0"))
TRANSACTION_COST = float(os.getenv("TRANSACTION_COST", "0.001"))
SIGNAL_LOOKBACK = int(os.getenv("SIGNAL_LOOKBACK", "168"))

# Serving.
MODEL_NAME = os.getenv("MODEL_NAME", "ensemble")
API_HOST = os.getenv("API_HOST", "0.0.0.0")
API_PORT = int(os.getenv("API_PORT", "8000"))
PAPER_TRADING_ENABLED = os.getenv("PAPER_TRADING_ENABLED", "true").lower() == "true"
