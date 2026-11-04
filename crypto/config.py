"""Central configuration for Crypto Volatility Forecast & Smart Signal Generator."""
from __future__ import annotations

import os
import sys
from pathlib import Path

# Base directory paths
ROOT_DIR = Path(__file__).resolve().parent
DATA_DIR = ROOT_DIR / "data"
RAW_DIR = DATA_DIR / "raw"
MARKET_RAW_DIR = RAW_DIR / "market"
ONCHAIN_RAW_DIR = RAW_DIR / "onchain"
SENTIMENT_RAW_DIR = RAW_DIR / "sentiment"
PROCESSED_DIR = DATA_DIR / "processed"
FEATURES_DIR = DATA_DIR / "features"
ARTIFACTS_DIR = ROOT_DIR / "artifacts"
MODEL_DIR = ARTIFACTS_DIR  # Alias for model storage
LOG_DIR = ROOT_DIR / "logs"
PAPER_DIR = ROOT_DIR / "paper_trading"

# Create all necessary pipeline directories
for _d in (
    MARKET_RAW_DIR,
    ONCHAIN_RAW_DIR,
    SENTIMENT_RAW_DIR,
    PROCESSED_DIR,
    FEATURES_DIR,
    ARTIFACTS_DIR,
    LOG_DIR,
    PAPER_DIR,
):
    _d.mkdir(parents=True, exist_ok=True)

# Load local environment file if present
_env_file = ROOT_DIR / ".env"
if _env_file.exists():
    try:
        for _line in _env_file.read_text(encoding="utf-8").splitlines():
            _line = _line.strip()
            if _line and not _line.startswith("#") and "=" in _line:
                _k, _v = _line.split("=", 1)
                os.environ.setdefault(_k.strip(), _v.strip())
    except Exception:
        pass

# Assets & Exchanges
ASSETS = [
    x.strip()
    for x in os.getenv("ASSETS", "BTC/USDT,ETH/USDT,SOL/USDT").split(",")
    if x.strip()
]
DEFAULT_SYMBOL = os.getenv("SYMBOL", "BTC/USDT")
SYMBOL = DEFAULT_SYMBOL
EXCHANGE_ID = os.getenv("EXCHANGE_ID", "bybit")
EXCHANGE_FALLBACKS = [
    x.strip()
    for x in os.getenv("EXCHANGE_FALLBACKS", "kraken,coinbase,binance").split(",")
    if x.strip()
]
TIMEFRAMES = ["1m", "5m", "15m", "30m", "1h", "4h", "1d"]
DEFAULT_TIMEFRAME = os.getenv("DEFAULT_TIMEFRAME", "1h")
LOOKBACK_DAYS = int(os.getenv("LOOKBACK_DAYS", "730"))  # 2 years

# Public data endpoints (no API keys required)
BLOCKCHAIN_CHARTS_BASE = os.getenv(
    "BLOCKCHAIN_CHARTS_BASE", "https://api.blockchain.info/charts"
)
FEAR_GREED_API_URL = os.getenv(
    "FEAR_GREED_API_URL", "https://api.alternative.me/fng/?limit=0&format=json"
)
NEWS_RSS_FEEDS = [
    x.strip()
    for x in os.getenv(
        "NEWS_RSS_FEEDS",
        "https://www.coindesk.com/arc/outboundfeeds/rss/,https://cointelegraph.com/rss,https://decrypt.co/feed,https://economictimes.indiatimes.com/markets/rssfeeds/1977021501.cms,https://www.livemint.com/rss/markets",
    ).split(",")
    if x.strip()
]

# Modeling & Feature Engineering parameters
SEQ_LEN = int(os.getenv("SEQ_LEN", "48"))
VOLATILITY_WINDOWS = [6, 12, 24, 72]
FORECAST_HORIZONS = [1, 6, 24]  # 1h, 6h, 24h forward
RSI_PERIOD = 14
ATR_PERIOD = 14
MACD_FAST = 12
MACD_SLOW = 26
MACD_SIGNAL = 9
BOLLINGER_PERIOD = 20
BOLLINGER_STD = 2.0
TRAIN_RATIO = float(os.getenv("TRAIN_RATIO", "0.70"))
VALID_RATIO = float(os.getenv("VALID_RATIO", "0.15"))
RANDOM_SEED = int(os.getenv("RANDOM_SEED", "42"))

# Signal Intelligence Thresholds (Configurable)
STRONG_BUY_DIR_THRESHOLD = float(os.getenv("STRONG_BUY_DIR_THRESHOLD", "0.62"))
STRONG_BUY_CONFIDENCE = float(os.getenv("STRONG_BUY_CONFIDENCE", "0.72"))
STRONG_BUY_VOL_RATIO_MAX = float(os.getenv("STRONG_BUY_VOL_RATIO_MAX", "1.30"))

STRONG_SELL_DIR_THRESHOLD = float(os.getenv("STRONG_SELL_DIR_THRESHOLD", "0.38"))
STRONG_SELL_CONFIDENCE = float(os.getenv("STRONG_SELL_CONFIDENCE", "0.72"))

BUY_DIR_THRESHOLD = float(os.getenv("BUY_DIR_THRESHOLD", "0.55"))
BUY_VOL_RATIO_MAX = float(os.getenv("BUY_VOL_RATIO_MAX", "1.40"))
SELL_DIR_THRESHOLD = float(os.getenv("SELL_DIR_THRESHOLD", "0.45"))
VOL_SHOCK_RATIO_THRESHOLD = float(os.getenv("VOL_SHOCK_RATIO_THRESHOLD", "1.80"))

MODEL_AGREEMENT_MIN_STRONG = float(os.getenv("MODEL_AGREEMENT_MIN_STRONG", "0.75"))
EVENT_RISK_HOURS_WINDOW = float(os.getenv("EVENT_RISK_HOURS_WINDOW", "4.0"))

# Risk & Trading controls
INITIAL_CAPITAL = float(os.getenv("INITIAL_CAPITAL", "10000.0"))
MAX_POSITION_RISK_PCT = float(os.getenv("MAX_POSITION_RISK_PCT", "0.02"))  # 2% max capital risk
STOP_LOSS_ATR_MULTIPLIER = float(os.getenv("STOP_LOSS_ATR_MULTIPLIER", "2.0"))
MAX_DRAWDOWN_LIMIT = float(os.getenv("MAX_DRAWDOWN_LIMIT", "0.20"))  # 20% max drawdown limit
TAKE_PROFIT_RR = float(os.getenv("TAKE_PROFIT_RR", "2.0"))  # 2:1 reward:risk ratio
TRANSACTION_COST = float(os.getenv("TRANSACTION_COST", "0.001"))  # 10 bps (0.1% per trade)
SLIPPAGE = float(os.getenv("SLIPPAGE", "0.0005"))  # 5 bps
SIGNAL_LOOKBACK = int(os.getenv("SIGNAL_LOOKBACK", "168"))

# Serving, Caching & Performance
MODEL_NAME = os.getenv("MODEL_NAME", "ensemble")
API_HOST = os.getenv("API_HOST", "0.0.0.0")
API_PORT = int(os.getenv("API_PORT", "8000"))
AGY_API_KEY = os.getenv("AGY_API_KEY", "")
API_SECRET_KEY = AGY_API_KEY or os.getenv("API_SECRET_KEY", "")
PAPER_TRADING_ENABLED = os.getenv("PAPER_TRADING_ENABLED", "true").lower() == "true"
DRIFT_P_VALUE_THRESHOLD = float(os.getenv("DRIFT_P_VALUE_THRESHOLD", "0.05"))
WEBSOCKET_ENABLED = os.getenv("WEBSOCKET_ENABLED", "true").lower() == "true"

# Service-Level Cache TTLs (seconds)
CACHE_TTL_PRICE = int(os.getenv("CACHE_TTL_PRICE", "2"))
CACHE_TTL_CANDLES = int(os.getenv("CACHE_TTL_CANDLES", "2"))
CACHE_TTL_NEWS = int(os.getenv("CACHE_TTL_NEWS", "180"))
CACHE_TTL_EVENTS = int(os.getenv("CACHE_TTL_EVENTS", "900"))
CACHE_TTL_SENTIMENT = int(os.getenv("CACHE_TTL_SENTIMENT", "300"))

# Configurable AI Assistant Provider Configuration
_is_testing = "pytest" in sys.modules or any("pytest" in str(arg).lower() for arg in sys.argv)
_gemini_active = bool(os.getenv("GEMINI_API_KEY") or (os.getenv("LLM_PROVIDER") == "gemini"))
if _is_testing:
    LLM_PROVIDER = os.getenv("TEST_LLM_PROVIDER", "deterministic")
else:
    LLM_PROVIDER = os.getenv("LLM_PROVIDER", "gemini" if _gemini_active else "deterministic")
LLM_MODEL = os.getenv("LLM_MODEL", "")
LLM_BASE_URL = os.getenv("LLM_BASE_URL", "")
LLM_API_KEY = os.getenv("LLM_API_KEY", "")
LLM_TIMEOUT = float(os.getenv("LLM_TIMEOUT", "15.0"))
LLM_FALLBACK_PROVIDER = os.getenv("LLM_FALLBACK_PROVIDER", "deterministic")

# Provider-specific aliases & backward compatibility
GEMINI_API_KEY = os.getenv("GEMINI_API_KEY", "") or (LLM_API_KEY if LLM_PROVIDER in ("gemini", "google") else "")
GEMINI_MODEL = os.getenv("GEMINI_MODEL", "") or (LLM_MODEL if LLM_PROVIDER in ("gemini", "google") and LLM_MODEL else "gemini-3.5-flash-lite")
GEMINI_BASE_URL = os.getenv("GEMINI_BASE_URL", "") or (LLM_BASE_URL if LLM_BASE_URL else "https://generativelanguage.googleapis.com/v1beta/openai")

OPENAI_API_KEY = os.getenv("OPENAI_API_KEY", "") or (LLM_API_KEY if LLM_PROVIDER in ("openai", "openai_compatible") else "")
OPENAI_BASE_URL = os.getenv("OPENAI_BASE_URL", "") or (LLM_BASE_URL if LLM_BASE_URL else "")
OPENROUTER_API_KEY = os.getenv("OPENROUTER_API_KEY", "") or (LLM_API_KEY if LLM_PROVIDER == "openrouter" else "")
XAI_API_KEY = os.getenv("XAI_API_KEY", os.getenv("GROK_API_KEY", "")) or (LLM_API_KEY if LLM_PROVIDER in ("xai", "grok", "xai_grok") else "")
XAI_BASE_URL = os.getenv("XAI_BASE_URL", "") or (LLM_BASE_URL if LLM_BASE_URL else "https://api.x.ai/v1")
HUGGINGFACE_API_KEY = os.getenv("HUGGINGFACE_API_KEY", "") or (LLM_API_KEY if LLM_PROVIDER in ("huggingface", "hf") else "")

