# Crypto Volatility Forecast and Smart Signal Generator - Project Knowledge Base


## FILE: app.py

"""Streamlit UI: live BTC chart, forecast dashboard, signal explanation and paper trading."""
from __future__ import annotations

import os
import time

import pandas as pd
import plotly.graph_objects as go
import requests
import streamlit as st

API_URL = os.getenv("API_URL", "http://localhost:8000")

st.set_page_config(page_title="BTC Volatility Intelligence", page_icon="â‚¿", layout="wide")

st.markdown("""
<style>
.block-container {padding-top: 1.2rem; max-width: 1450px;}
[data-testid="stMetricValue"] {font-size: 1.65rem;}
.signal-card {padding: 14px 18px; border-radius: 14px; border: 1px solid rgba(128,128,128,.25);}
.small {font-size: .85rem; opacity: .75;}
</style>
""", unsafe_allow_html=True)


def api_get(path, params=None):
    r = requests.get(f"{API_URL}{path}", params=params, timeout=25)
    r.raise_for_status()
    return r.json()


st.title("â‚¿ Crypto Volatility Forecast & Smart Signal Generator")
st.caption("BTC/USDT â€¢ 1H ensemble forecast with 1H/4H/1D live charts â€¢ GARCH + LSTM + Transformer â€¢ paper trading only")

with st.sidebar:
    st.header("Controls")
    timeframe = st.selectbox("Chart timeframe", ["1h", "4h", "1d"], index=0)
    chart_days = st.slider("Chart history (days)", 2, 90, 7)
    auto_refresh = st.checkbox("Auto-refresh live view", value=True)
    refresh_seconds = st.slider("Refresh seconds", 10, 120, 30)
    if st.button("Refresh now"):
        st.cache_data.clear()
        st.rerun()
    st.divider()
    health = api_get("/health")
    st.write(f"API: **{health.get('status', 'unknown')}**")
    st.write(f"Models: **{'ready' if health.get('models_ready') else 'not trained'}**")

try:
    market = api_get("/market", {"timeframe": timeframe, "days": chart_days})
    mdf = pd.DataFrame(market)
    mdf["open_time"] = pd.to_datetime(mdf["open_time"], utc=True)
    pred = api_get("/predict/latest", {"timeframe": timeframe})
except Exception as exc:
    st.error(f"Backend unavailable: {exc}")
    st.stop()

last = mdf.iloc[-1]
p1 = pred["predictions"]["1h"]
p6 = pred["predictions"]["6h"]
p24 = pred["predictions"]["24h"]
signal = pred["signal"]

c1, c2, c3, c4, c5 = st.columns(5)
c1.metric("BTC Price", f"${pred['current_price']:,.2f}")
c2.metric("Signal", signal["action"])
c3.metric("6H Up Probability", f"{p6['up_probability']*100:.1f}%")
c4.metric("6H Forecast Vol", f"{p6['volatility']*100:.2f}%")
c5.metric("Risk", signal["risk_level"])

fig = go.Figure(data=[go.Candlestick(x=mdf["open_time"], open=mdf["open"], high=mdf["high"], low=mdf["low"], close=mdf["close"])])
fig.update_layout(height=500, margin=dict(l=10, r=10, t=25, b=10), xaxis_rangeslider_visible=False, title=f"BTC/USDT â€” {timeframe}")
st.plotly_chart(fig, use_container_width=True)

left, right = st.columns([1.05, 1])
with left:
    st.subheader("Forecast horizon")
    forecast_df = pd.DataFrame([
        ["1H", p1["volatility"], p1["up_probability"]],
        ["6H", p6["volatility"], p6["up_probability"]],
        ["24H", p24["volatility"], p24["up_probability"]],
    ], columns=["Horizon", "Forecast volatility", "Up probability"])
    forecast_df["Forecast volatility"] = (forecast_df["Forecast volatility"] * 100).round(3).astype(str) + "%"
    forecast_df["Up probability"] = (forecast_df["Up probability"] * 100).round(1).astype(str) + "%"
    st.dataframe(forecast_df, hide_index=True, use_container_width=True)
    st.info(signal["reason"] + f" Confidence: {signal['confidence']*100:.0f}%. Forecast model cadence: 1H.")

with right:
    st.subheader("Paper trading")
    status = api_get("/paper/status", {"price": pred["current_price"]})
    s1, s2, s3 = st.columns(3)
    s1.metric("Equity", f"${status['equity']:,.2f}")
    s2.metric("Cash", f"${status['cash']:,.2f}")
    s3.metric("BTC Position", f"{status['position']:.6f}")
    st.caption("Orders below are simulated and are never sent to an exchange.")
    with st.form("trade_form"):
        side = st.selectbox("Action", ["BUY", "SELL", "REDUCE"])
        qty = st.number_input("BTC quantity", min_value=0.000001, value=0.001, format="%.6f")
        submitted = st.form_submit_button("Place paper order")
        if submitted:
            try:
                out = requests.post(f"{API_URL}/paper/trade", json={"side": side, "quantity": qty, "price": pred["current_price"]}, timeout=10)
                out.raise_for_status()
                st.success("Paper order recorded.")
            except Exception as exc:
                st.error(str(exc))

st.subheader("Why this signal?")
st.write(f"**{signal['action']}** is based on the ensemble's 6-hour direction probability, forecast volatility relative to the latest 24-hour historical volatility, and explicit risk thresholds. Volatility is treated as risk â€” not direction.")

if os.path.exists(os.path.join("data", "processed", "evaluation_report.json")):
    with st.expander("Model evaluation"):
        import json
        report = json.load(open(os.path.join("data", "processed", "evaluation_report.json"), encoding="utf-8"))
        st.json(report)

if auto_refresh:
    time.sleep(refresh_seconds)
    st.rerun()


## FILE: config.py

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


## FILE: main.py

"""End-to-end training pipeline: ingest -> clean -> enrich -> train -> evaluate."""
from __future__ import annotations

import json

import pandas as pd

from config import INITIAL_CAPITAL, PROCESSED_DIR, RAW_DIR
from data_pipeline.ingest_market_data import fetch_klines
from data_pipeline.ingest_onchain_data import build_onchain_dataset
from data_pipeline.ingest_sentiment_data import fetch_news, hourly_sentiment_index
from data_pipeline.preprocess import clean_ohlcv, merge_sources, add_targets
from data_pipeline.feature_engineering import engineer_features
from evaluation.backtest import Backtester
from evaluation.metrics import directional_accuracy, mae, rmse
from models.train import train_all


def run_pipeline(train_epochs: int = 20):
    print("[1/6] Fetching BTC 1h/4h/1d market data...")
    for tf in ("1h", "4h", "1d"):
        raw = fetch_klines(timeframe=tf)
        raw.to_csv(RAW_DIR / f"btc_{tf}_ohlcv.csv", index=False)

    print("[2/6] Fetching free on-chain and RSS sentiment data...")
    try:
        onchain = build_onchain_dataset("2years")
        onchain.to_csv(RAW_DIR / "onchain_btc.csv", index=False)
    except Exception as exc:
        print("On-chain source unavailable; continuing with neutral fallback:", exc)
        onchain = None
    try:
        sentiment = hourly_sentiment_index(fetch_news())
        sentiment.to_csv(RAW_DIR / "sentiment_btc.csv", index=False)
    except Exception as exc:
        print("News source unavailable; continuing with neutral fallback:", exc)
        sentiment = None

    print("[3/6] Cleaning, aligning and creating forward targets...")
    raw = pd.read_csv(RAW_DIR / "btc_1h_ohlcv.csv", parse_dates=["open_time"])
    base = clean_ohlcv(raw)
    merged = merge_sources(base, onchain, sentiment)
    targeted = add_targets(merged)
    features = engineer_features(targeted)
    for col in ["active_addresses", "tx_count", "hash_rate", "difficulty", "miners_revenue", "sentiment_score", "news_count"]:
        if col not in features.columns:
            features[col] = 0.0
    features.to_csv(PROCESSED_DIR / "btc_features.csv", index=False)

    print("[4/6] Training GARCH + LSTM + Transformer...")
    artifacts = train_all(epochs=train_epochs)

    print("[5/6] Evaluating forecast accuracy...")
    lstm_out = artifacts["lstm"].predict(artifacts["X_test"], verbose=0)
    trans_out = artifacts["transformer"].predict(artifacts["X_test"], verbose=0)
    neural_vol = (lstm_out["volatility"] + trans_out["volatility"]) / 2
    neural_dir = (lstm_out["direction"] + trans_out["direction"]) / 2

    forecast_metrics = {}
    for i, horizon in enumerate((1, 6, 24)):
        yv = artifacts["yv_test"][:, i]
        yd = artifacts["yd_test"][:, i]
        forecast_metrics[f"{horizon}h"] = {
            "volatility_rmse": rmse(yv, neural_vol[:, i]),
            "volatility_mae": mae(yv, neural_vol[:, i]),
            "directional_accuracy": directional_accuracy(yd, neural_dir[:, i] - 0.5),
        }

    print("[6/6] Running paper backtest on the held-out period...")
    test_df = artifacts["test"].iloc[artifacts["test"].shape[0] - len(neural_dir):].reset_index(drop=True)
    bt = Backtester(test_df, initial_capital=INITIAL_CAPITAL)
    results, equity, position = bt.run(neural_dir[:, 1], neural_vol[:, 1])
    report = {"forecast_metrics": forecast_metrics, "backtest": results}
    (PROCESSED_DIR / "evaluation_report.json").write_text(json.dumps(report, indent=2), encoding="utf-8")
    pd.DataFrame({"equity": equity, "position": position}).to_csv(PROCESSED_DIR / "backtest_curve.csv", index=False)
    print(json.dumps(report, indent=2))
    return report


if __name__ == "__main__":
    run_pipeline()


## FILE: README.md

# Crypto Volatility Forecast & Smart Signal Generator

An end-to-end BTC/USDT research and paper-trading system that forecasts **future volatility and price direction** for 1H, 6H and 24H horizons and converts the ensemble output into an explainable BUY / HOLD / SELL / REDUCE signal.

## What changed from the original skeleton

- Public market ingestion uses **CCXT** with exchange fallbacks; no market-data API key is required.
- Free on-chain enrichment uses public Blockchain.com chart data.
- News sentiment uses RSS feeds without CryptoPanic credentials. The pipeline is designed so a public GitHub dataset can seed historical sentiment later.
- Targets are genuinely forward-looking: future realized volatility and future return direction.
- LSTM and Transformer are multi-task models: volatility + direction for 1H/6H/24H.
- GARCH remains an interpretable statistical baseline and contributes to the live ensemble.
- Chronological train/validation/test splits avoid using the test period as validation.
- Signal generation no longer incorrectly assumes that low volatility implies BUY. Direction and risk are separate inputs.
- Backtesting uses held-out predictions and transaction costs.
- FastAPI exposes live market, prediction and paper-trading endpoints.
- Streamlit provides a user-friendly live BTC chart, forecast table, signal explanation and paper-trading controls.
- Docker Compose runs training, inference and the dashboard.

## Data sources

1. **CCXT / public exchange APIs** for live/historical OHLCV. Public market-data methods do not require API keys.
2. **Speirsy11/crypto-dataset** as a reproducible public-domain/CC0 historical seed option for research data.
3. **Speirsy11/crypto-news-dataset** as a historical crypto-news/sentiment research seed; it is enriched with FinBERT and published as Parquet.
4. **Blockchain.com charts API** for key free BTC on-chain series.

The application does not require Binance credentials for the research/paper-trading path. Exchange credentials should only be added for private account actions, and this project intentionally does not send live orders.

## Local run

```powershell
python -m venv .venv
.\.venv\Scripts\activate
python -m pip install --upgrade pip
python -m pip install -r requirements.txt
python main.py
python -m uvicorn serving.inference_api:app --host 0.0.0.0 --port 8000
streamlit run app.py
```

Open `http://localhost:8501`.

## Important note

Training TensorFlow models on a laptop can take time. Run `python main.py` once to generate `saved_models/` artifacts. The dashboard becomes fully predictive after the artifacts exist.

## Architecture

```text
Public exchange OHLCV â”€â”€â”
Free on-chain metrics â”€â”€â”€â”¼â”€> cleaning/alignment â”€> features/targets â”€> GARCH
RSS news sentiment â”€â”€â”€â”€â”€â”€â”˜                                     â”œâ”€â”€â”€â”€> LSTM
                                                               â”œâ”€â”€â”€â”€> Transformer
                                                               â””â”€â”€â”€â”€> Ensemble
                                                                      â”‚
                                                         direction + volatility
                                                                      â”‚
                                                         smart signal + risk
                                                                      â”‚
                                               FastAPI <â”€â”€> Streamlit dashboard
                                                                      â”‚
                                                               paper broker
```

## API endpoints

- `GET /health` â€” service/model readiness
- `GET /market?timeframe=1h&days=7` â€” live chart data
- `GET /predict/latest?timeframe=1h` â€” ensemble forecast and signal
- `GET /paper/status` â€” paper account state
- `POST /paper/trade` â€” record a simulated BUY/SELL/REDUCE market order
- `/docs` â€” interactive FastAPI documentation

## Safety boundary

This release deliberately has **no live-order path**. The UI's trading controls write to a local paper account only. Any future private exchange integration should be isolated behind a separate credentialed execution service and testnet-first validation.


## FILE: requirements.txt

pandas>=2.2
numpy>=1.26
scikit-learn>=1.5
ccxt>=4.4
requests>=2.32
feedparser>=6.0
vaderSentiment>=3.3
arch>=7.0
joblib>=1.4
tensorflow>=2.18
fastapi>=0.115
uvicorn[standard]>=0.30
pydantic>=2.9
streamlit>=1.40
plotly>=5.24
pytest>=8.0


## FILE: tests_smoke.py

import numpy as np
import pandas as pd

from data_pipeline.preprocess import add_targets
from data_pipeline.feature_engineering import engineer_features
from evaluation.metrics import rmse
from trading.signal_generator import SignalGenerator
from trading.risk_manager import RiskManager


def test_features_and_targets():
    n = 300
    t = pd.date_range("2025-01-01", periods=n, freq="h", tz="UTC")
    price = np.linspace(90000, 95000, n) + np.sin(np.arange(n)) * 100
    df = pd.DataFrame({"open_time": t, "open": price, "high": price + 100, "low": price - 100, "close": price, "volume": np.full(n, 1000.0)})
    out = engineer_features(add_targets(df))
    assert len(out) > 0
    assert {"realized_vol_1", "realized_vol_6", "realized_vol_24", "direction_24"}.issubset(out.columns)


def test_signal_is_not_volatility_only():
    sg = SignalGenerator()
    assert sg.generate(0.7, 0.01, 0.02).action == "BUY"
    assert sg.generate(0.3, 0.01, 0.02).action == "SELL"


def test_risk_manager():
    rm = RiskManager(10000)
    plan = rm.plan(100000, 1000, "long")
    assert plan.quantity > 0
    assert plan.stop_price < plan.entry_price


def test_rmse():
    assert rmse(np.array([1, 2]), np.array([1, 3])) > 0


## FILE: data_pipeline\feature_engineering.py

"""Technical, volatility, momentum, volume, and exogenous features."""
from __future__ import annotations

import numpy as np
import pandas as pd

from config import ATR_PERIOD, RSI_PERIOD


def compute_atr(df: pd.DataFrame, period: int = ATR_PERIOD) -> pd.DataFrame:
    out = df.copy()
    prev_close = out["close"].shift(1)
    tr = pd.concat([(out["high"] - out["low"]), (out["high"] - prev_close).abs(), (out["low"] - prev_close).abs()], axis=1).max(axis=1)
    out["atr"] = tr.rolling(period).mean()
    out["atr_pct"] = out["atr"] / out["close"]
    return out


def compute_rsi(df: pd.DataFrame, period: int = RSI_PERIOD) -> pd.DataFrame:
    out = df.copy()
    delta = out["close"].diff()
    gain = delta.clip(lower=0).ewm(alpha=1 / period, adjust=False).mean()
    loss = (-delta.clip(upper=0)).ewm(alpha=1 / period, adjust=False).mean()
    rs = gain / (loss + 1e-9)
    out["rsi"] = 100 - (100 / (1 + rs))
    return out


def compute_bollinger(df: pd.DataFrame, period: int = 20, num_std: float = 2.0) -> pd.DataFrame:
    out = df.copy()
    ma = out["close"].rolling(period).mean()
    std = out["close"].rolling(period).std()
    out["bb_upper"] = ma + num_std * std
    out["bb_lower"] = ma - num_std * std
    out["bb_width"] = (out["bb_upper"] - out["bb_lower"]) / (ma + 1e-9)
    out["bb_position"] = (out["close"] - out["bb_lower"]) / (out["bb_upper"] - out["bb_lower"] + 1e-9)
    return out


def compute_moving_averages(df: pd.DataFrame) -> pd.DataFrame:
    out = df.copy()
    for span in (12, 26, 50, 200):
        out[f"ema_{span}"] = out["close"].ewm(span=span, adjust=False).mean()
        out[f"ema_{span}_dist"] = out["close"] / out[f"ema_{span}"] - 1
    return out


def engineer_features(df: pd.DataFrame) -> pd.DataFrame:
    out = df.copy()
    if "log_return" not in out.columns:
        out["log_return"] = np.log(out["close"] / out["close"].shift(1))
    for w in (6, 12, 24, 72):
        col = f"hist_vol_{w}"
        if col not in out.columns:
            out[col] = out["log_return"].rolling(w).std() * np.sqrt(w)
    out = compute_atr(out)
    out = compute_rsi(out)
    out = compute_bollinger(out)
    out = compute_moving_averages(out)
    vol_mean = out["volume"].rolling(72).mean()
    vol_std = out["volume"].rolling(72).std()
    out["volume_z"] = (out["volume"] - vol_mean) / (vol_std + 1e-9)
    out["range_pct"] = (out["high"] - out["low"]) / out["close"]
    out["body_pct"] = (out["close"] - out["open"]) / out["open"]
    out = out.replace([np.inf, -np.inf], np.nan)
    return out.dropna().reset_index(drop=True)


## FILE: data_pipeline\ingest_market_data.py

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


## FILE: data_pipeline\ingest_onchain_data.py

"""Free Bitcoin on-chain ingestion using public Blockchain.com charts."""
from __future__ import annotations

import requests
import pandas as pd

from config import BLOCKCHAIN_CHARTS_BASE, RAW_DIR

CHARTS = {
    "active_addresses": "n-unique-addresses",
    "tx_count": "n-transactions",
    "hash_rate": "hash-rate",
    "difficulty": "difficulty",
    "miners_revenue": "miners-revenue",
}


def fetch_chart(chart_name: str, timespan: str = "2years") -> pd.DataFrame:
    url = f"{BLOCKCHAIN_CHARTS_BASE.rstrip('/')}/{chart_name}"
    r = requests.get(url, params={"timespan": timespan, "format": "json"}, timeout=20)
    r.raise_for_status()
    payload = r.json()
    values = payload.get("values", [])
    rows = pd.DataFrame(values)
    if rows.empty:
        return pd.DataFrame(columns=["timestamp", chart_name])
    rows["timestamp"] = pd.to_datetime(rows["x"], unit="s", utc=True)
    return rows[["timestamp", "y"]].rename(columns={"y": chart_name})


def build_onchain_dataset(timespan: str = "2years") -> pd.DataFrame:
    frames = [fetch_chart(chart, timespan) for chart in CHARTS.values()]
    merged = frames[0]
    for frame in frames[1:]:
        merged = merged.merge(frame, on="timestamp", how="outer")
    return merged.sort_values("timestamp").ffill()


def save_onchain(timespan: str = "2years"):
    df = build_onchain_dataset(timespan)
    out = RAW_DIR / "onchain_btc.csv"
    df.to_csv(out, index=False)
    return out


if __name__ == "__main__":
    out = save_onchain()
    print(f"Saved {out}")


## FILE: data_pipeline\ingest_sentiment_data.py

"""Keyless crypto-news sentiment ingestion from RSS feeds."""
from __future__ import annotations

import re
import requests
import pandas as pd
import feedparser
from vaderSentiment.vaderSentiment import SentimentIntensityAnalyzer

from config import NEWS_RSS_FEEDS, RAW_DIR

SIA = SentimentIntensityAnalyzer()


def _is_btc_related(text: str) -> bool:
    t = text.lower()
    return bool(re.search(r"\b(bitcoin|btc|crypto|cryptocurrency|digital asset)\b", t))


def fetch_news(feeds: list[str] | None = None) -> pd.DataFrame:
    rows: list[dict] = []
    for feed_url in feeds or NEWS_RSS_FEEDS:
        try:
            response = requests.get(feed_url, timeout=15, headers={"User-Agent": "crypto-volatility-research/1.0"})
            response.raise_for_status()
            feed = feedparser.parse(response.content)
            for item in feed.entries:
                title = str(item.get("title", "")).strip()
                published = item.get("published", item.get("updated"))
                if not title or not published or not _is_btc_related(title):
                    continue
                rows.append({"published_at": pd.to_datetime(published, utc=True), "title": title, "source": feed_url})
        except Exception:
            continue
    if not rows:
        return pd.DataFrame(columns=["published_at", "title", "source", "sentiment"])
    df = pd.DataFrame(rows).drop_duplicates(subset=["published_at", "title"])
    df["sentiment"] = df["title"].map(lambda t: SIA.polarity_scores(t)["compound"])
    return df.sort_values("published_at")


def hourly_sentiment_index(df: pd.DataFrame) -> pd.DataFrame:
    if df.empty:
        return pd.DataFrame(columns=["open_time", "sentiment_score", "news_count"])
    hourly = (
        df.set_index("published_at")
        .resample("1h")
        .agg(sentiment_score=("sentiment", "mean"), news_count=("title", "count"))
        .reset_index()
        .rename(columns={"published_at": "open_time"})
    )
    return hourly


def save_sentiment():
    news = fetch_news()
    hourly = hourly_sentiment_index(news)
    out = RAW_DIR / "sentiment_btc.csv"
    hourly.to_csv(out, index=False)
    return out


if __name__ == "__main__":
    out = save_sentiment()
    print(f"Saved {out}")


## FILE: data_pipeline\preprocess.py

"""Data quality, source alignment, and leak-safe forecast targets."""
from __future__ import annotations

import numpy as np
import pandas as pd


def clean_ohlcv(df: pd.DataFrame, timeframe: str = "1h") -> pd.DataFrame:
    out = df.copy()
    out["open_time"] = pd.to_datetime(out["open_time"], utc=True)
    out = out.drop_duplicates("open_time").sort_values("open_time").set_index("open_time")
    # We do not interpolate market candles: missing candles are data-quality issues, not synthetic prices.
    out = out[~out.index.duplicated(keep="last")]
    out = out.dropna(subset=["open", "high", "low", "close", "volume"])
    return out.reset_index()


def merge_sources(
    ohlcv: pd.DataFrame,
    onchain: pd.DataFrame | None = None,
    sentiment: pd.DataFrame | None = None,
) -> pd.DataFrame:
    """
    Merge market, on-chain, and sentiment data using UTC nanosecond-resolution
    timestamps so pandas merge_asof receives identical datetime dtypes.
    """
    out = ohlcv.copy()

    # Normalize market timestamps to one exact dtype.
    out["open_time"] = (
        pd.to_datetime(out["open_time"], utc=True)
        .dt.as_unit("ns")
    )

    out = out.sort_values("open_time").reset_index(drop=True)

    # -----------------------------
    # Merge on-chain data
    # -----------------------------
    if onchain is not None and not onchain.empty:
        oc = onchain.copy()

        oc["timestamp"] = (
            pd.to_datetime(oc["timestamp"], utc=True)
            .dt.as_unit("ns")
        )

        oc = (
            oc.sort_values("timestamp")
            .drop_duplicates("timestamp", keep="last")
            .reset_index(drop=True)
        )

        out = pd.merge_asof(
            out,
            oc,
            left_on="open_time",
            right_on="timestamp",
            direction="backward",
        )

        out = out.drop(columns=["timestamp"], errors="ignore")

    # -----------------------------
    # Merge sentiment data
    # -----------------------------
    if sentiment is not None and not sentiment.empty:
        se = sentiment.copy()

        se["open_time"] = (
            pd.to_datetime(se["open_time"], utc=True)
            .dt.as_unit("ns")
        )

        se = (
            se.sort_values("open_time")
            .drop_duplicates("open_time", keep="last")
            .reset_index(drop=True)
        )

        out = pd.merge_asof(
            out,
            se,
            on="open_time",
            direction="backward",
        )

    # -----------------------------
    # Handle missing exogenous data
    # -----------------------------
    neutral_cols = [
        "active_addresses",
        "tx_count",
        "hash_rate",
        "difficulty",
        "miners_revenue",
        "sentiment_score",
        "news_count",
    ]

    for col in neutral_cols:
        if col in out.columns:
            out[col] = (
                pd.to_numeric(out[col], errors="coerce")
                .ffill()
                .bfill()
                .fillna(0.0)
            )

    return out


def add_targets(df: pd.DataFrame) -> pd.DataFrame:
    out = df.copy()
    out["log_return"] = np.log(out["close"] / out["close"].shift(1))
    for h in (1, 6, 24):
        # Forward-looking target: future realized volatility based on returns after t.
        future_returns = out["log_return"].shift(-1)
        if h == 1:
            out[f"realized_vol_{h}"] = future_returns.abs()
        else:
            out[f"realized_vol_{h}"] = future_returns.rolling(h).std().shift(-(h - 1)) * np.sqrt(h)
        out[f"future_return_{h}"] = out["close"].shift(-h) / out["close"] - 1.0
        out[f"direction_{h}"] = (out[f"future_return_{h}"] > 0).astype(int)
    # Historical volatility features known at time t.
    for w in (6, 12, 24, 72):
        out[f"hist_vol_{w}"] = out["log_return"].rolling(w).std() * np.sqrt(w)
    return out.dropna().reset_index(drop=True)


## FILE: data_pipeline\__init__.py


## FILE: deployment\DEPLOYMENT.md

# Deployment

## Docker Compose

From the project root:

```bash
docker compose -f deployment/docker-compose.yml up --build
```

The dashboard is on `http://localhost:8501` and the API is on `http://localhost:8000/docs`.

The training container performs the full data/training pipeline. For a first build, it can take substantial CPU/RAM because TensorFlow trains both an LSTM and Transformer and GARCH is also fitted.

## Kubernetes

Build and push the image, replace `your-registry/volatility-inference:latest` in the manifests, then apply:

```bash
kubectl apply -f deployment/k8s/inference-deployment.yaml
kubectl apply -f deployment/k8s/dashboard-deployment.yaml
kubectl apply -f deployment/k8s/ingestion-cronjob.yaml
```

Training should be run as a controlled batch job or CI/CD step so that the resulting model artifacts are mounted/published before the inference deployment becomes ready.


## FILE: deployment\Dockerfile.inference.txt

FROM python:3.11-slim
WORKDIR /app
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt
COPY . .
EXPOSE 8000
CMD ["python", "-m", "uvicorn", "serving.inference_api:app", "--host", "0.0.0.0", "--port", "8000"]


## FILE: deployment\Dockerfile.ingestion.txt

FROM python:3.11-slim
WORKDIR /app
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt
COPY . .
CMD ["python", "-c", "from data_pipeline.ingest_market_data import save_market_data; save_market_data('1h')"]


## FILE: deployment\Dockerfile.training.txt

FROM python:3.11-slim
WORKDIR /app
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt
COPY . .
CMD ["python", "main.py"]


## FILE: evaluation\backtest.py

"""Leak-safe vectorized paper backtest driven by direction + volatility forecasts."""
from __future__ import annotations

import numpy as np
import pandas as pd

from evaluation.metrics import max_drawdown, sharpe_ratio, sortino_ratio


class Backtester:
    def __init__(self, df: pd.DataFrame, initial_capital=10000.0, transaction_cost=0.001):
        self.df = df.reset_index(drop=True).copy()
        self.initial_capital = float(initial_capital)
        self.cost = float(transaction_cost)

    def run(self, predicted_direction, predicted_volatility):
        pred_dir = np.asarray(predicted_direction, dtype=float)
        pred_vol = np.asarray(predicted_volatility, dtype=float)
        close = self.df["close"].to_numpy(dtype=float)
        n = min(len(close) - 1, len(pred_dir), len(pred_vol))
        close = close[:n + 1]
        pred_dir, pred_vol = pred_dir[:n], pred_vol[:n]

        # Signal at t is applied to return t -> t+1.
        vol_rank = pd.Series(pred_vol).rolling(168, min_periods=30).rank(pct=True).fillna(0.5).to_numpy()
        position = np.where((pred_dir >= 0.55) & (vol_rank < 0.75), 1.0,
                    np.where((pred_dir <= 0.45) | (vol_rank >= 0.90), 0.0, 0.5))
        returns = close[1:] / close[:-1] - 1
        changes = np.abs(np.diff(np.r_[0.0, position]))
        strategy_returns = position * returns - changes * self.cost
        equity = self.initial_capital * np.cumprod(1 + strategy_returns)
        results = {
            "final_equity": float(equity[-1]),
            "total_return_pct": float((equity[-1] / self.initial_capital - 1) * 100),
            "buy_and_hold_pct": float((close[-1] / close[0] - 1) * 100),
            "sharpe": sharpe_ratio(strategy_returns),
            "sortino": sortino_ratio(strategy_returns),
            "max_drawdown_pct": float(max_drawdown(equity) * 100),
            "num_trades": int(np.count_nonzero(changes > 0)),
        }
        return results, equity, position


## FILE: evaluation\metrics.py

"""Forecast and trading metrics."""
from __future__ import annotations

import numpy as np


def rmse(y_true, y_pred):
    return float(np.sqrt(np.mean((np.asarray(y_true) - np.asarray(y_pred)) ** 2)))


def mae(y_true, y_pred):
    return float(np.mean(np.abs(np.asarray(y_true) - np.asarray(y_pred))))


def directional_accuracy(y_true, y_pred):
    yt, yp = np.asarray(y_true), np.asarray(y_pred)
    return float(np.mean((yt > 0) == (yp > 0)))


def sharpe_ratio(returns, periods_per_year=8760):
    r = np.asarray(returns, dtype=float)
    if len(r) == 0 or np.std(r) == 0:
        return 0.0
    return float(np.sqrt(periods_per_year) * np.mean(r) / np.std(r, ddof=1))


def sortino_ratio(returns, periods_per_year=8760):
    r = np.asarray(returns, dtype=float)
    downside = r[r < 0]
    if len(r) == 0 or len(downside) == 0 or np.std(downside, ddof=1) == 0:
        return 0.0
    return float(np.sqrt(periods_per_year) * np.mean(r) / np.std(downside, ddof=1))


def max_drawdown(equity_curve):
    curve = np.asarray(equity_curve, dtype=float)
    peak = np.maximum.accumulate(curve)
    return float(np.min((curve - peak) / peak))


## FILE: evaluation\__init__.py


## FILE: models\garch_model.py

"""GARCH(1,1) volatility baseline."""
from __future__ import annotations

import joblib
import numpy as np
import pandas as pd
from arch import arch_model


def fit_garch(returns: pd.Series, p: int = 1, q: int = 1):
    clean = pd.Series(returns).dropna().astype(float) * 100
    return arch_model(clean, vol="GARCH", p=p, q=q, dist="t", mean="Zero").fit(disp="off")


def forecast_volatility(result, horizons=(1, 6, 24)) -> dict[int, float]:
    max_h = max(horizons)
    fc = result.forecast(horizon=max_h, reindex=False)
    variance = np.asarray(fc.variance.iloc[-1], dtype=float)
    return {h: float(np.sqrt(np.sum(variance[:h])) / 100) for h in horizons}


def save_result(result, path):
    joblib.dump(result, path)


## FILE: models\lstm_model.py

"""Multi-task LSTM: predicts volatility and direction for 1h/6h/24h horizons."""
from __future__ import annotations

import numpy as np
import tensorflow as tf
from tensorflow.keras import Model, layers


OUTPUT_HORIZONS = (1, 6, 24)


def build_lstm(seq_len: int, n_features: int, units: int = 64, dropout: float = 0.2) -> Model:
    inputs = layers.Input(shape=(seq_len, n_features), name="features")
    x = layers.LSTM(units, return_sequences=True)(inputs)
    x = layers.Dropout(dropout)(x)
    x = layers.LSTM(units // 2)(x)
    x = layers.Dropout(dropout)(x)
    x = layers.Dense(32, activation="relu")(x)
    vol = layers.Dense(3, activation="softplus", name="volatility")(x)
    direction = layers.Dense(3, activation="sigmoid", name="direction")(x)
    model = Model(inputs, {"volatility": vol, "direction": direction})
    model.compile(
        optimizer=tf.keras.optimizers.Adam(1e-3),
        loss={"volatility": "mse", "direction": "binary_crossentropy"},
        loss_weights={"volatility": 1.0, "direction": 0.5},
        metrics={"volatility": ["mae"], "direction": ["accuracy"]},
    )
    return model


def make_sequences(feature_array: np.ndarray, vol_targets: np.ndarray, dir_targets: np.ndarray, seq_len: int):
    X, y_vol, y_dir = [], [], []
    for i in range(len(feature_array) - seq_len + 1):
        X.append(feature_array[i:i + seq_len])
        y_vol.append(vol_targets[i + seq_len - 1])
        y_dir.append(dir_targets[i + seq_len - 1])
    return np.asarray(X), np.asarray(y_vol), np.asarray(y_dir)


## FILE: models\train.py

"""Leak-safe chronological training and artifact generation."""
from __future__ import annotations

import json
import os
import random

import joblib
import numpy as np
import pandas as pd
import tensorflow as tf
from sklearn.preprocessing import StandardScaler

from config import MODEL_DIR, PROCESSED_DIR, RANDOM_SEED, SEQ_LEN, TRAIN_RATIO, VALID_RATIO
from models.garch_model import fit_garch
from models.lstm_model import build_lstm, make_sequences
from models.transformer_model import build_transformer

FEATURE_COLS = [
    "log_return", "atr_pct", "rsi", "bb_width", "bb_position",
    "ema_12_dist", "ema_26_dist", "ema_50_dist", "ema_200_dist",
    "volume_z", "range_pct", "body_pct",
    "hist_vol_6", "hist_vol_12", "hist_vol_24", "hist_vol_72",
    "sentiment_score", "news_count", "active_addresses", "tx_count", "hash_rate", "difficulty",
]
HORIZONS = [1, 6, 24]


def _seed_everything():
    random.seed(RANDOM_SEED)
    np.random.seed(RANDOM_SEED)
    tf.random.set_seed(RANDOM_SEED)


def load_data():
    path = PROCESSED_DIR / "btc_features.csv"
    if not path.exists():
        raise FileNotFoundError(f"Missing {path}. Run the data pipeline first.")
    return pd.read_csv(path, parse_dates=["open_time"])


def _targets(df):
    vol = df[[f"realized_vol_{h}" for h in HORIZONS]].values.astype("float32")
    direction = df[[f"direction_{h}" for h in HORIZONS]].values.astype("float32")
    return vol, direction


def train_all(epochs=25, batch_size=128):
    _seed_everything()
    df = load_data()
    n = len(df)
    train_end = int(n * TRAIN_RATIO)
    valid_end = int(n * (TRAIN_RATIO + VALID_RATIO))
    if train_end <= SEQ_LEN or valid_end >= n:
        raise ValueError("Dataset is too small for configured split/sequence length.")

    train_df = df.iloc[:train_end].copy()
    valid_df = df.iloc[train_end:valid_end].copy()
    test_df = df.iloc[valid_end:].copy()

    scaler = StandardScaler().fit(train_df[FEATURE_COLS])
    joblib.dump(scaler, MODEL_DIR / "feature_scaler.pkl")

    def split_sequences(part):
        x = scaler.transform(part[FEATURE_COLS])
        v, d = _targets(part)
        return make_sequences(x, v, d, SEQ_LEN)

    X_train, yv_train, yd_train = split_sequences(train_df)
    # Validation/test include the tail of the preceding split so their first sequence has historical context.
    valid_context = df.iloc[train_end - SEQ_LEN:valid_end].copy()
    test_context = df.iloc[valid_end - SEQ_LEN:].copy()
    X_valid, yv_valid, yd_valid = split_sequences(valid_context)
    X_valid, yv_valid, yd_valid = X_valid[1:], yv_valid[1:], yd_valid[1:]
    X_test, yv_test, yd_test = split_sequences(test_context)
    X_test, yv_test, yd_test = X_test[1:], yv_test[1:], yd_test[1:]

    garch = fit_garch(train_df["log_return"])
    joblib.dump(garch, MODEL_DIR / "garch_btc.pkl")

    callbacks = [
        tf.keras.callbacks.EarlyStopping(monitor="val_loss", patience=4, restore_best_weights=True),
        tf.keras.callbacks.ReduceLROnPlateau(monitor="val_loss", factor=0.5, patience=2, min_lr=1e-5),
    ]

    lstm = build_lstm(SEQ_LEN, len(FEATURE_COLS))
    lstm.fit(
        X_train, {"volatility": yv_train, "direction": yd_train},
        validation_data=(X_valid, {"volatility": yv_valid, "direction": yd_valid}),
        epochs=epochs, batch_size=batch_size, verbose=1, callbacks=callbacks,
    )
    lstm.save(MODEL_DIR / "lstm_model.keras")

    transformer = build_transformer(SEQ_LEN, len(FEATURE_COLS))
    transformer.fit(
        X_train, {"volatility": yv_train, "direction": yd_train},
        validation_data=(X_valid, {"volatility": yv_valid, "direction": yd_valid}),
        epochs=epochs, batch_size=batch_size, verbose=1, callbacks=callbacks,
    )
    transformer.save(MODEL_DIR / "transformer_model.keras")

    metadata = {
        "symbol": "BTC/USDT", "sequence_length": SEQ_LEN, "features": FEATURE_COLS,
        "horizons": HORIZONS, "train_rows": len(train_df), "valid_rows": len(valid_df), "test_rows": len(test_df),
    }
    with open(MODEL_DIR / "metadata.json", "w", encoding="utf-8") as f:
        json.dump(metadata, f, indent=2)

    np.savez(MODEL_DIR / "test_predictions_cache.npz", yv=yv_test, yd=yd_test)
    return {"lstm": lstm, "transformer": transformer, "garch": garch, "test": test_df, "X_test": X_test, "yv_test": yv_test, "yd_test": yd_test}


if __name__ == "__main__":
    artifacts = train_all()
    print(f"Training complete. Test sequences: {len(artifacts['X_test'])}")


## FILE: models\transformer_model.py

"""Lightweight Transformer encoder for multi-task volatility + direction forecasting."""
from __future__ import annotations

import tensorflow as tf
from tensorflow.keras import Model, layers


def encoder_block(x, head_size=32, num_heads=4, ff_dim=64, dropout=0.1):
    attn = layers.MultiHeadAttention(num_heads=num_heads, key_dim=head_size, dropout=dropout)(x, x)
    x = layers.LayerNormalization(epsilon=1e-6)(x + layers.Dropout(dropout)(attn))
    ff = layers.Dense(ff_dim, activation="gelu")(x)
    ff = layers.Dense(x.shape[-1])(ff)
    return layers.LayerNormalization(epsilon=1e-6)(x + layers.Dropout(dropout)(ff))


def build_transformer(seq_len, n_features, head_size=32, num_heads=4, ff_dim=64, num_blocks=2, dropout=0.1):
    inputs = layers.Input(shape=(seq_len, n_features), name="features")
    x = inputs
    for _ in range(num_blocks):
        x = encoder_block(x, head_size, num_heads, ff_dim, dropout)
    x = layers.GlobalAveragePooling1D()(x)
    x = layers.Dense(48, activation="gelu")(x)
    x = layers.Dropout(dropout)(x)
    vol = layers.Dense(3, activation="softplus", name="volatility")(x)
    direction = layers.Dense(3, activation="sigmoid", name="direction")(x)
    model = Model(inputs, {"volatility": vol, "direction": direction})
    model.compile(
        optimizer=tf.keras.optimizers.Adam(1e-3),
        loss={"volatility": "mse", "direction": "binary_crossentropy"},
        loss_weights={"volatility": 1.0, "direction": 0.5},
        metrics={"volatility": ["mae"], "direction": ["accuracy"]},
    )
    return model


## FILE: models\__init__.py


## FILE: serving\inference_api.py

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


## FILE: serving\__init__.py


## FILE: trading\execution.py

"""Deterministic paper-trading engine. Never sends a live order."""
from __future__ import annotations

import json
from datetime import datetime, timezone
from pathlib import Path

from config import PAPER_DIR, TRANSACTION_COST


class PaperBroker:
    def __init__(self, capital: float, path: Path | None = None):
        self.starting_cash = float(capital)
        self.cash = float(capital)
        self.position = 0.0
        self.avg_entry = 0.0
        self.realized_pnl = 0.0
        self.path = path or (PAPER_DIR / "paper_account.json")
        if self.path.exists():
            self._load()

    def _load(self):
        data = json.loads(self.path.read_text(encoding="utf-8"))
        self.starting_cash = data.get("starting_cash", self.starting_cash)
        self.cash = data.get("cash", self.cash)
        self.position = data.get("position", self.position)
        self.avg_entry = data.get("avg_entry", self.avg_entry)
        self.realized_pnl = data.get("realized_pnl", self.realized_pnl)

    def _save(self):
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self.path.write_text(json.dumps(self.snapshot(), indent=2), encoding="utf-8")

    def snapshot(self, mark_price: float | None = None):
        equity = self.cash + (self.position * mark_price if mark_price is not None else 0.0)
        return {"starting_cash": self.starting_cash, "cash": self.cash, "position": self.position,
                "avg_entry": self.avg_entry, "realized_pnl": self.realized_pnl, "equity": equity}

    def market_order(self, side: str, quantity: float, price: float):
        quantity = float(quantity)
        price = float(price)
        fee = abs(quantity * price) * TRANSACTION_COST
        if side == "BUY":
            cost = quantity * price + fee
            if cost > self.cash:
                raise ValueError("Insufficient paper cash")
            new_pos = self.position + quantity
            self.avg_entry = ((self.position * self.avg_entry) + quantity * price) / max(new_pos, 1e-9)
            self.position = new_pos
            self.cash -= cost
        elif side in ("SELL", "REDUCE"):
            quantity = min(quantity, self.position)
            proceeds = quantity * price - fee
            self.realized_pnl += quantity * (price - self.avg_entry) - fee
            self.position -= quantity
            self.cash += proceeds
            if self.position <= 1e-12:
                self.position, self.avg_entry = 0.0, 0.0
        else:
            raise ValueError("side must be BUY or SELL/REDUCE")
        self._save()
        return self.snapshot(price)


## FILE: trading\risk_manager.py

"""Paper-trading risk controls: ATR stop, 2R target, position sizing and kill switch."""
from __future__ import annotations

from dataclasses import dataclass

from config import MAX_DRAWDOWN_LIMIT, MAX_POSITION_RISK_PCT, STOP_LOSS_ATR_MULTIPLIER, TAKE_PROFIT_RR


@dataclass
class TradePlan:
    side: str
    entry_price: float
    quantity: float
    stop_price: float
    take_profit: float
    risk_amount: float


class RiskManager:
    def __init__(self, capital: float):
        self.capital = float(capital)
        self.peak_capital = float(capital)
        self.trading_enabled = True

    def position_size(self, entry_price: float, stop_price: float) -> float:
        distance = abs(entry_price - stop_price)
        if distance <= 0 or not self.trading_enabled:
            return 0.0
        return (self.capital * MAX_POSITION_RISK_PCT) / distance

    def plan(self, entry_price: float, atr: float, side: str = "long") -> TradePlan:
        distance = max(float(atr) * STOP_LOSS_ATR_MULTIPLIER, entry_price * 0.005)
        stop = entry_price - distance if side == "long" else entry_price + distance
        target = entry_price + TAKE_PROFIT_RR * distance if side == "long" else entry_price - TAKE_PROFIT_RR * distance
        qty = self.position_size(entry_price, stop)
        return TradePlan(side, entry_price, qty, stop, target, qty * distance)

    def update_capital(self, new_capital: float) -> bool:
        self.capital = float(new_capital)
        self.peak_capital = max(self.peak_capital, self.capital)
        drawdown = (self.peak_capital - self.capital) / max(self.peak_capital, 1e-9)
        if drawdown >= MAX_DRAWDOWN_LIMIT:
            self.trading_enabled = False
        return self.trading_enabled


## FILE: trading\signal_generator.py

"""Explainable signal engine using direction + volatility + confidence."""
from __future__ import annotations

from dataclasses import dataclass
import numpy as np

from config import SIGNAL_LOOKBACK


@dataclass
class Signal:
    action: str
    confidence: float
    risk_level: str
    reason: str


class SignalGenerator:
    def __init__(self, lookback: int = SIGNAL_LOOKBACK):
        self.vol_history: list[float] = []
        self.lookback = lookback

    def generate(self, predicted_direction: float, predicted_volatility: float, reference_volatility: float | None = None) -> Signal:
        self.vol_history.append(float(predicted_volatility))
        hist = self.vol_history[-self.lookback:]
        ref = reference_volatility if reference_volatility is not None else float(np.median(hist))
        vol_ratio = predicted_volatility / max(ref, 1e-9)
        direction_score = abs(predicted_direction - 0.5) * 2
        confidence = float(np.clip(0.55 * direction_score + 0.45 * min(abs(vol_ratio - 1) + 0.35, 1), 0, 1))

        if vol_ratio >= 1.8:
            action, risk = "REDUCE", "HIGH"
            reason = "Forecast volatility is materially above its recent baseline."
        elif predicted_direction >= 0.55 and vol_ratio <= 1.5:
            action, risk = "BUY", "LOW" if vol_ratio < 1.0 else "MEDIUM"
            reason = "Positive direction probability with acceptable forecast risk."
        elif predicted_direction <= 0.45:
            action, risk = "SELL", "MEDIUM" if vol_ratio < 1.5 else "HIGH"
            reason = "Negative direction probability dominates."
        else:
            action, risk = "HOLD", "MEDIUM"
            reason = "Direction edge is weak relative to model uncertainty."
        return Signal(action, confidence, risk, reason)


## FILE: trading\__init__.py

