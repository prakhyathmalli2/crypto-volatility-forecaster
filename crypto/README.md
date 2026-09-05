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
Public exchange OHLCV ──┐
Free on-chain metrics ───┼─> cleaning/alignment ─> features/targets ─> GARCH
RSS news sentiment ──────┘                                     ├────> LSTM
                                                               ├────> Transformer
                                                               └────> Ensemble
                                                                      │
                                                         direction + volatility
                                                                      │
                                                         smart signal + risk
                                                                      │
                                               FastAPI <──> Streamlit dashboard
                                                                      │
                                                               paper broker
```

## API endpoints

- `GET /health` — service/model readiness
- `GET /market?timeframe=1h&days=7` — live chart data
- `GET /predict/latest?timeframe=1h` — ensemble forecast and signal
- `GET /paper/status` — paper account state
- `POST /paper/trade` — record a simulated BUY/SELL/REDUCE market order
- `/docs` — interactive FastAPI documentation

## Safety boundary

This release deliberately has **no live-order path**. The UI's trading controls write to a local paper account only. Any future private exchange integration should be isolated behind a separate credentialed execution service and testnet-first validation.
