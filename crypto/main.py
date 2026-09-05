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
