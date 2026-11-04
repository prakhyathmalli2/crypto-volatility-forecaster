"""Master orchestration script for the Crypto Volatility Forecaster & Signal Generator.

Executes the complete production ML pipeline end-to-end:
[Step 1] Ingest multi-source market, on-chain, and sentiment data
[Step 2] Clean and align sources with leak-safe backward as-of joins
[Step 3] Calculate forward realized volatility targets and engineered features
[Step 4] Fit and serialize training scaler and canonical feature schema
[Step 5] Train baseline Naive, GARCH(1,1), Deep LSTM, and Transformer models
[Step 6] Evaluate quantitative metrics and run held-out walk-forward backtest
[Step 7] Export evaluation report and backtest equity curve
"""
from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from config import (
    ARTIFACTS_DIR,
    ASSETS,
    DEFAULT_SYMBOL,
    DEFAULT_TIMEFRAME,
    INITIAL_CAPITAL,
    PROCESSED_DIR,
    RAW_DIR,
)
from data_pipeline.feature_engineering import (
    CANONICAL_FEATURES,
    engineer_features,
    fit_and_save_feature_store,
)
from data_pipeline.ingest_market_data import fetch_historical_ohlcv, save_market_dataset
from data_pipeline.ingest_onchain_data import build_onchain_dataset, save_onchain_dataset
from data_pipeline.ingest_sentiment_data import build_sentiment_dataset, save_sentiment_dataset
from data_pipeline.preprocess import (
    clean_ohlcv,
    compute_realized_volatility_targets,
    merge_sources_leak_free,
    split_chronological_datasets,
)
from evaluation.backtest import generate_comparative_evaluation_report
from models.train import train_all_models


def run_pipeline(train_epochs: int = 15, lookback_days: int = 730) -> dict[str, Any]:
    """Execute the full end-to-end machine learning and trading pipeline."""
    print("=" * 70)
    print(" CRYPTO VOLATILITY FORECASTER & SMART TRADING SIGNAL GENERATOR")
    print("=" * 70)

    # -------------------------------------------------------------
    # STEP 1: Multi-Source Data Ingestion
    # -------------------------------------------------------------
    print("\n[STEP 1/6] Ingesting multi-source historical & live market data...")
    market_paths = save_market_dataset(symbol=DEFAULT_SYMBOL, timeframe=DEFAULT_TIMEFRAME, lookback_days=lookback_days)
    print(f"  -> Market data saved: {market_paths['csv']}")

    print("[STEP 2/6] Ingesting on-chain metrics and Fear & Greed sentiment...")
    try:
        onchain_df, _ = build_onchain_dataset("2years")
        save_onchain_dataset("2years")
        print(f"  -> On-chain metrics collected: {len(onchain_df)} records")
    except Exception as exc:
        print(f"  -> On-chain fallback used: {exc}")
        onchain_df = None

    try:
        sentiment_df, _ = build_sentiment_dataset()
        save_sentiment_dataset()
        print(f"  -> Sentiment records collected: {len(sentiment_df)} days")
    except Exception as exc:
        print(f"  -> Sentiment fallback used: {exc}")
        sentiment_df = None

    # -------------------------------------------------------------
    # STEP 2 & 3: Cleaning, Leak-Safe Alignment & Feature Engineering
    # -------------------------------------------------------------
    print("\n[STEP 3/6] Cleaning, aligning sources with backward as-of join & calculating forward targets...")
    raw_ohlcv = pd.read_csv(market_paths["csv"], parse_dates=["open_time"])
    cleaned = clean_ohlcv(raw_ohlcv)
    aligned = merge_sources_leak_free(cleaned, onchain_df=onchain_df, sentiment_df=sentiment_df)
    targeted = compute_realized_volatility_targets(aligned)
    featured = engineer_features(targeted)

    for col in CANONICAL_FEATURES:
        if col not in featured.columns:
            featured[col] = 0.0

    featured.to_csv(PROCESSED_DIR / "btc_features.csv", index=False)
    print(f"  -> Engineered dataset: {len(featured)} rows with {len(CANONICAL_FEATURES)} features")

    # -------------------------------------------------------------
    # STEP 4: Chronological Splitting
    # -------------------------------------------------------------
    print("\n[STEP 4/6] Creating leak-safe chronological Train / Valid / Test splits...")
    splits = split_chronological_datasets(featured)
    print(f"  -> Train: {len(splits['train'])} | Valid: {len(splits['valid'])} | Test: {len(splits['test'])}")

    # -------------------------------------------------------------
    # STEP 5: Model Training
    # -------------------------------------------------------------
    print("\n[STEP 5/6] Fitting GARCH(1,1), Deep LSTM & Self-Attention Transformer...")
    training_results = train_all_models(epochs=train_epochs)

    # -------------------------------------------------------------
    # STEP 6: Quantitative Evaluation & Backtest Simulation
    # -------------------------------------------------------------
    print("\n[STEP 6/6] Evaluating forecast accuracy and simulating held-out walk-forward backtest...")
    lstm_preds = training_results["lstm"].predict(training_results["X_test"])
    trans_preds = training_results["transformer"].predict(training_results["X_test"])

    # Load test prediction cache
    cache = np.load(ARTIFACTS_DIR / "test_predictions.npz")
    naive_vol = cache["naive_vol"]
    garch_fc = training_results["garch"].forecast(horizons=(1, 6, 24))
    garch_vol = np.tile(
        [garch_fc[1], garch_fc[6], garch_fc[24]], (len(training_results["yv_test"]), 1)
    )

    report = generate_comparative_evaluation_report(
        test_df=training_results["test_df"],
        yv_test=training_results["yv_test"],
        yd_test=training_results["yd_test"],
        naive_vol=naive_vol,
        garch_vol=garch_vol,
        lstm_vol=lstm_preds["volatility"],
        lstm_dir=lstm_preds["direction"],
        transformer_vol=trans_preds["volatility"],
        transformer_dir=trans_preds["direction"],
    )

    print("\n" + "=" * 70)
    print(" MODEL PERFORMANCE COMPARISON (HELD-OUT TEST SET)")
    print("=" * 70)
    comp_df = pd.DataFrame(report["comparison_table"])
    print(comp_df.to_string(index=False))

    print("\n" + "=" * 70)
    print(" ENSEMBLE TRADING BACKTEST SUMMARY (0.1% FEE + SLIPPAGE)")
    print("=" * 70)
    for k, v in report["ensemble_backtest"].items():
        print(f"  {k:30s}: {v}")
    print("=" * 70)

    return report


if __name__ == "__main__":
    run_pipeline(train_epochs=15, lookback_days=730)
