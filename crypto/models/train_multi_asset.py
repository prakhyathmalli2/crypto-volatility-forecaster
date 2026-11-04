"""Asset-Specific Model Training, Chronological Validation, and Out-of-Sample Evaluation.

Provides dedicated, leak-safe models for:
1. BTC/USDT (Crypto Benchmark)
2. ETH/USDT (High-Beta Smart Contract Crypto)
3. ^NSEI (NIFTY 50 Indian Equity Index)

Guarantees:
- Strict chronological Train (70%) / Validation (15%) / Test (15%) splits
- Zero look-ahead bias: Scaler fit ONLY on training data
- Walk-forward out-of-sample performance evaluation tracking:
  * Directional Accuracy
  * Precision, Recall, F1-score
  * Brier Score (probability calibration)
  * Sharpe Ratio, Profit Factor, Max Drawdown, Win Rate, Avg Win/Loss Return
- Artifacts saved per asset under artifacts/<asset_slug>/
"""
from __future__ import annotations

import json
import os
import random
import sys
from pathlib import Path
from typing import Any

# Ensure project root is in sys.path
_PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(_PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(_PROJECT_ROOT))

import ccxt
import joblib
import numpy as np
import pandas as pd
import yfinance as yf

from config import ARTIFACTS_DIR, FORECAST_HORIZONS, PROCESSED_DIR, RANDOM_SEED, ROOT_DIR, SEQ_LEN
from data_pipeline.feature_engineering import CANONICAL_FEATURES, engineer_features
from data_pipeline.preprocess import clean_ohlcv, compute_realized_volatility_targets
from evaluation.backtest import Backtester
from evaluation.metrics import (
    average_trade_metrics,
    brier_score,
    correlation_score,
    directional_accuracy,
    mae,
    precision_recall_f1,
    rmse,
)
from models.garch_model import fit_garch, forecast_volatility, save_garch_artifact
from models.lstm_model import build_lstm, make_sequences
from models.transformer_model import build_transformer
from sklearn.preprocessing import StandardScaler


def seed_everything(seed: int = RANDOM_SEED):
    random.seed(seed)
    np.random.seed(seed)


def fetch_historical_dataset(symbol: str, timeframe: str = "1h", limit: int = 500) -> pd.DataFrame:
    """Fetch high-quality historical OHLCV data for the target asset."""
    if symbol in ("BTC/USDT", "ETH/USDT"):
        try:
            exchange = ccxt.binance({"enableRateLimit": True, "timeout": 15000})
            candles = exchange.fetch_ohlcv(symbol, timeframe=timeframe, limit=limit)
            if candles and len(candles) >= 100:
                df = pd.DataFrame(candles, columns=["timestamp", "open", "high", "low", "close", "volume"])
                df["open_time"] = pd.to_datetime(df.pop("timestamp"), unit="ms", utc=True)
                for col in ["open", "high", "low", "close", "volume"]:
                    df[col] = pd.to_numeric(df[col], errors="coerce")
                return clean_ohlcv(df)
        except Exception as e:
            print(f"CCXT fetch error for {symbol}: {e}")

    # Fallback or Equities via yfinance
    yf_symbol = "^NSEI" if symbol in ("^NSEI", "NIFTY 50", "NIFTY") else symbol.replace("/", "-")
    try:
        yf_df = yf.download(yf_symbol, period="1y", interval="1d" if yf_symbol == "^NSEI" else "1h", progress=False)
        if yf_df is not None and len(yf_df) >= 60:
            if isinstance(yf_df.columns, pd.MultiIndex):
                yf_df.columns = yf_df.columns.get_level_values(0)
            yf_df = yf_df.reset_index()
            col_map = {
                "Date": "open_time",
                "Datetime": "open_time",
                "Open": "open",
                "High": "high",
                "Low": "low",
                "Close": "close",
                "Volume": "volume",
            }
            yf_df = yf_df.rename(columns=col_map)
            yf_df["open_time"] = pd.to_datetime(yf_df["open_time"], utc=True)
            for col in ["open", "high", "low", "close", "volume"]:
                yf_df[col] = pd.to_numeric(yf_df[col], errors="coerce")
            return clean_ohlcv(yf_df[["open_time", "open", "high", "low", "close", "volume"]])
    except Exception as e:
        print(f"yfinance fetch error for {symbol}: {e}")

    # Local fallback for BTC if present
    if symbol == "BTC/USDT":
        p = ROOT_DIR / "data" / "raw" / "market" / "btc_usdt_1h.csv"
        if p.exists():
            df = pd.read_csv(p, parse_dates=["open_time"])
            return clean_ohlcv(df)

    raise RuntimeError(f"Unable to load sufficient historical data for {symbol}")


def train_and_validate_asset(
    symbol: str,
    asset_slug: str,
    epochs: int = 12,
    batch_size: int = 32,
    timeframe: str = "1h",
) -> dict[str, Any]:
    """Train asset-specific models with walk-forward validation and leak-safe chronological splits."""
    seed_everything()
    print(f"\n=======================================================")
    print(f"  ASSET PIPELINE: {symbol} (slug: {asset_slug})")
    print(f"=======================================================")

    # 1. Collect and clean historical data
    print(f"[{symbol}] Fetching and cleaning OHLCV data...")
    raw_df = fetch_historical_dataset(symbol, timeframe=timeframe, limit=500)
    print(f"[{symbol}] Loaded {len(raw_df)} clean candles.")

    # 2. Engineer features with zero look-ahead bias
    print(f"[{symbol}] Engineering technical and volatility features...")
    feat_df = engineer_features(raw_df)
    
    # 3. Add realized volatility targets
    print(f"[{symbol}] Computing forward-looking realized volatility targets...")
    df_with_targets = compute_realized_volatility_targets(feat_df, horizons=FORECAST_HORIZONS)

    # Filter canonical features
    feature_names = [col for col in CANONICAL_FEATURES if col in df_with_targets.columns]

    # 4. Strict Chronological Split: 70% Train | 15% Valid | 15% Test
    # Purge forward-looking label boundary rows so labels in one split never use prices from the next split
    n = len(df_with_targets)
    train_end = int(n * 0.70)
    valid_end = int(n * 0.85)
    max_horizon = max(FORECAST_HORIZONS)

    # Purge forward-label horizon rows from each split boundary to avoid leakage
    # Train split: exclude the last `max_horizon` rows so its forward targets never use validation data
    train_df = df_with_targets.iloc[: max(0, train_end - max_horizon)].copy().reset_index(drop=True)
    # Validation split: start after the purged horizon following the train split and also exclude its last horizon rows
    valid_start = min(train_end + max_horizon, valid_end)  # ensure we don't exceed valid_end
    valid_end_purged = max(train_end, valid_end - max_horizon)
    valid_df = df_with_targets.iloc[valid_start:valid_end_purged].copy().reset_index(drop=True)
    # Test split: starts at the original `valid_end` (no later split to leak into)
    test_df = df_with_targets.iloc[valid_end:].copy().reset_index(drop=True)

    print(f"[{symbol}] Split breakdown: Train={len(train_df)}, Valid={len(valid_df)}, Test={len(test_df)}")

    # 5. Fit Standard Scaler on TRAIN split ONLY (prevents data leakage)
    scaler = StandardScaler()
    scaler.fit(train_df[feature_names])

    X_train_raw = scaler.transform(train_df[feature_names])
    X_valid_raw = scaler.transform(valid_df[feature_names])
    X_test_raw = scaler.transform(test_df[feature_names])

    yv_train_raw = train_df[[f"realized_vol_{h}" for h in FORECAST_HORIZONS]].to_numpy(dtype=np.float32)
    yd_train_raw = train_df[[f"direction_{h}" for h in FORECAST_HORIZONS]].to_numpy(dtype=np.float32)

    yv_valid_raw = valid_df[[f"realized_vol_{h}" for h in FORECAST_HORIZONS]].to_numpy(dtype=np.float32)
    yd_valid_raw = valid_df[[f"direction_{h}" for h in FORECAST_HORIZONS]].to_numpy(dtype=np.float32)

    yv_test_raw = test_df[[f"realized_vol_{h}" for h in FORECAST_HORIZONS]].to_numpy(dtype=np.float32)
    yd_test_raw = test_df[[f"direction_{h}" for h in FORECAST_HORIZONS]].to_numpy(dtype=np.float32)

    # Dynamic sequence length (adapts to dataset size if dataset < 150)
    seq_len = min(SEQ_LEN, max(8, len(train_df) // 10))

    X_train, yv_train, yd_train = make_sequences(X_train_raw, yv_train_raw, yd_train_raw, seq_len)

    # Context concatenation for validation and test sequence windows
    valid_context = pd.concat([train_df.iloc[-seq_len:], valid_df]).reset_index(drop=True)
    X_val_ctx = scaler.transform(valid_context[feature_names])
    yv_val_ctx = valid_context[[f"realized_vol_{h}" for h in FORECAST_HORIZONS]].to_numpy(dtype=np.float32)
    yd_val_ctx = valid_context[[f"direction_{h}" for h in FORECAST_HORIZONS]].to_numpy(dtype=np.float32)
    X_valid, yv_valid, yd_valid = make_sequences(X_val_ctx, yv_val_ctx, yd_val_ctx, seq_len)

    test_context = pd.concat([valid_df.iloc[-seq_len:], test_df]).reset_index(drop=True)
    X_test_ctx = scaler.transform(test_context[feature_names])
    yv_test_ctx = test_context[[f"realized_vol_{h}" for h in FORECAST_HORIZONS]].to_numpy(dtype=np.float32)
    yd_test_ctx = test_context[[f"direction_{h}" for h in FORECAST_HORIZONS]].to_numpy(dtype=np.float32)
    X_test, yv_test, yd_test = make_sequences(X_test_ctx, yv_test_ctx, yd_test_ctx, seq_len)

    X_valid = X_valid[:len(valid_df)]
    yv_valid = yv_valid[:len(valid_df)]
    yd_valid = yd_valid[:len(valid_df)]

    X_test = X_test[:len(test_df)]
    yv_test = yv_test[:len(test_df)]
    yd_test = yd_test[:len(test_df)]

    # 6. Fit Statistical Baseline GARCH(1,1)
    print(f"[{symbol}] Fitting GARCH(1,1) econometric baseline...")
    garch_model = fit_garch(train_df["log_return"])

    # 7. Fit Multi-Task LSTM
    print(f"[{symbol}] Fitting Multi-Task Deep LSTM...")
    lstm = build_lstm(seq_len=seq_len, n_features=len(feature_names), units=36)
    lstm.fit(
        X_train, yv_train, yd_train,
        X_val=X_valid, y_vol_val=yv_valid, y_dir_val=yd_valid,
        epochs=epochs, batch_size=batch_size, lr=0.003
    )

    # 8. Fit Self-Attention Transformer
    print(f"[{symbol}] Fitting Self-Attention Transformer...")
    transformer = build_transformer(
        seq_len=seq_len, n_features=len(feature_names), d_model=32, num_heads=4, d_ff=64
    )
    transformer.fit(
        X_train, yv_train, yd_train,
        X_val=X_valid, y_vol_val=yv_valid, y_dir_val=yd_valid,
        epochs=epochs, batch_size=batch_size, lr=0.003
    )

    # 9. Conformal Uncertainty Quantiles on Validation Split
    val_preds_lstm = lstm.predict(X_valid)["volatility"]
    val_preds_trans = transformer.predict(X_valid)["volatility"]
    val_preds_ens = 0.5 * val_preds_lstm + 0.5 * val_preds_trans
    abs_errors = np.abs(yv_valid - val_preds_ens)
    uncertainty_quantiles = {
        f"{h}h": float(np.percentile(abs_errors[:, i], 90))
        for i, h in enumerate(FORECAST_HORIZONS)
    }

    # 10. Out-of-Sample Evaluation on Held-Out TEST Split
    print(f"[{symbol}] Generating Out-of-Sample Predictions on Test Split...")
    test_preds_lstm = lstm.predict(X_test)
    test_preds_trans = transformer.predict(X_test)

    garch_fc = forecast_volatility(garch_model, horizons=FORECAST_HORIZONS)
    garch_vol_test = np.column_stack([
        np.full(len(X_test), garch_fc[h]) for h in FORECAST_HORIZONS
    ])

    lstm_vol_test = test_preds_lstm["volatility"]
    lstm_dir_test = test_preds_lstm["direction"]
    trans_vol_test = test_preds_trans["volatility"]
    trans_dir_test = test_preds_trans["direction"]

    # Ensemble combinations
    ens_vol_test = 0.35 * garch_vol_test + 0.35 * lstm_vol_test + 0.30 * trans_vol_test
    ens_dir_test = 0.50 * lstm_dir_test + 0.50 * trans_dir_test

    # Horizon 1 (next period) metrics
    y_true_dir_1h = yd_test[:, 0]
    y_pred_dir_1h = ens_dir_test[:, 0]

    y_true_vol_1h = yv_test[:, 0]
    y_pred_vol_1h = ens_vol_test[:, 0]

    # A. Volatility Accuracy
    vol_rmse = rmse(y_true_vol_1h, y_pred_vol_1h)
    vol_mae = mae(y_true_vol_1h, y_pred_vol_1h)
    vol_corr = correlation_score(y_true_vol_1h, y_pred_vol_1h)

    # B. Directional Classification Metrics
    dir_acc = directional_accuracy(y_true_dir_1h, y_pred_dir_1h)
    clf_metrics = precision_recall_f1(y_true_dir_1h, y_pred_dir_1h, threshold=0.50)
    brier = brier_score(y_true_dir_1h, y_pred_dir_1h)

    # C. Trading Simulation on Out-of-Sample Test Set
    backtester = Backtester(test_df)
    sim_results, equity_curve, positions = backtester.run(
        predicted_direction=y_pred_dir_1h,
        predicted_volatility=y_pred_vol_1h,
    )
    trade_returns = np.diff(equity_curve) / equity_curve[:-1] if len(equity_curve) > 1 else np.array([])
    trade_metrics = average_trade_metrics(trade_returns)

    oos_report = {
        "asset": symbol,
        "asset_slug": asset_slug,
        "timeframe": timeframe,
        "test_sample_count": len(test_df),
        "test_date_range": [str(test_df["open_time"].iloc[0]), str(test_df["open_time"].iloc[-1])],
        "volatility_evaluation": {
            "rmse": round(vol_rmse, 6),
            "mae": round(vol_mae, 6),
            "pearson_correlation": round(vol_corr, 4),
        },
        "directional_classification": {
            "directional_accuracy_pct": round(dir_acc * 100.0, 2),
            "precision": clf_metrics["precision"],
            "recall": clf_metrics["recall"],
            "f1_score": clf_metrics["f1"],
            "brier_score": round(brier, 4),
            "confusion_matrix": {
                "tp": clf_metrics["true_positives"],
                "fp": clf_metrics["false_positives"],
                "tn": clf_metrics["true_negatives"],
                "fn": clf_metrics["false_negatives"],
            },
        },
        "trading_metrics": {
            "sharpe_ratio": round(sim_results["sharpe_ratio"], 2),
            "sortino_ratio": round(sim_results["sortino_ratio"], 2),
            "profit_factor": round(sim_results["profit_factor"], 2),
            "max_drawdown_pct": round(sim_results["max_drawdown_pct"], 2),
            "win_rate_pct": round(sim_results["win_rate_pct"], 2),
            "total_trades": sim_results["total_trades"],
            "strategy_return_pct": round(sim_results["total_net_return_pct"], 2),
            "buy_and_hold_return_pct": round(sim_results["buy_and_hold_return_pct"], 2),
            "avg_win_return_pct": trade_metrics["avg_win_return_pct"],
            "avg_loss_return_pct": trade_metrics["avg_loss_return_pct"],
        },
        "model_weights": {"garch": 0.35, "lstm": 0.35, "transformer": 0.30},
        "conformal_uncertainty_quantiles_90pct": uncertainty_quantiles,
    }

    # 11. Persist Artifacts
    target_dir = ARTIFACTS_DIR / asset_slug
    target_dir.mkdir(parents=True, exist_ok=True)
    (target_dir / "garch").mkdir(parents=True, exist_ok=True)
    (target_dir / "lstm").mkdir(parents=True, exist_ok=True)
    (target_dir / "transformer").mkdir(parents=True, exist_ok=True)

    joblib.dump(scaler, target_dir / "scaler.pkl")
    save_garch_artifact(garch_model, target_dir / "garch" / "garch_model.pkl")
    lstm.save(target_dir / "lstm" / "lstm_model")
    transformer.save(target_dir / "transformer" / "transformer_model")

    (target_dir / "feature_schema.json").write_text(json.dumps(feature_names, indent=2), encoding="utf-8")
    (target_dir / "test_metrics.json").write_text(json.dumps(oos_report, indent=2), encoding="utf-8")

    # If BTC, also update default root artifacts for backward compatibility
    if asset_slug == "btc_usdt":
        joblib.dump(scaler, ARTIFACTS_DIR / "scaler.pkl")
        save_garch_artifact(garch_model, ARTIFACTS_DIR / "garch" / "garch_model.pkl")
        lstm.save(ARTIFACTS_DIR / "lstm" / "lstm_model")
        transformer.save(ARTIFACTS_DIR / "transformer" / "transformer_model")
        (ARTIFACTS_DIR / "feature_schema.json").write_text(json.dumps(feature_names, indent=2), encoding="utf-8")
        (ARTIFACTS_DIR / "model_metadata.json").write_text(json.dumps(oos_report, indent=2), encoding="utf-8")

    print(f"[{symbol}] Completed. Directional Accuracy: {oos_report['directional_classification']['directional_accuracy_pct']}%, Sharpe: {oos_report['trading_metrics']['sharpe_ratio']}, Profit Factor: {oos_report['trading_metrics']['profit_factor']}")
    return oos_report


def run_all_asset_validations() -> dict[str, Any]:
    """Execute end-to-end training and out-of-sample validation for BTC, ETH, and NIFTY 50."""
    assets = [
        {"symbol": "BTC/USDT", "slug": "btc_usdt", "timeframe": "1h"},
        {"symbol": "ETH/USDT", "slug": "eth_usdt", "timeframe": "1h"},
        {"symbol": "^NSEI", "slug": "nsei", "timeframe": "1d"},
    ]

    master_results = {}
    for a in assets:
        try:
            res = train_and_validate_asset(
                symbol=a["symbol"],
                asset_slug=a["slug"],
                epochs=12,
                timeframe=a["timeframe"],
            )
            master_results[a["slug"]] = res
        except Exception as e:
            print(f"Failed pipeline for {a['symbol']}: {e}")
            import traceback
            traceback.print_exc()

    report_path = ARTIFACTS_DIR / "multi_asset_validation_report.json"
    report_path.write_text(json.dumps(master_results, indent=2), encoding="utf-8")
    print(f"\nSaved Multi-Asset Out-of-Sample Validation Report to: {report_path}")
    return master_results


if __name__ == "__main__":
    run_all_asset_validations()
