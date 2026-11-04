"""Comprehensive training orchestrator and model registry pipeline.

Fits and saves:
1. Naive Volatility Baseline (Historical rolling benchmark)
2. GARCH(1,1) Econometric Model
3. Multi-Task Deep Sequential LSTM Model
4. Multi-Head Self-Attention Transformer Model
5. Ensemble Predictor with Conformal Empirical Uncertainty Intervals
Saves all artifacts under artifacts/ with metadata versioning.
"""
from __future__ import annotations

import json
import random
from pathlib import Path
from typing import Any

import joblib
import numpy as np
import pandas as pd

from config import (
    ARTIFACTS_DIR,
    FORECAST_HORIZONS,
    PROCESSED_DIR,
    RANDOM_SEED,
    SEQ_LEN,
)
from data_pipeline.feature_engineering import CANONICAL_FEATURES, fit_and_save_feature_store
from models.garch_model import PureGARCH, fit_garch, forecast_volatility, save_garch_artifact
from models.lstm_model import LSTMModel, build_lstm, make_sequences
from models.transformer_model import TransformerModel, build_transformer


def seed_everything(seed: int = RANDOM_SEED):
    """Set global seeds for reproducibility across runs."""
    random.seed(seed)
    np.random.seed(seed)


def load_datasets() -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """Load chronological Train, Validation, and Test datasets."""
    p_train = PROCESSED_DIR / "train.csv"
    p_valid = PROCESSED_DIR / "valid.csv"
    p_test = PROCESSED_DIR / "test.csv"

    if not (p_train.exists() and p_valid.exists() and p_test.exists()):
        raise FileNotFoundError(
            f"Preprocessed data not found in {PROCESSED_DIR}. Run data pipeline first."
        )

    train_df = pd.read_csv(p_train, parse_dates=["open_time"])
    valid_df = pd.read_csv(p_valid, parse_dates=["open_time"])
    test_df = pd.read_csv(p_test, parse_dates=["open_time"])
    return train_df, valid_df, test_df


def extract_targets(df: pd.DataFrame) -> tuple[np.ndarray, np.ndarray]:
    """Extract volatility and direction target matrices."""
    vol_targets = df[[f"realized_vol_{h}" for h in FORECAST_HORIZONS]].to_numpy(dtype=np.float32)
    dir_targets = df[[f"direction_{h}" for h in FORECAST_HORIZONS]].to_numpy(dtype=np.float32)
    return vol_targets, dir_targets


def train_all_models(
    epochs: int = 15,
    batch_size: int = 64,
) -> dict[str, Any]:
    """Train complete model suite with leak-safe chronological splits."""
    seed_everything()
    print("[1/5] Loading chronological dataset splits...")
    train_df, valid_df, test_df = load_datasets()

    print("[2/5] Fitting standard scaler on training split ONLY...")
    scaler, feature_names = fit_and_save_feature_store(train_df, valid_df, test_df)

    # Scale feature arrays
    X_train_raw = scaler.transform(train_df[feature_names])
    X_valid_raw = scaler.transform(valid_df[feature_names])
    X_test_raw = scaler.transform(test_df[feature_names])

    yv_train_raw, yd_train_raw = extract_targets(train_df)
    yv_valid_raw, yd_valid_raw = extract_targets(valid_df)
    yv_test_raw, yd_test_raw = extract_targets(test_df)

    # Construct rolling 3D sequence tensors: (N - seq_len + 1, seq_len, n_features)
    X_train, yv_train, yd_train = make_sequences(X_train_raw, yv_train_raw, yd_train_raw, SEQ_LEN)

    # Context concatenation for validation and test sequence windows
    valid_context = pd.concat([train_df.iloc[-SEQ_LEN:], valid_df]).reset_index(drop=True)
    X_val_ctx = scaler.transform(valid_context[feature_names])
    yv_val_ctx, yd_val_ctx = extract_targets(valid_context)
    X_valid, yv_valid, yd_valid = make_sequences(X_val_ctx, yv_val_ctx, yd_val_ctx, SEQ_LEN)

    test_context = pd.concat([valid_df.iloc[-SEQ_LEN:], test_df]).reset_index(drop=True)
    X_test_ctx = scaler.transform(test_context[feature_names])
    yv_test_ctx, yd_test_ctx = extract_targets(test_context)
    X_test, yv_test, yd_test = make_sequences(X_test_ctx, yv_test_ctx, yd_test_ctx, SEQ_LEN)

    # Trim to match exactly the split length
    X_valid, yv_valid, yd_valid = X_valid[:len(valid_df)], yv_valid[:len(valid_df)], yd_valid[:len(valid_df)]
    X_test, yv_test, yd_test = X_test[:len(test_df)], yv_test[:len(test_df)], yd_test[:len(test_df)]

    # 3. Fit Baseline GARCH(1,1)
    print("[3/5] Fitting GARCH(1,1) statistical baseline...")
    garch_dir = ARTIFACTS_DIR / "garch"
    garch_dir.mkdir(parents=True, exist_ok=True)
    garch_model = fit_garch(train_df["log_return"])
    save_garch_artifact(garch_model, garch_dir / "garch_model.pkl")

    # 4. Train Deep Learning Multi-Task Models
    print("[4/5] Training Multi-Task LSTM model...")
    lstm_dir = ARTIFACTS_DIR / "lstm"
    lstm_dir.mkdir(parents=True, exist_ok=True)
    lstm = build_lstm(seq_len=SEQ_LEN, n_features=len(feature_names), units=48)
    lstm.fit(
        X_train, yv_train, yd_train,
        X_val=X_valid, y_vol_val=yv_valid, y_dir_val=yd_valid,
        epochs=epochs, batch_size=batch_size, lr=0.003
    )
    lstm.save(lstm_dir / "lstm_model")

    print("[5/5] Training Multi-Head Self-Attention Transformer model...")
    transformer_dir = ARTIFACTS_DIR / "transformer"
    transformer_dir.mkdir(parents=True, exist_ok=True)
    transformer = build_transformer(
        seq_len=SEQ_LEN, n_features=len(feature_names), d_model=32, num_heads=4, d_ff=64
    )
    transformer.fit(
        X_train, yv_train, yd_train,
        X_val=X_valid, y_vol_val=yv_valid, y_dir_val=yd_valid,
        epochs=epochs, batch_size=batch_size, lr=0.003
    )
    transformer.save(transformer_dir / "transformer_model")

    # Calculate empirical conformal uncertainty intervals on validation set
    val_preds_lstm = lstm.predict(X_valid)["volatility"]
    val_preds_trans = transformer.predict(X_valid)["volatility"]
    val_preds_ensemble = 0.5 * val_preds_lstm + 0.5 * val_preds_trans
    abs_errors = np.abs(yv_valid - val_preds_ensemble)
    uncertainty_quantiles = {
        f"{h}h": float(np.percentile(abs_errors[:, i], 90))
        for i, h in enumerate(FORECAST_HORIZONS)
    }

    # Baseline Naive Volatility on test split: rolling standard deviation of log returns
    naive_vol_test = np.column_stack([
        test_df[f"hist_vol_{h}"].fillna(test_df["log_return"].std()).to_numpy() if f"hist_vol_{h}" in test_df.columns
        else np.full(len(test_df), test_df["log_return"].std() * np.sqrt(h))
        for h in FORECAST_HORIZONS
    ])

    # Save model metadata
    model_metadata = {
        "model_version": "2.0.0",
        "random_seed": RANDOM_SEED,
        "sequence_length": SEQ_LEN,
        "n_features": len(feature_names),
        "features": feature_names,
        "horizons": list(FORECAST_HORIZONS),
        "conformal_uncertainty_quantiles_90pct": uncertainty_quantiles,
        "training_samples": len(X_train),
        "validation_samples": len(X_valid),
        "test_samples": len(X_test),
        "ensemble_weights": {"garch": 0.35, "lstm": 0.35, "transformer": 0.30},
    }
    (ARTIFACTS_DIR / "model_metadata.json").write_text(
        json.dumps(model_metadata, indent=2), encoding="utf-8"
    )

    # Cache test predictions for evaluation and backtesting
    np.savez_compressed(
        ARTIFACTS_DIR / "test_predictions.npz",
        X_test=X_test,
        yv_test=yv_test,
        yd_test=yd_test,
        naive_vol=naive_vol_test,
    )

    return {
        "garch": garch_model,
        "lstm": lstm,
        "transformer": transformer,
        "scaler": scaler,
        "metadata": model_metadata,
        "X_test": X_test,
        "yv_test": yv_test,
        "yd_test": yd_test,
        "test_df": test_df,
    }


if __name__ == "__main__":
    results = train_all_models(epochs=15)
    print("Training successfully complete! Model artifacts saved to artifacts/.")
