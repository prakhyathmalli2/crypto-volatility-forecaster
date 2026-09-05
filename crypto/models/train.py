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
