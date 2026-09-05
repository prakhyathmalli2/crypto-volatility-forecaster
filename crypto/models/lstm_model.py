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
