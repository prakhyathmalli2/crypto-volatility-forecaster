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
