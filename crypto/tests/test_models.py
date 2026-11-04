"""Unit tests for GARCH, LSTM, Transformer, and quantitative metrics."""
from __future__ import annotations

import numpy as np
import pytest

from evaluation.metrics import (
    directional_accuracy,
    mae,
    max_drawdown,
    rmse,
    sharpe_ratio,
)
from models.garch_model import PureGARCH, fit_garch, forecast_volatility
from models.lstm_model import LSTMModel, build_lstm
from models.transformer_model import TransformerModel, build_transformer


def test_garch_model_fit_and_forecast():
    np.random.seed(42)
    returns = np.random.normal(0, 0.015, 300)
    model = fit_garch(returns)
    assert model.fitted
    assert model.omega > 0
    assert model.alpha >= 0
    assert model.beta >= 0
    assert (model.alpha + model.beta) < 1.0

    fc = forecast_volatility(model, horizons=(1, 6, 24))
    assert len(fc) == 3
    assert fc[1] > 0
    assert fc[6] > fc[1]  # Cumulative volatility increases with horizon
    assert fc[24] > fc[6]


def test_lstm_forward_and_bounds():
    np.random.seed(42)
    model = build_lstm(seq_len=24, n_features=10, units=32)
    x = np.random.normal(0, 1, (3, 24, 10)).astype(np.float32)
    out = model.forward(x)

    assert "volatility" in out and "direction" in out
    assert out["volatility"].shape == (3, 3)
    assert out["direction"].shape == (3, 3)
    # Volatility head uses Softplus -> must be non-negative
    assert (out["volatility"] >= 0.0).all()
    # Direction head uses Sigmoid -> must be in [0, 1]
    assert (out["direction"] >= 0.0).all() and (out["direction"] <= 1.0).all()


def test_lstm_serialization(tmp_path):
    model = build_lstm(seq_len=12, n_features=8, units=16)
    save_file = tmp_path / "test_lstm"
    model.save(save_file)

    loaded = LSTMModel.load(save_file)
    x = np.random.normal(0, 1, (2, 12, 8)).astype(np.float32)
    orig_out = model.forward(x)
    load_out = loaded.forward(x)

    np.testing.assert_allclose(orig_out["volatility"], load_out["volatility"], rtol=1e-5)
    np.testing.assert_allclose(orig_out["direction"], load_out["direction"], rtol=1e-5)


def test_transformer_forward_and_bounds():
    np.random.seed(42)
    model = build_transformer(seq_len=24, n_features=10, d_model=32, num_heads=4)
    x = np.random.normal(0, 1, (2, 24, 10)).astype(np.float32)
    out = model.forward(x)

    assert out["volatility"].shape == (2, 3)
    assert out["direction"].shape == (2, 3)
    assert (out["volatility"] >= 0.0).all()
    assert (out["direction"] >= 0.0).all() and (out["direction"] <= 1.0).all()


def test_quantitative_metrics():
    y_true = np.array([0.02, 0.03, 0.01, 0.04])
    y_pred = np.array([0.025, 0.028, 0.015, 0.035])

    assert rmse(y_true, y_pred) > 0
    assert mae(y_true, y_pred) > 0
    assert directional_accuracy(np.array([0.01, -0.02]), np.array([0.7, 0.2])) == 1.0

    # Sharpe ratio on positive return stream
    returns = np.array([0.01, 0.02, 0.005, 0.015, 0.01])
    assert sharpe_ratio(returns) > 0

    # Max drawdown on declining equity curve
    equity = np.array([100.0, 120.0, 90.0, 110.0])  # Peak 120 -> trough 90 = -25%
    assert abs(max_drawdown(equity) - (-0.25)) < 1e-4
