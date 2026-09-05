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
