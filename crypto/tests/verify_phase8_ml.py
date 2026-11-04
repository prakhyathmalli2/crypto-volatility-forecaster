"""Phase 8 real ML verification script.
Tests model artifact loading, input/output tensor shapes, actual execution path, inference latency, and quantitative evaluation metrics.
"""
import json
import sys
import time
from pathlib import Path

# Add crypto root to sys.path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
sys.stdout.reconfigure(encoding="utf-8")

import joblib
import numpy as np
import torch

from config import ARTIFACTS_DIR
from models.garch_model import load_garch_artifact, forecast_volatility
from models.lstm_model import LSTMModel
from models.transformer_model import TransformerModel
from serving.live_stream import LiveStreamBuffer

def run_ml_audit():
    print("=================================================================")
    print("PHASE 8: REAL ML EXECUTION & QUANT AUDIT")
    print("=================================================================")

    # 1. Artifacts existence & loading
    garch_path = ARTIFACTS_DIR / "garch" / "garch_model.pkl"
    lstm_path = ARTIFACTS_DIR / "lstm" / "lstm_model"
    trans_path = ARTIFACTS_DIR / "transformer" / "transformer_model"
    scaler_path = ARTIFACTS_DIR / "scaler.pkl"
    meta_path = ARTIFACTS_DIR / "model_metadata.json"

    print("1. Checking saved artifacts:")
    print(f"   GARCH artifact: {garch_path.exists()} ({garch_path.stat().st_size if garch_path.exists() else 0} bytes)")
    print(f"   LSTM artifact: {lstm_path.with_suffix('.npz').exists()} ({lstm_path.with_suffix('.npz').stat().st_size if lstm_path.with_suffix('.npz').exists() else 0} bytes)")
    print(f"   Transformer artifact: {trans_path.with_suffix('.npz').exists()} ({trans_path.with_suffix('.npz').stat().st_size if trans_path.with_suffix('.npz').exists() else 0} bytes)")
    print(f"   Scaler artifact: {scaler_path.exists()} ({scaler_path.stat().st_size if scaler_path.exists() else 0} bytes)")
    print(f"   Metadata artifact: {meta_path.exists()} ({meta_path.stat().st_size if meta_path.exists() else 0} bytes)")

    # Load artifacts
    t0 = time.perf_counter()
    garch_model = load_garch_artifact(garch_path)
    lstm_model = LSTMModel.load(lstm_path)
    trans_model = TransformerModel.load(trans_path)
    scaler = joblib.load(scaler_path)
    metadata = json.loads(meta_path.read_text(encoding="utf-8")) if meta_path.exists() else {}
    load_time_ms = (time.perf_counter() - t0) * 1000.0
    print(f"\n2. All models loaded in {load_time_ms:.2f}ms")

    # 2. Check metadata specifications
    print(f"\n3. Model Specifications from metadata:")
    print(f"   Version: {metadata.get('model_version')}")
    print(f"   Sequence Length: {metadata.get('sequence_length')} periods")
    print(f"   Features ({metadata.get('n_features')}): {', '.join(metadata.get('features', [])[:6])} ...")
    print(f"   Horizons: {metadata.get('horizons')}")
    print(f"   Ensemble weights: {metadata.get('ensemble_weights')}")
    print(f"   Conformal 90% Quantiles: {metadata.get('conformal_uncertainty_quantiles_90pct')}")

    # 3. Test Live Inference Pipeline
    buffer = LiveStreamBuffer()
    print("\n4. Testing Live Buffer prepare_inference_tensor for BTC/USDT:")
    t0 = time.perf_counter()
    tensor, tail_df = buffer.prepare_inference_tensor(symbol="BTC/USDT", timeframe="1h")
    prep_time_ms = (time.perf_counter() - t0) * 1000.0
    print(f"   Tensor prepared in {prep_time_ms:.2f}ms")
    print(f"   Input Tensor Shape: {tensor.shape} (Expected: (1, 48, 28))")
    print(f"   Tensor dtype: {tensor.dtype}, Min: {tensor.min():.3f}, Max: {tensor.max():.3f}")

    # 4. Measure Model Forward Passes
    t0 = time.perf_counter()
    lstm_out = lstm_model.predict(tensor)
    lstm_lat_ms = (time.perf_counter() - t0) * 1000.0

    t0 = time.perf_counter()
    trans_out = trans_model.predict(tensor)
    trans_lat_ms = (time.perf_counter() - t0) * 1000.0

    t0 = time.perf_counter()
    garch_fc = forecast_volatility(garch_model, horizons=(1, 6, 24))
    garch_lat_ms = (time.perf_counter() - t0) * 1000.0

    print(f"\n5. Inference Latency & Outputs:")
    print(f"   Bi-LSTM:       {lstm_lat_ms:6.2f}ms | Volatility: {np.round(lstm_out['volatility'][0], 5)} | Direction: {np.round(lstm_out['direction'][0], 4)}")
    print(f"   Transformer:   {trans_lat_ms:6.2f}ms | Volatility: {np.round(trans_out['volatility'][0], 5)} | Direction: {np.round(trans_out['direction'][0], 4)}")
    print(f"   GARCH(1,1):    {garch_lat_ms:6.2f}ms | Forecasts:  {np.round(list(garch_fc.values()), 5)}")

    # 5. Ensemble & Conformal Bounds
    l_vol = lstm_out["volatility"][0]
    t_vol = trans_out["volatility"][0]
    g_vol = np.array([garch_fc[h] for h in (1, 6, 24)], dtype=float)
    ens_vol = (0.35 * g_vol) + (0.35 * l_vol) + (0.30 * t_vol)
    
    l_dir = lstm_out["direction"][0]
    t_dir = trans_out["direction"][0]
    ens_dir = (0.50 * l_dir) + (0.50 * t_dir)

    print(f"\n6. Weighted Multi-Model Ensemble Forecast:")
    print(f"   1h Horizon:  Vol = {ens_vol[0]:.5f} | Directional Prob = {ens_dir[0]*100:.1f}%")
    print(f"   6h Horizon:  Vol = {ens_vol[1]:.5f} | Directional Prob = {ens_dir[1]*100:.1f}%")
    print(f"   24h Horizon: Vol = {ens_vol[2]:.5f} | Directional Prob = {ens_dir[2]*100:.1f}%")

    quantiles = metadata.get("conformal_uncertainty_quantiles_90pct", {"1h": 0.0179, "6h": 0.0093, "24h": 0.0242})
    print(f"\n7. Conformal Calibration Intervals (90% Confidence):")
    for i, h in enumerate((1, 6, 24)):
        q = float(quantiles.get(f"{h}h", 0.01))
        v = float(ens_vol[i])
        print(f"   {h}h Interval: [{max(v - q, 0.0):.5f}, {v + q:.5f}] (Center: {v:.5f}, Half-width: {q:.5f})")

    # 6. Check Historical Test Predictions Archive
    test_npz = ARTIFACTS_DIR / "test_predictions.npz"
    if test_npz.exists():
        data = np.load(test_npz)
        print(f"\n8. Historical Test Set Verification:")
        print(f"   Keys in test_predictions.npz: {list(data.keys())}")
        if "y_true_vol" in data and "y_pred_vol" in data:
            y_true = data["y_true_vol"]
            y_pred = data["y_pred_vol"]
            mae = np.mean(np.abs(y_true - y_pred))
            rmse = np.sqrt(np.mean((y_true - y_pred) ** 2))
            print(f"   Measured Test Set Volatility MAE:  {mae:.5f}")
            print(f"   Measured Test Set Volatility RMSE: {rmse:.5f}")

if __name__ == "__main__":
    run_ml_audit()
