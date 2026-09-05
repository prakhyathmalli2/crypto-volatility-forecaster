"""Streamlit UI: live BTC chart, forecast dashboard, signal explanation and paper trading."""
from __future__ import annotations

import os
import time

import pandas as pd
import plotly.graph_objects as go
import requests
import streamlit as st

API_URL = os.getenv("API_URL", "http://localhost:8000")

st.set_page_config(page_title="BTC Volatility Intelligence", page_icon="₿", layout="wide")

st.markdown("""
<style>
.block-container {padding-top: 1.2rem; max-width: 1450px;}
[data-testid="stMetricValue"] {font-size: 1.65rem;}
.signal-card {padding: 14px 18px; border-radius: 14px; border: 1px solid rgba(128,128,128,.25);}
.small {font-size: .85rem; opacity: .75;}
</style>
""", unsafe_allow_html=True)


def api_get(path, params=None):
    r = requests.get(f"{API_URL}{path}", params=params, timeout=25)
    r.raise_for_status()
    return r.json()


st.title("₿ Crypto Volatility Forecast & Smart Signal Generator")
st.caption("BTC/USDT • 1H ensemble forecast with 1H/4H/1D live charts • GARCH + LSTM + Transformer • paper trading only")

with st.sidebar:
    st.header("Controls")
    timeframe = st.selectbox("Chart timeframe", ["1h", "4h", "1d"], index=0)
    chart_days = st.slider("Chart history (days)", 2, 90, 7)
    auto_refresh = st.checkbox("Auto-refresh live view", value=True)
    refresh_seconds = st.slider("Refresh seconds", 10, 120, 30)
    if st.button("Refresh now"):
        st.cache_data.clear()
        st.rerun()
    st.divider()
    health = api_get("/health")
    st.write(f"API: **{health.get('status', 'unknown')}**")
    st.write(f"Models: **{'ready' if health.get('models_ready') else 'not trained'}**")

try:
    market = api_get("/market", {"timeframe": timeframe, "days": chart_days})
    mdf = pd.DataFrame(market)
    mdf["open_time"] = pd.to_datetime(mdf["open_time"], utc=True)
    pred = api_get("/predict/latest", {"timeframe": timeframe})
except Exception as exc:
    st.error(f"Backend unavailable: {exc}")
    st.stop()

last = mdf.iloc[-1]
p1 = pred["predictions"]["1h"]
p6 = pred["predictions"]["6h"]
p24 = pred["predictions"]["24h"]
signal = pred["signal"]

c1, c2, c3, c4, c5 = st.columns(5)
c1.metric("BTC Price", f"${pred['current_price']:,.2f}")
c2.metric("Signal", signal["action"])
c3.metric("6H Up Probability", f"{p6['up_probability']*100:.1f}%")
c4.metric("6H Forecast Vol", f"{p6['volatility']*100:.2f}%")
c5.metric("Risk", signal["risk_level"])

fig = go.Figure(data=[go.Candlestick(x=mdf["open_time"], open=mdf["open"], high=mdf["high"], low=mdf["low"], close=mdf["close"])])
fig.update_layout(height=500, margin=dict(l=10, r=10, t=25, b=10), xaxis_rangeslider_visible=False, title=f"BTC/USDT — {timeframe}")
st.plotly_chart(fig, use_container_width=True)

left, right = st.columns([1.05, 1])
with left:
    st.subheader("Forecast horizon")
    forecast_df = pd.DataFrame([
        ["1H", p1["volatility"], p1["up_probability"]],
        ["6H", p6["volatility"], p6["up_probability"]],
        ["24H", p24["volatility"], p24["up_probability"]],
    ], columns=["Horizon", "Forecast volatility", "Up probability"])
    forecast_df["Forecast volatility"] = (forecast_df["Forecast volatility"] * 100).round(3).astype(str) + "%"
    forecast_df["Up probability"] = (forecast_df["Up probability"] * 100).round(1).astype(str) + "%"
    st.dataframe(forecast_df, hide_index=True, use_container_width=True)
    st.info(signal["reason"] + f" Confidence: {signal['confidence']*100:.0f}%. Forecast model cadence: 1H.")

with right:
    st.subheader("Paper trading")
    status = api_get("/paper/status", {"price": pred["current_price"]})
    s1, s2, s3 = st.columns(3)
    s1.metric("Equity", f"${status['equity']:,.2f}")
    s2.metric("Cash", f"${status['cash']:,.2f}")
    s3.metric("BTC Position", f"{status['position']:.6f}")
    st.caption("Orders below are simulated and are never sent to an exchange.")
    with st.form("trade_form"):
        side = st.selectbox("Action", ["BUY", "SELL", "REDUCE"])
        qty = st.number_input("BTC quantity", min_value=0.000001, value=0.001, format="%.6f")
        submitted = st.form_submit_button("Place paper order")
        if submitted:
            try:
                out = requests.post(f"{API_URL}/paper/trade", json={"side": side, "quantity": qty, "price": pred["current_price"]}, timeout=10)
                out.raise_for_status()
                st.success("Paper order recorded.")
            except Exception as exc:
                st.error(str(exc))

st.subheader("Why this signal?")
st.write(f"**{signal['action']}** is based on the ensemble's 6-hour direction probability, forecast volatility relative to the latest 24-hour historical volatility, and explicit risk thresholds. Volatility is treated as risk — not direction.")

if os.path.exists(os.path.join("data", "processed", "evaluation_report.json")):
    with st.expander("Model evaluation"):
        import json
        report = json.load(open(os.path.join("data", "processed", "evaluation_report.json"), encoding="utf-8"))
        st.json(report)

if auto_refresh:
    time.sleep(refresh_seconds)
    st.rerun()
