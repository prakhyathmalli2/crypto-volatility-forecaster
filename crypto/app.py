"""Streamlit Interactive Financial Dashboard & Research Terminal.

Synchronized with the v1 REST API:
- Multi-Asset Switcher (BTC/USDT, ETH/USDT, SOL/USDT)
- Multi-Horizon (1H, 6H, 24H) Volatility Forecasts with Conformal Uncertainty Intervals
- Directional Return Probability Gauge & 5-Factor Model Agreement Score
- Explainable ML Signal Intelligence (STRONG BUY, BUY, HOLD, SELL, STRONG SELL, REDUCE)
- Macro Events Calendar & Event-Risk Filter Warnings
- Prioritized Live Crypto News Feed
- Paper Trading Order Desk with Realized PnL and Execution Ledger
- Educational Trading Principles from Legendary Market Practitioners
- Link to High-Frequency TradingView Web Terminal
"""
from __future__ import annotations

import json
import os
import time
from pathlib import Path

import numpy as np
import pandas as pd
import requests
import streamlit as st

try:
    import plotly.graph_objects as go
    from plotly.subplots import make_subplots
    HAS_PLOTLY = True
except ImportError:
    HAS_PLOTLY = False

API_URL = os.getenv("API_URL", "http://127.0.0.1:8000")

st.set_page_config(
    page_title="AlphaVol Pro • Crypto Trading Intelligence",
    page_icon="⚡",
    layout="wide",
    initial_sidebar_state="expanded",
)

# Custom Financial Terminal Dark CSS
st.markdown(
    """
<style>
    .block-container { padding-top: 1.0rem; max-width: 1550px; }
    [data-testid="stMetricValue"] { font-size: 1.6rem; font-weight: 700; font-family: monospace; }
    .card {
        background-color: rgba(255, 255, 255, 0.04);
        border: 1px solid rgba(255, 255, 255, 0.08);
        border-radius: 8px;
        padding: 14px 18px;
        margin-bottom: 14px;
    }
    .badge-strong-buy { color: #000; background: #0ecb81; font-weight: 800; padding: 3px 8px; border-radius: 4px; }
    .badge-buy { color: #0ecb81; border: 1px solid #0ecb81; font-weight: 700; padding: 3px 8px; border-radius: 4px; }
    .badge-sell { color: #f6465d; border: 1px solid #f6465d; font-weight: 700; padding: 3px 8px; border-radius: 4px; }
    .badge-strong-sell { color: #fff; background: #f6465d; font-weight: 800; padding: 3px 8px; border-radius: 4px; }
    .badge-reduce { color: #000; background: #ff9900; font-weight: 800; padding: 3px 8px; border-radius: 4px; }
    .badge-hold { color: #00e5ff; border: 1px solid #00e5ff; font-weight: 700; padding: 3px 8px; border-radius: 4px; }
</style>
""",
    unsafe_allow_html=True,
)


def api_get(path: str, params: dict | None = None) -> dict:
    r = requests.get(f"{API_URL}{path}", params=params, timeout=25)
    r.raise_for_status()
    return r.json()


def api_post(path: str, payload: dict) -> dict:
    r = requests.post(f"{API_URL}{path}", json=payload, timeout=25)
    r.raise_for_status()
    return r.json()


# Sidebar Controls
with st.sidebar:
    st.markdown("### ◈ ALPHA**VOL** PRO")
    st.caption("Quantitative Crypto Trading Intelligence")

    symbol = st.selectbox("Asset Pair", ["BTC/USDT", "ETH/USDT", "SOL/USDT"], index=0)
    timeframe = st.selectbox("Timeframe", ["1m", "5m", "15m", "30m", "1h", "4h", "1d"], index=4)

    st.divider()
    st.markdown("#### 📡 System Telemetry")
    try:
        sys_status = api_get("/api/v1/system/status")
        cluster_status = sys_status.get("status", "HEALTHY")
        st.success(f"Cluster: {cluster_status}")
        exchange_name = sys_status.get("services", {}).get("market_api", {}).get("exchange", "EXCHANGE")
        st.caption(f"Exchange: {str(exchange_name).upper()}")
        ml_stat = sys_status.get("services", {}).get("ml_models", {}).get("status", "READY")
        st.caption(f"ML Models: {ml_stat}")
    except Exception:
        st.warning("FastAPI Offline. Launch backend via: python run.py api")

    st.divider()
    st.markdown("#### ⚡ Web Terminal")
    st.link_button("Open TradingView Terminal ↗", f"{API_URL}/", use_container_width=True)

    st.divider()
    auto_refresh = st.checkbox("Live Refresh", value=False)
    refresh_rate = st.slider("Interval (s)", 10, 120, 25)
    if st.button("🔄 Refresh Telemetry"):
        st.cache_data.clear()
        st.rerun()

st.title(f"📈 {symbol} Trading Intelligence Terminal")

# Fetch live market data, indicators, and signals
try:
    ticker = api_get(f"/api/v1/market/{symbol}")
    signal_data = api_get(f"/api/v1/signal/{symbol}", {"timeframe": timeframe})
    forecast_data = api_get(f"/api/v1/forecast/{symbol}", {"timeframe": timeframe})
    candles_data = api_get(f"/api/v1/market/{symbol}/candles", {"timeframe": timeframe, "limit": 100})
except Exception as exc:
    st.error(f"Failed to connect to API at {API_URL}: {exc}")
    st.info("Tip: Start the backend server with: python run.py api")
    st.stop()

# Header Metrics Row
cur_price = ticker["price"]
chg = ticker["change_24h_pct"]
sig_action = signal_data["signal"]
confidence = signal_data["confidence"]
agreement = signal_data["model_agreement_fraction"]

col1, col2, col3, col4, col5 = st.columns(5)
col1.metric("Spot Price", f"${cur_price:,.2f}", f"{chg:+.2f}%")
col2.metric("ML Signal", sig_action.replace("_", " "), f"Conf: {confidence*100:.0f}%")
col3.metric("Model Concord", agreement, f"{signal_data['model_agreement']*100:.0f}% Agreement")
col4.metric("1H Forecast Vol", f"{signal_data['volatility_forecast']*100:.2f}%", f"Regime: {signal_data['volatility_regime']}")
col5.metric("Event Risk", "HIGH" if signal_data.get("event_risk_high") else "NORMAL", f"Risk: {signal_data['risk_level']}")

# Interactive Chart with Indicators
if HAS_PLOTLY and candles_data:
    cdf = pd.DataFrame(candles_data)
    cdf["timestamp"] = pd.to_datetime(cdf["timestamp"])

    fig = make_subplots(
        rows=2,
        cols=1,
        shared_xaxes=True,
        vertical_spacing=0.04,
        row_heights=[0.75, 0.25],
        subplot_titles=(f"{symbol} Price & ML Signal Levels ({timeframe})", "Volume Histogram"),
    )

    # Candlestick trace
    fig.add_trace(
        go.Candlestick(
            x=cdf["timestamp"],
            open=cdf["open"],
            high=cdf["high"],
            low=cdf["low"],
            close=cdf["close"],
            name="Price",
        ),
        row=1,
        col=1,
    )

    # Volume trace
    colors = ["#0ecb81" if row["close"] >= row["open"] else "#f6465d" for _, row in cdf.iterrows()]
    fig.add_trace(
        go.Bar(
            x=cdf["timestamp"],
            y=cdf["volume"],
            name="Volume",
            marker_color=colors,
        ),
        row=2,
        col=1,
    )

    # Target & Stop Loss Lines on chart
    if sig_action in ("STRONG_BUY", "BUY"):
        fig.add_hline(y=signal_data["stop_loss"], line_dash="dash", line_color="#f6465d", annotation_text="Stop Loss", row=1, col=1)
        fig.add_hline(y=signal_data["take_profit"], line_dash="dash", line_color="#0ecb81", annotation_text="Take Profit (2R)", row=1, col=1)

    fig.update_layout(
        height=520,
        margin=dict(l=10, r=10, t=30, b=10),
        xaxis_rangeslider_visible=False,
        template="plotly_dark",
    )
    st.plotly_chart(fig, use_container_width=True)

# Tabs Drawer
tab_signal, tab_news, tab_events, tab_paper, tab_ai, tab_principles = st.tabs([
    "🎯 Signal Analysis", "📰 Prioritized News", "📅 Macro Events", "💼 Paper Trading", "🤖 AI Research Assistant", "📚 Principles"
])

with tab_signal:
    c_left, c_right = st.columns([1.1, 1.0])
    with c_left:
        st.markdown("#### 💡 Decision Strategy Rationale")
        reasons_md = "".join(f"- {r}\n" for r in signal_data.get("reasons", []))
        st.markdown(
            f"""
        <div class="card">
            <h5>Strategic Action: <b>{sig_action.replace('_', ' ')}</b></h5>
            <p><b>Market Regime:</b> <code>{signal_data['market_regime']}</code> | <b>Risk Assessment:</b> <code>{signal_data['risk_level']}</code></p>
            <p><b>Reasons:</b></p>
            {reasons_md}
        </div>
        """,
            unsafe_allow_html=True,
        )

    with c_right:
        st.markdown("#### 📐 Risk Management Trade Plan")
        st.markdown(
            f"""
        <div class="card">
            <p><b>Entry Reference:</b> ${signal_data['entry']:,.2f}</p>
            <p style="color:#f6465d"><b>Dynamic ATR Stop-Loss:</b> ${signal_data['stop_loss']:,.2f}</p>
            <p style="color:#0ecb81"><b>Take-Profit Target (2R):</b> ${signal_data['take_profit']:,.2f}</p>
            <p><b>Reward-to-Risk Ratio:</b> 1:{signal_data['risk_reward']:.1f}</p>
            <p><small>Enforcing maximum 2.0% capital risk rule with ATR volatility buffer.</small></p>
        </div>
        """,
            unsafe_allow_html=True,
        )

with tab_news:
    try:
        news_items = api_get("/api/v1/news", {"symbol": symbol, "limit": 10})
        for n in news_items:
            badge = "🔴 BREAKING" if n["importance"] == "BREAKING" else ("🟠 HIGH IMPACT" if n["importance"] == "HIGH IMPACT" else "⚪ NEWS")
            with st.expander(f"{badge} • {n['title']} ({n['source']})"):
                st.write(n["summary"])
                st.caption(f"Published: {n['published_at']} | Sentiment Polarity: {n['sentiment']:+.2f} | Tags: {', '.join(n['categories'])}")
                if n.get("url"):
                    st.link_button("Read Source Article ↗", n["url"])
    except Exception:
        st.caption("Unable to stream news headlines.")

with tab_events:
    try:
        events_items = api_get("/api/v1/events", {"symbol": symbol})
        st.dataframe(pd.DataFrame(events_items)[["importance", "event", "hours_until", "expected", "previous", "source"]], hide_index=True, use_container_width=True)
    except Exception:
        st.caption("Macro events calendar unavailable.")

with tab_paper:
    p_left, p_right = st.columns([1, 1])
    with p_left:
        st.markdown("##### Portfolio State")
        try:
            p_data = api_get("/api/v1/paper/account", {"symbol": symbol})["portfolio"]
            m1, m2, m3 = st.columns(3)
            m1.metric("Total Equity", f"${p_data['total_equity']:,.2f}", f"{p_data['total_return_pct']:+.2f}%")
            m2.metric("Available Cash", f"${p_data['cash']:,.2f}")
            m3.metric("Holdings", f"{p_data['crypto_position']:.4f}")
        except Exception:
            st.caption("Paper ledger offline.")

    with p_right:
        st.markdown("##### Place Simulated Order")
        with st.form("paper_form"):
            o_side = st.selectbox("Side", ["BUY", "SELL", "REDUCE"])
            o_qty = st.number_input("Quantity", min_value=0.001, max_value=10.0, value=0.01, step=0.005)
            if st.form_submit_button("⚡ Execute Simulated Order"):
                try:
                    res = api_post("/api/v1/paper/order", {"side": o_side, "quantity": o_qty, "price": cur_price, "symbol": symbol})
                    st.success(f"Executed {o_side} {o_qty} at ${cur_price:,.2f}!")
                    st.rerun()
                except Exception as exc:
                    st.error(f"Order failed: {exc}")

with tab_ai:
    st.markdown("#### 🤖 Quantitative AI Research Assistant")
    user_q = st.text_input("Ask about market movement, model signals, or risk:")
    if st.button("Query Assistant") and user_q:
        with st.spinner("Analyzing quantitative telemetry..."):
            try:
                ai_res = api_post("/api/v1/chat", {"query": user_q, "symbol": symbol})
                st.markdown(ai_res["response"])
                st.caption(f"Provider: {ai_res['provider']} • Tools executed: {', '.join(ai_res['tools_called'])}")
            except Exception as exc:
                st.error(f"Assistant query failed: {exc}")

with tab_principles:
    try:
        principle = api_get("/api/v1/principles", {"volatility_regime": signal_data["volatility_regime"], "signal_action": sig_action})
        st.markdown(
            f"""
        <div class="card">
            <h4>💡 {principle['concept']}</h4>
            <p style="font-size:1.15rem;font-style:italic">"{principle['rule_summary']}"</p>
            <p>{principle['detailed_explanation']}</p>
            <p><small style="color:#94a3b8">— {principle['source_attribution']}</small></p>
        </div>
        """,
            unsafe_allow_html=True,
        )
    except Exception:
        st.caption("Principles database unavailable.")

if auto_refresh:
    time.sleep(refresh_rate)
    st.rerun()
