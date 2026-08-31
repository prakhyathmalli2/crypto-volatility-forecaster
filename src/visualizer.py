"""
Executive Visualization Module using Plotly
Sleek, minimalist fintech styling with custom dark slate templates and clean typography.
"""

import plotly.graph_objects as go
from plotly.subplots import make_subplots
import pandas as pd
import numpy as np

# Premium Fintech Dark Palette
BG_COLOR = "#0f172a"        # Slate 900
CARD_BG = "#1e293b"         # Slate 800
BORDER_COLOR = "#334155"    # Slate 700
TEXT_MAIN = "#f8fafc"       # Slate 50
TEXT_MUTED = "#94a3b8"      # Slate 400
ACCENT_BLUE = "#38bdf8"     # Sky 400
ACCENT_GREEN = "#10b981"    # Emerald 500
ACCENT_RED = "#ef4444"      # Rose 500
ACCENT_AMBER = "#f59e0b"    # Amber 500
ACCENT_PURPLE = "#a855f7"   # Purple 500


def apply_fintech_theme(fig: go.Figure, height: int = 500) -> go.Figure:
    """Apply consistent, high-end dark fintech layout to any Plotly figure."""
    fig.update_layout(
        paper_bgcolor="rgba(0,0,0,0)",
        plot_bgcolor="rgba(0,0,0,0)",
        font=dict(family="Inter, -apple-system, BlinkMacSystemFont, 'Segoe UI', Roboto, sans-serif", color=TEXT_MAIN, size=12),
        height=height,
        margin=dict(l=20, r=20, t=40, b=20),
        xaxis=dict(
            gridcolor="rgba(255, 255, 255, 0.05)",
            zerolinecolor="rgba(255, 255, 255, 0.08)",
            showline=True,
            linecolor=BORDER_COLOR,
            tickfont=dict(color=TEXT_MUTED, size=11)
        ),
        yaxis=dict(
            gridcolor="rgba(255, 255, 255, 0.05)",
            zerolinecolor="rgba(255, 255, 255, 0.08)",
            showline=True,
            linecolor=BORDER_COLOR,
            tickfont=dict(color=TEXT_MUTED, size=11)
        ),
        hoverlabel=dict(
            bgcolor=CARD_BG,
            bordercolor=BORDER_COLOR,
            font=dict(color=TEXT_MAIN, size=12)
        )
    )
    return fig


def create_candlestick_chart(
    df: pd.DataFrame,
    signals_series: pd.Series = None,
    show_bb: bool = True,
    show_ema: bool = True
) -> go.Figure:
    """
    Sleek interactive candlestick chart with volume and model signal markers.
    """
    fig = make_subplots(
        rows=2, cols=1,
        shared_xaxes=True,
        vertical_spacing=0.03,
        row_heights=[0.78, 0.22]
    )

    # 1. Candlestick
    fig.add_trace(
        go.Candlestick(
            x=df.index,
            open=df['Open'],
            high=df['High'],
            low=df['Low'],
            close=df['Close'],
            name="OHLC",
            increasing=dict(line=dict(color=ACCENT_GREEN, width=1), fillcolor=ACCENT_GREEN),
            decreasing=dict(line=dict(color=ACCENT_RED, width=1), fillcolor=ACCENT_RED)
        ),
        row=1, col=1
    )

    # 2. Bollinger Bands
    if show_bb and 'Close' in df.columns:
        sma20 = df['Close'].rolling(20).mean()
        std20 = df['Close'].rolling(20).std()
        bb_upper = sma20 + (2 * std20)
        bb_lower = sma20 - (2 * std20)

        fig.add_trace(
            go.Scatter(x=df.index, y=bb_upper, line=dict(color='rgba(148, 163, 184, 0.35)', width=1, dash='dot'), name="Upper Band"),
            row=1, col=1
        )
        fig.add_trace(
            go.Scatter(x=df.index, y=bb_lower, line=dict(color='rgba(148, 163, 184, 0.35)', width=1, dash='dot'), fill='tonexty', fillcolor='rgba(148, 163, 184, 0.04)', name="Lower Band"),
            row=1, col=1
        )

    # 3. EMAs
    if show_ema and 'Close' in df.columns:
        ema9 = df['Close'].ewm(span=9, adjust=False).mean()
        ema21 = df['Close'].ewm(span=21, adjust=False).mean()
        fig.add_trace(go.Scatter(x=df.index, y=ema9, line=dict(color=ACCENT_BLUE, width=1.5), name="EMA 9"), row=1, col=1)
        fig.add_trace(go.Scatter(x=df.index, y=ema21, line=dict(color=ACCENT_AMBER, width=1.5), name="EMA 21"), row=1, col=1)

    # 4. Trading Signals Markers
    if signals_series is not None:
        aligned_signals = signals_series.reindex(df.index).fillna(0)
        buy_idx = df.index[aligned_signals == 1]
        sell_idx = df.index[aligned_signals == 0]

        if len(buy_idx) > 0:
            fig.add_trace(
                go.Scatter(
                    x=buy_idx,
                    y=df.loc[buy_idx, 'Low'] * 0.985,
                    mode='markers',
                    marker=dict(symbol='triangle-up', size=10, color=ACCENT_GREEN, line=dict(width=1, color="#ffffff")),
                    name='Buy Signal'
                ),
                row=1, col=1
            )

        if len(sell_idx) > 0:
            fig.add_trace(
                go.Scatter(
                    x=sell_idx,
                    y=df.loc[sell_idx, 'High'] * 1.015,
                    mode='markers',
                    marker=dict(symbol='triangle-down', size=10, color=ACCENT_RED, line=dict(width=1, color="#ffffff")),
                    name='Sell Signal'
                ),
                row=1, col=1
            )

    # 5. Volume Bar Chart
    colors = [ACCENT_GREEN if c >= o else ACCENT_RED for c, o in zip(df['Close'], df['Open'])]
    fig.add_trace(
        go.Bar(x=df.index, y=df['Volume'], marker_color=colors, opacity=0.7, name="Volume"),
        row=2, col=1
    )

    # Range Selector
    fig.update_xaxes(
        rangeselector=dict(
            buttons=list([
                dict(count=1, label="1M", step="month", stepmode="backward"),
                dict(count=3, label="3M", step="month", stepmode="backward"),
                dict(count=6, label="6M", step="month", stepmode="backward"),
                dict(count=1, label="1Y", step="year", stepmode="backward"),
                dict(step="all", label="ALL")
            ]),
            bgcolor=CARD_BG,
            activecolor=ACCENT_BLUE,
            font=dict(color=TEXT_MAIN, size=10)
        ),
        xaxis_rangeslider_visible=False,
        row=1, col=1
    )

    fig.update_layout(
        legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1, font=dict(size=11)),
        height=580
    )
    return apply_fintech_theme(fig, height=580)


def create_volatility_gauge_chart(current_vol: float, p33: float, p66: float) -> go.Figure:
    """
    Minimalist semi-circular gauge for current market volatility regime.
    """
    max_val = max(p66 * 1.8, current_vol * 1.3, 10.0)

    fig = go.Figure(go.Indicator(
        mode="gauge+number",
        value=current_vol,
        number={'suffix': "%", 'font': {'size': 32, 'color': TEXT_MAIN, 'family': "Inter"}},
        domain={'x': [0, 1], 'y': [0, 1]},
        gauge={
            'axis': {'range': [0, max_val], 'tickwidth': 1, 'tickcolor': TEXT_MUTED, 'tickfont': {'size': 10}},
            'bar': {'color': ACCENT_BLUE, 'thickness': 0.3},
            'bgcolor': CARD_BG,
            'borderwidth': 1,
            'bordercolor': BORDER_COLOR,
            'steps': [
                {'range': [0, p33], 'color': 'rgba(16, 185, 129, 0.25)'},    # Low (Green)
                {'range': [p33, p66], 'color': 'rgba(245, 158, 11, 0.25)'},   # Med (Amber)
                {'range': [p66, max_val], 'color': 'rgba(239, 68, 68, 0.25)'} # High (Red)
            ],
            'threshold': {
                'line': {'color': "#ffffff", 'width': 3},
                'thickness': 0.8,
                'value': current_vol
            }
        }
    ))

    fig.update_layout(
        paper_bgcolor="rgba(0,0,0,0)",
        plot_bgcolor="rgba(0,0,0,0)",
        height=240,
        margin=dict(l=20, r=20, t=30, b=10)
    )
    return fig


def create_volatility_chart(
    df: pd.DataFrame,
    garch_vol_series: pd.Series
) -> go.Figure:
    """
    Clean dual-pane chart for daily log returns and GARCH conditional volatility.
    """
    fig = make_subplots(
        rows=2, cols=1,
        shared_xaxes=True,
        vertical_spacing=0.06,
        subplot_titles=("Log Returns (%)", "GARCH(1,1) Volatility vs. 20D Realized Volatility (%)"),
        row_heights=[0.45, 0.55]
    )

    # 1. Log Returns
    fig.add_trace(
        go.Scatter(
            x=df.index,
            y=df['Log_Return_Pct'],
            line=dict(color=ACCENT_BLUE, width=1.2),
            name="Log Return %"
        ),
        row=1, col=1
    )

    # 2. Realized Vol
    realized_20d = df['Log_Return_Pct'].rolling(20).std()
    fig.add_trace(
        go.Scatter(
            x=df.index,
            y=realized_20d,
            line=dict(color='rgba(148, 163, 184, 0.5)', width=1.2, dash='dot'),
            name="20D Realized Vol"
        ),
        row=2, col=1
    )

    # 3. GARCH Vol
    aligned_garch = garch_vol_series.reindex(df.index)
    fig.add_trace(
        go.Scatter(
            x=df.index,
            y=aligned_garch,
            line=dict(color=ACCENT_RED, width=1.8),
            name="GARCH Conditional Vol"
        ),
        row=2, col=1
    )

    fig.update_layout(
        legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1, font=dict(size=11)),
        height=500
    )
    return apply_fintech_theme(fig, height=500)


def create_volatility_forecast_chart(forecast_df: pd.DataFrame, last_vol: float) -> go.Figure:
    """
    Forward volatility projection fan chart.
    """
    fig = go.Figure()

    horizons = ["Current"] + forecast_df['Horizon'].tolist()
    daily_vols = [last_vol] + forecast_df['Daily_Vol_Forecast_Pct'].tolist()

    fig.add_trace(
        go.Scatter(
            x=horizons,
            y=daily_vols,
            mode='lines+markers+text',
            text=[f"{v:.2f}%" for v in daily_vols],
            textposition="top center",
            textfont=dict(color=TEXT_MAIN, size=11),
            line=dict(color=ACCENT_AMBER, width=2.5),
            marker=dict(size=7, color="#ffffff", line=dict(color=ACCENT_AMBER, width=2)),
            name="Forecast Vol (%)"
        )
    )

    fig.update_layout(
        title=dict(text="Forward Multi-Step Volatility Projection", font=dict(size=14, color=TEXT_MAIN)),
        height=300,
        yaxis=dict(title="Daily Vol (%)")
    )
    return apply_fintech_theme(fig, height=300)


def create_confusion_matrix_chart(cm_data: list) -> go.Figure:
    """
    Annotated confusion matrix heatmap.
    """
    z = cm_data
    x = ['Predicted Down/Cash', 'Predicted Up/Buy']
    y = ['Actual Down/Cash', 'Actual Up/Buy']
    z_text = [[str(val) for val in row] for row in z]

    fig = go.Figure(data=go.Heatmap(
        z=z,
        x=x,
        y=y,
        text=z_text,
        texttemplate="%{text}",
        textfont={"size": 16, "color": "white", "family": "Inter"},
        colorscale=[[0, "#1e293b"], [1, "#0284c7"]],
        showscale=False
    ))

    fig.update_layout(
        title=dict(text="Out-of-Sample Confusion Matrix", font=dict(size=14, color=TEXT_MAIN)),
        height=280
    )
    return apply_fintech_theme(fig, height=280)


def create_backtest_chart(equity_df: pd.DataFrame) -> go.Figure:
    """
    Cumulative strategy portfolio value vs benchmark and drawdown chart.
    """
    fig = make_subplots(
        rows=2, cols=1,
        shared_xaxes=True,
        vertical_spacing=0.04,
        subplot_titles=("Cumulative Portfolio Valuation ($)", "Portfolio Drawdown (%)"),
        row_heights=[0.72, 0.28]
    )

    # 1. Equity Curves
    fig.add_trace(
        go.Scatter(
            x=equity_df.index,
            y=equity_df['Strategy_Equity'],
            line=dict(color=ACCENT_GREEN, width=2.2),
            name="ML + GARCH Strategy"
        ),
        row=1, col=1
    )

    fig.add_trace(
        go.Scatter(
            x=equity_df.index,
            y=equity_df['Benchmark_Equity'],
            line=dict(color=TEXT_MUTED, width=1.4, dash='dash'),
            name="Buy & Hold Benchmark"
        ),
        row=1, col=1
    )

    # 2. Drawdown
    fig.add_trace(
        go.Scatter(
            x=equity_df.index,
            y=equity_df['Drawdown'] * 100.0,
            line=dict(color=ACCENT_RED, width=1),
            fill='tozeroy',
            fillcolor='rgba(239, 68, 68, 0.18)',
            name="Drawdown %"
        ),
        row=2, col=1
    )

    fig.update_layout(
        legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1, font=dict(size=11)),
        height=540
    )
    return apply_fintech_theme(fig, height=540)


def create_feature_importance_chart(imp_df: pd.DataFrame, top_n: int = 8) -> go.Figure:
    """
    Feature importance horizontal bar chart.
    """
    top_df = imp_df.head(top_n).sort_values(by="Importance", ascending=True)

    fig = go.Figure(
        go.Bar(
            x=top_df['Importance_Pct'],
            y=top_df['Feature'],
            orientation='h',
            marker=dict(
                color=top_df['Importance_Pct'],
                colorscale=[[0, "#38bdf8"], [1, "#a855f7"]],
                showscale=False
            ),
            text=[f"{v:.1f}%" for v in top_df['Importance_Pct']],
            textposition='outside',
            textfont=dict(color=TEXT_MAIN, size=11)
        )
    )

    fig.update_layout(
        title=dict(text=f"Top {top_n} Predictive Feature Weights (%)", font=dict(size=14, color=TEXT_MAIN)),
        height=320,
        xaxis=dict(title="Importance (%)"),
        yaxis=dict(title="")
    )
    return apply_fintech_theme(fig, height=320)
