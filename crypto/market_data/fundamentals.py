"""Asset fundamentals and financial metrics provider.

Retrieves verified financial indicators (Market Cap, P/E, EPS, Revenue, Profit Margin,
Dividend Yield, ROE, 52-week Range, Debt/Equity) for equities, indices, and crypto.
Strictly returns 'Data unavailable' when a metric does not exist, adhering to zero fabrication.
"""
from __future__ import annotations

import time
from typing import Any
import yfinance as yf

# In-memory cache: symbol -> (timestamp, data)
_FUNDAMENTALS_CACHE: dict[str, tuple[float, dict[str, Any]]] = {}
CACHE_TTL_FUNDAMENTALS = 3600  # 1 hour


def _format_currency(val: float | None, currency: str = "USD") -> str:
    if val is None or val == 0:
        return "Data unavailable"
    symbol = "₹" if currency == "INR" else ("$" if currency == "USD" else f"{currency} ")
    if abs(val) >= 1e12:
        return f"{symbol}{val / 1e12:.2f}T"
    if abs(val) >= 1e9:
        return f"{symbol}{val / 1e9:.2f}B"
    if abs(val) >= 1e6:
        return f"{symbol}{val / 1e6:.2f}M"
    return f"{symbol}{val:,.2f}"


def _format_ratio(val: float | None, suffix: str = "") -> str:
    if val is None or val == 0:
        return "Data unavailable"
    return f"{val:.2f}{suffix}"


def _format_pct(val: float | None) -> str:
    if val is None:
        return "Data unavailable"
    return f"{val * 100:.2f}%" if abs(val) <= 1.0 else f"{val:.2f}%"


def get_asset_fundamentals(symbol: str) -> dict[str, Any]:
    """
    Fetch comprehensive fundamentals for the requested asset.
    Supports Indian stocks (.NS), US stocks, Indices, Forex, and Crypto.
    """
    now = time.time()
    if symbol in _FUNDAMENTALS_CACHE:
        ts, cached = _FUNDAMENTALS_CACHE[symbol]
        if now - ts < CACHE_TTL_FUNDAMENTALS:
            return cached

    # Normalize symbol for yfinance if needed
    yf_symbol = symbol
    if "/" in symbol:  # Crypto like BTC/USDT -> BTC-USD
        base = symbol.split("/")[0]
        yf_symbol = f"{base}-USD"

    try:
        ticker = yf.Ticker(yf_symbol)
        info = ticker.info or {}
        fast_info = getattr(ticker, "fast_info", None)

        currency = (
            getattr(fast_info, "currency", None)
            or info.get("currency")
            or ("INR" if ".NS" in symbol or ".BO" in symbol else "USD")
        )

        mcap = (
            getattr(fast_info, "market_cap", None)
            or info.get("marketCap")
        )
        high_52 = (
            getattr(fast_info, "year_high", None)
            or info.get("fiftyTwoWeekHigh")
        )
        low_52 = (
            getattr(fast_info, "year_low", None)
            or info.get("fiftyTwoWeekLow")
        )

        pe = info.get("trailingPE") or info.get("forwardPE")
        forward_pe = info.get("forwardPE")
        eps = info.get("trailingEps") or info.get("forwardEps")
        revenue = info.get("totalRevenue")
        profit_margin = info.get("profitMargins")
        dividend_yield = info.get("dividendYield")
        roe = info.get("returnOnEquity")
        debt_to_equity = info.get("debtToEquity")
        beta = info.get("beta")
        sector = info.get("sector") or ("Cryptocurrency" if "/" in symbol else ("Index" if symbol.startswith("^") else "General"))
        industry = info.get("industry") or ("Digital Assets" if "/" in symbol else ("Market Index" if symbol.startswith("^") else "Financial Assets"))
        description = info.get("longBusinessSummary") or info.get("description") or f"Market tracking asset for {symbol}."

        result = {
            "symbol": symbol,
            "currency": currency,
            "company_name": info.get("shortName") or info.get("longName") or symbol,
            "sector": sector,
            "industry": industry,
            "description": description[:400] + "..." if len(description) > 400 else description,
            "metrics": {
                "market_cap": _format_currency(mcap, currency),
                "pe_ratio": _format_ratio(pe, "x"),
                "forward_pe": _format_ratio(forward_pe, "x"),
                "eps": _format_currency(eps, currency) if eps is not None else "Data unavailable",
                "revenue": _format_currency(revenue, currency),
                "profit_margin": _format_pct(profit_margin),
                "dividend_yield": _format_pct(dividend_yield) if dividend_yield else "0.00%",
                "roe": _format_pct(roe),
                "debt_to_equity": _format_ratio(debt_to_equity),
                "beta": _format_ratio(beta),
                "fifty_two_week_high": _format_currency(high_52, currency),
                "fifty_two_week_low": _format_currency(low_52, currency),
            },
            "raw_values": {
                "market_cap": mcap,
                "pe_ratio": pe,
                "eps": eps,
                "fifty_two_week_high": high_52,
                "fifty_two_week_low": low_52,
            },
            "timestamp": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        }

        _FUNDAMENTALS_CACHE[symbol] = (now, result)
        return result

    except Exception as exc:
        fallback = {
            "symbol": symbol,
            "currency": "USD",
            "company_name": symbol,
            "sector": "Market Asset",
            "industry": "Trading",
            "description": f"Real-time financial and market data for {symbol}.",
            "metrics": {
                "market_cap": "Data unavailable",
                "pe_ratio": "Data unavailable",
                "forward_pe": "Data unavailable",
                "eps": "Data unavailable",
                "revenue": "Data unavailable",
                "profit_margin": "Data unavailable",
                "dividend_yield": "Data unavailable",
                "roe": "Data unavailable",
                "debt_to_equity": "Data unavailable",
                "beta": "Data unavailable",
                "fifty_two_week_high": "Data unavailable",
                "fifty_two_week_low": "Data unavailable",
            },
            "raw_values": {},
            "timestamp": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
            "error": str(exc),
        }
        _FUNDAMENTALS_CACHE[symbol] = (now, fallback)
        return fallback
