"""
Data Ingestion and Preprocessing Module
Fetches cryptocurrency price data from Yahoo Finance API with offline fallback
and robust multi-index handling.
"""

import pandas as pd
import numpy as np
import yfinance as yf
from datetime import datetime, timedelta
import logging

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Standard crypto ticker map
CRYPTO_TICKERS = {
    "Bitcoin (BTC)": "BTC-USD",
    "Ethereum (ETH)": "ETH-USD",
    "Solana (SOL)": "SOL-USD",
    "Binance Coin (BNB)": "BNB-USD",
    "Cardano (ADA)": "ADA-USD",
    "Ripple (XRP)": "XRP-USD",
    "Dogecoin (DOGE)": "DOGE-USD",
    "Avalanche (AVAX)": "AVAX-USD",
    "Chainlink (LINK)": "LINK-USD",
    "Polygon (POL/MATIC)": "POL-USD"
}


def _generate_synthetic_crypto_data(ticker: str = "BTC-USD", days: int = 730) -> pd.DataFrame:
    """
    Generate realistic synthetic cryptocurrency OHLCV data with volatility clustering
    as an offline fallback if internet connection or API fails.
    """
    logger.warning(f"Generating synthetic fallback data for {ticker} across {days} days.")
    np.random.seed(42)
    
    end_date = datetime.today()
    start_date = end_date - timedelta(days=days)
    dates = pd.date_range(start=start_date, end=end_date, freq='D')
    n = len(dates)

    # GARCH(1,1) simulated volatility clustering
    omega, alpha, beta = 0.05, 0.15, 0.80
    sigma2 = np.zeros(n)
    sigma2[0] = omega / (1 - alpha - beta)
    eps = np.zeros(n)
    z = np.random.standard_t(df=5, size=n)

    for t in range(1, n):
        sigma2[t] = omega + alpha * (eps[t-1]**2) + beta * sigma2[t-1]
        eps[t] = np.sqrt(sigma2[t]) * z[t]

    log_returns = eps / 100.0
    initial_price = 45000.0 if "BTC" in ticker else (2800.0 if "ETH" in ticker else 150.0)
    prices = initial_price * np.exp(np.cumsum(log_returns))

    highs = prices * (1 + np.abs(np.random.normal(0.01, 0.008, n)))
    lows = prices * (1 - np.abs(np.random.normal(0.01, 0.008, n)))
    opens = (highs + lows) / 2.0 + np.random.normal(0, 0.005 * prices, n)
    volumes = np.random.lognormal(20, 0.8, n)

    df = pd.DataFrame({
        'Open': opens,
        'High': highs,
        'Low': lows,
        'Close': prices,
        'Volume': volumes
    }, index=dates)

    df['Return'] = df['Close'].pct_change()
    df['Log_Return'] = np.log(df['Close'] / df['Close'].shift(1))
    df['Log_Return_Pct'] = df['Log_Return'] * 100.0
    df.dropna(inplace=True)
    return df


def fetch_crypto_data(
    ticker: str = "BTC-USD",
    start_date: str = None,
    end_date: str = None,
    period: str = "2y",
    interval: str = "1d"
) -> pd.DataFrame:
    """
    Fetch historical OHLCV data for a given cryptocurrency ticker.

    Args:
        ticker (str): Yahoo Finance ticker (e.g., 'BTC-USD').
        start_date (str, optional): 'YYYY-MM-DD' start date.
        end_date (str, optional): 'YYYY-MM-DD' end date.
        period (str): Valid yfinance periods ('1y', '2y', '3y', '5y', 'max').
        interval (str): Data frequency ('1d', '1h', etc.).

    Returns:
        pd.DataFrame: Cleaned dataframe with Open, High, Low, Close, Volume, Returns.
    """
    ticker_clean = ticker.strip().upper()
    if not ticker_clean.endswith("-USD") and not ("=" in ticker_clean):
        ticker_clean = f"{ticker_clean}-USD"

    try:
        logger.info(f"Fetching data for {ticker_clean} (period={period})...")
        
        if start_date and end_date:
            df = yf.download(ticker_clean, start=start_date, end=end_date, interval=interval, progress=False)
        else:
            df = yf.download(ticker_clean, period=period, interval=interval, progress=False)

        if df is None or df.empty:
            logger.warning(f"yfinance returned empty dataset for {ticker_clean}. Engaging fallback.")
            return _generate_synthetic_crypto_data(ticker_clean)

        # Handle multi-level columns if returned by yfinance
        if isinstance(df.columns, pd.MultiIndex):
            # Check whether 'Close' is in level 0 or level 1
            if 'Close' in df.columns.get_level_values(0):
                df.columns = df.columns.get_level_values(0)
            else:
                df.columns = df.columns.get_level_values(1)

        # Ensure required columns exist
        required_cols = ['Open', 'High', 'Low', 'Close', 'Volume']
        for col in required_cols:
            if col not in df.columns:
                raise KeyError(f"Missing expected column: {col}")

        df = df[required_cols].copy()
        
        # Cast to float
        for col in required_cols:
            df[col] = pd.to_numeric(df[col], errors='coerce')

        df.dropna(inplace=True)
        df.sort_index(inplace=True)

        # Deduplicate index
        df = df[~df.index.duplicated(keep='first')]

        # Compute price returns
        df['Return'] = df['Close'].pct_change()
        df['Log_Return'] = np.log(df['Close'] / df['Close'].shift(1))
        df['Log_Return_Pct'] = df['Log_Return'] * 100.0

        # Replace any accidental inf/-inf
        df.replace([np.inf, -np.inf], np.nan, inplace=True)
        df.dropna(subset=['Log_Return', 'Return', 'Close'], inplace=True)

        if len(df) < 30:
            logger.warning("Insufficient data points fetched (<30). Using synthetic dataset.")
            return _generate_synthetic_crypto_data(ticker_clean)

        logger.info(f"Successfully loaded {len(df)} records for {ticker_clean}.")
        return df

    except Exception as e:
        logger.error(f"Error fetching data for {ticker_clean}: {e}. Activating fallback generator.")
        return _generate_synthetic_crypto_data(ticker_clean)


def get_crypto_summary_metrics(df: pd.DataFrame) -> dict:
    """
    Compute high-level statistical summary metrics for a cryptocurrency series.
    """
    if df is None or df.empty:
        return {}

    latest_close = float(df['Close'].iloc[-1])
    prev_close = float(df['Close'].iloc[-2]) if len(df) > 1 else latest_close
    change_24h_pct = ((latest_close - prev_close) / prev_close) * 100.0

    daily_vol = float(df['Log_Return'].std())
    annualized_vol_pct = daily_vol * np.sqrt(365) * 100.0

    high_52w = float(df['High'].max())
    low_52w = float(df['Low'].min())

    first_close = float(df['Close'].iloc[0])
    total_return_pct = ((latest_close - first_close) / first_close) * 100.0

    return {
        "latest_price": latest_close,
        "change_24h_pct": round(change_24h_pct, 2),
        "daily_volatility_pct": round(daily_vol * 100.0, 2),
        "annualized_volatility_pct": round(annualized_vol_pct, 2),
        "52w_high": round(high_52w, 2),
        "52w_low": round(low_52w, 2),
        "total_return_pct": round(total_return_pct, 2),
        "total_data_points": len(df)
    }
