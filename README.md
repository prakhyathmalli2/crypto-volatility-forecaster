# crypto-volatility-forecaster
'Final year CSE major project - crypto volatility forecasting and trading signal generation'.

# Crypto Volatility Forecaster & Smart Trading Signal Generator

A final-year CSE major project that forecasts short-term cryptocurrency volatility 
and generates trading signals (Buy/Sell/Hold) by combining statistical volatility 
modeling with machine learning on historical price data.

## Problem Statement
Cryptocurrency markets are highly volatile and unpredictable, making risk assessment 
difficult for traders. This project builds a system that (1) forecasts near-term 
volatility using time-series statistical models, and (2) generates trading signals 
that adapt based on the forecasted volatility regime.

## Tech Stack
- **Language:** Python
- **Data:** pandas, numpy, CoinGecko API (historical OHLCV data)
- **Volatility Modeling:** arch (GARCH models), rolling statistics
- **Machine Learning:** scikit-learn / XGBoost (signal classification)
- **Dashboard/Deployment:** Streamlit, Streamlit Community Cloud
- **Version Control:** Git & GitHub

## Project Structure
## Methodology (high level)
1. Collect historical OHLCV crypto data via public API
2. Engineer features: returns, rolling volatility, RSI, MACD, Bollinger Bands
3. Forecast volatility using GARCH modeling
4. Train a classifier to generate trading signals, using forecasted volatility as an input feature
5. Backtest signal performance against historical data
6. Deploy an interactive dashboard showing live forecasts and signals

## Status
🚧 Work in progress — Week 1: Data collection & volatility fundamentals complete.

## Disclaimer
This project is built for academic and educational purposes. It is a backtested 
forecasting and signal-generation tool, not financial advice, and is not intended 
for live automated trading with real funds.

## Author
Prakhyath Malli — Final Year CSE
