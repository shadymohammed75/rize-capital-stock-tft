# src/preprocessing/feature_engineering.py
import pandas as pd
import talib as ta
import numpy as np


def calculate_technical_indicators(df: pd.DataFrame) -> pd.DataFrame:
    """Calculates all time-varying technical indicators using TA-Lib."""

    # Ensure the OHLCV columns are available
    close = df['close'].values.astype(np.float64)
    high = df['high'].values.astype(np.float64)
    low = df['low'].values.astype(np.float64)
    volume = df['volume'].values.astype(np.float64)

    # 1. Trend Indicators (SMA, EMA, MACD)
    # SMA (Simple Moving Average)
    df['SMA_20'] = ta.SMA(close, timeperiod=20)

    # EMA (Exponential Moving Average)
    df['EMA_20'] = ta.EMA(close, timeperiod=20)

    # MACD (Moving Average Convergence Divergence)
    # TA-Lib returns 3 series: MACD Line, Signal Line, and Histogram
    macd_line, macd_signal, macd_hist = ta.MACD(close, fastperiod=12, slowperiod=26, signalperiod=9)
    df['MACD_Line'] = macd_line
    df['MACD_Signal'] = macd_signal
    df['MACD_Hist'] = macd_hist  # This is your MACDh

    # 2. Momentum Indicators (RSI, Stochastic)
    # RSI (Relative Strength Index)
    df['RSI_14'] = ta.RSI(close, timeperiod=14)

    # STOCH (Stochastic Oscillator)
    # TA-Lib returns 2 series: SlowK and SlowD (the standard stochastic lines)
    stoch_k, stoch_d = ta.STOCH(high, low, close, fastk_period=14, slowk_period=3, slowd_period=3)
    df['STOCH_K'] = stoch_k
    df['STOCH_D'] = stoch_d

    # 3. Volatility & Volume Indicators
    # ATR (Average True Range)
    df['ATR_14'] = ta.ATR(high, low, close, timeperiod=14)

    # OBV (On-Balance Volume)
    df['OBV'] = ta.OBV(close, volume)  # Note: Volume is also required here

    # Drop rows with NaN values after indicator calculation (due to lookback)
    # These NaNs occur at the start of the series because the indicators haven't
    # accumulated enough data points (e.g., 26 days for MACD, 20 days for MA).
    df.dropna(inplace=True)

    return df


def create_date_features(df: pd.DataFrame) -> pd.DataFrame:
    """Creates time-varying known features."""

    # Ensure the index is a datetime object
    if not isinstance(df.index, pd.DatetimeIndex):
        df.index = pd.to_datetime(df.index)

    df['datetime'] = df.index

    # These are Time-Varying Known Inputs
    df['day_of_week'] = df['datetime'].dt.dayofweek.astype(str).astype('category')
    df['month'] = df['datetime'].dt.month.astype(str).astype('category')

    # --- THE FIX IS HERE ---
    # OLD (Calendar Days - Causes gaps/crashes):
    # df['time_idx'] = (df['datetime'] - df['datetime'].min()).dt.days

    # NEW (Trading Days - Contiguous):
    # This treats every row as t+1, ignoring weekend gaps.
    # This is standard for stock prediction and prevents the "Index 68" crash.
    df['time_idx'] = np.arange(len(df))

    return df.drop(columns=['datetime'])


def preprocess_data(df: pd.DataFrame) -> pd.DataFrame:
    """Combines indicator calculation and feature creation."""
    # 1. CLEANING STEP: Drop unnecessary columns
    print("Dropping unused raw columns: 'timestamp' and 'resolution'.")
    # Note: 'datetime' is the index name now, so it's not a column to drop.
    columns_to_drop = ['timestamp', 'resolution']

    # Use errors='ignore' as 'timestamp' might not be a column if you used it
    # to set the index in another scenario. 'resolution' should be dropped.
    df.drop(columns=columns_to_drop, inplace=True, errors='ignore')
    df = calculate_technical_indicators(df)
    df = create_date_features(df)
    return df