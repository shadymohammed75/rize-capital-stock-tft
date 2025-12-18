# src/utils/data_utils.py
import pandas as pd
from sklearn.preprocessing import StandardScaler


def scale_and_split(df: pd.DataFrame, target_column: str = 'Close'):
    """Scales continuous features and splits data for training."""

    # Identify continuous and categorical columns
    continuous_cols = [col for col in df.columns if df[col].dtype != 'category']
    categorical_cols = [col for col in df.columns if df[col].dtype == 'category']

    # --- Scaling Continuous Variables (Excluding 'time_idx' and Target for now)

    # Apply standard scaling
    scaler = StandardScaler()
    df[continuous_cols] = scaler.fit_transform(df[continuous_cols])

    # --- Train/Validation Split (Time-based split is crucial)

    # Use 80% for training, 20% for validation
    train_size = int(len(df) * 0.8)
    train_df = df.iloc[:train_size]
    val_df = df.iloc[train_size:]

    # In a real TFT implementation, you'd use a dedicated TimeSeriesDataSet object
    # to handle the lookback/forecast windowing, but this gets the data ready.

    return train_df, val_df, scaler, continuous_cols, categorical_cols