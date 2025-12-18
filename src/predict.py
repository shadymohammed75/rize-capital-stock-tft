import torch
import pandas as pd
import numpy as np
from pytorch_forecasting import TemporalFusionTransformer, TimeSeriesDataSet
import os
import warnings

# Suppress warnings for clean output
warnings.filterwarnings("ignore")

# --- CONFIGURATION ---
MODEL_PATH = r'C:\Users\dell\PycharmProjects\rizecapital_stock_prediction\src\src\checkpoints\epoch=4-val_loss=3.68.ckpt'
DATA_PATH = r"C:\Users\dell\PycharmProjects\rizecapital_stock_prediction\data\processed\AAPL_processed.pkl"


def add_future_dummy_rows(df: pd.DataFrame, horizon: int = 5):
    """
    Appends 'horizon' empty rows to the dataframe.
    Uses FORWARD FILL to carry over the last known values (price, indicators)
    into the future slots. This prevents the scaler from seeing '0.0' and
    crashing the prediction.
    """
    last_date = df.index[-1]
    last_time_idx = df['time_idx'].iloc[-1]

    # 1. Generate next 5 Business Days
    future_dates = pd.date_range(start=last_date, periods=horizon + 1, freq='B')[1:]

    # 2. Create container
    future_df = pd.DataFrame(index=future_dates)
    future_df['time_idx'] = range(last_time_idx + 1, last_time_idx + 1 + horizon)
    future_df['day_of_week'] = future_df.index.dayofweek.astype(str).astype('category')
    future_df['month'] = future_df.index.month.astype(str).astype('category')
    future_df['ticker'] = 'AAPL'

    # 3. FILL UNKNOWN COLS WITH LAST KNOWN VALUE (Forward Fill)
    # This is the critical fix. We take the last row of real data
    # and copy it into the future rows.
    last_row = df.iloc[-1]

    for col in df.columns:
        if col not in future_df.columns:
            # Copy the value from the last real row
            future_df[col] = last_row[col]

    # 4. Concatenate
    return pd.concat([df, future_df[df.columns]])

def predict_future():
    # 1. Load Data & Model
    if not os.path.exists(DATA_PATH) or not os.path.exists(MODEL_PATH):
        print("❌ Error: Files not found.")
        return

    print(f"Loading model from {os.path.basename(MODEL_PATH)}...")
    best_tft = TemporalFusionTransformer.load_from_checkpoint(MODEL_PATH, map_location=torch.device('cpu'))

    print(f"Loading data...")
    df = pd.read_pickle(DATA_PATH)

    # 2. PREPARE DATA
    # Use the LAST 180 days to be safe (plenty of history)
    history_df = df.tail(180).copy()

    # Add the future rows
    prediction_df = add_future_dummy_rows(history_df, horizon=5)

    print(f"Preparing inference dataset (Rows: {len(prediction_df)})...")

    # 3. CREATE DATASET MANUALLY
    try:
        # We set min_encoder_length=1 to prevent "filter" crashes
        inference_ds = TimeSeriesDataSet.from_parameters(
            best_tft.dataset_parameters,
            prediction_df,
            predict=True,
            stop_randomization=True,
            min_encoder_length=1
        )

        # Batch size can be larger now, but 32 is safe
        inference_dataloader = inference_ds.to_dataloader(train=False, batch_size=32, num_workers=0)

    except Exception as e:
        print(f"\n❌ Dataset Creation Failed: {e}")
        return

    # 4. PREDICT
    print("Calculating forecast...")
    try:
        # FIX IS HERE: Remove 'return_x=True' and do not unpack.
        # We only need the output dictionary.
        raw_prediction = best_tft.predict(inference_dataloader, mode="raw")

    except Exception as e:
        print(f"\n❌ Prediction Failed: {e}")
        return

    # 5. EXTRACT & PRINT
    # raw_prediction is a dictionary containing 'prediction' tensor.
    # Shape: (n_samples, horizon, quantiles)
    # We want the LAST sample (which corresponds to the latest time window)

    # Index 3 is the Median (0.5 quantile)
    forecast_tensor = raw_prediction['prediction'][-1, :, 3]
    forecast_values = forecast_tensor.tolist()

    print("\n" + "=" * 45)
    print(f"🚀 RIZE CAPITAL: AAPL FORECAST (Next 5 Days)")
    print("=" * 45)

    last_real_close = history_df['close'].iloc[-1]
    print(f"Ref Close: ${last_real_close:.2f} ({history_df.index[-1].date()})")
    print("-" * 45)

    # Get the dates from our prediction dataframe (the last 5 rows)
    future_dates = prediction_df.index[-5:]

    for date, price in zip(future_dates, forecast_values):
        change = ((price - last_real_close) / last_real_close) * 100
        arrow = "🟢" if change > 0 else "🔴"
        print(f"{date.date()}: ${price:.2f}  ({arrow} {change:+.2f}%)")

    print("=" * 45)


if __name__ == '__main__':
    predict_future()