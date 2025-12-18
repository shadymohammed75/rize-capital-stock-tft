# appl_main.py
import pandas as pd
import os
import torch
from pytorch_forecasting import TimeSeriesDataSet, NaNLabelEncoder
from torch.utils.data import DataLoader
from src.preprocessing.feature_engneering  import  preprocess_data
from src.utils.data_utils import scale_and_split


from src.models.tft_model import create_tft_model
from src.models.tft_trainer import train_tft_model

# --- Configuration ---
RAW_PATH = r'C:\Users\dell\PycharmProjects\rizecapital_stock_prediction\data\raw\stocks\AAPL_historical_data.csv'
PROCESSED_PATH = r'C:\Users\dell\PycharmProjects\rizecapital_stock_prediction\data\processed'

# TFT Hyperparameters
MAX_ENCODER_LENGTH = 60  # Lookback days
MAX_PREDICTION_LENGTH = 5  # Days to forecast (H=5)
PROCESSED_FILE = os.path.join(PROCESSED_PATH, 'AAPL_processed.pkl') #(the file)


def main_pipeline():
    # PHASE 1: Data Processing and Persistence 💾
    # --------------------------------------------------------------------------
    if os.path.exists(PROCESSED_FILE):
        # FIX 1: Use PROCESSED_FILE in the log message
        print(f"Loading pre-processed data from {PROCESSED_FILE}...")

        # CORRECT: Reading the FILE, which is what caused your previous error
        processed_df = pd.read_pickle(PROCESSED_FILE)
    else:
        print("Processed data not found. Starting feature engineering...")

        # 1. Load Raw Data
        df = pd.read_csv(RAW_PATH, index_col='datetime', parse_dates=True)

        # 2. Add Indicators, Date Features, and clean NaNs
        processed_df = preprocess_data(df.copy())

        # 3. Add Static Covariate (TFT Requirement)
        processed_df['ticker'] = 'AAPL'

        # 4. Save the Result
        # FIX 2: Correctly create the directory using PROCESSED_PATH
        os.makedirs(PROCESSED_PATH, exist_ok=True)

        # CORRECT: Saving to the FILE
        processed_df.to_pickle(PROCESSED_FILE)

        # FIX 3: Use PROCESSED_FILE in the log message
        print(f"Processed data (with indicators) saved to {PROCESSED_FILE}")

    print(f"Data preparation complete. Total samples: {len(processed_df)}.")

    # PHASE 2: TimeSeriesDataSet Setup and Training 🚀
    # --------------------------------------------------------------------------

    # 1. Time-based Split
    training_cutoff_index = int(len(processed_df) * 0.8)
    # The TimeSeriesDataSet needs the time_idx, not the date index, for splitting
    training_cutoff_value = processed_df['time_idx'].iloc[training_cutoff_index]

    # 2. Define the TimeSeriesDataSet (Training Data)
    print("Creating TimeSeriesDataSet...")
    training_data_set = TimeSeriesDataSet(
        processed_df[lambda x: x['time_idx'] <= training_cutoff_value],
        time_idx="time_idx",
        target="close",
        group_ids=["ticker"],  # The Static Covariate
        max_encoder_length=MAX_ENCODER_LENGTH,
        max_prediction_length=MAX_PREDICTION_LENGTH,

        # --- Feature Mapping ---
        static_categoricals=["ticker"],
        time_varying_known_categoricals=["day_of_week", "month"],
        # All continuous features that change over time and are unknown in the future
        time_varying_unknown_reals=[
            col for col in processed_df.columns
            if col not in ['time_idx', 'ticker', 'day_of_week', 'month']
        ],
        categorical_encoders={"day_of_week": NaNLabelEncoder(add_nan=True), "month": NaNLabelEncoder(add_nan=True)},
        allow_missing_timesteps=True,

    )

    # 3. Create Validation Data Set
    validation_data_set = TimeSeriesDataSet.from_dataset(
        training_data_set,
        processed_df,
        min_prediction_idx=training_cutoff_value,
        stop_randomization=True
    )

    # 4. Create DataLoaders
    batch_size = 64
    train_dataloader = training_data_set.to_dataloader(
        train=True, batch_size=batch_size, num_workers=0, drop_last=True
    )
    val_dataloader = validation_data_set.to_dataloader(
        train=False, batch_size=batch_size, num_workers=0
    )

    # 5. Initialize and Train the TFT Model
    tft_model = create_tft_model(training_data_set)

    # FIX IS HERE: Calculate and print the number of parameters directly
    n_params = sum(p.numel() for p in tft_model.parameters())
    print(f"Model initialized with {n_params:,} parameters.")

    # Call the trainer function
    best_tft = train_tft_model(tft_model, train_dataloader, val_dataloader, max_epochs=5)

    print("\n✅ Training Complete. Best model is ready for prediction and interpretation.")


if __name__ == '__main__':
    main_pipeline()




