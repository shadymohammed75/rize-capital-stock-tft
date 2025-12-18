# src/models/tft_model.py
from pytorch_forecasting.models import TemporalFusionTransformer
from pytorch_forecasting.metrics import QuantileLoss


def create_tft_model(training_data_set):
    """Initializes the Temporal Fusion Transformer model."""

    tft = TemporalFusionTransformer.from_dataset(
        training_data_set,
        # --- Core Architecture Parameters ---
        # INCREASED CAPACITY:
        hidden_size=128,  # Was 64. 128 allows for more complex pattern recognition.
        lstm_layers=2,  # Was 1. 2 layers helps capture deeper temporal dependencies.
        attention_head_size=4,

        # --- Training & Regularization Parameters ---
        dropout=0.15,  # Slight increase to prevent overfitting with the larger model.
        loss=QuantileLoss(),

        # --- Interpretability Output ---
        output_size=7,

        # --- Learning Rate ---
        log_interval=10,  # Set to 10 so you see logs in the console

        # LOWER LEARNING RATE:
        # Stock data is noisy. 1e-3 is often too fast and makes the model jump around.
        # 1e-4 is slower but much more stable for convergence.
        learning_rate=0.001,  # Keep 1e-3 for now, but if loss bounces, lower to 0.0001

        # Reduce 'weight_decay' to allow the model to learn slightly more aggressive patterns
        weight_decay=1e-2,
    )
    return tft