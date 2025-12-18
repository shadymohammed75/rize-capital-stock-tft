# src/models/tft_trainer.py

from lightning.pytorch.callbacks import EarlyStopping, ModelCheckpoint
from lightning.pytorch import Trainer
from pytorch_forecasting import TemporalFusionTransformer


def train_tft_model(model, train_dataloader, val_dataloader, max_epochs=50):
    # 1. Define Callbacks
    checkpoint_callback = ModelCheckpoint(
        monitor="val_loss",
        mode="min",
        save_top_k=1,
        dirpath="src/checkpoints",  # Ensure this path exists or matches your project structure
        filename="{epoch}-{val_loss:.2f}",
        verbose=True,  # Set to True so you see when a new best model is saved
    )

    early_stop_callback = EarlyStopping(
        monitor="val_loss",
        min_delta=1e-4,
        patience=10,
        verbose=True,
        mode="min"
    )

    # 2. Define the Lightning Trainer
    trainer = Trainer(
        max_epochs=max_epochs,
        accelerator="auto",
        callbacks=[early_stop_callback, checkpoint_callback],
        logger=True,  # Turn logger on (useful for TensorBoard later)

        # --- CRITICAL FIXES HERE ---
        # limit_train_batches=100,  <-- DELETE THIS LINE! It prevents full training.

        enable_progress_bar=True,

        # Add Gradient Clipping.
        # This prevents the model from crashing or diverging when it hits a "surprise" in stock data.
        gradient_clip_val=0.1,
    )

    # 3. Train the Model
    print(f"\nStarting TFT Model Training for {max_epochs} epochs...")
    trainer.fit(
        model,
        train_dataloaders=train_dataloader,
        val_dataloaders=val_dataloader,
    )

    # Load best model
    best_model_path = trainer.checkpoint_callback.best_model_path
    if best_model_path:
        print(f"✅ Loaded best model from: {best_model_path}")
        best_tft = TemporalFusionTransformer.load_from_checkpoint(best_model_path)
        return best_tft
    else:
        print("Warning: No best model checkpoint found.")
        return model