# RIZE CAPITAL: Probabilistic Stock Forecasting with TFT

![Python](https://img.shields.io/badge/python-3.10+-blue.svg)
![PyTorch](https://img.shields.io/badge/PyTorch-2.0+-EE4C2C.svg)
![Lightning](https://img.shields.io/badge/Lightning-2.0+-792EE5.svg)

An end-to-end quantitative trading project developing a **Temporal Fusion Transformer (TFT)** for multi-horizon price forecasting. This model moves beyond traditional point estimates to provide **probabilistic forecasts** (Quantile Regression), essential for institutional risk management and Value-at-Risk (VaR) analysis.

## 🚀 Key Features
- **Deep Learning Architecture:** Utilizes the TFT model to handle static metadata, known future covariates, and observed historical inputs.
- **Advanced Feature Engineering:** Integrated **TA-Lib** for high-fidelity technical indicators (RSI, MACD, OBV, SMA).
- **Quantile Loss Function:** Predicts 7 different confidence intervals (quantiles) to visualize market uncertainty.
- **Production Pipeline:** Includes automated data preprocessing, normalization, model checkpointing, and recursive 5-day inference.

## 📊 Model Performance
The model demonstrated strong convergence during training, successfully learning temporal dependencies in equity markets.
- **Baseline Validation Loss:** 133.08
- **Final Optimized Validation Loss:** **3.68**
- **Forecast Horizon:** 5 Trading Days

<img width="430" height="250" alt="image" src="https://github.com/user-attachments/assets/625468a2-7bd4-4215-93a0-e3263b352533" />


## 📁 Project Structure
```text
├── data/               # Raw and pre-processed AAPL datasets
├── src/
│   ├── models/         # TFT architecture and trainer logic
│   ├── preprocessing/  # TA-Lib feature engineering and scaling
│   ├── checkpoints/    # Saved model states (.ckpt)
│   ├── predict.py      # Inference script for forward-looking forecasts
│   └── main.py         # Entry point for training pipeline
└── requirements.txt    # Production dependencies

