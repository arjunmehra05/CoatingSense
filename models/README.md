# Models Directory

This directory should contain the trained model files after running the training scripts.

## Required Model Files
- `cnn_model.keras` - CNN model for coating image classification
- `lstm_model.keras` - LSTM model for sensor sequence analysis  
- `fusion_model.keras` - Fusion model combining CNN and LSTM outputs

## Training Instructions

1. **Generate synthetic data:**
   ```bash
   python training/data_generation.py
   ```

2. **Train individual models:**
   ```bash
   python training/cnn_training.py
   python training/lstm_training.py
   ```

3. **Train fusion model:**
   ```bash
   python training/fusion+demo.py
   ```

The trained models will be saved here automatically. Place the `.keras` files in this directory for the application to load them.