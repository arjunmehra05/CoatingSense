import numpy as np
import tensorflow as tf
import streamlit as st
from pathlib import Path
from keras.models import load_model   # ✅ correct loader

MODEL_DIR = Path(__file__).resolve().parent.parent / "models"


def _model_path(name: str) -> str:
    return str((MODEL_DIR / name).resolve())


@st.cache_resource
def load_models():
    print(f"[DEBUG] Loading models from: {MODEL_DIR}")
    print(f"[DEBUG] Current directory: {Path.cwd()}")
    print(f"[DEBUG] Models dir exists: {MODEL_DIR.exists()}")

    if MODEL_DIR.exists():
        print(f"[DEBUG] Files in models/: {sorted([p.name for p in MODEL_DIR.iterdir()])}")

    required_files = ["cnn_model.keras", "lstm_model.keras", "fusion_model.keras"]
    missing_files = [fname for fname in required_files if not (MODEL_DIR / fname).exists()]

    if missing_files:
        raise FileNotFoundError(
            f"Missing model files in {MODEL_DIR}: {', '.join(missing_files)}"
        )

    # ✅ Use Keras 3 loader
    keras_load_model = load_model

    try:
        cnn = keras_load_model(_model_path("cnn_model.keras"), compile=False)
        print("[DEBUG] CNN model loaded successfully")
    except Exception as e:
        print(f"[ERROR] Failed to load CNN: {e}")
        raise

    try:
        lstm = keras_load_model(_model_path("lstm_model.keras"), compile=False)
        print("[DEBUG] LSTM model loaded successfully")
    except Exception as e:
        print(f"[ERROR] Failed to load LSTM: {e}")
        raise

    try:
        fusion = keras_load_model(_model_path("fusion_model.keras"), compile=False)
        print("[DEBUG] Fusion model loaded successfully")
    except Exception as e:
        print(f"[ERROR] Failed to load Fusion: {e}")
        raise

    print("[DEBUG] All models loaded successfully!")
    return cnn, lstm, fusion


def run_inference(cnn_model, lstm_model, fusion_model, img_rgb, sensor_seq):
    print("[DEBUG] Starting inference...")

    try:
        cnn_out = cnn_model.predict(np.expand_dims(img_rgb, 0), verbose=0)[0]
        print(f"[DEBUG] CNN inference done: {cnn_out}")
    except Exception as e:
        print(f"[ERROR] CNN inference failed: {e}")
        raise

    try:
        lstm_out = lstm_model.predict(sensor_seq[np.newaxis], verbose=0)[0]
        print(f"[DEBUG] LSTM inference done: {lstm_out}")
    except Exception as e:
        print(f"[ERROR] LSTM inference failed: {e}")
        raise

    try:
        fusion_in = np.concatenate([cnn_out, lstm_out])[np.newaxis].astype(np.float32)
        fusion_out = fusion_model.predict(fusion_in, verbose=0)[0]
        pred = int(fusion_out.argmax())
        print(f"[DEBUG] Fusion inference done: {fusion_out}, pred: {pred}")
    except Exception as e:
        print(f"[ERROR] Fusion inference failed: {e}")
        raise

    return cnn_out, lstm_out, fusion_out, pred