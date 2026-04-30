import numpy as np
import tensorflow as tf
import streamlit as st
import os

MODEL_DIR = "models"


@st.cache_resource
def load_models():
    print(f"[DEBUG] Loading models from: {MODEL_DIR}")
    print(f"[DEBUG] Current directory: {os.getcwd()}")
    print(f"[DEBUG] Models dir exists: {os.path.exists(MODEL_DIR)}")
    if os.path.exists(MODEL_DIR):
        print(f"[DEBUG] Files in models/: {os.listdir(MODEL_DIR)}")
    
    try:
        cnn    = tf.keras.models.load_model(f'{MODEL_DIR}/cnn_model.keras')
        print("[DEBUG] CNN model loaded successfully")
    except Exception as e:
        print(f"[ERROR] Failed to load CNN: {e}")
        raise
    
    try:
        lstm   = tf.keras.models.load_model(f'{MODEL_DIR}/lstm_model.keras')
        print("[DEBUG] LSTM model loaded successfully")
    except Exception as e:
        print(f"[ERROR] Failed to load LSTM: {e}")
        raise
    
    try:
        fusion = tf.keras.models.load_model(f'{MODEL_DIR}/fusion_model.keras')
        print("[DEBUG] Fusion model loaded successfully")
    except Exception as e:
        print(f"[ERROR] Failed to load Fusion: {e}")
        raise
    
    print("[DEBUG] All models loaded successfully!")
    return cnn, lstm, fusion


def run_inference(cnn_model, lstm_model, fusion_model, img_rgb, sensor_seq):
    print("[DEBUG] Starting inference...")
    try:
        cnn_out    = cnn_model.predict(np.expand_dims(img_rgb, 0), verbose=0)[0]
        print(f"[DEBUG] CNN inference done: {cnn_out}")
    except Exception as e:
        print(f"[ERROR] CNN inference failed: {e}")
        raise
    
    try:
        lstm_out   = lstm_model.predict(sensor_seq[np.newaxis], verbose=0)[0]
        print(f"[DEBUG] LSTM inference done: {lstm_out}")
    except Exception as e:
        print(f"[ERROR] LSTM inference failed: {e}")
        raise
    
    try:
        fusion_in  = np.concatenate([cnn_out, lstm_out])[np.newaxis].astype(np.float32)
        fusion_out = fusion_model.predict(fusion_in, verbose=0)[0]
        pred       = int(fusion_out.argmax())
        print(f"[DEBUG] Fusion inference done: {fusion_out}, pred: {pred}")
    except Exception as e:
        print(f"[ERROR] Fusion inference failed: {e}")
        raise
    
    return cnn_out, lstm_out, fusion_out, pred