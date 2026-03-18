import numpy as np
import tensorflow as tf
import streamlit as st

MODEL_DIR = "models"


@st.cache_resource
def load_models():
    cnn    = tf.keras.models.load_model(f'{MODEL_DIR}/cnn_model.keras')
    lstm   = tf.keras.models.load_model(f'{MODEL_DIR}/lstm_model.keras')
    fusion = tf.keras.models.load_model(f'{MODEL_DIR}/fusion_model.keras')
    return cnn, lstm, fusion


def run_inference(cnn_model, lstm_model, fusion_model, img_rgb, sensor_seq):
    cnn_out    = cnn_model.predict(np.expand_dims(img_rgb, 0), verbose=0)[0]
    lstm_out   = lstm_model.predict(sensor_seq[np.newaxis], verbose=0)[0]
    fusion_in  = np.concatenate([cnn_out, lstm_out])[np.newaxis].astype(np.float32)
    fusion_out = fusion_model.predict(fusion_in, verbose=0)[0]
    pred       = int(fusion_out.argmax())
    return cnn_out, lstm_out, fusion_out, pred