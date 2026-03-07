import numpy as np
import cv2
import tensorflow as tf


# ─────────────────────────────────────────────
# GRAD-CAM
# ─────────────────────────────────────────────
def get_gradcam_heatmap(model, img_array):
    mobilenet       = model.get_layer('mobilenetv2_1.00_224')
    last_conv_layer = mobilenet.get_layer('out_relu')
    conv_model      = tf.keras.Model(inputs=mobilenet.input, outputs=last_conv_layer.output)

    preprocessed = tf.keras.applications.mobilenet_v2.preprocess_input(
        tf.cast(img_array, tf.float32)
    )
    preprocessed = tf.Variable(preprocessed)

    with tf.GradientTape() as tape:
        tape.watch(preprocessed)
        conv_outputs = conv_model(preprocessed, training=False)
        x            = model.get_layer('global_average_pooling2d')(conv_outputs)
        x            = model.get_layer('dropout')(x, training=False)
        x            = model.get_layer('dense')(x)
        x            = model.get_layer('dropout_1')(x, training=False)
        predictions  = model.get_layer('dense_1')(x)
        pred_class   = tf.argmax(predictions[0])
        class_score  = predictions[:, pred_class]

    grads        = tape.gradient(class_score, conv_outputs)
    pooled_grads = tf.reduce_mean(grads, axis=(0, 1, 2))
    conv_out     = conv_outputs[0]
    heatmap      = conv_out @ pooled_grads[..., tf.newaxis]
    heatmap      = tf.squeeze(heatmap)
    heatmap      = tf.maximum(heatmap, 0) / (tf.math.reduce_max(heatmap) + 1e-8)
    full_preds   = model(img_array, training=False)

    return heatmap.numpy(), pred_class.numpy(), full_preds[0].numpy()


def overlay_gradcam(img_bgr, heatmap, alpha=0.45):
    heatmap_resized = cv2.resize(heatmap, (img_bgr.shape[1], img_bgr.shape[0]))
    heatmap_uint8   = np.uint8(255 * heatmap_resized)
    heatmap_colored = cv2.applyColorMap(heatmap_uint8, cv2.COLORMAP_JET)
    return cv2.addWeighted(img_bgr, 1 - alpha, heatmap_colored, alpha, 0)


# ─────────────────────────────────────────────
# LSTM SALIENCY
# ─────────────────────────────────────────────
def get_lstm_saliency(model, sensor_seq):
    input_tensor = tf.Variable(sensor_seq[np.newaxis], dtype=tf.float32)
    with tf.GradientTape() as tape:
        predictions = model(input_tensor, training=False)
        pred_class  = tf.argmax(predictions[0])
        class_score = predictions[:, pred_class]
    grads    = tape.gradient(class_score, input_tensor)
    saliency = tf.abs(grads[0]).numpy()
    return saliency, pred_class.numpy(), predictions[0].numpy()


# ─────────────────────────────────────────────
# INSIGHT TEXT
# ─────────────────────────────────────────────
def gradcam_insight(heatmap, pred_class, probs, coating_state):
    class_names = ['Good', 'Degraded', 'Failed']
    heatmap_r   = cv2.resize(heatmap, (224, 224))
    concentration = float(np.std(heatmap_r))
    h, w    = heatmap_r.shape
    center  = heatmap_r[h//4:3*h//4, w//4:3*w//4].mean()
    edges   = heatmap_r.mean() - center
    focus_region = "central coating area" if center > edges else "image edges"
    predicted    = class_names[pred_class]
    confidence   = probs[pred_class]

    if concentration > 0.25:
        attention = "tightly focused on specific regions"
    elif concentration > 0.15:
        attention = "moderately focused across the coating"
    else:
        attention = "diffuse across the entire image"

    lines = [
        f"Predicted <b>{predicted}</b> with <b>{confidence:.0%} confidence</b>.",
        f"Model attention is <b>{attention}</b>, concentrated on the <b>{focus_region}</b>."
    ]

    if coating_state == 'good':
        if concentration < 0.2:
            lines.append("Diffuse attention is expected for intact coatings — no localized defects to focus on.")
        else:
            lines.append("Focused attention on a good coating may indicate the model is responding to lighting variation rather than actual defects.")
    elif coating_state == 'degraded':
        if concentration > 0.2:
            lines.append("The model successfully localized subtle crack or discoloration regions characteristic of early degradation.")
        else:
            lines.append("Degraded defects in this sample are subtle — the model may be relying on global texture shifts rather than specific defect locations.")
    elif coating_state == 'failed':
        if concentration > 0.25:
            lines.append("The model correctly identified prominent damage patches and deep cracks — strong visual evidence of coating failure.")
        else:
            lines.append("Despite severe damage, attention is diffuse — the model may be using overall color shift as a proxy for failure.")

    return " ".join(lines)


def lstm_insight(saliency, pred_class, probs, sensor_state):
    state_names   = ['Stable', 'Warning', 'Critical']
    channel_names = ['Temperature', 'Humidity', 'pH', 'Electrochemical', 'Optical']
    predicted     = state_names[pred_class]
    confidence    = probs[pred_class]

    channel_importance = saliency.mean(axis=0)
    top_ch_idx  = channel_importance.argmax()
    top_channel = channel_names[top_ch_idx]

    early = saliency[:17].mean()
    mid   = saliency[17:34].mean()
    late  = saliency[34:].mean()
    time_labels = ['early timesteps (0–17)', 'mid timesteps (17–34)', 'late timesteps (34–50)']
    top_time    = time_labels[np.argmax([early, mid, late])]

    lines = [
        f"Predicted <b>{predicted}</b> with <b>{confidence:.0%} confidence</b>.",
        f"Most influential sensor: <b>{top_channel}</b>. Critical decision region: <b>{top_time}</b>."
    ]

    if sensor_state == 'stable':
        lines.append("Low overall saliency is expected — stable sensors show minimal drift across all channels and timesteps.")
    elif sensor_state == 'warning':
        if 'late' in top_time:
            lines.append("Correct behavior — gradual drift accumulates toward later timesteps, which is where the model focuses for warning detection.")
        else:
            lines.append(f"The model detected early warning signals in the {top_time}, suggesting drift began earlier than expected.")
        lines.append(f"{top_channel} showed the strongest divergence from baseline, driving the warning classification.")
    elif sensor_state == 'critical':
        lines.append(f"The model identified accelerated multi-channel drift and spike events concentrated in {top_time}.")
        lines.append(f"{top_channel} was the dominant indicator, likely due to the sharpest deviation from safe operating range.")

    return " ".join(lines)


def fusion_insight(cnn_out, lstm_out, fusion_out, pred, coating_state, sensor_state):
    STATUS_LABELS = ['All Clear', 'Monitor', 'Alert', 'Critical']
    status     = STATUS_LABELS[pred]
    confidence = fusion_out[pred]
    cnn_class  = ['Good', 'Degraded', 'Failed'][np.argmax(cnn_out)]
    lstm_class = ['Stable', 'Warning', 'Critical'][np.argmax(lstm_out)]
    dominant   = "visual coating assessment (CNN)" if cnn_out.max() > lstm_out.max() else "sensor readings (LSTM)"

    lines = [
        f"Final decision: <b>{status}</b> at <b>{confidence:.0%} confidence</b>.",
        f"The fusion model weighted <b>{dominant}</b> more heavily in this decision.",
        f"CNN said <b>{cnn_class}</b> ({cnn_out.max():.0%}) — LSTM said <b>{lstm_class}</b> ({lstm_out.max():.0%})."
    ]

    if pred == 0:
        lines.append("Both modalities agree the instrument is in good condition. All Clear is only triggered when visual and sensor signals are simultaneously clean.")
    elif pred == 1:
        lines.append("One or both signals show early concern. Monitor status prompts a scheduled inspection without immediate action.")
    elif pred == 2:
        lines.append("A meaningful disagreement or single severe signal detected — one modality sees a problem the other may not reflect. Alert requires prompt inspection before next use.")
    elif pred == 3:
        lines.append("Both coating and sensor data indicate critical failure. This instrument should be removed from service immediately.")

    return " ".join(lines)


# ─────────────────────────────────────────────
# SHAP FOR FUSION
# ─────────────────────────────────────────────
FEATURE_NAMES  = ['CNN: Good', 'CNN: Degraded', 'CNN: Failed',
                  'LSTM: Stable', 'LSTM: Warning', 'LSTM: Critical']
STATUS_LABELS  = ['All Clear', 'Monitor', 'Alert', 'Critical']


def compute_shap_single(fusion_model, cnn_out, lstm_out, n_background=40):
    """
    Compute SHAP values for a single sample using a small random background.
    Returns shap_values list (one array per output class) and the input vector.
    """
    try:
        import shap
    except ImportError:
        return None, None

    x_single    = np.concatenate([cnn_out, lstm_out])[np.newaxis].astype(np.float32)

    # Small synthetic background centered around neutral probabilities
    rng        = np.random.RandomState(42)
    background = rng.dirichlet(np.ones(3), size=n_background).astype(np.float32)
    bg_cnn     = background
    bg_lstm    = rng.dirichlet(np.ones(3), size=n_background).astype(np.float32)
    bg         = np.concatenate([bg_cnn, bg_lstm], axis=1).astype(np.float32)

    def model_predict(x):
        return fusion_model.predict(x.astype(np.float32), verbose=0)

    explainer   = shap.KernelExplainer(model_predict, bg)
    shap_values = explainer.shap_values(x_single, nsamples=80)

    # Debug — remove after confirming shape
    import sys
    if isinstance(shap_values, list):
        print(f"[SHAP] list of {len(shap_values)} arrays, shapes: {[s.shape for s in shap_values]}", file=sys.stderr)
    else:
        print(f"[SHAP] single array shape: {np.array(shap_values).shape}", file=sys.stderr)

    return shap_values, x_single


def shap_insight(shap_values, pred):
    """Generate plain-English insight from SHAP values for the predicted class."""
    if shap_values is None:
        return "SHAP analysis unavailable — install the <b>shap</b> package to enable this."

    # Handle different output shapes same as shap_chart
    if isinstance(shap_values, list):
        idx = min(pred, len(shap_values) - 1)
        raw = shap_values[idx]
        sv  = raw[0] if raw.ndim == 2 else raw
    elif isinstance(shap_values, np.ndarray) and shap_values.ndim == 3:
        idx = min(pred, shap_values.shape[2] - 1)
        sv  = shap_values[0, :, idx]
    else:
        sv = shap_values[0] if shap_values.ndim == 2 else shap_values
    mean_abs    = np.abs(sv)
    sorted_idx  = np.argsort(mean_abs)[::-1]
    top_feat    = FEATURE_NAMES[sorted_idx[0]]
    top_dir     = "towards" if sv[sorted_idx[0]] > 0 else "away from"
    second_feat = FEATURE_NAMES[sorted_idx[1]]

    cnn_influence  = mean_abs[:3].sum()
    lstm_influence = mean_abs[3:].sum()
    total          = cnn_influence + lstm_influence + 1e-9
    cnn_pct        = cnn_influence / total * 100
    lstm_pct       = lstm_influence / total * 100
    dominant       = "CNN (visual)" if cnn_influence > lstm_influence else "LSTM (sensor)"

    lines = [
        f"SHAP attribution for <b>{STATUS_LABELS[pred]}</b> prediction:",
        f"Dominant modality: <b>{dominant}</b> ({cnn_pct:.0f}% CNN / {lstm_pct:.0f}% LSTM influence).",
        f"Strongest feature: <b>{top_feat}</b> pushed the decision <b>{top_dir}</b> {STATUS_LABELS[pred]}.",
        f"Second strongest: <b>{second_feat}</b>."
    ]

    if pred == 0:
        lines.append("For All Clear, both CNN: Good and LSTM: Stable should be the dominant positive features — any deviation from this suggests the model is compensating.")
    elif pred == 1:
        lines.append("Monitor decisions are typically driven by one modality showing mild degradation while the other remains near normal.")
    elif pred == 2:
        lines.append("Alert is often caused by a mismatch between modalities — one signal looks bad while the other hasn't fully deteriorated yet.")
    elif pred == 3:
        lines.append("Critical decisions should be driven by CNN: Failed and LSTM: Critical together. If only one dominates, the other signal may be lagging behind the actual condition.")

    return " ".join(lines)