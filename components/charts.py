import numpy as np
import cv2
import matplotlib.pyplot as plt
from components.explainability import overlay_gradcam

BG = '#0d1520'

CH_NAMES  = ['Temp', 'Humidity', 'pH', 'Electrochem', 'Optical']
CH_COLORS = ['#4fc3f7', '#4ade80', '#fb923c', '#c084fc', '#f87171']
CNN_COLORS  = ['#4ade80', '#fb923c', '#f87171']
LSTM_COLORS = ['#4ade80', '#fb923c', '#f87171']
FUSION_COLORS = ['#4ade80', '#a3e635', '#fb923c', '#f87171']


def dark_fig(w, h):
    fig, ax = plt.subplots(figsize=(w, h))
    fig.patch.set_facecolor(BG)
    ax.set_facecolor(BG)
    return fig, ax


def prob_chart(probs, labels, colors, title):
    fig, ax = dark_fig(4, 2.6)
    bars = ax.bar(labels, probs, color=colors, width=0.5, zorder=3)
    for bar, val in zip(bars, probs):
        ax.text(
            bar.get_x() + bar.get_width()/2, bar.get_height() + 0.025,
            f'{val:.2f}', ha='center', color='#4a6a88', fontsize=8, fontfamily='monospace'
        )
    ax.set_ylim(0, 1.2)
    ax.set_title(title, color='#2a4a6b', fontsize=8, fontfamily='monospace', pad=8, loc='left')
    ax.tick_params(colors='#2a4a6b', labelsize=7)
    ax.spines[['top', 'right', 'left']].set_visible(False)
    ax.spines['bottom'].set_color('#1a2332')
    ax.yaxis.set_visible(False)
    for lbl in ax.get_xticklabels():
        lbl.set_fontfamily('monospace')
        lbl.set_color('#4a6a88')
    fig.tight_layout(pad=1)
    return fig


def sensor_chart(sensor_seq):
    fig, ax = plt.subplots(figsize=(7, 2.6))
    fig.patch.set_facecolor(BG)
    ax.set_facecolor(BG)
    for i, (name, color) in enumerate(zip(CH_NAMES, CH_COLORS)):
        ax.plot(sensor_seq[:, i], label=name, color=color, linewidth=1.5, alpha=0.9)
    ax.set_ylim(-0.05, 1.2)
    ax.set_title('SENSOR CHANNELS — 50 TIMESTEPS', color='#2a4a6b', fontsize=7,
                 fontfamily='monospace', pad=8, loc='left')
    ax.tick_params(colors='#2a4a6b', labelsize=7)
    ax.spines[['top', 'right']].set_visible(False)
    ax.spines[['bottom', 'left']].set_color('#1a2332')
    legend = ax.legend(
        fontsize=7,
        loc='upper left',
        bbox_to_anchor=(1.01, 1),
        borderaxespad=0,
        frameon=True,
        framealpha=0.6,
        facecolor='#0b0f14',
        edgecolor='#1a2332',
        labelcolor='#8ba5c0'
    )
    for t in legend.get_texts():
        t.set_fontfamily('monospace')
    fig.subplots_adjust(right=0.82)
    return fig


def fusion_chart(fusion_out, pred):
    labels = ['All Clear', 'Monitor', 'Alert', 'Critical']
    fig, ax = dark_fig(4, 2.6)
    bars = ax.bar(labels, fusion_out, color=FUSION_COLORS, width=0.5, zorder=3)
    bars[pred].set_edgecolor('white')
    bars[pred].set_linewidth(2)
    for bar, val in zip(bars, fusion_out):
        ax.text(
            bar.get_x() + bar.get_width()/2, bar.get_height() + 0.025,
            f'{val:.2f}', ha='center', color='#4a6a88', fontsize=8, fontfamily='monospace'
        )
    ax.set_ylim(0, 1.2)
    ax.set_title('FUSION OUTPUT', color='#2a4a6b', fontsize=8, fontfamily='monospace', pad=8, loc='left')
    ax.tick_params(colors='#2a4a6b', labelsize=7)
    ax.spines[['top', 'right', 'left']].set_visible(False)
    ax.spines['bottom'].set_color('#1a2332')
    ax.yaxis.set_visible(False)
    for lbl in ax.get_xticklabels():
        lbl.set_fontfamily('monospace')
        lbl.set_color('#4a6a88')
    fig.tight_layout(pad=1)
    return fig


def gradcam_chart(img_bgr, heatmap):
    overlaid     = overlay_gradcam(img_bgr, heatmap)
    overlaid_rgb = cv2.cvtColor(overlaid, cv2.COLOR_BGR2RGB)
    fig, axes    = plt.subplots(1, 2, figsize=(4.5, 2.2))
    fig.patch.set_facecolor(BG)
    for ax in axes:
        ax.set_facecolor(BG)
    axes[0].imshow(cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB))
    axes[0].set_title('Original', color='#2a4a6b', fontsize=7, fontfamily='monospace')
    axes[0].axis('off')
    axes[1].imshow(overlaid_rgb)
    axes[1].set_title('Grad-CAM', color='#2a4a6b', fontsize=7, fontfamily='monospace')
    axes[1].axis('off')
    fig.tight_layout(pad=0.3)
    return fig


def shap_chart(shap_values, pred):
    """Bar chart of SHAP values for the predicted class."""
    FEATURE_NAMES = ['CNN: Good', 'CNN: Degraded', 'CNN: Failed',
                     'LSTM: Stable', 'LSTM: Warning', 'LSTM: Critical']
    STATUS_LABELS = ['All Clear', 'Monitor', 'Alert', 'Critical']

    # shap_values can be:
    #   list of 4 arrays (one per class) -> shape (n_samples, n_features) each
    #   single 2D array -> shape (n_samples, n_features)
    #   single 3D array -> shape (n_samples, n_features, n_classes)
    if isinstance(shap_values, list):
        idx = min(pred, len(shap_values) - 1)
        raw = shap_values[idx]
        sv  = raw[0] if raw.ndim == 2 else raw
    elif isinstance(shap_values, np.ndarray) and shap_values.ndim == 3:
        # shape (n_samples, n_features, n_classes)
        idx = min(pred, shap_values.shape[2] - 1)
        sv  = shap_values[0, :, idx]
    else:
        sv = shap_values[0] if shap_values.ndim == 2 else shap_values
    sorted_idx = np.argsort(np.abs(sv))[::-1]
    vals       = sv[sorted_idx]
    labels     = [FEATURE_NAMES[i] for i in sorted_idx]
    colors     = ['#4ade80' if v > 0 else '#f87171' for v in vals]

    fig, ax = plt.subplots(figsize=(4.5, 3))
    fig.patch.set_facecolor(BG)
    ax.set_facecolor(BG)

    bars = ax.barh(labels[::-1], vals[::-1], color=colors[::-1], height=0.55)
    ax.axvline(x=0, color='#2a4a6b', linewidth=0.8)

    # Value labels inside bars to avoid clipping
    for bar, val in zip(bars, vals[::-1]):
        if abs(val) > 0.005:
            x_pos = val / 2
            ax.text(x_pos, bar.get_y() + bar.get_height()/2,
                    f'{val:+.3f}', va='center', ha='center',
                    color='#ffffff', fontsize=6.5, fontfamily='monospace',
                    fontweight='bold')

    ax.set_title(f'SHAP VALUES — {STATUS_LABELS[pred].upper()} CLASS',
                 color='#2a4a6b', fontsize=7, fontfamily='monospace', pad=8, loc='left')
    ax.tick_params(colors='#4a6a88', labelsize=7)
    ax.spines[['top', 'right', 'bottom']].set_visible(False)
    ax.spines['left'].set_color('#1a2332')
    for lbl in ax.get_yticklabels():
        lbl.set_fontfamily('monospace')
        lbl.set_color('#8ba5c0')
    ax.set_xlabel('← pushes away   |   pushes toward →',
                  color='#1a3050', fontsize=6, fontfamily='monospace')

    fig.tight_layout(pad=1.2)
    return fig


def saliency_chart(saliency, sensor_seq):
    fig, axes = plt.subplots(2, 1, figsize=(4.5, 3.8))
    fig.patch.set_facecolor(BG)
    for ax in axes:
        ax.set_facecolor(BG)

    for i, (name, color) in enumerate(zip(CH_NAMES, CH_COLORS)):
        axes[0].plot(sensor_seq[:, i], label=name, color=color, linewidth=1.5)
    axes[0].set_title('RAW SENSOR READING', color='#2a4a6b', fontsize=7,
                      fontfamily='monospace', pad=6, loc='left')
    axes[0].set_ylim(-0.05, 1.2)
    legend = axes[0].legend(
        fontsize=7,
        loc='upper left',
        bbox_to_anchor=(1.01, 1),
        borderaxespad=0,
        frameon=True,
        framealpha=0.6,
        facecolor='#0b0f14',
        edgecolor='#1a2332',
        labelcolor='#8ba5c0'
    )
    for t in legend.get_texts():
        t.set_fontfamily('monospace')
    axes[0].tick_params(colors='#2a4a6b', labelsize=7)
    axes[0].spines[['top', 'right']].set_visible(False)
    axes[0].spines[['bottom', 'left']].set_color('#1a2332')

    im = axes[1].imshow(saliency.T, aspect='auto', cmap='hot', interpolation='nearest')
    axes[1].set_yticks(range(5))
    axes[1].set_yticklabels(CH_NAMES, fontsize=8, color='#4a6a88', fontfamily='monospace')
    axes[1].set_xlabel('Timestep', color='#2a4a6b', fontsize=7, fontfamily='monospace')
    axes[1].set_title('SALIENCY MAP — BRIGHTER = MORE IMPORTANT', color='#2a4a6b',
                      fontsize=7, fontfamily='monospace', pad=6, loc='left')
    axes[1].tick_params(colors='#2a4a6b', labelsize=7)

    # Use same fraction/pad as legend space so both plots align
    cbar = fig.colorbar(im, ax=axes[1], fraction=0.046, pad=0.01)
    cbar.ax.tick_params(colors='#2a4a6b', labelsize=6)
    cbar.ax.yaxis.set_tick_params(width=0.5)

    fig.subplots_adjust(right=0.82, hspace=0.4)
    return fig