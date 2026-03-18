import streamlit as st
import numpy as np
import cv2
import matplotlib.pyplot as plt
import random

from components.generators import generate_coating_image, generate_sensor_reading
from components.models import load_models, run_inference
from components.explainability import (
    get_gradcam_heatmap, get_lstm_saliency,
    gradcam_insight, lstm_insight, fusion_insight,
    compute_shap_single, shap_insight
)
from components.charts import (
    prob_chart, sensor_chart, fusion_chart,
    gradcam_chart, saliency_chart, shap_chart, CNN_COLORS, LSTM_COLORS
)

# ─────────────────────────────────────────────
# PAGE CONFIG
# ─────────────────────────────────────────────
st.set_page_config(
    page_title="CoatingSense",
    page_icon="🔬",
    layout="wide",
    initial_sidebar_state="collapsed"
)

# ─────────────────────────────────────────────
# STYLING
# ─────────────────────────────────────────────
st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=Space+Mono:wght@400;700&family=DM+Sans:wght@300;400;500;600&display=swap');

html, body, [class*="css"] {
    font-family: 'DM Sans', sans-serif;
    background-color: #080c10;
    color: #b8c5d0;
}
.stApp { background-color: #080c10; }

section[data-testid="stSidebar"] { display: none; }
button[data-testid="collapsedControl"] { display: none; }

.stTabs [data-baseweb="tab-list"] {
    background: #0b0f14;
    border-bottom: 1px solid #1a2332;
    padding: 0 32px;
    gap: 0;
}
.stTabs [data-baseweb="tab"] {
    font-family: 'Space Mono', monospace;
    font-size: 0.68rem;
    letter-spacing: 2px;
    text-transform: uppercase;
    color: #2a4a6b;
    padding: 14px 24px;
    border-bottom: 2px solid transparent;
    background: transparent;
}
.stTabs [aria-selected="true"] {
    color: #4fc3f7 !important;
    border-bottom: 2px solid #4fc3f7 !important;
    background: transparent !important;
}

.block-container { padding: 0 2rem 2rem 2rem !important; max-width: 1400px; }

.banner {
    padding: 16px 24px;
    border-radius: 4px;
    font-family: 'Space Mono', monospace;
    font-size: 1rem;
    font-weight: 700;
    letter-spacing: 1px;
    text-align: center;
    margin-bottom: 24px;
}
.banner-allclear { background:#0a2818; border:1px solid #1a5c38; color:#4ade80; }
.banner-monitor  { background:#1a2408; border:1px solid #3d5c10; color:#a3e635; }
.banner-alert    { background:#2a1a04; border:1px solid #6b3d0a; color:#fb923c; }
.banner-critical { background:#2a0808; border:1px solid #6b1414; color:#f87171; }

.card {
    background: #0d1520;
    border: 1px solid #1a2332;
    border-radius: 6px;
    padding: 16px 18px;
    margin-bottom: 10px;
}
.card-label {
    font-family: 'Space Mono', monospace;
    font-size: 0.58rem;
    color: #2a4a6b;
    letter-spacing: 3px;
    text-transform: uppercase;
    margin-bottom: 6px;
}
.card-value {
    font-family: 'Space Mono', monospace;
    font-size: 0.88rem;
    color: #e2eaf0;
    line-height: 1.5;
}

.section-label {
    font-family: 'Space Mono', monospace;
    font-size: 0.6rem;
    color: #2a4a6b;
    letter-spacing: 3px;
    text-transform: uppercase;
    border-bottom: 1px solid #1a2332;
    padding-bottom: 6px;
    margin-bottom: 14px;
    margin-top: 20px;
}

.insight-block {
    background: #0d1520;
    border-left: 3px solid #4fc3f7;
    border-radius: 0 6px 6px 0;
    padding: 14px 18px;
    margin-bottom: 10px;
    font-size: 0.88rem;
    line-height: 1.65;
    color: #8ba5c0;
}
.insight-block.good { border-left-color: #4ade80; }
.insight-block.warn { border-left-color: #fb923c; }
.insight-block.crit { border-left-color: #f87171; }
.insight-block.info { border-left-color: #4fc3f7; }

.insight-title {
    font-family: 'Space Mono', monospace;
    font-size: 0.68rem;
    letter-spacing: 2px;
    text-transform: uppercase;
    margin-bottom: 8px;
    color: #4fc3f7;
}
.insight-title.good { color: #4ade80; }
.insight-title.warn { color: #fb923c; }
.insight-title.crit { color: #f87171; }

.scenario-pill {
    display: inline-block;
    font-family: 'Space Mono', monospace;
    font-size: 0.7rem;
    padding: 4px 12px;
    border-radius: 20px;
    letter-spacing: 1px;
    text-transform: uppercase;
    margin-bottom: 16px;
}
.pill-allclear { background:#0a2818; color:#4ade80; border:1px solid #1a5c38; }
.pill-monitor  { background:#1a2408; color:#a3e635; border:1px solid #3d5c10; }
.pill-alert    { background:#2a1a04; color:#fb923c; border:1px solid #6b3d0a; }
.pill-critical { background:#2a0808; color:#f87171; border:1px solid #6b1414; }

.stButton > button {
    background: linear-gradient(135deg, #0d4f8c, #0a7ab5);
    color: #e2eaf0;
    border: 1px solid #1a6fa8;
    border-radius: 4px;
    font-family: 'Space Mono', monospace;
    font-size: 0.82rem;
    font-weight: 700;
    letter-spacing: 2px;
    padding: 14px 32px;
    text-transform: uppercase;
    transition: all 0.2s;
    min-width: 220px;
}
.stButton > button:hover {
    background: linear-gradient(135deg, #1060a8, #0d8fd4);
    border-color: #2a8fd4;
}

.feature-card {
    background: #0d1520;
    border: 1px solid #1a2332;
    border-radius: 8px;
    padding: 20px 22px;
    height: 100%;
}
.feature-icon  { font-size: 1.4rem; margin-bottom: 10px; }
.feature-title {
    font-family: 'Space Mono', monospace;
    font-size: 0.72rem;
    letter-spacing: 2px;
    text-transform: uppercase;
    color: #4fc3f7;
    margin-bottom: 8px;
}
.feature-desc { font-size: 0.85rem; color: #4a6a88; line-height: 1.6; }

#MainMenu {visibility: hidden;}
footer {visibility: hidden;}
header {visibility: hidden;}
</style>
""", unsafe_allow_html=True)


# ─────────────────────────────────────────────
# CONSTANTS
# ─────────────────────────────────────────────
SCENARIOS = ['all_clear', 'monitor', 'alert', 'critical']
SCENARIO_MAP = {
    'all_clear': ('good',     'stable'),
    'monitor':   ('degraded', 'stable'),
    'alert':     ('degraded', 'warning'),
    'critical':  ('failed',   'critical'),
}
STATUS_LABELS = {0: 'All Clear', 1: 'Monitor', 2: 'Alert', 3: 'Critical'}
STATUS_BANNER = {0: 'banner-allclear', 1: 'banner-monitor', 2: 'banner-alert', 3: 'banner-critical'}
STATUS_PILL   = {0: 'pill-allclear', 1: 'pill-monitor', 2: 'pill-alert', 3: 'pill-critical'}
STATUS_EMOJI  = {0: '🟢', 1: '🟡', 2: '🟠', 3: '🔴'}
INSIGHT_CLASS = {0: 'good', 1: 'info', 2: 'warn', 3: 'crit'}
REC_COLOR     = {'all_clear': 'good', 'monitor': 'info', 'alert': 'warn', 'critical': 'crit'}
CNN_LABEL_MAP  = ['Good', 'Degraded', 'Failed']
LSTM_LABEL_MAP = ['Stable', 'Warning', 'Critical']
RECOMMENDATIONS = {
    'all_clear': "Instrument coating is intact and storage conditions are within normal parameters. No action required. Continue standard monitoring schedule.",
    'monitor':   "Early signs of wear or environmental stress detected. Schedule an inspection within the next maintenance cycle. Do not escalate unless readings worsen.",
    'alert':     "Significant coating degradation or sensor anomaly detected. Inspect this instrument before next use. Do not use until cleared by a qualified technician.",
    'critical':  "Severe coating failure with critical sensor readings confirmed. Remove instrument from service immediately. Send for reconditioning or replacement.",
}


# ─────────────────────────────────────────────
# SESSION STATE INIT
# ─────────────────────────────────────────────
if 'result' not in st.session_state:
    st.session_state.result = None
if 'active_tab' not in st.session_state:
    st.session_state.active_tab = 0
if 'show_toast' not in st.session_state:
    st.session_state.show_toast = False
if 'toast_scenario' not in st.session_state:
    st.session_state.toast_scenario = ''
if 'toast_status' not in st.session_state:
    st.session_state.toast_status = ''
if 'toast_emoji' not in st.session_state:
    st.session_state.toast_emoji = ''


# ─────────────────────────────────────────────
# GENERATE FUNCTION - runs before tabs render
# ─────────────────────────────────────────────
def run_analysis():
    try:
        cnn_model, lstm_model, fusion_model = load_models()
    except Exception as e:
        st.error(f"Could not load models from `models/`. Make sure all three .keras files are present.\n\n`{e}`")
        return False

    scenario                    = random.choice(SCENARIOS)
    coating_state, sensor_state = SCENARIO_MAP[scenario]
    img                         = generate_coating_image(coating_state)
    sensor_seq                  = generate_sensor_reading(50, state=sensor_state)
    img_rgb                     = cv2.cvtColor(img, cv2.COLOR_BGR2RGB).astype(np.float32)

    cnn_out, lstm_out, fusion_out, pred = run_inference(
        cnn_model, lstm_model, fusion_model, img_rgb, sensor_seq
    )

    try:
        heatmap, _, _ = get_gradcam_heatmap(cnn_model, np.expand_dims(img_rgb, 0))
        gradcam_ok = True
    except Exception:
        heatmap, gradcam_ok = None, False

    saliency, _, _ = get_lstm_saliency(lstm_model, sensor_seq)

    # SHAP for fusion - runs fast with small background
    shap_values, _ = compute_shap_single(fusion_model, cnn_out, lstm_out, n_background=40)

    st.session_state.result = {
        'scenario':      scenario,
        'coating_state': coating_state,
        'sensor_state':  sensor_state,
        'img':           img,
        'sensor_seq':    sensor_seq,
        'cnn_out':       cnn_out,
        'lstm_out':      lstm_out,
        'fusion_out':    fusion_out,
        'pred':          pred,
        'heatmap':       heatmap,
        'gradcam_ok':    gradcam_ok,
        'saliency':      saliency,
        'shap_values':   shap_values,
    }
    st.session_state.active_tab    = 1
    st.session_state.show_toast    = True
    st.session_state.toast_scenario = scenario
    st.session_state.toast_status   = STATUS_LABELS[pred]
    st.session_state.toast_emoji    = STATUS_EMOJI[pred]
    return True


# ─────────────────────────────────────────────
# TABS
# ─────────────────────────────────────────────
tab_home, tab_results, tab_explain = st.tabs([
    "  🔬  CoatingSense  ",
    "  📊  Model Results  ",
    "  💡  Explainability  "
])


# ══════════════════════════════════════════════
# TAB 1 - HOME
# ══════════════════════════════════════════════
with tab_home:
    st.markdown("<br>", unsafe_allow_html=True)

    col_hero, col_spacer, col_btn = st.columns([3, 0.3, 1])
    with col_hero:
        st.markdown("""
        <div style="font-family:'Space Mono',monospace; font-size:2rem; font-weight:700;
                    color:#4fc3f7; letter-spacing:-1px; line-height:1.1;">
            CoatingSense
        </div>
        <div style="font-family:'Space Mono',monospace; font-size:0.65rem; color:#2a4a6b;
                    letter-spacing:4px; text-transform:uppercase; margin-top:6px; margin-bottom:20px;">
            Surgical Instrument Coating Monitor
        </div>
        <div style="font-size:1rem; color:#4a6a88; line-height:1.8; max-width:1000px;">
            An AI system that monitors <b style="color:#b8c5d0;">self-healing coatings</b> on surgical instruments
            by combining computer vision with environmental sensor data.
            The system fuses both signals into a single actionable status alert.
        </div>
        """, unsafe_allow_html=True)

    with col_btn:
        st.markdown("<br><br><br>", unsafe_allow_html=True)
        if st.button("⟳  Analyse New Sample"):
            with st.spinner("Generating sample and running inference..."):
                run_analysis()
            st.rerun()

    # Toast notification after generation
    if st.session_state.show_toast:
        st.session_state.show_toast = False
        scenario_display = st.session_state.toast_scenario.replace('_', ' ').title()
        st.markdown(f"""
        <div style="
            background: #0d1e14;
            border: 1px solid #1a5c38;
            border-left: 4px solid #4ade80;
            border-radius: 6px;
            padding: 14px 20px;
            margin: 16px 0;
            font-family: 'Space Mono', monospace;
            font-size: 0.78rem;
            color: #4ade80;
            display: flex;
            align-items: center;
            gap: 12px;
        ">
            ✓ &nbsp; Sample generated - <b>{scenario_display}</b> scenario &nbsp;·&nbsp;
            Result: <b>{st.session_state.toast_emoji} {st.session_state.toast_status}</b>
            &nbsp;·&nbsp;
            <span style="color:#2a6a48;">Switch to the Model Results tab to view →</span>
        </div>
        """, unsafe_allow_html=True)

    st.markdown("<br>", unsafe_allow_html=True)
    st.markdown('<div class="section-label">How it works</div>', unsafe_allow_html=True)

    f1, f2, f3, f4 = st.columns(4)
    features = [
        ("🖼️", "CNN - Coating Vision",
         "MobileNetV2 analyses a coating image and classifies it as Good, Degraded, or Failed based on visual defect patterns."),
        ("📡", "LSTM - Sensor Monitor",
         "A bidirectional LSTM reads 5-channel environmental sensor data over 50 timesteps and classifies storage conditions as Stable, Warning, or Critical."),
        ("🔀", "Fusion - Decision",
         "A lightweight MLP combines CNN and LSTM probability outputs into a final 4-class system status: All Clear, Monitor, Alert, or Critical."),
        ("🔍", "Explainability",
         "Grad-CAM heatmaps show where the CNN looked. Saliency maps show which sensor channels and timesteps drove the LSTM decision."),
    ]
    for col, (icon, title, desc) in zip([f1, f2, f3, f4], features):
        with col:
            st.markdown(f"""
            <div class="feature-card">
                <div class="feature-icon">{icon}</div>
                <div class="feature-title">{title}</div>
                <div class="feature-desc">{desc}</div>
            </div>
            """, unsafe_allow_html=True)

    st.markdown("<br>", unsafe_allow_html=True)
    st.markdown('<div class="section-label">Alert Levels</div>', unsafe_allow_html=True)

    a1, a2, a3, a4 = st.columns(4)
    alerts = [
        ('banner-allclear', '🟢', 'All Clear',  'Good coating + Stable sensors. No action needed.'),
        ('banner-monitor',  '🟡', 'Monitor',    'Early warning signs. Schedule inspection.'),
        ('banner-alert',    '🟠', 'Alert',      'Significant degradation. Inspect before next use.'),
        ('banner-critical', '🔴', 'Critical',   'Severe failure. Remove from service immediately.'),
    ]
    for col, (css, emoji, label, desc) in zip([a1, a2, a3, a4], alerts):
        with col:
            st.markdown(f"""
            <div class="banner {css}" style="margin-bottom:8px; font-size:0.85rem;">
                {emoji} {label}
            </div>
            <div style="font-size:0.8rem; color:#2a4a6b; text-align:center;
                        font-family:'Space Mono',monospace; line-height:1.5;">
                {desc}
            </div>
            """, unsafe_allow_html=True)

    st.markdown("<br>", unsafe_allow_html=True)
    st.markdown('<div class="section-label">Patent Context</div>', unsafe_allow_html=True)
    st.markdown("""
    <div style="font-size:0.85rem; color:#4a6a88; line-height:1.8; max-width:1500px;">
        This prototype supports patent claims around
        <b style="color:#b8c5d0;">automated coating integrity monitoring</b> using computer vision,
        <b style="color:#b8c5d0;">multi-modal sensor fusion</b> combining visual and environmental data,
        and <b style="color:#b8c5d0;">predictive maintenance scheduling</b> based on real-time assessment.
        Neither visual inspection nor sensor monitoring alone is sufficient -
        the fusion model catches scenarios that either individual model would miss.
    </div>
    """, unsafe_allow_html=True)


# ══════════════════════════════════════════════
# TAB 2 - MODEL RESULTS
# ══════════════════════════════════════════════
with tab_results:
    r = st.session_state.result

    if r is None:
        st.markdown("""
        <div style="height:50vh; display:flex; flex-direction:column; align-items:center;
                    justify-content:center; color:#1a3050; font-family:'Space Mono',monospace;
                    text-align:center; gap:14px;">
            <div style="font-size:2rem;">📊</div>
            <div style="font-size:0.65rem; letter-spacing:3px; text-transform:uppercase;">
                No results yet - go to CoatingSense tab and click Analyse New Sample
            </div>
        </div>
        """, unsafe_allow_html=True)
    else:
        pred = r['pred']

        st.markdown("<br>", unsafe_allow_html=True)
        st.markdown(f"""
        <div class="banner {STATUS_BANNER[pred]}">
            {STATUS_EMOJI[pred]} &nbsp; SYSTEM STATUS: {STATUS_LABELS[pred].upper()} &nbsp;·&nbsp; {r['fusion_out'][pred]:.0%} CONFIDENCE
        </div>
        <span class="scenario-pill {STATUS_PILL[pred]}">Scenario: {r['scenario'].replace('_', ' ')}</span>
        """, unsafe_allow_html=True)

        col_img, col_sensor = st.columns([1, 2])
        with col_img:
            st.markdown('<div class="section-label">Coating Image</div>', unsafe_allow_html=True)
            st.image(cv2.cvtColor(r['img'], cv2.COLOR_BGR2RGB), use_container_width=True)
            st.markdown(f"""
            <div class="card">
                <div class="card-label">Coating State</div>
                <div class="card-value">{r['coating_state'].capitalize()}</div>
            </div>""", unsafe_allow_html=True)

        with col_sensor:
            st.markdown('<div class="section-label">Sensor Reading</div>', unsafe_allow_html=True)
            sfig = sensor_chart(r['sensor_seq'])
            st.pyplot(sfig, use_container_width=True)
            plt.close(sfig)
            st.markdown(f"""
            <div class="card">
                <div class="card-label">Sensor State</div>
                <div class="card-value">{r['sensor_state'].capitalize()}</div>
            </div>""", unsafe_allow_html=True)

        st.markdown('<div class="section-label">Model Predictions</div>', unsafe_allow_html=True)
        c1, c2, c3 = st.columns(3)

        with c1:
            fig = prob_chart(r['cnn_out'], CNN_LABEL_MAP, CNN_COLORS, 'CNN - Coating Quality')
            st.pyplot(fig, use_container_width=True); plt.close(fig)
            st.markdown(f"""<div class="card"><div class="card-label">CNN Prediction</div>
            <div class="card-value">{CNN_LABEL_MAP[np.argmax(r['cnn_out'])]} · {r['cnn_out'].max():.0%}</div>
            </div>""", unsafe_allow_html=True)

        with c2:
            fig = prob_chart(r['lstm_out'], LSTM_LABEL_MAP, LSTM_COLORS, 'LSTM - Sensor Condition')
            st.pyplot(fig, use_container_width=True); plt.close(fig)
            st.markdown(f"""<div class="card"><div class="card-label">LSTM Prediction</div>
            <div class="card-value">{LSTM_LABEL_MAP[np.argmax(r['lstm_out'])]} · {r['lstm_out'].max():.0%}</div>
            </div>""", unsafe_allow_html=True)

        with c3:
            fig = fusion_chart(r['fusion_out'], pred)
            st.pyplot(fig, use_container_width=True); plt.close(fig)
            st.markdown(f"""<div class="card"><div class="card-label">Fusion Decision</div>
            <div class="card-value">{STATUS_LABELS[pred]} · {r['fusion_out'][pred]:.0%}</div>
            </div>""", unsafe_allow_html=True)

        st.markdown("<br>", unsafe_allow_html=True)
        col_again, _ = st.columns([1, 3])
        with col_again:
            if st.button("⟳  Generate Another Sample", key="btn_results"):
                with st.spinner("Generating new sample..."):
                    run_analysis()
                st.rerun()


# ══════════════════════════════════════════════
# TAB 3 - EXPLAINABILITY
# ══════════════════════════════════════════════
with tab_explain:
    r = st.session_state.result

    if r is None:
        st.markdown("""
        <div style="height:50vh; display:flex; flex-direction:column; align-items:center;
                    justify-content:center; color:#1a3050; font-family:'Space Mono',monospace;
                    text-align:center; gap:14px;">
            <div style="font-size:2rem;">💡</div>
            <div style="font-size:0.65rem; letter-spacing:3px; text-transform:uppercase;">
                No results yet - go to CoatingSense tab and click Analyse New Sample
            </div>
        </div>
        """, unsafe_allow_html=True)
    else:
        pred = r['pred']
        ic   = INSIGHT_CLASS[pred]

        st.markdown("<br>", unsafe_allow_html=True)

        # ── GRAD-CAM + CNN INSIGHT ──
        st.markdown('<div class="section-label">Grad-CAM - Where the CNN looked on the coating image</div>',
                    unsafe_allow_html=True)
        gc_col, cnn_insight_col = st.columns([1, 1])

        with gc_col:
            if r['gradcam_ok']:
                gcfig = gradcam_chart(r['img'], r['heatmap'])
                st.pyplot(gcfig, use_container_width=True)
                plt.close(gcfig)
                st.markdown("""<div style="font-size:0.72rem; color:#2a4a6b; font-family:'Space Mono',monospace; margin-top:4px;">
                Red/yellow = high attention &nbsp;·&nbsp; Blue = low attention
                </div>""", unsafe_allow_html=True)
            else:
                st.warning("Grad-CAM could not be computed for this sample.")

        with cnn_insight_col:
            cnn_text = gradcam_insight(r['heatmap'], int(np.argmax(r['cnn_out'])), r['cnn_out'], r['coating_state']) \
                if r['gradcam_ok'] else \
                f"CNN predicted <b>{CNN_LABEL_MAP[np.argmax(r['cnn_out'])]}</b> with <b>{r['cnn_out'].max():.0%} confidence</b>."
            st.markdown(f"""
            <div class="insight-block good" style="height:100%; font-size:1rem;">
                <div class="insight-title good" style="font-size:0.75rem;">CNN - Coating Analysis</div>
                {cnn_text}
            </div>""", unsafe_allow_html=True)

        # ── SALIENCY + LSTM INSIGHT ──
        st.markdown('<div class="section-label" style="margin-top:28px;">Saliency Map - Which sensors and timesteps drove the LSTM</div>',
                    unsafe_allow_html=True)
        sal_col, lstm_insight_col = st.columns([1, 1])

        with sal_col:
            salfig = saliency_chart(r['saliency'], r['sensor_seq'])
            st.pyplot(salfig, use_container_width=True)
            plt.close(salfig)
            st.markdown("""<div style="font-size:0.72rem; color:#2a4a6b; font-family:'Space Mono',monospace; margin-top:4px;">
            Brighter = more important &nbsp;·&nbsp; Rows = channels, Columns = timesteps
            </div>""", unsafe_allow_html=True)

        with lstm_insight_col:
            lstm_text = lstm_insight(r['saliency'], int(np.argmax(r['lstm_out'])), r['lstm_out'], r['sensor_state'])
            st.markdown(f"""
            <div class="insight-block info" style="height:100%; font-size:1rem;">
                <div class="insight-title info" style="font-size:0.75rem;">LSTM - Sensor Analysis</div>
                {lstm_text}
            </div>""", unsafe_allow_html=True)

        # ── SHAP + FUSION INSIGHT ──
        st.markdown('<div class="section-label" style="margin-top:28px;">SHAP - Feature Attribution for Fusion Model</div>',
                    unsafe_allow_html=True)
        shap_col, fusion_insight_col = st.columns([1, 1])

        with shap_col:
            if r.get('shap_values') is not None:
                sfig = shap_chart(r['shap_values'], pred)
                st.pyplot(sfig, use_container_width=True)
                plt.close(sfig)
                st.markdown("""<div style="font-size:0.72rem; color:#2a4a6b; font-family:'Space Mono',monospace; margin-top:4px;">
                Green = pushes toward class &nbsp;·&nbsp; Red = pushes away &nbsp;·&nbsp; Longer = stronger
                </div>""", unsafe_allow_html=True)
            else:
                st.info("Install the `shap` package to enable SHAP attribution: `pip install shap`")

        with fusion_insight_col:
            fusion_text = fusion_insight(
                r['cnn_out'], r['lstm_out'], r['fusion_out'], pred,
                r['coating_state'], r['sensor_state']
            )
            shap_text = shap_insight(r.get('shap_values'), pred)
            st.markdown(f"""
            <div class="insight-block {ic}" style="margin-bottom:10px; font-size:1rem;">
                <div class="insight-title {ic}" style="font-size:0.75rem;">Fusion - Decision Reasoning</div>
                {fusion_text}
            </div>
            <div class="insight-block {ic}" style="font-size:1rem;">
                <div class="insight-title {ic}" style="font-size:0.75rem;">SHAP - Fusion Attribution</div>
                {shap_text}
            </div>""", unsafe_allow_html=True)

        # ── CLINICAL RECOMMENDATION ──
        st.markdown('<div class="section-label" style="margin-top:28px;">Clinical Recommendation</div>',
                    unsafe_allow_html=True)
        st.markdown(f"""
        <div class="insight-block {REC_COLOR[r['scenario']]}" style="font-size:1rem; padding:24px 28px;">
            <div class="insight-title {REC_COLOR[r['scenario']]}" style="font-size:0.75rem; margin-bottom:12px;">Action Required</div>
            {RECOMMENDATIONS[r['scenario']]}
        </div>""", unsafe_allow_html=True)


# ─────────────────────────────────────────────
# AUTO-REDIRECT AFTER GENERATE
# Streamlit can't programmatically switch tabs,
# so we inject a JS click on the results tab
# ─────────────────────────────────────────────
if st.session_state.active_tab == 1:
    st.session_state.active_tab = 0   # reset so it doesn't keep firing
    st.markdown("""
    <script>
    const tabs = window.parent.document.querySelectorAll('[data-baseweb="tab"]');
    if (tabs.length >= 2) { tabs[1].click(); }
    </script>
    """, unsafe_allow_html=True)