"""
MLOmics: Streamlit Demo Application
====================================
Cancer subtype prediction from multi-omics CSV upload.

Launch:  streamlit run app/streamlit_app.py
"""

import json
import os
import sys

import joblib
import matplotlib
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import plotly.graph_objects as go
import shap
import streamlit as st
import streamlit.components.v1 as components
import torch

# ---------------------------------------------------------------------------
# Resolve project root (one level up from app/)
# ---------------------------------------------------------------------------
APP_DIR = os.path.dirname(os.path.abspath(__file__))
PROJECT_ROOT = os.path.dirname(APP_DIR)
sys.path.insert(0, PROJECT_ROOT)

from src.models import IntermediateFusionModel  # noqa: E402

ARTIFACT_DIR = os.path.join(APP_DIR, "model_artifacts")
RESULTS_DIR = os.path.join(PROJECT_ROOT, "results")

# Force non-interactive matplotlib backend (required for Streamlit)
matplotlib.use("Agg")

# Consistent chart color palette
CHART_COLORS = ["#3b82f6", "#06b6d4", "#10b981", "#f59e0b", "#ef4444", "#8b5cf6"]


# =====================================================================
# Cached Loading Functions — loaded once, reused across reruns
# =====================================================================

@st.cache_resource
def load_config_json():
    """Load demo configuration (feature names, class labels, modality dims)."""
    path = os.path.join(ARTIFACT_DIR, "config.json")
    with open(path) as f:
        return json.load(f)


@st.cache_resource
def load_scaler():
    """Load the plain StandardScaler fitted on concatenated training data."""
    return joblib.load(os.path.join(ARTIFACT_DIR, "scaler.pkl"))


@st.cache_resource
def load_xgb_model():
    """Load the best XGBoost model."""
    return joblib.load(os.path.join(ARTIFACT_DIR, "xgb_best.pkl"))


@st.cache_resource
def load_fusion_model():
    """Reconstruct and load the best IntermediateFusion model."""
    ckpt = torch.load(
        os.path.join(ARTIFACT_DIR, "fusion_best.pt"),
        map_location="cpu",
        weights_only=False,
    )
    model = IntermediateFusionModel(
        modality_dims=ckpt["modality_dims"],
        latent_dim=ckpt["latent_dim"],
        num_classes=ckpt["num_classes"],
        encoder_hidden=ckpt["encoder_hidden"],
        hidden_dim=ckpt["classifier_hidden"],
        dropout=ckpt["dropout"],
    )
    model.load_state_dict(ckpt["model_state_dict"])
    model.eval()
    return model


@st.cache_data
def load_model_comparison():
    """Load model comparison metrics CSV."""
    path = os.path.join(RESULTS_DIR, "metrics", "model_comparison.csv")
    if os.path.exists(path):
        return pd.read_csv(path)
    return None


@st.cache_data
def load_sample_csv_bytes():
    """Load sample CSV for download button."""
    path = os.path.join(APP_DIR, "sample_input.csv")
    if os.path.exists(path):
        with open(path, "rb") as f:
            return f.read()
    return None


# =====================================================================
# Helper Functions — Model logic (UNCHANGED)
# =====================================================================

def split_by_modality(X_concat: np.ndarray, cfg: dict) -> dict:
    """Split concatenated feature array into per-modality dict using config."""
    modality_order = cfg["modality_order"]
    modality_dims = cfg["modality_dims"]

    x_dict = {}
    start = 0
    for mod in modality_order:
        if mod not in modality_dims:
            continue
        dim = modality_dims[mod]
        x_dict[mod] = X_concat[:, start:start + dim]
        start += dim

    return x_dict


def predict_xgboost(model, X_scaled: np.ndarray):
    """Run XGBoost prediction. Returns (predicted_class, probabilities)."""
    probs = model.predict_proba(X_scaled)
    pred = np.argmax(probs, axis=1)
    return pred, probs


def predict_fusion(model, X_scaled: np.ndarray, cfg: dict):
    """Run IntermediateFusion prediction. Returns (predicted_class, probabilities)."""
    x_dict = split_by_modality(X_scaled, cfg)
    x_dict_tensor = {
        k: torch.tensor(v, dtype=torch.float32)
        for k, v in x_dict.items()
    }

    with torch.no_grad():
        logits = model(x_dict_tensor)
        probs = torch.softmax(logits, dim=1).numpy()
    pred = np.argmax(probs, axis=1)
    return pred, probs


def compute_shap_explanation(model, X_scaled: np.ndarray, sample_idx: int = 0):
    """Compute TreeSHAP for XGBoost. Returns shap.Explanation for the predicted class."""
    explainer = shap.TreeExplainer(model)
    shap_values = explainer.shap_values(X_scaled)

    pred_class = np.argmax(
        model.predict_proba(X_scaled[sample_idx:sample_idx + 1]), axis=1
    )[0]

    if isinstance(shap_values, list):
        sv = shap_values[pred_class][sample_idx]
    else:
        sv = shap_values[sample_idx, :, pred_class]

    base_val = explainer.expected_value
    if isinstance(base_val, (np.ndarray, list)):
        base_val = base_val[pred_class]

    return shap.Explanation(
        values=sv,
        base_values=float(base_val),
        data=X_scaled[sample_idx],
    )


# =====================================================================
# Plotly Theme Helper
# =====================================================================

def apply_dark_theme(fig):
    """Apply consistent theme to any Plotly figure (works on both light/dark)."""
    fig.update_layout(
        colorway=CHART_COLORS,
        plot_bgcolor="rgba(0,0,0,0)",
        paper_bgcolor="rgba(0,0,0,0)",
        font=dict(family="Inter, sans-serif", color="#64748b", size=12),
        xaxis=dict(
            gridcolor="rgba(128,128,128,0.15)",
            linecolor="rgba(128,128,128,0.15)",
            zerolinecolor="rgba(128,128,128,0.15)",
        ),
        yaxis=dict(
            gridcolor="rgba(128,128,128,0.15)",
            linecolor="rgba(128,128,128,0.15)",
            zerolinecolor="rgba(128,128,128,0.15)",
        ),
        legend=dict(
            bgcolor="rgba(0,0,0,0)",
            bordercolor="rgba(0,0,0,0)",
            font=dict(color="#64748b", size=11),
        ),
        margin=dict(l=50, r=20, t=50, b=50),
    )
    return fig


# =====================================================================
# UI Component Helpers
# =====================================================================

def render_prediction_card(sample_id: str, class_name: str, confidence: float):
    """Render a premium prediction card with gradient top border."""
    tier = "high" if confidence >= 0.8 else "moderate" if confidence >= 0.5 else "low"
    tier_label = {"high": "HIGH CONFIDENCE", "moderate": "MODERATE", "low": "LOW"}[tier]

    st.markdown(
        '<div class="pred-card {tier}">'
        '<div class="sample-id">{sid}</div>'
        '<div class="class-name">{cn}</div>'
        '<div class="confidence-bar-track">'
        '<div class="confidence-bar-fill" style="width:{pct}%;"></div>'
        '</div>'
        '<div class="confidence-text">'
        '<span>{conf}</span>'
        '<span style="font-size:0.72rem;font-weight:500;">{tl}</span>'
        '</div></div>'.format(
            tier=tier, sid=sample_id, cn=class_name,
            pct=int(confidence * 100), conf=f"{confidence:.1%}",
            tl=tier_label,
        ),
        unsafe_allow_html=True,
    )


def render_confidence_chart(probs: np.ndarray, cfg: dict, sample_idx: int = 0):
    """Render a Plotly bar chart showing per-class confidence."""
    class_names = cfg["class_names"]
    p = probs[sample_idx]
    pred_class = np.argmax(p)

    labels = [class_names.get(str(i), f"Class {i}") for i in range(len(p))]
    colors = [
        "#3b82f6" if i == pred_class else "rgba(128,128,128,0.15)"
        for i in range(len(p))
    ]

    fig = go.Figure(
        data=[
            go.Bar(
                x=labels,
                y=p,
                marker_color=colors,
                text=[f"{v:.1%}" for v in p],
                textposition="auto",
                textfont=dict(size=13, family="Inter"),
            )
        ]
    )
    fig.update_layout(
        title=dict(
            text="Prediction Confidence by Subtype",
            font=dict(size=15),
        ),
        yaxis_title="Probability",
        height=380,
    )
    apply_dark_theme(fig)
    fig.update_yaxes(range=[0, 1])
    st.plotly_chart(fig, use_container_width=True)


def render_shap_waterfall(model, X_scaled: np.ndarray, cfg: dict, sample_idx: int = 0):
    """Compute TreeSHAP and render waterfall plot for one sample."""
    explanation = compute_shap_explanation(model, X_scaled, sample_idx)
    explanation.feature_names = cfg["feature_names"]

    shap.plots.waterfall(explanation, max_display=15, show=False)
    fig = plt.gcf()
    fig.patch.set_facecolor("none")
    for a in fig.get_axes():
        a.set_facecolor("none")
        a.tick_params(colors="#64748b")
        a.xaxis.label.set_color("#64748b")
        a.yaxis.label.set_color("#64748b")
        if a.get_title():
            a.title.set_color("#64748b")
        for spine in a.spines.values():
            spine.set_edgecolor("#64748b")
    st.pyplot(fig, clear_figure=True)
    plt.close(fig)

    return explanation


def render_pipeline_diagram():
    """Render pipeline diagram with premium step cards."""
    steps = [
        ("📁", "CSV Upload", "Multi-omics data (features x samples)", "rgba(59,130,246,0.15)", "#3b82f6"),
        ("⚙️", "Preprocessing", "Transpose + StandardScaler normalization", "rgba(139,92,246,0.15)", "#8b5cf6"),
        ("🧠", "Model Inference", "XGBoost or Intermediate Fusion", "rgba(236,72,153,0.15)", "#ec4899"),
        ("🎯", "Prediction", "Subtype classification + confidence", "rgba(16,185,129,0.15)", "#10b981"),
        ("🔍", "Explanation", "SHAP waterfall feature importance", "rgba(6,182,212,0.15)", "#06b6d4"),
    ]

    parts = []
    for i, (icon, title, desc, bg_color, border_color) in enumerate(steps):
        parts.append(
            '<div class="pipeline-step">'
            '<div class="step-icon" style="background:{bg};border:1px solid {bc}30;">'
            '{icon}</div>'
            '<div class="step-text">'
            '<div class="step-title">{title}</div>'
            '<div class="step-desc">{desc}</div>'
            '</div></div>'.format(icon=icon, title=title, desc=desc, bg=bg_color, bc=border_color)
        )
        if i < len(steps) - 1:
            parts.append(
                '<div class="pipeline-arrow" style="color:{bc};">↓</div>'.format(bc=border_color)
            )

    html = '<div style="display:flex;flex-direction:column;gap:0;">' + "".join(parts) + "</div>"
    st.markdown(html, unsafe_allow_html=True)


def section_divider():
    st.markdown(
        '<div style="height:1px;background:linear-gradient(90deg,transparent,rgba(128,128,128,0.15),transparent);margin:28px 0;"></div>',
        unsafe_allow_html=True,
    )


# SVG chevron for custom <details> elements
_CHEVRON_SVG = (
    '<svg class="chevron-svg" width="14" height="14" viewBox="0 0 16 16" '
    'fill="currentColor"><path d="M6 2l6 6-6 6V2z"/></svg>'
)


# =====================================================================
# CSS Design System — Premium Dark Theme
# =====================================================================

CUSTOM_CSS = """
<style>
@import url('https://fonts.googleapis.com/css2?family=Inter:wght@300;400;500;600;700;800&display=swap');

/* ── ROOT VARIABLES ── */
:root {
    --bg-deepest: #0a0e1a;
    --bg-card: #111827;
    --bg-elevated: #1a2035;
    --bg-hover: #1e2a42;
    --border-subtle: rgba(255,255,255,0.06);
    --border-hover: rgba(255,255,255,0.12);
    --text-primary: #f1f5f9;
    --text-secondary: #94a3b8;
    --text-muted: #64748b;
    --accent-blue: #3b82f6;
    --accent-cyan: #06b6d4;
    --accent-green: #10b981;
    --accent-amber: #f59e0b;
    --accent-red: #ef4444;
    --accent-purple: #8b5cf6;
    --gradient-blue: linear-gradient(135deg, #3b82f6, #06b6d4);
    --gradient-green: linear-gradient(135deg, #10b981, #06b6d4);
    --gradient-amber: linear-gradient(135deg, #f59e0b, #ef4444);
    --radius: 12px;
    --shadow-card: 0 4px 24px rgba(0,0,0,0.25);
    --shadow-glow-blue: 0 0 20px rgba(59,130,246,0.15);
}

/* ── LIGHT THEME — override ALL variables ── */
body.light-theme {
    --bg-deepest: #f8fafc;
    --bg-card: #ffffff;
    --bg-elevated: #f1f5f9;
    --bg-hover: #e2e8f0;
    --border-subtle: rgba(0,0,0,0.08);
    --border-hover: rgba(0,0,0,0.15);
    --text-primary: #0f172a;
    --text-secondary: #475569;
    --text-muted: #64748b;
    --accent-blue: #2563eb;
    --accent-cyan: #0891b2;
    --accent-green: #059669;
    --accent-amber: #d97706;
    --accent-red: #dc2626;
    --accent-purple: #7c3aed;
    --gradient-blue: linear-gradient(135deg, #2563eb, #0891b2);
    --gradient-green: linear-gradient(135deg, #059669, #0891b2);
    --gradient-amber: linear-gradient(135deg, #d97706, #dc2626);
    --shadow-card: 0 1px 3px rgba(0,0,0,0.08), 0 1px 2px rgba(0,0,0,0.06);
    --shadow-glow-blue: 0 0 15px rgba(37,99,235,0.08);
}
/* Keep sidebar dark in light mode (brand element) */
body.light-theme [data-testid="stSidebar"] {
    background: linear-gradient(180deg, #1e40af 0%, #1e3a8a 100%) !important;
}
body.light-theme [data-testid="stSidebar"] h1,
body.light-theme [data-testid="stSidebar"] h2,
body.light-theme [data-testid="stSidebar"] h3 {
    color: #f1f5f9 !important;
}
body.light-theme [data-testid="stSidebar"] [data-testid="stMarkdownContainer"] p,
body.light-theme [data-testid="stSidebar"] label,
body.light-theme [data-testid="stSidebar"] span {
    color: #cbd5e1 !important;
}
body.light-theme [data-testid="stSidebar"] hr {
    border-color: rgba(255,255,255,0.15) !important;
}
body.light-theme [data-testid="stSidebar"] .stRadio > div[role="radiogroup"] > label[data-baseweb="radio"] {
    background: rgba(255,255,255,0.08) !important;
    border-color: rgba(255,255,255,0.12) !important;
    color: #e2e8f0 !important;
}
body.light-theme [data-testid="stSidebar"] .stRadio > div[role="radiogroup"] > label[data-baseweb="radio"]:hover {
    background: rgba(255,255,255,0.15) !important;
    border-color: rgba(255,255,255,0.3) !important;
}
body.light-theme [data-testid="stSidebar"] .stCaption {
    color: rgba(255,255,255,0.5) !important;
}
/* Sidebar custom details stay dark-styled in light mode */
body.light-theme .custom-details summary {
    background: rgba(255,255,255,0.08);
    border-color: rgba(255,255,255,0.12);
    color: #cbd5e1;
}
body.light-theme .custom-details summary:hover {
    background: rgba(255,255,255,0.15);
    border-color: rgba(255,255,255,0.25);
}
body.light-theme .custom-details .details-body {
    color: rgba(255,255,255,0.6);
}
body.light-theme .custom-details .details-body a {
    color: #38bdf8;
}
/* Main-area custom details adapt to light */
body.light-theme .custom-details-main summary {
    background: var(--bg-card);
    border-color: var(--border-subtle);
    color: var(--text-secondary);
}
body.light-theme .custom-details-main .details-body {
    color: var(--text-muted);
}
/* Disclaimer warm tones for light */
body.light-theme .disclaimer-box {
    background: #fffbeb;
    color: #92400e;
    border-color: #f59e0b;
}
/* Light scrollbar */
body.light-theme ::-webkit-scrollbar-track { background: #f1f5f9; }
body.light-theme ::-webkit-scrollbar-thumb { background: #cbd5e1; }
body.light-theme ::-webkit-scrollbar-thumb:hover { background: #94a3b8; }
/* Light-mode shadow overrides */
body.light-theme .pred-card, body.light-theme .pipeline-step, body.light-theme .metric-highlight {
    box-shadow: 0 1px 3px rgba(0,0,0,0.08), 0 2px 8px rgba(0,0,0,0.04);
}
body.light-theme .stDownloadButton > button {
    color: #2563eb !important;
}
body.light-theme .stPlotlyChart {
    box-shadow: 0 1px 3px rgba(0,0,0,0.06);
}

/* ── GLOBAL TYPOGRAPHY ── */
html, body, [class*="st-"], .stMarkdown, .stText, p, span, label, div {
    font-family: 'Inter', -apple-system, BlinkMacSystemFont, sans-serif !important;
    -webkit-font-smoothing: antialiased;
}

/* ── PAGE BACKGROUND ── */
.stApp {
    background:
        radial-gradient(ellipse at 20% 0%, rgba(59,130,246,0.08) 0%, transparent 50%),
        radial-gradient(ellipse at 80% 100%, rgba(6,182,212,0.05) 0%, transparent 50%),
        var(--bg-deepest) !important;
}
.stApp > header {
    background: transparent !important;
    backdrop-filter: none !important;
    border-bottom: none !important;
    box-shadow: none !important;
}
[data-testid="stHeader"] {
    background: transparent !important;
    backdrop-filter: none !important;
    border-bottom: none !important;
    box-shadow: none !important;
}

/* ── MAIN CONTENT AREA ── */
.block-container {
    padding-top: 2rem !important;
    padding-bottom: 2rem !important;
    max-width: 1200px !important;
}

/* ── SIDEBAR ── */
[data-testid="stSidebar"] {
    background: linear-gradient(180deg, #0f172a 0%, #0a0e1a 100%) !important;
    border-right: 1px solid var(--border-subtle) !important;
}
[data-testid="stSidebar"] [data-testid="stMarkdownContainer"] p,
[data-testid="stSidebar"] [data-testid="stMarkdownContainer"] span,
[data-testid="stSidebar"] label,
[data-testid="stSidebar"] .stRadio label span {
    color: var(--text-secondary) !important;
}
[data-testid="stSidebar"] h1, [data-testid="stSidebar"] h2, [data-testid="stSidebar"] h3 {
    color: var(--text-primary) !important;
}
[data-testid="stSidebar"] hr {
    border-color: var(--border-subtle) !important;
}
[data-testid="stSidebar"] .stRadio > div[role="radiogroup"] > label[data-baseweb="radio"] {
    background: var(--bg-card) !important;
    border: 1px solid var(--border-subtle) !important;
    border-radius: 8px !important;
    padding: 8px 12px !important;
    margin-bottom: 4px !important;
    transition: all 0.2s ease !important;
}
[data-testid="stSidebar"] .stRadio > div[role="radiogroup"] > label[data-baseweb="radio"]:hover {
    border-color: var(--accent-blue) !important;
    background: var(--bg-elevated) !important;
}
[data-testid="stSidebar"] .stCaption, [data-testid="stSidebar"] small {
    color: var(--text-muted) !important;
}

/* Material Icons ligature text is replaced with SVGs via JS below.
   No CSS blanket-hide here — it would override the JS replacements. */

/* ── Custom <details> expanders with SVG chevrons ── */
.custom-details {
    border-radius: 8px;
    margin-bottom: 8px;
    overflow: hidden;
}
.custom-details summary {
    cursor: pointer;
    display: flex;
    align-items: center;
    gap: 8px;
    padding: 10px 14px;
    border-radius: 8px;
    background: var(--bg-card);
    border: 1px solid var(--border-subtle);
    list-style: none;
    font-weight: 500;
    font-size: 0.9rem;
    color: var(--text-secondary);
    transition: all 0.2s ease;
}
.custom-details summary:hover {
    border-color: var(--accent-blue);
    background: var(--bg-elevated);
}
.custom-details summary::-webkit-details-marker { display: none; }
.custom-details summary::marker { display: none; content: ""; }
.custom-details .chevron-svg {
    flex-shrink: 0;
    transition: transform 0.2s;
}
.custom-details[open] .chevron-svg {
    transform: rotate(90deg);
}
.custom-details .details-body {
    padding: 12px 14px;
    font-size: 0.84rem;
    line-height: 1.6;
    color: var(--text-muted);
}
.custom-details .details-body a {
    color: var(--accent-cyan);
}
/* Main-area variant */
.custom-details-main summary {
    background: var(--bg-card);
    border: 1px solid var(--border-subtle);
    color: var(--text-secondary);
}
.custom-details-main summary:hover {
    background: var(--bg-elevated);
    border-color: var(--border-hover);
}
.custom-details-main .details-body {
    color: var(--text-muted);
}

/* ── TABS — Pill style ── */
.stTabs [data-baseweb="tab-list"] {
    gap: 0px;
    background: var(--bg-card);
    border-radius: 10px;
    padding: 4px;
    border: 1px solid var(--border-subtle);
}
.stTabs [data-baseweb="tab"] {
    border-radius: 8px;
    padding: 8px 20px;
    color: var(--text-muted) !important;
    font-weight: 500;
    font-size: 0.9rem;
    background: transparent;
    border: none !important;
    transition: all 0.2s ease;
}
.stTabs [data-baseweb="tab"]:hover {
    color: var(--text-primary) !important;
    background: var(--bg-elevated);
}
.stTabs [aria-selected="true"] {
    background: var(--accent-blue) !important;
    color: white !important;
    font-weight: 600;
}
.stTabs [data-baseweb="tab-highlight"],
.stTabs [data-baseweb="tab-border"] {
    display: none;
}

/* ── MAIN HEADER ── */
.main-header {
    font-size: 1.6rem;
    font-weight: 700;
    color: var(--accent-blue);
    margin-bottom: 2px;
    letter-spacing: -0.02em;
}
.sub-header {
    font-size: 0.95rem;
    color: var(--text-muted);
    margin-bottom: 16px;
}

/* ── DISCLAIMER ── */
.disclaimer-box {
    background: rgba(245,158,11,0.08);
    border: 1px solid rgba(245,158,11,0.2);
    border-left: 3px solid var(--accent-amber);
    border-radius: 8px;
    padding: 12px 18px;
    font-size: 0.84rem;
    color: #fbbf24;
    margin-bottom: 20px;
}

/* ── PREDICTION CARDS ── */
.pred-card {
    background: var(--bg-card);
    border: 1px solid var(--border-subtle);
    border-radius: var(--radius);
    padding: 20px 24px;
    position: relative;
    overflow: hidden;
    transition: all 0.25s ease;
    min-height: 120px;
    margin-bottom: 14px;
}
.pred-card:hover {
    border-color: var(--border-hover);
    transform: translateY(-2px);
    box-shadow: var(--shadow-card);
}
.pred-card::before {
    content: '';
    position: absolute;
    top: 0; left: 0; right: 0;
    height: 3px;
    border-radius: var(--radius) var(--radius) 0 0;
}
.pred-card.high::before { background: var(--gradient-green); }
.pred-card.moderate::before { background: var(--gradient-blue); }
.pred-card.low::before { background: var(--gradient-amber); }
.pred-card .sample-id {
    font-size: 0.72rem;
    color: var(--text-muted);
    text-transform: uppercase;
    letter-spacing: 1px;
    font-weight: 600;
    margin-bottom: 6px;
}
.pred-card .class-name {
    font-size: 1.35rem;
    font-weight: 700;
    color: var(--text-primary);
    margin-bottom: 12px;
    letter-spacing: -0.01em;
}
.pred-card .confidence-bar-track {
    width: 100%;
    height: 4px;
    background: rgba(255,255,255,0.06);
    border-radius: 2px;
    margin-bottom: 8px;
    overflow: hidden;
}
.pred-card .confidence-bar-fill {
    height: 100%;
    border-radius: 2px;
    transition: width 0.8s ease;
}
.pred-card.high .confidence-bar-fill { background: var(--gradient-green); }
.pred-card.moderate .confidence-bar-fill { background: var(--gradient-blue); }
.pred-card.low .confidence-bar-fill { background: var(--gradient-amber); }
.pred-card .confidence-text {
    font-size: 0.82rem;
    font-weight: 600;
    display: flex;
    justify-content: space-between;
    align-items: center;
}
.pred-card.high .confidence-text { color: var(--accent-green); }
.pred-card.moderate .confidence-text { color: var(--accent-blue); }
.pred-card.low .confidence-text { color: var(--accent-amber); }

/* ── CARD SHADOW DEPTH ── */
.pred-card, .pipeline-step, .metric-highlight {
    box-shadow: 0 1px 2px rgba(0,0,0,0.3), 0 4px 16px rgba(0,0,0,0.2), 0 0 1px rgba(255,255,255,0.05) inset;
}

/* ── WELCOME PAGE ── */
.welcome-title {
    font-size: 2rem;
    font-weight: 800;
    color: var(--text-primary);
    letter-spacing: -0.03em;
    line-height: 1.2;
    margin-bottom: 12px;
}
.welcome-desc {
    font-size: 1rem;
    color: var(--text-secondary);
    line-height: 1.7;
    margin-bottom: 20px;
}

/* ── Pipeline steps ── */
.pipeline-step {
    background: var(--bg-card);
    border: 1px solid var(--border-subtle);
    border-radius: 10px;
    padding: 14px 18px;
    margin-bottom: 0;
    display: flex;
    align-items: center;
    gap: 14px;
    transition: all 0.2s ease;
}
.pipeline-step:hover {
    border-color: var(--accent-blue);
    background: var(--bg-elevated);
}
.pipeline-step .step-icon {
    width: 36px;
    height: 36px;
    border-radius: 8px;
    display: flex;
    align-items: center;
    justify-content: center;
    font-size: 1rem;
    flex-shrink: 0;
    background: var(--bg-elevated);
}
.pipeline-step .step-title {
    font-size: 0.88rem;
    font-weight: 600;
    color: var(--text-primary);
}
.pipeline-step .step-desc {
    font-size: 0.75rem;
    color: var(--text-muted);
}
.pipeline-arrow {
    text-align: center;
    color: var(--accent-cyan);
    font-size: 0.85rem;
    line-height: 1;
    padding: 3px 0;
    opacity: 0.5;
}

/* ── BEST MODEL BANNER ── */
.best-model-banner {
    background: linear-gradient(135deg, rgba(16,185,129,0.1) 0%, rgba(6,182,212,0.08) 100%);
    border: 1px solid rgba(16,185,129,0.2);
    border-radius: var(--radius);
    padding: 14px 20px;
    margin-bottom: 20px;
    display: flex;
    align-items: center;
    gap: 10px;
}
.best-model-banner .trophy { font-size: 1.2rem; }
.best-model-banner .banner-text {
    font-size: 0.92rem;
    font-weight: 600;
    color: var(--accent-green);
}

/* ── METRIC HIGHLIGHT CARDS ── */
.metric-highlight {
    background: var(--bg-card);
    border: 1px solid var(--border-subtle);
    border-radius: var(--radius);
    padding: 16px 20px;
    text-align: center;
}
.metric-highlight .metric-value {
    font-size: 1.8rem;
    font-weight: 800;
    background: var(--gradient-blue);
    -webkit-background-clip: text;
    -webkit-text-fill-color: transparent;
    background-clip: text;
}
.metric-highlight .metric-label {
    font-size: 0.78rem;
    color: var(--text-muted);
    text-transform: uppercase;
    letter-spacing: 0.5px;
    margin-top: 4px;
}

/* ── FILE UPLOADER ── */
[data-testid="stFileUploader"] {
    background: var(--bg-card) !important;
    border: 1px dashed var(--border-hover) !important;
    border-radius: var(--radius) !important;
    padding: 12px !important;
}
[data-testid="stFileUploader"]:hover {
    border-color: var(--accent-blue) !important;
}

/* ── DATAFRAME ── */
[data-testid="stDataFrame"] {
    background: var(--bg-card) !important;
    border: 1px solid var(--border-subtle) !important;
    border-radius: var(--radius) !important;
    padding: 8px !important;
    box-shadow: var(--shadow-card);
    overflow: hidden;
}

/* ── SELECT BOXES ── */
[data-baseweb="select"] {
    background: var(--bg-card) !important;
    border-radius: 8px !important;
}

/* ── DOWNLOAD BUTTONS ── */
.stDownloadButton > button {
    background: linear-gradient(135deg, rgba(59,130,246,0.12), rgba(6,182,212,0.08)) !important;
    border: 1px solid rgba(59,130,246,0.25) !important;
    color: #60a5fa !important;
    border-radius: 8px !important;
    font-size: 0.85rem !important;
    font-weight: 600 !important;
    padding: 10px 20px !important;
    transition: all 0.2s ease !important;
}
.stDownloadButton > button:hover {
    background: linear-gradient(135deg, rgba(59,130,246,0.2), rgba(6,182,212,0.15)) !important;
    border-color: rgba(59,130,246,0.4) !important;
    color: #93c5fd !important;
    transform: translateY(-1px) !important;
    box-shadow: 0 4px 12px rgba(59,130,246,0.15) !important;
}

/* ── EXPANDER (Streamlit native fallback) ── */
[data-testid="stExpander"] {
    background: var(--bg-card) !important;
    border: 1px solid var(--border-subtle) !important;
    border-radius: 8px !important;
}

/* ── PLOTLY ── */
.js-plotly-plot .plotly .main-svg {
    border-radius: var(--radius);
}
.stPlotlyChart {
    background: var(--bg-card) !important;
    border: 1px solid var(--border-subtle) !important;
    border-radius: var(--radius) !important;
    padding: 16px !important;
    box-shadow: var(--shadow-card);
    margin-bottom: 16px;
}

/* ── SECTION TITLES ── */
.section-title {
    font-size: 1.4rem;
    font-weight: 700;
    color: var(--text-primary);
    margin-bottom: 4px;
    letter-spacing: -0.02em;
}
.section-subtitle {
    font-size: 0.9rem;
    color: var(--text-muted);
    margin-bottom: 20px;
}

/* ── FOOTER ── */
.app-footer {
    text-align: center;
    padding: 32px 0 16px 0;
    border-top: 1px solid var(--border-subtle);
    margin-top: 40px;
}
.app-footer .footer-brand {
    font-size: 0.85rem;
    font-weight: 600;
    color: var(--text-secondary);
    margin-bottom: 4px;
}
.app-footer .footer-sub {
    font-size: 0.78rem;
    color: var(--text-muted);
    line-height: 1.6;
}
.app-footer a {
    color: var(--accent-cyan) !important;
    text-decoration: none !important;
}
.app-footer a:hover {
    text-decoration: underline !important;
}

/* ── SCROLLBAR ── */
::-webkit-scrollbar { width: 6px; height: 6px; }
::-webkit-scrollbar-track { background: var(--bg-deepest); }
::-webkit-scrollbar-thumb { background: var(--text-muted); border-radius: 3px; }
::-webkit-scrollbar-thumb:hover { background: var(--text-secondary); }

/* ── HIDE STREAMLIT DEFAULT FOOTER ── */
footer {visibility: hidden;}
</style>
"""

# JS: Replace Material Icons ligature text with inline SVGs
ICON_FIX_JS = """
<script>
(function() {
    /* Detect Streamlit light/dark theme and set body class */
    function detectTheme() {
        try {
            var doc = window.parent.document;
            var el = doc.querySelector('[data-testid="stAppViewContainer"]')
                     || doc.querySelector('.stApp')
                     || doc.body;
            var bg = window.parent.getComputedStyle(el).backgroundColor;
            var m = bg.match(/\\d+/g);
            if (m) {
                var avg = (parseInt(m[0]) + parseInt(m[1]) + parseInt(m[2])) / 3;
                if (avg < 128) {
                    doc.body.classList.add('dark-theme');
                    doc.body.classList.remove('light-theme');
                } else {
                    doc.body.classList.add('light-theme');
                    doc.body.classList.remove('dark-theme');
                }
            }
        } catch(e) {}
    }

    var svgIcons = {
        'keyboard_double_arrow_left':
            '<svg width="20" height="20" viewBox="0 0 24 24" fill="none" '
            + 'stroke="currentColor" stroke-width="2.2" stroke-linecap="round" '
            + 'stroke-linejoin="round"><polyline points="11 17 6 12 11 7"/>'
            + '<polyline points="18 17 13 12 18 7"/></svg>',
        'keyboard_double_arrow_right':
            '<svg width="20" height="20" viewBox="0 0 24 24" fill="none" '
            + 'stroke="currentColor" stroke-width="2.2" stroke-linecap="round" '
            + 'stroke-linejoin="round"><polyline points="13 17 18 12 13 7"/>'
            + '<polyline points="6 17 11 12 6 7"/></svg>',
    };
    function fixIcons() {
        try {
            var doc = window.parent.document;
            var pat = /^(keyboard_|arrow_|chevron_|expand_|navigate_|menu$|more_)/;
            var walker = doc.createTreeWalker(
                doc.body, NodeFilter.SHOW_TEXT, null, false
            );
            var nodes = [];
            while (walker.nextNode()) nodes.push(walker.currentNode);
            nodes.forEach(function(node) {
                var txt = node.textContent.trim();
                if (!txt || !pat.test(txt)) return;
                var el = node.parentElement;
                if (!el || el.tagName === 'BODY' || el.dataset.svgDone) return;
                if (svgIcons[txt]) {
                    el.innerHTML = svgIcons[txt];
                    el.style.display = 'inline-flex';
                    el.style.alignItems = 'center';
                    el.style.justifyContent = 'center';
                } else {
                    el.style.display = 'none';
                }
                el.dataset.svgDone = '1';
            });
        } catch(e) {}
    }
    detectTheme();
    fixIcons();
    setInterval(function() { detectTheme(); fixIcons(); }, 800);
})();
</script>
"""


# =====================================================================
# Streamlit App Layout
# =====================================================================

def main():
    st.set_page_config(
        page_title="MLOmics — Cancer Subtype Classifier",
        page_icon="🧬",
        layout="wide",
        initial_sidebar_state="expanded",
    )

    # --- Inject CSS + JS ---
    st.markdown(CUSTOM_CSS, unsafe_allow_html=True)
    components.html(ICON_FIX_JS, height=0, width=0)

    # --- Sidebar ---
    with st.sidebar:
        st.markdown('''
<div style="padding: 8px 0 20px 0; border-bottom: 1px solid rgba(255,255,255,0.06); margin-bottom: 20px;">
    <div style="font-size: 2rem; font-weight: 800; background: linear-gradient(135deg, #3b82f6, #06b6d4); -webkit-background-clip: text; -webkit-text-fill-color: transparent; background-clip: text; letter-spacing: -0.03em; line-height: 1.1;">
        🧬 MLOmics
    </div>
    <div style="font-size: 0.78rem; color: #64748b; margin-top: 6px; line-height: 1.4;">
        Latent-Fusion Multi-Omics<br>Cancer Subtype Classifier
    </div>
</div>
''', unsafe_allow_html=True)

        model_choice = st.radio(
            "Select Model",
            ["XGBoost (Baseline)", "Intermediate Fusion (Deep)"],
            help="XGBoost uses early fusion (concatenation). "
                 "Intermediate Fusion uses per-modality encoders -> latent concat -> MLP.",
        )

        st.markdown("---")

        st.markdown(
            '<details class="custom-details"><summary>{chev} About this project</summary>'
            '<div class="details-body">'
            "<strong>MLOmics</strong> is a BSc CS Final Year Project that classifies "
            "TCGA cancer subtypes using multi-omics data (mRNA, miRNA, Methylation, CNV).<br><br>"
            "<strong>Models:</strong><br>"
            "- <strong>XGBoost</strong> — gradient-boosted trees on concatenated features<br>"
            "- <strong>Intermediate Fusion</strong> — per-modality neural encoders -> "
            "latent concatenation -> MLP classifier<br><br>"
            "<strong>Cancer type:</strong> GS-BRCA (5 subtypes)<br>"
            "<strong>Features:</strong> 15,366 total<br><br>"
            '<a href="https://github.com/deaneeth/multi-omics-cancer-subtype-classifier" '
            'target="_blank">GitHub Repository</a>'
            "</div></details>".format(chev=_CHEVRON_SVG),
            unsafe_allow_html=True,
        )

        st.markdown(
            '<details class="custom-details"><summary>{chev} How to use</summary>'
            '<div class="details-body">'
            "1. Select a model in the sidebar<br>"
            "2. Upload a CSV in <strong>MLOmics format</strong>:<br>"
            "&nbsp;&nbsp;&nbsp;- Rows = features, Columns = samples<br>"
            "&nbsp;&nbsp;&nbsp;- First column = feature names (index)<br>"
            "3. View prediction, confidence, and SHAP explanation<br>"
            "4. Check the <strong>Model Comparison</strong> tab for benchmarks"
            "</div></details>".format(chev=_CHEVRON_SVG),
            unsafe_allow_html=True,
        )

        st.markdown("---")
        st.caption("v0.4 · Seeds: 42 · 5-fold CV")

    # --- Main content ---
    st.markdown(
        '<p class="main-header">🧬 MLOmics Cancer Subtype Classifier</p>',
        unsafe_allow_html=True,
    )
    st.markdown(
        '<p class="sub-header">'
        "Latent-Fusion Multi-Omics Classifier for Cancer Subtype Prediction"
        "</p>",
        unsafe_allow_html=True,
    )

    # Disclaimer
    st.markdown(
        '<div class="disclaimer-box">'
        "⚠️ <strong>Academic research prototype.</strong> "
        "Not for clinical use. Results are for educational and research purposes only. "
        "This tool should not be used for medical diagnosis or treatment decisions."
        "</div>",
        unsafe_allow_html=True,
    )

    # Load shared resources
    try:
        cfg = load_config_json()
        scaler = load_scaler()
    except Exception as e:
        st.error(f"Failed to load model artifacts: {e}")
        st.stop()

    # --- Tabs ---
    tab_predict, tab_compare = st.tabs(["Prediction", "Model Comparison"])

    # =================================================================
    # TAB 1: PREDICTION
    # =================================================================
    with tab_predict:

        uploaded_file = st.file_uploader(
            "Choose a CSV file",
            type=["csv"],
            help="Upload a CSV with features as rows and samples as columns.",
            label_visibility="collapsed",
        )

        if uploaded_file is None:
            # ------- WELCOME PANEL -------
            col_left, col_right = st.columns([3, 2], gap="large")

            with col_left:
                st.markdown('''
<div style="margin-bottom: 24px;">
    <div style="font-size: 2.4rem; font-weight: 800; letter-spacing: -0.04em; line-height: 1.15; margin-bottom: 12px;">
        <span style="color: var(--text-primary);">Welcome to </span>
        <span style="background: linear-gradient(135deg, #3b82f6, #06b6d4); -webkit-background-clip: text; -webkit-text-fill-color: transparent; background-clip: text;">MLOmics</span>
    </div>
    <div style="font-size: 1rem; color: var(--text-secondary); line-height: 1.7; max-width: 520px;">
        Upload a multi-omics CSV file to classify cancer subtypes using machine learning.
        The system supports both tree-based (XGBoost) and deep learning (Intermediate Fusion)
        models trained on the MLOmics benchmark dataset.
    </div>
</div>
''', unsafe_allow_html=True)
                st.markdown('''
<div style="display: flex; flex-wrap: wrap; gap: 8px; margin-bottom: 12px;">
    <span style="background: rgba(59,130,246,0.12); color: #60a5fa; padding: 4px 12px; border-radius: 20px; font-size: 0.8rem; font-weight: 500;">mRNA (5,000)</span>
    <span style="background: rgba(6,182,212,0.12); color: #22d3ee; padding: 4px 12px; border-radius: 20px; font-size: 0.8rem; font-weight: 500;">miRNA (366)</span>
    <span style="background: rgba(16,185,129,0.12); color: #34d399; padding: 4px 12px; border-radius: 20px; font-size: 0.8rem; font-weight: 500;">Methylation (5,000)</span>
    <span style="background: rgba(245,158,11,0.12); color: #fbbf24; padding: 4px 12px; border-radius: 20px; font-size: 0.8rem; font-weight: 500;">CNV (5,000)</span>
</div>
<div style="font-size: 0.88rem; color: var(--text-muted); margin-bottom: 20px;">
    Cancer type: <strong style="color: var(--text-secondary);">GS-BRCA</strong> — 5 molecular subtypes
</div>
''', unsafe_allow_html=True)

                sample_bytes = load_sample_csv_bytes()
                if sample_bytes:
                    st.download_button(
                        "📥 Download sample CSV",
                        data=sample_bytes,
                        file_name="sample_input.csv",
                        mime="text/csv",
                    )

            with col_right:
                render_pipeline_diagram()

            st.markdown("---")
            st.markdown("##### ↑ Upload a CSV file above to get started")

        else:
            # ------- PREDICTION FLOW -------
            try:
                # --- Parse & transpose ---
                raw_df = pd.read_csv(uploaded_file, index_col=0)
                st.caption(
                    f"Raw CSV: {raw_df.shape[0]} features x {raw_df.shape[1]} samples"
                )

                data_df = raw_df.T
                st.caption(
                    f"After transpose: {data_df.shape[0]} samples x "
                    f"{data_df.shape[1]} features"
                )

                # --- Feature alignment ---
                expected_features = cfg["feature_names"]
                if data_df.shape[1] != len(expected_features):
                    st.error(
                        f"Feature count mismatch: uploaded CSV has "
                        f"{data_df.shape[1]} features, but the model expects "
                        f"{len(expected_features)}. Ensure the CSV contains "
                        f"exactly these modalities: "
                        f"{', '.join(cfg['modality_order'])}."
                    )
                    st.stop()

                X_raw = data_df.values.astype(np.float64)

                # --- Scale ---
                X_scaled = scaler.transform(X_raw)

                # --- Predict ---
                section_divider()
                st.markdown('<div class="section-title">🎯 Prediction Results</div>', unsafe_allow_html=True)

                if model_choice == "XGBoost (Baseline)":
                    xgb_model = load_xgb_model()
                    preds, probs = predict_xgboost(xgb_model, X_scaled)
                else:
                    fusion_model = load_fusion_model()
                    preds, probs = predict_fusion(fusion_model, X_scaled, cfg)

                # Display prediction cards
                sample_ids = list(data_df.index)
                n_samples = len(sample_ids)

                if n_samples <= 3:
                    cols = st.columns(n_samples)
                    for i, sid in enumerate(sample_ids):
                        pred_class = int(preds[i])
                        class_name = cfg["class_names"].get(
                            str(pred_class), f"Class {pred_class}"
                        )
                        confidence = float(probs[i, pred_class])
                        with cols[i]:
                            render_prediction_card(sid, class_name, confidence)
                else:
                    for i, sid in enumerate(sample_ids):
                        pred_class = int(preds[i])
                        class_name = cfg["class_names"].get(
                            str(pred_class), f"Class {pred_class}"
                        )
                        confidence = float(probs[i, pred_class])
                        render_prediction_card(sid, class_name, confidence)

                # --- Export predictions ---
                export_df = pd.DataFrame({
                    "Sample": sample_ids,
                    "Predicted Subtype": [
                        cfg["class_names"].get(str(int(p)), f"Class {p}")
                        for p in preds
                    ],
                    "Confidence": [
                        float(probs[i, int(preds[i])]) for i in range(n_samples)
                    ],
                })
                st.download_button(
                    "📥 Export predictions as CSV",
                    data=export_df.to_csv(index=False),
                    file_name="mlomics_predictions.csv",
                    mime="text/csv",
                )

                # --- Confidence chart ---
                section_divider()
                st.markdown('<div class="section-title">Confidence Distribution</div>', unsafe_allow_html=True)
                sample_to_explain = 0
                if n_samples > 1:
                    sample_to_explain = st.selectbox(
                        "Select sample for detailed view",
                        range(n_samples),
                        format_func=lambda idx: sample_ids[idx],
                    )
                render_confidence_chart(probs, cfg, sample_to_explain)

                # --- SHAP Explanation ---
                section_divider()
                st.markdown('<div class="section-title">Feature Importance (SHAP)</div>', unsafe_allow_html=True)
                st.markdown(
                    "The waterfall plot shows which genomic features influenced "
                    "this prediction. **Red bars** push toward the predicted class; "
                    "**blue bars** push away from it. Features are ranked by "
                    "absolute impact."
                )

                st.markdown(
                    '<details class="custom-details custom-details-main">'
                    "<summary>{chev} What is SHAP?</summary>"
                    '<div class="details-body">'
                    "<strong>SHAP (SHapley Additive exPlanations)</strong> is a "
                    "game-theoretic approach to explain individual predictions. Each "
                    "feature receives a contribution score showing how much it pushed "
                    "the prediction toward or away from a particular class."
                    "</div></details>".format(chev=_CHEVRON_SVG),
                    unsafe_allow_html=True,
                )

                if model_choice == "XGBoost (Baseline)":
                    st.caption(
                        "Computing TreeSHAP waterfall plot for the selected sample..."
                    )
                    with st.spinner("Running TreeSHAP..."):
                        xgb_model = load_xgb_model()
                        render_shap_waterfall(
                            xgb_model, X_scaled, cfg, sample_to_explain
                        )
                else:
                    fusion_shap_path = os.path.join(
                        ARTIFACT_DIR, "fusion_shap_sample.pkl"
                    )
                    if os.path.exists(fusion_shap_path):
                        st.caption(
                            "Showing precomputed DeepSHAP values for a reference "
                            "sample."
                        )
                        precomputed = joblib.load(fusion_shap_path)
                        shap.plots.waterfall(
                            precomputed, max_display=15, show=False
                        )
                        fig = plt.gcf()
                        fig.patch.set_facecolor("none")
                        for a in fig.get_axes():
                            a.set_facecolor("none")
                            a.tick_params(colors="#94a3b8")
                            a.xaxis.label.set_color("#94a3b8")
                            a.yaxis.label.set_color("#94a3b8")
                            if a.get_title():
                                a.title.set_color("#64748b")
                            for spine in a.spines.values():
                                spine.set_edgecolor("#64748b")
                        st.pyplot(fig, clear_figure=True)
                        plt.close(fig)
                    else:
                        st.info(
                            "Deep attribution (Integrated Gradients) for the "
                            "fusion model is available in the full analysis "
                            "notebook (`04_explainability.ipynb`). Real-time SHAP "
                            "for neural networks requires precomputation."
                        )

            except Exception as e:
                st.error(
                    "Could not process the uploaded file. Please check the format "
                    "matches the sample CSV."
                )
                with st.expander("Technical details"):
                    st.code(str(e))

    # =================================================================
    # TAB 2: MODEL COMPARISON
    # =================================================================
    with tab_compare:
        st.markdown('<div class="section-title">Model Performance Comparison</div>', unsafe_allow_html=True)
        st.markdown(
            "Patient-level stratified 5-fold cross-validation across all "
            "trained models."
        )

        comp_df = load_model_comparison()
        if comp_df is None:
            st.warning("No model comparison data found.")
        else:
            # --- Best model banner ---
            best_row = comp_df.loc[comp_df["F1_mean"].idxmax()]
            st.markdown(
                '<div class="best-model-banner">'
                '<span class="trophy">🏆</span>'
                '<span class="banner-text">Best model: {model} on '
                "{cancer} — F1 = {f1:.3f} +/- {std:.3f}</span>"
                "</div>".format(
                    model=best_row["Model"],
                    cancer=best_row["Cancer"],
                    f1=best_row["F1_mean"],
                    std=best_row["F1_std"],
                ),
                unsafe_allow_html=True,
            )

            # --- Metric highlight cards ---
            mc1, mc2, mc3 = st.columns(3)
            with mc1:
                st.markdown(
                    '<div class="metric-highlight">'
                    '<div class="metric-value">{:.3f}</div>'
                    '<div class="metric-label">Best F1 Score</div>'
                    "</div>".format(best_row["F1_mean"]),
                    unsafe_allow_html=True,
                )
            with mc2:
                st.markdown(
                    '<div class="metric-highlight">'
                    '<div class="metric-value">{:.3f}</div>'
                    '<div class="metric-label">Precision</div>'
                    "</div>".format(best_row["Precision_mean"]),
                    unsafe_allow_html=True,
                )
            with mc3:
                st.markdown(
                    '<div class="metric-highlight">'
                    '<div class="metric-value">{:.3f}</div>'
                    '<div class="metric-label">Recall</div>'
                    "</div>".format(best_row["Recall_mean"]),
                    unsafe_allow_html=True,
                )

            st.markdown("")  # spacer

            # --- Filter by cancer type ---
            cancer_types = sorted(comp_df["Cancer"].unique())
            selected_cancer = st.selectbox(
                "Filter by cancer type", ["All"] + cancer_types
            )

            display_df = comp_df.copy()
            if selected_cancer != "All":
                display_df = display_df[display_df["Cancer"] == selected_cancer]

            # --- Styled table ---
            format_dict = {
                c: "{:.3f}" for c in display_df.columns
                if c.endswith("_mean") or c.endswith("_std")
            }
            st.dataframe(
                display_df.style.format(format_dict).highlight_max(
                    subset=["F1_mean", "Precision_mean", "Recall_mean"],
                    color="rgba(16,185,129,0.15)",
                ),
                use_container_width=True,
            )
            st.caption(
                "Higher is better for all metrics. "
                "Highlighted cells indicate best per column."
            )

            # --- Plotly grouped bar chart ---
            st.markdown("#### F1 Score Comparison")
            fig = go.Figure()

            models = display_df["Model"].unique()
            for i, model_name in enumerate(models):
                model_data = display_df[display_df["Model"] == model_name]
                fig.add_trace(
                    go.Bar(
                        name=model_name,
                        x=model_data["Cancer"],
                        y=model_data["F1_mean"],
                        error_y=dict(
                            type="data",
                            array=model_data["F1_std"].tolist(),
                            visible=True,
                        ),
                        marker_color=CHART_COLORS[i % len(CHART_COLORS)],
                        text=[f"{v:.3f}" for v in model_data["F1_mean"]],
                        textposition="outside",
                        textfont=dict(size=12),
                    )
                )

            fig.update_layout(
                barmode="group",
                yaxis_title="F1 Score (mean +/- std)",
                xaxis_title="Cancer Type",
                height=480,
                legend=dict(
                    orientation="h", yanchor="bottom", y=1.02,
                    xanchor="right", x=1,
                ),
            )
            apply_dark_theme(fig)
            fig.update_traces(marker=dict(line=dict(width=0)))
            st.plotly_chart(fig, use_container_width=True)

            # --- Radar chart ---
            st.markdown("#### Multi-Metric Profile")
            metric_cols = [
                "Precision_mean", "Recall_mean", "F1_mean",
                "NMI_mean", "ARI_mean",
            ]
            metric_labels = ["Precision", "Recall", "F1", "NMI", "ARI"]

            radar_cancer = (
                selected_cancer if selected_cancer != "All"
                else "GS-BRCA"
            )
            radar_df = comp_df[comp_df["Cancer"] == radar_cancer]

            if len(radar_df) == 0:
                st.info("No data for selected cancer type in radar view.")
            else:
                fig_radar = go.Figure()
                for i, model_name in enumerate(radar_df["Model"].unique()):
                    row = radar_df[radar_df["Model"] == model_name].iloc[0]
                    values = [row[c] for c in metric_cols] + [row[metric_cols[0]]]
                    labels = metric_labels + [metric_labels[0]]

                    fig_radar.add_trace(
                        go.Scatterpolar(
                            r=values,
                            theta=labels,
                            fill="toself",
                            name=model_name,
                            opacity=0.6,
                            line=dict(
                                color=CHART_COLORS[i % len(CHART_COLORS)]
                            ),
                        )
                    )

                fig_radar.update_layout(
                    polar=dict(
                        radialaxis=dict(
                            visible=True, range=[0, 1],
                            gridcolor="rgba(128,128,128,0.2)",
                            color="#64748b",
                        ),
                        angularaxis=dict(
                            gridcolor="rgba(128,128,128,0.2)",
                            color="#64748b",
                        ),
                        bgcolor="rgba(0,0,0,0)",
                    ),
                    height=500,
                    title=dict(
                        text=f"Multi-Metric Profile ({radar_cancer})",
                        font=dict(size=15),
                    ),
                    legend=dict(
                        orientation='h',
                        yanchor='top', y=-0.15,
                        xanchor='center', x=0.5,
                        font=dict(size=11),
                    ),
                    margin=dict(t=40, b=80, l=80, r=80),
                )
                apply_dark_theme(fig_radar)
                if selected_cancer == "All":
                    st.caption(
                        "Showing GS-BRCA by default. Select a specific cancer "
                        "type above to change."
                    )
                st.plotly_chart(fig_radar, use_container_width=True)

    # =================================================================
    # FOOTER
    # =================================================================
    st.markdown(
        """
        <div class="app-footer">
            <div class="footer-brand">MLOmics</div>
            <div class="footer-sub">
                BSc Computer Science Final Year Project<br>
                Latent-Fusion Multi-Omics Classifier for Cancer Subtype Prediction<br>
                <a href="https://github.com/deaneeth/multi-omics-cancer-subtype-classifier"
                   target="_blank">GitHub Repository</a> ·
                Powered by PyTorch, XGBoost, SHAP<br>
                <em>Academic research prototype. Not for clinical use.</em>
            </div>
        </div>
        """,
        unsafe_allow_html=True,
    )


if __name__ == "__main__":
    main()
