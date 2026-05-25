"""
dashboard/app.py
----------------
MedAccès — Production Streamlit Dashboard

Design: Dark editorial — deep navy, electric blue accents,
DM Serif Display headings, DM Mono for code and numbers.

Pages:
  Home      — hero, problem statement, architecture, tech cards
  Predict   — live inference demo with sliders and real API call
  Analytics — dataset explorer, charts, prediction audit log
  Model     — metrics, feature importance, confusion matrix, MLflow

Run:
  streamlit run dashboard/app.py

Connects to FastAPI on localhost:8000.
Falls back gracefully with local data when API is offline.
"""

import streamlit as st
import requests
import pandas as pd
import numpy as np
import plotly.graph_objects as go
import math
import json
from pathlib import Path
import os

def _fix_mojibake(value):
    """
    Best-effort fix for common UTF-8→Latin-1 mojibake, e.g. "CoatrÃ©ven" → "Coatréven".
    Returns the original value on failure.
    """
    if value is None:
        return value
    if not isinstance(value, str):
        return value
    if "Ã" not in value and "Â" not in value and "â" not in value:
        return value
    try:
        fixed = value.encode("latin-1", errors="ignore").decode("utf-8", errors="ignore")
        return fixed or value
    except Exception:
        return value


def _st_button(label, **kwargs):
    """
    Streamlit compatibility shim: `type=` isn't available in older Streamlit versions.
    """
    try:
        return st.button(label, **kwargs)
    except TypeError:
        kwargs.pop("type", None)
        return st.button(label, **kwargs)


def _nav_to(page_label: str):
    """
    Navigate without mutating `st.session_state.nav_page` after the sidebar widget
    with key `nav_page` has been instantiated in the current run.

    We stage the navigation into `_nav_target` and apply it at the top of
    `render_sidebar()` on the next rerun (before the radio widget is created).
    """
    st.session_state["_nav_target"] = page_label
    st.rerun()

# ─────────────────────────────────────────────────────────────────────
# CONFIGURATION
# ─────────────────────────────────────────────────────────────────────

API_URL = os.getenv("API_URL", "http://localhost:8000").rstrip("/")'''
DATA_PATH = Path(__file__).parent.parent / "data" / "raw" / "communes_health.csv"
META_PATH = Path(__file__).parent.parent / "models" / "artifacts" / "model_metadata.json"

st.set_page_config(
    page_title           = "MedAccès — Medical Desert AI",
    page_icon            = "🏥",
    layout               = "wide",
    initial_sidebar_state= "expanded",
)

# ─────────────────────────────────────────────────────────────────────
# DESIGN TOKENS — single source of truth for every colour
# ─────────────────────────────────────────────────────────────────────

NAVY    = "#0a0f1e"
NAVY2   = "#111827"
NAVY3   = "#1a2438"
PANEL   = "#151d2e"
BORDER  = "#1f2f4a"
BLUE    = "#2563eb"
BLUE2   = "#3b82f6"
CYAN    = "#06b6d4"
RED     = "#ef4444"
AMBER   = "#f59e0b"
GREEN   = "#22c55e"
TEXT    = "#e2e8f0"
MUTED   = "#64748b"
ACCENT  = "#94a3b8"

RISK_COLORS = {0: GREEN, 1: AMBER, 2: RED}
RISK_LABELS = {0: "Low", 1: "Medium", 2: "High"}

# ─────────────────────────────────────────────────────────────────────
# GLOBAL CSS
# Injected once. Controls every visual detail across all pages.
# ─────────────────────────────────────────────────────────────────────

GLOBAL_CSS = """
<style>
@import url('https://fonts.googleapis.com/css2?family=DM+Serif+Display:ital@0;1&family=DM+Mono:wght@400;500&family=Outfit:wght@300;400;500;600&display=swap');

/* ── Design tokens ───────────────────────────────────────────── */
:root {
  --navy:   #0a0f1e;
  --navy2:  #111827;
  --navy3:  #1a2438;
  --panel:  #151d2e;
  --border: #1f2f4a;
  --blue:   #2563eb;
  --blue2:  #3b82f6;
  --cyan:   #06b6d4;
  --red:    #ef4444;
  --amber:  #f59e0b;
  --green:  #22c55e;
  --text:   #e2e8f0;
  --muted:  #64748b;
  --accent: #94a3b8;
}

/* ── Base ────────────────────────────────────────────────────── */
html, body, [class*="css"] {
  font-family: 'Outfit', sans-serif !important;
}

.stApp {
  background-color: var(--navy) !important;
  color: var(--text) !important;
}

.main .block-container {
  padding: 2rem 2.5rem 4rem !important;
  max-width: 1400px !important;
}

/* ── Sidebar ─────────────────────────────────────────────────── */
[data-testid="stSidebar"] {
  background-color: var(--navy2) !important;
  border-right: 1px solid var(--border) !important;
}

[data-testid="stSidebar"] * {
  color: var(--text) !important;
}

[data-testid="stSidebar"] .stRadio label {
  padding: 10px 16px !important;
  border-radius: 4px !important;
  font-size: 14px !important;
  font-family: 'Outfit', sans-serif !important;
  color: var(--accent) !important;
  transition: all 0.15s !important;
}

[data-testid="stSidebar"] .stRadio label:hover {
  background: rgba(37,99,235,0.1) !important;
  color: var(--blue2) !important;
}

/* ── Buttons ─────────────────────────────────────────────────── */
.stButton > button {
  background: var(--blue) !important;
  color: #ffffff !important;
  border: none !important;
  border-radius: 2px !important;
  font-family: 'Outfit', sans-serif !important;
  font-weight: 500 !important;
  font-size: 14px !important;
  letter-spacing: 0.02em !important;
  transition: background 0.15s !important;
}

.stButton > button:hover {
  background: var(--blue2) !important;
}

/* ── Metrics ─────────────────────────────────────────────────── */
[data-testid="metric-container"] {
  background: var(--panel) !important;
  border: 1px solid var(--border) !important;
  border-radius: 4px !important;
  padding: 20px 24px !important;
}

[data-testid="metric-container"] label {
  font-family: 'DM Mono', monospace !important;
  font-size: 10px !important;
  text-transform: uppercase !important;
  letter-spacing: 0.08em !important;
  color: var(--muted) !important;
}

[data-testid="metric-container"] [data-testid="stMetricValue"] {
  font-family: 'DM Serif Display', serif !important;
  font-size: 32px !important;
  color: #f1f5f9 !important;
}

/* ── Sliders ─────────────────────────────────────────────────── */
.stSlider > div > div > div {
  background: var(--blue) !important;
}

/* ── Select/Input ────────────────────────────────────────────── */
.stSelectbox > div > div,
.stTextInput > div > div > input {
  background: var(--navy3) !important;
  border-color: var(--border) !important;
  color: var(--text) !important;
  border-radius: 2px !important;
}

/* ── Tabs ────────────────────────────────────────────────────── */
.stTabs [data-baseweb="tab-list"] {
  background: transparent !important;
  border-bottom: 1px solid var(--border) !important;
}

.stTabs [data-baseweb="tab"] {
  background: transparent !important;
  color: var(--muted) !important;
  font-family: 'DM Mono', monospace !important;
  font-size: 11px !important;
  text-transform: uppercase !important;
  letter-spacing: 0.06em !important;
}

.stTabs [aria-selected="true"] {
  color: var(--blue2) !important;
  border-bottom: 2px solid var(--blue2) !important;
}

/* ── Divider ─────────────────────────────────────────────────── */
hr {
  border-color: var(--border) !important;
  margin: 2rem 0 !important;
}

/* ── Expander ────────────────────────────────────────────────── */
.streamlit-expanderHeader {
  background: var(--navy3) !important;
  border: 1px solid var(--border) !important;
  border-radius: 4px !important;
  color: var(--accent) !important;
  font-family: 'DM Mono', monospace !important;
  font-size: 12px !important;
}

/* ── Code blocks ─────────────────────────────────────────────── */
.stCodeBlock {
  border: 1px solid var(--border) !important;
  border-radius: 4px !important;
}

/* ── Dataframe ───────────────────────────────────────────────── */
.stDataFrame {
  border: 1px solid var(--border) !important;
  border-radius: 4px !important;
}

/* ── Spinner ─────────────────────────────────────────────────── */
.stSpinner > div { border-top-color: var(--blue2) !important; }

/* ── Hide branding ───────────────────────────────────────────── */
#MainMenu, footer, header { visibility: hidden; }

/* ── Pulse animation ─────────────────────────────────────────── */
@keyframes pulse {
  0%,100% { opacity: 1; }
  50%      { opacity: 0.3; }
}
</style>
"""


def inject_css():
    st.markdown(GLOBAL_CSS, unsafe_allow_html=True)


# ─────────────────────────────────────────────────────────────────────
# HTML COMPONENT LIBRARY
# Pure functions returning HTML strings.
# Every custom visual block is defined here once.
# ─────────────────────────────────────────────────────────────────────

def html_section_label(text: str) -> str:
    """Uppercase monospace label with trailing rule."""
    return f"""
    <div style="display:flex;align-items:center;gap:12px;
      font-family:'DM Mono',monospace;font-size:10px;
      text-transform:uppercase;letter-spacing:0.1em;
      color:{MUTED};margin-bottom:16px;">
      {text}
      <div style="flex:1;height:1px;background:{BORDER};"></div>
    </div>"""


def html_live_tag(text: str) -> str:
    """Animated pulsing cyan tag."""
    return f"""
    <div style="display:inline-flex;align-items:center;gap:8px;
      font-family:'DM Mono',monospace;font-size:11px;
      font-weight:500;letter-spacing:0.08em;
      text-transform:uppercase;color:{CYAN};
      border:1px solid rgba(6,182,212,0.3);
      padding:5px 14px;border-radius:2px;margin-bottom:24px;">
      <span style="width:6px;height:6px;border-radius:50%;
        background:{CYAN};display:inline-block;
        animation:pulse 2s infinite;"></span>
      {text}
    </div>"""


def html_hero_heading(title_html: str, subtitle: str) -> str:
    """Large serif hero heading."""
    return f"""
    <h1 style="font-family:'DM Serif Display',serif;
      font-size:clamp(40px,5.5vw,70px);font-weight:400;
      line-height:1.05;letter-spacing:-0.02em;
      color:#f1f5f9;margin:0 0 12px;">{title_html}</h1>
    <p style="font-size:17px;color:{ACCENT};font-weight:300;
      max-width:620px;margin:0 0 32px;line-height:1.6;">{subtitle}</p>"""


def html_stat_card(value: str, label: str) -> str:
    """Single stat card with serif number."""
    return f"""
    <div style="background:{PANEL};border:1px solid {BORDER};
      border-radius:4px;padding:20px 24px;text-align:center;">
      <div style="font-family:'DM Serif Display',serif;
        font-size:36px;line-height:1;color:#f1f5f9;
        margin-bottom:6px;">{value}</div>
      <div style="font-family:'DM Mono',monospace;font-size:10px;
        text-transform:uppercase;letter-spacing:0.08em;
        color:{MUTED};">{label}</div>
    </div>"""


def html_badge_strip(badges: list) -> str:
    """Row of monospace tech stack badges."""
    items = "".join(
        f"""<span style="font-size:10px;font-family:'DM Mono',monospace;
          text-transform:uppercase;letter-spacing:0.06em;
          padding:3px 10px;border-radius:2px;
          border:1px solid {BORDER};color:{MUTED};">{b}</span>"""
        for b in badges
    )
    return f'<div style="display:flex;flex-wrap:wrap;gap:8px;margin-top:20px;">{items}</div>'


def html_risk_badge(label: str) -> str:
    """Coloured pill: Low / Medium / High."""
    cfgs = {
        "High":   (f"rgba(239,68,68,0.15)",  "#f87171", f"rgba(239,68,68,0.3)"),
        "Medium": (f"rgba(245,158,11,0.15)", "#fbbf24", f"rgba(245,158,11,0.3)"),
        "Low":    (f"rgba(34,197,94,0.15)",  "#4ade80", f"rgba(34,197,94,0.3)"),
    }
    bg, txt, bdr = cfgs.get(label, cfgs["Low"])
    return f"""
    <span style="font-family:'DM Mono',monospace;font-size:11px;
      font-weight:500;text-transform:uppercase;letter-spacing:0.08em;
      padding:4px 14px;border-radius:2px;
      background:{bg};color:{txt};border:1px solid {bdr};">{label} Risk</span>"""


def html_prob_bar(label: str, value: float, color: str) -> str:
    """Single probability bar row."""
    pct = f"{value:.1f}%"
    return f"""
    <div style="display:flex;align-items:center;gap:10px;margin-bottom:10px;">
      <span style="font-family:'DM Mono',monospace;font-size:11px;
        color:{MUTED};width:52px;flex-shrink:0;">{label}</span>
      <div style="flex:1;height:4px;background:{NAVY3};border-radius:2px;overflow:hidden;">
        <div style="height:100%;border-radius:2px;
          background:{color};width:{value}%;"></div>
      </div>
      <span style="font-family:'DM Mono',monospace;font-size:11px;
        color:{ACCENT};width:40px;text-align:right;">{pct}</span>
    </div>"""


def html_rec_item(text: str) -> str:
    """Single recommendation row."""
    text = _fix_mojibake(text)
    return f"""
    <div style="display:flex;gap:10px;font-size:13px;
      color:{ACCENT};margin-bottom:8px;line-height:1.5;">
      <span style="color:{BLUE2};font-family:'DM Mono',monospace;
        flex-shrink:0;">→</span>{text}
    </div>"""


def html_fi_bar(feature: str, importance_pct: float, max_pct: float) -> str:
    """Feature importance row."""
    width = (importance_pct / max_pct) * 100 if max_pct else 0
    return f"""
    <div style="display:flex;align-items:center;gap:10px;margin-bottom:10px;">
      <span style="font-family:'DM Mono',monospace;font-size:11px;
        color:{MUTED};width:190px;flex-shrink:0;letter-spacing:0.03em;">{feature}</span>
      <div style="flex:1;height:4px;background:{NAVY3};border-radius:2px;overflow:hidden;">
        <div style="height:100%;border-radius:2px;
          background:{BLUE2};width:{width:.1f}%;"></div>
      </div>
      <span style="font-family:'DM Mono',monospace;font-size:11px;
        color:{ACCENT};width:44px;text-align:right;">{importance_pct:.1f}%</span>
    </div>"""


def html_arch_node(label: str, icon: str, highlight: bool = False) -> str:
    """Architecture flow node."""
    bg  = f"rgba(37,99,235,0.08)" if highlight else NAVY3
    bdr = f"rgba(37,99,235,0.5)"  if highlight else BORDER
    clr = BLUE2                   if highlight else ACCENT
    return f"""
    <div style="background:{bg};border:1px solid {bdr};
      border-radius:4px;padding:10px 16px;
      font-size:12px;font-family:'DM Mono',monospace;
      color:{clr};letter-spacing:0.04em;
      display:inline-flex;align-items:center;gap:8px;">{icon}&nbsp;{label}</div>"""


def html_card(content: str, border_color: str = None, extra_style: str = "") -> str:
    """Generic bordered card container."""
    bdr = f"border-left:3px solid {border_color};" if border_color else ""
    return f"""
    <div style="background:{PANEL};border:1px solid {BORDER};
      {bdr}border-radius:4px;padding:24px;{extra_style}">{content}</div>"""


def html_info_row(label: str, value: str, value_color: str = "#f1f5f9") -> str:
    """Key-value info row in monospace."""
    return f"""
    <div style="display:flex;justify-content:space-between;
      padding:8px 0;border-bottom:1px solid {BORDER};
      font-family:'DM Mono',monospace;font-size:12px;">
      <span style="color:{MUTED};">{label}</span>
      <span style="color:{value_color};">{value}</span>
    </div>"""


# ─────────────────────────────────────────────────────────────────────
# PLOTLY SHARED LAYOUT
# Apply to every fig.update_layout() call for consistency.
# ─────────────────────────────────────────────────────────────────────

def plotly_layout(**overrides) -> dict:
    base = dict(
        paper_bgcolor = "rgba(0,0,0,0)",
        plot_bgcolor  = "rgba(0,0,0,0)",
        font          = dict(family="DM Mono, monospace", color=ACCENT, size=11),
        margin        = dict(l=10, r=10, t=30, b=10),
        xaxis         = dict(gridcolor=BORDER, showline=False, zeroline=False,
                             tickfont=dict(size=10, color=MUTED)),
        yaxis         = dict(gridcolor=BORDER, showline=False, zeroline=False,
                             tickfont=dict(size=10, color=MUTED)),
        legend        = dict(bgcolor="rgba(0,0,0,0)", bordercolor=BORDER,
                             borderwidth=1, font=dict(size=10)),
    )
    base.update(overrides)
    return base


PLOTLY_CFG = {"displayModeBar": False}


# ─────────────────────────────────────────────────────────────────────
# API HELPERS
# ─────────────────────────────────────────────────────────────────────

def api_get(endpoint: str):
    try:
        r = requests.get(f"{API_URL}{endpoint}", timeout=8)
        return r.json() if r.status_code == 200 else None
    except Exception:
        return None


def api_post(endpoint: str, payload: dict):
    try:
        r = requests.post(f"{API_URL}{endpoint}", json=payload, timeout=10)
        return r.json() if r.status_code == 200 else None
    except Exception:
        return None


def api_online() -> bool:
    try:
        return requests.get(f"{API_URL}/health", timeout=4).status_code == 200
    except Exception:
        return False


def local_metadata() -> dict | None:
    if META_PATH.exists():
        with open(META_PATH) as f:
            return json.load(f)
    return None


def local_dataset() -> pd.DataFrame | None:
    if not DATA_PATH.exists():
        return None
    df = pd.read_csv(DATA_PATH)
    for col in ["commune_name", "department_code", "commune_code"]:
        if col in df.columns:
            df[col] = df[col].astype(str).map(_fix_mojibake)
    return df


# ─────────────────────────────────────────────────────────────────────
# OFFLINE PREDICTION HEURISTIC
# Used when the FastAPI backend is not running.
# Mirrors the model's key decision factors — for demo purposes only.
# ─────────────────────────────────────────────────────────────────────

RECS = {
    "High": [
        "⚠️ URGENT: Apply for Zone Sous-Dense (ZSD) designation",
        "Request deployment from pacte contre les déserts médicaux",
        "Deploy mobile medical units for immediate coverage",
        "Fast-track Maison de Santé Pluriprofessionnelle creation",
        "Partner with medical schools for rural internship placements",
    ],
    "Medium": [
        "Prioritize recruitment of 2+ new GPs within 18 months",
        "Establish a Maison de Santé Pluriprofessionnelle (MSP)",
        "Expand teleconsultation infrastructure",
        "Apply for Zone d'Intervention Prioritaire (ZIP) status",
    ],
    "Low": [
        "Maintain current healthcare infrastructure",
        "Monitor GP age distribution for upcoming retirements",
        "Continue telemedicine investment to stay resilient",
    ],
}


def offline_predict(f: dict) -> dict:
    """Heuristic risk score when API is offline."""
    score  = 0
    score += max(0, (50  - f.get("specialist_density",  30)) / 50) * 35
    score += max(0, (10  - f.get("gp_count",              5)) / 10) * 30
    score += (f.get("elderly_ratio", 0.25) - 0.15) / 0.35 * 15
    score += max(0, (f.get("avg_gp_age", 52) - 50) / 20) * 10
    score -= f.get("urban_score",  1) * 5
    score -= f.get("wealth_index", 0.5) * 8
    score -= f.get("population_growth_rate", 0) * 50
    score  = max(0, min(100, score))

    if score > 60:
        lvl, lbl = 2, "High"
        ph = min(0.99, 0.55 + score / 200)
        pm = (1 - ph) * 0.6
        pl = 1 - ph - pm
    elif score > 35:
        lvl, lbl = 1, "Medium"
        pm = min(0.80, 0.40 + score / 200)
        ph = (1 - pm) * 0.4
        pl = 1 - pm - ph
    else:
        lvl, lbl = 0, "Low"
        pl = min(0.99, 0.60 + (100 - score) / 200)
        pm = (1 - pl) * 0.5
        ph = 1 - pl - pm

    conf = max(pl, pm, ph)
    return {
        "risk_level":      lvl,
        "risk_label":      lbl,
        "confidence":      round(conf, 4),
        "probabilities":   {"low": round(pl,4), "medium": round(pm,4), "high": round(ph,4)},
        "recommendations": RECS[lbl],
        "model_version":   "offline-heuristic",
    }


# ─────────────────────────────────────────────────────────────────────
# PRESET COMMUNE PROFILES
# ─────────────────────────────────────────────────────────────────────

PROFILES = {
    "Rural · Creuse (expected High)": {
        "commune_name":"Ahun","commune_code":"23001","department_code":"23",
        "population_log":7.1,"urban_score":0,"elderly_ratio":0.38,
        "gp_count":1,"specialist_density":25.0,"pharmacy_score":1.5,
        "wealth_index":0.32,"population_growth_rate":-0.02,
        "avg_gp_age":61.0,"teleconsult_score":1.5,
    },
    "Peri-urban · Indre (expected Medium)": {
        "commune_name":"La Châtre","commune_code":"36032","department_code":"36",
        "population_log":9.4,"urban_score":1,"elderly_ratio":0.24,
        "gp_count":8,"specialist_density":60.0,"pharmacy_score":2.8,
        "wealth_index":0.52,"population_growth_rate":-0.01,
        "avg_gp_age":54.0,"teleconsult_score":2.5,
    },
    "Urban · Paris (expected Low)": {
        "commune_name":"Paris 8e","commune_code":"75108","department_code":"75",
        "population_log":11.5,"urban_score":3,"elderly_ratio":0.16,
        "gp_count":85,"specialist_density":180.0,"pharmacy_score":4.8,
        "wealth_index":0.92,"population_growth_rate":0.02,
        "avg_gp_age":49.0,"teleconsult_score":4.5,
    },
    "Custom": {},
}


# ─────────────────────────────────────────────────────────────────────
# SIDEBAR
# ─────────────────────────────────────────────────────────────────────

def render_sidebar() -> str:
    # Apply staged navigation before instantiating the sidebar widget keyed `nav_page`
    if "_nav_target" in st.session_state:
        st.session_state["nav_page"] = st.session_state.pop("_nav_target")

    # Brand
    st.sidebar.markdown(f"""
    <div style="padding:24px 16px 20px;">
      <div style="font-family:'DM Serif Display',serif;
        font-size:26px;color:#f1f5f9;line-height:1;">
        Med<span style="color:{BLUE2};font-style:italic;">Accès</span></div>
      <div style="font-family:'DM Mono',monospace;font-size:10px;
        text-transform:uppercase;letter-spacing:0.08em;
        color:{MUTED};margin-top:6px;">Medical Desert AI · France</div>
    </div>
    <hr style="border-color:{BORDER};margin:0 16px 16px;">
    """, unsafe_allow_html=True)

    # Nav (keyed so pages can "redirect" by updating session_state)
    pages = ["🏠  Home", "🎯  Predict", "📊  Analytics", "🤖  Model"]
    if "nav_page" not in st.session_state:
        st.session_state["nav_page"] = pages[0]
    page = st.sidebar.radio(
        "nav",
        pages,
        label_visibility="collapsed",
        key="nav_page",
    )

    # API status
    online = api_online()
    sc = GREEN if online else RED
    sl = "API · online" if online else "API · offline"

    st.sidebar.markdown("<div style='height:24px'></div>", unsafe_allow_html=True)
    st.sidebar.markdown(f"""
    <div style="display:flex;align-items:center;gap:8px;
      padding:10px 16px;margin:0 8px;
      border:1px solid {BORDER};border-radius:4px;background:{NAVY};">
      <div style="width:7px;height:7px;border-radius:50%;background:{sc};"></div>
      <span style="font-family:'DM Mono',monospace;font-size:11px;
        text-transform:uppercase;letter-spacing:0.06em;color:{sc};">{sl}</span>
    </div>
    """, unsafe_allow_html=True)

    # Footer
    st.sidebar.markdown(f"""
    <div style="position:fixed;bottom:0;left:0;width:250px;
      padding:16px;border-top:1px solid {BORDER};background:{NAVY2};">
      <div style="font-family:'DM Mono',monospace;font-size:10px;
        text-transform:uppercase;letter-spacing:0.06em;color:{MUTED};
        margin-bottom:4px;"> Data For Betterment</div>
      <div style="font-family:'DM Mono',monospace;font-size:10px;
        color:#374151;">· France · 2025</div>
    </div>
    """, unsafe_allow_html=True)

    return page


# ─────────────────────────────────────────────────────────────────────
# PAGE 1 — HOME
# ─────────────────────────────────────────────────────────────────────

def page_home():

    # Hero
    st.markdown(html_live_tag("Medacces Project · Data For Betterment · FRANCE"),
                unsafe_allow_html=True)
    st.markdown(html_hero_heading(
        f'Med<em style="color:{BLUE2};">Accès</em>',
        "AI-powered medical desert risk prediction across 34,000 French communes — "
        "built end-to-end with FastAPI, XGBoost, and MLflow."
    ), unsafe_allow_html=True)

    # CTA (jump into demo / metrics)
    cta1, cta2, _ = st.columns([1, 1, 2])
    with cta1:
        if st.button("🎯  Open live demo", use_container_width=True, key="cta_open_demo"):
            _nav_to("🎯  Predict")
    with cta2:
        if st.button("🤖  View model metrics", use_container_width=True, key="cta_view_model"):
            _nav_to("🤖  Model")

    # Meta row
    st.markdown(f"""
    <div style="display:flex;flex-wrap:wrap;gap:28px;margin-bottom:36px;">
      <span style="font-size:13px;color:{MUTED};">📍 France · 87% medically underserved</span>
      <span style="font-size:13px;color:{MUTED};">👥 8M+ people lack GP access</span>
      <span style="font-size:13px;color:{MUTED};">🧠 XGBoost · 63.7% accuracy</span>
      <span style="font-size:13px;color:{MUTED};">🔬 DREES 2024 data · official thresholds</span>
    </div>
    """, unsafe_allow_html=True)

    # Stat cards
    c1, c2, c3, c4 = st.columns(4)
    for col, val, lbl in zip(
        [c1, c2, c3, c4],
        ["34k", "10", "3", "35"],
        ["Communes analysed", "ML features", "Risk classes", "pytest tests"],
    ):
        with col:
            st.markdown(html_stat_card(val, lbl), unsafe_allow_html=True)

    # Badge strip
    st.markdown(html_badge_strip([
        "FastAPI", "XGBoost", "MLflow", "scikit-learn",
        "PostgreSQL", "Docker", "Railway", "Pydantic",
        "pytest", "SQLAlchemy", "Streamlit", "Plotly",
    ]), unsafe_allow_html=True)

    st.markdown("<hr>", unsafe_allow_html=True)

    # Problem / Solution
    st.markdown(html_section_label("The problem · the solution"), unsafe_allow_html=True)

    col_prob, col_sol = st.columns(2, gap="large")
    with col_prob:
        st.markdown(html_card(f"""
          <div style="font-family:'DM Mono',monospace;font-size:10px;
            text-transform:uppercase;letter-spacing:0.08em;
            color:{RED};margin-bottom:12px;">Healthcare crisis</div>
          <p style="font-size:14px;color:{ACCENT};line-height:1.7;margin:0;">
            87% of France is affected by medical deserts.
            Over <strong style="color:#f1f5f9;">8 million people</strong> cannot
            find a GP within a reasonable distance. The French government's
            <em>pacte de lutte contre les déserts médicaux</em> (2025)
            identified 151 priority zones — but no automated risk scoring
            system exists to guide allocation.
          </p>
        """, border_color=RED), unsafe_allow_html=True)

    with col_sol:
        st.markdown(html_card(f"""
          <div style="font-family:'DM Mono',monospace;font-size:10px;
            text-transform:uppercase;letter-spacing:0.08em;
            color:{GREEN};margin-bottom:12px;">The solution</div>
          <p style="font-size:14px;color:{ACCENT};line-height:1.7;margin:0;">
            MedAccès predicts which communes are at risk
            <strong style="color:#f1f5f9;">before</strong> they become deserts —
            enabling proactive resource allocation. An XGBoost classifier trained
            on INSEE demographic data and DREES healthcare supply features
            scores every commune as <strong style="color:{GREEN};">Low</strong> /
            <strong style="color:{AMBER};">Medium</strong> /
            <strong style="color:{RED};">High</strong> risk with confidence scores
            and government-aligned recommendations.
          </p>
        """, border_color=GREEN), unsafe_allow_html=True)

    st.markdown("<hr>", unsafe_allow_html=True)

    # Architecture
    st.markdown(html_section_label("System architecture"), unsafe_allow_html=True)

    steps = [
        ("INSEE API",       "🗄",  False),
        ("Feature eng.",    "⚙",   False),
        ("XGBoost",         "🧠",  True),
        ("MLflow",          "📈",  False),
        ("FastAPI",         "🔌",  True),
        ("PostgreSQL",      "🗃",  False),
        ("Railway · Live",  "☁",  True),
    ]
    arch_html = f'<div style="display:flex;flex-wrap:wrap;align-items:center;gap:8px;margin-bottom:24px;">'
    for i, (lbl, ico, hl) in enumerate(steps):
        arch_html += html_arch_node(lbl, ico, hl)
        if i < len(steps) - 1:
            arch_html += f'<span style="color:{BORDER};font-family:monospace;font-size:18px;">→</span>'
    arch_html += "</div>"
    st.markdown(arch_html, unsafe_allow_html=True)

    st.markdown("<hr>", unsafe_allow_html=True)

    # Tech cards
    st.markdown(html_section_label("Engineering highlights"), unsafe_allow_html=True)

    tech_cards = [
        ("FastAPI + Pydantic",      "🔌",
         "10 production endpoints · background tasks · CORS middleware · lifespan startup · global error handlers"),
        ("XGBoost + scikit-learn",  "🧠",
         "sklearn Pipeline (scaler + model) · 3-model comparison · stratified CV · joblib serialization"),
        ("MLflow experiment tracking", "📈",
         "Every run logged · hyperparams · metrics per class · model artifact · SQLite backend"),
        ("PostgreSQL + SQLAlchemy", "🗃",
         "Prediction audit log · drift monitoring · GROUP BY aggregations · alembic migrations"),
        ("Docker · multi-stage",    "🐳",
         "Builder + runtime stages · model trained at build time · docker-compose · Railway CI/CD"),
        ("pytest · 35 tests",       "🧪",
         "In-memory SQLite · dependency_overrides · domain sanity tests · AAA pattern · conftest fixtures"),
    ]

    cols = st.columns(3)
    for i, (title, icon, desc) in enumerate(tech_cards):
        with cols[i % 3]:
            st.markdown(f"""
            <div style="background:{NAVY3};border:1px solid {BORDER};
              border-radius:4px;padding:20px;margin-bottom:12px;">
              <div style="font-size:20px;margin-bottom:10px;">{icon}</div>
              <div style="font-size:13px;font-weight:500;
                color:#f1f5f9;margin-bottom:6px;">{title}</div>
              <div style="font-size:12px;color:{MUTED};line-height:1.5;">{desc}</div>
            </div>
            """, unsafe_allow_html=True)

    # API endpoints table
    st.markdown("<hr>", unsafe_allow_html=True)
    st.markdown(html_section_label("API endpoints"), unsafe_allow_html=True)

    endpoints = [
        ("POST", "/predict/commune",      "Single commune ML inference · returns risk + confidence + recs",    True),
        ("POST", "/predict/batch",        "Batch inference up to 500 communes · matrix operation",            True),
        ("GET",  "/model/info",           "Current model metadata · metrics · feature importance",            False),
        ("GET",  "/model/health",         "Liveness check · used by load balancers and monitoring",           False),
        ("POST", "/model/retrain",        "Trigger background retraining · returns 202 Accepted",             True),
        ("GET",  "/communes/stats",       "Aggregated prediction statistics · GROUP BY department",           False),
        ("GET",  "/communes/dataset",     "Browse training dataset · filter by dept / risk · paginated",      False),
        ("GET",  "/communes/predictions", "Prediction audit log · newest first · filterable",                 False),
        ("GET",  "/",                     "API navigation root",                                              False),
        ("GET",  "/health",               "App liveness check",                                               False),
    ]

    method_colors = {"GET": BLUE2, "POST": GREEN}
    for method, path, desc, important in endpoints:
        mc = method_colors.get(method, ACCENT)
        bg = f"rgba(37,99,235,0.05)" if important else "transparent"
        st.markdown(f"""
        <div style="display:flex;align-items:center;gap:16px;
          padding:10px 12px;border-bottom:1px solid {BORDER};
          background:{bg};border-radius:2px;">
          <span style="font-family:'DM Mono',monospace;font-size:11px;
            color:{mc};font-weight:500;width:36px;flex-shrink:0;">{method}</span>
          <span style="font-family:'DM Mono',monospace;font-size:12px;
            color:#f1f5f9;width:220px;flex-shrink:0;">{path}</span>
        <span style="font-size:12px;color:{MUTED};">{desc}</span>
        </div>
        """, unsafe_allow_html=True)

    if st.button("🎯  Go to Predict", use_container_width=True, key="home_go_predict"):
        _nav_to("🎯  Predict")


# ─────────────────────────────────────────────────────────────────────
# PAGE 2 — PREDICT
# ─────────────────────────────────────────────────────────────────────

def page_predict():

    st.markdown(html_section_label("Live inference demo"), unsafe_allow_html=True)
    st.markdown(html_hero_heading(
        "Predict commune risk",
        "Submit commune features to the XGBoost model and receive a full risk assessment"
    ), unsafe_allow_html=True)

    # Profile selector (matches the HTML design: 3 quick presets + Custom)
    quick_profiles = [
        "Rural · Creuse (expected High)",
        "Peri-urban · Indre (expected Medium)",
        "Urban · Paris (expected Low)",
        "Custom",
    ]
    if "predict_profile" not in st.session_state:
        st.session_state["predict_profile"] = quick_profiles[0]

    st.markdown(
        "<div style=\"font-size:11px;font-family:'DM Mono',monospace;"
        "text-transform:uppercase;letter-spacing:0.06em;color:var(--muted);"
        "margin-bottom:6px;\">Commune profile</div>",
        unsafe_allow_html=True,
    )
    b1, b2, b3, b4 = st.columns(4)
    for col, label in zip(
        [b1, b2, b3, b4],
        ["Rural · Creuse", "Peri-urban", "Urban · Paris", "Custom"],
    ):
        full = next((p for p in quick_profiles if p.startswith(label)), label)
        if label == "Custom":
            full = "Custom"
        with col:
            is_active = st.session_state["predict_profile"].startswith(label) if label != "Custom" else (st.session_state["predict_profile"] == "Custom")
            if _st_button(
                label,
                key=f"prof_{label}",
                use_container_width=True,
                type="primary" if is_active else "secondary",
            ):
                st.session_state["predict_profile"] = full
                st.rerun()

    # Fallback selectbox (useful on very small screens / keyboard nav)
    profile_name = st.selectbox(
        "Load preset profile",
        quick_profiles,
        index=quick_profiles.index(st.session_state["predict_profile"]),
        label_visibility="collapsed",
    )
    if profile_name != st.session_state["predict_profile"]:
        st.session_state["predict_profile"] = profile_name

    P = PROFILES[st.session_state["predict_profile"]]

    st.markdown("<div style='height:12px'></div>", unsafe_allow_html=True)

    col_form, col_result = st.columns([1, 1], gap="large")

    # ── LEFT: Form ────────────────────────────────────────────────────
    with col_form:

        st.markdown(html_section_label("Commune identifiers"), unsafe_allow_html=True)
        commune_name    = st.text_input("Commune name",    value=P.get("commune_name", ""))
        commune_code    = st.text_input("INSEE code",      value=P.get("commune_code", ""))
        department_code = st.text_input("Department code", value=P.get("department_code", ""))

        st.markdown("<div style='height:8px'></div>", unsafe_allow_html=True)
        st.markdown(html_section_label("Demographics"), unsafe_allow_html=True)

        # Population: user sets raw value, we log-transform it
        pop_raw      = st.slider("Population",
            min_value=50, max_value=2_000_000,
            value=int(math.expm1(P.get("population_log", 7.1))),
            format="%d",
            help="Automatically log-transformed for the ML model",
        )
        pop_log = round(math.log1p(pop_raw), 4)
        st.markdown(f"""
        <div style="font-family:'DM Mono',monospace;font-size:11px;
          color:{MUTED};margin-top:-8px;margin-bottom:12px;">
          log₁p({pop_raw:,}) = {pop_log}</div>
        """, unsafe_allow_html=True)

        urban_score = st.select_slider("Urban classification",
            options=[0, 1, 2, 3],
            value=P.get("urban_score", 0),
            format_func=lambda x: ["0 — Rural","1 — Peri-urban","2 — Small city","3 — Urban"][x],
        )

        elderly_ratio = st.slider("Elderly ratio (65+)",
            0.05, 0.55, float(P.get("elderly_ratio", 0.38)), 0.01, format="%.2f",
            help="Proportion of population aged 65+",
        )

        pop_growth = st.slider("Population growth rate",
            -0.05, 0.05, float(P.get("population_growth_rate", -0.02)), 0.001, format="%.3f",
            help="Negative = population declining → doctors leave",
        )

        st.markdown("<div style='height:8px'></div>", unsafe_allow_html=True)
        st.markdown(html_section_label("Healthcare supply"), unsafe_allow_html=True)

        gp_count = st.slider("GP count",
            0, 200, int(P.get("gp_count", 1)),
            help="Number of general practitioners in the commune",
        )

        specialist_density = st.slider("Specialist density (per 100k)",
            0.0, 300.0, float(P.get("specialist_density", 25.0)), 1.0,
        )

        pharmacy_score = st.slider("Pharmacy access score",
            1.0, 5.0, float(P.get("pharmacy_score", 1.5)), 0.1, format="%.1f",
            help="1 = no pharmacy within 10km · 5 = multiple pharmacies",
        )

        avg_gp_age = st.slider("Average GP age",
            35.0, 72.0, float(P.get("avg_gp_age", 61.0)), 0.5, format="%.1f",
            help="Older average age = higher imminent retirement risk",
        )

        teleconsult_score = st.slider("Teleconsultation score",
            1.0, 5.0, float(P.get("teleconsult_score", 1.5)), 0.1, format="%.1f",
            help="1 = no telemedicine coverage · 5 = excellent infrastructure",
        )

        st.markdown("<div style='height:8px'></div>", unsafe_allow_html=True)
        st.markdown(html_section_label("Socioeconomic"), unsafe_allow_html=True)

        wealth_index = st.slider("Wealth index",
            0.0, 1.0, float(P.get("wealth_index", 0.32)), 0.01, format="%.2f",
            help="0 = deprived · 1 = affluent",
        )

        st.markdown("<div style='height:16px'></div>", unsafe_allow_html=True)
        run_btn = st.button("⚡  Run inference", use_container_width=True, key="predict_run_inference")

    # ── RIGHT: Result ─────────────────────────────────────────────────
    with col_result:

        payload = {
            "commune_name":           commune_name or None,
            "commune_code":           commune_code or None,
            "department_code":        department_code or None,
            "population_log":         pop_log,
            "urban_score":            urban_score,
            "elderly_ratio":          elderly_ratio,
            "gp_count":               gp_count,
            "specialist_density":     specialist_density,
            "pharmacy_score":         pharmacy_score,
            "wealth_index":           wealth_index,
            "population_growth_rate": pop_growth,
            "avg_gp_age":             avg_gp_age,
            "teleconsult_score":      teleconsult_score,
        }

        if run_btn:
            with st.spinner("Running inference..."):
                result = api_post("/predict/commune", payload)
                if result is None:
                    result = offline_predict(payload)
                    result["_source"] = "offline"
                else:
                    result["_source"] = "api"
            st.session_state["pred_result"]  = result
            st.session_state["pred_payload"] = payload

        result = st.session_state.get("pred_result")

        if result is None:
            # Empty state
            st.markdown(f"""
            <div style="background:{PANEL};border:1px dashed {BORDER};
              border-radius:4px;padding:64px 24px;text-align:center;
              margin-top:80px;">
              <div style="font-size:36px;margin-bottom:16px;">🎯</div>
              <div style="font-family:'DM Mono',monospace;font-size:12px;
                text-transform:uppercase;letter-spacing:0.06em;color:#374151;">
                Configure features on the left<br>then click Run inference
              </div>
            </div>
            """, unsafe_allow_html=True)
            return

        # Unpack result
        risk_label = result.get("risk_label", "Low")
        confidence = result.get("confidence", 0.0)
        probs      = result.get("probabilities", {})
        recs       = result.get("recommendations", [])
        source     = result.get("_source", "api")
        version    = result.get("model_version", "—")

        src_color  = GREEN if source == "api" else AMBER
        src_text   = "Live API prediction" if source == "api" else "Offline · heuristic (API not running)"

        risk_hex   = {"High": RED, "Medium": AMBER, "Low": GREEN}.get(risk_label, GREEN)

        st.markdown(html_section_label("Prediction result"), unsafe_allow_html=True)

        # Source indicator
        st.markdown(f"""
        <div style="font-family:'DM Mono',monospace;font-size:10px;
          text-transform:uppercase;letter-spacing:0.06em;
          color:{src_color};margin-bottom:16px;
          display:flex;align-items:center;gap:6px;">
          <span style="width:6px;height:6px;border-radius:50%;
            background:{src_color};display:inline-block;"></span>
          {src_text}
        </div>
        """, unsafe_allow_html=True)

        # Main result card
        st.markdown(f"""
        <div style="background:{PANEL};border:1px solid {BORDER};
          border-top:3px solid {risk_hex};
          border-radius:0 0 4px 4px;padding:24px;margin-bottom:16px;">
          <div style="display:flex;align-items:flex-start;
            justify-content:space-between;margin-bottom:16px;">
            <div>
              {html_risk_badge(risk_label)}
              <div style="font-family:'DM Mono',monospace;font-size:11px;
                color:{MUTED};margin-top:10px;">
                {_fix_mojibake(commune_name) or '—'} · Dept {_fix_mojibake(department_code) or '—'}
              </div>
            </div>
            <div style="text-align:right;">
              <div style="font-family:'DM Serif Display',serif;
                font-size:44px;line-height:1;color:#f1f5f9;">
                {confidence:.1%}</div>
              <div style="font-family:'DM Mono',monospace;font-size:10px;
                text-transform:uppercase;letter-spacing:0.08em;color:{MUTED};">
                confidence</div>
            </div>
          </div>
          {html_prob_bar("Low",    probs.get("low",    0) * 100, GREEN)}
          {html_prob_bar("Medium", probs.get("medium", 0) * 100, AMBER)}
          {html_prob_bar("High",   probs.get("high",   0) * 100, RED)}
        </div>
        """, unsafe_allow_html=True)

        # Donut chart
        fig_donut = go.Figure(go.Pie(
            labels   = ["Low risk", "Medium risk", "High risk"],
            values   = [probs.get("low",0), probs.get("medium",0), probs.get("high",0)],
            hole     = 0.72,
            marker   = dict(colors=[GREEN, AMBER, RED]),
            textinfo = "none",
            hovertemplate="%{label}: %{percent}<extra></extra>",
        ))
        fig_donut.update_layout(
            **plotly_layout(height=200, margin=dict(l=0,r=0,t=0,b=0), showlegend=False),
            annotations=[dict(
                text=f"<b>{confidence:.0%}</b>",
                x=0.5, y=0.5,
                font=dict(size=22, color="#f1f5f9", family="DM Serif Display"),
                showarrow=False,
            )],
        )
        st.plotly_chart(fig_donut, use_container_width=True, config=PLOTLY_CFG)

        # Recommendations
        st.markdown(html_section_label("Government-aligned recommendations"),
                    unsafe_allow_html=True)
        recs_html = "".join(html_rec_item(r) for r in recs)
        st.markdown(html_card(recs_html), unsafe_allow_html=True)

        # Model version
        st.markdown(f"""
        <div style="font-family:'DM Mono',monospace;font-size:10px;
          color:{MUTED};margin-top:12px;text-align:right;">
          model version: {version}
        </div>
        """, unsafe_allow_html=True)

        # Raw JSON
        with st.expander("View raw API response"):
            clean = {k: v for k, v in result.items() if k != "_source"}
            st.code(json.dumps(clean, indent=2), language="json")

    # ── Batch section ─────────────────────────────────────────────────
    st.markdown("<hr>", unsafe_allow_html=True)
    st.markdown(html_section_label("Batch scoring · POST /predict/batch"),
                unsafe_allow_html=True)
    st.markdown(f"""
    <p style="font-size:14px;color:{MUTED};margin-bottom:16px;">
      Scores all three reference communes in one request.
      The batch endpoint accepts up to 500 communes and processes
      them in a single matrix operation — much faster than looping
      over the single-prediction endpoint.
    </p>
    """, unsafe_allow_html=True)

    if st.button("⚡  Run batch inference on 3 reference communes", key="predict_run_batch"):
        batch_communes = [
            PROFILES["Rural · Creuse (expected High)"],
            PROFILES["Peri-urban · Indre (expected Medium)"],
            PROFILES["Urban · Paris (expected Low)"],
        ]
        with st.spinner("Scoring 3 communes..."):
            batch_resp = api_post("/predict/batch", {"communes": batch_communes})

        if batch_resp:
            results_list = batch_resp.get("results", [])
        else:
            results_list = [offline_predict(p) for p in batch_communes]
            for i, p in enumerate(batch_communes):
                results_list[i]["commune_name"] = p.get("commune_name", "")

        # Enrich names
        names = ["Ahun · Creuse", "La Châtre · Indre", "Paris 8e"]
        cols3 = st.columns(3)
        for i, (col, res) in enumerate(zip(cols3, results_list)):
            with col:
                rl    = res.get("risk_label", "Low")
                conf  = res.get("confidence", 0)
                r_hex = {"High": RED, "Medium": AMBER, "Low": GREEN}.get(rl, GREEN)
                cname = _fix_mojibake(res.get("commune_name")) or names[i]

                st.markdown(f"""
                <div style="background:{PANEL};border:1px solid {BORDER};
                  border-top:3px solid {r_hex};
                  border-radius:0 0 4px 4px;padding:20px;">
                  <div style="font-family:'DM Mono',monospace;font-size:10px;
                    text-transform:uppercase;letter-spacing:0.06em;
                    color:{MUTED};margin-bottom:10px;">{cname}</div>
                  {html_risk_badge(rl)}
                  <div style="font-family:'DM Serif Display',serif;
                    font-size:32px;color:#f1f5f9;margin-top:14px;">
                    {conf:.1%}</div>
                  <div style="font-family:'DM Mono',monospace;font-size:10px;
                    text-transform:uppercase;letter-spacing:0.06em;
                    color:{MUTED};">confidence</div>
                </div>
                """, unsafe_allow_html=True)


# ─────────────────────────────────────────────────────────────────────
# PAGE 3 — ANALYTICS
# ─────────────────────────────────────────────────────────────────────

def page_analytics():

    st.markdown(html_section_label("Data · statistics · audit log"),
                unsafe_allow_html=True)
    st.markdown(html_hero_heading(
        "Analytics",
        "Explore the commune dataset and track every prediction made"
    ), unsafe_allow_html=True)

    # Load data
    df = None
    api_data = api_get("/communes/dataset?limit=500")
    if api_data and api_data.get("data"):
        df = pd.DataFrame(api_data["data"])
    if df is None:
        df = local_dataset()
    if df is None:
        st.warning("Dataset not found. Run: `python ml/data_ingestion.py`")
        return

    total  = len(df)
    high_n = len(df[df["medical_desert_risk"] == 2])
    med_n  = len(df[df["medical_desert_risk"] == 1])
    low_n  = len(df[df["medical_desert_risk"] == 0])
    dept_n = df["department_code"].nunique()

    # Summary metrics
    st.markdown(html_section_label("Dataset overview"), unsafe_allow_html=True)
    c1, c2, c3, c4, c5 = st.columns(5)
    with c1: st.metric("Total communes", f"{total:,}")
    with c2: st.metric("🔴 High risk",   f"{high_n:,}", f"{high_n/total:.0%}")
    with c3: st.metric("🟡 Medium risk", f"{med_n:,}",  f"{med_n/total:.0%}")
    with c4: st.metric("🟢 Low risk",    f"{low_n:,}",  f"{low_n/total:.0%}")
    with c5: st.metric("Departments",    str(dept_n))

    st.markdown("<div style='height:8px'></div>", unsafe_allow_html=True)

    # Risk distribution + top at-risk departments
    col_pie, col_bar = st.columns(2, gap="large")

    with col_pie:
        st.markdown(html_section_label("Risk distribution"), unsafe_allow_html=True)
        dist = df["medical_desert_risk"].value_counts().sort_index()

        fig_pie = go.Figure(go.Pie(
            labels   = [RISK_LABELS[i] for i in dist.index],
            values   = dist.values,
            marker   = dict(colors=[RISK_COLORS[i] for i in dist.index]),
            hole     = 0.6,
            textinfo = "none",
            hovertemplate="%{label}: %{value} (%{percent})<extra></extra>",
        ))
        fig_pie.update_layout(
            **plotly_layout(height=280, showlegend=True,
                legend=dict(orientation="h", x=0.5, xanchor="center",
                            y=-0.1, font=dict(size=11, color=ACCENT))),
            annotations=[dict(
                text=f"<b>{total:,}</b>",
                x=0.5, y=0.5,
                font=dict(size=18, color="#f1f5f9", family="DM Serif Display"),
                showarrow=False,
            )],
        )
        st.plotly_chart(fig_pie, use_container_width=True, config=PLOTLY_CFG)

    with col_bar:
        st.markdown(html_section_label("Top 10 at-risk departments"), unsafe_allow_html=True)
        dept_risk = (
            df[df["medical_desert_risk"] == 2]
            .groupby("department_code").size()
            .sort_values(ascending=True).tail(10)
        )
        fig_bar = go.Figure(go.Bar(
            x=dept_risk.values, y=dept_risk.index,
            orientation="h",
            marker=dict(
                color=dept_risk.values,
                colorscale=[[0, AMBER], [1, RED]],
                line=dict(width=0),
            ),
            hovertemplate="Dept %{y}: %{x} high-risk communes<extra></extra>",
        ))
        fig_bar.update_layout(
            **plotly_layout(height=280),
            xaxis=dict(title=None, gridcolor=BORDER),
            yaxis=dict(title=None, gridcolor=BORDER),
        )
        st.plotly_chart(fig_bar, use_container_width=True, config=PLOTLY_CFG)

    # Feature distributions
    st.markdown("<hr>", unsafe_allow_html=True)
    st.markdown(html_section_label("Feature distributions by risk level"),
                unsafe_allow_html=True)
    st.markdown(f"""
    <p style="font-size:14px;color:{MUTED};margin-bottom:16px;">
      Box plots showing how each feature differs across risk classes.
      A good feature shows clearly separated distributions.
    </p>
    """, unsafe_allow_html=True)

    feat_map = {
        "gp_density_per_100k": "GP density (per 100k)",
        "elderly_ratio":        "Elderly ratio",
        "avg_gp_age":           "Average GP age",
        "wealth_index":         "Wealth index",
    }

    fc1, fc2 = st.columns(2)
    for idx, (feat, label) in enumerate(feat_map.items()):
        if feat not in df.columns:
            continue
        with (fc1 if idx % 2 == 0 else fc2):
            fig_box = go.Figure()
            for r in [0, 1, 2]:
                subset = df[df["medical_desert_risk"] == r][feat].dropna()
                fig_box.add_trace(go.Box(
                    y=subset, name=RISK_LABELS[r],
                    marker=dict(color=RISK_COLORS[r], size=3),
                    line=dict(color=RISK_COLORS[r], width=1.5),
                    fillcolor=RISK_COLORS[r] + "22",
                    boxmean=True,
                ))
            fig_box.update_layout(
                **plotly_layout(
                    title=dict(text=label, font=dict(size=13, color=ACCENT)),
                    height=260,
                    showlegend=False,
                )
            )
            st.plotly_chart(fig_box, use_container_width=True, config=PLOTLY_CFG)

    # GP density vs elderly ratio scatter
    st.markdown("<hr>", unsafe_allow_html=True)
    st.markdown(html_section_label("GP density vs elderly ratio"),
                unsafe_allow_html=True)
    st.markdown(f"""
    <p style="font-size:14px;color:{MUTED};margin-bottom:8px;">
      The two most predictive features plotted against each other.
      High-risk communes cluster in the top-left: high elderly ratio,
      low GP density.
    </p>
    """, unsafe_allow_html=True)

    df_sc = df.dropna(subset=["gp_density_per_100k","elderly_ratio","medical_desert_risk"])
    df_sc = df_sc.sample(min(600, len(df_sc)), random_state=42)

    fig_sc = go.Figure()
    for r in [2, 1, 0]:
        sub = df_sc[df_sc["medical_desert_risk"] == r]
        fig_sc.add_trace(go.Scatter(
            x=sub["elderly_ratio"], y=sub["gp_density_per_100k"],
            mode="markers",
            name=f"{RISK_LABELS[r]} risk",
            marker=dict(color=RISK_COLORS[r], size=5, opacity=0.7),
            hovertemplate=(
                f"Elderly: %{{x:.0%}}<br>GP density: %{{y:.0f}}<br>"
                f"Risk: {RISK_LABELS[r]}<extra></extra>"
            ),
        ))
    fig_sc.update_layout(
        **plotly_layout(height=340),
        xaxis=dict(title="Elderly ratio (65+)", tickformat=".0%"),
        yaxis=dict(title="GP density per 100k"),
    )
    st.plotly_chart(fig_sc, use_container_width=True, config=PLOTLY_CFG)

    # Prediction audit log
    st.markdown("<hr>", unsafe_allow_html=True)
    st.markdown(html_section_label("Prediction audit log"), unsafe_allow_html=True)

    logs_resp = api_get("/communes/predictions?limit=50")
    if logs_resp and logs_resp.get("predictions"):
        logs   = logs_resp["predictions"]
        df_log = pd.DataFrame(logs)
        total_logged = logs_resp.get("total", 0)

        st.markdown(f"""
        <div style="font-family:'DM Mono',monospace;font-size:11px;
          color:{MUTED};margin-bottom:12px;">
          {total_logged} predictions logged · showing 50 most recent
        </div>
        """, unsafe_allow_html=True)

        keep = [c for c in ["id","commune_name","department_code",
                             "risk_label","confidence","created_at"]
                if c in df_log.columns]
        df_log = df_log[keep]
        if "commune_name" in df_log.columns:
            df_log["commune_name"] = df_log["commune_name"].astype(str).map(_fix_mojibake)
        if "confidence" in df_log.columns:
            df_log["confidence"] = df_log["confidence"].apply(lambda x: f"{x:.1%}")

        st.dataframe(df_log, use_container_width=True, hide_index=True)
    else:
        st.markdown(f"""
        <div style="background:{PANEL};border:1px dashed {BORDER};
          border-radius:4px;padding:24px;text-align:center;
          font-family:'DM Mono',monospace;font-size:12px;
          text-transform:uppercase;letter-spacing:0.06em;color:#374151;">
          No predictions logged yet · go to Predict page to run inference
        </div>
        """, unsafe_allow_html=True)

    # Dataset explorer
    st.markdown("<hr>", unsafe_allow_html=True)
    st.markdown(html_section_label("Dataset explorer"), unsafe_allow_html=True)

    fc, rc = st.columns(2)
    with fc:
        dept_filter = st.text_input("Filter by department code", placeholder="e.g. 23")
    with rc:
        risk_filter = st.selectbox("Filter by risk level",
                                   ["All", "Low (0)", "Medium (1)", "High (2)"])

    df_view = df.copy()
    if dept_filter:
        df_view = df_view[df_view["department_code"].astype(str) == dept_filter]
    if risk_filter != "All":
        rv = int(risk_filter.split("(")[1].rstrip(")"))
        df_view = df_view[df_view["medical_desert_risk"] == rv]

    st.markdown(f"""
    <div style="font-family:'DM Mono',monospace;font-size:11px;
      color:{MUTED};margin-bottom:12px;">{len(df_view):,} communes</div>
    """, unsafe_allow_html=True)

    show_cols = [c for c in [
        "commune_name","department_code","population",
        "gp_density_per_100k","elderly_ratio",
        "avg_gp_age","wealth_index","medical_desert_risk",
    ] if c in df_view.columns]

    st.dataframe(df_view[show_cols].head(100),
                 use_container_width=True, hide_index=True)


# ─────────────────────────────────────────────────────────────────────
# PAGE 4 — MODEL
# ─────────────────────────────────────────────────────────────────────

def page_model():

    st.markdown(html_section_label("ML model · performance · MLflow"),
                unsafe_allow_html=True)
    st.markdown(html_hero_heading(
        "Model Performance",
        "XGBoost classifier — metrics, feature importance, confusion matrix, experiment tracking"
    ), unsafe_allow_html=True)

    # Load metadata
    meta = api_get("/model/info") or local_metadata()
    if meta is None:
        st.warning("Model metadata not found. Run: `python ml/pipeline.py`")
        return

    metrics = meta.get("metrics", {})
    fi      = meta.get("feature_importance", {})

    # Key metrics
    st.markdown(html_section_label("Key metrics"), unsafe_allow_html=True)

    c1, c2, c3, c4 = st.columns(4)
    acc  = metrics.get("accuracy",    0)
    f1   = metrics.get("f1_weighted", 0)
    cvf1 = metrics.get("cv_f1_mean",  0)
    cvsd = metrics.get("cv_f1_std",   0)

    with c1: st.metric("Algorithm",    meta.get("model_name","—").upper())
    with c2: st.metric("Accuracy",     f"{acc:.2%}",  f"+{(acc-0.33):.0%} vs random")
    with c3: st.metric("F1 weighted",  f"{f1:.4f}")
    with c4: st.metric("CV F1 5-fold", f"{cvf1:.4f}", f"±{cvsd:.4f}")

    st.markdown(f"""
    <div style="font-family:'DM Mono',monospace;font-size:11px;
      color:{MUTED};margin:12px 0 24px;">
      Version: {meta.get('version','—')} &nbsp;·&nbsp;
      Trained: {meta.get('trained_at','—')[:19]} &nbsp;·&nbsp;
      Features: {len(meta.get('features',[]))}
    </div>
    """, unsafe_allow_html=True)

    st.markdown("<hr>", unsafe_allow_html=True)

    # Feature importance + Confusion matrix
    col_fi, col_cm = st.columns(2, gap="large")

    with col_fi:
        st.markdown(html_section_label("Feature importance · XGBoost"),
                    unsafe_allow_html=True)

        if fi:
            fi_sorted = sorted(fi.items(), key=lambda x: x[1], reverse=True)
            max_val   = fi_sorted[0][1] * 100

            # Text bars
            for feat, imp in fi_sorted:
                st.markdown(html_fi_bar(feat, imp * 100, max_val),
                            unsafe_allow_html=True)

            st.markdown("<div style='height:12px'></div>", unsafe_allow_html=True)

            # Plotly bar chart
            fi_df = pd.DataFrame(fi_sorted, columns=["feature","importance"])
            fi_df = fi_df.sort_values("importance")

            fig_fi = go.Figure(go.Bar(
                x=fi_df["importance"] * 100,
                y=fi_df["feature"],
                orientation="h",
                marker=dict(
                    color=fi_df["importance"] * 100,
                    colorscale=[[0, NAVY3], [1, BLUE2]],
                    line=dict(width=0),
                ),
                hovertemplate="%{y}: %{x:.2f}%<extra></extra>",
            ))
            fig_fi.update_layout(
                **plotly_layout(height=320,
                    xaxis=dict(title="Importance (%)", gridcolor=BORDER),
                    yaxis=dict(title=None),
                    margin=dict(l=10,r=10,t=10,b=10)),
            )
            st.plotly_chart(fig_fi, use_container_width=True, config=PLOTLY_CFG)

    with col_cm:
        st.markdown(html_section_label("Confusion matrix"),
                    unsafe_allow_html=True)
        st.markdown(f"""
        <p style="font-size:13px;color:{MUTED};margin-bottom:12px;line-height:1.5;">
          Rows = actual class · Columns = predicted class.
          Strong diagonal = good discrimination.
        </p>
        """, unsafe_allow_html=True)

        cm = meta.get("confusion_matrix")
        if cm:
            cm_arr = np.array(cm)
            labels = ["Low", "Medium", "High"]

            fig_cm = go.Figure(go.Heatmap(
                z=cm_arr, x=labels, y=labels,
                colorscale=[[0, NAVY], [0.5, "#1f3a6e"], [1, BLUE]],
                text=cm_arr, texttemplate="%{text}",
                textfont=dict(size=18, color="#f1f5f9"),
                showscale=False,
                hovertemplate="Actual: %{y}<br>Predicted: %{x}<br>Count: %{z}<extra></extra>",
            ))
            fig_cm.update_layout(
                **plotly_layout(height=300, margin=dict(l=10,r=10,t=10,b=10)),
                xaxis=dict(title="Predicted", side="bottom", gridcolor="rgba(0,0,0,0)"),
                yaxis=dict(title="Actual",    gridcolor="rgba(0,0,0,0)"),
            )
            st.plotly_chart(fig_cm, use_container_width=True, config=PLOTLY_CFG)

        # Per-class report
        cr = meta.get("class_report", {})
        if cr:
            st.markdown("<div style='height:8px'></div>", unsafe_allow_html=True)
            st.markdown(html_section_label("Per-class report"),
                        unsafe_allow_html=True)
            for cls in ["Low", "Medium", "High"]:
                if cls not in cr:
                    continue
                d   = cr[cls]
                clr = {"Low": GREEN, "Medium": AMBER, "High": RED}[cls]
                st.markdown(f"""
                <div style="display:flex;align-items:center;gap:16px;
                  padding:10px 0;border-bottom:1px solid {BORDER};
                  font-family:'DM Mono',monospace;font-size:12px;">
                  <span style="color:{clr};width:60px;flex-shrink:0;
                    text-transform:uppercase;letter-spacing:0.06em;">{cls}</span>
                  <span style="color:{MUTED};width:110px;">
                    Prec: <span style="color:#f1f5f9;">{d.get('precision',0):.3f}</span></span>
                  <span style="color:{MUTED};width:110px;">
                    Rec: <span style="color:#f1f5f9;">{d.get('recall',0):.3f}</span></span>
                  <span style="color:{MUTED};width:110px;">
                    F1: <span style="color:#f1f5f9;">{d.get('f1-score',0):.3f}</span></span>
                  <span style="color:#374151;margin-left:auto;">
                    n={int(d.get('support',0)):,}</span>
                </div>
                """, unsafe_allow_html=True)

    # Model comparison
    st.markdown("<hr>", unsafe_allow_html=True)
    st.markdown(html_section_label("Model comparison · all candidates"),
                unsafe_allow_html=True)
    st.markdown(f"""
    <p style="font-size:14px;color:{MUTED};margin-bottom:16px;">
      Three model architectures were trained and compared.
      The winner is selected by cross-validation F1 score —
      the most honest metric (averaged over 5 folds, not a single split).
    </p>
    """, unsafe_allow_html=True)

    models_comp = [
        ("Logistic Regression", 0.5813, 0.5701, 0.5744, False),
        ("Random Forest",       0.6120, 0.6034, 0.6089, False),
        ("XGBoost",             0.6367, 0.6203, 0.6361, True),
    ]

    fig_comp = go.Figure()
    metric_labels = ["Accuracy", "F1 weighted", "CV F1"]
    for name, acc_v, f1_v, cv_v, winner in models_comp:
        color = BLUE2 if winner else "#374151"
        width = 2.0   if winner else 1.0
        fig_comp.add_trace(go.Scatter(
            x=metric_labels, y=[acc_v, f1_v, cv_v],
            mode="lines+markers",
            name=name + (" ✓" if winner else ""),
            line=dict(color=color, width=width),
            marker=dict(color=color, size=8),
            hovertemplate=f"{name}<br>%{{x}}: %{{y:.4f}}<extra></extra>",
        ))
    fig_comp.update_layout(
        **plotly_layout(height=300,
            yaxis=dict(range=[0.50,0.70], title="Score", gridcolor=BORDER),
            xaxis=dict(title=None),
            legend=dict(orientation="h", x=0.5, xanchor="center", y=-0.25)),
    )
    st.plotly_chart(fig_comp, use_container_width=True, config=PLOTLY_CFG)

    # Comparison table
    df_c = pd.DataFrame({
        "Model":   [m[0] for m in models_comp],
        "Accuracy":       [f"{m[1]:.4f}" for m in models_comp],
        "F1 weighted":    [f"{m[2]:.4f}" for m in models_comp],
        "CV F1 (5-fold)": [f"{m[3]:.4f}" for m in models_comp],
        "Status":         ["—", "—", "✓ Selected"],
    })
    st.dataframe(df_c, use_container_width=True, hide_index=True)

    # Feature documentation
    st.markdown("<hr>", unsafe_allow_html=True)
    st.markdown(html_section_label("Feature documentation"), unsafe_allow_html=True)

    FEAT_DOCS = {
        "population_log":          ("log₁p(population)",      "Log transform compresses Paris vs. tiny village scale gap."),
        "urban_score":             ("Urban classification",    "0=rural, 1=peri-urban, 2=small city, 3=urban. INSEE typology."),
        "elderly_ratio":           ("Elderly ratio",           "Proportion aged 65+. Higher = more GP demand, lower supply."),
        "gp_count":                ("GP count",                "Absolute number of GPs. Directly measures supply."),
        "specialist_density":      ("Specialist density",      "Specialists per 100k. Top feature — correlated with GP access."),
        "pharmacy_score":          ("Pharmacy access",         "1–5 scale. First contact point where GPs are absent."),
        "wealth_index":            ("Wealth index",            "0–1 proxy. Wealthier areas can attract and retain doctors."),
        "population_growth_rate":  ("Growth rate",             "Negative = declining → doctors leave and don't return."),
        "avg_gp_age":              ("Avg GP age",              "Forward-looking signal. High age = imminent retirement wave."),
        "teleconsult_score":       ("Teleconsult score",       "1–5. Telemedicine partially compensates for GP shortage."),
    }

    features = meta.get("features", list(FEAT_DOCS.keys()))
    fd1, fd2 = st.columns(2)
    for i, feat in enumerate(features):
        doc = FEAT_DOCS.get(feat, (feat, ""))
        imp = fi.get(feat, 0)
        with (fd1 if i % 2 == 0 else fd2):
            st.markdown(f"""
            <div style="display:flex;gap:14px;padding:12px 0;
              border-bottom:1px solid {BORDER};">
              <div style="font-family:'DM Mono',monospace;font-size:10px;
                text-transform:uppercase;letter-spacing:0.06em;
                color:{BLUE2};width:44px;flex-shrink:0;padding-top:2px;">
                {imp*100:.1f}%</div>
              <div>
                <div style="font-size:13px;font-weight:500;
                  color:#f1f5f9;margin-bottom:2px;">{doc[0]}</div>
                <div style="font-size:12px;color:{MUTED};
                  line-height:1.4;">{doc[1]}</div>
              </div>
            </div>
            """, unsafe_allow_html=True)

    # MLflow section
    st.markdown("<hr>", unsafe_allow_html=True)
    st.markdown(html_section_label("MLflow experiment tracking"), unsafe_allow_html=True)

    st.markdown(html_card(f"""
      <div style="font-family:'DM Mono',monospace;font-size:12px;
        color:{ACCENT};line-height:2.2;">
        Every training run is automatically logged to MLflow, including:<br>
        <span style="color:{BLUE2};">→</span>&nbsp;
          Model hyperparameters (n_estimators, max_depth, learning_rate)<br>
        <span style="color:{BLUE2};">→</span>&nbsp;
          Evaluation metrics (accuracy, F1, CV F1 per fold, ±std)<br>
        <span style="color:{BLUE2};">→</span>&nbsp;
          Per-class precision · recall · F1 for Low / Medium / High<br>
        <span style="color:{BLUE2};">→</span>&nbsp;
          Full sklearn Pipeline artifact (scaler + model bundled together)<br>
        <span style="color:{BLUE2};">→</span>&nbsp;
          Training dataset size, feature count, test/train split info
      </div>
    """), unsafe_allow_html=True)

    st.markdown("<div style='height:16px'></div>", unsafe_allow_html=True)

    ml1, ml2 = st.columns(2)
    with ml1:
        st.code(
            "# Launch MLflow UI\nmlflow ui \\\n"
            "  --backend-store-uri \\\n"
            "  sqlite:///models/mlruns/mlflow.db\n\n"
            "# Open: http://localhost:5000",
            language="bash",
        )
    with ml2:
        st.code(
            "# Trigger retraining via API\ncurl -X POST \\\n"
            "  http://localhost:8000/model/retrain\n\n"
            "# Check new model:\ncurl http://localhost:8000/model/info",
            language="bash",
        )

    st.markdown("<div style='height:8px'></div>", unsafe_allow_html=True)
    if st.button("🔄  Trigger model retraining via API", key="model_trigger_retrain"):
        resp = api_post("/model/retrain", {})
        if resp:
            st.success(f"✅ {resp.get('message','Retraining started.')}")
        else:
            st.warning("API offline. Run locally: `python ml/pipeline.py`")

    # Quick start guide
    st.markdown("<hr>", unsafe_allow_html=True)
    st.markdown(html_section_label("Quick start"), unsafe_allow_html=True)

    st.code(
        "# 1. Install dependencies\npip install -r requirements.txt\n\n"
        "# 2. Generate dataset\npython ml/data_ingestion.py\n\n"
        "# 3. Train the model\npython ml/pipeline.py\n\n"
        "# 4. Start the API\nuvicorn app.main:app --reload\n\n"
        "# 5. Start this dashboard\nstreamlit run dashboard/app.py\n\n"
        "# 6. Run tests\npytest tests/ -v",
        language="bash",
    )


# ─────────────────────────────────────────────────────────────────────
# MAIN
# ─────────────────────────────────────────────────────────────────────

def main():
    inject_css()
    page = render_sidebar()

    if   "Home"      in page: page_home()
    elif "Predict"   in page: page_predict()
    elif "Analytics" in page: page_analytics()
    elif "Model"     in page: page_model()


if __name__ == "__main__":
    main()
