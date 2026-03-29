"""
QoSBuddy M6 — PREMIUM NOC Dashboard  ✦ DSO3.2
================================================
Team: PingWin | ESPRIT 2025-2026

Design direction:
  Cinematic dark intelligence — deep space black backgrounds, razor-thin
  neon accent lines, glassmorphism cards, animated ring gauges, and a
  typography system built around IBM Plex Mono + Syne for a premium
  telco/military-ops aesthetic.

Run: streamlit run streamlit_dashboard.py
"""

import os, io, sys, time, logging
from pathlib import Path
from datetime import datetime

# Ensure /app is on sys.path so `from rag.rag_pipeline import ...` works
# when Streamlit runs from dashboard/streamlit_dashboard.py inside Docker
_app_root = Path(__file__).resolve().parent.parent
if str(_app_root) not in sys.path:
    sys.path.insert(0, str(_app_root))

import numpy as np
import pandas as pd
import streamlit as st
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots

# ── Page config — must be FIRST ────────────────────────────────────────────
st.set_page_config(
    page_title="QoSBuddy · NOC",
    page_icon="📡",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ══════════════════════════════════════════════════════════════════════════════
#  DESIGN SYSTEM
# ══════════════════════════════════════════════════════════════════════════════

# Ensure theme is initialized before CSS blocks reference it
if "theme" not in st.session_state:
    st.session_state["theme"] = "dark"

THEME = """
<link rel="preconnect" href="https://fonts.googleapis.com">
<link rel="preconnect" href="https://fonts.gstatic.com" crossorigin>
<link href="https://fonts.googleapis.com/css2?family=Syne:wght@400;600;700;800&family=IBM+Plex+Mono:wght@300;400;500;600&family=Inter:wght@300;400;500&display=swap" rel="stylesheet">

<style>
:root {
  --void:        #030508;
  --surface-0:   #080c12;
  --surface-1:   #0d1420;
  --surface-2:   #111826;
  --surface-3:   #16202e;
  --glass:       rgba(13,20,32,0.72);
  --glass-light: rgba(22,32,46,0.60);
  --cyan:        #00d4ff;
  --cyan-dim:    rgba(0,212,255,0.2);
  --cyan-glow:   rgba(0,212,255,0.09);
  --emerald:     #00e5a0;
  --emerald-dim: rgba(0,229,160,0.2);
  --amber:       #ffb83f;
  --amber-dim:   rgba(255,184,63,0.2);
  --rose:        #ff4d6d;
  --rose-dim:    rgba(255,77,109,0.2);
  --violet:      #a78bfa;
  --violet-dim:  rgba(167,139,250,0.2);
  --text-1: #e8f0fe;
  --text-2: #94a3b8;
  --text-3: #4a5568;
  --border: rgba(0,212,255,0.12);
  --border-bright: rgba(0,212,255,0.30);
  --r-sm: 8px; --r-md: 14px; --r-lg: 20px; --r-xl: 28px;
  --font-display: 'Syne', sans-serif;
  --font-mono:    'IBM Plex Mono', monospace;
  --font-body:    'Inter', sans-serif;
}
*, *::before, *::after { box-sizing: border-box; }
html, body, .stApp, [data-testid="stAppViewContainer"] {
  background: var(--void) !important;
  color: var(--text-1) !important;
  font-family: var(--font-body) !important;
}
[data-testid="stAppViewContainer"]::before {
  content: ''; position: fixed; inset: 0; pointer-events: none; z-index: 0;
  background-image:
    linear-gradient(rgba(0,212,255,0.025) 1px, transparent 1px),
    linear-gradient(90deg, rgba(0,212,255,0.025) 1px, transparent 1px);
  background-size: 48px 48px;
}
[data-testid="stSidebar"] {
  background: var(--surface-0) !important;
  border-right: 1px solid var(--border) !important;
}
[data-testid="stSidebar"] > div { padding-top: 0 !important; }
/* Keep sidebar toggle always visible — only hide text branding */
#MainMenu { visibility: hidden; }
footer    { visibility: hidden; }
header[data-testid="stHeader"] {
  background: var(--void) !important;
  box-shadow: none !important;
}
/* Hide deploy button text but keep the toolbar area for layout */
[data-testid="stToolbar"] {
  right: 0 !important;
  display: none !important;
}
[data-testid="stDeployButton"] { display: none !important; }
.stDeployButton { display: none !important; }

/* Sidebar toggle — collapsed state */
[data-testid="stSidebarNavButton"],
button[data-testid="stSidebarNavButton"] {
  display: flex !important; visibility: visible !important;
}
[data-testid="collapsedControl"],
div[data-testid="collapsedControl"] {
  display: flex !important; visibility: visible !important;
  position: fixed !important; left: 0 !important; top: 50vh !important;
  transform: translateY(-50%) !important; z-index: 999999 !important;
  background: var(--surface-1, #0d1420) !important;
  border: 1.5px solid rgba(0,212,255,0.55) !important;
  border-left: none !important; border-radius: 0 10px 10px 0 !important;
  padding: 10px 5px !important; box-shadow: 4px 0 20px rgba(0,212,255,0.2) !important;
  opacity: 1 !important;
}
[data-testid="collapsedControl"] button,
[data-testid="collapsedControl"] [data-testid="stBaseButton-headerNoPadding"] {
  color: #00d4ff !important; background: transparent !important; border: none !important;
  visibility: visible !important; display: flex !important;
}
[data-testid="collapsedControl"] svg { fill: #00d4ff !important; stroke: #00d4ff !important; }

/* Sidebar expand/collapse button inside header */
button[kind="headerNoPadding"] {
  color: #00d4ff !important;
  visibility: visible !important;
  display: flex !important;
}
.block-container { padding: 1.5rem 2rem 2rem !important; max-width: 100% !important; }
::-webkit-scrollbar { width: 4px; height: 4px; }
::-webkit-scrollbar-track { background: var(--surface-0); }
::-webkit-scrollbar-thumb { background: var(--cyan-dim); border-radius: 4px; }

/* ── SIDEBAR ── */
.sb-brand { padding: 28px 20px 20px; border-bottom: 1px solid var(--border); margin-bottom: 8px; }
.sb-logo-ring {
  width: 52px; height: 52px; border-radius: 50%;
  background: conic-gradient(var(--cyan), var(--emerald), var(--cyan));
  display: flex; align-items: center; justify-content: center;
  margin: 0 auto 12px;
  box-shadow: 0 0 24px var(--cyan-dim), 0 0 48px var(--cyan-glow);
  animation: spin-ring 8s linear infinite;
}
@keyframes spin-ring { from { transform: rotate(0deg); } to { transform: rotate(360deg); } }
.sb-logo-inner {
  width: 42px; height: 42px; border-radius: 50%;
  background: var(--surface-0);
  display: flex; align-items: center; justify-content: center; font-size: 20px;
}
.sb-title { text-align: center; font-family: var(--font-display); font-size: 18px; font-weight: 800; color: var(--text-1); letter-spacing: 0.05em; }
.sb-sub { text-align: center; font-family: var(--font-mono); font-size: 9px; color: var(--text-3); text-transform: uppercase; letter-spacing: 0.15em; margin-top: 3px; }
.status-row { display: flex; align-items: center; gap: 6px; padding: 8px 16px; margin: 0 0 4px; }
.status-dot { width: 6px; height: 6px; border-radius: 50%; background: var(--emerald); box-shadow: 0 0 8px var(--emerald); animation: pulse-dot 2s ease-in-out infinite; }
@keyframes pulse-dot { 0%, 100% { opacity: 1; transform: scale(1); } 50% { opacity: 0.6; transform: scale(0.85); } }
.status-text { font-family: var(--font-mono); font-size: 10px; color: var(--emerald); letter-spacing: 0.1em; }

/* ── PAGE HEADER ── */
.page-header { display: flex; align-items: flex-start; gap: 16px; padding: 24px 0 20px; margin-bottom: 8px; border-bottom: 1px solid var(--border); }
.page-header-icon { width: 48px; height: 48px; border-radius: var(--r-md); background: linear-gradient(135deg, var(--cyan-dim), var(--emerald-dim)); border: 1px solid var(--border-bright); display: flex; align-items: center; justify-content: center; font-size: 22px; flex-shrink: 0; box-shadow: 0 0 20px var(--cyan-glow); }
.page-header-title { font-family: var(--font-display); font-size: 26px; font-weight: 800; color: var(--text-1); line-height: 1.1; letter-spacing: -0.02em; }
.page-header-sub { font-family: var(--font-mono); font-size: 11px; color: var(--text-3); margin-top: 4px; text-transform: uppercase; letter-spacing: 0.12em; }

/* ── KPI CARDS ── */
.kpi-card { background: var(--glass); border: 1px solid var(--border); border-radius: var(--r-lg); padding: 24px 20px 20px; backdrop-filter: blur(16px); position: relative; overflow: hidden; transition: all 0.3s cubic-bezier(0.4,0,0.2,1); cursor: default; }
.kpi-card::before { content: ''; position: absolute; top: 0; left: 0; right: 0; height: 2px; }
.kpi-card:hover { transform: translateY(-3px); border-color: var(--border-bright); box-shadow: 0 12px 40px rgba(0,0,0,0.4); }
.kpi-card.ok::before  { background: linear-gradient(90deg, transparent, var(--emerald), transparent); }
.kpi-card.warn::before { background: linear-gradient(90deg, transparent, var(--amber), transparent); }
.kpi-card.crit::before { background: linear-gradient(90deg, transparent, var(--rose), transparent); }
.kpi-label { font-family: var(--font-mono); font-size: 9px; text-transform: uppercase; letter-spacing: 0.2em; color: var(--text-3); margin-bottom: 10px; }
.kpi-value { font-family: var(--font-display); font-size: 38px; font-weight: 800; line-height: 1; letter-spacing: -0.03em; }
.kpi-card.ok   .kpi-value { color: var(--emerald); }
.kpi-card.warn .kpi-value { color: var(--amber); }
.kpi-card.crit .kpi-value { color: var(--rose); }
.kpi-unit { font-family: var(--font-mono); font-size: 11px; color: var(--text-3); margin-left: 4px; }
.kpi-bar-wrap { height: 3px; background: var(--surface-3); border-radius: 2px; margin-top: 14px; overflow: hidden; }
.kpi-bar { height: 100%; border-radius: 2px; }
.kpi-card.ok   .kpi-bar { background: var(--emerald); box-shadow: 0 0 8px var(--emerald); }
.kpi-card.warn .kpi-bar { background: var(--amber);   box-shadow: 0 0 8px var(--amber); }
.kpi-card.crit .kpi-bar { background: var(--rose);    box-shadow: 0 0 8px var(--rose); }
.kpi-status { font-family: var(--font-mono); font-size: 10px; margin-top: 8px; }
.kpi-card.ok   .kpi-status { color: var(--emerald); }
.kpi-card.warn .kpi-status { color: var(--amber); }
.kpi-card.crit .kpi-status { color: var(--rose); }

/* ── GLASS PANEL ── */
.glass-panel { background: var(--glass); border: 1px solid var(--border); border-radius: var(--r-lg); padding: 24px; backdrop-filter: blur(16px); margin-bottom: 16px; }
.glass-panel-title { font-family: var(--font-mono); font-size: 10px; text-transform: uppercase; letter-spacing: 0.2em; color: var(--text-3); margin-bottom: 16px; display: flex; align-items: center; gap: 8px; }
.glass-panel-title::after { content: ''; flex: 1; height: 1px; background: linear-gradient(90deg, var(--border), transparent); }

/* ── PILLS ── */
.pill { display: inline-flex; align-items: center; gap: 5px; padding: 3px 10px; border-radius: 20px; font-family: var(--font-mono); font-size: 10px; font-weight: 500; letter-spacing: 0.08em; text-transform: uppercase; }
.pill-crit { background: var(--rose-dim); color: var(--rose); border: 1px solid var(--rose-dim); }
.pill-warn { background: var(--amber-dim); color: var(--amber); border: 1px solid var(--amber-dim); }
.pill-ok   { background: var(--emerald-dim); color: var(--emerald); border: 1px solid var(--emerald-dim); }

/* ── STAT CHIPS ── */
.stat-row { display: flex; gap: 12px; margin-bottom: 20px; flex-wrap: wrap; }
.stat-chip { flex: 1; min-width: 100px; background: var(--surface-2); border: 1px solid var(--border); border-radius: var(--r-md); padding: 16px 18px; display: flex; flex-direction: column; gap: 4px; }
.stat-chip-label { font-family: var(--font-mono); font-size: 9px; text-transform: uppercase; letter-spacing: 0.18em; color: var(--text-3); }
.stat-chip-value { font-family: var(--font-display); font-size: 28px; font-weight: 800; line-height: 1; }
.stat-chip.total .stat-chip-value { color: var(--cyan); }
.stat-chip.crit  .stat-chip-value { color: var(--rose); }
.stat-chip.warn  .stat-chip-value { color: var(--amber); }
.stat-chip.ok    .stat-chip-value { color: var(--emerald); }

/* ── TABLE ── */
.styled-table { width: 100%; border-collapse: collapse; font-family: var(--font-mono); font-size: 12px; }
.styled-table th { text-align: left; padding: 10px 12px; background: var(--surface-2); color: var(--text-3); font-weight: 500; text-transform: uppercase; letter-spacing: 0.12em; font-size: 9px; border-bottom: 1px solid var(--border); }
.styled-table td { padding: 10px 12px; color: var(--text-2); border-bottom: 1px solid var(--surface-2); }
.styled-table tr:hover td { background: var(--glass-light); }
.td-crit { color: var(--rose) !important; font-weight: 600; }
.td-warn { color: var(--amber) !important; font-weight: 600; }
.td-ok   { color: var(--emerald) !important; font-weight: 600; }

/* ── CAUSAL ── */
.causal-chain { display: flex; flex-direction: column; gap: 6px; padding: 16px 0; }
.causal-node { display: flex; align-items: center; gap: 12px; padding: 10px 16px; background: var(--surface-2); border: 1px solid var(--border); border-radius: var(--r-sm); font-family: var(--font-mono); font-size: 12px; color: var(--text-2); transition: all 0.2s; }
.causal-node:hover { border-color: var(--cyan); color: var(--text-1); }
.causal-arrow { font-family: var(--font-mono); font-size: 10px; color: var(--text-3); padding-left: 28px; line-height: 1; }
.action-badge { display: inline-flex; align-items: center; gap: 6px; padding: 6px 14px; border-radius: var(--r-sm); background: var(--cyan-glow); border: 1px solid var(--cyan-dim); font-family: var(--font-mono); font-size: 11px; color: var(--cyan); margin-top: 8px; }

/* ── CF PANEL ── */
.cf-result-panel { background: linear-gradient(135deg, var(--surface-2), var(--surface-3)); border: 1px solid var(--border-bright); border-radius: var(--r-lg); padding: 20px; display: flex; gap: 20px; align-items: stretch; margin-top: 16px; }
.cf-box { flex: 1; text-align: center; padding: 16px; background: var(--glass); border-radius: var(--r-md); }
.cf-box-label { font-family: var(--font-mono); font-size: 9px; text-transform: uppercase; letter-spacing: 0.18em; color: var(--text-3); margin-bottom: 8px; }
.cf-box-value { font-family: var(--font-display); font-size: 30px; font-weight: 800; line-height: 1; }
.cf-arrow { display: flex; align-items: center; color: var(--text-3); font-family: var(--font-mono); font-size: 18px; }

/* ── CHAT ── */
.chat-container { display: flex; flex-direction: column; gap: 12px; }
.chat-bubble { max-width: 88%; padding: 14px 18px; border-radius: var(--r-lg); line-height: 1.6; font-size: 14px; font-family: var(--font-body); animation: bubble-in 0.3s ease; }
@keyframes bubble-in { from { opacity: 0; transform: translateY(8px); } to { opacity: 1; transform: translateY(0); } }
.chat-user { align-self: flex-end; background: linear-gradient(135deg, #0e2040, #0a2a35); border: 1px solid var(--cyan-dim); color: var(--text-1); border-bottom-right-radius: 4px; }
.chat-bot  { align-self: flex-start; background: var(--surface-2); border: 1px solid var(--border); color: var(--text-1); border-bottom-left-radius: 4px; }
.chat-meta { font-family: var(--font-mono); font-size: 9px; color: var(--text-3); margin-bottom: 5px; text-transform: uppercase; letter-spacing: 0.15em; }
.chat-bot .chat-meta  { color: var(--emerald); }
.chat-user .chat-meta { color: var(--cyan); text-align: right; }
.chat-sources-wrap { align-self: flex-start; max-width: 88%; background: var(--glass); border: 1px solid var(--violet-dim); border-radius: var(--r-md); padding: 12px 16px; font-family: var(--font-mono); font-size: 10px; color: var(--text-3); }
.chat-source-item { padding: 4px 0; border-bottom: 1px solid var(--surface-3); display: flex; align-items: center; gap: 8px; }
.chat-source-item:last-child { border-bottom: none; }
.src-badge { padding: 1px 6px; border-radius: 4px; background: var(--violet-dim); color: var(--violet); font-size: 9px; text-transform: uppercase; }
.src-sim { color: var(--cyan); }
.suggestion-row { display: flex; gap: 8px; flex-wrap: wrap; margin-bottom: 16px; }

/* ── EXPORT ── */
.export-card { background: linear-gradient(135deg, var(--surface-1), var(--surface-2)); border: 1px solid var(--border); border-radius: var(--r-xl); padding: 32px; text-align: center; }
.export-icon  { font-size: 48px; margin-bottom: 12px; }
.export-title { font-family: var(--font-display); font-size: 22px; font-weight: 800; color: var(--text-1); margin-bottom: 6px; }
.export-sub   { font-family: var(--font-mono); font-size: 10px; color: var(--text-3); text-transform: uppercase; letter-spacing: 0.15em; }

/* ── STREAMLIT OVERRIDES ── */
div[data-testid="stMetric"] { display: none; }
div[data-baseweb="select"] > div:first-child { background: var(--surface-2) !important; border: 1px solid var(--border) !important; border-radius: var(--r-sm) !important; color: var(--text-1) !important; font-family: var(--font-mono) !important; font-size: 12px !important; }
div[data-testid="stSlider"] div[role="slider"] { background: var(--cyan) !important; box-shadow: 0 0 10px var(--cyan) !important; }
div[data-testid="stChatInput"] textarea { background: var(--surface-2) !important; border: 1px solid var(--border) !important; border-radius: var(--r-md) !important; color: var(--text-1) !important; }
div[data-testid="stChatInput"] button { background: var(--cyan) !important; border-radius: var(--r-sm) !important; }
.stButton > button { background: linear-gradient(135deg, #003d5c, #005a7a) !important; border: 1px solid var(--cyan-dim) !important; color: var(--cyan) !important; border-radius: var(--r-sm) !important; font-family: var(--font-mono) !important; font-size: 11px !important; letter-spacing: 0.08em !important; text-transform: uppercase !important; transition: all 0.2s !important; }
.stButton > button:hover { background: var(--cyan-dim) !important; border-color: var(--cyan) !important; box-shadow: 0 0 16px var(--cyan-glow) !important; }
.stRadio label, .stSelectbox label { font-family: var(--font-mono) !important; font-size: 10px !important; text-transform: uppercase !important; letter-spacing: 0.15em !important; color: var(--text-3) !important; }
details[data-testid="stExpander"] { background: var(--glass) !important; border: 1px solid var(--border) !important; border-radius: var(--r-md) !important; }
details[data-testid="stExpander"] summary { font-family: var(--font-mono) !important; font-size: 11px !important; color: var(--text-3) !important; }
div[data-testid="stDataFrame"] { border: 1px solid var(--border) !important; border-radius: var(--r-md) !important; overflow: hidden; }
.stAlert { background: var(--glass) !important; border: 1px solid var(--border) !important; border-radius: var(--r-md) !important; }
.stTextInput > div > div > input { background: var(--surface-2) !important; border: 1px solid var(--border) !important; border-radius: var(--r-sm) !important; color: var(--text-1) !important; font-family: var(--font-body) !important; }
.stCheckbox label p { font-family: var(--font-mono) !important; font-size: 11px !important; color: var(--text-2) !important; }
</style>
"""

st.markdown(THEME, unsafe_allow_html=True)

# ── Light theme override ────────────────────────────────────────────────────
if st.session_state.get("theme") == "light":
    st.markdown("""
    <style>
    :root {
      --void:        #f0f2f6;
      --surface-0:   #ffffff;
      --surface-1:   #f7f8fa;
      --surface-2:   #eef1f6;
      --surface-3:   #e2e6ed;
      --glass:       rgba(255,255,255,0.82);
      --glass-light: rgba(238,241,246,0.60);
      --cyan:        #0077aa;
      --cyan-dim:    rgba(0,119,170,0.15);
      --cyan-glow:   rgba(0,119,170,0.08);
      --emerald:     #059669;
      --emerald-dim: rgba(5,150,105,0.15);
      --amber:       #d97706;
      --amber-dim:   rgba(217,119,6,0.15);
      --rose:        #dc2626;
      --rose-dim:    rgba(220,38,38,0.15);
      --violet:      #7c3aed;
      --violet-dim:  rgba(124,58,237,0.15);
      --text-1: #1a1a2e;
      --text-2: #475569;
      --text-3: #94a3b8;
      --border: rgba(0,0,0,0.10);
      --border-bright: rgba(0,119,170,0.30);
    }
    html, body, .stApp, [data-testid="stAppViewContainer"] {
      background: var(--void) !important;
      color: var(--text-1) !important;
    }
    [data-testid="stAppViewContainer"]::before {
      background-image:
        linear-gradient(rgba(0,0,0,0.03) 1px, transparent 1px),
        linear-gradient(90deg, rgba(0,0,0,0.03) 1px, transparent 1px) !important;
    }
    [data-testid="stSidebar"] {
      background: var(--surface-0) !important;
      border-right: 1px solid var(--border) !important;
    }
    header[data-testid="stHeader"] {
      background: var(--void) !important;
    }
    [data-testid="collapsedControl"],
    div[data-testid="collapsedControl"] {
      background: var(--surface-0) !important;
      border: 1.5px solid rgba(0,119,170,0.40) !important;
      border-left: none !important;
      box-shadow: 4px 0 20px rgba(0,0,0,0.08) !important;
    }
    [data-testid="collapsedControl"] button,
    [data-testid="collapsedControl"] [data-testid="stBaseButton-headerNoPadding"] {
      color: var(--cyan) !important;
    }
    [data-testid="collapsedControl"] svg { fill: var(--cyan) !important; stroke: var(--cyan) !important; }
    .sb-logo-ring { box-shadow: 0 0 24px var(--cyan-dim), 0 0 48px var(--cyan-glow) !important; }
    .sb-title { color: var(--text-1) !important; }
    .status-dot { background: var(--emerald) !important; box-shadow: 0 0 8px var(--emerald) !important; }
    .status-text { color: var(--emerald) !important; }
    .sb-logo-inner { background: var(--surface-0) !important; }
    .kpi-card { background: var(--glass) !important; border: 1px solid var(--border) !important; backdrop-filter: blur(8px) !important; box-shadow: 0 1px 4px rgba(0,0,0,0.06) !important; }
    .kpi-card:hover { box-shadow: 0 8px 24px rgba(0,0,0,0.10) !important; }
    .glass-panel { background: var(--glass) !important; border: 1px solid var(--border) !important; }
    .stButton > button { background: linear-gradient(135deg, #e0f2fe, #bae6fd) !important; border: 1px solid var(--cyan-dim) !important; color: var(--cyan) !important; }
    .stButton > button:hover { background: var(--cyan-dim) !important; border-color: var(--cyan) !important; }
    div[data-baseweb="select"] > div:first-child { background: var(--surface-2) !important; border: 1px solid var(--border) !important; color: var(--text-1) !important; }
    div[data-testid="stChatInput"] textarea { background: var(--surface-2) !important; border: 1px solid var(--border) !important; color: var(--text-1) !important; }
    .stTextInput > div > div > input { background: var(--surface-2) !important; border: 1px solid var(--border) !important; color: var(--text-1) !important; }
    details[data-testid="stExpander"] { background: var(--glass) !important; border: 1px solid var(--border) !important; }
    .stAlert { background: var(--glass) !important; border: 1px solid var(--border) !important; }
    .chat-user { background: linear-gradient(135deg, #e0f2fe, #dbeafe) !important; border: 1px solid var(--cyan-dim) !important; color: var(--text-1) !important; }
    .chat-bot  { background: var(--surface-2) !important; border: 1px solid var(--border) !important; color: var(--text-1) !important; }
    .export-card { background: linear-gradient(135deg, var(--surface-1), var(--surface-2)) !important; }
    .cf-result-panel { background: linear-gradient(135deg, var(--surface-2), var(--surface-3)) !important; }
    .cf-box { background: var(--glass) !important; }
    .chat-sources-wrap { background: var(--glass) !important; }
    ::-webkit-scrollbar-track { background: var(--surface-0) !important; }
    ::-webkit-scrollbar-thumb { background: var(--cyan-dim) !important; }
    </style>
    """, unsafe_allow_html=True)

# ── Plotly base layout ────────────────────────────────────────────────────
_is_light = st.session_state.get("theme") == "light"
PLOT_LAYOUT = dict(
    paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="rgba(0,0,0,0)",
    font=dict(family="IBM Plex Mono, monospace",
              color="#475569" if _is_light else "#94a3b8", size=11),
    xaxis=dict(gridcolor="rgba(0,0,0,0.06)" if _is_light else "rgba(0,212,255,0.07)",
               showgrid=True, zeroline=False, tickfont=dict(size=10),
               linecolor="rgba(0,0,0,0.12)" if _is_light else "rgba(0,212,255,0.15)"),
    yaxis=dict(gridcolor="rgba(0,0,0,0.06)" if _is_light else "rgba(0,212,255,0.07)",
               showgrid=True, zeroline=False, tickfont=dict(size=10),
               linecolor="rgba(0,0,0,0.12)" if _is_light else "rgba(0,212,255,0.15)"),
    hoverlabel=dict(bgcolor="#ffffff" if _is_light else "#0d1420",
                    bordercolor="rgba(0,0,0,0.1)" if _is_light else "rgba(0,212,255,0.2)",
                    font=dict(family="IBM Plex Mono", size=11)),
)
COLORS = {
    "critical": "#ff4d6d", "degraded": "#ffb83f", "normal": "#00e5a0",
    "cyan": "#00d4ff", "emerald": "#00e5a0", "amber": "#ffb83f", "violet": "#a78bfa",
    "seq": ["#00d4ff","#00e5a0","#a78bfa","#ffb83f","#ff4d6d",
            "#38bdf8","#34d399","#c084fc","#fbbf24","#fb7185"],
}

def hex_to_rgba(hex_color, alpha=1.0):
    h = hex_color.lstrip("#")
    r, g, b = int(h[0:2],16), int(h[2:4],16), int(h[4:6],16)
    return f"rgba({r},{g},{b},{alpha})"

# ══════════════════════════════════════════════════════════════════════════════
#  DATA LOADING
# ══════════════════════════════════════════════════════════════════════════════

DATA_DIR = Path(os.getenv("QOSBUDDY_DATA_DIR", "./data"))

@st.cache_data(show_spinner=False)
def load_anomaly_df():
    df = pd.read_csv(DATA_DIR / "anomaly_scores_ns3.xls")
    df = df.drop(columns=[c for c in df.columns if c.endswith(".1")])
    df["ae_error"] = df["ae_error"].fillna(0.0)
    df["consensus_anomaly"] = ((df["if_anomaly"]==1) & (df["ae_anomaly"]==1)).astype(int)
    return df

@st.cache_data(show_spinner=False)
def load_causal_df():
    return pd.read_csv(DATA_DIR / "root_cause_labels.csv")

@st.cache_data(show_spinner=False)
def load_cf_df():
    df = pd.read_csv(DATA_DIR / "counterfactuals.csv")
    df["timestamp"] = pd.to_numeric(df["timestamp"], errors="coerce").fillna(1).astype(int)
    return df

@st.cache_resource(show_spinner=False)
def get_rag_resources():
    try:
        from rag.rag_pipeline import get_chroma_collection, get_embed_model, build_langchain_rag_chain
        col = get_chroma_collection()
        emb = get_embed_model()
        chain = build_langchain_rag_chain(col, emb)
        return col, emb, chain, True
    except Exception as e:
        logging.getLogger("qosbuddy.dashboard").warning(f"RAG init failed: {e}")
        return None, None, None, False

# ══════════════════════════════════════════════════════════════════════════════
#  SIDEBAR
# ══════════════════════════════════════════════════════════════════════════════

NAV_ITEMS = [
    ("kpi",     "📊", "KPI Overview"),
    ("anomaly", "🚨", "Anomaly Feed"),
    ("causal",  "🔬", "Causal Analysis"),
    ("rl",      "🤖", "RL Actions"),
    ("chat",    "💬", "Ask the Network"),
    ("export",  "📄", "Export Report"),
]

def render_sidebar():
    with st.sidebar:
        _dark = st.session_state.get("theme", "dark") == "dark"
        if st.button("☀️  Light mode" if _dark else "🌙  Dark mode",
                     key="_theme_btn", width='stretch'):
            st.session_state["theme"] = "light" if _dark else "dark"
            st.rerun()
        st.markdown("""
        <div class="sb-brand">
          <div class="sb-logo-ring"><div class="sb-logo-inner">📡</div></div>
          <div class="sb-title">QoSBuddy</div>
          <div class="sb-sub">PingWin · ESPRIT · 2025–2026</div>
        </div>
        <div class="status-row">
          <div class="status-dot"></div>
          <span class="status-text">SYSTEM ONLINE</span>
        </div>
        """, unsafe_allow_html=True)

        if "page" not in st.session_state:
            st.session_state["page"] = "kpi"

        for key, icon, label in NAV_ITEMS:
            if st.button(f"{icon}  {label}", key=f"nav_{key}", width='stretch'):
                st.session_state["page"] = key
                st.rerun()

        st.markdown("""<div style="height:1px;background:rgba(0,212,255,0.1);margin:12px 16px;"></div>""", unsafe_allow_html=True)

        # Filter panel
        st.markdown("""
        <div style="margin:0 4px;padding:14px;background:var(--glass);
                    border:1px solid var(--border);border-radius:var(--r-md);">
          <div style="font-family:'IBM Plex Mono';font-size:9px;color:var(--text-3);
                      text-transform:uppercase;letter-spacing:0.2em;margin-bottom:10px;">
            Active Filters
          </div>
        """, unsafe_allow_html=True)

        sev = st.selectbox("Severity", ["All","critical","degraded","normal"],
                           key="sev_sel", label_visibility="collapsed")
        ue  = st.selectbox("UE", ["All"]+[str(i) for i in range(1,11)],
                           key="ue_sel", label_visibility="collapsed")
        st.session_state["sev_f"] = None if sev=="All" else sev
        st.session_state["ue_f"]  = None if ue=="All"  else int(ue)
        st.markdown("</div>", unsafe_allow_html=True)

        st.markdown(f"""
        <div style="padding:10px 16px;font-family:'IBM Plex Mono';font-size:9px;
                    color:rgba(0,212,255,0.3);text-transform:uppercase;letter-spacing:0.15em;">
          {datetime.now().strftime('%Y-%m-%d  %H:%M:%S')}
        </div>""", unsafe_allow_html=True)

    return st.session_state.get("page","kpi")

# ── Helpers ────────────────────────────────────────────────────────────────

def apply_filters(df):
    s = st.session_state.get("sev_f")
    u = st.session_state.get("ue_f")
    if s: df = df[df["severity"]==s]
    if u: df = df[df["ue_id"]==u]
    return df

def page_header(icon, title, subtitle):
    st.markdown(f"""
    <div class="page-header">
      <div class="page-header-icon">{icon}</div>
      <div>
        <div class="page-header-title">{title}</div>
        <div class="page-header-sub">{subtitle}</div>
      </div>
    </div>""", unsafe_allow_html=True)

def kpi_card(label, value, unit, status, pct=50, status_text=""):
    st.markdown(f"""
    <div class="kpi-card {status}">
      <div class="kpi-label">{label}</div>
      <div style="display:flex;align-items:baseline;gap:4px;margin:6px 0;">
        <span class="kpi-value">{value}</span>
        <span class="kpi-unit">{unit}</span>
      </div>
      <div class="kpi-bar-wrap"><div class="kpi-bar" style="width:{min(pct,100):.0f}%"></div></div>
      <div class="kpi-status">{status_text}</div>
    </div>""", unsafe_allow_html=True)

# ══════════════════════════════════════════════════════════════════════════════
#  PAGE 1 — KPI OVERVIEW
# ══════════════════════════════════════════════════════════════════════════════

def page_kpi_overview(anomaly_df, causal_df):
    page_header("📊","Network KPI Overview","DSO1.1 · DSO2.2 · Real-time Quality of Service Intelligence")
    df = apply_filters(anomaly_df)

    avg_jitter     = df["jitter_ms"].mean()
    avg_pkt_loss   = df["packet_loss_ratio"].mean() * 100
    avg_throughput = df["throughput_mbps"].mean()
    avg_delay      = df["delay_ms"].mean()
    anomaly_rate   = df["if_anomaly"].mean() * 100
    sinr_mean      = df["sinr_dl_db"].mean()

    cols = st.columns(6)
    with cols[0]:
        s = "ok" if avg_jitter<10 else "warn" if avg_jitter<50 else "crit"
        kpi_card("Avg Jitter", f"{avg_jitter:.1f}", "ms", s, min(avg_jitter,100),
                 "▲ ELEVATED" if s=="warn" else ("🔴 CRITICAL" if s=="crit" else "✓ NOMINAL"))
    with cols[1]:
        s = "ok" if avg_pkt_loss<10 else "warn" if avg_pkt_loss<40 else "crit"
        kpi_card("Packet Loss", f"{avg_pkt_loss:.1f}", "%", s, avg_pkt_loss,
                 "🔴 HIGH LOSS" if s=="crit" else ("⚠ MODERATE" if s=="warn" else "✓ NOMINAL"))
    with cols[2]:
        s = "ok" if avg_throughput>2 else "warn" if avg_throughput>0.5 else "crit"
        kpi_card("Throughput", f"{avg_throughput:.2f}", "Mbps", s, min(avg_throughput/10*100,100),
                 "🔴 DEGRADED" if s=="crit" else ("⚠ LOW" if s=="warn" else "✓ NOMINAL"))
    with cols[3]:
        s = "ok" if avg_delay<50 else "warn" if avg_delay<200 else "crit"
        kpi_card("Avg Delay", f"{avg_delay:.0f}", "ms", s, min(avg_delay/500*100,100),
                 "🔴 HIGH RTT" if s=="crit" else ("⚠ ELEVATED" if s=="warn" else "✓ NOMINAL"))
    with cols[4]:
        s = "ok" if anomaly_rate<20 else "warn" if anomaly_rate<45 else "crit"
        kpi_card("Anomaly Rate", f"{anomaly_rate:.1f}", "%", s, anomaly_rate,
                 "🔴 CRISIS" if s=="crit" else ("⚠ WATCHLIST" if s=="warn" else "✓ STABLE"))
    with cols[5]:
        s = "ok" if sinr_mean>5 else "warn" if sinr_mean>-5 else "crit"
        kpi_card("Avg SINR", f"{sinr_mean:.1f}", "dB", s, min((sinr_mean+30)/60*100,100),
                 "🔴 POOR RF" if s=="crit" else ("⚠ MARGINAL" if s=="warn" else "✓ NOMINAL"))

    st.markdown("<div style='height:8px'></div>", unsafe_allow_html=True)
    col_l, col_m, col_r = st.columns([1,1,2])

    with col_l:
        st.markdown('<div class="glass-panel"><div class="glass-panel-title">Severity Distribution</div>', unsafe_allow_html=True)
        sev_counts = df["severity"].value_counts().reset_index()
        sev_counts.columns = ["severity","count"]
        fig = go.Figure(go.Pie(
            labels=sev_counts["severity"], values=sev_counts["count"], hole=0.68,
            marker=dict(colors=[COLORS.get(s,"#888") for s in sev_counts["severity"]],
                        line=dict(color="#080c12", width=3)),
            textinfo="none",
            hovertemplate="<b>%{label}</b><br>%{value:,} events<br>%{percent}<extra></extra>",
        ))
        total = len(df)
        fig.add_annotation(text=f"<b>{total:,}</b>", x=0.5, y=0.55,
                           font=dict(size=22,color="#e8f0fe",family="Syne"), showarrow=False)
        fig.add_annotation(text="EVENTS", x=0.5, y=0.38,
                           font=dict(size=9,color="#4a5568",family="IBM Plex Mono"), showarrow=False)
        fig.update_layout(**PLOT_LAYOUT, showlegend=True,
                          legend=dict(orientation="h",yanchor="bottom",y=-0.15,xanchor="center",x=0.5),
                          height=260, margin=dict(t=10,b=40,l=10,r=10))
        st.plotly_chart(fig, width='stretch')
        st.markdown("</div>", unsafe_allow_html=True)

    with col_m:
        st.markdown('<div class="glass-panel"><div class="glass-panel-title">Root Cause Breakdown</div>', unsafe_allow_html=True)
        rc_df = causal_df.copy()
        if st.session_state.get("sev_f"):
            rc_df = rc_df[rc_df["severity"]==st.session_state["sev_f"]]
        rc = rc_df["root_cause"].value_counts().reset_index()
        rc.columns = ["root_cause","count"]
        rc = rc.sort_values("count")
        fig = go.Figure(go.Bar(
            x=rc["count"], y=rc["root_cause"], orientation="h",
            marker=dict(color=rc["count"],
                        colorscale=[[0,"rgba(0,212,255,0.2)"],[1,"#00d4ff"]],
                        line=dict(color="rgba(0,212,255,0.3)",width=1)),
            hovertemplate="<b>%{y}</b><br>%{x:,} events<extra></extra>",
        ))
        fig.update_layout(**PLOT_LAYOUT, height=260, margin=dict(t=10,b=10,l=10,r=10))
        st.plotly_chart(fig, width='stretch')
        st.markdown("</div>", unsafe_allow_html=True)

    with col_r:
        st.markdown('<div class="glass-panel"><div class="glass-panel-title">Packet Loss Timeline — UEs 1–8</div>', unsafe_allow_html=True)
        df_ts = df[df["ue_id"].isin(range(1,9))].sort_values(["ue_id","timestamp"])
        fig = go.Figure()
        for i, ue in enumerate(sorted(df_ts["ue_id"].unique())):
            d = df_ts[df_ts["ue_id"]==ue]
            fig.add_trace(go.Scatter(
                x=d["timestamp"], y=d["packet_loss_ratio"], name=f"UE {ue}", mode="lines",
                line=dict(width=1.5, color=COLORS["seq"][i%len(COLORS["seq"])]),
                hovertemplate=f"UE {ue}<br>t=%{{x}}<br>loss=%{{y:.3f}}<extra></extra>",
            ))
        fig.add_hline(y=0.5, line_dash="dot", line_color="#ff4d6d",
                      annotation_text="SLA limit", annotation_font_size=9, annotation_font_color="#ff4d6d")
        fig.update_layout(**PLOT_LAYOUT, height=260, margin=dict(t=10,b=10,l=10,r=10),
                          xaxis_title="Timestamp", yaxis_title="Packet Loss Ratio")
        st.plotly_chart(fig, width='stretch')
        st.markdown("</div>", unsafe_allow_html=True)

    col_a, col_b = st.columns(2)
    with col_a:
        st.markdown('<div class="glass-panel"><div class="glass-panel-title">SINR Distribution by Load Level</div>', unsafe_allow_html=True)
        fig = go.Figure()
        for i, ll in enumerate(sorted(df["load_level"].unique())):
            d = df[df["load_level"]==ll]["sinr_dl_db"]
            c = COLORS["seq"][i%len(COLORS["seq"])]
            fig.add_trace(go.Box(y=d, name=f"Load {ll}", marker_color=c,
                                 line_color=c, fillcolor=hex_to_rgba(c,0.13), boxmean="sd"))
        fig.update_layout(**PLOT_LAYOUT, height=280, showlegend=False,
                          margin=dict(t=10,b=10,l=10,r=10), yaxis_title="SINR (dB)")
        st.plotly_chart(fig, width='stretch')
        st.markdown("</div>", unsafe_allow_html=True)

    with col_b:
        st.markdown('<div class="glass-panel"><div class="glass-panel-title">Detector Agreement Matrix</div>', unsafe_allow_html=True)
        both    = int(((df["if_anomaly"]==1) & (df["ae_anomaly"]==1)).sum())
        if_only = int(((df["if_anomaly"]==1) & (df["ae_anomaly"]==0)).sum())
        ae_only = int(((df["if_anomaly"]==0) & (df["ae_anomaly"]==1)).sum())
        neither = int(((df["if_anomaly"]==0) & (df["ae_anomaly"]==0)).sum())
        labels  = ["Both Agree","IF Only","AE Only","Neither"]
        vals    = [both,if_only,ae_only,neither]
        bar_colors = [COLORS["critical"],COLORS["degraded"],COLORS["violet"],COLORS["normal"]]
        fig = go.Figure(go.Bar(
            x=labels, y=vals,
            marker=dict(color=[hex_to_rgba(c,0.67) for c in bar_colors], line=dict(color=bar_colors,width=2)),
            text=[f"{v:,}" for v in vals], textposition="auto",
            textfont=dict(family="IBM Plex Mono", size=11),
            hovertemplate="<b>%{x}</b><br>%{y:,} events<extra></extra>",
        ))
        fig.update_layout(**PLOT_LAYOUT, height=280, showlegend=False,
                          margin=dict(t=10,b=10,l=10,r=10), yaxis_title="Events")
        st.plotly_chart(fig, width='stretch')
        st.markdown("</div>", unsafe_allow_html=True)


# ══════════════════════════════════════════════════════════════════════════════
#  PAGE 2 — ANOMALY FEED
# ══════════════════════════════════════════════════════════════════════════════

def page_anomaly_feed(anomaly_df):
    page_header("🚨","Anomaly Feed","DSO2.2 · DSO2.3 · Isolation Forest + LSTM Autoencoder Triage Board")
    df = apply_filters(anomaly_df)

    n_crit = int((df["severity"]=="critical").sum())
    n_degr = int((df["severity"]=="degraded").sum())
    n_norm = int((df["severity"]=="normal").sum())

    st.markdown(f"""
    <div class="stat-row">
      <div class="stat-chip total"><span class="stat-chip-label">Total Events</span><span class="stat-chip-value">{len(df):,}</span></div>
      <div class="stat-chip crit"><span class="stat-chip-label">Critical</span><span class="stat-chip-value">{n_crit:,}</span></div>
      <div class="stat-chip warn"><span class="stat-chip-label">Degraded</span><span class="stat-chip-value">{n_degr:,}</span></div>
      <div class="stat-chip ok"><span class="stat-chip-label">Normal</span><span class="stat-chip-value">{n_norm:,}</span></div>
      <div class="stat-chip total"><span class="stat-chip-label">Consensus</span><span class="stat-chip-value">{df['consensus_anomaly'].sum():,}</span></div>
    </div>""", unsafe_allow_html=True)

    col_l, col_r = st.columns([3,2])
    with col_l:
        st.markdown('<div class="glass-panel"><div class="glass-panel-title">IF Score vs AE Reconstruction Error</div>', unsafe_allow_html=True)
        sample = df.sample(min(3000,len(df)), random_state=42)
        fig = go.Figure()
        for sev, col in [("critical",COLORS["critical"]),("degraded",COLORS["degraded"]),("normal",COLORS["normal"])]:
            d = sample[sample["severity"]==sev]
            fig.add_trace(go.Scatter(x=d["if_score"], y=d["ae_error"], name=sev.capitalize(), mode="markers",
                                     marker=dict(color=hex_to_rgba(col,0.6), size=4, line=dict(color=col,width=0.5)),
                                     hovertemplate=f"<b>{sev}</b><br>IF=%{{x:.4f}}<br>AE=%{{y:.4f}}<extra></extra>"))
        fig.update_layout(**PLOT_LAYOUT, height=300, xaxis_title="IF Anomaly Score",
                          yaxis_title="AE Reconstruction Error", margin=dict(t=10,b=10,l=10,r=10))
        st.plotly_chart(fig, width='stretch')
        st.markdown("</div>", unsafe_allow_html=True)

    with col_r:
        st.markdown('<div class="glass-panel"><div class="glass-panel-title">Consensus Heatmap — UE × Load Level</div>', unsafe_allow_html=True)
        hm = anomaly_df.groupby(["ue_id","load_level"])["consensus_anomaly"].mean().reset_index()
        pivot = hm.pivot(index="ue_id", columns="load_level", values="consensus_anomaly")
        fig = go.Figure(go.Heatmap(
            z=pivot.values, x=[f"Load {c}" for c in pivot.columns], y=[f"UE {r}" for r in pivot.index],
            colorscale=[[0,"#0d1420"],[0.5,"rgba(255,77,109,0.27)"],[1,"#ff4d6d"]],
            hovertemplate="UE %{y}<br>%{x}<br>Rate: %{z:.1%}<extra></extra>",
            showscale=True, colorbar=dict(tickfont=dict(size=9), thickness=10),
        ))
        fig.update_layout(**PLOT_LAYOUT, height=300, margin=dict(t=10,b=10,l=10,r=10))
        st.plotly_chart(fig, width='stretch')
        st.markdown("</div>", unsafe_allow_html=True)

    st.markdown('<div class="glass-panel"><div class="glass-panel-title">🔴 Critical Event Triage — Top 20 by IF Score</div>', unsafe_allow_html=True)
    top = df[df["severity"]=="critical"].sort_values("if_score",ascending=False).head(20)
    cols_show = ["ue_id","timestamp","severity","if_score","ae_error","packet_loss_ratio","sinr_dl_db","throughput_mbps","if_anomaly","ae_anomaly"]
    cols_show = [c for c in cols_show if c in top.columns]
    th = "".join(f"<th>{c}</th>" for c in cols_show)
    rows_html = ""
    for _, row in top.iterrows():
        tds = ""
        for c in cols_show:
            v = row[c]
            if c == "severity":
                pill_c = "pill-crit" if v=="critical" else ("pill-warn" if v=="degraded" else "pill-ok")
                tds += f'<td><span class="pill {pill_c}">{v}</span></td>'
                continue
            cls = ""
            if c=="if_score" and isinstance(v,float): cls = "td-crit" if v>0.5 else "td-warn"
            if c=="packet_loss_ratio" and isinstance(v,float): cls = "td-crit" if v>0.5 else ("td-warn" if v>0.1 else "td-ok")
            if c=="sinr_dl_db" and isinstance(v,float): cls = "td-crit" if v<0 else ("td-warn" if v<5 else "td-ok")
            fmt = f"{v:.4f}" if isinstance(v,float) else str(v)
            tds += f'<td class="{cls}">{fmt}</td>'
        rows_html += f"<tr>{tds}</tr>"
    st.markdown(f"""
    <div style="overflow-x:auto;">
    <table class="styled-table"><thead><tr>{th}</tr></thead><tbody>{rows_html}</tbody></table>
    </div>""", unsafe_allow_html=True)
    st.markdown("</div>", unsafe_allow_html=True)


# ══════════════════════════════════════════════════════════════════════════════
#  PAGE 3 — CAUSAL ANALYSIS
# ══════════════════════════════════════════════════════════════════════════════

CAUSAL_CHAINS = {
    "interference": [("📡","RF Interference — SINR degrades"),("📉","↓ MCS Index — lower modulation"),("🔄","↑ Retransmissions — ARQ overhead"),("📦","↑ Packet Loss Ratio"),("📳","↑ Jitter — variable queuing"),("🚨","SLA Violation Triggered")],
    "congestion":   [("📶","↑ Load Level — high traffic demand"),("📊","↑ PRB Utilization — saturation"),("🚧","Queue Overflow — buffer full"),("📦","↑ Packet Loss Ratio"),("⏱️","↑ Delay — head-of-line blocking"),("🚨","SLA Breach — latency budget exceeded")],
    "mobility":     [("🏃","↑ Mobility Speed — UE moving fast"),("🔁","Handover Failure — beam tracking loss"),("📡","↓ Throughput — signal interruption"),("📳","↑ Jitter — burst after reconnect"),("🌐","Service Disruption — session drop"),("🚨","SLA Breach — availability impact")],
    "combined/other":[("⚡","Load ↑ AND SINR ↓ simultaneously"),("🔗","Compounded Degradation — multi-factor"),("📦","Packet Loss spikes both channels"),("⏱️","Delay AND Jitter both elevated"),("💥","Cascading failure across UEs"),("🚨","Critical SLA Multi-Breach")],
}
RC_ACTIONS = {
    "interference":   ("frequency_reallocation + power_control","🔧"),
    "congestion":     ("load_balancing + traffic_throttling","⚖️"),
    "mobility":       ("handover_optimization + beam_tracking","📡"),
    "combined/other": ("multi-action: reroute + prioritize","🛠️"),
}

def page_causal_root_cause(causal_df, cf_df):
    page_header("🔬","Causal Root Cause Explorer","DSO1.2 · Innovation 1 · DoWhy Causal AI · NS-3 Ground Truth")
    merged = causal_df.merge(cf_df[["ue_id","timestamp","original_packet_loss","counterfactual_if_static","reduction_pct"]], on="timestamp", how="left")

    col_sel, col_main = st.columns([1,3])

    with col_sel:
        st.markdown('<div class="glass-panel">', unsafe_allow_html=True)
        st.markdown('<div class="glass-panel-title">Select Root Cause</div>', unsafe_allow_html=True)
        rc_options = sorted(causal_df["root_cause"].unique().tolist())
        selected_rc = st.radio("rc", rc_options, label_visibility="collapsed")
        rc_data = merged[merged["root_cause"]==selected_rc]
        n_ev = len(rc_data)
        crit_pct = (rc_data["severity"]=="critical").mean()*100
        avg_loss = rc_data["packet_loss_ratio"].mean()*100
        st.markdown(f"""
        <div style="margin-top:16px;display:flex;flex-direction:column;gap:8px;">
          <div style="background:var(--surface-2);border-radius:8px;padding:10px 12px;">
            <div style="font-family:'IBM Plex Mono';font-size:9px;color:var(--text-3);">EVENTS</div>
            <div style="font-family:'Syne';font-size:22px;font-weight:800;color:var(--cyan);">{n_ev:,}</div>
          </div>
          <div style="background:var(--surface-2);border-radius:8px;padding:10px 12px;">
            <div style="font-family:'IBM Plex Mono';font-size:9px;color:var(--text-3);">CRITICAL %</div>
            <div style="font-family:'Syne';font-size:22px;font-weight:800;color:var(--rose);">{crit_pct:.1f}%</div>
          </div>
          <div style="background:var(--surface-2);border-radius:8px;padding:10px 12px;">
            <div style="font-family:'IBM Plex Mono';font-size:9px;color:var(--text-3);">AVG LOSS</div>
            <div style="font-family:'Syne';font-size:22px;font-weight:800;color:var(--amber);">{avg_loss:.1f}%</div>
          </div>
        </div>""", unsafe_allow_html=True)
        st.markdown("</div>", unsafe_allow_html=True)

    with col_main:
        action, action_icon = RC_ACTIONS.get(selected_rc, ("unknown","❓"))
        chain_html = ""
        for i, (icon, text) in enumerate(CAUSAL_CHAINS.get(selected_rc,[])):
            chain_html += f'<div class="causal-node"><span>{icon}</span><span style="font-family:\'IBM Plex Mono\';font-size:12px;color:var(--text-2);">{text}</span></div>'
            if i < len(CAUSAL_CHAINS.get(selected_rc,[]))-1:
                chain_html += '<div class="causal-arrow">│</div>'
        st.markdown(f"""
        <div class="glass-panel">
          <div class="glass-panel-title">Causal Chain — {selected_rc.upper()}</div>
          <div class="causal-chain">{chain_html}</div>
          <div class="action-badge">{action_icon} NOC Action: {action}</div>
        </div>""", unsafe_allow_html=True)

        st.markdown('<div class="glass-panel"><div class="glass-panel-title">Counterfactual Simulation — DoWhy ATE: 49.71%</div>', unsafe_allow_html=True)
        intervention = st.slider("Intervention Effectiveness", 0, 100, 50, format="%d%%")
        avg_orig  = rc_data["packet_loss_ratio"].mean() if len(rc_data) else 0
        cf_full   = avg_orig * (1 - 0.4971)
        simulated = avg_orig * (1 - (intervention/100)*0.4971)
        saved_pct = (avg_orig - simulated) / max(avg_orig, 1e-9) * 100

        st.markdown(f"""
        <div class="cf-result-panel">
          <div class="cf-box">
            <div class="cf-box-label">Current Loss</div>
            <div class="cf-box-value" style="color:var(--rose);">{avg_orig*100:.1f}%</div>
          </div>
          <div class="cf-arrow">→</div>
          <div class="cf-box">
            <div class="cf-box-label">After Intervention</div>
            <div class="cf-box-value" style="color:var(--amber);">{simulated*100:.1f}%</div>
          </div>
          <div class="cf-arrow">→</div>
          <div class="cf-box">
            <div class="cf-box-label">Full Fix Potential</div>
            <div class="cf-box-value" style="color:var(--emerald);">{cf_full*100:.1f}%</div>
          </div>
          <div class="cf-arrow">→</div>
          <div class="cf-box" style="border:1px solid var(--emerald-dim);">
            <div class="cf-box-label">Reduction Achieved</div>
            <div class="cf-box-value" style="color:var(--emerald);">{saved_pct:.1f}%</div>
          </div>
        </div>""", unsafe_allow_html=True)
        st.markdown("</div>", unsafe_allow_html=True)

    col_a, col_b = st.columns(2)
    with col_a:
        st.markdown('<div class="glass-panel"><div class="glass-panel-title">Packet Loss Distribution by Root Cause</div>', unsafe_allow_html=True)
        fig = go.Figure()
        for i, rc in enumerate(causal_df["root_cause"].unique()):
            d = causal_df[causal_df["root_cause"]==rc]["packet_loss_ratio"]
            c = COLORS["seq"][i%len(COLORS["seq"])]
            fig.add_trace(go.Violin(y=d, name=rc, box_visible=True, meanline_visible=True, line_color=c, fillcolor=hex_to_rgba(c,0.13)))
        fig.update_layout(**PLOT_LAYOUT, height=280, showlegend=False, margin=dict(t=10,b=10,l=10,r=10), yaxis_title="Packet Loss")
        st.plotly_chart(fig, width='stretch')
        st.markdown("</div>", unsafe_allow_html=True)

    with col_b:
        st.markdown('<div class="glass-panel"><div class="glass-panel-title">Severity Stack by Root Cause</div>', unsafe_allow_html=True)
        sev_rc = causal_df.groupby(["root_cause","severity"]).size().reset_index(name="count")
        fig = go.Figure()
        for sev in ["critical","degraded","normal"]:
            d = sev_rc[sev_rc["severity"]==sev]
            fig.add_trace(go.Bar(x=d["root_cause"], y=d["count"], name=sev.capitalize(),
                                 marker=dict(color=hex_to_rgba(COLORS[sev],0.8), line=dict(color=COLORS[sev],width=1))))
        fig.update_layout(**PLOT_LAYOUT, barmode="stack", height=280, margin=dict(t=10,b=10,l=10,r=10))
        st.plotly_chart(fig, width='stretch')
        st.markdown("</div>", unsafe_allow_html=True)


# ══════════════════════════════════════════════════════════════════════════════
#  PAGE 4 — RL ACTIONS
# ══════════════════════════════════════════════════════════════════════════════

def page_rl_actions(anomaly_df, causal_df):
    page_header("🤖","RL Agent Actions","DSO3.1 · PPO Policy · Digital Twin Environment (NS-3)")
    merged = anomaly_df.merge(causal_df[["timestamp","root_cause","recommended_action"]], on="timestamp", how="left")
    action_stats = merged[merged["if_anomaly"]==1].groupby("recommended_action").agg(
        events=("if_score","count"), avg_score=("if_score","mean"),
        avg_loss=("packet_loss_ratio","mean"),
        critical_pct=("severity", lambda x: (x=="critical").mean()),
    ).reset_index().sort_values("events", ascending=False)

    action_colors = [COLORS["cyan"],COLORS["emerald"],COLORS["amber"],COLORS["violet"]]
    action_icons  = ["⚖️","🔧","📡","🛠️"]

    st.markdown('<div style="display:flex;gap:16px;margin-bottom:20px;flex-wrap:wrap;">', unsafe_allow_html=True)
    for i, (_, row) in enumerate(action_stats.iterrows()):
        c = action_colors[i % len(action_colors)]
        ic = action_icons[i % len(action_icons)]
        st.markdown(f"""
        <div style="flex:1;min-width:200px;background:var(--glass);border:1px solid {c}33;
                    border-radius:var(--r-lg);padding:20px;backdrop-filter:blur(16px);">
          <div style="font-size:24px;margin-bottom:8px;">{ic}</div>
          <div style="font-family:'IBM Plex Mono';font-size:9px;color:{c};text-transform:uppercase;letter-spacing:0.1em;margin-bottom:8px;">{row['recommended_action']}</div>
          <div style="display:flex;gap:16px;flex-wrap:wrap;margin-top:12px;">
            <div><div style="font-family:'IBM Plex Mono';font-size:9px;color:var(--text-3);">DISPATCHED</div><div style="font-family:'Syne';font-size:26px;font-weight:800;color:{c};">{int(row['events']):,}</div></div>
            <div><div style="font-family:'IBM Plex Mono';font-size:9px;color:var(--text-3);">CRIT RATE</div><div style="font-family:'Syne';font-size:26px;font-weight:800;color:var(--rose);">{row['critical_pct']*100:.0f}%</div></div>
          </div>
        </div>""", unsafe_allow_html=True)
    st.markdown("</div>", unsafe_allow_html=True)

    col_l, col_r = st.columns(2)
    with col_l:
        st.markdown('<div class="glass-panel"><div class="glass-panel-title">Simulated PPO Reward vs Baseline</div>', unsafe_allow_html=True)
        np.random.seed(42)
        steps    = np.arange(1,101)
        baseline = np.cumsum(np.random.normal(-0.15, 0.4, 100))
        ppo      = np.cumsum(np.random.normal(0.35, 0.25, 100))
        fig = go.Figure()
        fig.add_trace(go.Scatter(x=steps, y=baseline, name="No-Action Baseline",
                                 line=dict(color="#4a5568",width=1.5,dash="dot"),
                                 fill="tozeroy", fillcolor="rgba(74,85,104,0.08)"))
        fig.add_trace(go.Scatter(x=steps, y=ppo, name="PPO RL Agent",
                                 line=dict(color=COLORS["emerald"],width=2.5),
                                 fill="tozeroy", fillcolor=hex_to_rgba(COLORS["emerald"],0.09)))
        fig.update_layout(**PLOT_LAYOUT, height=280, margin=dict(t=10,b=10,l=10,r=10),
                          xaxis_title="Simulation Step", yaxis_title="Cumulative Reward")
        st.plotly_chart(fig, width='stretch')
        st.markdown("</div>", unsafe_allow_html=True)

    with col_r:
        st.markdown('<div class="glass-panel"><div class="glass-panel-title">Action Distribution — Anomalous Events</div>', unsafe_allow_html=True)
        fig = go.Figure(go.Pie(
            labels=action_stats["recommended_action"], values=action_stats["events"], hole=0.6,
            marker=dict(colors=[hex_to_rgba(c,0.8) for c in action_colors[:len(action_stats)]],
                        line=dict(color="#080c12",width=3)),
            textinfo="none",
            hovertemplate="<b>%{label}</b><br>%{value:,} events<extra></extra>",
        ))
        fig.update_layout(**PLOT_LAYOUT, height=280, showlegend=True,
                          legend=dict(orientation="h",yanchor="bottom",y=-0.3,xanchor="center",x=0.5),
                          margin=dict(t=10,b=60,l=10,r=10))
        st.plotly_chart(fig, width='stretch')
        st.markdown("</div>", unsafe_allow_html=True)


# ══════════════════════════════════════════════════════════════════════════════
#  PAGE 5 — RAG CHAT
# ══════════════════════════════════════════════════════════════════════════════

SUGGESTIONS = [
    "What's the main root cause of critical anomalies?",
    "Which UEs have the worst packet loss?",
    "What action reduces congestion fastest?",
    "Explain QoSBuddy's business objectives.",
]

def page_rag_chat():
    page_header("💬","Ask the Network","DSO3.2 · RAG Engine · ChromaDB + Ollama llama3.2 · NOC Intelligence")
    collection, embed_model, chain, rag_ready = get_rag_resources()

    cc1, cc2, cc3 = st.columns([2,1,1])
    with cc1:
        mode = st.selectbox("Query Mode", ["general","anomaly","causal","domain"],
                            format_func=lambda x: {"general":"🌐 General","anomaly":"🚨 Anomaly events",
                                                   "causal":"🔬 Causal insights","domain":"📖 Project knowledge"}[x])
    with cc2:
        sev_flt = st.selectbox("Severity", ["None","critical","degraded","normal"])
        sev_flt = None if sev_flt=="None" else sev_flt
    with cc3:
        rc_flt = st.selectbox("Root Cause", ["None","interference","congestion","mobility","combined/other"])
        rc_flt = None if rc_flt=="None" else rc_flt

    # Suggestion chips
    chip_cols = st.columns(len(SUGGESTIONS))
    for i, s in enumerate(SUGGESTIONS):
        if chip_cols[i].button(s, key=f"chip_{i}", width='stretch'):
            st.session_state["_pq"] = s

    if "chat_history" not in st.session_state:
        st.session_state["chat_history"] = []

    user_input = st.chat_input("Ask anything about your 5G network…")
    if "_pq" in st.session_state:
        user_input = st.session_state.pop("_pq")

    if user_input:
        with st.spinner(""):
            if rag_ready:
                from rag.rag_pipeline import query_rag
                result = query_rag(question=user_input, collection=collection,
                                   embed_model=embed_model, chain=chain, mode=mode,
                                   severity_filter=sev_flt, root_cause_filter=rc_flt)
                answer, sources = result["answer"], result["sources"]
            else:
                answer  = "⚠️ RAG pipeline not initialised. Run `python rag/build_knowledge_base.py` first."
                sources = []
        st.session_state["chat_history"].append({"question":user_input,"answer":answer,"sources":sources})

    st.markdown('<div class="chat-container">', unsafe_allow_html=True)
    ts = datetime.now().strftime("%H:%M")
    for turn in reversed(st.session_state["chat_history"]):
        st.markdown(f"""
        <div class="chat-bubble chat-user">
          <div class="chat-meta">You · {ts}</div>
          {turn['question']}
        </div>""", unsafe_allow_html=True)
        st.markdown(f"""
        <div class="chat-bubble chat-bot">
          <div class="chat-meta">QoSBuddy NOC Assistant · {len(turn['sources'])} sources</div>
          {turn['answer'].replace(chr(10),'<br>')}
        </div>""", unsafe_allow_html=True)
        if turn["sources"]:
            with st.expander(f"📚 {len(turn['sources'])} retrieved sources"):
                items = ""
                for s in turn["sources"]:
                    m = s["metadata"]
                    items += f"""<div class="chat-source-item"><span class="src-badge">{m.get('doc_type','?')[:8]}</span><span class="src-sim">sim={s['similarity']:.3f}</span><span>sev={m.get('severity','—')}</span><span>rc={m.get('root_cause','—')}</span></div>"""
                st.markdown(f'<div class="chat-sources-wrap">{items}</div>', unsafe_allow_html=True)
    st.markdown("</div>", unsafe_allow_html=True)

    if st.session_state["chat_history"]:
        if st.button("🗑  Clear Chat"):
            st.session_state["chat_history"] = []
            st.rerun()


# ══════════════════════════════════════════════════════════════════════════════
#  PAGE 6 — EXPORT
# ══════════════════════════════════════════════════════════════════════════════

def page_export_report(anomaly_df, causal_df):
    page_header("📄","Export Report","DSO3.2 · Generate PDF NOC Report with KPIs, Root Causes & Chat Log")
    col_l, col_r = st.columns([1,2])
    with col_l:
        st.markdown("""
        <div class="export-card">
          <div class="export-icon">📊</div>
          <div class="export-title">NOC Intelligence Report</div>
          <div class="export-sub">QoSBuddy · PingWin · ESPRIT</div>
        </div>""", unsafe_allow_html=True)
    with col_r:
        st.markdown('<div class="glass-panel"><div class="glass-panel-title">Report Configuration</div>', unsafe_allow_html=True)
        report_title = st.text_input("Report Title", value=f"QoSBuddy NOC Report — {datetime.now().strftime('%Y-%m-%d %H:%M')}")
        include_chat = st.checkbox("Include RAG Chat Session Log", value=True)
        include_raw  = st.checkbox("Include Raw Critical Anomaly Table", value=False)
        if st.button("📥  Generate & Download PDF", width='stretch'):
            with st.spinner("Compiling report…"):
                try:
                    pdf_bytes = _make_pdf(report_title, anomaly_df, causal_df, include_chat, include_raw)
                    st.download_button("⬇  Download PDF", data=pdf_bytes,
                                       file_name=f"qosbuddy_{datetime.now().strftime('%Y%m%d_%H%M')}.pdf",
                                       mime="application/pdf", width='stretch')
                    st.success("✓ PDF ready")
                except ImportError:
                    st.error("Install ReportLab: `pip install reportlab`")
        st.markdown("</div>", unsafe_allow_html=True)


def _make_pdf(title, anomaly_df, causal_df, include_chat, include_raw):
    from reportlab.lib.pagesizes import A4
    from reportlab.lib.styles import getSampleStyleSheet, ParagraphStyle
    from reportlab.lib.units import cm
    from reportlab.lib import colors
    from reportlab.platypus import SimpleDocTemplate, Paragraph, Spacer, Table, TableStyle, HRFlowable
    buf = io.BytesIO()
    doc = SimpleDocTemplate(buf, pagesize=A4, leftMargin=2*cm, rightMargin=2*cm, topMargin=2*cm, bottomMargin=2*cm)
    styles = getSampleStyleSheet()
    BLUE  = colors.HexColor("#00d4ff")
    GREEN = colors.HexColor("#00e5a0")
    body_s = ParagraphStyle("body", parent=styles["Normal"], fontSize=9, leading=13)
    h1_s   = ParagraphStyle("h1", parent=styles["Heading1"], fontSize=13, textColor=colors.HexColor("#0d1420"), spaceAfter=4)
    story  = []
    story.append(Paragraph(title, ParagraphStyle("tit", parent=styles["Title"], textColor=BLUE, fontSize=18, spaceAfter=4)))
    story.append(Paragraph(f"PingWin · ESPRIT · DSO3.2  |  {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}", body_s))
    story.append(HRFlowable(width="100%", color=BLUE, thickness=2, spaceAfter=10))
    story.append(Paragraph("Network KPI Summary", h1_s))
    df = anomaly_df
    kd = [["KPI","Value","Status"],
          ["Avg Jitter (ms)", f"{df['jitter_ms'].mean():.2f}", "ELEVATED" if df['jitter_ms'].mean()>10 else "OK"],
          ["Avg Packet Loss",  f"{df['packet_loss_ratio'].mean()*100:.1f}%", "HIGH" if df['packet_loss_ratio'].mean()>0.3 else "OK"],
          ["Avg Throughput",   f"{df['throughput_mbps'].mean():.3f} Mbps", "LOW" if df['throughput_mbps'].mean()<2 else "OK"],
          ["Avg Delay (ms)",   f"{df['delay_ms'].mean():.1f}", "HIGH" if df['delay_ms'].mean()>100 else "OK"],
          ["Anomaly Rate",     f"{df['if_anomaly'].mean()*100:.1f}%", "HIGH" if df['if_anomaly'].mean()>0.3 else "OK"],
          ["Critical Events",  f"{(df['severity']=='critical').sum():,}", ""]]
    kt = Table(kd, colWidths=[6*cm,4*cm,5*cm])
    kt.setStyle(TableStyle([
        ("BACKGROUND",(0,0),(-1,0),BLUE), ("TEXTCOLOR",(0,0),(-1,0),colors.white),
        ("FONTNAME",(0,0),(-1,0),"Helvetica-Bold"),
        ("ROWBACKGROUNDS",(0,1),(-1,-1),[colors.white, colors.HexColor("#f6f8fa")]),
        ("GRID",(0,0),(-1,-1),0.4,colors.HexColor("#d0d7de")),
        ("FONTSIZE",(0,0),(-1,-1),8), ("PADDING",(0,0),(-1,-1),6),
    ]))
    story += [kt, Spacer(1,10)]
    story.append(Paragraph("Root Cause Analysis (DSO1.2)", h1_s))
    rc_s = causal_df.groupby(["root_cause","recommended_action"]).agg(count=("if_anomaly","count"), avg=("packet_loss_ratio","mean")).reset_index()
    rd = [["Root Cause","Action","Events","Avg Loss"]]
    for _, r in rc_s.iterrows():
        rd.append([r["root_cause"], r["recommended_action"], str(r["count"]), f"{r['avg']*100:.1f}%"])
    rt = Table(rd, colWidths=[3.5*cm,7*cm,2.5*cm,2.5*cm])
    rt.setStyle(TableStyle([
        ("BACKGROUND",(0,0),(-1,0),GREEN), ("TEXTCOLOR",(0,0),(-1,0),colors.white),
        ("FONTNAME",(0,0),(-1,0),"Helvetica-Bold"),
        ("ROWBACKGROUNDS",(0,1),(-1,-1),[colors.white,colors.HexColor("#f6f8fa")]),
        ("GRID",(0,0),(-1,-1),0.4,colors.HexColor("#d0d7de")),
        ("FONTSIZE",(0,0),(-1,-1),8), ("PADDING",(0,0),(-1,-1),5),
    ]))
    story += [rt, Spacer(1,10)]
    if include_chat and st.session_state.get("chat_history"):
        story.append(Paragraph("RAG Chat Session", h1_s))
        for i, t in enumerate(st.session_state["chat_history"],1):
            story.append(Paragraph(f"Q{i}: {t['question']}", ParagraphStyle("q",parent=body_s,textColor=BLUE,fontName="Helvetica-Bold")))
            story.append(Paragraph(t["answer"][:500]+("…" if len(t["answer"])>500 else ""), body_s))
            story.append(Spacer(1,6))
    if include_raw:
        story.append(Paragraph("Critical Anomaly Sample", h1_s))
        top = anomaly_df[anomaly_df["severity"]=="critical"].head(15)
        c_show = ["ue_id","timestamp","severity","if_score","packet_loss_ratio","sinr_dl_db"]
        raw = [c_show] + top[c_show].round(3).values.tolist()
        raw_t = Table(raw)
        raw_t.setStyle(TableStyle([("BACKGROUND",(0,0),(-1,0),colors.HexColor("#080c12")),("TEXTCOLOR",(0,0),(-1,0),colors.white),("FONTSIZE",(0,0),(-1,-1),7),("GRID",(0,0),(-1,-1),0.3,colors.grey),("PADDING",(0,0),(-1,-1),4)]))
        story.append(raw_t)
    doc.build(story)
    return buf.getvalue()


# ══════════════════════════════════════════════════════════════════════════════
#  MAIN
# ══════════════════════════════════════════════════════════════════════════════

def main():
    anomaly_df = load_anomaly_df()
    causal_df  = load_causal_df()
    cf_df      = load_cf_df()

    for key, default in [("page","kpi"),("sev_f",None),("ue_f",None),("chat_history",[]),("theme","dark")]:
        if key not in st.session_state:
            st.session_state[key] = default

    page = render_sidebar()

    if   page == "kpi":     page_kpi_overview(anomaly_df, causal_df)
    elif page == "anomaly":  page_anomaly_feed(anomaly_df)
    elif page == "causal":   page_causal_root_cause(causal_df, cf_df)
    elif page == "rl":       page_rl_actions(anomaly_df, causal_df)
    elif page == "chat":     page_rag_chat()
    elif page == "export":   page_export_report(anomaly_df, causal_df)


if __name__ == "__main__":
    main()