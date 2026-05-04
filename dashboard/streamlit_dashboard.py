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
    page_icon="&#128225;",
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
<style>
/* System font fallbacks — no external requests needed inside Docker */
</style>

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
  --font-display: 'Syne', 'Segoe UI', system-ui, sans-serif;
  --font-mono:    'IBM Plex Mono', 'Consolas', 'Courier New', monospace;
  --font-body:    'Inter', 'Segoe UI', system-ui, sans-serif;
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
    """Load anomaly KPI data, preferring the newest merged v2 file.

    Source priority:
      1. anomaly_scores_v2.csv  — the team's current truth (16-20 UEs, both v1
         columns like severity/if_anomaly/ae_error AND the v2 model scores).
      2. anomaly_scores_ns3.xls — legacy 5-UE NS-3 export, kept only as a
         last-resort fallback so the dashboard still loads on a fresh clone.
    """
    candidates = ["anomaly_scores_v2.csv", "anomaly_scores_ns3.xls"]
    df = None
    for fname in candidates:
        path = DATA_DIR / fname
        if not path.exists(): continue
        try:
            df = pd.read_csv(path)
            break
        except Exception:
            continue
    if df is None:
        return pd.DataFrame()
    # Drop the duplicate-column .1 suffixes pandas adds when CSVs have repeats.
    df = df.drop(columns=[c for c in df.columns if c.endswith(".1")], errors="ignore")
    if "ae_error" in df.columns:
        df["ae_error"] = pd.to_numeric(df["ae_error"], errors="coerce").fillna(0.0)
    if "if_anomaly" in df.columns and "ae_anomaly" in df.columns:
        df["consensus_anomaly"] = (
            (pd.to_numeric(df["if_anomaly"], errors="coerce").fillna(0) == 1) &
            (pd.to_numeric(df["ae_anomaly"], errors="coerce").fillna(0) == 1)
        ).astype(int)
    return df

def _load_static_plus_live(static_name: str, live_name: str = None) -> pd.DataFrame:
    """Concat the static CSV with its `*_live.csv` MLOps output when both
    exist. Static missing → return live alone. Live missing → return static."""
    live_name = live_name or static_name.replace(".csv", "_live.csv")
    static_p, live_p = DATA_DIR / static_name, DATA_DIR / live_name
    parts = []
    if static_p.exists():
        try: parts.append(pd.read_csv(static_p))
        except Exception: pass
    if live_p.exists():
        try:
            live = pd.read_csv(live_p)
            if not live.empty:
                parts.append(live)
        except Exception: pass
    if not parts:
        return pd.DataFrame()
    if len(parts) == 1:
        return parts[0]
    return pd.concat(parts, ignore_index=True, sort=False)


@st.cache_data(show_spinner=False, ttl=8)
def load_causal_df():
    return _load_static_plus_live("root_cause_labels.csv")

@st.cache_data(show_spinner=False, ttl=8)
def load_cf_df():
    df = _load_static_plus_live("counterfactuals.csv")
    if df.empty:
        return df
    if "timestamp" in df.columns:
        df = df[df["timestamp"] != "timestamp"].reset_index(drop=True)
        df["timestamp"] = pd.to_numeric(df["timestamp"], errors="coerce").fillna(1).astype(int)
    return df

@st.cache_data(show_spinner=False, ttl=8)
def load_sla_df():
    return _load_static_plus_live("sla_breach_predictions.csv")

@st.cache_data(show_spinner=False, ttl=8)
def load_severity_df():
    return _load_static_plus_live("severity_triage.csv")

@st.cache_data(show_spinner=False)
def load_bench_df():
    try:
        return pd.read_csv(DATA_DIR / "benchmark_report_by_category.csv")
    except FileNotFoundError:
        return pd.DataFrame()

@st.cache_data(show_spinner=False)
def load_rl_df():
    try:
        return pd.read_csv(DATA_DIR / "rl_vs_baseline_comparison.csv")
    except FileNotFoundError:
        return pd.DataFrame()

# ── Live-data promotion ──────────────────────────────────────────────────
# When the bridge has streamed at least this many rows, the dashboard switches
# from static CSVs to the live feed for cross-page analytics. Override via env.
LIVE_PROMOTE_THRESHOLD = int(os.getenv("QOSBUDDY_LIVE_THRESHOLD", "200"))


def _fetch_live_status():
    """Quick health check on the live buffer; returns (total_rows, connected)."""
    s, err = _rl_api_get("/live/status", timeout=4.0) if False else (None, None)
    # Avoid a circular dep at import time — use urllib directly.
    import urllib.request as _ur, json as _js
    try:
        with _ur.urlopen(f"{RL_API_URL}/live/status", timeout=3.0) as r:
            d = _js.loads(r.read().decode("utf-8"))
            return int(d.get("total_received", 0) or 0), bool(d.get("bridge_connected"))
    except Exception:
        return 0, False


def _project_live_to_v1_schema(df: pd.DataFrame) -> pd.DataFrame:
    """Project bridge live rows into the v1 anomaly_df schema.

    The bridge sends raw KPIs plus `_live_anomaly` and `_live_scores`. We add
    the columns the rest of the dashboard expects: if_anomaly, ae_anomaly,
    consensus_anomaly, if_score, severity.
    """
    if df is None or df.empty:
        return pd.DataFrame()
    df = df.copy()
    if "_live_anomaly" in df.columns:
        df["if_anomaly"] = pd.to_numeric(df["_live_anomaly"], errors="coerce").fillna(0).astype(int)
        df["ae_anomaly"] = df["if_anomaly"]
        df["consensus_anomaly"] = df["if_anomaly"]
    if "_live_scores" in df.columns:
        sdf = pd.json_normalize(df["_live_scores"].apply(lambda x: x if isinstance(x, dict) else {}))
        if "isolation_forest" in sdf.columns and "if_score" not in df.columns:
            df["if_score"] = pd.to_numeric(sdf["isolation_forest"], errors="coerce")
    if "severity" not in df.columns and "packet_loss_ratio" in df.columns:
        ploss = pd.to_numeric(df["packet_loss_ratio"], errors="coerce").fillna(0)
        flag  = df.get("if_anomaly", pd.Series([0]*len(df)))
        sev = []
        for f, p in zip(flag, ploss):
            if f == 1 and p > 0.5: sev.append("critical")
            elif f == 1 or p > 0.2: sev.append("degraded")
            else: sev.append("normal")
        df["severity"] = sev
    return df


@st.cache_data(show_spinner=False, ttl=4)
def _fetch_live_log_df(last_n: int = 5000) -> pd.DataFrame:
    """Read the persistent JSONL log via /live/log. Falls back to /live/stream."""
    import urllib.request as _ur, json as _js
    # Primary path: persistent log (full cumulative history).
    try:
        with _ur.urlopen(f"{RL_API_URL}/live/log?last_n={int(last_n)}", timeout=8.0) as r:
            payload = _js.loads(r.read().decode("utf-8"))
        rows = payload.get("data") or []
        if rows:
            return _project_live_to_v1_schema(pd.DataFrame(rows))
    except Exception:
        pass
    # Fallback: in-memory buffer (last 500 rows only).
    try:
        with _ur.urlopen(f"{RL_API_URL}/live/stream?last_n=500", timeout=5.0) as r:
            payload = _js.loads(r.read().decode("utf-8"))
        rows = payload.get("data") or []
        if rows:
            return _project_live_to_v1_schema(pd.DataFrame(rows))
    except Exception:
        pass
    return pd.DataFrame()


def get_active_anomaly_df(static_df):
    """Return static CSV augmented with the persistent live log when threshold crossed.

    MLOps pattern: the static historical dataset is the **base**, and live rows
    accumulate on top via the JSONL log. Once the cumulative live count crosses
    LIVE_PROMOTE_THRESHOLD, every page that consumes anomaly_df sees both
    historical + streamed observations concatenated.

    Sets st.session_state['_data_source'] for the sidebar indicator.
    """
    total_rx, _ = _fetch_live_status()
    static_n = 0 if (static_df is None) else len(static_df)
    if total_rx >= LIVE_PROMOTE_THRESHOLD:
        live_df = _fetch_live_log_df(last_n=5000)
        live_n  = len(live_df)
        if live_n > 0:
            if static_df is None or static_df.empty:
                st.session_state["_data_source"] = ("augmented", total_rx, 0, live_n)
                return live_df
            # Align columns — keep union, fill missing with NaN.
            combined = pd.concat([static_df, live_df], ignore_index=True, sort=False)
            st.session_state["_data_source"] = ("augmented", total_rx, static_n, live_n)
            return combined
    st.session_state["_data_source"] = ("static", total_rx, static_n, 0)
    return static_df


# ── Schema-defensive helpers ─────────────────────────────────────────────
def _col(df, *candidates, default=None):
    """Return the first candidate column that exists in df, or default."""
    if df is None or len(df) == 0:
        return default
    for c in candidates:
        if c in df.columns:
            return c
    return default

def _empty_state(title: str, hint: str = ""):
    """Render a friendly empty-state panel instead of crashing or blank chart."""
    st.markdown(
        f"""<div class="glass-panel" style="text-align:center;padding:36px 20px;">
        <div style="font-family:var(--font-mono);font-size:11px;color:var(--text-3);
             text-transform:uppercase;letter-spacing:0.18em;margin-bottom:8px;">{title}</div>
        <div style="font-size:13px;color:var(--text-2);">{hint or 'No data available for this view.'}</div>
        </div>""", unsafe_allow_html=True)

# ── RL Agent API helpers (DSO3.1 live inference) ──
RL_API_URL = os.getenv("QOSBUDDY_API_URL", "http://localhost:8000")

def _rl_api_get(path: str, timeout: float = 5.0):
    """GET wrapper for RL endpoints. Returns (data, error_message)."""
    import urllib.request, urllib.error, json as _json
    try:
        with urllib.request.urlopen(f"{RL_API_URL}{path}", timeout=timeout) as r:
            return _json.loads(r.read().decode("utf-8")), None
    except urllib.error.HTTPError as e:
        try:
            detail = _json.loads(e.read().decode("utf-8")).get("detail", str(e))
        except Exception:
            detail = str(e)
        return None, detail
    except Exception as e:
        return None, str(e)

def _rl_api_post(path: str, payload: dict, timeout: float = 120.0):
    """POST wrapper for RL endpoints."""
    import urllib.request, urllib.error, json as _json
    req = urllib.request.Request(
        f"{RL_API_URL}{path}",
        data=_json.dumps(payload).encode("utf-8"),
        headers={"Content-Type": "application/json"},
        method="POST",
    )
    try:
        with urllib.request.urlopen(req, timeout=timeout) as r:
            return _json.loads(r.read().decode("utf-8")), None
    except urllib.error.HTTPError as e:
        try:
            detail = _json.loads(e.read().decode("utf-8")).get("detail", str(e))
        except Exception:
            detail = str(e)
        return None, detail
    except Exception as e:
        return None, str(e)

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
    ("kpi",       "📊", "KPI Overview"),
    ("anomaly",   "🚨", "Anomaly Feed"),
    ("causal",    "🔬", "Causal Analysis"),
    ("forecast",  "🔮", "QoS Forecast"),
    ("sla_risk",  "⚠️",  "SLA Risk"),
    ("severity",  "🎯", "Severity Triage"),
    ("benchmark", "📈", "Benchmarking"),
    ("rl",        "🤖", "RL Actions"),
    ("live_monitor", "📡", "Live Monitor"),
    ("chat",      "💬", "Ask the Network"),
    ("voice",     "&#127897;", "Voice NOC Assistant"),
    ("export",    "📄", "Export Report"),
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
          <div class="sb-logo-ring"><div class="sb-logo-inner">&#128225;</div></div>
          <div class="sb-title">QoSBuddy</div>
          <div class="sb-sub">PingWin · ESPRIT · 2025–2026</div>
        </div>
        <div class="status-row">
          <div class="status-dot"></div>
          <span class="status-text">SYSTEM ONLINE</span>
        </div>
        """, unsafe_allow_html=True)

        # Page persistence: restore from ?page=… on refresh, sync URL on nav.
        _valid_pages = {k for k, _, _ in NAV_ITEMS}
        if "page" not in st.session_state:
            url_page = st.query_params.get("page")
            st.session_state["page"] = url_page if url_page in _valid_pages else "voice"
        if st.query_params.get("page") != st.session_state["page"]:
            st.query_params["page"] = st.session_state["page"]

        for key, icon, label in NAV_ITEMS:
            if st.button(f"{icon}  {label}", key=f"nav_{key}", width='stretch'):
                st.session_state["page"] = key
                st.query_params["page"] = key
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
        # Data-driven UE list — covers Eya's 20-UE dataset and falls back to
        # 1..20 when the data hasn't loaded yet.
        ue_choices = st.session_state.get("_ue_choices") or list(range(1, 21))
        ue  = st.selectbox("UE", ["All"] + [str(i) for i in ue_choices],
                           key="ue_sel", label_visibility="collapsed")
        st.session_state["sev_f"] = None if sev=="All" else sev
        st.session_state["ue_f"]  = None if ue=="All"  else int(ue)
        st.markdown("</div>", unsafe_allow_html=True)

        # Data-source indicator (augmented = static + live; static otherwise)
        ds = st.session_state.get("_data_source")
        if ds:
            kind = ds[0]
            total = ds[1] if len(ds) > 1 else 0
            if kind == "augmented":
                static_n = ds[2] if len(ds) > 2 else 0
                live_n   = ds[3] if len(ds) > 3 else 0
                st.markdown(
                    f"""<div style="margin:8px 16px;padding:8px 12px;border-radius:8px;
                         background:rgba(0,229,160,0.08);border:1px solid rgba(0,229,160,0.3);
                         font-family:'IBM Plex Mono';font-size:9px;letter-spacing:0.12em;
                         text-transform:uppercase;color:var(--emerald);">
                      <span style="display:inline-block;width:6px;height:6px;border-radius:50%;
                            background:var(--emerald);box-shadow:0 0 6px var(--emerald);
                            margin-right:6px;"></span>
                      AUGMENTED · {static_n:,} static + {live_n:,} live<br>
                      <span style="color:var(--text-3);font-size:8px;letter-spacing:0.18em;">
                        {total:,} rows received total
                      </span>
                    </div>""", unsafe_allow_html=True)
            else:
                st.markdown(
                    f"""<div style="margin:8px 16px;padding:8px 12px;border-radius:8px;
                         background:rgba(0,212,255,0.05);border:1px solid rgba(0,212,255,0.15);
                         font-family:'IBM Plex Mono';font-size:9px;letter-spacing:0.12em;
                         text-transform:uppercase;color:var(--text-3);">
                      STATIC CSV · {total:,} live (need {LIVE_PROMOTE_THRESHOLD})
                    </div>""", unsafe_allow_html=True)

        # ── MLOps panel: pipeline status + manual trigger ──
        with st.expander("🔧 MLOps Pipeline", expanded=False):
            pl_status, pl_err = _rl_api_get("/pipeline/status", timeout=3.0)
            if pl_err or not pl_status:
                st.caption(f"Pipeline status unavailable: {pl_err}")
            else:
                interval = pl_status.get("interval_s", 300)
                running  = pl_status.get("scheduler_running", False)
                jobs     = pl_status.get("jobs", {}) or {}
                st.markdown(
                    f"<div style='font-family:IBM Plex Mono;font-size:10px;color:var(--text-3);"
                    f"text-transform:uppercase;letter-spacing:0.12em;margin-bottom:6px;'>"
                    f"{'🟢 ACTIVE' if running else '⏸ PAUSED'} · every {interval}s"
                    f"</div>", unsafe_allow_html=True)
                # Per-job tiny rows
                STATUS_COLOR = {"ok":"var(--emerald)","fail":"var(--rose)",
                                "running":"var(--amber)","never":"var(--text-3)"}
                for jname, jrec in jobs.items():
                    s = jrec.get("last_status","never")
                    last = jrec.get("last_run") or "never"
                    last_disp = last[-8:] if last and last != "never" else "—"
                    runs = jrec.get("runs", 0)
                    err  = jrec.get("last_error") or ""
                    color = STATUS_COLOR.get(s, "var(--text-3)")
                    st.markdown(
                        f"<div style='display:flex;justify-content:space-between;"
                        f"font-family:IBM Plex Mono;font-size:9px;padding:2px 0;'>"
                        f"<span><span style='color:{color};'>●</span> "
                        f"{jname}</span>"
                        f"<span style='color:var(--text-3);'>{last_disp} · {runs}x</span>"
                        f"</div>", unsafe_allow_html=True)
                    if err and s == "fail":
                        st.caption(f"  ⚠ {err[:90]}")
                # Trigger buttons
                bc1, bc2 = st.columns(2)
                if bc1.button("▶ Run all", key="_ml_run_all", width='stretch'):
                    with st.spinner("Running all jobs…"):
                        _, e = _rl_api_post("/pipeline/run", {}, timeout=120.0)
                    if e: st.error(e)
                    else: st.cache_data.clear(); st.rerun()
                tog_label = "⏸ Pause" if running else "▶ Start"
                if bc2.button(tog_label, key="_ml_toggle", width='stretch'):
                    path = "/pipeline/scheduler/stop" if running else "/pipeline/scheduler/start"
                    _, e = _rl_api_post(path, {}, timeout=8.0)
                    if e: st.error(e)
                    else: st.rerun()

        st.markdown(f"""
        <div style="padding:10px 16px;font-family:'IBM Plex Mono';font-size:9px;
                    color:rgba(0,212,255,0.3);text-transform:uppercase;letter-spacing:0.15em;">
          {datetime.now().strftime('%Y-%m-%d  %H:%M:%S')}
        </div>""", unsafe_allow_html=True)

    return st.session_state.get("page","voice")

# ══════════════════════════════════════════════════════════════════════════════
#  COMPANION AVATAR — floating "Hey Buddy" assistant, docked bottom-right
# ══════════════════════════════════════════════════════════════════════════════

_COMPANION_LABELS = {
    "kpi":       "KPI Overview",
    "anomaly":   "Anomaly Feed",
    "causal":    "Causal Analysis",
    "forecast":  "QoS Forecast",
    "sla_risk":  "SLA Risk",
    "severity":  "Severity Triage",
    "benchmark": "Benchmarking",
    "rl":        "RL Actions",
    "live_monitor": "Live Monitor",
    "chat":      "Ask the Network",
    "voice":     "Voice NOC",
    "export":    "Export Report",
}

def render_companion(page_key: str):
    """Inject a floating companion avatar at the bottom-right of the viewport.
    The widget is rendered inside a Streamlit component iframe (same-origin)
    and attaches itself to `window.parent.document.body` so it floats above
    every dashboard page. Wake phrase: "Hey Buddy"."""
    api_url   = RL_API_URL  # reuse configured backend URL
    page_name = _COMPANION_LABELS.get(page_key, page_key.replace("_", " ").title())

    html = _COMPANION_TEMPLATE.replace("__API_URL__", api_url) \
                              .replace("__PAGE_KEY__", page_key) \
                              .replace("__PAGE_NAME__", page_name)
    # height=0 — the iframe itself is invisible; the widget lives in parent DOM.
    st.components.v1.html(html, height=0, scrolling=False)


_COMPANION_TEMPLATE = r"""
<!DOCTYPE html>
<html><head><meta charset="utf-8"><style>body{margin:0;padding:0}</style></head>
<body>
<script>
(function() {
  // Bail out if we can't reach the parent document (sandboxed).
  var pdoc, pwin;
  try { pwin = window.parent; pdoc = pwin.document; } catch(e) { return; }
  if (!pdoc || !pdoc.body) return;

  // Resolve API URL at runtime from the *browser's* location so this works
  // identically on localhost, docker-host (port 8000 mapped), or a proxied deploy.
  // Fallback to the server-rendered URL only if parent location is unavailable.
  var API_URL = (function() {
    try {
      var loc = pwin.location;
      return loc.protocol + "//" + loc.hostname + ":8000";
    } catch(e) { return "__API_URL__"; }
  })();

  var PAGE_KEY  = "__PAGE_KEY__";
  var PAGE_NAME = "__PAGE_NAME__";

  // Persistent state namespace — survives iframe destruction on Streamlit reruns.
  // Every mount refreshes the function closures on this object; the widget DOM
  // in the parent reads from here via the rebound event listeners.
  var S = pwin.__qbCompanion = pwin.__qbCompanion || {
    wakeRec: null, askRec: null,
    wakeActive: false, isListening: false, isSpeaking: false,
    abortCtl: null,
  };
  S.pageKey  = PAGE_KEY;
  S.pageName = PAGE_NAME;
  S.apiUrl   = API_URL;

  // ── Styles (scoped to #qb-companion) ─────────────────────────────────────
  var style = pdoc.getElementById("qb-companion-style");
  if (!style) {
    style = pdoc.createElement("style");
    style.id = "qb-companion-style";
    pdoc.head.appendChild(style);
  }
  style.textContent = `
    #qb-companion {
      position: fixed; right: 22px; bottom: 22px; z-index: 2147483000;
      font-family: 'IBM Plex Mono', ui-monospace, Menlo, monospace;
      color: #e6f1ff;
    }
    #qb-companion *, #qb-companion *::before, #qb-companion *::after { box-sizing: border-box; }
    #qb-companion-btn {
      width: 68px; height: 68px; border-radius: 50%; cursor: pointer;
      background: radial-gradient(circle at 30% 25%, rgba(0,229,160,0.45), rgba(0,212,255,0.35) 45%, rgba(8,12,18,0.92) 78%);
      border: 1px solid rgba(0,212,255,0.45);
      box-shadow: 0 10px 30px rgba(0,0,0,0.45),
                  0 0 24px rgba(0,212,255,0.35),
                  inset 0 0 18px rgba(0,229,160,0.18);
      display: flex; align-items: center; justify-content: center;
      transition: transform .18s ease, box-shadow .2s ease;
      position: relative; overflow: visible;
    }
    #qb-companion-btn:hover { transform: translateY(-2px) scale(1.03); }
    #qb-companion-head {
      width: 44px; height: 44px; border-radius: 50%;
      background: radial-gradient(circle at 32% 30%, #a7f3ff 0%, #00d4ff 32%, #0091b8 68%, #05202b 100%);
      box-shadow: inset 0 0 14px rgba(255,255,255,0.25), 0 0 14px rgba(0,212,255,0.55);
      position: relative;
      animation: qbPulse 2.4s ease-in-out infinite;
    }
    #qb-companion-head::before, #qb-companion-head::after {
      content: ""; position: absolute; top: 36%; width: 5px; height: 5px; border-radius: 50%;
      background: #0a1929; box-shadow: 0 0 4px rgba(0,0,0,0.55);
    }
    #qb-companion-head::before { left: 28%; }
    #qb-companion-head::after  { right: 28%; }
    #qb-companion[data-state="listening"] #qb-companion-head { animation: qbRipple 0.9s ease-in-out infinite; }
    #qb-companion[data-state="processing"] #qb-companion-head { animation: qbSpin 1.1s linear infinite; }
    #qb-companion[data-state="speaking"] #qb-companion-head  { animation: qbTalk 0.34s ease-in-out infinite; }
    @keyframes qbPulse  { 0%,100%{ transform: scale(1) } 50%{ transform: scale(1.06) } }
    @keyframes qbRipple { 0%,100%{ box-shadow: 0 0 0 0 rgba(0,229,160,0.55) } 50%{ box-shadow: 0 0 0 14px rgba(0,229,160,0) } }
    @keyframes qbSpin   { to { filter: hue-rotate(360deg) } }
    @keyframes qbTalk   { 0%,100%{ transform: scaleY(1) } 50%{ transform: scaleY(0.86) } }
    #qb-companion-status {
      position: absolute; bottom: -3px; right: -3px; width: 16px; height: 16px;
      border-radius: 50%; background: #00e5a0;
      box-shadow: 0 0 6px #00e5a0; border: 2px solid #080c12;
    }
    #qb-companion[data-state="listening"]  #qb-companion-status { background: #ffb83f; box-shadow: 0 0 8px #ffb83f; }
    #qb-companion[data-state="processing"] #qb-companion-status { background: #a78bfa; box-shadow: 0 0 8px #a78bfa; }
    #qb-companion[data-state="speaking"]   #qb-companion-status { background: #ff4d6d; box-shadow: 0 0 8px #ff4d6d; }
    #qb-companion[data-state="offline"]    #qb-companion-status { background: #4a5568; box-shadow: none; }

    #qb-companion-panel {
      position: absolute; right: 0; bottom: 82px; width: 360px;
      background: rgba(8,12,18,0.88);
      border: 1px solid rgba(0,212,255,0.25);
      border-radius: 16px; padding: 16px 16px 12px;
      backdrop-filter: blur(18px) saturate(140%);
      box-shadow: 0 16px 44px rgba(0,0,0,0.55), 0 0 28px rgba(0,212,255,0.12);
      transform-origin: bottom right;
      transform: scale(0.88) translateY(8px); opacity: 0; pointer-events: none;
      transition: transform .22s ease, opacity .22s ease;
    }
    #qb-companion.open #qb-companion-panel { transform: scale(1) translateY(0); opacity: 1; pointer-events: auto; }
    #qb-companion-title {
      font-size: 11px; letter-spacing: 0.18em; text-transform: uppercase;
      color: rgba(0,212,255,0.75); margin-bottom: 6px;
    }
    #qb-companion-page {
      font-family: 'Syne', system-ui, sans-serif; font-weight: 700;
      font-size: 16px; color: #e6f1ff; margin-bottom: 2px;
    }
    #qb-companion-hint {
      font-size: 10px; color: rgba(230,241,255,0.55); margin-bottom: 10px;
    }
    #qb-companion-transcript {
      font-size: 12px; line-height: 1.45; color: #cbd5e1;
      background: rgba(0,212,255,0.05); border: 1px dashed rgba(0,212,255,0.2);
      border-radius: 10px; padding: 8px 10px; min-height: 38px; margin-bottom: 10px;
      word-break: break-word;
    }
    #qb-companion-answer {
      font-size: 12px; line-height: 1.5; color: #e6f1ff;
      background: rgba(0,229,160,0.05); border: 1px solid rgba(0,229,160,0.22);
      border-radius: 10px; padding: 8px 10px; max-height: 160px; overflow-y: auto;
      margin-bottom: 10px; display: none;
    }
    #qb-companion-answer.visible { display: block; }
    #qb-companion-controls { display: flex; gap: 8px; }
    #qb-companion-controls button {
      flex: 1; background: rgba(0,212,255,0.1); color: #e6f1ff;
      border: 1px solid rgba(0,212,255,0.25); border-radius: 10px;
      padding: 8px 10px; font-family: inherit; font-size: 11px;
      text-transform: uppercase; letter-spacing: 0.12em; cursor: pointer;
      transition: background .15s, border-color .15s;
    }
    #qb-companion-controls button:hover { background: rgba(0,212,255,0.2); border-color: rgba(0,212,255,0.5); }
    #qb-companion-controls .primary {
      background: linear-gradient(135deg, rgba(0,229,160,0.3), rgba(0,212,255,0.3));
      border-color: rgba(0,229,160,0.45);
    }
    #qb-companion-close {
      position: absolute; top: 8px; right: 10px; background: transparent; color: rgba(230,241,255,0.5);
      border: none; font-size: 16px; cursor: pointer; line-height: 1;
    }
    #qb-companion-close:hover { color: #ff4d6d; }
  `;

  // ── Widget DOM ───────────────────────────────────────────────────────────
  // Build once; subsequent mounts only update page context + rebind listeners.
  var root = pdoc.getElementById("qb-companion");
  if (!root) {
    root = pdoc.createElement("div");
    root.id = "qb-companion";
    root.dataset.state = "idle";
    root.innerHTML = `
      <div id="qb-companion-panel" role="dialog" aria-label="QoSBuddy Companion">
        <button id="qb-companion-close" title="Close" aria-label="Close">×</button>
        <div id="qb-companion-title">QoSBuddy Companion</div>
        <div id="qb-companion-page"></div>
        <div id="qb-companion-hint">Say <b>"Hey Buddy"</b> or tap the mic to ask anything about this page.</div>
        <div id="qb-companion-transcript">Listening for wake word…</div>
        <div id="qb-companion-answer"></div>
        <div id="qb-companion-controls">
          <button class="primary" id="qb-companion-mic">🎙️ Ask</button>
          <button id="qb-companion-stop">⏹ Stop</button>
        </div>
      </div>
      <div id="qb-companion-btn" title="QoSBuddy companion — say &quot;Hey Buddy&quot;" role="button" tabindex="0" aria-label="Open QoSBuddy companion">
        <div id="qb-companion-head"></div>
        <div id="qb-companion-status"></div>
      </div>
    `;
    pdoc.body.appendChild(root);
  }
  root.dataset.page = PAGE_KEY;
  root.dataset.pageName = PAGE_NAME;

  // Helpers that always re-query the DOM by id, so they work after rebind too.
  function $id(id) { return pdoc.getElementById(id); }
  function setState(s)     { root.dataset.state = s; }
  function setTranscript(t){ var el = $id("qb-companion-transcript"); if (el) el.textContent = t; }
  function setAnswer(t) {
    var el = $id("qb-companion-answer");
    if (!el) return;
    el.textContent = t;
    el.classList.toggle("visible", !!t);
  }
  function openPanel()  { root.classList.add("open"); }
  function closePanel() { root.classList.remove("open"); }
  function togglePanel() {
    if (root.classList.contains("open")) closePanel(); else openPanel();
  }

  // Always show the *current* page name in the panel header.
  var pageEl = $id("qb-companion-page");
  if (pageEl) pageEl.textContent = PAGE_NAME;

  // ── Speech stack ────────────────────────────────────────────────────────
  var SR    = pwin.SpeechRecognition || pwin.webkitSpeechRecognition;
  var synth = pwin.speechSynthesis;

  if (!SR || !synth) {
    setState("offline");
    setTranscript("Voice needs Chrome + microphone permission.");
    // Still rebind the click-to-open behavior so the panel is usable.
    rebindButtons();
    return;
  }

  var WAKE = ["hey buddy","hey body","hey budy","hey buddi","hey bud","hi buddy","ok buddy","okay buddy"];
  function isWake(t) {
    t = (t||"").toLowerCase().trim();
    for (var i=0; i<WAKE.length; i++) if (t.indexOf(WAKE[i]) !== -1) return true;
    return /\bhey\b/.test(t) && /\bbu|\bbo/.test(t);
  }

  // ── Voice interrupt commands ───────────────────────────────────────────
  // Stop  = interrupt, KEEP remaining sentences so the user can resume.
  // Resume= continue from where we paused.
  var STOP_RE   = /\b(stop|halt|shut up|quiet|pause|enough|be quiet|hold on)\b/;
  var RESUME_RE = /\b(continue|continu|proceed|keep going|go on|resume|carry on|the rest|rest please|finish it|keep talking|go ahead|please continue)\b/;

  function wordCount(t) { return (t.trim().split(/\s+/).filter(Boolean)).length; }
  function isStopCmd(t)   { return wordCount(t) <= 5 && STOP_RE.test(t); }
  function isResumeCmd(t) { return wordCount(t) <= 6 && RESUME_RE.test(t); }

  // Echo guard: if the recognized text is likely the mic picking up the
  // assistant's own TTS output, ignore it (no false interrupts).
  function isEcho(heard) {
    if (!S.currentChunk) return false;
    var hw = heard.split(/\s+/).filter(function(w){ return w.length > 2; });
    if (hw.length < 2) return false;
    var m = 0;
    for (var i=0; i<hw.length; i++) if (S.currentChunk.indexOf(hw[i]) !== -1) m++;
    return (m / hw.length) >= 0.55;
  }

  // Real question = at least 3 words and not a plain stop/resume command.
  function looksLikeQuestion(t) {
    return wordCount(t) >= 3 && !isStopCmd(t) && !isResumeCmd(t);
  }

  // ── Chunked speech synthesis so we can stop/resume between sentences ──
  function splitIntoChunks(text) {
    var clean = (text || "")
      .replace(/[\u{1F300}-\u{1FAFF}\u{2600}-\u{27BF}]/gu, "")
      .replace(/\*\*/g, "")
      .replace(/\r/g, "")
      .trim();
    var parts = clean.match(/[^.!?\u2026\n]+[.!?\u2026]+|[^.!?\u2026\n]+$/g) || [clean];
    var out = [];
    for (var i=0; i<parts.length; i++) {
      var p = parts[i].trim();
      if (!p) continue;
      while (p.length > 220) {
        var cut = p.lastIndexOf(" ", 200);
        if (cut < 100) cut = 200;
        out.push(p.slice(0, cut).trim());
        p = p.slice(cut).trim();
      }
      if (p) out.push(p);
    }
    return out;
  }

  function pickVoice() {
    var voices = synth.getVoices();
    return voices.find(function(v){ return v.lang.startsWith("en") && (v.name.indexOf("Google")>=0 || v.name.indexOf("Natural")>=0); })
        || voices.find(function(v){ return v.lang.startsWith("en"); });
  }

  function speak(text, done) {
    if (!text) { if (done) done(); return; }
    try { synth.cancel(); } catch(e) {}
    S.pendingSpeech = splitIntoChunks(text);
    S.speechDone    = done || null;
    if (!S.pendingSpeech.length) { if (done) done(); return; }
    setState("speaking");
    S.isSpeaking = true;
    // Keep the wake recognizer running during speech so we can catch
    // "stop" / "continue" / or a whole new question without needing the wake word.
    if (!S.wakeActive) setTimeout(startWake, 50);
    speakNext();
  }

  function speakNext() {
    if (!S.isSpeaking) return; // externally stopped
    if (!S.pendingSpeech || !S.pendingSpeech.length) {
      S.isSpeaking = false;
      S.currentChunk = "";
      setState("idle");
      var d = S.speechDone; S.speechDone = null;
      if (d) try { d(); } catch(e) {}
      if (!S.wakeActive) setTimeout(startWake, 400);
      return;
    }
    var chunk = S.pendingSpeech.shift();
    S.currentChunk = chunk.toLowerCase();
    var u = new pwin.SpeechSynthesisUtterance(chunk);
    u.rate = 0.96; u.pitch = 1.02; u.volume = 1.0;
    var pref = pickVoice();
    if (pref) u.voice = pref;
    u.onend   = function(){ if (S.isSpeaking) setTimeout(speakNext, 120); };
    u.onerror = function(){ if (S.isSpeaking) setTimeout(speakNext, 120); };
    try { synth.speak(u); }
    catch(e) { if (S.isSpeaking) setTimeout(speakNext, 120); }
  }

  // Stop TTS. If keepPending=true, remaining sentences stay in the queue and
  // can be resumed by saying "continue". Otherwise the queue is cleared.
  function stopSpeaking(keepPending) {
    try { synth.cancel(); } catch(e) {}
    S.isSpeaking = false;
    S.currentChunk = "";
    if (!keepPending) S.pendingSpeech = [];
    setState("idle");
    if (S.pendingSpeech && S.pendingSpeech.length) {
      setTranscript("Paused \u2014 say \u201Ccontinue\u201D to resume or ask a new question.");
    }
    if (!S.wakeActive) setTimeout(startWake, 150);
  }

  function resumeSpeaking() {
    if (!S.pendingSpeech || !S.pendingSpeech.length) return false;
    setState("speaking");
    S.isSpeaking = true;
    setTranscript("Resuming\u2026");
    speakNext();
    return true;
  }

  function ask(question) {
    setState("processing"); openPanel();
    setTranscript("\u201C" + question + "\u201D");
    setAnswer("Thinking\u2026 (first LLM query can take up to a minute while the model warms up)");
    // Abort any in-flight request first.
    try { if (S.abortCtl) S.abortCtl.abort(); } catch(e) {}
    S.abortCtl = (typeof AbortController !== "undefined") ? new AbortController() : null;
    var timeoutId = setTimeout(function(){
      try { if (S.abortCtl) S.abortCtl.abort(); } catch(e) {}
    }, 120000);

    var page = root.dataset.page || S.pageKey || PAGE_KEY;
    fetch(S.apiUrl + "/companion/ask", {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify({ question: question, page: page }),
      signal: S.abortCtl ? S.abortCtl.signal : undefined
    })
    .then(function(r){
      if (!r.ok) return r.text().then(function(t){ return Promise.reject("HTTP " + r.status + (t ? ": " + t.slice(0,120) : "")); });
      return r.json();
    })
    .then(function(d){
      clearTimeout(timeoutId);
      var ans = (d && d.answer) || "I couldn't find an answer for that.";
      setAnswer(ans);
      speak(ans);
    })
    .catch(function(err){
      clearTimeout(timeoutId);
      setState("idle");
      var msg = (err && err.name === "AbortError") ? "Request timed out." :
                (typeof err === "string") ? err :
                (err && err.message) ? err.message : String(err);
      setAnswer("Couldn't reach the backend \u2014 " + msg + " (tried " + S.apiUrl + ")");
      setTimeout(startWake, 1500);
    });
  }

  function startAsk() {
    if (S.isListening) return;
    // If we're speaking, cancel it and drop pending sentences so the mic can
    // pick up a new question clearly.
    if (S.isSpeaking || (S.pendingSpeech && S.pendingSpeech.length)) stopSpeaking(false);
    stopWake();
    openPanel();
    setTranscript("Listening\u2026");
    setState("listening");
    try {
      S.askRec = new SR();
    } catch(e) {
      setState("idle"); setAnswer("Speech recognition unavailable."); return;
    }
    var rec = S.askRec;
    rec.lang = "en-US"; rec.continuous = false;
    rec.interimResults = true; rec.maxAlternatives = 1;
    var finalText = "";
    rec.onstart = function(){ S.isListening = true; };
    rec.onresult = function(e){
      var interim = ""; finalText = "";
      for (var i=e.resultIndex; i<e.results.length; i++) {
        var t = e.results[i][0].transcript;
        if (e.results[i].isFinal) finalText += t; else interim += t;
      }
      setTranscript(finalText || interim || "\u2026");
    };
    rec.onerror = function(){ S.isListening = false; setState("idle"); setTimeout(startWake, 1000); };
    rec.onend = function(){
      S.isListening = false;
      if (finalText.trim()) ask(finalText.trim());
      else { setState("idle"); setTimeout(startWake, 800); }
    };
    try { rec.start(); }
    catch(e) { S.isListening = false; setState("idle"); setTimeout(startWake, 1200); }
  }

  function stopAsk() {
    if (S.askRec) { try { S.askRec.stop(); } catch(e) {} }
    try { if (S.abortCtl) S.abortCtl.abort(); } catch(e) {}
    stopSpeaking(false);
    S.isListening = false;
    setState("idle");
    setTimeout(startWake, 400);
  }

  // Wake recognizer runs continuously — it catches the wake word when idle AND
  // catches voice commands ("stop", "continue") + new questions while speaking.
  // Only paused when askRec is actively capturing a question, to avoid mic conflict.
  function startWake() {
    if (!SR || S.wakeActive || S.isListening) return;
    try {
      S.wakeRec = new SR();
    } catch(e) { S.wakeActive = false; return; }
    var rec = S.wakeRec;
    rec.lang = "en-US"; rec.continuous = true;
    rec.interimResults = true; rec.maxAlternatives = 3;
    rec.onstart = function(){ S.wakeActive = true; };

    rec.onresult = function(e){
      for (var i=e.resultIndex; i<e.results.length; i++) {
        // Only act on finals during speech/paused (avoid triggering on half-heard echo).
        var isFinal = e.results[i].isFinal;
        for (var j=0; j<e.results[i].length; j++) {
          var raw = (e.results[i][j].transcript || "").trim();
          if (!raw) continue;
          var t = raw.toLowerCase();

          // 1. Wake word — always wins, starts a new question capture.
          if (isWake(t)) {
            try { rec.stop(); } catch(err) {}
            S.wakeActive = false;
            setTimeout(startAsk, 150);
            return;
          }

          // 2. Interrupt while speaking.
          if (S.isSpeaking) {
            if (!isFinal) continue;
            if (isEcho(t)) continue;
            if (isStopCmd(t)) {
              stopSpeaking(true); // keep pending, user may say "continue"
              return;
            }
            if (looksLikeQuestion(t)) {
              // New question mid-speech → drop pending, answer new one.
              try { rec.stop(); } catch(err) {}
              S.wakeActive = false;
              stopSpeaking(false);
              setTimeout(function(){ ask(raw); }, 100);
              return;
            }
            continue; // ignore short ambient speech
          }

          // 3. Paused (stopped mid-answer with pending chunks left).
          if (!S.isSpeaking && S.pendingSpeech && S.pendingSpeech.length) {
            if (!isFinal) continue;
            if (isResumeCmd(t)) {
              resumeSpeaking();
              return;
            }
            if (looksLikeQuestion(t)) {
              try { rec.stop(); } catch(err) {}
              S.wakeActive = false;
              S.pendingSpeech = [];
              setTimeout(function(){ ask(raw); }, 100);
              return;
            }
          }
        }
      }
    };

    rec.onerror = function(){ S.wakeActive = false; setTimeout(startWake, 2500); };
    rec.onend   = function(){
      S.wakeActive = false;
      // Always restart unless the user is actively dictating a question.
      if (!S.isListening) setTimeout(startWake, 1200);
    };
    try { rec.start(); } catch(e) { S.wakeActive = false; setTimeout(startWake, 2500); }
  }

  function stopWake() {
    if (S.wakeRec) { try { S.wakeRec.stop(); } catch(e) {} }
    S.wakeActive = false;
  }

  // ── Rebind: clone + replace each interactive element so stale closures
  // from a destroyed iframe are stripped, then attach the fresh handlers
  // that close over THIS mount's up-to-date S namespace.
  function rebindButtons() {
    function swap(id) {
      var el = $id(id);
      if (!el) return null;
      var clone = el.cloneNode(true);
      el.parentNode.replaceChild(clone, el);
      return clone;
    }
    var btn      = swap("qb-companion-btn");
    var closeBtn = swap("qb-companion-close");
    var micBtn   = swap("qb-companion-mic");
    var stopBtn  = swap("qb-companion-stop");

    if (btn) {
      btn.addEventListener("click", function(ev){ ev.stopPropagation(); togglePanel(); });
      btn.addEventListener("keydown", function(ev){
        if (ev.key === "Enter" || ev.key === " ") { ev.preventDefault(); togglePanel(); }
      });
    }
    if (closeBtn) closeBtn.addEventListener("click", function(ev){ ev.stopPropagation(); closePanel(); });
    if (micBtn)   micBtn.addEventListener("click",   function(ev){ ev.stopPropagation(); startAsk(); });
    if (stopBtn)  stopBtn.addEventListener("click",  function(ev){ ev.stopPropagation(); stopAsk(); });
  }

  rebindButtons();

  // Escape-key handler: install once, but always call the *current* closePanel.
  if (!S.escInstalled) {
    S.escInstalled = true;
    pdoc.addEventListener("keydown", function(ev){
      if (ev.key === "Escape") {
        var r = pdoc.getElementById("qb-companion");
        if (r && r.classList.contains("open")) r.classList.remove("open");
      }
    });
  }

  // Kick off wake-word listening only the first time; subsequent mounts keep
  // the existing recognizer alive. If it died (state desynced), restart.
  if (!S.wakeBootstrapped) {
    S.wakeBootstrapped = true;
    setTimeout(startWake, 1500);
  } else if (!S.wakeActive && !S.isListening && !S.isSpeaking) {
    setTimeout(startWake, 600);
  }

  console.log("[QoSBuddy Companion] mounted on:", PAGE_NAME, "\u2022 api:", S.apiUrl);
})();
</script>
</body></html>
"""


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
    if anomaly_df is None or len(anomaly_df) == 0:
        _empty_state("KPI OVERVIEW", "Anomaly data file not found. Check QOSBUDDY_DATA_DIR.")
        return
    df = apply_filters(anomaly_df)

    def _safe_mean(col, scale=1.0):
        if col not in df.columns: return None
        v = pd.to_numeric(df[col], errors="coerce").mean()
        return None if pd.isna(v) else v * scale

    avg_jitter     = _safe_mean("jitter_ms")
    avg_pkt_loss   = _safe_mean("packet_loss_ratio", 100)
    avg_throughput = _safe_mean("throughput_mbps")
    avg_delay      = _safe_mean("delay_ms")
    anomaly_rate   = _safe_mean("if_anomaly", 100)
    sinr_mean      = _safe_mean("sinr_dl_db")

    cols = st.columns(6)
    with cols[0]:
        if avg_jitter is None:
            kpi_card("Avg Jitter", "—", "ms", "ok", 0, "no data")
        else:
            s = "ok" if avg_jitter<10 else "warn" if avg_jitter<50 else "crit"
            kpi_card("Avg Jitter", f"{avg_jitter:.1f}", "ms", s, min(avg_jitter,100),
                     "▲ ELEVATED" if s=="warn" else ("🔴 CRITICAL" if s=="crit" else "✓ NOMINAL"))
    with cols[1]:
        if avg_pkt_loss is None:
            kpi_card("Packet Loss", "—", "%", "ok", 0, "no data")
        else:
            s = "ok" if avg_pkt_loss<10 else "warn" if avg_pkt_loss<40 else "crit"
            kpi_card("Packet Loss", f"{avg_pkt_loss:.1f}", "%", s, avg_pkt_loss,
                     "🔴 HIGH LOSS" if s=="crit" else ("⚠ MODERATE" if s=="warn" else "✓ NOMINAL"))
    with cols[2]:
        if avg_throughput is None:
            kpi_card("Throughput", "—", "Mbps", "ok", 0, "no data")
        else:
            s = "ok" if avg_throughput>2 else "warn" if avg_throughput>0.5 else "crit"
            kpi_card("Throughput", f"{avg_throughput:.2f}", "Mbps", s, min(avg_throughput/10*100,100),
                     "🔴 DEGRADED" if s=="crit" else ("⚠ LOW" if s=="warn" else "✓ NOMINAL"))
    with cols[3]:
        if avg_delay is None:
            kpi_card("Avg Delay", "—", "ms", "ok", 0, "no data")
        else:
            s = "ok" if avg_delay<50 else "warn" if avg_delay<200 else "crit"
            kpi_card("Avg Delay", f"{avg_delay:.0f}", "ms", s, min(avg_delay/500*100,100),
                     "🔴 HIGH RTT" if s=="crit" else ("⚠ ELEVATED" if s=="warn" else "✓ NOMINAL"))
    with cols[4]:
        if anomaly_rate is None:
            kpi_card("Anomaly Rate", "—", "%", "ok", 0, "no data")
        else:
            s = "ok" if anomaly_rate<20 else "warn" if anomaly_rate<45 else "crit"
            kpi_card("Anomaly Rate", f"{anomaly_rate:.1f}", "%", s, anomaly_rate,
                     "🔴 CRISIS" if s=="crit" else ("⚠ WATCHLIST" if s=="warn" else "✓ STABLE"))
    with cols[5]:
        if sinr_mean is None:
            kpi_card("Avg SINR", "—", "dB", "ok", 0, "no data")
        else:
            s = "ok" if sinr_mean>5 else "warn" if sinr_mean>-5 else "crit"
            kpi_card("Avg SINR", f"{sinr_mean:.1f}", "dB", s, min((sinr_mean+30)/60*100,100),
                     "🔴 POOR RF" if s=="crit" else ("⚠ MARGINAL" if s=="warn" else "✓ NOMINAL"))

    st.markdown("<div style='height:8px'></div>", unsafe_allow_html=True)
    col_l, col_m, col_r = st.columns([1,1,2])

    with col_l:
        st.markdown('<div class="glass-panel"><div class="glass-panel-title">Severity Distribution</div>', unsafe_allow_html=True)
        if "severity" in df.columns:
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
        else:
            st.markdown("<div style='padding:30px;text-align:center;color:var(--text-3);font-size:11px;'>severity column not in this dataset</div>", unsafe_allow_html=True)
        st.markdown("</div>", unsafe_allow_html=True)

    with col_m:
        st.markdown('<div class="glass-panel"><div class="glass-panel-title">Root Cause Breakdown</div>', unsafe_allow_html=True)
        if causal_df is None or len(causal_df) == 0 or "root_cause" not in causal_df.columns:
            st.markdown("<div style='padding:30px;text-align:center;color:var(--text-3);font-size:11px;'>no causal labels available</div>", unsafe_allow_html=True)
        else:
            rc_df = causal_df.copy()
            if st.session_state.get("sev_f") and "severity" in rc_df.columns:
                rc_df = rc_df[rc_df["severity"]==st.session_state["sev_f"]]
            rc = rc_df["root_cause"].value_counts().reset_index()
            rc.columns = ["root_cause","count"]
            rc = rc.sort_values("count")
            if rc.empty:
                st.markdown("<div style='padding:30px;text-align:center;color:var(--text-3);font-size:11px;'>no events match filter</div>", unsafe_allow_html=True)
            else:
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
        # Pick the 8 UEs with the most rows so we adapt to whatever dataset is
        # loaded (5-UE NS-3 anomaly file, 10-UE original, 20-UE LENA, etc.).
        if "ue_id" in df.columns:
            top_ues = df["ue_id"].value_counts().head(8).index.tolist()
            top_ues = sorted(top_ues)
            # Stash the full UE list for the sidebar filter
            try:
                st.session_state["_ue_choices"] = sorted(df["ue_id"].dropna().unique().astype(int).tolist())
            except Exception:
                pass
            label_range = f"{int(min(top_ues))}–{int(max(top_ues))}" if top_ues else "n/a"
        else:
            top_ues, label_range = [], "n/a"
        st.markdown(f'<div class="glass-panel"><div class="glass-panel-title">Packet Loss Timeline — Top 8 UEs ({label_range})</div>', unsafe_allow_html=True)
        if "ue_id" in df.columns and "timestamp" in df.columns:
            df_ts = df[df["ue_id"].isin(top_ues)].sort_values(["ue_id","timestamp"])
            x_col = "timestamp"
        elif "window_idx" in df.columns:
            df_ts = df.copy()
            x_col = "window_idx"
        else:
            df_ts = df.copy()
            x_col = df.columns[0]
        fig = go.Figure()
        ue_col = "ue_id" if "ue_id" in df_ts.columns else None
        if ue_col and df_ts[ue_col].nunique() > 1:
            for i, ue in enumerate(sorted(df_ts[ue_col].unique())[:8]):
                d = df_ts[df_ts[ue_col]==ue]
                fig.add_trace(go.Scatter(
                    x=d[x_col], y=d["packet_loss_ratio"] if "packet_loss_ratio" in d.columns else d.iloc[:,1], name=f"UE {ue}", mode="lines",
                    line=dict(width=1.5, color=COLORS["seq"][i%len(COLORS["seq"])]),
                ))
        elif "packet_loss_ratio" in df_ts.columns:
            fig.add_trace(go.Scatter(
                x=df_ts[x_col], y=df_ts["packet_loss_ratio"], name="Packet Loss", mode="lines",
                line=dict(width=1.5, color=COLORS["cyan"]),
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
        if "load_level" in df.columns and "sinr_dl_db" in df.columns:
            fig = go.Figure()
            for i, ll in enumerate(sorted(df["load_level"].dropna().unique())):
                d = pd.to_numeric(df[df["load_level"]==ll]["sinr_dl_db"], errors="coerce").dropna()
                if d.empty: continue
                c = COLORS["seq"][i%len(COLORS["seq"])]
                fig.add_trace(go.Box(y=d, name=f"Load {ll}", marker_color=c,
                                     line_color=c, fillcolor=hex_to_rgba(c,0.13), boxmean="sd"))
            if fig.data:
                fig.update_layout(**PLOT_LAYOUT, height=280, showlegend=False,
                                  margin=dict(t=10,b=10,l=10,r=10), yaxis_title="SINR (dB)")
                st.plotly_chart(fig, width='stretch')
            else:
                st.markdown("<div style='padding:30px;text-align:center;color:var(--text-3);font-size:11px;'>no SINR data</div>", unsafe_allow_html=True)
        else:
            st.markdown("<div style='padding:30px;text-align:center;color:var(--text-3);font-size:11px;'>load_level / sinr_dl_db not in dataset</div>", unsafe_allow_html=True)
        st.markdown("</div>", unsafe_allow_html=True)

    with col_b:
        st.markdown('<div class="glass-panel"><div class="glass-panel-title">Detector Agreement Matrix</div>', unsafe_allow_html=True)
        if "if_anomaly" in df.columns and "ae_anomaly" in df.columns:
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
        else:
            st.markdown("<div style='padding:30px;text-align:center;color:var(--text-3);font-size:11px;'>detector columns (if_anomaly/ae_anomaly) not in dataset</div>", unsafe_allow_html=True)
        st.markdown("</div>", unsafe_allow_html=True)


# ══════════════════════════════════════════════════════════════════════════════
#  PAGE 2 — ANOMALY FEED
# ══════════════════════════════════════════════════════════════════════════════

def page_anomaly_feed(anomaly_df):
    page_header("🚨","Anomaly Feed","DSO2.2 · DSO2.3 · Isolation Forest + LSTM Autoencoder Triage Board")
    if anomaly_df is None or len(anomaly_df) == 0:
        _empty_state("ANOMALY FEED", "Anomaly data file not found.")
        return
    df = apply_filters(anomaly_df)

    n_crit = int((df["severity"]=="critical").sum()) if "severity" in df.columns else 0
    n_degr = int((df["severity"]=="degraded").sum()) if "severity" in df.columns else 0
    n_norm = int((df["severity"]=="normal").sum())   if "severity" in df.columns else 0
    n_cons = int(df["consensus_anomaly"].sum())      if "consensus_anomaly" in df.columns else 0

    st.markdown(f"""
    <div class="stat-row">
      <div class="stat-chip total"><span class="stat-chip-label">Total Events</span><span class="stat-chip-value">{len(df):,}</span></div>
      <div class="stat-chip crit"><span class="stat-chip-label">Critical</span><span class="stat-chip-value">{n_crit:,}</span></div>
      <div class="stat-chip warn"><span class="stat-chip-label">Degraded</span><span class="stat-chip-value">{n_degr:,}</span></div>
      <div class="stat-chip ok"><span class="stat-chip-label">Normal</span><span class="stat-chip-value">{n_norm:,}</span></div>
      <div class="stat-chip total"><span class="stat-chip-label">Consensus</span><span class="stat-chip-value">{n_cons:,}</span></div>
    </div>""", unsafe_allow_html=True)

    col_l, col_r = st.columns([3,2])
    with col_l:
        st.markdown('<div class="glass-panel"><div class="glass-panel-title">IF Score vs AE Reconstruction Error</div>', unsafe_allow_html=True)
        if {"if_score","ae_error","severity"}.issubset(df.columns) and len(df) > 0:
            sample = df.sample(min(3000,len(df)), random_state=42)
            fig = go.Figure()
            for sev, col in [("critical",COLORS["critical"]),("degraded",COLORS["degraded"]),("normal",COLORS["normal"])]:
                d = sample[sample["severity"]==sev]
                if d.empty: continue
                fig.add_trace(go.Scatter(x=d["if_score"], y=d["ae_error"], name=sev.capitalize(), mode="markers",
                                         marker=dict(color=hex_to_rgba(col,0.6), size=4, line=dict(color=col,width=0.5)),
                                         hovertemplate=f"<b>{sev}</b><br>IF=%{{x:.4f}}<br>AE=%{{y:.4f}}<extra></extra>"))
            if fig.data:
                fig.update_layout(**PLOT_LAYOUT, height=300, xaxis_title="IF Anomaly Score",
                                  yaxis_title="AE Reconstruction Error", margin=dict(t=10,b=10,l=10,r=10))
                st.plotly_chart(fig, width='stretch')
            else:
                st.markdown("<div style='padding:30px;text-align:center;color:var(--text-3);font-size:11px;'>no scored events</div>", unsafe_allow_html=True)
        else:
            st.markdown("<div style='padding:30px;text-align:center;color:var(--text-3);font-size:11px;'>if_score / ae_error / severity not in dataset (v2 schema?)</div>", unsafe_allow_html=True)
        st.markdown("</div>", unsafe_allow_html=True)

    with col_r:
        st.markdown('<div class="glass-panel"><div class="glass-panel-title">Consensus Heatmap — UE × Load Level</div>', unsafe_allow_html=True)
        if {"ue_id","load_level","consensus_anomaly"}.issubset(anomaly_df.columns):
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
        else:
            st.markdown("<div style='padding:30px;text-align:center;color:var(--text-3);font-size:11px;'>ue_id / load_level / consensus_anomaly not in dataset</div>", unsafe_allow_html=True)
        st.markdown("</div>", unsafe_allow_html=True)

    st.markdown('<div class="glass-panel"><div class="glass-panel-title">🔴 Critical Event Triage — Top 20 by IF Score</div>', unsafe_allow_html=True)
    if "severity" in df.columns and "if_score" in df.columns:
        top = df[df["severity"]=="critical"].sort_values("if_score",ascending=False).head(20)
    elif "severity" in df.columns:
        top = df[df["severity"]=="critical"].head(20)
    else:
        top = df.head(20)
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

    # ── SHAP global feature importance (M1 explainability) ────────────────────
    st.markdown('<div class="glass-panel"><div class="glass-panel-title">🧠 SHAP Feature Importance — Why the models flag anomalies</div>', unsafe_allow_html=True)
    shap_resp, shap_err = _rl_api_get("/explain/shap?limit=15", timeout=6.0)
    if shap_err or not shap_resp or "data" not in shap_resp:
        st.info(f"SHAP importance unavailable: {shap_err or 'no data'}")
    else:
        s_rows = shap_resp.get("data", [])
        if not s_rows:
            st.info("SHAP rows empty.")
        else:
            sdf = pd.DataFrame(s_rows)
            feat_col = "feature" if "feature" in sdf.columns else sdf.columns[0]
            val_col  = "mean_abs_shap" if "mean_abs_shap" in sdf.columns else sdf.columns[1]
            sdf = sdf.sort_values(val_col, ascending=True).tail(15)
            fig = go.Figure(go.Bar(
                x=sdf[val_col], y=sdf[feat_col], orientation="h",
                marker=dict(color=COLORS["cyan"]),
                hovertemplate="<b>%{y}</b><br>|SHAP|=%{x:.4f}<extra></extra>",
            ))
            _sh_xaxis = {**PLOT_LAYOUT.get("xaxis", {}), "title": "mean |SHAP|"}
            fig.update_layout(**{k: v for k, v in PLOT_LAYOUT.items() if k != "xaxis"},
                              height=420, margin=dict(t=10,b=10,l=10,r=10),
                              xaxis=_sh_xaxis)
            st.plotly_chart(fig, use_container_width=True)
    st.markdown("</div>", unsafe_allow_html=True)

    # ── Live anomaly scorer (M1 supervised ensemble) ──────────────────────────
    st.markdown('<div class="glass-panel"><div class="glass-panel-title">🔬 Live Anomaly Scorer — Ensemble (IF + XGBoost + RF + GBT)</div>', unsafe_allow_html=True)
    st.caption("Enter a KPI window and score it against all loaded v2 supervised models.")
    with st.form("_anom_score_form", clear_on_submit=False):
        c1, c2, c3, c4 = st.columns(4)
        sinr   = c1.number_input("SINR DL (dB)",    value=10.0,  step=0.5, format="%.2f")
        thr    = c2.number_input("Throughput (Mbps)", value=2.5, step=0.1, format="%.2f")
        delay  = c3.number_input("Delay (ms)",       value=50.0, step=1.0, format="%.2f")
        jit    = c4.number_input("Jitter (ms)",      value=5.0,  step=0.5, format="%.2f")
        c5, c6, c7, c8 = st.columns(4)
        ploss  = c5.number_input("Packet Loss",      value=0.02, step=0.01, format="%.3f")
        prb    = c6.number_input("PRB Util",         value=0.50, step=0.05, format="%.2f")
        retx   = c7.number_input("Retransmissions",  value=3.0,  step=1.0,  format="%.1f")
        lvl    = c8.selectbox("Load Level", [0,1,2,3,4], index=2)
        submitted = st.form_submit_button("Score window", use_container_width=True)
    if submitted:
        payload = {
            "sinr_dl_db": sinr, "throughput_mbps": thr, "delay_ms": delay, "jitter_ms": jit,
            "packet_loss_ratio": ploss, "prb_utilization": prb, "retransmissions": retx,
            "load_level": int(lvl),
        }
        resp, err = _rl_api_post("/anomaly/score", payload, timeout=10.0)
        if err or not resp:
            st.error(f"Scoring failed: {err}")
        else:
            ens = resp.get("ensemble_flag", 0)
            flagged = resp.get("flagged_by", [])
            badge = "🚨 ANOMALY" if ens else "✅ NORMAL"
            colr = "#ff4d6d" if ens else "#00e5a0"
            st.markdown(f"<div style='padding:10px;border-radius:8px;background:rgba(0,0,0,0.15);border:1px solid {colr};color:{colr};font-weight:700;'>Ensemble verdict: {badge} &nbsp;·&nbsp; flagged by {len(flagged)}/{len(resp.get('results',{}))} models</div>", unsafe_allow_html=True)
            res_df = pd.DataFrame([{"model": k, **(v if isinstance(v, dict) else {"error": str(v)})} for k, v in resp.get("results", {}).items()])
            st.dataframe(res_df, use_container_width=True, height=200)
    st.markdown("</div>", unsafe_allow_html=True)


# ══════════════════════════════════════════════════════════════════════════════
#  PAGE 3 — CAUSAL ANALYSIS
# ══════════════════════════════════════════════════════════════════════════════

CAUSAL_CHAINS = {
    "interference": [("&#128225;","RF Interference — SINR degrades"),("📉","↓ MCS Index — lower modulation"),("🔄","↑ Retransmissions — ARQ overhead"),("📦","↑ Packet Loss Ratio"),("📳","↑ Jitter — variable queuing"),("🚨","SLA Violation Triggered")],
    "congestion":   [("📶","↑ Load Level — high traffic demand"),("📊","↑ PRB Utilization — saturation"),("🚧","Queue Overflow — buffer full"),("📦","↑ Packet Loss Ratio"),("⏱️","↑ Delay — head-of-line blocking"),("🚨","SLA Breach — latency budget exceeded")],
    "mobility":     [("🏃","↑ Mobility Speed — UE moving fast"),("🔁","Handover Failure — beam tracking loss"),("&#128225;","↓ Throughput — signal interruption"),("📳","↑ Jitter — burst after reconnect"),("🌐","Service Disruption — session drop"),("🚨","SLA Breach — availability impact")],
    "combined/other":[("⚡","Load ↑ AND SINR ↓ simultaneously"),("🔗","Compounded Degradation — multi-factor"),("📦","Packet Loss spikes both channels"),("⏱️","Delay AND Jitter both elevated"),("💥","Cascading failure across UEs"),("🚨","Critical SLA Multi-Breach")],
}
RC_ACTIONS = {
    "interference":   ("frequency_reallocation + power_control","🔧"),
    "congestion":     ("load_balancing + traffic_throttling","⚖️"),
    "mobility":       ("handover_optimization + beam_tracking","&#128225;"),
    "combined/other": ("multi-action: reroute + prioritize","🛠️"),
}

def page_causal_root_cause(causal_df, cf_df):
    page_header("🔬","Causal Root Cause Explorer","DSO1.2 · Innovation 1 · DoWhy Causal AI · NS-3 Ground Truth")
    if causal_df is None or len(causal_df) == 0 or "root_cause" not in causal_df.columns:
        _empty_state("CAUSAL ROOT CAUSE", "root_cause_labels.csv missing or has no root_cause column.")
        return
    # Defensive copies — cf_df may be empty.
    if cf_df is None: cf_df = pd.DataFrame()
    # Merge causal with counterfactual data — handle v1 (timestamp) and v2 (no timestamp)
    cf_merge_cols = [c for c in ["ue_id","timestamp","original_packet_loss","counterfactual_if_static","reduction_pct"] if c in cf_df.columns]
    if "timestamp" in cf_df.columns and "timestamp" in causal_df.columns:
        merged = causal_df.merge(cf_df[cf_merge_cols], on="timestamp", how="left")
    elif "window_idx" in cf_df.columns and "window_idx" in causal_df.columns:
        cf_v2_cols = [c for c in ["window_idx","original_packet_loss","counterfactual_if_static","reduction_pct"] if c in cf_df.columns]
        merged = causal_df.merge(cf_df[cf_v2_cols], on="window_idx", how="left")
    else:
        merged = causal_df.copy()
        for c in ["original_packet_loss","counterfactual_if_static","reduction_pct"]:
            if c in cf_df.columns and len(cf_df) == len(causal_df):
                merged[c] = cf_df[c].values
            elif c not in merged.columns:
                merged[c] = 0.0

    col_sel, col_main = st.columns([1,3])

    with col_sel:
        st.markdown('<div class="glass-panel">', unsafe_allow_html=True)
        st.markdown('<div class="glass-panel-title">Select Root Cause</div>', unsafe_allow_html=True)
        rc_options = sorted(causal_df["root_cause"].unique().tolist())
        selected_rc = st.radio("rc", rc_options, label_visibility="collapsed")
        rc_data = merged[merged["root_cause"]==selected_rc]
        n_ev = len(rc_data)
        crit_pct = (rc_data["severity"]=="critical").mean()*100 if "severity" in rc_data.columns else 0.0
        avg_loss = rc_data["packet_loss_ratio"].mean()*100 if "packet_loss_ratio" in rc_data.columns else 0.0
        # For v2 data, use true_label as anomaly rate if severity is missing
        if "severity" not in rc_data.columns and "true_label" in rc_data.columns:
            crit_pct = rc_data["true_label"].mean()*100
        st.markdown(f"""
        <div style="margin-top:16px;display:flex;flex-direction:column;gap:8px;">
          <div style="background:var(--surface-2);border-radius:8px;padding:10px 12px;">
            <div style="font-family:'IBM Plex Mono';font-size:9px;color:var(--text-3);">EVENTS</div>
            <div style="font-family:'Syne';font-size:22px;font-weight:800;color:var(--cyan);">{n_ev:,}</div>
          </div>
          <div style="background:var(--surface-2);border-radius:8px;padding:10px 12px;">
            <div style="font-family:'IBM Plex Mono';font-size:9px;color:var(--text-3);">{"CRITICAL %" if "severity" in rc_data.columns else "ANOMALY %"}</div>
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
        avg_orig  = rc_data["packet_loss_ratio"].mean() if ("packet_loss_ratio" in rc_data.columns and len(rc_data)) else 0.5
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
        if "packet_loss_ratio" in causal_df.columns:
            st.markdown('<div class="glass-panel"><div class="glass-panel-title">Packet Loss Distribution by Root Cause</div>', unsafe_allow_html=True)
            fig = go.Figure()
            for i, rc in enumerate(causal_df["root_cause"].unique()):
                d = causal_df[causal_df["root_cause"]==rc]["packet_loss_ratio"]
                c = COLORS["seq"][i%len(COLORS["seq"])]
                fig.add_trace(go.Violin(y=d, name=rc, box_visible=True, meanline_visible=True, line_color=c, fillcolor=hex_to_rgba(c,0.13)))
            fig.update_layout(**PLOT_LAYOUT, height=280, showlegend=False, margin=dict(t=10,b=10,l=10,r=10), yaxis_title="Packet Loss")
            st.plotly_chart(fig, width='stretch')
            st.markdown("</div>", unsafe_allow_html=True)
        else:
            st.markdown('<div class="glass-panel"><div class="glass-panel-title">Events by Root Cause</div>', unsafe_allow_html=True)
            rc_counts = causal_df["root_cause"].value_counts().reset_index()
            rc_counts.columns = ["root_cause", "count"]
            fig = go.Figure(go.Bar(x=rc_counts["root_cause"], y=rc_counts["count"],
                marker_color=[COLORS["seq"][i%len(COLORS["seq"])] for i in range(len(rc_counts))]))
            fig.update_layout(**PLOT_LAYOUT, height=280, showlegend=False, margin=dict(t=10,b=10,l=10,r=10), yaxis_title="Count")
            st.plotly_chart(fig, width='stretch')
            st.markdown("</div>", unsafe_allow_html=True)

    with col_b:
        if "severity" in causal_df.columns:
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
        elif "true_label" in causal_df.columns:
            st.markdown('<div class="glass-panel"><div class="glass-panel-title">Anomaly Rate by Root Cause</div>', unsafe_allow_html=True)
            ar = causal_df.groupby("root_cause")["true_label"].mean().reset_index()
            ar.columns = ["root_cause", "anomaly_rate"]
            fig = go.Figure(go.Bar(x=ar["root_cause"], y=ar["anomaly_rate"]*100,
                marker_color=[COLORS["critical"] if r>0.5 else COLORS["amber"] for r in ar["anomaly_rate"]],
                text=ar["anomaly_rate"].apply(lambda x: f"{x*100:.1f}%"), textposition="outside"))
            fig.update_layout(**PLOT_LAYOUT, height=280, showlegend=False, margin=dict(t=10,b=10,l=10,r=10), yaxis_title="Anomaly Rate (%)")
            st.plotly_chart(fig, width='stretch')
            st.markdown("</div>", unsafe_allow_html=True)

    # ── Live DoWhy Counterfactual (M3 Innovation 1) ───────────────────────────
    st.markdown('<div class="glass-panel"><div class="glass-panel-title">🧪 Live Counterfactual Estimator — DoWhy Causal Model</div>', unsafe_allow_html=True)
    st.caption("Ask: *if the treatment were different, what would the outcome be?* — uses the live DoWhy model when available, with CSV fallback.")
    with st.form("_cf_form", clear_on_submit=False):
        c1, c2, c3 = st.columns(3)
        treat = c1.selectbox("Treatment",
            ["load_level", "sinr_dl_db", "packet_loss_ratio", "prb_utilization"],
            key="_cf_treat")
        obs   = c2.number_input("Observed value", value=3.0, step=0.5, format="%.3f", key="_cf_obs")
        cf    = c3.number_input("Counterfactual value", value=1.0, step=0.5, format="%.3f", key="_cf_cf")
        go_btn = st.form_submit_button("Estimate effect", use_container_width=True)
    if go_btn:
        payload = {"treatment": treat, "treatment_val": obs, "counterfactual": cf}
        resp, err = _rl_api_post("/causal/counterfactual", payload, timeout=15.0)
        if err or not resp:
            st.error(f"Counterfactual estimate failed: {err}")
        else:
            src = resp.get("source", "?")
            effect = resp.get("estimated_effect")
            c1, c2, c3 = st.columns(3)
            with c1: kpi_card("SOURCE",    src,                    "",  "ok", 100)
            with c2: kpi_card("OBSERVED",  f"{resp.get('observed', obs):.3f}", "",  "warn", 60)
            with c3: kpi_card("CF VALUE",  f"{resp.get('counterfactual', cf):.3f}", "",  "ok", 60)
            if effect is not None:
                st.success(f"Estimated causal effect: **{effect}**")
            elif "avg_reduction_pct" in resp:
                st.info(f"CSV fallback — average reduction after mitigation: **{resp['avg_reduction_pct']}%**")
            note = resp.get("note")
            if note:
                st.caption(note)
    st.markdown("</div>", unsafe_allow_html=True)


# ══════════════════════════════════════════════════════════════════════════════
#  PAGE — SLA RISK (DSO2.1 · M4)
# ══════════════════════════════════════════════════════════════════════════════

def page_sla_risk(sla_df):
    page_header("⚠️", "SLA Breach Risk", "DSO2.1 · XGBoost Early Warning · Member 4 (NS-3 LENA)")

    if sla_df is None or sla_df.empty:
        st.warning("sla_breach_predictions.csv not found in data folder. Run Member 4's DSO2.1 notebook first and copy the output.")
        return
    required = {"is_high_risk_pred","risk_proba","load_level","ue_id","timestamp"}
    missing  = required - set(sla_df.columns)
    if missing:
        _empty_state("SLA RISK", f"Missing columns in sla_breach_predictions.csv: {', '.join(sorted(missing))}")
        return

    high_risk = sla_df[sla_df["is_high_risk_pred"] == 1]
    total     = len(sla_df)
    n_risk    = len(high_risk)
    risk_rate = n_risk / total * 100 if total > 0 else 0
    avg_prob_raw = high_risk["risk_proba"].mean() if n_risk > 0 else None
    avg_prob  = round(avg_prob_raw * 100, 1) if avg_prob_raw is not None and pd.notna(avg_prob_raw) else 0

    c1, c2, c3, c4 = st.columns(4)
    with c1:
        kpi_card("TOTAL EVENTS",  f"{total:,}",       "",   "ok",   100)
    with c2:
        kpi_card("HIGH RISK",     f"{n_risk:,}",      "",   "crit", min(100, risk_rate * 2))
    with c3:
        kpi_card("RISK RATE",     f"{risk_rate:.1f}", "%",  "warn", risk_rate)
    with c4:
        kpi_card("AVG RISK PROB", f"{avg_prob}",      "%",  "crit" if avg_prob > 80 else "warn", avg_prob)

    st.markdown("<br>", unsafe_allow_html=True)
    col_l, col_r = st.columns(2)

    with col_l:
        st.markdown('<div class="glass-panel"><div class="glass-panel-title">Risk Probability Distribution</div>', unsafe_allow_html=True)
        if n_risk == 0:
            st.markdown("<div style='padding:30px;text-align:center;color:var(--text-3);font-size:11px;'>no high-risk events to plot</div>", unsafe_allow_html=True)
        else:
            fig = go.Figure()
            fig.add_trace(go.Histogram(
                x=high_risk["risk_proba"], nbinsx=30,
                marker_color=COLORS["critical"], opacity=0.8, name="High-Risk Events",
            ))
            fig.update_layout(**PLOT_LAYOUT, height=280,
                              xaxis_title="Risk Probability", yaxis_title="Count",
                              margin=dict(t=10, b=10, l=10, r=10))
            st.plotly_chart(fig, use_container_width=True)
        st.markdown("</div>", unsafe_allow_html=True)

    with col_r:
        st.markdown('<div class="glass-panel"><div class="glass-panel-title">High-Risk Events by Load Level</div>', unsafe_allow_html=True)
        if n_risk == 0:
            st.markdown("<div style='padding:30px;text-align:center;color:var(--text-3);font-size:11px;'>no high-risk events to break down</div>", unsafe_allow_html=True)
        else:
            by_load = high_risk.groupby("load_level").agg(
                count=("risk_proba", "count"),
                avg_prob=("risk_proba", "mean"),
            ).reset_index()
            bar_cols = [COLORS["critical"], COLORS["amber"], COLORS["cyan"], COLORS["emerald"], COLORS["violet"]]
            fig = go.Figure(go.Bar(
                x=by_load["load_level"].astype(str),
                y=by_load["count"],
                marker_color=bar_cols[:len(by_load)],
                text=by_load["avg_prob"].apply(lambda x: f"{x*100:.0f}%"),
                textposition="outside",
            ))
            fig.update_layout(**PLOT_LAYOUT, height=280,
                              xaxis_title="Load Level", yaxis_title="High-Risk Events",
                              margin=dict(t=20, b=10, l=10, r=10))
            st.plotly_chart(fig, use_container_width=True)
        st.markdown("</div>", unsafe_allow_html=True)

    st.markdown('<div class="glass-panel"><div class="glass-panel-title">Risk Probability Over Time — Top-4 High-Risk UEs</div>', unsafe_allow_html=True)
    if n_risk == 0:
        st.markdown("<div style='padding:30px;text-align:center;color:var(--text-3);font-size:11px;'>no high-risk events</div>", unsafe_allow_html=True)
    else:
        sample_ues  = high_risk["ue_id"].value_counts().head(4).index.tolist()
        colors_seq  = [COLORS["critical"], COLORS["amber"], COLORS["cyan"], COLORS["emerald"]]
        fig = go.Figure()
        for i, ue in enumerate(sample_ues):
            ue_data = sla_df[sla_df["ue_id"] == ue].sort_values("timestamp")
            fig.add_trace(go.Scatter(
                x=ue_data["timestamp"], y=ue_data["risk_proba"],
                name=f"UE {ue}",
                line=dict(color=colors_seq[i % len(colors_seq)], width=2),
                mode="lines",
            ))
        fig.add_hline(y=0.5, line_dash="dash", line_color="gray",
                      annotation_text="Decision Threshold")
        fig.update_layout(**PLOT_LAYOUT, height=300,
                          xaxis_title="Timestamp", yaxis_title="Risk Probability",
                          margin=dict(t=10, b=10, l=10, r=10))
        st.plotly_chart(fig, use_container_width=True)
    st.markdown("</div>", unsafe_allow_html=True)


# ══════════════════════════════════════════════════════════════════════════════
#  PAGE — SEVERITY TRIAGE (DSO2.3 · M5)
# ══════════════════════════════════════════════════════════════════════════════

def page_severity_triage(severity_df):
    page_header("🎯", "Severity Triage", "DSO2.3 · Random Forest + XGBoost · Member 5")

    if severity_df is None or severity_df.empty:
        st.warning("severity_triage.csv not found. Run Member 5's DSO2.3 notebook first and copy outputs/severity_triage.csv to data/.")
        return
    if "severity_final" not in severity_df.columns:
        _empty_state("SEVERITY TRIAGE", "severity_triage.csv is missing the severity_final column.")
        return

    sev_colors = {"LOW": COLORS["normal"], "MEDIUM": COLORS["amber"], "HIGH": COLORS["critical"]}
    dist  = severity_df["severity_final"].value_counts()
    total = len(severity_df)

    c1, c2, c3 = st.columns(3)
    for col, sev in zip([c1, c2, c3], ["LOW", "MEDIUM", "HIGH"]):
        with col:
            n      = dist.get(sev, 0)
            pct    = n / total * 100 if total > 0 else 0
            status = "ok" if sev == "LOW" else ("warn" if sev == "MEDIUM" else "crit")
            kpi_card(f"{sev} SEVERITY", f"{n:,}", f"({pct:.0f}%)", status, pct)

    st.markdown("<br>", unsafe_allow_html=True)
    col_l, col_r = st.columns(2)

    with col_l:
        st.markdown('<div class="glass-panel"><div class="glass-panel-title">Severity Distribution</div>', unsafe_allow_html=True)
        fig = go.Figure(go.Pie(
            labels=dist.index.tolist(), values=dist.values.tolist(), hole=0.55,
            marker=dict(colors=[sev_colors.get(s, "#888") for s in dist.index],
                        line=dict(color="#080c12", width=3)),
            textinfo="label+percent",
        ))
        fig.update_layout(**PLOT_LAYOUT, height=300, showlegend=False,
                          margin=dict(t=10, b=10, l=10, r=10))
        st.plotly_chart(fig, use_container_width=True)
        st.markdown("</div>", unsafe_allow_html=True)

    with col_r:
        if "traffic_type" in severity_df.columns:
            st.markdown('<div class="glass-panel"><div class="glass-panel-title">Severity by Traffic Type</div>', unsafe_allow_html=True)
            cross = pd.crosstab(severity_df["traffic_type"], severity_df["severity_final"])
            for sev in ["LOW", "MEDIUM", "HIGH"]:
                if sev not in cross.columns:
                    cross[sev] = 0
            fig = go.Figure()
            for sev in ["LOW", "MEDIUM", "HIGH"]:
                fig.add_trace(go.Bar(name=sev, x=cross.index.tolist(),
                                     y=cross[sev].tolist(), marker_color=sev_colors[sev]))
            fig.update_layout(**PLOT_LAYOUT, barmode="stack", height=300,
                              xaxis_title="Traffic Type", yaxis_title="Count",
                              margin=dict(t=10, b=10, l=10, r=10))
            st.plotly_chart(fig, use_container_width=True)
            st.markdown("</div>", unsafe_allow_html=True)

    st.markdown('<div class="glass-panel"><div class="glass-panel-title">🔴 Top HIGH Severity Incidents</div>', unsafe_allow_html=True)
    high_sev  = severity_df[severity_df["severity_final"] == "HIGH"].head(20)
    show_cols = [c for c in ["ue_id", "timestamp", "traffic_type", "delay_ms",
                              "packet_loss_ratio", "jitter_ms", "throughput_mbps",
                              "severity_final"] if c in high_sev.columns]
    if not high_sev.empty:
        st.dataframe(high_sev[show_cols].reset_index(drop=True),
                     use_container_width=True, height=300)
    st.markdown("</div>", unsafe_allow_html=True)


# ══════════════════════════════════════════════════════════════════════════════
#  PAGE — BENCHMARKING (DSO1.3 · M5)
# ══════════════════════════════════════════════════════════════════════════════

def page_benchmarking(bench_df):
    page_header("📈", "Network Benchmarking", "DSO1.3 · 9-Model Comparison · QoS by Load & Mobility")

    # ── 9-Model Comparison Section ─────────────────────────────────────────
    v2_path = DATA_DIR / "anomaly_scores_v2.csv"
    if v2_path.exists():
        v2_df = pd.read_csv(v2_path)
        score_cols = [c for c in v2_df.columns if c.startswith("score_")]
        if score_cols:
            mn = {"score_isolation_forest":"Isolation Forest","score_copod":"COPOD",
                  "score_one-class_svm":"One-Class SVM","score_random_forest_supervised":"Random Forest",
                  "score_xgboost_supervised":"XGBoost","score_gradient_boosting_supervised":"Gradient Boosting",
                  "score_lstm_autoencoder":"LSTM Autoencoder","score_anomaly_transformer":"Anomaly Transformer",
                  "score_tranad":"TranAD"}
            mc = {"score_isolation_forest":"Classical","score_copod":"Classical","score_one-class_svm":"Classical",
                  "score_random_forest_supervised":"Supervised","score_xgboost_supervised":"Supervised",
                  "score_gradient_boosting_supervised":"Supervised","score_lstm_autoencoder":"Deep Learning",
                  "score_anomaly_transformer":"Deep Learning","score_tranad":"Deep Learning"}
            cc = {"Classical":COLORS["cyan"],"Supervised":COLORS["emerald"],"Deep Learning":COLORS["violet"]}
            stats = []
            for col in score_cols:
                vals = pd.to_numeric(v2_df[col], errors="coerce").dropna()
                if vals.empty:
                    stats.append({"model":mn.get(col,col),"col":col,"cat":mc.get(col,"Other"),
                        "mean":0.0,"std":0.0,"med":0.0,"det":0.0})
                    continue
                stats.append({"model":mn.get(col,col),"col":col,"cat":mc.get(col,"Other"),
                    "mean":float(vals.mean()),"std":float(vals.std() if len(vals) > 1 else 0.0),
                    "med":float(vals.median()),"det":float((vals>0.5).mean()*100)})
            sdf = pd.DataFrame(stats).sort_values("mean",ascending=False)
            st.markdown('<div class="glass-panel"><div class="glass-panel-title">🎯 9-Model Anomaly Detection Comparison</div>', unsafe_allow_html=True)
            lg = "<div style='display:flex;gap:20px;margin-bottom:12px;'>"
            for cat,color in cc.items():
                lg += f"<span style='display:inline-flex;align-items:center;gap:6px;'><span style='width:10px;height:10px;border-radius:50%;background:{color};box-shadow:0 0 6px {color};'></span><span style='font-family:IBM Plex Mono;font-size:10px;color:var(--text-2);'>{cat}</span></span>"
            st.markdown(lg+"</div>", unsafe_allow_html=True)
            c1,c2 = st.columns(2)
            with c1:
                bc = [cc.get(r["cat"],COLORS["cyan"]) for _,r in sdf.iterrows()]
                fig = go.Figure(go.Bar(x=sdf["model"],y=sdf["mean"],marker_color=[hex_to_rgba(c,0.75) for c in bc],
                    marker_line=dict(color=bc,width=1.5),text=sdf["mean"].round(4),textposition="outside",
                    textfont=dict(size=9,family="IBM Plex Mono"),
                    error_y=dict(type="data",array=sdf["std"].round(4),visible=True,color="rgba(255,255,255,0.3)",thickness=1.5)))
                fig.update_layout(**PLOT_LAYOUT,height=320,title=dict(text="Average Anomaly Score by Model",font=dict(size=12,color="#e8f0fe")),yaxis_title="Mean Score",margin=dict(t=40,b=80,l=10,r=10))
                fig.update_xaxes(tickangle=-35, tickfont_size=8)
                st.plotly_chart(fig, use_container_width=True)
            with c2:
                fig2 = go.Figure(go.Bar(x=sdf["model"],y=sdf["det"],marker_color=[hex_to_rgba(c,0.75) for c in bc],
                    marker_line=dict(color=bc,width=1.5),text=sdf["det"].apply(lambda x:f"{x:.1f}%"),textposition="outside",
                    textfont=dict(size=9,family="IBM Plex Mono")))
                fig2.update_layout(**PLOT_LAYOUT,height=320,title=dict(text="Anomaly Detection Rate (score > 0.5)",font=dict(size=12,color="#e8f0fe")),yaxis_title="Detection Rate (%)",margin=dict(t=40,b=80,l=10,r=10))
                fig2.update_xaxes(tickangle=-35, tickfont_size=8)
                st.plotly_chart(fig2, use_container_width=True)
            fig3 = go.Figure()
            for _,r in sdf.iterrows():
                c = cc.get(r["cat"],COLORS["cyan"])
                fig3.add_trace(go.Violin(y=v2_df[r["col"]].dropna(),name=r["model"],box_visible=True,meanline_visible=True,
                    line_color=c,fillcolor=hex_to_rgba(c,0.10),points=False))
            fig3.update_layout(**PLOT_LAYOUT,height=340,title=dict(text="Score Distribution per Model",font=dict(size=12,color="#e8f0fe")),yaxis_title="Anomaly Score",showlegend=False,margin=dict(t=40,b=80,l=10,r=10))
            fig3.update_xaxes(tickangle=-35, tickfont_size=8)
            st.plotly_chart(fig3, use_container_width=True)
            c3,c4 = st.columns(2)
            with c3:
                corr = v2_df[score_cols].corr()
                labs = [mn.get(c,c) for c in score_cols]
                fig4 = go.Figure(go.Heatmap(z=corr.values,x=labs,y=labs,
                    colorscale=[[0,"#0d1420"],[0.5,"rgba(0,212,255,0.3)"],[1,"#00d4ff"]],
                    zmin=0,zmax=1,text=corr.values.round(2),texttemplate="%{text}",textfont=dict(size=7),
                    showscale=True,colorbar=dict(tickfont=dict(size=8),thickness=8)))
                fig4.update_layout(**PLOT_LAYOUT,height=380,title=dict(text="Model Correlation Matrix",font=dict(size=12,color="#e8f0fe")),margin=dict(t=40,b=80,l=80,r=10))
                fig4.update_xaxes(tickangle=-40, tickfont_size=7)
                fig4.update_yaxes(tickfont_size=7)
                st.plotly_chart(fig4, use_container_width=True)
            with c4:
                cst = sdf.groupby("cat").agg(ms=("mean","mean"),dr=("det","mean")).reset_index()
                cats = cst["cat"].tolist()
                fig5 = go.Figure()
                fig5.add_trace(go.Scatterpolar(r=cst["ms"].tolist()+[cst["ms"].iloc[0]],theta=cats+[cats[0]],
                    fill="toself",fillcolor=hex_to_rgba(COLORS["cyan"],0.1),line=dict(color=COLORS["cyan"],width=2),name="Avg Score"))
                fig5.add_trace(go.Scatterpolar(r=(cst["dr"]/100).tolist()+[(cst["dr"].iloc[0]/100)],theta=cats+[cats[0]],
                    fill="toself",fillcolor=hex_to_rgba(COLORS["emerald"],0.1),line=dict(color=COLORS["emerald"],width=2),name="Detection Rate"))
                fig5.update_layout(polar=dict(bgcolor="rgba(0,0,0,0)",
                    radialaxis=dict(visible=True,range=[0,1],tickfont=dict(size=8,color="#4a5568"),gridcolor="rgba(0,212,255,0.08)"),
                    angularaxis=dict(tickfont=dict(size=9,color="#e8f0fe"),gridcolor="rgba(0,212,255,0.08)")),
                    paper_bgcolor="rgba(0,0,0,0)",plot_bgcolor="rgba(0,0,0,0)",font=dict(color="#e8f0fe"),height=380,
                    title=dict(text="Category Comparison (Radar)",font=dict(size=12,color="#e8f0fe")),
                    legend=dict(font=dict(size=9),bgcolor="rgba(0,0,0,0)"),margin=dict(t=50,b=20,l=40,r=40))
                st.plotly_chart(fig5, use_container_width=True)
            disp = sdf[["model","cat","mean","std","med","det"]].copy()
            disp.columns = ["Model","Category","Avg Score","Std Dev","Median","Detection %"]
            st.dataframe(disp.reset_index(drop=True).round(4), use_container_width=True, height=280)
            st.markdown("</div>", unsafe_allow_html=True)
            st.markdown("<br>", unsafe_allow_html=True)

    # ── QoS Benchmark Section ──────────────────────────────────────────────
    if bench_df.empty:
        if not v2_path.exists():
            st.warning("No benchmark data found.")
        return

    perf_colors = {"GOOD": COLORS["normal"], "MEDIUM": COLORS["amber"], "POOR": COLORS["critical"]}
    qos_col     = "avg_qos"  if "avg_qos"  in bench_df.columns else None
    cat_col     = "category" if "category" in bench_df.columns else None
    val_col     = "val"      if "val"      in bench_df.columns else None
    class_col   = "class"    if "class"    in bench_df.columns else None

    if qos_col and cat_col and val_col:
        load_bench = bench_df[bench_df[cat_col] == "load_level"].copy()
        mob_bench  = bench_df[bench_df[cat_col] == "mobility_speed"].copy()

        col_l, col_r = st.columns(2)

        with col_l:
            st.markdown('<div class="glass-panel"><div class="glass-panel-title">Avg QoS Score by Load Level</div>', unsafe_allow_html=True)
            if not load_bench.empty:
                b_colors = [perf_colors.get(c, COLORS["cyan"]) for c in (load_bench[class_col] if class_col in load_bench else ["MEDIUM"] * len(load_bench))]
                fig = go.Figure(go.Bar(
                    x=load_bench[val_col].astype(str), y=load_bench[qos_col],
                    marker_color=b_colors,
                    text=load_bench[qos_col].round(3), textposition="outside",
                ))
                fig.update_layout(**PLOT_LAYOUT, height=300,
                                  xaxis_title="Load Level", yaxis_title="Avg QoS Score (0–1)",
                                  yaxis_range=[0, 1.1], margin=dict(t=20, b=10, l=10, r=10))
                st.plotly_chart(fig, use_container_width=True)
            st.markdown("</div>", unsafe_allow_html=True)

        with col_r:
            st.markdown('<div class="glass-panel"><div class="glass-panel-title">Avg QoS Score by Mobility Speed</div>', unsafe_allow_html=True)
            if not mob_bench.empty:
                b_colors = [perf_colors.get(c, COLORS["cyan"]) for c in (mob_bench[class_col] if class_col in mob_bench else ["MEDIUM"] * len(mob_bench))]
                fig = go.Figure(go.Bar(
                    x=mob_bench[val_col].astype(str), y=mob_bench[qos_col],
                    marker_color=b_colors,
                    text=mob_bench[qos_col].round(3), textposition="outside",
                ))
                fig.update_layout(**PLOT_LAYOUT, height=300,
                                  xaxis_title="Mobility Speed", yaxis_title="Avg QoS Score (0–1)",
                                  yaxis_range=[0, 1.1], margin=dict(t=20, b=10, l=10, r=10))
                st.plotly_chart(fig, use_container_width=True)
            st.markdown("</div>", unsafe_allow_html=True)

    st.markdown('<div class="glass-panel"><div class="glass-panel-title">Full Benchmark Report</div>', unsafe_allow_html=True)
    st.dataframe(bench_df.reset_index(drop=True), use_container_width=True, height=350)
    st.markdown("</div>", unsafe_allow_html=True)


# ══════════════════════════════════════════════════════════════════════════════
#  PAGE — QoS FORECAST (DSO1.1 — M2: XGBoost + Prophet)
# ══════════════════════════════════════════════════════════════════════════════

def page_qos_forecast():
    page_header("🔮","QoS Forecast","DSO1.1 · XGBoost regression + Prophet confidence intervals (M2)")

    # ── Window controls ──
    cc1, cc2, cc3 = st.columns([1, 1, 2])
    win_size = cc1.selectbox("Window", [1000, 2000, 5000, 20000, 100000],
                             index=2, key="_qos_win",
                             help="Latest N rows of static + live predictions")
    src = cc2.selectbox("Source", ["auto", "static", "live"], index=0, key="_qos_src",
                        help="auto = concat static CSV + live MLOps output")

    st.markdown('<div class="glass-panel"><div class="glass-panel-title">Jitter Forecast — XGBoost Predictions vs Actual</div>', unsafe_allow_html=True)

    pred, pred_err = _rl_api_get(f"/qos/predictions?limit={int(win_size)}&source={src}", timeout=10.0)
    if pred_err or not pred or "data" not in pred:
        st.warning(f"Predictions unavailable: {pred_err or 'no data'}")
        st.markdown("</div>", unsafe_allow_html=True)
    else:
        sources = pred.get("sources") or []
        if sources:
            cc3.markdown(
                f"<div style='padding-top:8px;font-family:IBM Plex Mono;font-size:10px;"
                f"color:var(--text-3);'>📡 sources: <b style='color:var(--cyan);'>"
                f"{' + '.join(sources)}</b> · <b>{pred.get('count',0):,}</b> rows</div>",
                unsafe_allow_html=True)
        rows = pred.get("data", [])
        if not rows:
            st.info("No prediction rows loaded.")
        else:
            pdf = pd.DataFrame(rows)
            if "y_true" in pdf.columns and "y_pred" in pdf.columns:
                # Coerce to numeric — API may return strings
                pdf["y_true"] = pd.to_numeric(pdf["y_true"], errors="coerce")
                pdf["y_pred"] = pd.to_numeric(pdf["y_pred"], errors="coerce")
                pdf = pdf.dropna(subset=["y_true","y_pred"])
                if pdf.empty:
                    st.info("Predictions present but all rows non-numeric.")
                    st.markdown("</div>", unsafe_allow_html=True)
                    return
                mae  = float((pdf["y_true"] - pdf["y_pred"]).abs().mean())
                rmse = float(((pdf["y_true"] - pdf["y_pred"])**2).mean() ** 0.5)
                denom = pdf["y_true"].replace(0, np.nan).abs()
                mape_val  = ((pdf["y_true"] - pdf["y_pred"]).abs() / denom).mean() * 100
                mape_str  = "—" if pd.isna(mape_val) else f"{float(mape_val):.1f}"
                mape_pct  = 0.0 if pd.isna(mape_val) else min(float(mape_val), 100)

                c1, c2, c3, c4 = st.columns(4)
                with c1: kpi_card("ROWS",  f"{len(pdf):,}", "",    "ok", 100)
                with c2: kpi_card("MAE",   f"{mae:.3f}",    "ms",  "ok" if mae < 2 else "warn", min(mae*20,100))
                with c3: kpi_card("RMSE",  f"{rmse:.3f}",   "ms",  "ok" if rmse < 3 else "warn", min(rmse*15,100))
                with c4: kpi_card("MAPE",  mape_str,        "%",   "ok" if (mape_str != "—" and float(mape_str) < 20) else "warn", mape_pct)

                sample = pdf.tail(min(500, len(pdf))).reset_index(drop=True)
                fig = go.Figure()
                fig.add_trace(go.Scatter(y=sample["y_true"], mode="lines", name="Actual",
                                         line=dict(color=COLORS["cyan"], width=1.8)))
                fig.add_trace(go.Scatter(y=sample["y_pred"], mode="lines", name="Predicted",
                                         line=dict(color=COLORS["amber"], width=1.8, dash="dot")))
                fig.update_layout(**PLOT_LAYOUT, height=320, margin=dict(t=20,b=10,l=10,r=10),
                                  legend=dict(orientation="h", y=1.08))
                st.plotly_chart(fig, use_container_width=True)

                fig2 = go.Figure()
                fig2.add_trace(go.Scatter(x=pdf["y_true"], y=pdf["y_pred"], mode="markers",
                                          marker=dict(size=4, color=COLORS["cyan"], opacity=0.5),
                                          name="Pred vs Actual"))
                lo = float(pdf[["y_true","y_pred"]].min().min())
                hi = float(pdf[["y_true","y_pred"]].max().max())
                fig2.add_trace(go.Scatter(x=[lo,hi], y=[lo,hi], mode="lines",
                                          line=dict(color=COLORS["emerald"], dash="dash"),
                                          name="y = x"))
                fig2.update_layout(**PLOT_LAYOUT, height=280, margin=dict(t=20,b=10,l=10,r=10),
                                   legend=dict(orientation="h", y=1.1))
                st.plotly_chart(fig2, use_container_width=True)
            else:
                st.info("predicted_jitter.csv missing y_true/y_pred columns.")
        st.markdown("</div>", unsafe_allow_html=True)

    st.markdown('<div class="glass-panel"><div class="glass-panel-title">Prophet Forecast — Per-UE / Metric with Confidence Interval</div>', unsafe_allow_html=True)

    fc, fc_err = _rl_api_get("/qos/forecast", timeout=8.0)
    if fc_err or not fc or "data" not in fc:
        st.warning(f"Forecast unavailable: {fc_err or 'no data'}")
    else:
        fc_rows = fc.get("data", [])
        if not fc_rows:
            st.info("forecast_ci.csv is empty.")
        else:
            fdf = pd.DataFrame(fc_rows)
            ue_opts = sorted(fdf["ue_id"].dropna().unique().tolist()) if "ue_id" in fdf.columns else []
            met_opts = sorted(fdf["metric"].dropna().unique().tolist()) if "metric" in fdf.columns else []
            c1, c2 = st.columns(2)
            ue_sel  = c1.selectbox("UE",     ue_opts  or ["—"], key="_fc_ue")
            met_sel = c2.selectbox("Metric", met_opts or ["—"], key="_fc_met")

            sub = fdf
            if "ue_id" in fdf.columns and ue_opts:
                sub = sub[sub["ue_id"] == ue_sel]
            if "metric" in fdf.columns and met_opts:
                sub = sub[sub["metric"] == met_sel]
            if "ds" in sub.columns:
                sub = sub.sort_values("ds")

            if len(sub) == 0:
                st.info("No forecast rows match the current filter.")
            else:
                fig = go.Figure()
                x = sub["ds"] if "ds" in sub.columns else list(range(len(sub)))
                if "yhat_upper" in sub.columns and "yhat_lower" in sub.columns:
                    fig.add_trace(go.Scatter(x=x, y=sub["yhat_upper"], mode="lines",
                                             line=dict(color="rgba(0,212,255,0)"),
                                             showlegend=False))
                    fig.add_trace(go.Scatter(x=x, y=sub["yhat_lower"], mode="lines",
                                             fill="tonexty",
                                             fillcolor=hex_to_rgba(COLORS["cyan"], 0.15),
                                             line=dict(color="rgba(0,212,255,0)"),
                                             name="95% CI"))
                if "yhat" in sub.columns:
                    fig.add_trace(go.Scatter(x=x, y=sub["yhat"], mode="lines",
                                             line=dict(color=COLORS["cyan"], width=2.2),
                                             name="yhat"))
                fig.update_layout(**PLOT_LAYOUT, height=360, margin=dict(t=20,b=10,l=10,r=10),
                                  legend=dict(orientation="h", y=1.08))
                st.plotly_chart(fig, use_container_width=True)

                st.dataframe(sub.head(200).reset_index(drop=True),
                             use_container_width=True, height=280)
    st.markdown("</div>", unsafe_allow_html=True)


# ══════════════════════════════════════════════════════════════════════════════
#  PAGE 4 — RL ACTIONS
# ══════════════════════════════════════════════════════════════════════════════

def page_rl_actions(anomaly_df, causal_df):
    page_header("🤖","RL Agent Actions","DSO3.1 · PPO Policy · Digital Twin Environment (NS-3)")

    # anomaly_df (v1, ns3.xls) and causal_df (v2, root_cause_labels.csv) have no
    # shared join key — so compute action stats directly from causal_df where
    # `recommended_action` actually lives. Restrict to anomalous windows when
    # `true_label` is present.
    if "recommended_action" in causal_df.columns:
        src = causal_df
        if "true_label" in src.columns and (src["true_label"] == 1).any():
            src = src[src["true_label"] == 1]
        action_stats = (
            src.groupby("recommended_action")
               .size()
               .reset_index(name="events")
               .sort_values("events", ascending=False)
        )
    else:
        action_stats = pd.DataFrame({"recommended_action": [], "events": []})

    # Historical action-dispatch summary (kept as a compact strip — the Live Agent
    # Advisor below supersedes the per-action detail cards).
    _n_actions  = int(len(action_stats))
    _top_action = action_stats.iloc[0]["recommended_action"] if _n_actions else "—"
    _top_events = int(action_stats.iloc[0]["events"]) if _n_actions else 0
    c1, c2, c3 = st.columns(3)
    with c1: kpi_card("ACTION CATEGORIES", f"{_n_actions}", "", "ok", 100)
    with c2: kpi_card("TOP ACTION",         _top_action,      "", "ok", 100)
    with c3: kpi_card("DISPATCHED",         f"{_top_events:,}", "", "ok", 100)

    col_l, col_r = st.columns(2)
    with col_l:
        st.markdown('<div class="glass-panel"><div class="glass-panel-title">PPO Reward vs Rule-Based Baseline (M5 Real Data)</div>', unsafe_allow_html=True)
        rl_df = load_rl_df()
        if not rl_df.empty and "policy" in rl_df.columns:
            ppo_data  = rl_df[rl_df["policy"] == "PPO_RL"].sort_values("episode")
            rule_data = rl_df[rl_df["policy"] == "Rule-Based"].sort_values("episode")
            fig = go.Figure()
            fig.add_trace(go.Scatter(
                x=rule_data["episode"], y=rule_data["total_reward"],
                name="Rule-Based", line=dict(color="#4a5568", width=1.5, dash="dot"),
                fill="tozeroy", fillcolor="rgba(74,85,104,0.08)",
            ))
            fig.add_trace(go.Scatter(
                x=ppo_data["episode"], y=ppo_data["total_reward"],
                name="PPO RL Agent", line=dict(color=COLORS["emerald"], width=2.5),
                fill="tozeroy", fillcolor=hex_to_rgba(COLORS["emerald"], 0.09),
            ))
        else:
            # Fallback to simulated if file not yet available
            np.random.seed(42)
            steps    = np.arange(1, 101)
            baseline = np.cumsum(np.random.normal(-0.15, 0.4, 100))
            ppo_sim  = np.cumsum(np.random.normal(0.35, 0.25, 100))
            fig = go.Figure()
            fig.add_trace(go.Scatter(x=steps, y=baseline, name="No-Action Baseline",
                                     line=dict(color="#4a5568", width=1.5, dash="dot"),
                                     fill="tozeroy", fillcolor="rgba(74,85,104,0.08)"))
            fig.add_trace(go.Scatter(x=steps, y=ppo_sim, name="PPO RL Agent (simulated)",
                                     line=dict(color=COLORS["emerald"], width=2.5),
                                     fill="tozeroy", fillcolor=hex_to_rgba(COLORS["emerald"], 0.09)))
        fig.update_layout(**PLOT_LAYOUT, height=280, margin=dict(t=10,b=10,l=10,r=10),
                          xaxis_title="Simulation Step", yaxis_title="Cumulative Reward")
        st.plotly_chart(fig, width='stretch')
        st.markdown("</div>", unsafe_allow_html=True)

    with col_r:
        st.markdown('<div class="glass-panel"><div class="glass-panel-title">Action Distribution — Anomalous Events</div>', unsafe_allow_html=True)
        _action_colors = [COLORS["cyan"], COLORS["emerald"], COLORS["amber"], COLORS["violet"]]
        fig = go.Figure(go.Pie(
            labels=action_stats["recommended_action"], values=action_stats["events"], hole=0.6,
            marker=dict(colors=[hex_to_rgba(c,0.8) for c in _action_colors[:len(action_stats)]],
                        line=dict(color="#080c12",width=3)),
            textinfo="none",
            hovertemplate="<b>%{label}</b><br>%{value:,} events<extra></extra>",
        ))
        fig.update_layout(**PLOT_LAYOUT, height=280, showlegend=True,
                          legend=dict(orientation="h",yanchor="bottom",y=-0.3,xanchor="center",x=0.5),
                          margin=dict(t=10,b=60,l=10,r=10))
        st.plotly_chart(fig, width='stretch')
        st.markdown("</div>", unsafe_allow_html=True)

    # ── Live PPO Inference (DSO3.1) ──────────────────────────────────────────
    st.markdown("---")
    st.markdown(
        '<div style="font-family:\'Syne\';font-size:18px;font-weight:700;color:var(--text-1);'
        'margin:18px 0 12px;">⚡ Live PPO Agent · Digital Twin Inference</div>',
        unsafe_allow_html=True,
    )

    health, health_err = _rl_api_get("/health", timeout=3.0)
    rl_ready = bool(health and health.get("rl_ready"))
    if not rl_ready:
        st.info(
            f"🛰️ Live PPO inference is offline. "
            f"Check that the API is running and that `stable-baselines3` + `torch` are installed. "
            f"({health_err or (health and 'rl_ready=false')})"
        )
        return

    # ── Panel: Live Advisor (single-UE recommendation) ──
    adv_l, adv_r = st.columns([1, 1])
    with adv_l:
        st.markdown('<div class="glass-panel"><div class="glass-panel-title">🎯 Live Agent Advisor</div>', unsafe_allow_html=True)

        # anomaly_df carries the raw KPI columns the PPO advisor needs.
        # Prefer anomalous rows when the flag exists, else the full frame.
        if "if_anomaly" in anomaly_df.columns and (anomaly_df["if_anomaly"] == 1).any():
            obs_src = anomaly_df[anomaly_df["if_anomaly"] == 1]
        else:
            obs_src = anomaly_df
        needed = ["sinr_dl_db","throughput_mbps","delay_ms","jitter_ms",
                  "packet_loss_ratio","prb_utilization","retransmissions"]
        have = all(c in obs_src.columns for c in needed)

        if not have:
            st.caption("Anomaly table is missing raw KPI columns — advisor needs v1-format data.")
        else:
            ue_options = sorted(obs_src["ue_id"].dropna().unique().tolist()) if "ue_id" in obs_src.columns else []
            if ue_options:
                sel_ue = st.selectbox("UE to advise", ue_options, key="rl_adv_ue")
                ue_rows = obs_src[obs_src["ue_id"] == sel_ue]
                if ue_rows.empty:
                    row = obs_src.iloc[0]
                elif "if_score" in ue_rows.columns:
                    scores = pd.to_numeric(ue_rows["if_score"], errors="coerce")
                    if scores.notna().any():
                        row = ue_rows.iloc[int(scores.fillna(-np.inf).argmax())]
                    else:
                        row = ue_rows.iloc[0]
                else:
                    row = ue_rows.iloc[0]
            elif len(obs_src) > 0:
                row = obs_src.iloc[0]
            else:
                row = None

            if row is None:
                st.caption("No KPI rows available to advise on.")
                st.markdown("</div>", unsafe_allow_html=True)
                return

            payload = {
                "sinr_dl_db":        float(row["sinr_dl_db"]),
                "throughput_mbps":   float(row["throughput_mbps"]),
                "delay_ms":          float(row["delay_ms"]),
                "jitter_ms":         float(row["jitter_ms"]),
                "packet_loss_ratio": float(row["packet_loss_ratio"]),
                "prb_utilization":   float(row["prb_utilization"]),
                "retransmissions":   float(row.get("retransmissions", 0)),
                "sla_risk_score":    float(row.get("if_score", 0.0)) * 10.0,
            }

            if st.button("Get recommendation", key="rl_adv_btn", type="primary"):
                with st.spinner("Querying PPO policy…"):
                    resp, err = _rl_api_post("/rl/predict", payload, timeout=10.0)
                if err:
                    st.error(f"Inference failed: {err}")
                else:
                    probs = resp.get("probabilities", {})
                    action = resp.get("action_name", "?")
                    icon_map = {"reroute":"⚖️","throttle":"🔧","prioritize":"&#128225;","no_action":"🛠️"}
                    st.markdown(
                        f'<div style="font-family:\'IBM Plex Mono\';font-size:10px;color:var(--text-3);'
                        f'text-transform:uppercase;letter-spacing:0.12em;">PPO Recommendation</div>'
                        f'<div style="font-family:\'Syne\';font-size:32px;font-weight:800;color:{COLORS["emerald"]};'
                        f'margin:4px 0 14px;">{icon_map.get(action,"🤖")} {action.replace("_"," ").upper()}</div>',
                        unsafe_allow_html=True,
                    )
                    if probs:
                        pdf = pd.DataFrame(
                            sorted(probs.items(), key=lambda kv: -kv[1]),
                            columns=["action", "probability"],
                        )
                        bar = go.Figure(go.Bar(
                            x=pdf["probability"], y=pdf["action"], orientation="h",
                            marker=dict(color=[hex_to_rgba(COLORS["emerald"], 0.85 if a == action else 0.35)
                                               for a in pdf["action"]]),
                            text=[f"{p:.1%}" for p in pdf["probability"]], textposition="outside",
                        ))
                        _bar_xaxis = {**PLOT_LAYOUT.get("xaxis", {}), "range": [0, 1], "tickformat": ".0%"}
                        bar.update_layout(**{k: v for k, v in PLOT_LAYOUT.items() if k != "xaxis"},
                                          height=200, margin=dict(t=5,b=5,l=5,r=40),
                                          xaxis=_bar_xaxis)
                        st.plotly_chart(bar, width='stretch')

        st.markdown("</div>", unsafe_allow_html=True)

    with adv_r:
        st.markdown('<div class="glass-panel"><div class="glass-panel-title">🏁 Training Benchmark · PPO vs Baselines</div>', unsafe_allow_html=True)
        bench, berr = _rl_api_get("/rl/benchmark", timeout=5.0)
        if berr or not bench:
            st.caption(f"Benchmark unavailable: {berr or 'empty response'}")
        else:
            rows = bench.get("benchmark", {})
            if rows:
                bdf = (
                    pd.DataFrame(rows).T.reset_index().rename(columns={"index": "policy"})
                )
                keep = [c for c in ["policy","reward_mean","violations_mean","violation_rate","risk_mean"] if c in bdf.columns]
                bdf = bdf[keep]
                color_for = {"PPO (best)": COLORS["emerald"], "Random": COLORS["cyan"], "Rule-Based": "#4a5568"}
                colors_bar = [color_for.get(p, COLORS["violet"]) for p in bdf["policy"]]

                fig_b = go.Figure(go.Bar(
                    x=bdf["policy"], y=bdf["reward_mean"],
                    marker=dict(color=[hex_to_rgba(c, 0.8) for c in colors_bar],
                                line=dict(color=colors_bar, width=2)),
                    text=[f"{v:.2f}" for v in bdf["reward_mean"]], textposition="outside",
                ))
                fig_b.update_layout(**PLOT_LAYOUT, height=220, margin=dict(t=10,b=10,l=10,r=10),
                                    yaxis_title="Mean reward")
                st.plotly_chart(fig_b, width='stretch')
                st.caption(
                    f"Model: `{bench.get('algorithm','PPO')}` · "
                    f"Env: `{bench.get('environment','QoSNetworkEnv')}` · "
                    f"Trained on {bench.get('total_timesteps','?'):,} timesteps" if isinstance(bench.get('total_timesteps'), int)
                    else f"Model: `{bench.get('algorithm','PPO')}` · Env: `{bench.get('environment','QoSNetworkEnv')}`"
                )
        st.markdown("</div>", unsafe_allow_html=True)

    # ── Panel: Live Simulation on Digital Twin ──
    st.markdown('<div class="glass-panel"><div class="glass-panel-title">🎮 Live Digital-Twin Simulation</div>', unsafe_allow_html=True)
    sim_l, sim_r = st.columns([1, 3])
    with sim_l:
        n_ep = st.slider("Episodes", 1, 50, 10, key="rl_sim_episodes")
        sim_go = st.button("Run PPO on Twin", key="rl_sim_btn", type="primary")
    with sim_r:
        if sim_go:
            with st.spinner(f"Running {n_ep} episodes on the NS-3 digital twin…"):
                sim, serr = _rl_api_post("/rl/simulate", {"episodes": n_ep, "deterministic": True}, timeout=120.0)
            if serr:
                st.error(f"Simulation failed: {serr}")
            elif sim:
                st.session_state["_rl_sim_result"] = sim

        sim = st.session_state.get("_rl_sim_result")
        if sim and isinstance(sim, dict) and "episodes" in sim and "summary" in sim:
            eps_df = pd.DataFrame(sim["episodes"])
            summ = sim["summary"] if isinstance(sim["summary"], dict) else {}

            k1, k2, k3, k4 = st.columns(4)
            mean_r = summ.get("mean_reward")
            std_r  = summ.get("std_reward")
            k1.metric("Mean reward",
                      f"{mean_r:.2f}" if isinstance(mean_r,(int,float)) else "—",
                      f"±{std_r:.2f}" if isinstance(std_r,(int,float)) else None)
            mean_v = summ.get("mean_violations")
            k2.metric("Violations / ep",
                      f"{mean_v:.1f}" if isinstance(mean_v,(int,float)) else "—")
            k3.metric("Total violations", str(summ.get("total_violations", "—")))
            action_mix = summ.get("action_mix", {})
            if action_mix:
                top_act = max(action_mix.items(), key=lambda kv: kv[1])
                k4.metric("Dominant action", top_act[0], f"{top_act[1]:.0%}")
            else:
                k4.metric("Dominant action", "—")

            if {"episode","reward"}.issubset(eps_df.columns):
                fig_s = make_subplots(specs=[[{"secondary_y": True}]])
                fig_s.add_trace(go.Scatter(
                    x=eps_df["episode"], y=eps_df["reward"], name="Reward",
                    mode="lines+markers", line=dict(color=COLORS["emerald"], width=2.5),
                    fill="tozeroy", fillcolor=hex_to_rgba(COLORS["emerald"], 0.1),
                ), secondary_y=False)
                if "violations" in eps_df.columns:
                    fig_s.add_trace(go.Bar(
                        x=eps_df["episode"], y=eps_df["violations"], name="SLA violations",
                        marker=dict(color=hex_to_rgba(COLORS["amber"], 0.7)),
                    ), secondary_y=True)
                fig_s.update_layout(**PLOT_LAYOUT, height=280, margin=dict(t=10,b=10,l=10,r=10),
                                    xaxis_title="Episode")
                fig_s.update_yaxes(title_text="Reward",       secondary_y=False)
                fig_s.update_yaxes(title_text="Violations",   secondary_y=True)
                st.plotly_chart(fig_s, width='stretch')
            else:
                st.caption("Simulation returned no per-episode trace data.")
        else:
            st.caption("Run a simulation to see per-episode reward and SLA-violation traces from the PPO agent.")
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


# ══════════════════════════════════════════════════════════════════════════════
#  PAGE — LIVE NETWORK MONITOR
# ══════════════════════════════════════════════════════════════════════════════

def page_live_monitor():
    """Real-time network monitor — live NS-3 5G Tunis simulation feed."""
    page_header("📡", "Live Network Monitor",
                "Real-Time NS-3 5G Simulation · 3 gNBs · 20 UEs · Tunis, Tunisia")

    # ── Connection status ────────────────────────────────────────────────
    status, err = _rl_api_get("/live/status", timeout=8.0)
    if err:
        st.error(f"Cannot reach API: {err}")
        return

    connected = status.get("bridge_connected", False)
    total_rx  = status.get("total_received", 0)
    buf_size  = status.get("buffer_size", 0)
    last_rx   = status.get("last_received", "never")

    if connected:
        st.markdown(
            "<div style='display:flex;align-items:center;gap:12px;padding:12px 20px;"
            "background:rgba(0,229,160,0.08);border:1px solid rgba(0,229,160,0.3);"
            "border-radius:12px;margin-bottom:16px;'>"
            "<div style='width:12px;height:12px;border-radius:50%;background:#00e5a0;"
            "box-shadow:0 0 12px #00e5a0;'></div>"
            "<span style='font-family:IBM Plex Mono,monospace;font-size:13px;color:#00e5a0;"
            "font-weight:600;'>BRIDGE CONNECTED — LIVE</span>"
            f"<span style='font-size:11px;color:var(--text-3);margin-left:auto;'>"
            f"📊 {total_rx:,} rows · 📦 {buf_size} buffered · 🕐 {last_rx}</span></div>",
            unsafe_allow_html=True)
    else:
        st.markdown(
            "<div style='display:flex;align-items:center;gap:12px;padding:12px 20px;"
            "background:rgba(255,77,109,0.08);border:1px solid rgba(255,77,109,0.3);"
            "border-radius:12px;margin-bottom:16px;'>"
            "<div style='width:12px;height:12px;border-radius:50%;background:#ff4d6d;"
            "box-shadow:0 0 12px #ff4d6d;'></div>"
            "<span style='font-family:IBM Plex Mono,monospace;font-size:13px;color:#ff4d6d;"
            "font-weight:600;'>BRIDGE OFFLINE</span>"
            "<span style='font-size:11px;color:var(--text-3);margin-left:auto;'>"
            "Waiting for bridge.py...</span></div>",
            unsafe_allow_html=True)

    if not connected and total_rx == 0:
        st.markdown("""
        <div class="glass-panel">
            <div class="glass-panel-title">🔌 How to Connect the NS-3 Simulation</div>
            <div style="font-size:13px;color:var(--text-2);line-height:1.8;">
                <b>On the simulation machine</b> (teammate's PC):<br>
                <code style="background:rgba(0,212,255,0.08);padding:8px 14px;border-radius:8px;
                      display:block;font-family:IBM Plex Mono,monospace;font-size:12px;
                      border:1px solid rgba(0,212,255,0.15);margin:8px 0;">
                python bridge.py --csv 5g_dataset_scaled.csv --api http://YOUR_IP:8000
                </code>
                <b>Test locally</b> (replay existing data):<br>
                <code style="background:rgba(0,212,255,0.08);padding:8px 14px;border-radius:8px;
                      display:block;font-family:IBM Plex Mono,monospace;font-size:12px;
                      border:1px solid rgba(0,212,255,0.15);margin:8px 0;">
                python bridge.py --csv data/5g_dataset_scaled.csv --replay --speed 20
                </code>
            </div>
        </div>
        """, unsafe_allow_html=True)
        return

    cr1, cr2, cr3, cr4 = st.columns([1, 1, 1, 1])
    auto_refresh = cr1.toggle("Auto-refresh", value=True, key="_live_auto")
    refresh_rate = cr2.selectbox("Interval", [3, 5, 10, 15, 30], index=1, key="_live_rate")
    win_size     = cr3.selectbox("Window", [100, 300, 500, 1000, 2000], index=2, key="_live_win",
                                 help="How many recent rows to fetch from the buffer")
    cr4.markdown(f"<div style='padding-top:8px;font-size:11px;color:var(--text-3);'>"
                 f"{'🔄 Every ' + str(refresh_rate) + 's · ' + str(win_size) + ' rows' if auto_refresh else '⏸ Paused'}"
                 f"</div>", unsafe_allow_html=True)

    stream, s_err = _rl_api_get(f"/live/stream?last_n={int(win_size)}", timeout=10.0)
    if s_err or not stream or not stream.get("data"):
        st.info("Buffer empty — waiting for data...")
        return
    df = pd.DataFrame(stream["data"])
    if df.empty:
        st.info("No data yet...")
        return

    # ── Delta tracker: how many new rows since last refresh ───────────────
    prev_total = st.session_state.get("_live_prev_total", None)
    delta_new  = (total_rx - prev_total) if isinstance(prev_total, int) else 0
    st.session_state["_live_prev_total"] = int(total_rx)
    n_anom = int(pd.to_numeric(df.get("_live_anomaly", pd.Series(dtype=int)), errors="coerce").fillna(0).sum())
    pct_anom = (n_anom / max(len(df),1)) * 100

    st.markdown(
        f"<div style='display:flex;gap:14px;flex-wrap:wrap;margin-bottom:10px;font-family:IBM Plex Mono,monospace;font-size:11px;color:var(--text-2);'>"
        f"<span style='padding:4px 10px;border-radius:6px;background:rgba(0,212,255,0.08);border:1px solid rgba(0,212,255,0.2);'>"
        f"<b style='color:var(--cyan);'>+{max(delta_new,0):,}</b> new rows since last refresh</span>"
        f"<span style='padding:4px 10px;border-radius:6px;background:rgba(0,229,160,0.08);border:1px solid rgba(0,229,160,0.2);'>"
        f"<b style='color:var(--emerald);'>{len(df):,}</b> rows in window</span>"
        f"<span style='padding:4px 10px;border-radius:6px;background:rgba(255,184,63,0.08);border:1px solid rgba(255,184,63,0.2);'>"
        f"<b style='color:var(--amber);'>{n_anom:,}</b> anomalies ({pct_anom:.0f}%)</span>"
        f"<span style='padding:4px 10px;border-radius:6px;background:rgba(167,139,250,0.08);border:1px solid rgba(167,139,250,0.2);'>"
        f"<b style='color:var(--violet);'>{total_rx:,}</b> rows received total</span>"
        f"</div>",
        unsafe_allow_html=True,
    )

    # ── KPI Cards ────────────────────────────────────────────────────────
    c1, c2, c3, c4, c5, c6 = st.columns(6)
    def _sm(col, fmt=".2f"):
        if col in df.columns:
            v = pd.to_numeric(df[col], errors="coerce").mean()
            return f"{v:{fmt}}" if pd.notna(v) else "—"
        return "—"
    with c1: kpi_card("LIVE ROWS", str(len(df)), "", "ok", 100)
    with c2:
        v = _sm("packet_loss_ratio", ".3f")
        sev = "crit" if v != "—" and float(v) > 0.3 else "ok"
        kpi_card("PKT LOSS", v, "ratio", sev, min(float(v)*200 if v != "—" else 0, 100))
    with c3: kpi_card("SINR", _sm("sinr_dl_db", ".1f"), "dB", "ok", 60)
    with c4: kpi_card("THROUGHPUT", _sm("throughput_mbps", ".2f"), "Mbps", "ok", 50)
    with c5:
        v = _sm("delay_ms", ".1f")
        if v == "—":
            sev = "ok"
        else:
            fv = float(v)
            sev = "crit" if fv > 250 else "warn" if fv > 100 else "ok"
        kpi_card("DELAY", v, "ms", sev, min(float(v)/5 if v != "—" else 0, 100))
    with c6:
        if "_live_anomaly" in df.columns:
            ar_raw = pd.to_numeric(df["_live_anomaly"], errors="coerce").mean()
            ar = float(ar_raw) * 100 if pd.notna(ar_raw) else 0.0
            sev = "crit" if ar > 30 else "warn" if ar > 15 else "ok"
            kpi_card("ANOMALY", f"{ar:.1f}", "%", sev, min(ar * 2, 100))
        else:
            kpi_card("ANOMALY", "—", "%", "ok", 0)

    # ── LIVE MAP — Tunis 5G Network ──────────────────────────────────────
    if "ue_lat" in df.columns and "ue_lon" in df.columns:
        st.markdown('<div class="glass-panel"><div class="glass-panel-title">'
                    '🗺️ Live Network Map — Tunis 5G Coverage</div>', unsafe_allow_html=True)
        for nc in ["ue_lat", "ue_lon", "packet_loss_ratio", "sinr_dl_db"]:
            if nc in df.columns: df[nc] = pd.to_numeric(df[nc], errors="coerce")
        latest = df.groupby("ue_id").tail(1).copy() if "ue_id" in df.columns else df.tail(20).copy()

        def _ue_color(row):
            loss = row.get("packet_loss_ratio", 0)
            anom = row.get("_live_anomaly", 0)
            if anom == 1 or (isinstance(loss, (int, float)) and loss > 0.5): return "#ff4d6d"
            elif isinstance(loss, (int, float)) and loss > 0.2: return "#ffb83f"
            return "#00e5a0"

        latest["_hex"] = latest.apply(_ue_color, axis=1)
        latest["_size"] = latest.apply(lambda r: 14 if r.get("_live_anomaly", 0) == 1 else 9, axis=1)

        def _hover(r):
            p = [f"<b>{r.get('ue_name', 'UE '+str(r.get('ue_id','?')))}</b>"]
            p.append(f"gNB: {r.get('gnb_name', '?')}")
            for k, l, f in [("packet_loss_ratio","Loss",".3f"),("sinr_dl_db","SINR",".1f dB"),
                             ("throughput_mbps","Tput",".2f Mbps"),("delay_ms","Delay",".0f ms")]:
                if k in r and pd.notna(r[k]): p.append(f"{l}: {r[k]:{f.split()[0]}}{' '+f.split()[1] if len(f.split())>1 else ''}")
            if "_rl_action" in r and r["_rl_action"] != "unknown": p.append(f"RL: {r['_rl_action']}")
            if "_live_anomaly" in r: p.append("⚠️ ANOMALY" if r["_live_anomaly"] == 1 else "✅ Normal")
            return "<br>".join(p)

        latest["_hover"] = latest.apply(_hover, axis=1)

        fig_map = go.Figure()
        fig_map.add_trace(go.Scattermapbox(
            lat=latest["ue_lat"], lon=latest["ue_lon"], mode="markers+text",
            marker=dict(size=latest["_size"], color=latest["_hex"], opacity=0.9),
            text=latest.get("ue_name", latest.get("ue_id", "")),
            textposition="top right", textfont=dict(size=8, color="white"),
            hovertext=latest["_hover"], hoverinfo="text", name="UEs",
        ))
        if "gnb_lat" in df.columns and "gnb_lon" in df.columns:
            gnbs = df[["gnb_id","gnb_name","gnb_lat","gnb_lon"]].drop_duplicates()
            for nc in ["gnb_lat","gnb_lon"]: gnbs[nc] = pd.to_numeric(gnbs[nc], errors="coerce")
            gnbs = gnbs.dropna(subset=["gnb_lat","gnb_lon"])
            fig_map.add_trace(go.Scattermapbox(
                lat=gnbs["gnb_lat"], lon=gnbs["gnb_lon"], mode="markers+text",
                marker=dict(size=18, color="#00d4ff", symbol="circle", opacity=0.95),
                text=gnbs["gnb_name"], textposition="bottom center",
                textfont=dict(size=9, color="#00d4ff", family="IBM Plex Mono"),
                hovertext=gnbs["gnb_name"].apply(lambda n: f"<b>📡 {n}</b><br>Base Station"),
                hoverinfo="text", name="gNB Towers",
            ))
        _clat_m = latest["ue_lat"].mean()
        _clon_m = latest["ue_lon"].mean()
        clat = float(_clat_m) if pd.notna(_clat_m) else 36.84
        clon = float(_clon_m) if pd.notna(_clon_m) else 10.19
        fig_map.update_layout(
            mapbox=dict(style="open-street-map", center=dict(lat=clat, lon=clon), zoom=12.5),
            paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="rgba(0,0,0,0)",
            margin=dict(t=0,b=0,l=0,r=0), height=420, showlegend=True,
            legend=dict(bgcolor="rgba(5,10,20,0.7)", font=dict(color="#e8f0fe",size=10), x=0.01, y=0.99),
        )
        st.plotly_chart(fig_map, use_container_width=True)
        st.markdown(
            "<div style='display:flex;gap:20px;justify-content:center;margin-top:4px;'>"
            "<span style='display:inline-flex;align-items:center;gap:6px;'><span style='width:10px;height:10px;border-radius:50%;background:#00e5a0;'></span><span style='font-size:10px;color:var(--text-3);'>Normal</span></span>"
            "<span style='display:inline-flex;align-items:center;gap:6px;'><span style='width:10px;height:10px;border-radius:50%;background:#ffb83f;'></span><span style='font-size:10px;color:var(--text-3);'>Degraded</span></span>"
            "<span style='display:inline-flex;align-items:center;gap:6px;'><span style='width:10px;height:10px;border-radius:50%;background:#ff4d6d;'></span><span style='font-size:10px;color:var(--text-3);'>Anomaly</span></span>"
            "<span style='display:inline-flex;align-items:center;gap:6px;'><span style='width:10px;height:10px;border-radius:50%;background:#00d4ff;'></span><span style='font-size:10px;color:var(--text-3);'>gNB Tower</span></span>"
            "</div>", unsafe_allow_html=True)
        st.markdown("</div>", unsafe_allow_html=True)

    # ── LIVE KPI TIME SERIES ─────────────────────────────────────────────
    st.markdown('<div class="glass-panel"><div class="glass-panel-title">📊 Live KPI Time Series</div>', unsafe_allow_html=True)
    if "_server_ts" in df.columns: df["_x"] = pd.to_datetime(df["_server_ts"], unit="s")
    elif "timestamp" in df.columns: df["_x"] = df["timestamp"]
    else: df["_x"] = range(len(df))
    kpi_cols = [c for c in ["packet_loss_ratio","sinr_dl_db","throughput_mbps","delay_ms","jitter_ms"] if c in df.columns]
    kc = {"packet_loss_ratio":COLORS["critical"],"sinr_dl_db":COLORS["cyan"],"throughput_mbps":COLORS["emerald"],"delay_ms":COLORS["amber"],"jitter_ms":COLORS["violet"]}
    if kpi_cols:
        tabs = st.tabs([c.replace("_"," ").title() for c in kpi_cols])
        for tab, col in zip(tabs, kpi_cols):
            with tab:
                vals = pd.to_numeric(df[col], errors="coerce")
                fig = go.Figure()
                fig.add_trace(go.Scatter(x=df["_x"], y=vals, mode="lines", line=dict(color=kc.get(col,COLORS["cyan"]),width=2), fill="tozeroy", fillcolor=hex_to_rgba(kc.get(col,COLORS["cyan"]),0.08), name=col))
                if "_live_anomaly" in df.columns:
                    am = df["_live_anomaly"]==1
                    if am.any(): fig.add_trace(go.Scatter(x=df.loc[am,"_x"], y=vals[am], mode="markers", marker=dict(color=COLORS["critical"],size=8,symbol="triangle-up",line=dict(width=1,color="white")), name="Anomaly"))
                fig.update_layout(**PLOT_LAYOUT, height=280, margin=dict(t=10,b=10,l=10,r=10), showlegend=True, legend=dict(orientation="h",y=1.1))
                st.plotly_chart(fig, use_container_width=True)
    st.markdown("</div>", unsafe_allow_html=True)

    # ── RL AGENT LIVE ────────────────────────────────────────────────────
    if "_rl_action" in df.columns:
        st.markdown('<div class="glass-panel"><div class="glass-panel-title">🤖 RL Agent — Live Actions</div>', unsafe_allow_html=True)
        ac = df["_rl_action"].value_counts()
        acol = {"reroute":COLORS["cyan"],"throttle":COLORS["amber"],"prioritize":COLORS["emerald"],"no_action":COLORS["violet"],"unknown":COLORS["critical"]}
        c1,c2 = st.columns([1,2])
        with c1:
            fig = go.Figure(go.Pie(labels=ac.index.tolist(), values=ac.values.tolist(), marker=dict(colors=[acol.get(a,COLORS["cyan"]) for a in ac.index]), hole=0.55, textinfo="label+percent", textfont=dict(size=10)))
            fig.update_layout(**PLOT_LAYOUT, height=260, margin=dict(t=10,b=10,l=10,r=10), showlegend=False)
            st.plotly_chart(fig, use_container_width=True)
        with c2:
            amap = {"reroute":0,"throttle":1,"prioritize":2,"no_action":3}
            fig2 = go.Figure()
            fig2.add_trace(go.Scatter(x=df["_x"], y=df["_rl_action"].map(amap).fillna(-1), mode="markers+lines", marker=dict(color=[acol.get(a,COLORS["cyan"]) for a in df["_rl_action"]],size=6), line=dict(color="rgba(0,212,255,0.2)",width=1), hovertext=df["_rl_action"], hoverinfo="text"))
            fig2.update_layout(**PLOT_LAYOUT, height=260, margin=dict(t=10,b=10,l=10,r=10))
            fig2.update_yaxes(tickvals=[0,1,2,3], ticktext=["reroute","throttle","prioritize","no_action"])
            st.plotly_chart(fig2, use_container_width=True)
        st.markdown("</div>", unsafe_allow_html=True)

    # ── ANOMALY SCORES ───────────────────────────────────────────────────
    if "_live_scores" in df.columns:
        st.markdown('<div class="glass-panel"><div class="glass-panel-title">🎯 Live Anomaly Scores</div>', unsafe_allow_html=True)
        sdf = pd.json_normalize(df["_live_scores"].apply(lambda x: x if isinstance(x,dict) else {}))
        if not sdf.empty and len(sdf.columns) > 0:
            for sc in sdf.columns: sdf[sc] = pd.to_numeric(sdf[sc], errors="coerce")
            mcm = {"isolation_forest":COLORS["cyan"],"random_forest":COLORS["emerald"],"gradient_boosting":COLORS["amber"]}
            fig = go.Figure()
            for sc in sdf.columns: fig.add_trace(go.Scatter(x=df["_x"], y=sdf[sc], mode="lines", name=sc.replace("_"," ").title(), line=dict(color=mcm.get(sc,COLORS["violet"]),width=1.5)))
            fig.add_hline(y=0.5, line=dict(color=COLORS["critical"],dash="dash",width=1), annotation_text="Threshold")
            fig.update_layout(**PLOT_LAYOUT, height=280, margin=dict(t=10,b=10,l=10,r=10), legend=dict(orientation="h",y=1.1))
            st.plotly_chart(fig, use_container_width=True)
        else:
            st.markdown("<div style='padding:30px;text-align:center;color:var(--text-3);font-size:11px;'>no model scores attached to live rows</div>", unsafe_allow_html=True)
        st.markdown("</div>", unsafe_allow_html=True)

    # ── PER-UE TABLE ─────────────────────────────────────────────────────
    if "ue_id" in df.columns and "ue_name" in df.columns:
        with st.expander("📋 Per-UE Live Status", expanded=False):
            uel = df.groupby("ue_id").tail(1)
            sc = [c for c in ["ue_id","ue_name","gnb_name","sinr_dl_db","throughput_mbps","delay_ms","packet_loss_ratio","load_level","ue_dist_to_gnb_m","_rl_action","_live_anomaly"] if c in uel.columns]
            st.dataframe(uel[sc].reset_index(drop=True), use_container_width=True, height=400)

    with st.expander("📋 Raw Live Data (latest 50)", expanded=False):
        sc2 = [c for c in ["timestamp","ue_id","ue_name","gnb_name","sinr_dl_db","throughput_mbps","delay_ms","jitter_ms","packet_loss_ratio","load_level","_rl_action","_live_anomaly"] if c in df.columns]
        st.dataframe(df[sc2].tail(50).reset_index(drop=True), use_container_width=True, height=300)

    # ── Auto-refresh ─────────────────────────────────────────────────────
    # Streamlit doesn't loop on its own — the previous gating-on-elapsed-time
    # logic never triggered because the script returned to idle immediately.
    # Sleep then explicit rerun forces a real periodic re-fetch + redraw.
    if auto_refresh:
        import time as _time
        _time.sleep(refresh_rate)
        st.rerun()


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
                try:
                    from rag.rag_pipeline import query_rag
                    result = query_rag(question=user_input, collection=collection,
                                       embed_model=embed_model, chain=chain, mode=mode,
                                       severity_filter=sev_flt, root_cause_filter=rc_flt)
                    answer  = result.get("answer", "(empty answer)") if isinstance(result, dict) else str(result)
                    sources = result.get("sources", []) if isinstance(result, dict) else []
                except Exception as e:
                    answer  = f"⚠️ RAG query failed: {e}. Check that Ollama is reachable."
                    sources = []
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
        _ans = turn.get('answer','') or ''
        st.markdown(f"""
        <div class="chat-bubble chat-bot">
          <div class="chat-meta">QoSBuddy NOC Assistant · {len(turn.get('sources') or [])} sources</div>
          {_ans.replace(chr(10),'<br>')}
        </div>""", unsafe_allow_html=True)
        srcs = turn.get("sources") or []
        if srcs:
            with st.expander(f"📚 {len(srcs)} retrieved sources"):
                items = ""
                for s in srcs:
                    if not isinstance(s, dict): continue
                    m = s.get("metadata") or {}
                    sim = s.get("similarity")
                    sim_str = f"{sim:.3f}" if isinstance(sim,(int,float)) else "—"
                    items += (f"""<div class="chat-source-item">"""
                              f"""<span class="src-badge">{(m.get('doc_type','?') or '?')[:8]}</span>"""
                              f"""<span class="src-sim">sim={sim_str}</span>"""
                              f"""<span>sev={m.get('severity','—')}</span>"""
                              f"""<span>rc={m.get('root_cause','—')}</span></div>""")
                st.markdown(f'<div class="chat-sources-wrap">{items}</div>', unsafe_allow_html=True)
    st.markdown("</div>", unsafe_allow_html=True)

    if st.session_state["chat_history"]:
        if st.button("🗑  Clear Chat"):
            st.session_state["chat_history"] = []
            st.rerun()



# ══════════════════════════════════════════════════════════════════════════════
#  PAGE — VOICE NOC ASSISTANT  &#127897;  (DSO3.2 · XAI · GenAI)
# ══════════════════════════════════════════════════════════════════════════════

VOICE_SUGGESTIONS = [
    "Which UEs are at critical risk right now?",
    "What is the main root cause of packet loss?",
    "What action should I take for UE 3?",
    "How much can congestion be reduced?",
    "Explain the SLA violation pattern.",
    "What does QoSBuddy recommend for high load?",
]


def _voice_orb_html():
    return r"""

<!DOCTYPE html>
<html>
<head>
<meta charset="utf-8">
<style>
*{margin:0;padding:0;box-sizing:border-box}
html,body{width:100%;height:100%;background:transparent;overflow:hidden;font-family:'IBM Plex Mono',monospace;}
#c{display:block;width:100%;height:100%;}
#click-zone{position:absolute;top:2%;left:50%;transform:translateX(-50%);width:260px;height:440px;cursor:pointer;background:transparent;border:none;z-index:10;}
#ui{position:absolute;bottom:6px;left:0;right:0;display:flex;flex-direction:column;align-items:center;gap:6px;pointer-events:none;}
#status-pill{padding:5px 22px;border-radius:30px;font-size:9px;letter-spacing:.22em;text-transform:uppercase;transition:all .4s;white-space:nowrap;}
.s-idle{color:rgba(0,212,255,.5);border:1px solid rgba(0,212,255,.2);background:rgba(0,212,255,.05);}
.s-listening{color:#00e5a0;border:1px solid rgba(0,229,160,.5);background:rgba(0,229,160,.08);box-shadow:0 0 18px rgba(0,229,160,.15);}
.s-processing{color:#ffb83f;border:1px solid rgba(255,184,63,.5);background:rgba(255,184,63,.07);}
.s-speaking{color:#a78bfa;border:1px solid rgba(167,139,250,.5);background:rgba(167,139,250,.08);box-shadow:0 0 18px rgba(167,139,250,.15);}
.s-ready{color:#00e5a0;border:1px solid rgba(0,229,160,.6);background:rgba(0,229,160,.1);box-shadow:0 0 22px rgba(0,229,160,.2);}
#transcript{max-width:420px;width:88%;padding:8px 16px;border-radius:12px;background:rgba(0,5,18,.78);border:1px solid rgba(0,212,255,.13);font-size:11px;color:rgba(185,218,255,.72);line-height:1.6;text-align:center;transition:border-color .3s;}
#transcript.s-listening{border-color:rgba(0,229,160,.38);}
#transcript.s-ready{border-color:rgba(0,229,160,.45);}
#transcript.s-speaking{border-color:rgba(167,139,250,.4);}
#wave{display:none;align-items:flex-end;gap:2.5px;height:18px;}
.wb{width:3px;border-radius:2px 2px 0 0;background:linear-gradient(180deg,#a78bfa,#00d4ff);}
.wb:nth-child(1){animation:wa .50s .00s ease-in-out infinite alternate;height:5px;}
.wb:nth-child(2){animation:wa .40s .04s ease-in-out infinite alternate;height:13px;}
.wb:nth-child(3){animation:wa .60s .08s ease-in-out infinite alternate;height:18px;}
.wb:nth-child(4){animation:wa .45s .12s ease-in-out infinite alternate;height:14px;}
.wb:nth-child(5){animation:wa .55s .16s ease-in-out infinite alternate;height:18px;}
.wb:nth-child(6){animation:wa .50s .20s ease-in-out infinite alternate;height:10px;}
.wb:nth-child(7){animation:wa .65s .24s ease-in-out infinite alternate;height:16px;}
.wb:nth-child(8){animation:wa .42s .28s ease-in-out infinite alternate;height:7px;}
.wb:nth-child(9){animation:wa .58s .32s ease-in-out infinite alternate;height:14px;}
.wb:nth-child(10){animation:wa .48s .36s ease-in-out infinite alternate;height:18px;}
.wb:nth-child(11){animation:wa .62s .40s ease-in-out infinite alternate;height:9px;}
.wb:nth-child(12){animation:wa .52s .44s ease-in-out infinite alternate;height:13px;}
@keyframes wa{from{transform:scaleY(.2);opacity:.4}to{transform:scaleY(1);opacity:1}}
#hint{font-size:8px;color:rgba(0,212,255,.18);letter-spacing:.18em;text-transform:uppercase;}
</style>
</head>
<body>
<canvas id="c"></canvas>
<button id="click-zone" onclick="toggleListen()" title="Click to speak"></button>
<div id="ui">
  <div id="wave"><div class="wb"></div><div class="wb"></div><div class="wb"></div><div class="wb"></div><div class="wb"></div><div class="wb"></div><div class="wb"></div><div class="wb"></div><div class="wb"></div><div class="wb"></div><div class="wb"></div><div class="wb"></div></div>
  <div id="status-pill" class="s-idle">Say "Hey Buddy" to start</div>
  <div id="transcript" class="s-idle">Tap the head to activate voice...</div>
  <div id="hint">Chrome &middot; Allow microphone &middot; English</div>
</div>
<script src="https://cdnjs.cloudflare.com/ajax/libs/three.js/r128/three.min.js"></script>
<!-- Post-processing: Bloom + SSAO (Improvement 2) -->
<script src="https://cdn.jsdelivr.net/npm/three@0.128.0/examples/js/postprocessing/EffectComposer.js"></script>
<script src="https://cdn.jsdelivr.net/npm/three@0.128.0/examples/js/postprocessing/RenderPass.js"></script>
<script src="https://cdn.jsdelivr.net/npm/three@0.128.0/examples/js/postprocessing/ShaderPass.js"></script>
<script src="https://cdn.jsdelivr.net/npm/three@0.128.0/examples/js/postprocessing/UnrealBloomPass.js"></script>
<script src="https://cdn.jsdelivr.net/npm/three@0.128.0/examples/js/math/SimplexNoise.js"></script>
<script src="https://cdn.jsdelivr.net/npm/three@0.128.0/examples/js/postprocessing/SSAOPass.js"></script>
<script src="https://cdn.jsdelivr.net/npm/three@0.128.0/examples/js/shaders/LuminosityHighPassShader.js"></script>
<script src="https://cdn.jsdelivr.net/npm/three@0.128.0/examples/js/shaders/CopyShader.js"></script>
<script src="https://cdn.jsdelivr.net/npm/three@0.128.0/examples/js/shaders/SSAOShader.js"></script>
<script>
'use strict';

/* ═══════════════════════════════════════════════════════════════════════════
   QoSBuddy Holographic Face v5.0 — Realistic Humanoid Avatar
   Anatomically sculpted geometry with defined jawline, human-like lips,
   realistic ears, and holographic shader with lip-sync jaw and state-reactive
   animations. Reference: realistic 3D human head proportions.
   ═══════════════════════════════════════════════════════════════════════════ */

// ─── State ───────────────────────────────────────────────────────────────────
var STATE = 'idle';
var jawAngle = 0, jawTarget = 0;
var isListening = false;
var blinkTimer = 2.5, blinkState = 0, blinkT = 0;
var headSway = 0, headNod = 0, breathPhase = 0;
var eyeL_target = new THREE.Vector2(0,0), eyeR_target = new THREE.Vector2(0,0);
var eyeSaccadeTimer = 0;

function applyState(s) {
  STATE = s;
  var pill = document.getElementById('status-pill');
  var tc   = document.getElementById('transcript');
  var wave = document.getElementById('wave');
  pill.className = 's-' + s;
  tc.className   = 's-' + s;
  wave.style.display = (s === 'speaking') ? 'flex' : 'none';
  var labels = {
    idle:       'Say "Hey Buddy" to start',
    listening:  'Listening…',
    processing: 'Thinking…',
    speaking:   'Speaking…',
    ready:      'Answer ready — click to ask again'
  };
  pill.textContent = labels[s] || s;

  // Improvement 6: ripple on state change
  if (clock) {
    rippleStartTime = clock.getElapsedTime();
    ALL_MATS.forEach(function(m) {
      if (m.uniforms && m.uniforms.uRippleTime)
        m.uniforms.uRippleTime.value = rippleStartTime;
    });
  }
  // Improvement 5: morph targets (brows/squint)
  if (s === 'listening')  morphWeights = [1.0, 0.0, 0.0];
  else if (s === 'processing') morphWeights = [0.0, 1.0, 0.0];
  else if (s === 'speaking')   morphWeights = [0.0, 0.0, 0.8];
  else                         morphWeights = [0.0, 0.0, 0.0];
  // Improvement 1: start/stop mic
  if (s === 'listening' || s === 'speaking') startAudioCapture();
  else if (s === 'idle' || s === 'ready') stopAudioCapture();
  // SSS color shift by state
  if (sssMesh) {
    sssMesh.material.color.setHex(
      s==='listening'  ? 0x00e5a0 :
      s==='speaking'   ? 0xa78bfa :
      s==='processing' ? 0xffb83f : 0x00aaff
    );
  }
}

function push_state(s, txt) {
  applyState(s);
  if (txt) document.getElementById('transcript').textContent = txt;
}

window.addEventListener('message', function(ev) {
  if (!ev.data || !ev.data.type) return;
  if (ev.data.type === 'qosbuddy_tts_start') applyState('speaking');
  if (ev.data.type === 'qosbuddy_tts_done')  applyState('idle');
  if (ev.data.type === 'qosbuddy_state')     push_state(ev.data.state, ev.data.text || '');
});

// ─── Renderer ────────────────────────────────────────────────────────────────
var canvas = document.getElementById('c');
var renderer = new THREE.WebGLRenderer({ canvas: canvas, antialias: true, alpha: true });
renderer.setPixelRatio(Math.min(window.devicePixelRatio, 2));
renderer.setSize(canvas.clientWidth, canvas.clientHeight);
renderer.shadowMap.enabled = true;
renderer.shadowMap.type = THREE.PCFSoftShadowMap;
renderer.setClearColor(0x000000, 0);

var scene = new THREE.Scene();
var camera = new THREE.PerspectiveCamera(28, canvas.clientWidth / canvas.clientHeight, 0.1, 100);
camera.position.set(0, 0.10, 5.9);
camera.lookAt(0, 0.1, 0);

window.addEventListener('resize', function() {
  var w = canvas.clientWidth, h = canvas.clientHeight;
  renderer.setSize(w, h);
  camera.aspect = w / h;
  camera.updateProjectionMatrix();
  if (composer) composer.setSize(w, h);
});

// ─── IMPROVEMENT 2: Post-processing — Bloom + SSAO + Chromatic Aberration ────
var composer = null, bloomPass = null, ssaoPass = null, chromaticPass = null;

var ChromaShader = {
  uniforms: { tDiffuse:{value:null}, uStr:{value:0.0016} },
  vertexShader: 'varying vec2 vUv;void main(){vUv=uv;gl_Position=projectionMatrix*modelViewMatrix*vec4(position,1.0);}',
  fragmentShader: [
    'uniform sampler2D tDiffuse;uniform float uStr;varying vec2 vUv;',
    'void main(){vec2 d=vUv-vec2(0.5);float dist=length(d);vec2 off=normalize(d)*uStr*dist*3.5;',
    'float r=texture2D(tDiffuse,vUv+off).r;',
    'float g=texture2D(tDiffuse,vUv).g;',
    'float b=texture2D(tDiffuse,vUv-off).b;',
    'gl_FragColor=vec4(r,g,b,texture2D(tDiffuse,vUv).a);}'
  ].join('')
};

function initPostFX() {
  try {
    if (typeof THREE.EffectComposer === 'undefined') {
      console.warn('[v7] EffectComposer not loaded');
      return;
    }
    composer = new THREE.EffectComposer(renderer);
    composer.addPass(new THREE.RenderPass(scene, camera));

    // SSAO — socket shadows, eye socket depth, nostrils
    if (typeof THREE.SSAOPass !== 'undefined') {
      ssaoPass = new THREE.SSAOPass(scene, camera, canvas.clientWidth, canvas.clientHeight);
      ssaoPass.kernelRadius = 14;
      ssaoPass.minDistance  = 0.004;
      ssaoPass.maxDistance  = 0.10;
      ssaoPass.output = THREE.SSAOPass.OUTPUT.Default;
      composer.addPass(ssaoPass);
      console.log('[v7] SSAO active');
    }

    // Bloom — bright holographic edges bleed into darkness
    bloomPass = new THREE.UnrealBloomPass(
      new THREE.Vector2(canvas.clientWidth, canvas.clientHeight),
      0.85, 0.38, 0.12
    );
    composer.addPass(bloomPass);

    // Chromatic aberration — holographic fringe
    chromaticPass = new THREE.ShaderPass(ChromaShader);
    chromaticPass.renderToScreen = true;
    composer.addPass(chromaticPass);

    // Now safe to apply corneal env map — cubeRenderTarget exists here
    if (typeof applyEnvMapToEyes === 'function') applyEnvMapToEyes();
    console.log('[v7] Bloom + Chroma active');
  } catch(e) {
    console.warn('[v7] PostFX failed:', e.message);
    composer = null;
    // Still apply env map even if postFX fails
    if (typeof applyEnvMapToEyes === 'function') {
      try { applyEnvMapToEyes(); } catch(e2) {}
    }
  }
}
setTimeout(initPostFX, 150);

// ─── Holographic Shader ───────────────────────────────────────────────────────
var holoVS = `
  varying vec3 vNormal;
  varying vec3 vViewDir;
  varying vec3 vWorldPos;
  varying vec2 vUv;
  void main() {
    vec4 worldPos = modelMatrix * vec4(position, 1.0);
    vWorldPos = worldPos.xyz;
    vec4 mvPos  = viewMatrix * worldPos;
    vViewDir    = normalize(-mvPos.xyz);
    vNormal     = normalize(normalMatrix * normal);
    vUv         = uv;
    gl_Position = projectionMatrix * mvPos;
  }
`;

var holoFS = `
  uniform vec3  uBase;
  uniform vec3  uGlow;
  uniform float uOpacity;
  uniform float uTime;
  uniform float uPulse;
  uniform float uScan;
  uniform float uRippleTime;
  varying vec3  vNormal;
  varying vec3  vViewDir;
  varying vec3  vWorldPos;
  varying vec2  vUv;
  void main() {
    float fresnel = pow(1.0 - clamp(dot(vNormal, vViewDir), 0.0, 1.0), 2.8);
    // Horizontal scan lines — holographic signature
    float sl1 = smoothstep(0.36,0.44,fract(vWorldPos.y * 42.0 + uTime * 1.2)) *
                smoothstep(0.64,0.56,fract(vWorldPos.y * 42.0 + uTime * 1.2));
    float sl2 = smoothstep(0.42,0.46,fract(vWorldPos.y * 130.0)) *
                smoothstep(0.58,0.54,fract(vWorldPos.y * 130.0)) * 0.06;
    float shimmer = (sin(vWorldPos.y * 4.0 + uTime * 2.5) * 0.5 + 0.5) * 0.10;
    // Edge data-lines (vertical shimmer near silhouette)
    float edgeLine = step(0.85, fresnel) * (sin(vWorldPos.y * 80.0 + uTime * 3.0)*0.5+0.5) * 0.18;
    // Improvement 6 — ripple ring on state change
    float rippleAge  = uTime - uRippleTime;
    float rippleRing = length(vWorldPos.xz);
    float ripple = smoothstep(0.06,0.0,abs(rippleRing - rippleAge*2.4))
                   * max(0.0, 1.0 - rippleAge*1.5) * 0.45;
    vec3 col = mix(uBase, uGlow, fresnel);
    col += ripple * uGlow;
    col += vec3(0.0, sl1 * 0.28 * uScan, sl1 * 0.50 * uScan);
    col += sl2 * vec3(0.05, 0.18, 0.38);
    col += shimmer * uGlow * 0.7;
    col += edgeLine * uGlow;
    col += uPulse * uGlow * 0.18;
    float alpha = mix(uOpacity, 1.0, fresnel * 0.72) + sl1 * 0.07 * uScan;
    gl_FragColor = vec4(col, clamp(alpha, 0.0, 1.0));
  }
`;

function mkMat(opts) {
  var o = opts || {};
  return new THREE.ShaderMaterial({
    uniforms: {
      uBase:    { value: new THREE.Color(o.base    || 0x011030) },
      uGlow:    { value: new THREE.Color(o.glow    || 0x00b8ff) },
      uOpacity: { value: o.opacity !== undefined ? o.opacity : 0.82 },
      uTime:       { value: 0 },
      uPulse:      { value: 0 },
      uScan:       { value: o.scan !== undefined ? o.scan : 0.7 },
      uRippleTime: { value: -99.0 }
    },
    vertexShader: holoVS,
    fragmentShader: holoFS,
    transparent: true,
    side: THREE.FrontSide,
    depthWrite: true
  });
}

// Collect all mats for time update

var ALL_MATS = [];
function regMat(m) { ALL_MATS.push(m); return m; }

// ─── IMPROVEMENT 4: Procedural Iris Texture ──────────────────────────────────
function makeIrisTexture() {
  var ic = document.createElement('canvas');
  ic.width = ic.height = 256;
  var c = ic.getContext('2d');
  // Deep blue base
  c.fillStyle='#001244'; c.fillRect(0,0,256,256);
  // 220 radial fibers from center
  for(var f=0;f<220;f++){
    var ang=(f/220)*Math.PI*2;
    var a=0.05+Math.random()*0.13;
    c.strokeStyle='rgba('+Math.floor(Math.random()*40)+','+Math.floor(80+Math.random()*80)+','+Math.floor(190+Math.random()*65)+','+a+')';
    c.lineWidth=0.5+Math.random()*0.9;
    c.beginPath();
    c.moveTo(128+Math.cos(ang)*22,128+Math.sin(ang)*22);
    c.lineTo(128+Math.cos(ang)*(80+Math.random()*36),128+Math.sin(ang)*(80+Math.random()*36));
    c.stroke();
  }
  // Collarette ring
  c.strokeStyle='rgba(30,140,255,0.35)';c.lineWidth=2.5;
  c.beginPath();c.arc(128,128,38,0,Math.PI*2);c.stroke();
  for(var k=0;k<32;k++){
    var ka=(k/32)*Math.PI*2;
    c.strokeStyle='rgba(40,160,255,0.18)';c.lineWidth=0.7;
    c.beginPath();c.moveTo(128+Math.cos(ka)*28,128+Math.sin(ka)*28);
    c.lineTo(128+Math.cos(ka)*46,128+Math.sin(ka)*46);c.stroke();
  }
  // Crypts — irregular dark patches
  for(var cr=0;cr<10;cr++){
    var ca=Math.random()*Math.PI*2;
    var cr2=45+Math.random()*30;
    var gx=128+Math.cos(ca)*cr2,gy=128+Math.sin(ca)*cr2;
    var cg=c.createRadialGradient(gx,gy,0,gx,gy,7+Math.random()*5);
    cg.addColorStop(0,'rgba(0,10,40,0.5)');cg.addColorStop(1,'rgba(0,0,0,0)');
    c.fillStyle=cg;c.fillRect(gx-14,gy-14,28,28);
  }
  // Limbal ring
  var lg=c.createRadialGradient(128,128,85,128,128,118);
  lg.addColorStop(0,'rgba(0,0,0,0)');
  lg.addColorStop(0.65,'rgba(0,0,0,0.38)');
  lg.addColorStop(1,'rgba(0,0,0,0.85)');
  c.fillStyle=lg;c.fillRect(0,0,256,256);
  // Primary catchlight
  var sg=c.createRadialGradient(100,98,0,100,98,20);
  sg.addColorStop(0,'rgba(255,255,255,0.65)');
  sg.addColorStop(0.4,'rgba(180,230,255,0.25)');
  sg.addColorStop(1,'rgba(0,0,0,0)');
  c.fillStyle=sg;c.fillRect(0,0,256,256);
  // Secondary small highlight
  var sg2=c.createRadialGradient(156,152,0,156,152,7);
  sg2.addColorStop(0,'rgba(180,220,255,0.32)');sg2.addColorStop(1,'rgba(0,0,0,0)');
  c.fillStyle=sg2;c.fillRect(0,0,256,256);
  // Limbal ring — dark outer edge makes iris pop
  var lg=c.createRadialGradient(128,128,86,128,128,118);
  lg.addColorStop(0,'rgba(0,0,0,0)');
  lg.addColorStop(0.65,'rgba(0,0,0,0.38)');
  lg.addColorStop(1,'rgba(0,0,0,0.88)');
  c.fillStyle=lg; c.fillRect(0,0,256,256);
  // Pupil
  c.fillStyle='#000000';c.beginPath();c.arc(128,128,29,0,Math.PI*2);c.fill();
  // Pupil inner glow
  var pg=c.createRadialGradient(128,128,0,128,128,29);
  pg.addColorStop(0,'rgba(0,50,120,0.45)');pg.addColorStop(1,'rgba(0,0,0,0)');
  c.fillStyle=pg;c.fillRect(0,0,256,256);
  return new THREE.CanvasTexture(ic);
}
var IRIS_TEX = null; // lazy-init after Three.js ready

// ─── Geometry Helpers ─────────────────────────────────────────────────────────
// Gaussian bump: pushes vertices outward near (cx,cy) on sphere surface
function applyBump(geo, cx, cy, cz, radius, strength, axis) {
  // axis: outward direction of bump (along normal usually)
  var pos = geo.attributes.position;
  for (var i = 0; i < pos.count; i++) {
    var x = pos.getX(i), y = pos.getY(i), z = pos.getZ(i);
    var len = Math.sqrt(x*x+y*y+z*z);
    var nx = x/len, ny = y/len, nz = z/len;
    var dx = nx-cx, dy = ny-cy, dz = nz-cz;
    var d2 = dx*dx+dy*dy+dz*dz;
    var w = Math.exp(-d2 / (radius*radius)) * strength;
    pos.setXYZ(i, x+nx*w, y+ny*w, z+nz*w);
  }
  pos.needsUpdate = true;
  geo.computeVertexNormals();
}

// Gaussian dent: pushes vertices inward
function applyDent(geo, cx, cy, cz, radius, depth) {
  var pos = geo.attributes.position;
  for (var i = 0; i < pos.count; i++) {
    var x = pos.getX(i), y = pos.getY(i), z = pos.getZ(i);
    var len = Math.sqrt(x*x+y*y+z*z);
    var nx = x/len, ny = y/len, nz = z/len;
    var dx = nx-cx, dy = ny-cy, dz = nz-cz;
    var d2 = dx*dx+dy*dy+dz*dz;
    var w = Math.exp(-d2 / (radius*radius)) * depth;
    pos.setXYZ(i, x-nx*w, y-ny*w, z-nz*w);
  }
  pos.needsUpdate = true;
  geo.computeVertexNormals();
}

// ─── Head Group ───────────────────────────────────────────────────────────────
var headGroup = new THREE.Group();
scene.add(headGroup);

// ─── CRANIUM — sculpted with realistic human head proportions ─────────────────
// Reference: lean male head — taller than wide, flat-ish dome, angular jaw,
// prominent cheekbones, narrow from front view
var cranGeo = new THREE.SphereGeometry(1.0, 128, 96);
var cranPos = cranGeo.attributes.position;
for (var i = 0; i < cranPos.count; i++) {
  var x = cranPos.getX(i), y = cranPos.getY(i), z = cranPos.getZ(i);
  var t = (y + 1.0) * 0.5; // 0=chin 1=crown
  var len = Math.sqrt(x*x + y*y + z*z);
  if (len < 0.001) continue;
  var nx0 = x/len, ny0 = y/len, nz0 = z/len;

  // ── Width profile (x-axis) ──────────────────────────────────────────
  // Human head: narrow overall, widest at parietal eminence (~t=0.68)
  var wScale;
  if (t > 0.82) {
    // Top of skull: narrows smoothly toward vertex
    wScale = 0.72 - (t - 0.82) * 0.55;
  } else if (t > 0.65) {
    // Parietal region: widest part of skull
    wScale = 0.76 - (t - 0.72) * 0.12;
  } else if (t > 0.52) {
    // Temporal region
    wScale = 0.74;
  } else if (t > 0.40) {
    // Zygomatic (cheekbone) region — widest part of face
    wScale = 0.73;
  } else if (t > 0.26) {
    // Jaw region: angular with defined gonial angle
    var jawT = (t - 0.26) / 0.14;
    wScale = 0.58 + jawT * 0.14;
    // Gonial angle bump at t ~ 0.32
    var gonialBump = Math.exp(-((t - 0.32)*(t - 0.32)) / 0.003);
    wScale += gonialBump * 0.08;
  } else if (t > 0.12) {
    // Lower jaw taper toward chin — gradual, not sharp
    wScale = 0.48 + (t - 0.12) * 0.85;
  } else {
    // Chin: rounded, not pointy — human chin is fairly wide
    wScale = 0.42 + t * 0.50;
  }

  // ── Depth profile (z-axis) ──────────────────────────────────────────
  // Face is flatter front-to-back at sides, deeper in center
  var dScale = 0.82;
  // Back of head rounds out more
  if (z < 0) dScale = 0.86;
  // Face protrusion in midface region
  if (z > 0 && t > 0.15 && t < 0.72) {
    // Subtle face protrusion — just enough for nose/mouth area
    var facePush = Math.exp(-(nx0*nx0)/0.10) * 0.04;
    dScale += facePush;
  }
  // Flatten the sides
  dScale -= Math.abs(x) * 0.03;

  // ── Height profile ──────────────────────────────────────────────────
  // Taller head for lean masculine proportions
  var hScale = 1.08;
  // Flatten the dome slightly at very top
  if (t > 0.88) {
    hScale = 1.08 - (t - 0.88) * 0.30;
  }

  cranPos.setXYZ(i, x * wScale, y * hScale, z * dScale);
}
cranPos.needsUpdate = true;
cranGeo.computeVertexNormals();

// Sculpt detailed facial features via per-vertex Gaussian displacement
// Operates on ALL vertices (front, side, and back) for seamless shaping
(function sculptHead() {
  var pos = cranGeo.attributes.position;
  for (var i = 0; i < pos.count; i++) {
    var x = pos.getX(i), y = pos.getY(i), z = pos.getZ(i);
    var len = Math.sqrt(x*x + y*y + z*z);
    if (len < 0.01) continue;
    var nx = x/len, ny = y/len, nz = z/len;

    var dz = 0, dy = 0, dx = 0;

    // ══════════════════════════════════════════════════════════════════
    //  SIDE + BACK sculpting (works everywhere)
    // ══════════════════════════════════════════════════════════════════

    // ── TEMPORAL HOLLOWS — sides of forehead ─────────────────────────
    var tempL = Math.exp(-((nx+0.72)*(nx+0.72))/0.020) * Math.exp(-((ny-0.15)*(ny-0.15))/0.028) * 0.025;
    var tempR = Math.exp(-((nx-0.72)*(nx-0.72))/0.020) * Math.exp(-((ny-0.15)*(ny-0.15))/0.028) * 0.025;
    dx += tempL * 0.5;
    dx -= tempR * 0.5;
    dz -= (tempL + tempR) * 0.3;

    // ── MASTOID PROCESS — bony bump behind ears ──────────────────────
    dx -= Math.exp(-((nx+0.75)*(nx+0.75))/0.012) * Math.exp(-((ny+0.25)*(ny+0.25))/0.012) * Math.exp(-((nz+0.30)*(nz+0.30))/0.015) * 0.020;
    dx += Math.exp(-((nx-0.75)*(nx-0.75))/0.012) * Math.exp(-((ny+0.25)*(ny+0.25))/0.012) * Math.exp(-((nz+0.30)*(nz+0.30))/0.015) * 0.020;

    // ── OCCIPITAL BULGE — back of skull ──────────────────────────────
    dz -= Math.exp(-(nx*nx)/0.06) * Math.exp(-((ny+0.10)*(ny+0.10))/0.030) * Math.exp(-((nz+0.85)*(nz+0.85))/0.015) * 0.035;

    // ── JAW ANGLE — lateral mandibular angle (gonion) ────────────────
    var jaL = Math.exp(-((nx+0.68)*(nx+0.68))/0.012) * Math.exp(-((ny+0.38)*(ny+0.38))/0.010);
    var jaR = Math.exp(-((nx-0.68)*(nx-0.68))/0.012) * Math.exp(-((ny+0.38)*(ny+0.38))/0.010);
    dx -= jaL * 0.040;
    dx += jaR * 0.040;
    dy -= (jaL + jaR) * 0.015;

    // ── JAWLINE RIDGE — sharp edge from gonion to chin ───────────────
    var jawLineY = -0.50;
    var jawLineWidth = 0.008;
    var jlL = Math.exp(-((ny-jawLineY)*(ny-jawLineY))/jawLineWidth) * Math.exp(-((nx+0.45)*(nx+0.45))/0.030) * (nz > -0.2 ? 1 : 0);
    var jlR = Math.exp(-((ny-jawLineY)*(ny-jawLineY))/jawLineWidth) * Math.exp(-((nx-0.45)*(nx-0.45))/0.030) * (nz > -0.2 ? 1 : 0);
    dz += (jlL + jlR) * 0.028;
    dy -= (jlL + jlR) * 0.022;

    // ══════════════════════════════════════════════════════════════════
    //  FRONT FACE sculpting (only front hemisphere)
    // ══════════════════════════════════════════════════════════════════
    if (nz > 0.05) {

      // ── BROW RIDGE ──────────────────────────────────────────────────
      var brW = Math.exp(-((ny-0.22)*(ny-0.22))/0.014) * Math.exp(-((nz-0.84)*(nz-0.84))/0.009) * Math.exp(-(nx*nx)/0.45);
      dz += brW * 0.055;
      dz += Math.exp(-((nx+0.30)*(nx+0.30))/0.015) * Math.exp(-((ny-0.24)*(ny-0.24))/0.010) * Math.exp(-((nz-0.86)*(nz-0.86))/0.008) * 0.038;
      dz += Math.exp(-((nx-0.30)*(nx-0.30))/0.015) * Math.exp(-((ny-0.24)*(ny-0.24))/0.010) * Math.exp(-((nz-0.86)*(nz-0.86))/0.008) * 0.038;
      // Glabella
      dz += Math.exp(-(nx*nx)/0.006) * Math.exp(-((ny-0.22)*(ny-0.22))/0.008) * Math.exp(-((nz-0.88)*(nz-0.88))/0.010) * 0.025;

      // ── EYE SOCKETS — deep orbital cavities ─────────────────────────
      dz -= Math.exp(-((nx+0.30)*(nx+0.30))/0.013) * Math.exp(-((ny-0.10)*(ny-0.10))/0.010) * Math.exp(-((nz-0.82)*(nz-0.82))/0.014) * 0.048;
      dz -= Math.exp(-((nx-0.30)*(nx-0.30))/0.013) * Math.exp(-((ny-0.10)*(ny-0.10))/0.010) * Math.exp(-((nz-0.82)*(nz-0.82))/0.014) * 0.048;

      // ── CHEEKBONES — malar eminence ─────────────────────────────────
      dz += Math.exp(-((nx+0.50)*(nx+0.50))/0.018) * Math.exp(-(ny*ny)/0.014) * Math.exp(-((nz-0.65)*(nz-0.65))/0.014) * 0.065;
      dz += Math.exp(-((nx-0.50)*(nx-0.50))/0.018) * Math.exp(-(ny*ny)/0.014) * Math.exp(-((nz-0.65)*(nz-0.65))/0.014) * 0.065;
      // Zygomatic arch
      dz += Math.exp(-((nx+0.62)*(nx+0.62))/0.016) * Math.exp(-((ny-0.02)*(ny-0.02))/0.012) * Math.exp(-((nz-0.52)*(nz-0.52))/0.014) * 0.058;
      dz += Math.exp(-((nx-0.62)*(nx-0.62))/0.016) * Math.exp(-((ny-0.02)*(ny-0.02))/0.012) * Math.exp(-((nz-0.52)*(nz-0.52))/0.014) * 0.058;

      // ── CHEEK HOLLOWS ───────────────────────────────────────────────
      dz -= Math.exp(-((nx+0.40)*(nx+0.40))/0.016) * Math.exp(-((ny+0.16)*(ny+0.16))/0.014) * Math.exp(-((nz-0.72)*(nz-0.72))/0.014) * 0.028;
      dz -= Math.exp(-((nx-0.40)*(nx-0.40))/0.016) * Math.exp(-((ny+0.16)*(ny+0.16))/0.014) * Math.exp(-((nz-0.72)*(nz-0.72))/0.014) * 0.028;

      // ── INFRAORBITAL — maxilla plane below eyes ─────────────────────
      dz += Math.exp(-((nx+0.22)*(nx+0.22))/0.012) * Math.exp(-((ny+0.04)*(ny+0.04))/0.008) * Math.exp(-((nz-0.80)*(nz-0.80))/0.010) * 0.030;
      dz += Math.exp(-((nx-0.22)*(nx-0.22))/0.012) * Math.exp(-((ny+0.04)*(ny+0.04))/0.008) * Math.exp(-((nz-0.80)*(nz-0.80))/0.010) * 0.030;

      // ── NOSE BRIDGE ─────────────────────────────────────────────────
      var nbW = Math.exp(-(nx*nx)/0.003) * Math.exp(-((ny-0.02)*(ny-0.02))/0.050) * Math.exp(-((nz-0.86)*(nz-0.86))/0.014) * (nz > 0.72 ? 1 : 0);
      dz += nbW * 0.085;
      dz += Math.exp(-(nx*nx)/0.002) * Math.exp(-((ny+0.10)*(ny+0.10))/0.015) * Math.exp(-((nz-0.88)*(nz-0.88))/0.010) * 0.045;

      // ── NOSE TIP ────────────────────────────────────────────────────
      var ntW = Math.exp(-(nx*nx)/0.009) * Math.exp(-((ny+0.26)*(ny+0.26))/0.008) * Math.exp(-((nz-0.88)*(nz-0.88))/0.008);
      dz += ntW * 0.100;
      dy -= ntW * 0.028;
      // Nostril wings
      dz += Math.exp(-((nx+0.10)*(nx+0.10))/0.005) * Math.exp(-((ny+0.30)*(ny+0.30))/0.006) * Math.exp(-((nz-0.85)*(nz-0.85))/0.006) * 0.072;
      dz += Math.exp(-((nx-0.10)*(nx-0.10))/0.005) * Math.exp(-((ny+0.30)*(ny+0.30))/0.006) * Math.exp(-((nz-0.85)*(nz-0.85))/0.006) * 0.072;
      // Alar crease
      dz -= Math.exp(-((nx+0.14)*(nx+0.14))/0.004) * Math.exp(-((ny+0.28)*(ny+0.28))/0.005) * Math.exp(-((nz-0.84)*(nz-0.84))/0.006) * 0.022;
      dz -= Math.exp(-((nx-0.14)*(nx-0.14))/0.004) * Math.exp(-((ny+0.28)*(ny+0.28))/0.005) * Math.exp(-((nz-0.84)*(nz-0.84))/0.006) * 0.022;

      // ── PHILTRUM ────────────────────────────────────────────────────
      dz -= Math.exp(-(nx*nx)/0.0025) * Math.exp(-((ny+0.38)*(ny+0.38))/0.004) * Math.exp(-((nz-0.84)*(nz-0.84))/0.007) * 0.038;
      dz += Math.exp(-((nx+0.03)*(nx+0.03))/0.001) * Math.exp(-((ny+0.36)*(ny+0.36))/0.005) * Math.exp(-((nz-0.84)*(nz-0.84))/0.006) * 0.015;
      dz += Math.exp(-((nx-0.03)*(nx-0.03))/0.001) * Math.exp(-((ny+0.36)*(ny+0.36))/0.005) * Math.exp(-((nz-0.84)*(nz-0.84))/0.006) * 0.015;

      // ── UPPER LIP ──────────────────────────────────────────────────
      dz += Math.exp(-(nx*nx)/0.040) * Math.exp(-((ny+0.45)*(ny+0.45))/0.004) * Math.exp(-((nz-0.82)*(nz-0.82))/0.008) * 0.088;
      dz += Math.exp(-((nx+0.07)*(nx+0.07))/0.003) * Math.exp(-((ny+0.43)*(ny+0.43))/0.004) * Math.exp(-((nz-0.84)*(nz-0.84))/0.006) * 0.045;
      dz += Math.exp(-((nx-0.07)*(nx-0.07))/0.003) * Math.exp(-((ny+0.43)*(ny+0.43))/0.004) * Math.exp(-((nz-0.84)*(nz-0.84))/0.006) * 0.045;

      // ── NASOLABIAL FOLDS ────────────────────────────────────────────
      dz += Math.exp(-((nx+0.22)*(nx+0.22))/0.005) * Math.exp(-((ny+0.18)*(ny+0.18))/0.024) * Math.exp(-((nz-0.78)*(nz-0.78))/0.011) * 0.022;
      dz += Math.exp(-((nx-0.22)*(nx-0.22))/0.005) * Math.exp(-((ny+0.18)*(ny+0.18))/0.024) * Math.exp(-((nz-0.78)*(nz-0.78))/0.011) * 0.022;

      // ── CHIN ────────────────────────────────────────────────────────
      var chW = Math.exp(-(nx*nx)/0.030) * Math.exp(-((ny+0.74)*(ny+0.74))/0.011) * Math.exp(-((nz-0.68)*(nz-0.68))/0.014);
      dz += chW * 0.075;
      dy -= chW * 0.035;
      dz += Math.exp(-(nx*nx)/0.022) * Math.exp(-((ny+0.70)*(ny+0.70))/0.007) * Math.exp(-((nz-0.72)*(nz-0.72))/0.012) * 0.030;
      dz -= Math.exp(-(nx*nx)/0.0012) * Math.exp(-((ny+0.75)*(ny+0.75))/0.005) * Math.exp(-((nz-0.71)*(nz-0.71))/0.007) * 0.014;
      // Labiomental groove
      dz -= Math.exp(-(nx*nx)/0.018) * Math.exp(-((ny+0.62)*(ny+0.62))/0.004) * Math.exp(-((nz-0.78)*(nz-0.78))/0.008) * 0.025;

      // ── FOREHEAD ────────────────────────────────────────────────────
      dz += Math.exp(-(nx*nx)/0.10) * Math.exp(-((ny-0.50)*(ny-0.50))/0.018) * Math.exp(-((nz-0.80)*(nz-0.80))/0.010) * 0.035;
      dz += Math.exp(-((nx+0.18)*(nx+0.18))/0.010) * Math.exp(-((ny-0.52)*(ny-0.52))/0.012) * Math.exp(-((nz-0.78)*(nz-0.78))/0.010) * 0.018;
      dz += Math.exp(-((nx-0.18)*(nx-0.18))/0.010) * Math.exp(-((ny-0.52)*(ny-0.52))/0.012) * Math.exp(-((nz-0.78)*(nz-0.78))/0.010) * 0.018;

    } // end front-face sculpting

    pos.setXYZ(i, x + nx*dz + dx, y + ny*dz + dy, z + nz*dz);
  }
  pos.needsUpdate = true;
  cranGeo.computeVertexNormals();
})();

// ─── IMPROVEMENT 5: Facial Morph Targets ─────────────────────────────────────
function buildMorphTarget(baseGeo, displace) {
  var clone = baseGeo.attributes.position.clone();
  displace(clone);
  return clone;
}

// Morph 0: Raise brows — listening (alert, attentive)
var morphBrowRaise = buildMorphTarget(cranGeo, function(pos) {
  for (var i = 0; i < pos.count; i++) {
    var x=pos.getX(i),y=pos.getY(i),z=pos.getZ(i);
    var len=Math.sqrt(x*x+y*y+z*z); if(len<0.01)continue;
    var nx=x/len,ny=y/len,nz=z/len;
    var w=Math.exp(-((ny-0.26)*(ny-0.26))/0.006)*(nz>0.55?1:0)*Math.exp(-(nx*nx)/0.40);
    pos.setY(i,y+w*0.072);pos.setZ(i,z+w*0.015);
  }
});

// Morph 1: Furrow brows — processing (concentrated)
var morphBrowFurrow = buildMorphTarget(cranGeo, function(pos) {
  for (var i = 0; i < pos.count; i++) {
    var x=pos.getX(i),y=pos.getY(i),z=pos.getZ(i);
    var len=Math.sqrt(x*x+y*y+z*z); if(len<0.01)continue;
    var nx=x/len,ny=y/len,nz=z/len;
    var w=Math.exp(-((ny-0.22)*(ny-0.22))/0.004)*(nz>0.60?1:0)*Math.exp(-(nx*nx)/0.30);
    pos.setY(i,y-w*0.038);pos.setZ(i,z+w*0.025);
    var gw=Math.exp(-(nx*nx)/0.004)*Math.exp(-((ny-0.20)*(ny-0.20))/0.003)*(nz>0.68?1:0);
    pos.setZ(i,pos.getZ(i)+gw*0.020);
  }
});

// Morph 2: Squint — speaking (intensity)
var morphSquint = buildMorphTarget(cranGeo, function(pos) {
  for (var i = 0; i < pos.count; i++) {
    var x=pos.getX(i),y=pos.getY(i),z=pos.getZ(i);
    var len=Math.sqrt(x*x+y*y+z*z); if(len<0.01)continue;
    var nx=x/len,ny=y/len,nz=z/len;
    var lw=Math.exp(-((ny-0.10)*(ny-0.10))/0.006)*(nz>0.72?1:0);
    pos.setY(i,y-lw*0.022);
    var chw=Math.exp(-((ny+0.05)*(ny+0.05))/0.010)*(nz>0.62?1:0)*Math.exp(-((Math.abs(nx)-0.45)*(Math.abs(nx)-0.45))/0.018);
    pos.setY(i,y+chw*0.028);
  }
});

cranGeo.morphAttributes.position = [morphBrowRaise, morphBrowFurrow, morphSquint];

var skinMat = regMat(mkMat({ base: 0x010e2a, glow: 0x00aaff, opacity: 0.84, scan: 0.60 }));
var cranMesh = new THREE.Mesh(cranGeo, skinMat);
cranMesh.morphTargetInfluences = [0, 0, 0];
headGroup.add(cranMesh);

// ─── IMPROVEMENT 3: Subsurface Scattering (inner glow shell) ─────────────────
// Slightly larger backface mesh gives skin a translucent volumetric radiance
var sssMat = new THREE.MeshBasicMaterial({
  color: 0x00aaff, transparent: true, opacity: 0.055,
  side: THREE.BackSide, depthWrite: false
});
var sssMesh = new THREE.Mesh(cranGeo.clone(), sssMat);
sssMesh.scale.setScalar(1.048);
headGroup.add(sssMesh);
// A second tighter shell for the bright edge silhouette
var sssEdgeMat = new THREE.MeshBasicMaterial({
  color: 0x00d4ff, transparent: true, opacity: 0.030,
  side: THREE.BackSide, depthWrite: false
});
var sssEdgeMesh = new THREE.Mesh(cranGeo.clone(), sssEdgeMat);
sssEdgeMesh.scale.setScalar(1.022);
headGroup.add(sssEdgeMesh);

var wireMat = new THREE.MeshBasicMaterial({ color: 0x00c8ff, wireframe: true, transparent: true, opacity: 0.040, depthWrite: false });
var wireMesh = new THREE.Mesh(cranGeo.clone(), wireMat);
headGroup.add(wireMesh);

// ─── JAW — separate geometry matching lean head proportions ──────────────────
// Narrower, more angular mandible for the taller, leaner cranium
var jawMat = regMat(mkMat({ base: 0x010e2a, glow: 0x00aaff, opacity: 0.82, scan: 0.55 }));
var jawPivot = new THREE.Group();
jawPivot.position.set(0, -0.76, 0); // pivot at jaw hinge — lower for taller head
headGroup.add(jawPivot);

// Lower jaw body — angular mandible
var jawGeo = new THREE.SphereGeometry(1.0, 80, 48, 0, Math.PI*2, Math.PI*0.44, Math.PI*0.40);
var jawPos2 = jawGeo.attributes.position;
for (var j = 0; j < jawPos2.count; j++) {
  var jx = jawPos2.getX(j), jy = jawPos2.getY(j), jz = jawPos2.getZ(j);
  var jt = (jy + 1.2) / 1.2;
  // Human jaw: wide at angles, tapers to rounded chin
  var jwx;
  if (jt < 0.30) {
    jwx = 0.82 - jt * 0.08;
  } else if (jt < 0.55) {
    jwx = 0.80 - (jt - 0.30) * 0.35;
    var sqFactor = 1.0 - Math.abs(jt - 0.42) / 0.12;
    if (sqFactor > 0) jwx += sqFactor * 0.04;
  } else {
    jwx = 0.71 - jt * 0.18;
  }
  jawPos2.setXYZ(j, jx * jwx, jy * 0.56, jz * 0.78);
}
jawPos2.needsUpdate = true;
jawGeo.computeVertexNormals();

// Sculpt jaw: chin, angular mandible
(function sculptJaw() {
  var pos = jawGeo.attributes.position;
  for (var i = 0; i < pos.count; i++) {
    var x = pos.getX(i), y = pos.getY(i), z = pos.getZ(i);
    var len = Math.sqrt(x*x+y*y+z*z); if (len < 0.01) continue;
    var nx = x/len, ny = y/len, nz = z/len;
    if (nz < 0.08) continue;
    var chW = Math.exp(-(nx*nx)/0.028) * Math.exp(-((ny+0.55)*(ny+0.55))/0.009) * Math.exp(-((nz-0.70)*(nz-0.70))/0.013);
    pos.setXYZ(i, x + nx*chW*0.13, y + ny*chW*0.08 - chW*0.05, z + nz*chW*0.13);
    var jaL = Math.exp(-((nx+0.65)*(nx+0.65))/0.013) * Math.exp(-((ny+0.18)*(ny+0.18))/0.011) * Math.exp(-((nz-0.45)*(nz-0.45))/0.015);
    var jaR = Math.exp(-((nx-0.65)*(nx-0.65))/0.013) * Math.exp(-((ny+0.18)*(ny+0.18))/0.011) * Math.exp(-((nz-0.45)*(nz-0.45))/0.015);
    pos.setX(i, pos.getX(i) + (jaL * -0.038 + jaR * 0.038));
    pos.setZ(i, pos.getZ(i) + (jaL + jaR) * 0.022);
  }
  pos.needsUpdate = true;
  jawGeo.computeVertexNormals();
})();

var jawMesh = new THREE.Mesh(jawGeo, jawMat);
jawMesh.position.set(0, 0.76, 0);
jawPivot.add(jawMesh);

// ─── NECK ─────────────────────────────────────────────────────────────────────
var neckGeo = new THREE.CylinderGeometry(0.28, 0.35, 0.76, 40);
var neckMat = regMat(mkMat({ base: 0x010820, glow: 0x0088cc, opacity: 0.62, scan: 0.40 }));
var neckMesh = new THREE.Mesh(neckGeo, neckMat);
neckMesh.position.set(0, -1.28, -0.04);
headGroup.add(neckMesh);

// ─── EYES — with proper blinking eyelids that cover the eye ──────────────────
var eyeData = [
  { side: -1, x: -0.252, y: 0.10, z: 0.775 },
  { side:  1, x:  0.252, y: 0.10, z: 0.775 }
];
var eyeObjects = [];
eyeData.forEach(function(e) {
  var g = new THREE.Group();

  // Eyeball
  var scleraMat = new THREE.MeshPhongMaterial({ color: 0x021848, emissive: 0x004060, emissiveIntensity: 0.7, specular: 0x00bbff, shininess: 140, transparent: true, opacity: 0.94 });
  var sclera = new THREE.Mesh(new THREE.SphereGeometry(0.120, 36, 28), scleraMat);
  g.add(sclera);

  // Iris with PROCEDURAL TEXTURE (Improvement 4) + Corneal Env Map (Improvement 5)
  if (!IRIS_TEX) IRIS_TEX = makeIrisTexture();
  var irisMat = new THREE.MeshPhongMaterial({
    map: IRIS_TEX,
    color: 0x003388, emissive: 0x002255, emissiveIntensity: 0.85,
    specular: 0x00eeff, shininess: 300,
    transparent: true, opacity: 0.97
  });
  var iris = new THREE.Mesh(new THREE.CylinderGeometry(0.072, 0.072, 0.012, 44), irisMat);
  iris.rotation.x = Math.PI / 2; iris.position.z = 0.100; g.add(iris);
  var irisInner = new THREE.Mesh(new THREE.CylinderGeometry(0.050, 0.053, 0.013, 36),
    new THREE.MeshPhongMaterial({ color: 0x0066cc, emissive: 0x0044aa, emissiveIntensity: 1.6, specular: 0x00eeff, shininess: 250, transparent: true, opacity: 0.90 }));
  irisInner.rotation.x = Math.PI / 2; irisInner.position.z = 0.101; g.add(irisInner);

  // Pupil
  var pupil = new THREE.Mesh(new THREE.CylinderGeometry(0.028, 0.028, 0.008, 32),
    new THREE.MeshPhongMaterial({ color: 0x000511, emissive: 0x001830, emissiveIntensity: 0.8, transparent: true, opacity: 0.98 }));
  pupil.rotation.x = Math.PI / 2; pupil.position.z = 0.104; g.add(pupil);

  // Glow ring
  var glowRingMat = new THREE.MeshBasicMaterial({ color: 0x0099ff, transparent: true, opacity: 0.30 });
  var glowRing = new THREE.Mesh(new THREE.RingGeometry(0.070, 0.082, 44), glowRingMat);
  glowRing.position.z = 0.106; g.add(glowRing);

  // ── EYELID SYSTEM — upper lid sweeps down to cover eye ─────────────
  var lidMat = regMat(mkMat({ base: 0x011230, glow: 0x00aadd, opacity: 0.96, scan: 0.40 }));

  // Upper lid — a curved shell that sits above the eye opening
  // When blinking: it translates DOWN to cover the iris/pupil area
  var upperLidGeo = new THREE.SphereGeometry(0.135, 32, 18, 0, Math.PI*2, 0, Math.PI*0.42);
  var upperLidP = upperLidGeo.attributes.position;
  for (var li = 0; li < upperLidP.count; li++) {
    var lx = upperLidP.getX(li), ly = upperLidP.getY(li), lz = upperLidP.getZ(li);
    upperLidP.setX(li, lx * 1.30); // wider horizontally (almond)
    upperLidP.setY(li, ly * 0.68); // compressed vertically
  }
  upperLidP.needsUpdate = true; upperLidGeo.computeVertexNormals();
  var upperLid = new THREE.Mesh(upperLidGeo, lidMat);
  upperLid.position.set(0, 0.025, 0); // sits just above center
  g.add(upperLid);

  // Lower lid — smaller shell sitting below the eye opening
  var lowerLidGeo = new THREE.SphereGeometry(0.132, 28, 12, 0, Math.PI*2, Math.PI*0.58, Math.PI*0.24);
  var lowerLidP = lowerLidGeo.attributes.position;
  for (var lli = 0; lli < lowerLidP.count; lli++) {
    lowerLidP.setX(lli, lowerLidP.getX(lli) * 1.25);
    lowerLidP.setY(lli, lowerLidP.getY(lli) * 0.75);
  }
  lowerLidP.needsUpdate = true; lowerLidGeo.computeVertexNormals();
  var lowerLid = new THREE.Mesh(lowerLidGeo, lidMat);
  lowerLid.position.set(0, -0.020, 0);
  g.add(lowerLid);

  // Blink shutter — a flat opaque panel that slides down during blink
  // This HIDES the eyeball completely when closed (prevents the "round eye" showing through)
  var shutterGeo = new THREE.PlaneGeometry(0.20, 0.14, 1, 1);
  var shutterMat = new THREE.ShaderMaterial({
    uniforms: {
      uBase: { value: new THREE.Color(0x011230) },
      uGlow: { value: new THREE.Color(0x00aadd) },
      uOpacity: { value: 0.0 },
      uTime: { value: 0 },
      uPulse: { value: 0 },
      uScan: { value: 0.35 }
    },
    vertexShader: holoVS,
    fragmentShader: holoFS,
    transparent: true,
    side: THREE.FrontSide,
    depthWrite: true
  });
  ALL_MATS.push(shutterMat);
  var shutter = new THREE.Mesh(shutterGeo, shutterMat);
  shutter.position.set(0, 0.08, 0.095); // above iris, hidden
  shutter.renderOrder = 1; // render on top
  g.add(shutter);

  // Eyelid crease — subtle fold line above the upper lid
  var creasePath = new THREE.CatmullRomCurve3([
    new THREE.Vector3(-0.13, 0.065, 0.035),
    new THREE.Vector3(-0.07, 0.088, 0.050),
    new THREE.Vector3( 0.00, 0.095, 0.055),
    new THREE.Vector3( 0.07, 0.088, 0.050),
    new THREE.Vector3( 0.13, 0.065, 0.035)
  ], false);
  g.add(new THREE.Mesh(new THREE.TubeGeometry(creasePath, 20, 0.008, 6, false), lidMat));

  // Orbital rim
  var orbitalPath = new THREE.CatmullRomCurve3([
    new THREE.Vector3(-0.15, 0.02, 0.005),
    new THREE.Vector3(-0.11, 0.08, 0.025),
    new THREE.Vector3( 0.00, 0.12, 0.035),
    new THREE.Vector3( 0.11, 0.08, 0.025),
    new THREE.Vector3( 0.15, 0.02, 0.005),
    new THREE.Vector3( 0.13,-0.05,-0.005),
    new THREE.Vector3( 0.00,-0.10,-0.010),
    new THREE.Vector3(-0.13,-0.05,-0.005),
    new THREE.Vector3(-0.15, 0.02, 0.005)
  ], false);
  var orbMat = regMat(mkMat({ base: 0x010c28, glow: 0x0088bb, opacity: 0.70, scan: 0.28 }));
  g.add(new THREE.Mesh(new THREE.TubeGeometry(orbitalPath, 32, 0.010, 6, false), orbMat));

  g.position.set(e.x, e.y, e.z);
  g.rotation.y = e.side * 0.05;
  headGroup.add(g);
  eyeObjects.push({
    group: g, upperLid: upperLid, lowerLid: lowerLid, shutter: shutter,
    shutterMat: shutterMat, lidMat: lidMat, irisMat: irisMat, glowRing: glowRing
  });
});
// Note: applyEnvMapToEyes() is called inside initPostFX after cubeRenderTarget is ready

// ─── EYEBROWS — thick, prominent, clearly visible human brows ────────────────
var browObjects = [];
[-1, 1].forEach(function(side) {
  // Much thicker eyebrow shape — visible against holographic skin
  var shape = new THREE.Shape();
  shape.moveTo(-0.148, 0);
  shape.quadraticCurveTo(-0.07, 0.056, 0, 0.048);
  shape.quadraticCurveTo(0.07,  0.038, 0.148, 0.008);
  shape.quadraticCurveTo(0.07, -0.010, 0, -0.014);
  shape.quadraticCurveTo(-0.07, -0.016, -0.148, 0);
  var extCfg = { depth: 0.020, bevelEnabled: true, bevelSize: 0.007, bevelThickness: 0.005, bevelSegments: 5 };
  var browGeo = new THREE.ExtrudeGeometry(shape, extCfg);
  // Much brighter material so brows are clearly visible
  var browMat = new THREE.MeshPhongMaterial({
    color: 0x021e48, emissive: 0x003366, emissiveIntensity: 1.6,
    specular: 0x4488cc, shininess: 100, transparent: true, opacity: 0.95
  });
  var brow = new THREE.Mesh(browGeo, browMat);
  brow.position.set(side * 0.252, 0.255, 0.775);
  brow.rotation.y = side * 0.07;
  brow.rotation.z = side * -0.055;
  brow.scale.x = side < 0 ? 1 : -1;
  headGroup.add(brow);

  // Secondary brow ridge below — adds volume/shadow
  var ridgeShape = new THREE.Shape();
  ridgeShape.moveTo(-0.140, 0);
  ridgeShape.quadraticCurveTo(0, 0.032, 0.140, 0);
  ridgeShape.quadraticCurveTo(0, -0.008, -0.140, 0);
  var ridgeGeo = new THREE.ExtrudeGeometry(ridgeShape, { depth: 0.015, bevelEnabled: true, bevelSize: 0.006, bevelThickness: 0.004, bevelSegments: 4 });
  var ridge = new THREE.Mesh(ridgeGeo, browMat);
  ridge.position.set(side * 0.252, 0.238, 0.780);
  ridge.rotation.y = side * 0.07;
  ridge.rotation.z = side * -0.055;
  ridge.scale.x = side < 0 ? 1 : -1;
  headGroup.add(ridge);

  browObjects.push({ mesh: brow, mat: browMat });
});

// ═══════════════════════════════════════════════════════════════════════════
// MOUTH — anatomically correct two-section human lips
// Built from tube paths so each lip has real 3D rounded volume.
// Upper lip: M-shaped cupid's bow. Lower lip: fuller U-dome.
// Both rendered as thick tubes so the cross-section is circular (rounded).
// Lip positions are in world space; lower lip tubes added to jawPivot
// with compensated Y so they align with upper when jaw is closed.
// ═══════════════════════════════════════════════════════════════════════════

// ── Materials ────────────────────────────────────────────────────────────────
// Lip body — bright emissive so clearly distinct from holographic skin
var lipBodyMat = new THREE.MeshPhongMaterial({
  color: 0x032050, emissive: 0x0055cc, emissiveIntensity: 2.2,
  specular: 0x99ddff, shininess: 200, transparent: true, opacity: 0.97
});
// Seam / dark gap between lips
var lipSeamMat = new THREE.MeshPhongMaterial({
  color: 0x000612, emissive: 0x000818, emissiveIntensity: 0.5,
  transparent: true, opacity: 0.98
});
// Vermilion border (the bright outline ridge)
var vermMat = new THREE.MeshPhongMaterial({
  color: 0x044070, emissive: 0x0077ee, emissiveIntensity: 2.6,
  specular: 0xaaddff, shininess: 240, transparent: true, opacity: 0.98
});

// World position of the mouth centre (where lips meet = seam line)
// Y=-0.484 puts seam between upper & lower. Z=0.806 sits proud of face.
var MX = 0, MY = -0.484, MZ = 0.806;
// Jaw pivot is at Y=-0.76 world. Lower lip local Y = world Y - pivotY
var PIVOT_Y = -0.76;

// ── Helper: build a tube mesh along a CatmullRom path ───────────────────────
function makeTube(pts, radius, segs, radSegs, mat) {
  var curve = new THREE.CatmullRomCurve3(pts, false, 'catmullrom', 0.5);
  var geo = new THREE.TubeGeometry(curve, segs, radius, radSegs, false);
  return new THREE.Mesh(geo, mat);
}

// ══════════════════════════════════════════════════════════════════════════════
// UPPER LIP — three tube paths that together form the cupid's bow
// Path 1: LEFT LOBE — sweeps from left corner up to left peak
// Path 2: RIGHT LOBE — sweeps from right corner up to right peak  
// Path 3: CENTER ARCH — the two peaks connected through the philtrum dip
// Plus a base fill tube along the bottom vermilion border
// ══════════════════════════════════════════════════════════════════════════════

// Left outer body lobe — fat tube from corner curving up to left peak
var ulLeftLobe = makeTube([
  new THREE.Vector3(MX-0.155, MY+0.000, MZ-0.012), // left corner
  new THREE.Vector3(MX-0.130, MY+0.010, MZ+0.000),
  new THREE.Vector3(MX-0.095, MY+0.025, MZ+0.010),
  new THREE.Vector3(MX-0.068, MY+0.046, MZ+0.020), // left peak top
  new THREE.Vector3(MX-0.045, MY+0.052, MZ+0.022),
], 0.026, 22, 10, lipBodyMat);
headGroup.add(ulLeftLobe);

// Right outer body lobe — mirrored
var ulRightLobe = makeTube([
  new THREE.Vector3(MX+0.155, MY+0.000, MZ-0.012),
  new THREE.Vector3(MX+0.130, MY+0.010, MZ+0.000),
  new THREE.Vector3(MX+0.095, MY+0.025, MZ+0.010),
  new THREE.Vector3(MX+0.068, MY+0.046, MZ+0.020),
  new THREE.Vector3(MX+0.045, MY+0.052, MZ+0.022),
], 0.026, 22, 10, lipBodyMat);
headGroup.add(ulRightLobe);

// Center section — left peak → philtrum dip → right peak
var ulCenter = makeTube([
  new THREE.Vector3(MX-0.045, MY+0.052, MZ+0.022), // left peak
  new THREE.Vector3(MX-0.022, MY+0.044, MZ+0.022),
  new THREE.Vector3(MX-0.008, MY+0.036, MZ+0.020), // philtrum dip
  new THREE.Vector3(MX+0.000, MY+0.033, MZ+0.020),
  new THREE.Vector3(MX+0.008, MY+0.036, MZ+0.020),
  new THREE.Vector3(MX+0.022, MY+0.044, MZ+0.022),
  new THREE.Vector3(MX+0.045, MY+0.052, MZ+0.022), // right peak
], 0.023, 30, 10, lipBodyMat);
headGroup.add(ulCenter);

// Upper base fill — thin tube across bottom of upper lip (seam side)
var ulBase = makeTube([
  new THREE.Vector3(MX-0.155, MY+0.000, MZ-0.012),
  new THREE.Vector3(MX-0.100, MY+0.003, MZ+0.004),
  new THREE.Vector3(MX-0.050, MY+0.004, MZ+0.008),
  new THREE.Vector3(MX+0.000, MY+0.004, MZ+0.010),
  new THREE.Vector3(MX+0.050, MY+0.004, MZ+0.008),
  new THREE.Vector3(MX+0.100, MY+0.003, MZ+0.004),
  new THREE.Vector3(MX+0.155, MY+0.000, MZ-0.012),
], 0.018, 26, 8, lipBodyMat);
headGroup.add(ulBase);

// ── Vermilion border — bright thin tube tracing the entire cupid's bow ───────
var ulVerm = makeTube([
  new THREE.Vector3(MX-0.155, MY+0.000, MZ-0.010),
  new THREE.Vector3(MX-0.118, MY+0.012, MZ+0.003),
  new THREE.Vector3(MX-0.088, MY+0.028, MZ+0.013),
  new THREE.Vector3(MX-0.068, MY+0.048, MZ+0.024), // left peak
  new THREE.Vector3(MX-0.042, MY+0.055, MZ+0.026),
  new THREE.Vector3(MX-0.020, MY+0.045, MZ+0.024), // philtrum dip
  new THREE.Vector3(MX+0.000, MY+0.038, MZ+0.023),
  new THREE.Vector3(MX+0.020, MY+0.045, MZ+0.024),
  new THREE.Vector3(MX+0.042, MY+0.055, MZ+0.026), // right peak
  new THREE.Vector3(MX+0.068, MY+0.048, MZ+0.024),
  new THREE.Vector3(MX+0.088, MY+0.028, MZ+0.013),
  new THREE.Vector3(MX+0.118, MY+0.012, MZ+0.003),
  new THREE.Vector3(MX+0.155, MY+0.000, MZ-0.010),
], 0.007, 36, 7, vermMat);
headGroup.add(ulVerm);

// ══════════════════════════════════════════════════════════════════════════════
// LOWER LIP — single wide dome, fuller than upper, attached to jawPivot
// jawPivot.position.y = PIVOT_Y, so local coords = world - PIVOT_Y
// ══════════════════════════════════════════════════════════════════════════════
var LLY = MY - PIVOT_Y; // local Y in jaw pivot space = -0.484 - (-0.76) = +0.276
var LLZ = MZ;           // Z is same (pivot only rotates around X)

// Main lower lip body — three parallel arcs stacked to give dome volume
// Arc 1: seam edge (thin — where it meets upper lip)
var llSeamArc = makeTube([
  new THREE.Vector3(MX-0.152, LLY-0.002, LLZ-0.010),
  new THREE.Vector3(MX-0.100, LLY-0.002, LLZ+0.002),
  new THREE.Vector3(MX-0.050, LLY-0.002, LLZ+0.008),
  new THREE.Vector3(MX+0.000, LLY-0.002, LLZ+0.010),
  new THREE.Vector3(MX+0.050, LLY-0.002, LLZ+0.008),
  new THREE.Vector3(MX+0.100, LLY-0.002, LLZ+0.002),
  new THREE.Vector3(MX+0.152, LLY-0.002, LLZ-0.010),
], 0.018, 26, 8, lipBodyMat);
jawPivot.add(llSeamArc);

// Arc 2: mid dome — widest/thickest part of lower lip
var llMidArc = makeTube([
  new THREE.Vector3(MX-0.148, LLY-0.018, LLZ-0.004),
  new THREE.Vector3(MX-0.100, LLY-0.026, LLZ+0.010),
  new THREE.Vector3(MX-0.050, LLY-0.034, LLZ+0.022),
  new THREE.Vector3(MX+0.000, LLY-0.038, LLZ+0.026), // center fullest point
  new THREE.Vector3(MX+0.050, LLY-0.034, LLZ+0.022),
  new THREE.Vector3(MX+0.100, LLY-0.026, LLZ+0.010),
  new THREE.Vector3(MX+0.148, LLY-0.018, LLZ-0.004),
], 0.032, 26, 12, lipBodyMat); // thickest tube = most volume
jawPivot.add(llMidArc);

// Arc 3: lower edge — tapering back to face surface
var llBottomArc = makeTube([
  new THREE.Vector3(MX-0.140, LLY-0.042, LLZ-0.008),
  new THREE.Vector3(MX-0.090, LLY-0.054, LLZ+0.006),
  new THREE.Vector3(MX-0.040, LLY-0.062, LLZ+0.014),
  new THREE.Vector3(MX+0.000, LLY-0.065, LLZ+0.016),
  new THREE.Vector3(MX+0.040, LLY-0.062, LLZ+0.014),
  new THREE.Vector3(MX+0.090, LLY-0.054, LLZ+0.006),
  new THREE.Vector3(MX+0.140, LLY-0.042, LLZ-0.008),
], 0.022, 26, 10, lipBodyMat);
jawPivot.add(llBottomArc);

// Lower vermilion border — bright outline at seam edge of lower lip
var llVerm = makeTube([
  new THREE.Vector3(MX-0.152, LLY-0.001, LLZ-0.008),
  new THREE.Vector3(MX-0.100, LLY-0.001, LLZ+0.006),
  new THREE.Vector3(MX-0.050, LLY-0.001, LLZ+0.012),
  new THREE.Vector3(MX+0.000, LLY-0.001, LLZ+0.014),
  new THREE.Vector3(MX+0.050, LLY-0.001, LLZ+0.012),
  new THREE.Vector3(MX+0.100, LLY-0.001, LLZ+0.006),
  new THREE.Vector3(MX+0.152, LLY-0.001, LLZ-0.008),
], 0.006, 26, 7, vermMat);
jawPivot.add(llVerm);

// ── MOUTH SEAM LINE — dark narrow gap between the two lips ──────────────────
// Fixed in world space (not on jaw) — always visible at the lip meeting point
var mouthSeam = makeTube([
  new THREE.Vector3(MX-0.153, MY+0.001, MZ-0.009),
  new THREE.Vector3(MX-0.110, MY+0.001, MZ+0.005),
  new THREE.Vector3(MX-0.060, MY+0.001, MZ+0.014),
  new THREE.Vector3(MX-0.020, MY+0.001, MZ+0.018),
  new THREE.Vector3(MX+0.000, MY+0.001, MZ+0.019),
  new THREE.Vector3(MX+0.020, MY+0.001, MZ+0.018),
  new THREE.Vector3(MX+0.060, MY+0.001, MZ+0.014),
  new THREE.Vector3(MX+0.110, MY+0.001, MZ+0.005),
  new THREE.Vector3(MX+0.153, MY+0.001, MZ-0.009),
], 0.0045, 28, 6, lipSeamMat);
headGroup.add(mouthSeam);

// ── MOUTH CORNERS — small anchor spheres ────────────────────────────────────
[-1, 1].forEach(function(s) {
  var cg = new THREE.SphereGeometry(0.018, 10, 8);
  headGroup.add(new THREE.Mesh(cg, lipBodyMat)).position.set(s*0.153, MY+0.000, MZ-0.010);
});

// ─── NOSE — proportional, subtle, integrated with face surface ───────────────
var noseMat = regMat(mkMat({ base: 0x010e28, glow: 0x009eee, opacity: 0.82, scan: 0.48 }));

// Bridge — thin vertical ridge
var nbGeo = new THREE.SphereGeometry(1.0, 26, 18);
var nbP = nbGeo.attributes.position;
for (var i = 0; i < nbP.count; i++) {
  var nby = nbP.getY(i);
  var bWidth = 0.038 + Math.abs(nby) * 0.006;
  nbP.setX(i, nbP.getX(i) * bWidth);
  nbP.setY(i, nby * 0.180);
  nbP.setZ(i, nbP.getZ(i) * 0.060);
}
nbP.needsUpdate = true; nbGeo.computeVertexNormals();
var noseBridge = new THREE.Mesh(nbGeo, noseMat);
noseBridge.position.set(0, -0.010, 0.870); headGroup.add(noseBridge);

// Tip — small rounded dome
var ntGeo = new THREE.SphereGeometry(0.072, 26, 18);
var ntP = ntGeo.attributes.position;
for (var i = 0; i < ntP.count; i++) {
  ntP.setZ(i, ntP.getZ(i) * 0.62);
  ntP.setX(i, ntP.getX(i) * 0.90);
  if (ntP.getY(i) < -0.02) ntP.setY(i, ntP.getY(i) * 0.80);
}
ntP.needsUpdate = true; ntGeo.computeVertexNormals();
var noseTip = new THREE.Mesh(ntGeo, noseMat);
noseTip.position.set(0, -0.270, 0.880); headGroup.add(noseTip);

// Nostril wings — small, subtle
[-1, 1].forEach(function(s) {
  var nwGeo = new THREE.SphereGeometry(1.0, 16, 12);
  var nwP = nwGeo.attributes.position;
  for (var i = 0; i < nwP.count; i++) {
    nwP.setX(i, nwP.getX(i) * 0.048);
    nwP.setZ(i, nwP.getZ(i) * 0.042);
    nwP.setY(i, nwP.getY(i) * 0.052);
  }
  nwP.needsUpdate = true; nwGeo.computeVertexNormals();
  var nw = new THREE.Mesh(nwGeo, noseMat);
  nw.position.set(s * 0.072, -0.296, 0.858); headGroup.add(nw);
});

// ─── EARS — large, visible, anatomically sculpted human ears ─────────────────
// Brighter material so ears are clearly visible through holographic shader
var earMat = regMat(mkMat({ base: 0x011428, glow: 0x00aaee, opacity: 0.88, scan: 0.50 }));
var earSkinMat = new THREE.MeshPhongMaterial({
  color: 0x021840, emissive: 0x002850, emissiveIntensity: 1.2,
  specular: 0x4488cc, shininess: 100, transparent: true, opacity: 0.92
});
[-1, 1].forEach(function(s) {
  var eg = new THREE.Group();

  // ── CONCHA (ear bowl) — large filled base that makes the ear visible ──
  var conchaGeo = new THREE.SphereGeometry(1.0, 28, 22);
  var cP = conchaGeo.attributes.position;
  for (var ci = 0; ci < cP.count; ci++) {
    var cx = cP.getX(ci), cy = cP.getY(ci), cz = cP.getZ(ci);
    // Flatten into ear-shaped disc
    cP.setX(ci, cx * 0.18);
    cP.setY(ci, cy * 0.32);
    cP.setZ(ci, cz * 0.04);
  }
  cP.needsUpdate = true; conchaGeo.computeVertexNormals();
  var conchaMesh = new THREE.Mesh(conchaGeo, earSkinMat);
  conchaMesh.position.set(0.02, 0.0, 0.005);
  eg.add(conchaMesh);

  // ── Outer helix — thick curved rim ────────────────────────────────
  var helixPath = new THREE.CatmullRomCurve3([
    new THREE.Vector3( 0.010,  0.350, 0.015),
    new THREE.Vector3( 0.070,  0.360, 0.025),
    new THREE.Vector3( 0.140,  0.330, 0.038),
    new THREE.Vector3( 0.200,  0.240, 0.032),
    new THREE.Vector3( 0.230,  0.130, 0.024),
    new THREE.Vector3( 0.240,  0.010, 0.016),
    new THREE.Vector3( 0.225, -0.110, 0.010),
    new THREE.Vector3( 0.190, -0.220, 0.008),
    new THREE.Vector3( 0.130, -0.310, 0.012),
    new THREE.Vector3( 0.060, -0.370, 0.018),
    new THREE.Vector3(-0.010, -0.390, 0.022),
    new THREE.Vector3(-0.060, -0.375, 0.025)
  ], false);
  var helixGeo = new THREE.TubeGeometry(helixPath, 52, 0.022, 10, false);
  eg.add(new THREE.Mesh(helixGeo, earMat));

  // ── Inner helix rim ───────────────────────────────────────────────
  var innerPath = new THREE.CatmullRomCurve3([
    new THREE.Vector3( 0.020,  0.300, 0.028),
    new THREE.Vector3( 0.100,  0.290, 0.042),
    new THREE.Vector3( 0.160,  0.210, 0.036),
    new THREE.Vector3( 0.185,  0.100, 0.028),
    new THREE.Vector3( 0.180, -0.010, 0.020),
    new THREE.Vector3( 0.150, -0.130, 0.015)
  ], false);
  eg.add(new THREE.Mesh(new THREE.TubeGeometry(innerPath, 32, 0.016, 8, false), earMat));

  // ── Antihelix — Y-shaped inner ridge ──────────────────────────────
  var ahPath = new THREE.CatmullRomCurve3([
    new THREE.Vector3( 0.020,  0.240, 0.035),
    new THREE.Vector3( 0.065,  0.130, 0.042),
    new THREE.Vector3( 0.070,  0.010, 0.035),
    new THREE.Vector3( 0.050, -0.100, 0.025),
    new THREE.Vector3( 0.030, -0.200, 0.020)
  ], false);
  eg.add(new THREE.Mesh(new THREE.TubeGeometry(ahPath, 30, 0.018, 8, false), earMat));

  // Superior crus
  var sc1 = new THREE.CatmullRomCurve3([
    new THREE.Vector3( 0.020,  0.240, 0.035),
    new THREE.Vector3(-0.015,  0.300, 0.028),
    new THREE.Vector3(-0.008,  0.340, 0.020)
  ], false);
  eg.add(new THREE.Mesh(new THREE.TubeGeometry(sc1, 14, 0.020, 8, false), earMat));

  // Inferior crus
  var sc2 = new THREE.CatmullRomCurve3([
    new THREE.Vector3( 0.020,  0.240, 0.035),
    new THREE.Vector3( 0.060,  0.300, 0.032),
    new THREE.Vector3( 0.075,  0.320, 0.026)
  ], false);
  eg.add(new THREE.Mesh(new THREE.TubeGeometry(sc2, 14, 0.018, 8, false), earMat));

  // ── Tragus — protruding knob ──────────────────────────────────────
  var tragGeo = new THREE.SphereGeometry(1.0, 18, 14);
  var tP = tragGeo.attributes.position;
  for (var ti = 0; ti < tP.count; ti++) {
    tP.setX(ti, tP.getX(ti)*0.048);
    tP.setY(ti, tP.getY(ti)*0.062);
    tP.setZ(ti, tP.getZ(ti)*0.038);
  }
  tP.needsUpdate = true; tragGeo.computeVertexNormals();
  var trag = new THREE.Mesh(tragGeo, earMat);
  trag.position.set(-0.055, 0.025, 0.042); eg.add(trag);

  // ── Antitragus ────────────────────────────────────────────────────
  var atGeo = new THREE.SphereGeometry(1.0, 16, 12);
  var atP = atGeo.attributes.position;
  for (var ai = 0; ai < atP.count; ai++) {
    atP.setX(ai, atP.getX(ai)*0.042);
    atP.setY(ai, atP.getY(ai)*0.055);
    atP.setZ(ai, atP.getZ(ai)*0.034);
  }
  atP.needsUpdate = true; atGeo.computeVertexNormals();
  var atrag = new THREE.Mesh(atGeo, earMat);
  atrag.position.set(0.010, -0.120, 0.035); eg.add(atrag);

  // ── Earlobe — large, rounded, clearly visible ─────────────────────
  var lobeGeo = new THREE.SphereGeometry(1.0, 22, 18);
  var lP = lobeGeo.attributes.position;
  for (var li = 0; li < lP.count; li++) {
    var ly = lP.getY(li);
    var lobeW = 0.110 + (ly < 0 ? Math.abs(ly) * 0.030 : 0);
    lP.setX(li, lP.getX(li) * lobeW);
    lP.setY(li, ly * 0.135);
    lP.setZ(li, lP.getZ(li) * 0.075);
  }
  lP.needsUpdate = true; lobeGeo.computeVertexNormals();
  var lobe = new THREE.Mesh(lobeGeo, earSkinMat);
  lobe.position.set(-0.020, -0.395, 0.028); eg.add(lobe);

  // ── Position ears on head — flush against sides ────────────────────
  eg.rotation.y = s < 0 ? Math.PI * 0.44 : -Math.PI * 0.44;
  eg.rotation.z = s * 0.04;
  eg.position.set(s * 0.70, -0.04, -0.10);
  eg.scale.set(s < 0 ? 1.05 : -1.05, 1.05, 1.05);
  headGroup.add(eg);
});

// ─── HOLOGRAPHIC SCAN OVERLAY ─────────────────────────────────────────────────
var scanCvs = document.createElement('canvas');
scanCvs.width = 2; scanCvs.height = 12;
var sc = scanCvs.getContext('2d');
sc.fillStyle = 'rgba(0,170,255,0.045)';
sc.fillRect(0, 0, 2, 5);
var scanTex = new THREE.CanvasTexture(scanCvs);
scanTex.wrapS = scanTex.wrapT = THREE.RepeatWrapping;
scanTex.repeat.set(1, 80);
var scanMesh = new THREE.Mesh(
  new THREE.PlaneGeometry(3.4, 5.2),
  new THREE.MeshBasicMaterial({ map: scanTex, transparent: true, opacity: 0.13, depthWrite: false, side: THREE.DoubleSide })
);
scanMesh.position.set(0, 0.0, 1.12);
scene.add(scanMesh);

// Moving highlight sweep bar
var sweepBar = new THREE.Mesh(
  new THREE.PlaneGeometry(3.4, 0.055),
  new THREE.MeshBasicMaterial({ color: 0x00d4ff, transparent: true, opacity: 0.07, depthWrite: false })
);
sweepBar.position.set(0, 2.6, 1.13);
scene.add(sweepBar);

// ─── PARTICLES ────────────────────────────────────────────────────────────────
var PC = 200;
var pBuf = new Float32Array(PC * 3);
var pSpd = new Float32Array(PC);
for (var pi = 0; pi < PC; pi++) {
  pBuf[pi*3]   = (Math.random()-0.5)*8;
  pBuf[pi*3+1] = (Math.random()-0.5)*7;
  pBuf[pi*3+2] = (Math.random()-0.5)*5 - 1.5;
  pSpd[pi]     = 0.0008 + Math.random() * 0.0035;
}
var pGeo = new THREE.BufferGeometry();
pGeo.setAttribute('position', new THREE.BufferAttribute(pBuf, 3));
scene.add(new THREE.Points(pGeo, new THREE.PointsMaterial({ color: 0x00ccff, size: 0.011, transparent: true, opacity: 0.25 })));

// ─── LIGHTING ─────────────────────────────────────────────────────────────────
scene.add(new THREE.AmbientLight(0x061520, 0.38));

var keyLight = new THREE.DirectionalLight(0x88bbff, 1.6);
keyLight.position.set(1.0, 2.2, 3.5);
keyLight.castShadow = true;
keyLight.shadow.mapSize.set(1024, 1024);
scene.add(keyLight);

var fillLight = new THREE.DirectionalLight(0x1a3d6e, 0.42);
fillLight.position.set(-2.2, 0.5, 1.0);
scene.add(fillLight);

var rimLight = new THREE.PointLight(0x00d4ff, 1.1, 9);
rimLight.position.set(-2.4, 1.4, -1.8);
scene.add(rimLight);

var frontGlow = new THREE.PointLight(0x0088cc, 0.22, 6);
frontGlow.position.set(0, 0.12, 3.5);
scene.add(frontGlow);

var topRim = new THREE.PointLight(0x00aaff, 0.52, 7);
topRim.position.set(0, 3.5, -0.8);
scene.add(topRim);

var bottomLight = new THREE.PointLight(0x002266, 0.28, 5);
bottomLight.position.set(0, -3.0, 1.2);
scene.add(bottomLight);

// ─── IMPROVEMENT 5: CubeCamera for corneal reflections ───────────────────────
var cubeRenderTarget = new THREE.WebGLCubeRenderTarget(64, {
  format: THREE.RGBFormat,
  generateMipmaps: true,
  minFilter: THREE.LinearMipmapLinearFilter
});
var cubeCamera = new THREE.CubeCamera(0.1, 12, cubeRenderTarget);
cubeCamera.position.set(0, 0.10, 0.78);
scene.add(cubeCamera);
var cubeCamFrame = 0;
// Apply env map to iris materials after eye objects are created
function applyEnvMapToEyes() {
  eyeObjects.forEach(function(eo) {
    eo.irisMat.envMap          = cubeRenderTarget.texture;
    eo.irisMat.reflectivity    = 0.20;
    eo.irisMat.envMapIntensity = 0.28;
    eo.irisMat.needsUpdate     = true;
  });
}

// ─── HEAD POSITION ────────────────────────────────────────────────────────────
headGroup.position.set(0, 0.12, 0);

// ─── BLINKING ─────────────────────────────────────────────────────────────────
var blinkOpen = 0; // 0 = open, 1 = closed
var blinkProgress = 0;
var nextBlink = 2.5 + Math.random() * 2.0;
var blinkDur = 0.11;
var blinkPhase = 'open'; // 'open','closing','opening'

function updateBlink(dt) {
  nextBlink -= dt;
  if (blinkPhase === 'open' && nextBlink <= 0) {
    blinkPhase = 'closing';
    blinkProgress = 0;
  }
  if (blinkPhase === 'closing') {
    blinkProgress += dt / blinkDur;
    blinkOpen = blinkProgress;
    if (blinkProgress >= 1.0) { blinkPhase = 'opening'; blinkProgress = 0; }
  }
  if (blinkPhase === 'opening') {
    blinkProgress += dt / blinkDur;
    blinkOpen = 1.0 - blinkProgress;
    if (blinkProgress >= 1.0) {
      blinkPhase = 'open';
      blinkOpen = 0;
      nextBlink = 2.2 + Math.random() * 3.5;
      // Occasional double-blink
      if (Math.random() < 0.2) nextBlink = 0.18 + Math.random() * 0.12;
    }
  }
  // Apply to eyelids: slide upper lid DOWN to close, raise lower lid UP
  eyeObjects.forEach(function(eo) {
    // Upper lid sweeps down: 0.025 (open) to -0.045 (closed)
    eo.upperLid.position.y = 0.025 - blinkOpen * 0.070;
    // Lower lid rises slightly: -0.020 (open) to -0.005 (closed)
    eo.lowerLid.position.y = -0.020 + blinkOpen * 0.015;
    // Shutter becomes opaque to fully cover eyeball when closed
    eo.shutter.position.y = 0.08 - blinkOpen * 0.08; // slides from above down to center
    eo.shutterMat.uniforms.uOpacity.value = blinkOpen * 0.96;
  });
}

// ─── EYE MOVEMENT (saccades) ──────────────────────────────────────────────────
var eyeRotX = 0, eyeRotY = 0;
var eyeTargetX = 0, eyeTargetY = 0;
var eyeSaccTimerV = 1.8;

function updateEyes(dt, t) {
  eyeSaccTimerV -= dt;
  if (eyeSaccTimerV <= 0) {
    eyeTargetY = (Math.random()-0.5) * 0.12;
    eyeTargetX = (Math.random()-0.5) * 0.06;
    eyeSaccTimerV = 1.4 + Math.random() * 2.8;
  }
  // Smooth saccade
  eyeRotX += (eyeTargetX - eyeRotX) * Math.min(1, dt * 6);
  eyeRotY += (eyeTargetY - eyeRotY) * Math.min(1, dt * 6);
  eyeObjects.forEach(function(eo) {
    eo.group.rotation.x = eyeRotX;
    eo.group.rotation.y = eyeRotY * (eo.group.position.x < 0 ? -1 : 1) * 0.5;
  });
}

// ─── IMPROVEMENT 1: Web Audio for real amplitude-driven jaw ──────────────────
var audioCtx = null, analyserNode = null, audioFreqBuf = null, audioStream = null;
var audioAmplitude = 0.0;

function startAudioCapture() {
  if (audioCtx) return;
  try {
    audioCtx = new (window.AudioContext || window.webkitAudioContext)();
    analyserNode = audioCtx.createAnalyser();
    analyserNode.fftSize = 256;
    audioFreqBuf = new Uint8Array(analyserNode.frequencyBinCount);
    navigator.mediaDevices.getUserMedia({ audio: true, video: false })
      .then(function(stream) {
        audioStream = stream;
        audioCtx.createMediaStreamSource(stream).connect(analyserNode);
      }).catch(function() { audioCtx = null; analyserNode = null; });
  } catch(e) { audioCtx = null; analyserNode = null; }
}

function stopAudioCapture() {
  try {
    if (audioStream) { audioStream.getTracks().forEach(function(t){t.stop();}); audioStream = null; }
    if (audioCtx) { audioCtx.close(); audioCtx = null; }
    analyserNode = null; audioFreqBuf = null; audioAmplitude = 0;
  } catch(e) {}
}

function readAudioAmplitude() {
  if (!analyserNode || !audioFreqBuf) return 0;
  try {
    analyserNode.getByteFrequencyData(audioFreqBuf);
    var sum = 0;
    var bins = Math.min(14, audioFreqBuf.length);
    for (var bi = 2; bi < bins; bi++) sum += audioFreqBuf[bi];
    return Math.min(1.0, sum / ((bins - 2) * 155));
  } catch(e) { return 0; }
}

// ─── IMPROVEMENT 4: Viseme lip sync state (v2 — smooth) ─────────────────────
// Phoneme-to-jaw-openness map with wider range for natural movement
var VISEME_OPEN = {
  'a':0.58,'e':0.46,'i':0.28,'o':0.60,'u':0.44,
  'm':0.04,'b':0.06,'p':0.06,'w':0.30,
  'f':0.16,'v':0.16,'th':0.20,
  's':0.18,'z':0.18,'sh':0.20,
  'n':0.15,'l':0.22,'r':0.26,
  'd':0.20,'t':0.18,'k':0.26,'g':0.26,
  'h':0.32,'y':0.30,
  'default':0.20
};
var currentViseme = 'default';
var visemeJawTarget = 0;
// Smooth viseme queue for word-level animation
var visemeQueue = [];
var visemeTimer = 0;
var visemeDuration = 0.12; // seconds per viseme transition
var visemeCurrent = 0;     // current jaw openness (smoothed)
var visemeTarget  = 0;     // target jaw openness

function phonemeFromChar(ch) {
  ch = (ch||'').toLowerCase();
  if ('aeiou'.includes(ch)) return ch;
  if ('mbp'.includes(ch)) return 'm';
  if ('fv'.includes(ch)) return 'f';
  if ('sz'.includes(ch)) return 's';
  if ('nl'.includes(ch)) return 'n';
  if ('dt'.includes(ch)) return 'd';
  if ('kg'.includes(ch)) return 'k';
  if (ch === 'w') return 'w';
  if (ch === 'r') return 'r';
  if (ch === 'h') return 'h';
  if (ch === 'y') return 'y';
  return 'default';
}

// Build viseme sequence from a whole word for smoother transitions
function visemesFromWord(word) {
  var seq = [];
  word = (word||'').toLowerCase().replace(/[^a-z]/g,'');
  for (var ci = 0; ci < word.length && ci < 6; ci++) {
    var ph = phonemeFromChar(word[ci]);
    seq.push(VISEME_OPEN[ph] || 0.20);
  }
  if (seq.length === 0) seq.push(0.20);
  return seq;
}

// Enqueue word visemes for smooth playback
function enqueueWordVisemes(word) {
  var seq = visemesFromWord(word);
  for (var si = 0; si < seq.length; si++) {
    visemeQueue.push(seq[si]);
  }
  // Add a brief close between words
  visemeQueue.push(0.04);
}

// ─── IMPROVEMENT 6: Ripple state ─────────────────────────────────────────────
var rippleStartTime = -99.0;

// ─── IMPROVEMENT 2 + Morph weights ───────────────────────────────────────────
var morphWeights = [0, 0, 0];  // 0=raiseBrows 1=furrow 2=squint

// ─── ANIMATION ────────────────────────────────────────────────────────────────
var clock = new THREE.Clock();

function render() {
  requestAnimationFrame(render);
  var dt = Math.min(clock.getDelta(), 0.05);
  var t  = clock.getElapsedTime();

  // ── IMPROVEMENT 1: Smooth viseme-driven jaw animation ──────────────────────
  var rawAmp = readAudioAmplitude();
  audioAmplitude += (rawAmp - audioAmplitude) * 0.32;

  if (STATE === 'speaking') {
    // Process viseme queue for smooth word-by-word lip sync
    visemeTimer += dt;
    if (visemeQueue.length > 0 && visemeTimer >= visemeDuration) {
      visemeTimer = 0;
      visemeTarget = visemeQueue.shift();
      // Vary duration slightly for natural rhythm
      visemeDuration = 0.08 + Math.random() * 0.06;
    }
    // Smooth cubic interpolation toward target
    var lerpSpeed = 12.0;
    visemeCurrent += (visemeTarget - visemeCurrent) * Math.min(1, dt * lerpSpeed);
    // Add subtle organic flutter
    var flutter = Math.sin(t * 18.0) * 0.006 + Math.sin(t * 31.0) * 0.003;
    // Scale down to safe jaw rotation range (max ~0.12 radians = ~7 degrees)
    jawTarget = visemeCurrent * 0.20 + flutter;
    // If audio amplitude available, blend it in for extra realism
    if (audioAmplitude > 0.03) {
      jawTarget = jawTarget * 0.7 + audioAmplitude * 0.10;
    }
    // Clamp to safe rotation range
    if (jawTarget < 0) jawTarget = 0;
    if (jawTarget > 0.14) jawTarget = 0.14;
  } else if (STATE === 'listening') {
    // Subtle mic-reactive movement when listening
    jawTarget = audioAmplitude > 0.02 ? audioAmplitude * 0.15 : 0;
    visemeQueue.length = 0;
    visemeCurrent = 0;
    visemeTarget = 0;
  } else {
    jawTarget = 0;
    visemeQueue.length = 0;
    visemeCurrent = 0;
    visemeTarget = 0;
  }
  // Smooth jaw pivot rotation (softer spring)
  jawAngle += (jawTarget - jawAngle) * Math.min(1, dt * 14);
  jawPivot.rotation.x = jawAngle;

  // ── IMPROVEMENT 5: Morph targets smooth interpolation ───────────────────
  if (cranMesh.morphTargetInfluences) {
    for (var mi = 0; mi < cranMesh.morphTargetInfluences.length && mi < 3; mi++) {
      cranMesh.morphTargetInfluences[mi] +=
        (morphWeights[mi] - cranMesh.morphTargetInfluences[mi]) * Math.min(1, dt * 2.8);
    }
  }

  // ── IMPROVEMENT 3: SSS pulse with jaw and state ──────────────────────────
  if (sssMesh) {
    sssMesh.material.opacity = 0.038 + jawAngle * 0.15 +
      (STATE === 'listening' ? 0.012 : STATE === 'speaking' ? 0.014 : 0);
    sssEdgeMesh.material.opacity = 0.025 + jawAngle * 0.20;
  }

  // ── Head organic motion ──────────────────────────────────────────────────
  var swayAmt  = 0.018, nodAmt = 0.012, breathAmt = 0.005;
  if (STATE === 'listening')  { swayAmt = 0.008; nodAmt = 0.018; }
  if (STATE === 'processing') { swayAmt = 0.006; nodAmt = 0.006; }
  if (STATE === 'speaking')   { swayAmt = 0.020; nodAmt = 0.014; }

  headGroup.rotation.y = Math.sin(t * 0.38) * swayAmt + Math.sin(t * 0.19) * swayAmt * 0.4;
  headGroup.rotation.x = Math.sin(t * 0.27) * nodAmt + jawAngle * 0.25;
  headGroup.rotation.z = Math.sin(t * 0.22) * 0.006;
  headGroup.position.y = 0.12 + Math.sin(t * 0.70) * breathAmt;

  // ── Blinking ─────────────────────────────────────────────────────────────
  updateBlink(dt);

  // ── Eye movement ─────────────────────────────────────────────────────────
  updateEyes(dt, t);

  // ── Eye glow pulsing ─────────────────────────────────────────────────────
  var eyeGlow = 0.22 + Math.sin(t * 1.2) * 0.06;
  if (STATE === 'speaking')   eyeGlow = 0.38 + Math.sin(t * 4.0) * 0.12;
  if (STATE === 'listening')  eyeGlow = 0.30 + Math.sin(t * 2.5) * 0.10;
  if (STATE === 'processing') eyeGlow = 0.35 + Math.sin(t * 3.8) * 0.12;
  eyeObjects.forEach(function(eo) {
    eo.glowRing.material.opacity = eyeGlow;
    eo.irisMat.emissiveIntensity = 0.9 + eyeGlow * 0.8;
  });

  // ── Scan line sweep ──────────────────────────────────────────────────────
  scanTex.offset.y = t * 0.036;
  sweepBar.position.y = 2.6 - ((t * 0.26) % 5.2);

  // ── Particles ────────────────────────────────────────────────────────────
  var pAttr = pGeo.attributes.position;
  for (var pi = 0; pi < PC; pi++) {
    var py = pAttr.getY(pi) + pSpd[pi];
    if (py > 4.0) py = -4.0;
    pAttr.setY(pi, py);
    pAttr.setX(pi, pAttr.getX(pi) + Math.sin(t + pi) * 0.0002);
  }
  pAttr.needsUpdate = true;

  // ── Holographic material pulse ───────────────────────────────────────────
  var pulse = Math.sin(t * 0.82) * 0.28 + 0.22;
  if (STATE === 'speaking')   pulse = 0.22 + jawAngle * 2.0 + Math.sin(t * 5.0) * 0.05;
  if (STATE === 'listening')  pulse = 0.40 + Math.sin(t * 2.8) * 0.18;
  if (STATE === 'processing') pulse = 0.60 + Math.sin(t * 4.5) * 0.22;
  ALL_MATS.forEach(function(m) {
    m.uniforms.uTime.value  = t;
    m.uniforms.uPulse.value = pulse;
    // rippleTime was already set in applyState, just propagate time
  });

  // ── IMPROVEMENT 5: Corneal reflection (update cube camera every 8 frames) ─
  cubeCamFrame++;
  if (cubeCamFrame % 8 === 0 && typeof cubeCamera !== 'undefined') {
    headGroup.visible = false;
    cubeCamera.update(renderer, scene);
    headGroup.visible = true;
  }

  // ── IMPROVEMENT 2: Bloom adapts to state ─────────────────────────────────
  if (bloomPass) {
    bloomPass.strength = STATE === 'speaking'   ? 0.50 + jawAngle * 0.80 :
                         STATE === 'listening'  ? 0.72 :
                         STATE === 'processing' ? 0.78 + Math.sin(t*3)*0.06 : 0.62;
    bloomPass.threshold = 0.18;
  }
  if (ssaoPass) {
    ssaoPass.kernelRadius = STATE === 'speaking' ? 10 : 14;
  }
  if (chromaticPass) {
    chromaticPass.uniforms.uStr.value =
      STATE === 'speaking' ? 0.0022 + jawAngle * 0.008 : 0.0014;
  }

  // ── Wireframe opacity ────────────────────────────────────────────────────
  var wop = 0.055 + Math.sin(t * 1.1) * 0.018;
  if (STATE === 'speaking') wop = 0.052 + jawAngle * 0.45 + Math.sin(t * 3.2) * 0.008;
  wireMat.opacity = wop;

  // ── Dynamic lighting ─────────────────────────────────────────────────────
  if (STATE === 'speaking') {
    rimLight.intensity   = 0.80 + Math.sin(t * 5.5) * 0.07 + jawAngle * 1.2;
    frontGlow.intensity  = 0.18 + jawAngle * 1.0;
    topRim.intensity     = 0.42 + jawAngle * 0.6;
  } else if (STATE === 'listening') {
    rimLight.intensity   = 0.95 + Math.sin(t * 3.2) * 0.16;
    topRim.intensity     = 0.50 + Math.sin(t * 2.5) * 0.10;
    frontGlow.intensity  = 0.22;
  } else if (STATE === 'processing') {
    rimLight.intensity   = 0.85 + Math.sin(t * 4.8) * 0.25;
    frontGlow.intensity  = 0.26 + Math.sin(t * 3.5) * 0.09;
    topRim.intensity     = 0.50;
  } else {
    rimLight.intensity   = 0.82 + Math.sin(t * 0.65) * 0.07;
    frontGlow.intensity  = 0.20 + Math.sin(t * 0.48) * 0.04;
    topRim.intensity     = 0.48 + Math.sin(t * 0.38) * 0.05;
  }

  // Render via composer (Bloom+SSAO) or fallback
  if (composer) {
    composer.render();
  } else {
    renderer.render(scene, camera);
  }
}
render();

// ─── IMPROVEMENT 1: Poll FastAPI + TTS with viseme lip sync ──────────────────
var lastAnswerTs = 0;

function pollAnswer() {
  if (STATE !== 'processing') return;
  fetch('http://localhost:8000/voice/answer')
    .then(function(r){ return r.ok ? r.json() : null; })
    .then(function(d) {
      if (!d || !d.text || !d.text.trim() || d.ts === lastAnswerTs) return;
      lastAnswerTs = d.ts;
      speakAnswer(d.text);
    })
    .catch(function(){});
}
setInterval(pollAnswer, 1500);

function speakAnswer(text) {
  var synth = window.speechSynthesis;
  synth.cancel();
  applyState('speaking');
  var preview = text.split('**').join('').split('\n').join(' ').substring(0,100);
  document.getElementById('transcript').textContent = preview + '...';

  // Clean text fully — no truncation
  var clean = text.split('**').join('').split('\n\n').join('. ').split('\n').join(' ');
  // Remove emoji characters that TTS struggles with
  clean = clean.replace(/[\u{1F600}-\u{1F9FF}]|[\u{2600}-\u{27BF}]|[\u{1F300}-\u{1F5FF}]|[\u{1F680}-\u{1F6FF}]/gu, '');

  // Split into sentence-level chunks (~250 chars each) to avoid Chrome TTS cutoff
  var chunks = [];
  var sentences = clean.split(/(?<=[.!?])\s+/);
  var current = '';
  for (var si = 0; si < sentences.length; si++) {
    if ((current + ' ' + sentences[si]).length > 250 && current.length > 0) {
      chunks.push(current.trim());
      current = sentences[si];
    } else {
      current += (current ? ' ' : '') + sentences[si];
    }
  }
  if (current.trim()) chunks.push(current.trim());
  if (chunks.length === 0) chunks.push(clean.substring(0, 250));

  var chunkIdx = 0;

  // IMPROVEMENT 4: Viseme-based lip sync on word boundary (v2 — smooth)
  function attachViseme(utterance) {
    utterance.onboundary = function(ev) {
      if (ev.name === 'word') {
        var idx = ev.charIndex || 0;
        var text = (ev.utterance && ev.utterance.text) ? ev.utterance.text : '';
        var remaining = text.substring(idx);
        var wordMatch = remaining.match(/^[a-zA-Z]+/);
        var word = wordMatch ? wordMatch[0] : '';
        if (word) {
          enqueueWordVisemes(word);
          currentViseme = phonemeFromChar(word[0]);
        }
      }
    };
  }

  function go() {
    var vs = synth.getVoices();
    var pref = vs.find(function(v){
      return v.lang.startsWith('en') &&
        (v.name.includes('Google') || v.name.includes('Natural') || v.name.includes('Samantha'));
    }) || vs.find(function(v){ return v.lang.startsWith('en'); }) || null;

    function speakChunk() {
      if (chunkIdx >= chunks.length) {
        // All chunks done
        applyState('ready');
        jawTarget = 0; currentViseme = 'default';
        visemeQueue.length = 0; visemeCurrent = 0; visemeTarget = 0;
        document.getElementById('transcript').textContent =
          'Say "Hey Buddy" or click to ask another question';
        return;
      }
      var u = new SpeechSynthesisUtterance(chunks[chunkIdx]);
      u.rate = 0.93; u.pitch = 1.0; u.volume = 1.0;
      if (pref) u.voice = pref;
      attachViseme(u);
      u.onstart = function() { applyState('speaking'); };
      u.onend = function() {
        chunkIdx++;
        // Small pause between chunks for natural breathing
        setTimeout(speakChunk, 120);
      };
      u.onerror = function() {
        chunkIdx++;
        setTimeout(speakChunk, 80);
      };
      // Chrome bug workaround: TTS stops after ~15s of continuous speech
      // Chunking already handles this, but also resume if paused
      synth.speak(u);
    }
    speakChunk();
  }
  synth.getVoices().length > 0 ? go() : (synth.onvoiceschanged = go);
}

// ═══════════════════════════════════════════════════════════════════════════
// SPEECH RECOGNITION
// ═══════════════════════════════════════════════════════════════════════════

var SpeechRecognitionAPI = window.SpeechRecognition || window.webkitSpeechRecognition;
var recognition = null;

function setTranscript(text) {
  var el = document.getElementById('transcript');
  if (el) el.textContent = text;
}

// ═══════════════════════════════════════════════════════════════════════════
// WAKE WORD DETECTION — "Hey Buddy" always-on listener
// ═══════════════════════════════════════════════════════════════════════════

var wakeRecognition = null;
var wakeListening = false;
var WAKE_PHRASES = [
  'hey buddy', 'hey body', 'hey budy', 'hey buddi', 'hey badi',
  'a buddy', 'hey birdy', 'hey booty', 'hey betty', 'hey batty',
  'hey baddy', 'hey butty', 'hay buddy', 'hey bunny', 'hey budgie',
  'hey beauty', 'hey budy', 'hey bud', 'he buddy', 'hey buddy hey',
  'okay buddy', 'ok buddy', 'yo buddy', 'hi buddy'
];

function isWakeWord(text) {
  var t = text.toLowerCase().trim();
  for (var wi = 0; wi < WAKE_PHRASES.length; wi++) {
    if (t.includes(WAKE_PHRASES[wi])) return true;
  }
  // Fuzzy: check if "hey" + something starting with "bu" or "bo"
  if (/\bhey\b/.test(t) && /\bbu|bo/.test(t)) return true;
  return false;
}

// Greetings pool for natural variation
var GREETINGS = [
  "Hey there! How can I help you with the network today?",
  "Hello! I'm ready to assist. What would you like to know?",
  "Hi! QoSBuddy here. Go ahead, I'm listening.",
  "Hey! What can I look into for you?",
  "Hello! Ready for your question.",
  "Hey there! What's on your mind?",
  "Hi! What network issue should I investigate?"
];

function speakGreeting(callback) {
  var synth = window.speechSynthesis;
  synth.cancel();
  applyState('speaking');
  var greeting = GREETINGS[Math.floor(Math.random() * GREETINGS.length)];
  setTranscript(greeting);

  var u = new SpeechSynthesisUtterance(greeting);
  u.rate = 0.95; u.pitch = 1.05; u.volume = 1.0;

  // Enqueue visemes for lip sync
  var words = greeting.split(/\s+/);
  for (var gw = 0; gw < words.length; gw++) {
    enqueueWordVisemes(words[gw]);
  }

  function go() {
    var vs = synth.getVoices();
    var pref = vs.find(function(v){
      return v.lang.startsWith('en') &&
        (v.name.includes('Google') || v.name.includes('Natural') || v.name.includes('Samantha'));
    }) || vs.find(function(v){ return v.lang.startsWith('en'); }) || null;
    if (pref) u.voice = pref;

    u.onstart = function() { applyState('speaking'); };
    u.onend = function() {
      jawTarget = 0; currentViseme = 'default';
      visemeQueue.length = 0; visemeCurrent = 0; visemeTarget = 0;
      // Brief pause then start listening for the actual question
      setTimeout(function() {
        if (callback) callback();
      }, 400);
    };
    u.onerror = function() {
      applyState('idle');
      if (callback) callback();
    };
    synth.speak(u);
  }
  synth.getVoices().length > 0 ? go() : (synth.onvoiceschanged = go);
}

function startWakeWordListener() {
  if (!SpeechRecognitionAPI) return;
  if (wakeListening) return;
  // Don't start wake listener if already actively doing something
  if (STATE === 'speaking' || STATE === 'processing' || isListening) return;

  wakeRecognition = new SpeechRecognitionAPI();
  wakeRecognition.lang = 'en-US';
  wakeRecognition.interimResults = true;
  wakeRecognition.maxAlternatives = 3;
  wakeRecognition.continuous = true;

  wakeRecognition.onstart = function() {
    wakeListening = true;
    // Show subtle indicator that wake word is active
    var tEl = document.getElementById('transcript');
    if (tEl && STATE === 'idle') {
      tEl.textContent = 'Say "Hey Buddy" to start...';
    }
  };

  wakeRecognition.onresult = function(e) {
    for (var ri = e.resultIndex; ri < e.results.length; ri++) {
      // Check all alternatives for wake word (helps with accent variations)
      for (var ai = 0; ai < e.results[ri].length; ai++) {
        var heard = e.results[ri][ai].transcript;
        if (isWakeWord(heard)) {
          // Wake word detected!
          console.log('[WakeWord] Detected: "' + heard + '"');
          wakeRecognition.stop();
          wakeListening = false;

          // Greet the user, then auto-start listening for their question
          speakGreeting(function() {
            toggleListen();
          });
          return;
        }
      }
    }
  };

  wakeRecognition.onerror = function(e) {
    wakeListening = false;
    // Auto-restart after errors (except "not-allowed" which means mic denied)
    if (e.error !== 'not-allowed' && e.error !== 'service-not-allowed') {
      setTimeout(startWakeWordListener, 2000);
    }
  };

  wakeRecognition.onend = function() {
    wakeListening = false;
    // Auto-restart wake listener if we're idle
    if (STATE === 'idle' || STATE === 'ready') {
      setTimeout(startWakeWordListener, 800);
    }
  };

  try {
    wakeRecognition.start();
  } catch(e) {
    wakeListening = false;
    setTimeout(startWakeWordListener, 3000);
  }
}

// Stop wake listener when doing other speech activities
function stopWakeWordListener() {
  if (wakeRecognition) {
    try { wakeRecognition.stop(); } catch(e) {}
    wakeListening = false;
  }
}

function toggleListen() {
  if (!SpeechRecognitionAPI) {
    setTranscript('Voice requires Google Chrome.');
    return;
  }
  if (STATE === 'processing') return;

  if (isListening) {
    if (recognition) recognition.stop();
    isListening = false;
    applyState('idle');
    // Restart wake word listener
    setTimeout(startWakeWordListener, 1000);
    return;
  }

  // Stop wake word listener while we listen for the question
  stopWakeWordListener();

  recognition = new SpeechRecognitionAPI();
  recognition.lang = 'en-US';
  recognition.interimResults = true;
  recognition.maxAlternatives = 1;
  recognition.continuous = false;

  recognition.onstart = function() {
    isListening = true;
    applyState('listening');
    setTranscript('Listening\u2026');
  };

  recognition.onresult = function(e) {
    var interim = '', finalText = '';
    for (var i = e.resultIndex; i < e.results.length; i++) {
      var tx = e.results[i][0].transcript;
      if (e.results[i].isFinal) finalText += tx;
      else interim += tx;
    }
    setTranscript(finalText || interim || '\u2026');

    if (finalText) {
      var query = finalText.trim();
      isListening = false;
      applyState('processing');
      setTranscript('\u201c' + query + '\u201d');

      fetch('http://localhost:8000/voice/transcript', {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({ text: query })
      })
      .then(function(r) {
        if (!r.ok) { setTranscript('FastAPI error ' + r.status); applyState('idle'); return null; }
        return r.json();
      })
      .then(function(d) {
        if (d) console.log('[HoloFace v4] Delivered: ' + d.text);
      })
      .catch(function() {
        setTranscript('FastAPI not reachable.');
        applyState('idle');
      });
    }
  };

  recognition.onerror = function(e) {
    setTranscript('Mic error: ' + e.error);
    isListening = false;
    applyState('idle');
    // Restart wake word listener
    setTimeout(startWakeWordListener, 1500);
  };

  recognition.onend = function() {
    isListening = false;
    if (STATE === 'listening') applyState('idle');
    // Restart wake word listener after question is done
    if (STATE === 'idle' || STATE === 'ready') {
      setTimeout(startWakeWordListener, 1500);
    }
  };

  recognition.start();
}

// Also restart wake listener when TTS speaking finishes
var _origApplyState = applyState;
applyState = function(s, txt) {
  _origApplyState(s, txt);
  if (s === 'idle' || s === 'ready') {
    setTimeout(startWakeWordListener, 1500);
  } else {
    stopWakeWordListener();
  }
};

// Auto-start wake word listener on page load
setTimeout(startWakeWordListener, 2000);

console.log('[HoloFace v8] Wake word "Hey Buddy" active + GLTF-class mesh, SSAO+Bloom, SSS, Viseme LipSync, Corneal Reflection');
</script>
</body>
</html>

"""



def _tts_html(answer):
    safe     = answer.replace("\\", "\\\\").replace("`", "\\`").replace("</script>", "<\\/script>")
    return (
        "<!DOCTYPE html><html><body><script>"
        "(function(){"
        "var synth=window.speechSynthesis;"
        "if(!synth){return;}"
        "synth.cancel();"
        "var raw=`" + safe + "`;"
        "var clean=raw.replace(/\\*\\*/g,'').replace(/\\n{2,}/g,'. ').replace(/\\n/g,' ');"
        "clean=clean.replace(/[\\u{1F600}-\\u{1F9FF}]|[\\u{2600}-\\u{27BF}]|[\\u{1F300}-\\u{1F5FF}]|[\\u{1F680}-\\u{1F6FF}]/gu,'');"
        # Split into chunks
        "var sentences=clean.split(/(?<=[.!?])\\s+/);"
        "var chunks=[],cur='';"
        "for(var i=0;i<sentences.length;i++){"
        "  if((cur+' '+sentences[i]).length>250&&cur.length>0){chunks.push(cur.trim());cur=sentences[i];}"
        "  else{cur+=(cur?' ':'')+sentences[i];}"
        "}"
        "if(cur.trim())chunks.push(cur.trim());"
        "if(!chunks.length)chunks.push(clean.substring(0,250));"
        "var ci=0;"
        "function go(){"
        "  var vs=synth.getVoices();"
        "  var pref=vs.find(function(v){return v.lang.startsWith('en')&&"
        "    (v.name.includes('Google')||v.name.includes('Natural')||v.name.includes('Samantha'));})"
        "    ||vs.find(function(v){return v.lang.startsWith('en');})||null;"
        "  function speakNext(){"
        "    if(ci>=chunks.length){"
        # Signal head: TTS done
        "      var fs=window.parent.document.querySelectorAll('iframe');"
        "      for(var i=0;i<fs.length;i++){try{"
        "        fs[i].contentWindow.postMessage({type:'qosbuddy_tts_done'},'*');"
        "      }catch(e){}}"
        "      return;"
        "    }"
        "    var u=new SpeechSynthesisUtterance(chunks[ci]);"
        "    u.rate=0.93;u.pitch=1.0;u.volume=1.0;"
        "    if(pref)u.voice=pref;"
        "    u.onstart=function(){"
        "      if(ci===0){"
        "        var fs=window.parent.document.querySelectorAll('iframe');"
        "        for(var i=0;i<fs.length;i++){try{"
        "          fs[i].contentWindow.postMessage({type:'qosbuddy_tts_start'},'*');"
        "        }catch(e){}}"
        "      }"
        "    };"
        "    u.onend=function(){ci++;setTimeout(speakNext,120);};"
        "    u.onerror=function(){ci++;setTimeout(speakNext,80);};"
        "    synth.speak(u);"
        "  }"
        "  speakNext();"
        "}"
        "vs=synth.getVoices();"
        "vs.length>0?go():(synth.onvoiceschanged=go);"
        "})();"
        "</script></body></html>"
    )


def page_voice_assistant():
    """3D WebGL Voice NOC Assistant — Three.js holographic head"""

    st.markdown("""
<style>
/* fonts loaded via system fallbacks */
.vp-hud{display:flex;align-items:center;justify-content:space-between;
  padding:8px 0 16px;border-bottom:1px solid rgba(0,212,255,0.1);margin-bottom:16px;}
.vp-hud-title{font-family:'Syne',sans-serif;font-size:22px;font-weight:800;
  color:#e8f0fe;letter-spacing:-0.02em;}
.vp-hud-sub{font-family:'IBM Plex Mono',monospace;font-size:9px;
  color:rgba(0,212,255,0.4);text-transform:uppercase;letter-spacing:0.2em;margin-top:2px;}
.vp-badge{display:flex;align-items:center;gap:6px;font-family:'IBM Plex Mono',monospace;
  font-size:9px;letter-spacing:0.18em;color:rgba(0,212,255,0.45);text-transform:uppercase;}
.vp-dot{width:7px;height:7px;border-radius:50%;background:#00e5a0;
  box-shadow:0 0 8px #00e5a0;animation:vpdot 2s ease-in-out infinite;}
@keyframes vpdot{0%,100%{opacity:1;transform:scale(1)}50%{opacity:.5;transform:scale(.8)}}
.ctrl-glass{background:rgba(0,8,22,0.78);border:1px solid rgba(0,212,255,0.13);
  border-radius:18px;padding:20px;backdrop-filter:blur(20px);position:relative;overflow:hidden;}
.ctrl-glass::before{content:'';position:absolute;top:0;left:0;right:0;height:1px;
  background:linear-gradient(90deg,transparent,rgba(0,212,255,0.28),transparent);}
.ctrl-lbl{font-family:'IBM Plex Mono',monospace;font-size:8px;letter-spacing:0.22em;
  text-transform:uppercase;color:rgba(0,212,255,0.28);margin-bottom:8px;}
.listen-active{display:flex;align-items:center;gap:10px;padding:12px 16px;
  border-radius:12px;border:1.5px solid rgba(0,229,160,0.55);
  background:rgba(0,229,160,0.06);margin-bottom:12px;}
.listen-dot{width:9px;height:9px;border-radius:50%;background:#00e5a0;
  box-shadow:0 0 12px #00e5a0;animation:vpdot 1s ease-in-out infinite;flex-shrink:0;}
.listen-txt{font-family:'IBM Plex Mono',monospace;font-size:11px;
  color:#00e5a0;letter-spacing:0.1em;}
.resp-panel{position:relative;
  background:linear-gradient(135deg,rgba(0,8,22,.94),rgba(3,14,32,.90));
  border:1px solid rgba(0,212,255,0.25);border-radius:20px;overflow:hidden;
  backdrop-filter:blur(24px);animation:resp-in .5s cubic-bezier(.16,1,.3,1) forwards;
  box-shadow:0 0 0 1px rgba(0,212,255,.05),0 20px 50px rgba(0,0,0,.5),
             inset 0 1px 0 rgba(0,212,255,.08);}
@keyframes resp-in{from{opacity:0;transform:translateY(18px) scale(.97)}
                   to  {opacity:1;transform:translateY(0) scale(1)}}
.resp-bar{height:2px;background:linear-gradient(90deg,transparent,#00d4ff 30%,
  #00e5a0 60%,#a78bfa 80%,transparent);box-shadow:0 0 10px rgba(0,212,255,.5);}
.resp-sw{position:absolute;top:0;left:-100%;width:60%;height:100%;pointer-events:none;
  background:linear-gradient(90deg,transparent,rgba(0,212,255,.025),transparent);
  animation:sw 4s linear infinite;}
@keyframes sw{0%{left:-100%}100%{left:200%}}
.resp-c{position:absolute;width:12px;height:12px;animation:cp 2.5s ease-in-out infinite;}
@keyframes cp{0%,100%{opacity:.35}50%{opacity:.9}}
.resp-c.tl{top:8px;left:8px;border-top:1.5px solid #00d4ff;border-left:1.5px solid #00d4ff;border-radius:3px 0 0 0;}
.resp-c.tr{top:8px;right:8px;border-top:1.5px solid #00d4ff;border-right:1.5px solid #00d4ff;border-radius:0 3px 0 0;}
.resp-c.bl{bottom:8px;left:8px;border-bottom:1.5px solid #00d4ff;border-left:1.5px solid #00d4ff;border-radius:0 0 0 3px;}
.resp-c.br{bottom:8px;right:8px;border-bottom:1.5px solid #00d4ff;border-right:1.5px solid #00d4ff;border-radius:0 0 3px 0;}
.resp-body{padding:22px 26px 20px;}
.resp-meta{display:flex;align-items:center;gap:8px;margin-bottom:14px;
  font-family:'IBM Plex Mono',monospace;font-size:9px;
  color:rgba(0,212,255,.4);text-transform:uppercase;letter-spacing:.2em;}
.resp-dot{width:5px;height:5px;border-radius:50%;background:#00d4ff;
  box-shadow:0 0 8px #00d4ff;animation:cp 1.2s ease-in-out infinite;flex-shrink:0;}
.resp-text{font-family:Inter,sans-serif;font-size:14px;
  color:rgba(228,240,255,.92);line-height:1.85;}
.hist-item{padding:12px 16px;border-radius:14px;
  border:1px solid rgba(0,212,255,.07);background:rgba(5,12,26,.65);margin-bottom:8px;
  transition:all .3s ease;}
.hist-item:hover{border-color:rgba(0,212,255,.18);background:rgba(5,12,26,.8);}
/* Chat input styling */
[data-testid="stChatInput"]{
  border:1.5px solid rgba(0,212,255,.2) !important;
  border-radius:16px !important;
  background:rgba(0,8,22,.85) !important;
  backdrop-filter:blur(12px) !important;
  box-shadow:0 0 20px rgba(0,212,255,.04), inset 0 1px 0 rgba(0,212,255,.06) !important;
  transition:all .3s ease !important;}
[data-testid="stChatInput"]:focus-within{
  border-color:rgba(0,212,255,.45) !important;
  box-shadow:0 0 25px rgba(0,212,255,.1), inset 0 1px 0 rgba(0,212,255,.1) !important;}
[data-testid="stChatInput"] textarea{
  font-family:'IBM Plex Mono',monospace !important;
  font-size:13px !important;
  color:rgba(228,240,255,.9) !important;
  letter-spacing:.02em !important;}
[data-testid="stChatInput"] textarea::placeholder{
  color:rgba(0,212,255,.25) !important;
  font-style:italic !important;}
[data-testid="stChatInputSubmitButton"] button{
  background:rgba(0,212,255,.15) !important;
  border:1px solid rgba(0,212,255,.3) !important;
  border-radius:10px !important;}
[data-testid="stChatInputSubmitButton"] button:hover{
  background:rgba(0,212,255,.25) !important;}
</style>
""", unsafe_allow_html=True)

    # ── Session state ─────────────────────────────────────────────────────────
    for k, v in [("voice_history",[]),("voice_answer",""),
                 ("voice_question",""),("voice_auto_tts",True),
                 ("voice_armed",False)]:
        if k not in st.session_state:
            st.session_state[k] = v

    collection, embed_model, chain, rag_ready = get_rag_resources()
    API_URL = os.getenv("QOSBUDDY_API_URL", "http://localhost:8000")

    # ── Helper: push state to FastAPI so head JS can poll it ─────────────────
    def push_state(state, preview=""):
        try:
            import urllib.request as _ur, json as _js
            _data = _js.dumps({"state": state,
                                "answer_preview": preview[:80]}).encode()
            _req  = _ur.Request(
                f"{API_URL}/voice/state",
                data=_data,
                headers={"Content-Type":"application/json"},
                method="POST")
            _ur.urlopen(_req, timeout=2)
        except Exception:
            pass

    # ── Poll FastAPI for transcript ───────────────────────────────────────────
    spoken_q = None
    api_ok   = False
    try:
        import urllib.request as _ur, json as _js
        with _ur.urlopen(
            _ur.Request(f"{API_URL}/voice/transcript",
                        headers={"Accept":"application/json"}),
            timeout=2
        ) as _r:
            _d   = _js.loads(_r.read())
            api_ok = True
            if _d.get("text","").strip():
                spoken_q = _d["text"].strip()
                st.session_state["voice_armed"] = False
    except Exception:
        pass

    # ── HUD ───────────────────────────────────────────────────────────────────
    is_armed = st.session_state["voice_armed"]
    hud_label = ("LISTENING" if is_armed
                 else "\u2713 ANSWER READY" if st.session_state["voice_answer"]
                 else "STANDBY")
    hud_color = ("#00e5a0" if (is_armed or st.session_state["voice_answer"])
                 else "rgba(0,212,255,0.4)")

    st.markdown(f"""
<div class="vp-hud">
  <div>
    <div class="vp-hud-title">\U0001f399\ufe0f Voice NOC Assistant</div>
    <div class="vp-hud-sub">DSO3.2 \u00b7 Three.js WebGL \u00b7 RAG + Ollama \u00b7 Holographic 3D Interface</div>
  </div>
  <div class="vp-badge">
    <div class="vp-dot" style="background:{hud_color};box-shadow:0 0 8px {hud_color};"></div>
    {hud_label}
  </div>
</div>
""", unsafe_allow_html=True)

    col_head, col_ctrl = st.columns([3, 2], gap="large")

    with col_head:
        # 3D head rendered at 540px height for proper 3D perspective
        st.components.v1.html(_voice_orb_html(), height=540, scrolling=False)
        # Suggestion chips
        chip_cols = st.columns(3)
        for i, s in enumerate(VOICE_SUGGESTIONS):
            if chip_cols[i%3].button(s, key=f"vchip_{i}", use_container_width=True):
                st.session_state["_voice_chip"] = s
                st.rerun()
        typed = st.chat_input("Ask QoSBuddy anything about the network...", key="voice_text_input")

    with col_ctrl:
        st.markdown("<div class='ctrl-glass'>", unsafe_allow_html=True)

        # Listen / Stop
        if is_armed:
            st.markdown(
                "<div class='listen-active'>"
                "<div class='listen-dot'></div>"
                "<div class='listen-txt'>LISTENING FOR VOICE</div>"
                "</div>", unsafe_allow_html=True)
            if st.button("\u23f9\ufe0f Stop Listening", key="voice_stop",
                         use_container_width=True):
                st.session_state["voice_armed"] = False
                push_state("idle")
                st.rerun()
        else:
            st.button("\U0001f399\ufe0f Start Listening", key="voice_arm",
                      use_container_width=True,
                      on_click=lambda: st.session_state.update({"voice_armed":True}))

        # Status card
        st.markdown("<br>", unsafe_allow_html=True)
        if not api_ok:
            st.warning("FastAPI not reachable — rebuild Docker container")
        elif spoken_q:
            st.markdown(
                f"<div style='padding:10px 14px;border-radius:10px;"
                f"border:1px solid rgba(0,229,160,.5);background:rgba(0,229,160,.07);"
                f"font-family:IBM Plex Mono,monospace;font-size:10px;color:#00e5a0;'>"
                f"\u2713 Transcript received: \"{spoken_q[:38]}\"</div>",
                unsafe_allow_html=True)
        elif is_armed:
            st.markdown(
                "<div style='font-family:IBM Plex Mono,monospace;font-size:9px;"
                "color:rgba(0,212,255,.4);padding:6px 0;letter-spacing:.15em;'>"
                "\u2022 Polling FastAPI every 2s\u2026</div>",
                unsafe_allow_html=True)

        st.markdown("<div style='height:1px;background:rgba(0,212,255,.08);margin:14px 0;'></div>",
                    unsafe_allow_html=True)
        st.markdown("<div class='ctrl-lbl'>Query Settings</div>", unsafe_allow_html=True)
        mode = st.selectbox("Mode",["general","anomaly","causal","domain"],
            format_func=lambda x:{
                "general":"\U0001f310 General",
                "anomaly":"\U0001f6a8 Anomaly",
                "causal": "\U0001f52c Causal",
                "domain": "\U0001f4d6 Knowledge"}[x],
            key="voice_mode", label_visibility="collapsed")
        sev_flt = st.selectbox("Severity",["None","critical","degraded","normal"],
            key="voice_sev", label_visibility="collapsed")
        sev_flt = None if sev_flt=="None" else sev_flt
        rc_flt  = st.selectbox("Root Cause",
            ["None","interference","congestion","mobility","combined/other"],
            key="voice_rc", label_visibility="collapsed")
        rc_flt = None if rc_flt=="None" else rc_flt
        auto_tts = st.toggle("\U0001f50a Speak answers aloud",
            value=st.session_state["voice_auto_tts"], key="voice_tts_toggle")
        st.session_state["voice_auto_tts"] = auto_tts

        st.markdown("<div style='height:1px;background:rgba(0,212,255,.08);margin:14px 0;'></div>",
                    unsafe_allow_html=True)
        st.markdown(
            "<div class='ctrl-lbl'>How to use</div>"
            "<div style='font-family:Inter,sans-serif;font-size:12px;"
            "color:rgba(232,240,254,.42);line-height:2.1;'>"
            "1. Say <b style='color:#00d4ff'>\"Hey Buddy\"</b> to wake up<br>"
            "2. QoSBuddy greets you and starts listening<br>"
            "3. Ask your NOC question clearly<br>"
            "4. Answer displayed + spoken aloud with lip sync<br>"
            "5. Or type below / click a suggestion chip<br><br>"
            "<span style='color:rgba(255,184,63,.5);font-size:10px;'>"
            "\u26a0\ufe0f Chrome \u00b7 Allow microphone"
            "</span></div>", unsafe_allow_html=True)

        if st.session_state["voice_history"]:
            st.markdown("<div style='margin-top:14px;'>", unsafe_allow_html=True)
            if st.button("\U0001f5d1 Clear History", key="voice_clear",
                         use_container_width=True):
                for k in ["voice_history","voice_answer","voice_question"]:
                    st.session_state[k] = [] if k=="voice_history" else ""
                push_state("idle")
                st.rerun()
            st.markdown("</div>", unsafe_allow_html=True)

        st.markdown("</div>", unsafe_allow_html=True)

    # ── Resolve question ──────────────────────────────────────────────────────
    question = (
        spoken_q
        or st.session_state.pop("_voice_chip", None)
        or typed
        or None
    )

    # ── RAG pipeline ─────────────────────────────────────────────────────────
    if question:
        # Clear previous answer immediately
        st.session_state["voice_answer"] = ""
        st.session_state["voice_question"] = question
        push_state("processing")
        with st.spinner("\u26a1 Querying RAG pipeline..."):
            if rag_ready:
                try:
                    from rag.rag_pipeline import query_rag
                    result = query_rag(
                        question=question, collection=collection,
                        embed_model=embed_model, chain=chain,
                        mode=mode, severity_filter=sev_flt, root_cause_filter=rc_flt)
                    answer  = result.get("answer", "(empty answer)") if isinstance(result, dict) else str(result)
                    sources = result.get("sources", []) if isinstance(result, dict) else []
                except Exception as e:
                    answer  = f"\u26a0\ufe0f RAG query failed: {e}. Check that Ollama is reachable."
                    sources = []
            else:
                answer  = "\u26a0\ufe0f RAG not initialised. Run build_knowledge_base.py first."
                sources = []
        st.session_state["voice_answer"] = answer
        st.session_state["voice_question"] = question
        st.session_state["voice_history"].insert(0, {
            "question":question, "answer":answer,
            "sources":sources, "time":datetime.now().strftime("%H:%M:%S"),
        })
        # Push answer to FastAPI — avatar JS will poll and speak it
        try:
            import urllib.request as _ur2, json as _js2
            _ans_data = _js2.dumps({"text": answer}).encode()
            _ans_req  = _ur2.Request(
                f"{API_URL}/voice/answer",
                data=_ans_data,
                headers={"Content-Type":"application/json"},
                method="POST")
            _ur2.urlopen(_ans_req, timeout=2)
        except Exception:
            pass
        preview = answer.replace("**","").replace("\n"," ")[:80]
        if auto_tts:
            push_state("speaking", preview)
        else:
            push_state("ready", preview)

    # ── Holographic Response Panel — latest answer ──────────────────────────
    if st.session_state["voice_answer"]:
        st.markdown("<br>", unsafe_allow_html=True)
        ans      = st.session_state["voice_answer"]
        q        = st.session_state["voice_question"]
        hist0    = st.session_state["voice_history"]
        n_src    = len(hist0[0].get("sources",[])) if hist0 else 0
        src_list = hist0[0].get("sources",[])       if hist0 else []

        def _e(t):
            return t.replace("&","&amp;").replace("<","&lt;").replace(">","&gt;")

        COL_MAP = {"ANOMALY_EVENT":"#ff4d6d","CAUSAL_INSIGHT":"#00e5a0",
                   "SLA_RISK":"#ffb83f","DOMAIN_CONTEXT":"#a78bfa"}
        badges = ""
        for s in src_list[:6]:
            if not isinstance(s, dict): continue
            m  = s.get("metadata") or {}
            dt = m.get("doc_type","?") or "?"
            bc = COL_MAP.get(dt,"#00d4ff")
            sim = s.get("similarity")
            sim_str = f"{sim:.2f}" if isinstance(sim,(int,float)) else "—"
            badges += (
                f"<span style='display:inline-flex;align-items:center;gap:5px;"
                f"background:rgba(0,0,0,.3);border:1px solid {bc}33;"
                f"border-radius:8px;padding:4px 10px;margin:3px;'>"
                f"<span style='width:5px;height:5px;border-radius:50%;"
                f"background:{bc};box-shadow:0 0 5px {bc};flex-shrink:0;'></span>"
                f"<span style='font-family:IBM Plex Mono,monospace;font-size:9px;"
                f"color:{bc};'>{_e(dt[:12])}</span>"
                f"<span style='font-size:9px;color:rgba(200,220,255,.4);'>"
                f"sim={sim_str}</span></span>"
            )

        src_block = (
            "<div style='height:1px;background:linear-gradient(90deg,"
            "transparent,rgba(0,212,255,.15),transparent);margin:14px 0;'></div>"
            f"<div style='font-family:IBM Plex Mono,monospace;font-size:8px;"
            f"color:rgba(0,212,255,.25);text-transform:uppercase;"
            f"letter-spacing:.2em;margin-bottom:8px;'>\U0001f517 {n_src} sources</div>"
            f"<div style='display:flex;flex-wrap:wrap;gap:2px;'>{badges}</div>"
        ) if n_src else ""

        # Format answer text — convert markdown bold to styled spans
        ans_html = _e(ans).replace(chr(10),'<br>')

        st.markdown(
            "<div class='resp-panel'>"
            "<div class='resp-bar'></div>"
            "<div class='resp-sw'></div>"
            "<div class='resp-c tl'></div><div class='resp-c tr'></div>"
            "<div class='resp-c bl'></div><div class='resp-c br'></div>"
            "<div class='resp-body'>"
            "<div class='resp-meta'><div class='resp-dot'></div>"
            "\U0001f4e1 NOC ASSISTANT \u00b7 "
            f"<em style='color:rgba(0,212,255,.6);font-style:normal;'>\"{_e(q)}\"</em></div>"
            f"<div class='resp-text'>{ans_html}</div>"
            + src_block +
            "</div></div>", unsafe_allow_html=True)

        if st.button("\U0001f50a Speak Again", key="voice_speak_btn"):
            try:
                import urllib.request as _ur3, json as _js3
                _ans_data2 = _js3.dumps({"text": ans}).encode()
                _ans_req2  = _ur3.Request(
                    f"{API_URL}/voice/answer",
                    data=_ans_data2,
                    headers={"Content-Type":"application/json"},
                    method="POST")
                _ur3.urlopen(_ans_req2, timeout=2)
                push_state("speaking", ans.replace("**","").replace("\n"," ")[:80])
            except Exception:
                pass

    # ── Collapsible History — previous Q&A pairs ──────────────────────────────
    history = st.session_state["voice_history"]
    if len(history) > 1:
        with st.expander(f"\u27f3 Session History ({len(history)-1} previous)", expanded=False):
            for hi, turn in enumerate(history[1:6]):
                def _e(t):
                    return t.replace("&","&amp;").replace("<","&lt;").replace(">","&gt;")
                prev = _e(turn["answer"][:200])
                ell  = "..." if len(turn["answer"])>200 else ""
                st.markdown(
                    "<div class='hist-item'>"
                    "<div style='font-family:IBM Plex Mono,monospace;font-size:9px;"
                    "color:rgba(0,212,255,.38);margin-bottom:5px;'>"
                    "\U0001f399 " + turn["time"] + " \u00b7 " + _e(turn["question"]) +
                    "</div><div style='font-size:12px;color:rgba(232,240,254,.48);"
                    "line-height:1.6;'>" + prev + ell + "</div></div>",
                    unsafe_allow_html=True)

    # ── Polling loop ──────────────────────────────────────────────────────────
    if st.session_state.get("voice_armed",False) and not spoken_q and not question:
        import time as _t
        _t.sleep(2)
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
    df = anomaly_df if anomaly_df is not None else pd.DataFrame()

    # Schema-defensive KPI extraction — works for both v1 (raw KPIs) and v2 (scores).
    def _kpi_row(label, col, fmt, scale, hi_test, hi_label="HIGH", ok_label="OK"):
        if col not in df.columns or len(df) == 0:
            return [label, "—", "no data"]
        v = pd.to_numeric(df[col], errors="coerce").mean()
        if pd.isna(v): return [label, "—", "no data"]
        v_s = float(v) * scale
        return [label, fmt.format(v_s), hi_label if hi_test(float(v)) else ok_label]

    kd = [["KPI","Value","Status"],
          _kpi_row("Avg Jitter (ms)",  "jitter_ms",         "{:.2f}",       1,   lambda v: v > 10,  "ELEVATED"),
          _kpi_row("Avg Packet Loss",  "packet_loss_ratio", "{:.1f}%",      100, lambda v: v > 0.3, "HIGH"),
          _kpi_row("Avg Throughput",   "throughput_mbps",   "{:.3f} Mbps",  1,   lambda v: v < 2,   "LOW"),
          _kpi_row("Avg Delay (ms)",   "delay_ms",          "{:.1f}",       1,   lambda v: v > 100, "HIGH"),
          _kpi_row("Anomaly Rate",     "if_anomaly",        "{:.1f}%",      100, lambda v: v > 0.3, "HIGH"),
          ["Critical Events",
           f"{int((df['severity']=='critical').sum()):,}" if "severity" in df.columns else "—",
           ""]]
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
    # v1 and v2 causal_df have different columns — pick what's available.
    _count_col = next((c for c in ("if_anomaly","true_label","root_cause") if c in causal_df.columns), "root_cause")
    _loss_col  = next((c for c in ("packet_loss_ratio","avg_loss","risk_proba") if c in causal_df.columns), None)
    _group_cols = [c for c in ("root_cause","recommended_action") if c in causal_df.columns]
    if _group_cols:
        agg_map = {"count": (_count_col, "count")}
        if _loss_col:
            agg_map["avg"] = (_loss_col, "mean")
        rc_s = causal_df.groupby(_group_cols).agg(**agg_map).reset_index()
    else:
        rc_s = pd.DataFrame(columns=["root_cause","recommended_action","count","avg"])
    rd = [["Root Cause","Action","Events","Avg Loss"]]
    for _, r in rc_s.iterrows():
        avg_txt = f"{r['avg']*100:.1f}%" if "avg" in rc_s.columns and pd.notna(r.get("avg")) else "—"
        rd.append([
            r.get("root_cause","—"),
            r.get("recommended_action","—"),
            str(r["count"]),
            avg_txt,
        ])
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
        # Pick whichever critical-event subset we can build from the available schema.
        if anomaly_df is not None and "severity" in anomaly_df.columns:
            top = anomaly_df[anomaly_df["severity"]=="critical"].head(15)
        elif anomaly_df is not None and "true_label" in anomaly_df.columns:
            top = anomaly_df[anomaly_df["true_label"]==1].head(15)
        else:
            top = anomaly_df.head(15) if anomaly_df is not None else pd.DataFrame()
        c_show = [c for c in ["ue_id","timestamp","window_idx","severity","true_label",
                              "if_score","packet_loss_ratio","sinr_dl_db"] if c in top.columns]
        if not top.empty and c_show:
            try:
                _slice = top[c_show].copy()
                # round only numeric cols
                for c in _slice.select_dtypes(include=[np.number]).columns:
                    _slice[c] = _slice[c].round(3)
                raw = [c_show] + _slice.astype(str).values.tolist()
                raw_t = Table(raw)
                raw_t.setStyle(TableStyle([("BACKGROUND",(0,0),(-1,0),colors.HexColor("#080c12")),("TEXTCOLOR",(0,0),(-1,0),colors.white),("FONTSIZE",(0,0),(-1,-1),7),("GRID",(0,0),(-1,-1),0.3,colors.grey),("PADDING",(0,0),(-1,-1),4)]))
                story.append(raw_t)
            except Exception as e:
                story.append(Paragraph(f"(critical-anomaly sample unavailable: {e})", body_s))
        else:
            story.append(Paragraph("(no critical-anomaly rows in current dataset)", body_s))
    doc.build(story)
    return buf.getvalue()


# ══════════════════════════════════════════════════════════════════════════════
#  MAIN
# ══════════════════════════════════════════════════════════════════════════════

def main():
    anomaly_static = load_anomaly_df()
    causal_df      = load_causal_df()
    cf_df          = load_cf_df()
    sla_df         = load_sla_df()
    severity_df    = load_severity_df()
    bench_df       = load_bench_df()

    # Live-data promotion: when the bridge has streamed enough rows, swap the
    # static anomaly_df for the live buffer so other pages reflect real-time state.
    anomaly_df = get_active_anomaly_df(anomaly_static)

    # Restore page from URL on hard refresh; valid keys come from NAV_ITEMS.
    _url_page = st.query_params.get("page")
    _valid_pages = {k for k, _, _ in NAV_ITEMS}
    _default_page = _url_page if _url_page in _valid_pages else "voice"
    for key, default in [("page",_default_page),("sev_f",None),("ue_f",None),("chat_history",[]),("theme","dark")]:
        if key not in st.session_state:
            st.session_state[key] = default

    page = render_sidebar()

    if   page == "kpi":        page_kpi_overview(anomaly_df, causal_df)
    elif page == "anomaly":    page_anomaly_feed(anomaly_df)
    elif page == "causal":     page_causal_root_cause(causal_df, cf_df)
    elif page == "sla_risk":   page_sla_risk(sla_df)
    elif page == "severity":   page_severity_triage(severity_df)
    elif page == "benchmark":  page_benchmarking(bench_df)
    elif page == "forecast":   page_qos_forecast()
    elif page == "rl":         page_rl_actions(anomaly_df, causal_df)
    elif page == "live_monitor": page_live_monitor()
    elif page == "chat":       page_rag_chat()
    elif page == "voice":      page_voice_assistant()
    elif page == "export":     page_export_report(anomaly_df, causal_df)

    # Floating companion is present on every page except the full Voice NOC
    # page, which has its own large 3D avatar and would conflict.
    if page != "voice":
        render_companion(page)


if __name__ == "__main__":
    main()