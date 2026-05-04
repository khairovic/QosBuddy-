"""
QoSBuddy M6 — FastAPI Backend v2 (DSO3.2)
===========================================
REST endpoints consumed by:
  • Streamlit dashboard → GET /kpi/summary, GET /anomalies, POST /rag/query
  • External NOC systems → GET /health, GET /docs (Swagger UI)
  • M1/M2 pipeline → POST /ingest/anomaly (push new anomaly events)

v2 changes:
  • Loads anomaly_scores_v2.csv (multi-model scores) with v1 fallback
  • Loads updated root_cause_labels.csv (detection-based root causes)
  • New endpoint: GET /models/benchmark — returns model comparison data
  • New endpoint: GET /anomalies/v2 — returns v2 anomaly window data
  • Unified data loader with version detection
"""

import logging
import os
from collections import deque
import threading
from contextlib import asynccontextmanager
from datetime import datetime
from pathlib import Path
from typing import Optional, List, Dict, Any

import pandas as pd
import numpy as np
from fastapi import FastAPI, BackgroundTasks, HTTPException, Query
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field

log = logging.getLogger("qosbuddy.api")
logging.basicConfig(level=logging.INFO)

# ─────────────────────────────────────────────
# Global State
# ─────────────────────────────────────────────

_collection    = None
_embed_model   = None
_chain         = None
_anomaly_df    = None   # v1: anomaly_scores_ns3.xls
_anomaly_v2_df = None   # v2: anomaly_scores_v2.csv (multi-model)
_causal_df     = None   # root_cause_labels.csv (v1 or v2)
_cf_df         = None   # counterfactuals.csv
_sla_df        = None   # sla_breach_predictions.csv
_severity_df   = None   # severity_triage.csv
_bench_df      = None   # benchmark_report_by_category.csv
_rl_df         = None   # rl_vs_baseline_comparison.csv
_rl_policy     = None   # rl_agent.RLPolicy — PPO agent (DSO3.1)
_jitter_pred_df = None  # predicted_jitter.csv (M2, DSO1.1)
_forecast_ci_df = None  # forecast_ci.csv (M2, Prophet CI per UE/metric)
_shap_df       = None   # shap_importance_ns3.xls (M1 SHAP explanations)
_dowhy_model   = None   # dowhy_model.pkl (M3, live counterfactual)
_anomaly_models = {}    # v2 supervised models (IF, XGB) loaded from anomaly_models/
_data_version  = "v1"   # "v1" or "v2"

# Live simulation buffer (circular, keeps last 500 rows from bridge.py)
_LIVE_BUFFER_SIZE = 500
_live_buffer: deque = deque(maxlen=_LIVE_BUFFER_SIZE)
_live_lock = threading.Lock()
_live_stats = {
    "total_received": 0,
    "last_received_ts": None,
    "bridge_connected": False,
    "rows_in_buffer": 0,
}

DATA_DIR = Path(os.getenv("QOSBUDDY_DATA_DIR", "./data"))

# ── Live data persistence ────────────────────────────────────────────────
# Append-only JSONL log of every row received from the bridge. Survives
# container restarts and is read directly by the dashboard for the cumulative
# augmented dataset that powers the cross-page analytics.
LIVE_LOG_PATH = DATA_DIR / "live_log.jsonl"
_live_log_lock = threading.Lock()


def _append_live_log(scored_rows: List[dict]) -> int:
    """Append scored rows to the JSONL log. Returns the number of lines written."""
    if not scored_rows:
        return 0
    import json as _json
    try:
        DATA_DIR.mkdir(parents=True, exist_ok=True)
        # Drop non-JSON-serializable values (numpy types, etc.) defensively.
        def _clean(r):
            out = {}
            for k, v in r.items():
                try:
                    _json.dumps(v)
                    out[k] = v
                except (TypeError, ValueError):
                    out[k] = str(v)
            return out
        with _live_log_lock, open(LIVE_LOG_PATH, "a", encoding="utf-8") as f:
            for r in scored_rows:
                f.write(_json.dumps(_clean(r), ensure_ascii=False) + "\n")
        return len(scored_rows)
    except Exception as e:
        log.warning(f"live_log append failed: {e}")
        return 0


def _count_live_log_lines() -> int:
    """Cheap line count for restoring the cumulative counter on startup."""
    if not LIVE_LOG_PATH.exists():
        return 0
    try:
        with open(LIVE_LOG_PATH, "rb") as f:
            return sum(1 for _ in f)
    except Exception:
        return 0


def _safe_load_csv(filename: str, label: str) -> Optional[pd.DataFrame]:
    """Load a CSV file safely, returning None on failure."""
    path = DATA_DIR / filename
    if not path.exists():
        log.warning(f"{label}: {filename} not found")
        return None
    try:
        df = pd.read_csv(path)
        log.info(f"{label}: {len(df)} rows loaded")
        return df
    except Exception as e:
        log.warning(f"{label}: failed to load — {e}")
        return None


@asynccontextmanager
async def lifespan(app: FastAPI):
    """Load all heavy resources on startup; release on shutdown."""
    global _collection, _embed_model, _chain
    global _anomaly_df, _anomaly_v2_df, _causal_df, _cf_df
    global _sla_df, _severity_df, _bench_df, _rl_df, _rl_policy, _data_version
    global _jitter_pred_df, _forecast_ci_df, _shap_df, _dowhy_model, _anomaly_models

    log.info("QoSBuddy M6 API v2 starting up…")

    # ── Restore cumulative counter from persistent live log ──
    try:
        prior = _count_live_log_lines()
        if prior > 0:
            _live_stats["total_received"] = prior
            log.info(f"Live log: restored {prior:,} cumulative rows from {LIVE_LOG_PATH.name}")
    except Exception as e:
        log.warning(f"Live log restore skipped: {e}")

    # ── Load anomaly scores (prefer v2, fallback v1) ──
    _anomaly_v2_df = _safe_load_csv("anomaly_scores_v2.csv", "Anomaly scores v2")
    if _anomaly_v2_df is not None:
        _data_version = "v2"
        log.info("Using anomaly_scores_v2.csv (multi-model format)")
    else:
        _data_version = "v1"

    # Always try to load v1 for backward compatibility
    v1_path = DATA_DIR / "anomaly_scores_ns3.xls"
    if v1_path.exists():
        try:
            _anomaly_df = pd.read_csv(v1_path)
            dupe_cols = [c for c in _anomaly_df.columns if c.endswith(".1")]
            _anomaly_df = _anomaly_df.drop(columns=dupe_cols, errors="ignore")
            if "ae_error" in _anomaly_df.columns:
                _anomaly_df["ae_error"] = _anomaly_df["ae_error"].fillna(0.0)
            log.info(f"Anomaly scores v1: {len(_anomaly_df)} rows loaded")
        except Exception as e:
            log.warning(f"v1 anomaly scores: {e}")

    # ── Load root cause labels ──
    _causal_df = _safe_load_csv("root_cause_labels.csv", "Root cause labels")
    if _causal_df is not None:
        # v1 format may need ue_id injection
        if "ue_id" not in _causal_df.columns and _anomaly_df is not None:
            if len(_causal_df) == len(_anomaly_df):
                _causal_df.insert(0, "ue_id", _anomaly_df["ue_id"].values)
                log.info("Injected ue_id into causal_df from v1 anomaly data")

    # ── Load counterfactuals ──
    _cf_df = _safe_load_csv("counterfactuals.csv", "Counterfactuals")
    if _cf_df is not None and "timestamp" in _cf_df.columns:
        _cf_df = _cf_df[_cf_df["timestamp"] != "timestamp"].reset_index(drop=True)
        _cf_df["timestamp"] = pd.to_numeric(
            _cf_df["timestamp"], errors="coerce"
        ).fillna(1).astype(int)

    # ── Load optional upstream files ──
    _sla_df      = _safe_load_csv("sla_breach_predictions.csv", "SLA breach predictions")
    _severity_df = _safe_load_csv("severity_triage.csv", "Severity triage")
    _bench_df    = _safe_load_csv("benchmark_report_by_category.csv", "Benchmark report")
    _rl_df       = _safe_load_csv("rl_vs_baseline_comparison.csv", "RL comparison")

    # ── M2 DSO1.1 — jitter/throughput forecasts ──
    _jitter_pred_df = _safe_load_csv("predicted_jitter.csv", "M2 jitter predictions")
    _forecast_ci_df = _safe_load_csv("forecast_ci.csv",     "M2 Prophet forecasts")

    # ── M1 SHAP explanations (optional xls) ──
    shap_path = DATA_DIR / "shap_importance_ns3.xls"
    if shap_path.exists():
        try:
            _shap_df = pd.read_csv(shap_path)
            log.info(f"SHAP importance: {len(_shap_df)} rows loaded")
        except Exception as e:
            try:
                _shap_df = pd.read_excel(shap_path)
                log.info(f"SHAP importance: {len(_shap_df)} rows loaded (excel)")
            except Exception as e2:
                log.warning(f"SHAP importance load failed: {e2}")

    # ── Load RAG stack ──
    try:
        from rag.rag_pipeline import (
            get_chroma_collection, get_embed_model, build_langchain_rag_chain
        )
        _collection  = get_chroma_collection()
        _embed_model = get_embed_model()
        _chain       = build_langchain_rag_chain(_collection, _embed_model)
        log.info(f"RAG stack ready — {_collection.count()} documents in KB")
    except Exception as e:
        log.warning(f"RAG stack not available: {e}")

    # ── Load PPO RL agent (DSO3.1 — M1 handoff) ──
    try:
        from rl_agent import RLPolicy
        _rl_policy = RLPolicy()
        _rl_policy.load()
        if _rl_policy.ready:
            log.info("RL policy ready — PPO loaded on digital twin")
        else:
            log.warning(f"RL policy unavailable: {_rl_policy.error}")
    except Exception as e:
        log.warning(f"RL agent import failed: {e}")

    # ── Load M3 DoWhy model for live counterfactuals ──
    dowhy_path = Path(__file__).resolve().parent.parent / "causal_models" / "dowhy_model.pkl"
    if dowhy_path.exists():
        try:
            import pickle
            with open(dowhy_path, "rb") as f:
                _dowhy_model = pickle.load(f)
            log.info(f"DoWhy model loaded from {dowhy_path.name}")
        except Exception as e:
            log.warning(f"DoWhy model load failed: {e}")
    else:
        log.info("DoWhy model not present (causal_models/dowhy_model.pkl)")

    # ── Load M1 v2 supervised anomaly models for live scoring ──
    # Each pickle is a dict {model, features, threshold|best_cont}. We keep
    # the bundle so /anomaly/score can pad inputs to each model's feature list.
    models_dir = Path(__file__).resolve().parent.parent / "anomaly_models"
    if models_dir.exists():
        for name, rel in [
            ("isolation_forest",  "isolation_forest_ns3.pkl"),
            ("random_forest",     "random_forest_v2.pkl"),
            ("gradient_boosting", "gradient_boosting_v2.pkl"),
        ]:
            fp = models_dir / rel
            if not fp.exists():
                continue
            try:
                import joblib
                _anomaly_models[name] = joblib.load(fp)
                log.info(f"Loaded anomaly model: {name}")
            except Exception as e:
                log.warning(f"Anomaly model {name} load failed: {e}")

    # ── MLOps: register scoring jobs + optionally auto-start scheduler ──
    try:
        from mlops import runner as _mlops_runner, register_default_jobs
        register_default_jobs()
        if _mlops_runner.auto_start_on_boot:
            _mlops_runner.start_scheduler()
            log.info("MLOps scheduler auto-started")
        else:
            log.info("MLOps registered but scheduler is paused (set QOSBUDDY_PIPELINE_AUTO=true to auto-start)")
    except Exception as e:
        log.warning(f"MLOps init skipped: {e}")

    log.info(f"Startup complete. Data version: {_data_version}")
    yield
    log.info("QoSBuddy M6 API shutting down")
    try:
        from mlops import runner as _mlops_runner
        _mlops_runner.stop_scheduler()
    except Exception:
        pass


app = FastAPI(
    title="QoSBuddy M6 API",
    description=(
        "DSO3.2 — RAG + GenAI + NOC Dashboard backend.\n\n"
        "Supports both v1 (anomaly_scores_ns3) and v2 (multi-model anomaly_scores) data formats."
    ),
    version="2.0.0",
    lifespan=lifespan,
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)


# ─────────────────────────────────────────────
# Pydantic Schemas
# ─────────────────────────────────────────────

class AnomalyEvent(BaseModel):
    """Schema for real-time anomaly push (v1 format)."""
    ue_id:             int
    timestamp:         int
    severity:          str
    if_score:          float
    if_anomaly:        int
    ae_error:          float
    ae_anomaly:        int
    sinr_dl_db:        float
    throughput_mbps:   float
    delay_ms:          float
    jitter_ms:         float
    packet_loss_ratio: float
    prb_utilization:   float
    retransmissions:   int
    load_level:        int
    root_cause:        Optional[str] = None
    recommended_action: Optional[str] = None
    cf_reduction_pct:  Optional[float] = None


class AnomalyEventV2(BaseModel):
    """Schema for v2 anomaly push (multi-model scores)."""
    window_idx:                          int
    true_label:                          int
    score_isolation_forest:              Optional[float] = None
    score_copod:                         Optional[float] = None
    score_one_class_svm:                 Optional[float] = None
    score_random_forest_supervised:      Optional[float] = None
    score_xgboost_supervised:            Optional[float] = None
    score_gradient_boosting_supervised:  Optional[float] = None
    score_lstm_autoencoder:              Optional[float] = None
    score_anomaly_transformer:           Optional[float] = None
    score_tranad:                        Optional[float] = None
    root_cause:                          Optional[str] = None
    recommended_action:                  Optional[str] = None


class RAGQueryRequest(BaseModel):
    question:          str
    mode:              str = Field("general", description="general|anomaly|causal|domain|sla|model_comparison")
    ue_id:             Optional[int]   = None
    severity_filter:   Optional[str]   = None
    root_cause_filter: Optional[str]   = None


class RAGQueryResponse(BaseModel):
    answer:    str
    sources:   List[Dict[str, Any]]
    question:  str
    mode:      str


class CounterfactualRequest(BaseModel):
    """Live DoWhy counterfactual — 'what would jitter be if X didn't happen?'"""
    treatment:      str   = Field(..., description="name of the treatment variable")
    treatment_val:  float = Field(..., description="observed treatment value")
    counterfactual: float = Field(0.0,  description="counterfactual treatment value")


class AnomalyScoreRequest(BaseModel):
    """Score a single KPI window against the v2 supervised models."""
    sinr_dl_db:        float
    throughput_mbps:   float
    delay_ms:          float
    jitter_ms:         float
    packet_loss_ratio: float
    prb_utilization:   float
    retransmissions:   float
    load_level:        int = 2


class RLObservation(BaseModel):
    """8-dim observation for PPO inference. Order matches QoSNetworkEnv.OBS_COLS."""
    sinr_dl_db:        float
    throughput_mbps:   float
    delay_ms:          float
    jitter_ms:         float
    packet_loss_ratio: float
    prb_utilization:   float
    retransmissions:   float
    sla_risk_score:    float = 0.0


class RLSimulateRequest(BaseModel):
    episodes:      int  = Field(10, ge=1, le=100)
    deterministic: bool = True


class KPISummary(BaseModel):
    total_events:        int
    anomaly_rate_pct:    float
    avg_jitter_ms:       float
    avg_packet_loss_pct: float
    avg_throughput_mbps: float
    avg_delay_ms:        float
    critical_count:      int
    degraded_count:      int
    normal_count:        int
    top_root_cause:      str
    data_version:        str


# ─────────────────────────────────────────────
# Endpoints
# ─────────────────────────────────────────────

@app.get("/health")
def health():
    """Health check — confirms API, data, and RAG are all live."""
    try:
        kb_docs = _collection.count() if _collection else 0
    except Exception:
        kb_docs = 0          # ChromaDB collection deleted/reset — non-fatal
    return {
        "status":       "ok",
        "data_version": _data_version,
        "rag_ready":    _collection is not None,
        "kb_docs":      kb_docs,
        "v1_loaded":    _anomaly_df is not None,
        "v2_loaded":    _anomaly_v2_df is not None,
        "v1_rows":      len(_anomaly_df) if _anomaly_df is not None else 0,
        "v2_rows":      len(_anomaly_v2_df) if _anomaly_v2_df is not None else 0,
        "rl_ready":     bool(_rl_policy and _rl_policy.ready),
    }


@app.get("/kpi/summary", response_model=KPISummary)
def get_kpi_summary(
    severity: Optional[str] = Query(None),
    ue_id:    Optional[int] = Query(None),
):
    """Real-time KPI summary for the Streamlit dashboard."""
    if _anomaly_df is None and _anomaly_v2_df is None:
        raise HTTPException(503, "No anomaly data loaded")

    # Use v1 data for KPI metrics (has raw KPI columns)
    if _anomaly_df is not None:
        df = _anomaly_df.copy()
        if severity: df = df[df["severity"] == severity]
        if ue_id:    df = df[df["ue_id"] == ue_id]

        if len(df) == 0:
            raise HTTPException(404, "No data matching filters")

        top_rc = "unknown"
        if _causal_df is not None and "root_cause" in _causal_df.columns:
            rc = _causal_df["root_cause"].value_counts()
            top_rc = rc.index[0] if len(rc) > 0 else "unknown"

        return KPISummary(
            total_events=len(df),
            anomaly_rate_pct=round(float(df["if_anomaly"].mean() * 100), 2),
            avg_jitter_ms=round(float(df["jitter_ms"].mean()), 3),
            avg_packet_loss_pct=round(float(df["packet_loss_ratio"].mean() * 100), 2),
            avg_throughput_mbps=round(float(df["throughput_mbps"].mean()), 3),
            avg_delay_ms=round(float(df["delay_ms"].mean()), 2),
            critical_count=int((df["severity"] == "critical").sum()),
            degraded_count=int((df["severity"] == "degraded").sum()),
            normal_count=int((df["severity"] == "normal").sum()),
            top_root_cause=top_rc,
            data_version=_data_version,
        )
    else:
        # v2 only — limited KPI info
        df = _anomaly_v2_df.copy()
        top_rc = "unknown"
        if _causal_df is not None and "root_cause" in _causal_df.columns:
            rc = _causal_df["root_cause"].value_counts()
            top_rc = rc.index[0] if len(rc) > 0 else "unknown"

        anomaly_rate = float(df["true_label"].mean() * 100) if "true_label" in df.columns else 0.0

        return KPISummary(
            total_events=len(df),
            anomaly_rate_pct=round(anomaly_rate, 2),
            avg_jitter_ms=0.0,
            avg_packet_loss_pct=0.0,
            avg_throughput_mbps=0.0,
            avg_delay_ms=0.0,
            critical_count=int(df["true_label"].sum()) if "true_label" in df.columns else 0,
            degraded_count=0,
            normal_count=int((df["true_label"] == 0).sum()) if "true_label" in df.columns else 0,
            top_root_cause=top_rc,
            data_version=_data_version,
        )


@app.get("/anomalies")
def get_anomalies(
    severity:     Optional[str]  = Query(None),
    ue_id:        Optional[int]  = Query(None),
    limit:        int            = Query(100, le=5000),
    only_flagged: bool           = Query(False),
):
    """Return v1 anomaly events as JSON — used by Streamlit triage board."""
    if _anomaly_df is None:
        raise HTTPException(503, "v1 anomaly data not loaded")

    df = _anomaly_df.copy()
    if severity:     df = df[df["severity"] == severity]
    if ue_id:        df = df[df["ue_id"] == ue_id]
    if only_flagged: df = df[df["if_anomaly"] == 1]

    df = df.sort_values("if_score", ascending=False).head(limit)
    return {"count": len(df), "data": df.to_dict(orient="records")}


@app.get("/anomalies/v2")
def get_anomalies_v2(
    label:  Optional[int] = Query(None, description="Filter by true_label (0=normal, 1=anomaly)"),
    root_cause: Optional[str] = Query(None),
    limit:  int = Query(200, le=5000),
):
    """Return v2 multi-model anomaly scores with root cause labels."""
    if _anomaly_v2_df is None:
        raise HTTPException(503, "v2 anomaly data not loaded")

    df = _anomaly_v2_df.copy()

    # Merge root causes if available
    if _causal_df is not None and "window_idx" in _causal_df.columns:
        rc_cols = ["window_idx", "root_cause", "recommended_action"]
        rc_cols = [c for c in rc_cols if c in _causal_df.columns]
        df = df.merge(_causal_df[rc_cols], on="window_idx", how="left")

    if label is not None:
        df = df[df["true_label"] == label]
    if root_cause and "root_cause" in df.columns:
        df = df[df["root_cause"] == root_cause]

    df = df.head(limit)

    # Model score statistics
    score_cols = [c for c in df.columns if c.startswith("score_")]
    model_stats = {}
    for col in score_cols:
        model_name = col.replace("score_", "")
        model_stats[model_name] = {
            "mean": round(float(df[col].mean()), 4),
            "std":  round(float(df[col].std()), 4),
            "min":  round(float(df[col].min()), 4),
            "max":  round(float(df[col].max()), 4),
        }

    return {
        "count":       len(df),
        "model_stats": model_stats,
        "data":        df.head(limit).to_dict(orient="records"),
    }


@app.get("/models/benchmark")
def get_model_benchmark():
    """
    Return model performance comparison from v2 anomaly scores.
    Computes basic separation metrics for each model.
    """
    if _anomaly_v2_df is None:
        raise HTTPException(503, "v2 anomaly data not loaded")

    df = _anomaly_v2_df.copy()
    score_cols = [c for c in df.columns if c.startswith("score_")]

    if "true_label" not in df.columns:
        raise HTTPException(503, "true_label column missing")

    results = []
    for col in score_cols:
        model_name = col.replace("score_", "").replace("_", " ").title()
        normal_scores  = df.loc[df["true_label"] == 0, col].dropna()
        anomaly_scores = df.loc[df["true_label"] == 1, col].dropna()

        if len(normal_scores) == 0 or len(anomaly_scores) == 0:
            continue

        results.append({
            "model":              model_name,
            "column":             col,
            "normal_mean":        round(float(normal_scores.mean()), 4),
            "anomaly_mean":       round(float(anomaly_scores.mean()), 4),
            "separation_ratio":   round(float(anomaly_scores.mean() / (normal_scores.mean() + 1e-12)), 2),
            "normal_count":       int(len(normal_scores)),
            "anomaly_count":      int(len(anomaly_scores)),
        })

    return {
        "models":       results,
        "total_windows": len(df),
        "anomaly_rate":  round(float(df["true_label"].mean() * 100), 2),
    }


@app.post("/ingest/anomaly")
async def ingest_anomaly(event: AnomalyEvent, background_tasks: BackgroundTasks):
    """M1 → M6 real-time push endpoint (v1 format)."""
    global _anomaly_df
    new_row = pd.DataFrame([event.model_dump()])
    if _anomaly_df is not None:
        _anomaly_df = pd.concat([_anomaly_df, new_row], ignore_index=True)

    if _collection is not None and _embed_model is not None:
        background_tasks.add_task(_upsert_event_to_kb, event)

    return {"status": "accepted", "ue_id": event.ue_id, "timestamp": event.timestamp}


def _upsert_event_to_kb(event: AnomalyEvent):
    """Background task: embed and upsert a single anomaly event into ChromaDB."""
    from rag.build_knowledge_base import SEVERITY_EMOJI, ROOT_CAUSE_EXPLANATIONS

    sev   = event.severity
    rc    = event.root_cause or "unknown"
    emoji = SEVERITY_EMOJI.get(sev, "⚪")
    rc_explain = ROOT_CAUSE_EXPLANATIONS.get(rc, rc)

    doc = f"""
[ANOMALY EVENT — LIVE] UE {event.ue_id} | Timestamp {event.timestamp} | Severity: {emoji} {sev.upper()}
Detection: IF={'YES' if event.if_anomaly else 'NO'} (score={event.if_score:.4f}) | AE={'YES' if event.ae_anomaly else 'NO'} (error={event.ae_error:.4f})
KPIs: sinr={event.sinr_dl_db:.2f}dB | throughput={event.throughput_mbps:.3f}Mbps | delay={event.delay_ms:.2f}ms | jitter={event.jitter_ms:.3f}ms | pkt_loss={event.packet_loss_ratio:.3f}
Root Cause: {rc} — {rc_explain}
Action: {event.recommended_action or 'pending analysis'}
Counterfactual reduction: {event.cf_reduction_pct or 0:.1f}%
""".strip()

    embedding = _embed_model.encode([doc], normalize_embeddings=True).tolist()
    _collection.upsert(
        documents=[doc],
        metadatas=[{
            "doc_type":   "ANOMALY_EVENT",
            "ue_id":      event.ue_id,
            "timestamp":  event.timestamp,
            "severity":   sev,
            "root_cause": rc,
            "if_anomaly": event.if_anomaly,
            "if_score":   round(event.if_score, 6),
            "ae_error":   round(event.ae_error, 6),
        }],
        ids=[f"live_ue{event.ue_id}_ts{event.timestamp}"],
        embeddings=embedding,
    )
    log.info(f"Upserted live event: UE {event.ue_id} ts={event.timestamp}")


@app.post("/rag/query", response_model=RAGQueryResponse)
def rag_query(req: RAGQueryRequest):
    """RAG question-answering endpoint."""
    if _collection is None or _embed_model is None:
        raise HTTPException(503, "RAG stack not initialised")

    from rag.rag_pipeline import query_rag
    result = query_rag(
        question=req.question,
        collection=_collection,
        embed_model=_embed_model,
        chain=_chain,
        mode=req.mode,
        ue_id=req.ue_id,
        severity_filter=req.severity_filter,
        root_cause_filter=req.root_cause_filter,
    )
    return RAGQueryResponse(**result)


@app.get("/causal/summary")
def get_causal_summary():
    """Root cause statistics."""
    if _causal_df is None:
        raise HTTPException(503, "Root cause data not loaded")

    group_cols = ["root_cause"]
    if "recommended_action" in _causal_df.columns:
        group_cols.append("recommended_action")

    agg_dict = {}
    if "true_label" in _causal_df.columns:
        agg_dict["true_label"] = ["count", "mean"]
    elif "if_anomaly" in _causal_df.columns:
        agg_dict["if_anomaly"] = ["count", "mean"]
    else:
        agg_dict["root_cause"] = "count"

    summary = _causal_df.groupby(group_cols).agg(**{
        "count": (list(agg_dict.keys())[0], "count"),
        "anomaly_rate": (list(agg_dict.keys())[0], "mean"),
    }).reset_index()

    return {"data": summary.round(4).to_dict(orient="records")}


@app.get("/counterfactuals")
def get_counterfactuals(ue_id: Optional[int] = Query(None)):
    """Counterfactual analysis endpoint."""
    if _cf_df is None:
        raise HTTPException(503, "Counterfactual data not loaded")

    df = _cf_df.copy()
    if ue_id:
        df = df[df["ue_id"] == ue_id]

    return {
        "count":             len(df),
        "avg_reduction_pct": round(float(df["reduction_pct"].mean()), 2),
        "avg_original_loss": round(float(df["original_packet_loss"].mean()), 4),
        "avg_cf_loss":       round(float(df["counterfactual_if_static"].mean()), 4),
        "sample":            df.head(20).to_dict(orient="records"),
    }


@app.get("/sla/risk")
def get_sla_risk(
    ue_id:     Optional[int] = Query(None),
    high_only: bool          = Query(True),
    limit:     int           = Query(200, le=5000),
):
    """M4 DSO2.1 — SLA breach risk predictions."""
    if _sla_df is None:
        raise HTTPException(503, "SLA data not loaded")

    df = _sla_df.copy()
    if ue_id:     df = df[df["ue_id"] == ue_id]
    if high_only: df = df[df["is_high_risk_pred"] == 1]

    df = df.sort_values("risk_proba", ascending=False).head(limit)
    return {
        "count":           len(df),
        "total_high_risk": int((_sla_df["is_high_risk_pred"] == 1).sum()),
        "total_rows":      len(_sla_df),
        "risk_rate_pct":   round(float((_sla_df["is_high_risk_pred"] == 1).mean() * 100), 2),
        "data":            df.to_dict(orient="records"),
    }


@app.get("/severity/summary")
def get_severity_summary(ue_id: Optional[int] = Query(None)):
    """M5 DSO2.3 — Severity triage classification."""
    if _severity_df is None:
        raise HTTPException(503, "Severity data not loaded")

    df = _severity_df.copy()
    if ue_id: df = df[df["ue_id"] == ue_id]

    dist = df["severity_final"].value_counts().to_dict()
    by_traffic = (
        df.groupby(["traffic_type", "severity_final"])
        .size().reset_index(name="count")
        .to_dict(orient="records")
    ) if "traffic_type" in df.columns else []

    return {
        "total":        len(df),
        "distribution": dist,
        "by_traffic":   by_traffic,
        "sample":       df.head(50).to_dict(orient="records"),
    }


@app.get("/benchmark/summary")
def get_benchmark_summary():
    """M5 DSO1.3 — QoS benchmarking by load level and mobility speed."""
    if _bench_df is None:
        raise HTTPException(503, "Benchmark data not loaded")

    return {"count": len(_bench_df), "data": _bench_df.to_dict(orient="records")}


@app.get("/rl/summary")
def get_rl_summary():
    """M5 DSO3.1 — PPO RL Agent vs Rule-Based Baseline comparison."""
    if _rl_df is None:
        raise HTTPException(503, "RL comparison data not loaded")

    summary = (
        _rl_df.groupby("policy")
        .agg(
            mean_reward     = ("total_reward",   "mean"),
            std_reward      = ("total_reward",   "std"),
            mean_violations = ("sla_violations", "mean"),
            std_violations  = ("sla_violations", "std"),
            episodes        = ("episode",        "count"),
        )
        .reset_index()
        .round(3)
    )

    ppo  = summary[summary["policy"] == "PPO_RL"]["mean_violations"].values
    rule = summary[summary["policy"] == "Rule-Based"]["mean_violations"].values
    improvement = 0.0
    if len(ppo) > 0 and len(rule) > 0 and rule[0] > 0:
        improvement = round((1 - ppo[0] / rule[0]) * 100, 1)

    return {
        "summary":            summary.to_dict(orient="records"),
        "improvement_pct":    improvement,
        "episodes_evaluated": int(_rl_df["episode"].nunique()),
        "raw_data":           _rl_df.to_dict(orient="records"),
    }


@app.post("/rl/predict")
def rl_predict(obs: RLObservation, deterministic: bool = Query(True)):
    """Live PPO inference — given current KPIs, return the recommended action
    and the full action probability distribution."""
    if not (_rl_policy and _rl_policy.ready):
        raise HTTPException(503, f"RL policy not ready: {_rl_policy.error if _rl_policy else 'not initialised'}")
    obs_list = [
        obs.sinr_dl_db, obs.throughput_mbps, obs.delay_ms, obs.jitter_ms,
        obs.packet_loss_ratio, obs.prb_utilization, obs.retransmissions, obs.sla_risk_score,
    ]
    try:
        return _rl_policy.predict(obs_list, deterministic=deterministic)
    except Exception as e:
        raise HTTPException(500, f"Inference failed: {e}")


@app.post("/rl/simulate")
def rl_simulate(req: RLSimulateRequest):
    """Run live episodes on the NS-3 digital twin using the trained PPO.
    Returns per-episode rewards + SLA violations + action mix."""
    if not (_rl_policy and _rl_policy.ready):
        raise HTTPException(503, f"RL policy not ready: {_rl_policy.error if _rl_policy else 'not initialised'}")
    try:
        return _rl_policy.simulate(episodes=req.episodes, deterministic=req.deterministic)
    except Exception as e:
        raise HTTPException(500, f"Simulation failed: {e}")


@app.get("/rl/benchmark")
def rl_benchmark():
    """Training-time benchmark (PPO vs Random vs Rule-Based) from rl_agent/config.json."""
    if not (_rl_policy and _rl_policy.ready):
        raise HTTPException(503, f"RL policy not ready: {_rl_policy.error if _rl_policy else 'not initialised'}")
    return _rl_policy.benchmark()


# ── M2 DSO1.1 — QoS Forecasts (Member 2) ──────────────────────────────────────

@app.get("/qos/predictions")
def qos_predictions(
    ue_id:  Optional[int] = Query(None),
    limit:  int  = Query(2000, le=100000),
    source: str  = Query("auto", regex="^(auto|static|live)$",
                         description="auto = static + live concat; static = CSV only; live = MLOps output only"),
):
    """Return XGBoost jitter predictions (y_true vs y_pred) from DSO1.1.

    `auto` concatenates the static `predicted_jitter.csv` with the live
    `predicted_jitter_live.csv` produced by the MLOps `score_qos_forecast`
    job, so the page reflects both historical eval and live scoring.
    """
    parts: List[pd.DataFrame] = []
    src_used: List[str] = []

    if source in ("auto", "static") and _jitter_pred_df is not None:
        parts.append(_jitter_pred_df.copy()); src_used.append("static")

    live_path = DATA_DIR / "predicted_jitter_live.csv"
    if source in ("auto", "live") and live_path.exists():
        try:
            live = pd.read_csv(live_path)
            if not live.empty:
                parts.append(live); src_used.append("live")
        except Exception as e:
            log.warning(f"qos predictions live read failed: {e}")

    if not parts:
        raise HTTPException(503, "DSO1.1 predictions not available (no static CSV, no live output)")

    df = pd.concat(parts, ignore_index=True, sort=False) if len(parts) > 1 else parts[0]
    if ue_id is not None and "ue_id" in df.columns:
        df = df[df["ue_id"] == ue_id]
    # Take the *tail* — newest rows are at the end of the live log
    df = df.tail(limit) if limit > 0 else df

    metrics = {}
    if {"y_true", "y_pred"}.issubset(df.columns):
        yt = pd.to_numeric(df["y_true"], errors="coerce")
        yp = pd.to_numeric(df["y_pred"], errors="coerce")
        ok = yt.notna() & yp.notna()
        if ok.any():
            err = (yp[ok] - yt[ok])
            metrics = {
                "n":    int(ok.sum()),
                "mae":  round(float(err.abs().mean()), 4),
                "rmse": round(float(np.sqrt((err ** 2).mean())), 4),
                "bias": round(float(err.mean()), 4),
            }
    return {
        "count":   int(len(df)),
        "sources": src_used,
        "metrics": metrics,
        "data":    df.to_dict(orient="records"),
    }


@app.get("/qos/forecast")
def qos_forecast(
    ue_id:  Optional[int] = Query(None),
    metric: Optional[str] = Query(None, description="jitter_ms | throughput_mbps"),
    limit:  int = Query(500, le=5000),
):
    """Prophet forecast with confidence intervals (DSO1.1)."""
    if _forecast_ci_df is None:
        raise HTTPException(503, "DSO1.1 forecast not loaded (forecast_ci.csv)")
    df = _forecast_ci_df.copy()
    if ue_id is not None and "ue_id" in df.columns:
        df = df[df["ue_id"] == ue_id]
    if metric and "metric" in df.columns:
        df = df[df["metric"] == metric]
    df = df.head(limit)
    metrics_available = sorted(_forecast_ci_df["metric"].dropna().unique().tolist()) if "metric" in _forecast_ci_df.columns else []
    ues_available = sorted(_forecast_ci_df["ue_id"].dropna().unique().tolist()) if "ue_id" in _forecast_ci_df.columns else []
    return {
        "count":    len(df),
        "metrics_available": metrics_available,
        "ues_available":     ues_available[:50],
        "data":     df.to_dict(orient="records"),
    }


# ── M1 SHAP explainability (Member 1) ─────────────────────────────────────────

@app.get("/explain/shap")
def explain_shap(limit: int = Query(30, le=500)):
    """Global feature importance from M1's SHAP analysis."""
    if _shap_df is None:
        raise HTTPException(503, "SHAP importance not loaded (shap_importance_ns3.xls)")
    df = _shap_df.copy()
    return {"count": len(df), "data": df.head(limit).to_dict(orient="records")}


# ── M3 Live Counterfactual (Member 3) ─────────────────────────────────────────

@app.post("/causal/counterfactual")
def causal_counterfactual(req: CounterfactualRequest):
    """Live DoWhy counterfactual estimate. Falls back to CSV lookup if model absent."""
    if _dowhy_model is not None:
        try:
            # DoWhy `CausalModel` workflow: identify → estimate → scale to (cf - obs).
            model = _dowhy_model
            identified_estimand = model.identify_effect(proceed_when_unidentifiable=True)
            estimate = model.estimate_effect(
                identified_estimand,
                method_name="backdoor.linear_regression",
                target_units="ate",
            )
            ate = float(estimate.value)
            delta = float(req.counterfactual) - float(req.treatment_val)
            return {
                "source":            "dowhy_model",
                "treatment":         req.treatment,
                "observed":          req.treatment_val,
                "counterfactual":    req.counterfactual,
                "ate":               round(ate, 6),
                "estimated_effect":  round(ate * delta, 6),
                "method":            "backdoor.linear_regression",
            }
        except Exception as e:
            log.warning(f"DoWhy live estimate failed: {e}; falling back to CSV")

    if _cf_df is None:
        raise HTTPException(503, "Neither DoWhy model nor counterfactuals.csv available")
    df = _cf_df
    return {
        "source":        "counterfactuals_csv",
        "treatment":     req.treatment,
        "observed":      req.treatment_val,
        "counterfactual": req.counterfactual,
        "note":          "Live DoWhy estimate unavailable; serving pre-computed CF aggregate.",
        "avg_reduction_pct": round(float(df["reduction_pct"].mean()), 2) if "reduction_pct" in df.columns else None,
    }


# ── M1 Live Anomaly Scoring (Member 1) ────────────────────────────────────────

_ANOMALY_FEATURES = [
    "sinr_dl_db", "throughput_mbps", "delay_ms", "jitter_ms",
    "packet_loss_ratio", "prb_utilization", "retransmissions", "load_level",
]


def _build_feature_vector(req: "AnomalyScoreRequest", expected: list[str]) -> np.ndarray:
    """Pad the 8 base KPIs onto whatever feature list the bundled model expects.
    Missing engineered features default to 0.0 — acceptable for a UI smoke test,
    and documented in the response so the user knows."""
    provided = {f: float(getattr(req, f)) for f in _ANOMALY_FEATURES}
    vec = [provided.get(col, 0.0) for col in expected]
    return np.array([vec], dtype=np.float32)


@app.post("/anomaly/score")
def anomaly_score(req: AnomalyScoreRequest):
    """Score a single KPI window with all loaded v2 supervised models."""
    if not _anomaly_models:
        raise HTTPException(503, "No anomaly models loaded (anomaly_models/ empty)")

    results = {}
    for name, bundle in _anomaly_models.items():
        try:
            model   = bundle["model"] if isinstance(bundle, dict) else bundle
            feat    = bundle.get("features") if isinstance(bundle, dict) else None
            thr     = (bundle.get("threshold") if isinstance(bundle, dict) else None) or 0.5
            if feat is None:
                feat = _ANOMALY_FEATURES
            x = _build_feature_vector(req, feat)

            if name == "isolation_forest":
                score = float(-model.score_samples(x)[0])
                flag  = int(model.predict(x)[0] == -1)
            else:
                proba = model.predict_proba(x)[0]
                score = float(proba[-1])
                flag  = int(score >= thr)
            results[name] = {
                "score":      round(score, 6),
                "is_anomaly": flag,
                "n_features_used": len(feat),
            }
        except Exception as e:
            results[name] = {"error": str(e)}

    flagged = [n for n, r in results.items() if isinstance(r.get("is_anomaly"), int) and r["is_anomaly"]]
    return {
        "features": {f: getattr(req, f) for f in _ANOMALY_FEATURES},
        "note":     "Engineered features (rolling means, spike flags, etc.) default to 0 — inference is best-effort from 8 base KPIs.",
        "results":  results,
        "ensemble_flag": 1 if len(flagged) >= max(1, len(results) // 2) else 0,
        "flagged_by":    flagged,
    }


# ── Companion widget: answers contextual questions from every dashboard page ──

_PAGE_CONTEXT = {
    "kpi": {
        "short": "the KPI Overview dashboard",
        "overview": (
            "This is the KPI Overview — the live pulse of the 5G network. "
            "You see six top cards: average jitter in milliseconds, packet loss ratio, "
            "throughput in megabits per second, one-way delay, anomaly rate, and average SINR in decibels. "
            "Below them are a severity donut, a root-cause breakdown, per-UE packet-loss timelines, "
            "SINR distribution by load level, and detector agreement. "
            "Colors follow a traffic-light rule: green means healthy, amber is elevated, red is critical."
        ),
        "how_to_read": (
            "Start with the six KPI cards. If jitter is above 30 ms or packet loss above 5 percent, "
            "that is already degraded service. Cross-check against the anomaly rate card — a high rate "
            "with otherwise healthy KPIs usually points to edge-case subtle anomalies rather than hard outages."
        ),
    },
    "anomaly": {
        "short": "the Anomaly Feed page",
        "overview": (
            "This is the Anomaly Feed, task DSO2.2 and DSO2.3. "
            "It shows the consensus output of our Isolation Forest plus LSTM Autoencoder ensemble, "
            "with SHAP bars explaining which KPI pushed each event over the threshold. "
            "The live scorer at the bottom lets you paste a KPI vector and get an instant verdict."
        ),
        "how_to_read": (
            "Rows flagged by both detectors are high-confidence anomalies — act on those first. "
            "Use the SHAP panel to see whether jitter, loss, or SINR dominated; that tells you where to start debugging."
        ),
    },
    "causal": {
        "short": "the Causal Root Cause Explorer",
        "overview": (
            "This is the Causal Analysis page, task DSO1.2. "
            "It runs a DoWhy causal model over the network KPIs, shows counterfactual estimates "
            "(what packet loss would have looked like without the anomaly), and the live estimator "
            "lets you ask 'what if jitter were 5 ms lower?' style questions."
        ),
        "how_to_read": (
            "The bar chart shows estimated causal effect per root cause. Larger absolute effect means "
            "that root cause contributed more to the degradation. The counterfactual line shows the gap "
            "between observed and 'clean-world' packet loss."
        ),
    },
    "forecast": {
        "short": "the QoS Forecast page",
        "overview": (
            "This is the QoS Forecast page, task DSO1.1. "
            "It overlays XGBoost jitter and throughput predictions against actuals, plus Prophet "
            "confidence bands per UE and per metric. Widening bands mean the model is less sure."
        ),
        "how_to_read": (
            "Pick a UE and a metric. If the predicted line diverges from actuals and the Prophet bands "
            "are narrow, that is a real regime shift worth investigating. Wide bands with a stable actual "
            "just mean volatility in the training window."
        ),
    },
    "sla_risk": {
        "short": "the SLA Breach Risk page",
        "overview": (
            "This is the SLA Breach Risk page, task DSO2.1. "
            "It shows XGBoost early-warning predictions for upcoming SLA breaches, broken down by load level."
        ),
        "how_to_read": (
            "High-risk rows at low load usually indicate configuration issues; high risk under heavy load "
            "typically points to capacity. Sort by predicted probability and act on the top slice."
        ),
    },
    "severity": {
        "short": "the Severity Triage page",
        "overview": (
            "This is the Severity Triage page. It classifies every event as critical, degraded, or normal, "
            "and visualizes the breakdown per UE and per root cause."
        ),
        "how_to_read": (
            "The critical slice is your oncall queue. Degraded is for the next-business-day triage. "
            "Use the per-UE facet to see whether one device is dragging the fleet numbers."
        ),
    },
    "benchmark": {
        "short": "the Benchmarking page",
        "overview": (
            "This is the Benchmarking page. It compares our anomaly detectors head-to-head on F1, "
            "precision, and recall, so you can judge which model to trust on new data."
        ),
        "how_to_read": (
            "Prefer models with high F1 and balanced precision-recall. A model with very high precision "
            "but low recall misses real anomalies; the opposite floods the NOC with false positives."
        ),
    },
    "rl": {
        "short": "the RL Actions page",
        "overview": (
            "This is the RL Actions page, task DSO3.1. "
            "It shows the PPO agent's live advisor (what action it would recommend for the current KPI vector), "
            "the historical distribution of recommended actions over anomalous events, and a digital-twin simulation."
        ),
        "how_to_read": (
            "The top card names the most common recommended action across anomalies: monitor closely, "
            "analyze time-series, investigate multivariate, inspect individual, or high-confidence alert. "
            "The pie chart shows frequency; the advisor panel takes a KPI vector and returns the policy action."
        ),
    },
    "chat": {
        "short": "the Ask the Network RAG chat",
        "overview": (
            "This is the Ask the Network page — the full RAG chat interface. "
            "It retrieves relevant KPI windows from ChromaDB and answers with llama3.2 via Ollama, "
            "with optional filtering by UE, severity, and root cause."
        ),
        "how_to_read": (
            "Pick the mode first: general, per-UE, or root-cause. Then ask in plain English. "
            "The retrieved context sources are shown below the answer so you can audit the evidence."
        ),
    },
    "voice": {
        "short": "the Voice NOC Assistant page",
        "overview": (
            "This is the Voice NOC Assistant. A full 3D avatar reacts to your voice — say 'Hey Buddy' "
            "to trigger listening, then ask any question. The avatar lip-syncs to the TTS answer."
        ),
        "how_to_read": (
            "The status dot on the avatar tells you the state: green idle, amber listening, "
            "purple processing, red speaking."
        ),
    },
    "export": {
        "short": "the Export Report page",
        "overview": (
            "This is the Export Report page. It produces a branded PDF summary with KPIs, "
            "anomaly highlights, causal findings, and forecasts — ready for handoff."
        ),
        "how_to_read": (
            "Pick the time range and the sections you want, then click Generate. The PDF is written to disk and downloadable."
        ),
    },
    "live_monitor": {
        "short": "the Live Network Monitor page",
        "overview": (
            "This is the Live Network Monitor — a real-time view of the NS-3 simulation feed. "
            "It connects to the teammate's NS-3 LENA simulator via bridge.py, which watches "
            "the simulation output CSV and streams new rows to the API over HTTP. "
            "Each incoming row is scored by anomaly detection models (IF, RF, GBT) and the PPO RL agent "
            "in real-time. The page shows live KPI time series, anomaly flags, RL action recommendations, "
            "and model score charts that auto-refresh every few seconds."
        ),
        "how_to_read": (
            "Check the green/red status banner at the top — green means bridge.py is actively sending data. "
            "The KPI cards show current averages. The time series tabs let you switch between metrics. "
            "Red triangle markers on the charts indicate anomaly-flagged points. "
            "The RL Agent section shows what action the PPO recommends for each data point."
        ),
    },
}

# Keywords that trigger the fast, LLM-free overview response.
_OVERVIEW_KEYWORDS = (
    "what is this", "what's this", "what is on", "what's on", "what does this page",
    "explain this", "explain the", "explain it", "give me an overview", "overview",
    "content of this", "content of the page", "what does this show", "what am i looking at",
    "tell me about this page", "what does the page", "what page", "what is the",
    "what are we", "help me understand", "walk me through", "describe this page",
    "describe the", "how do i read", "how to read", "how does this work",
)
_HOW_KEYWORDS = ("how do i read", "how to read", "how do i interpret", "interpret", "read the", "read this")


def _companion_fast_answer(page_key: str, question: str) -> Optional[str]:
    """Return a pre-baked answer if the question is an overview-style ask. None otherwise."""
    ctx = _PAGE_CONTEXT.get((page_key or "").lower())
    if not ctx:
        return None
    q = (question or "").lower().strip()
    if not q:
        return ctx.get("overview")
    # Any overview-style phrasing → rich overview.
    if any(kw in q for kw in _OVERVIEW_KEYWORDS):
        ans = ctx.get("overview", "")
        how = ctx.get("how_to_read", "")
        if any(kw in q for kw in _HOW_KEYWORDS) and how:
            return how
        return (ans + " " + how).strip() if how else ans
    return None


class CompanionAskRequest(BaseModel):
    question: str = Field(..., min_length=1)
    page:     Optional[str] = Field(None, description="current dashboard page key")


@app.post("/companion/ask")
def companion_ask(req: CompanionAskRequest):
    """Answer a question about the current page.

    Strategy:
    1. Fast path — if the question is overview-style, return the pre-baked page explanation
       immediately (no LLM, sub-100ms). This handles 'explain this page', 'what is this',
       'give me an overview', 'how do I read this', etc.
    2. Specific path — hand to the RAG chain for targeted questions. If RAG fails or is
       unavailable, fall back to the rich page overview so the user is never stuck.
    """
    page_key  = (req.page or "").lower()
    ctx       = _PAGE_CONTEXT.get(page_key, {})
    page_hint = ctx.get("short", "")

    # 1. Fast path
    fast = _companion_fast_answer(page_key, req.question)
    if fast:
        return {"answer": fast, "page": req.page, "source": "context"}

    # 2. RAG path
    if _collection is not None and _embed_model is not None and _chain is not None:
        try:
            from rag.rag_pipeline import query_rag
            framed = (
                f"The user is currently viewing {page_hint}. "
                f"Answer this question briefly and clearly, as a NOC companion, in two to four sentences: {req.question}"
            ) if page_hint else req.question
            result = query_rag(
                question=framed,
                collection=_collection,
                embed_model=_embed_model,
                chain=_chain,
                mode="general",
                ue_id=None,
                severity_filter=None,
                root_cause_filter=None,
            )
            answer = (result.get("answer") or "").strip()
            if answer:
                return {"answer": answer, "page": req.page, "source": "rag"}
        except Exception as e:
            log.warning(f"Companion RAG call failed: {e}")

    # 3. Fallback — always return the rich page overview so the user never sees a blank error.
    fallback = ctx.get("overview") or "I don't have context for this page yet."
    how = ctx.get("how_to_read")
    if how:
        fallback = fallback + " " + how
    return {"answer": fallback, "page": req.page, "source": "fallback"}


# ── Live NS-3 Bridge endpoints ────────────────────────────────────────────────

def _score_and_buffer(rows: list):
    """Score rows with ML models + RL agent and push to buffer.
    Runs in a background thread so /ingest/live returns immediately."""
    import time as _time

    def _score_row(row):
        row["_server_ts"]  = _time.time()
        row["_server_iso"] = datetime.now().isoformat()

        # Anomaly models
        if _anomaly_models:
            try:
                scores = {}
                for name, bundle in _anomaly_models.items():
                    model = bundle["model"] if isinstance(bundle, dict) else bundle
                    feat  = bundle.get("features") if isinstance(bundle, dict) else None
                    if feat is None:
                        feat = ["sinr_dl_db", "throughput_mbps", "delay_ms",
                                "jitter_ms", "packet_loss_ratio", "prb_utilization",
                                "retransmissions", "load_level"]
                    vec = [float(row.get(f, 0.0)) for f in feat]
                    x   = np.array([vec], dtype=np.float32)
                    if name == "isolation_forest":
                        scores[name] = float(-model.score_samples(x)[0])
                    else:
                        try:
                            scores[name] = float(model.predict_proba(x)[0][-1])
                        except Exception:
                            scores[name] = 0.0
                row["_live_scores"]  = scores
                row["_live_anomaly"] = int(
                    sum(1 for s in scores.values() if s > 0.5)
                    >= max(1, len(scores) // 2)
                )
            except Exception:
                row["_live_scores"]  = {}
                row["_live_anomaly"] = 0

        # RL agent
        if _rl_policy and _rl_policy.ready:
            try:
                obs  = [float(row.get(k, 0)) for k in
                        ["sinr_dl_db","throughput_mbps","delay_ms","jitter_ms",
                         "packet_loss_ratio","prb_utilization","retransmissions"]] + [0.0]
                pred = _rl_policy.predict(obs)
                row["_rl_action"] = pred["action_name"]
                row["_rl_probs"]  = pred["probabilities"]
            except Exception:
                row["_rl_action"] = "unknown"
                row["_rl_probs"]  = {}
        return row

    scored = [_score_row(r) for r in rows]
    with _live_lock:
        for r in scored:
            _live_buffer.append(r)
        # NOTE: total_received is incremented in /ingest/live (the entry point);
        # don't double-count here. We just refresh the buffer-size stat.
        _live_stats["bridge_connected"]  = True
        _live_stats["rows_in_buffer"]    = len(_live_buffer)
    # Persist scored rows to the JSONL log (outside the buffer lock so file
    # I/O doesn't block the live read path).
    _append_live_log(scored)


@app.post("/ingest/live")
async def ingest_live(payload: dict):
    """Receive live rows from bridge.py.
    Returns immediately — scoring happens in a background thread."""
    rows = payload.get("rows", [])
    if not rows:
        raise HTTPException(400, "No rows in payload")

    # Stamp arrival time right away so bridge_connected stays fresh
    import time as _time
    ts  = _time.time()
    iso = datetime.now().isoformat()
    for row in rows:
        row.setdefault("_server_ts",  ts)
        row.setdefault("_server_iso", iso)

    # Update stats immediately so the dashboard sees the connection
    with _live_lock:
        _live_stats["total_received"]   += len(rows)
        _live_stats["last_received_ts"]  = iso
        _live_stats["bridge_connected"]  = True

    # Score + buffer in background — never blocks the bridge
    t = threading.Thread(target=_score_and_buffer, args=(rows,), daemon=True)
    t.start()

    return {
        "status":         "ok",
        "received":       len(rows),
        "buffer_size":    len(_live_buffer),
        "total_received": _live_stats["total_received"],
    }


@app.get("/live/stream")
def live_stream(
    last_n: int = Query(50, le=500),
    since_ts: Optional[float] = Query(None),
):
    """Dashboard polls this for latest live data from the buffer."""
    with _live_lock:
        if since_ts:
            rows = [r for r in _live_buffer if r.get("_server_ts", 0) > since_ts]
        else:
            rows = list(_live_buffer)[-last_n:]
    return {
        "count": len(rows),
        "buffer_total": len(_live_buffer),
        "stats": dict(_live_stats),
        "data": rows,
    }


@app.get("/live/status")
def live_status():
    """Check if bridge.py is connected and sending data."""
    connected = False
    if _live_stats["last_received_ts"]:
        try:
            last = datetime.fromisoformat(_live_stats["last_received_ts"])
            age = (datetime.now() - last).total_seconds()
            connected = age < 30
        except Exception:
            pass
    # Persistent log size for the dashboard's augmentation logic.
    log_size = 0
    log_bytes = 0
    try:
        if LIVE_LOG_PATH.exists():
            log_bytes = LIVE_LOG_PATH.stat().st_size
            log_size  = _count_live_log_lines()
    except Exception:
        pass
    return {
        "bridge_connected": connected,
        "total_received": _live_stats["total_received"],
        "buffer_size": len(_live_buffer),
        "last_received": _live_stats["last_received_ts"],
        "log_rows":  log_size,
        "log_bytes": log_bytes,
        "log_path":  str(LIVE_LOG_PATH.name),
    }


@app.get("/live/log")
def live_log(last_n: int = Query(2000, ge=1, le=50000)):
    """Read the last N rows from the persistent JSONL log.

    Used by the dashboard to feed the cumulative augmented dataset into the
    KPI / Anomaly / RL pages. Returns rows oldest-first so concat with the
    static CSV preserves chronological order.
    """
    import json as _json
    if not LIVE_LOG_PATH.exists():
        return {"count": 0, "data": []}
    try:
        # Tail-read the last N lines. For very large files we read from the end
        # by chunks; for typical sizes here, reading whole and slicing is fine.
        with open(LIVE_LOG_PATH, "r", encoding="utf-8") as f:
            lines = f.readlines()
        tail = lines[-last_n:]
        rows = []
        for line in tail:
            line = line.strip()
            if not line: continue
            try:
                rows.append(_json.loads(line))
            except Exception:
                continue
        return {"count": len(rows), "log_total": len(lines), "data": rows}
    except Exception as e:
        log.warning(f"/live/log read failed: {e}")
        return {"count": 0, "data": [], "error": str(e)}


# ── MLOps pipeline endpoints ────────────────────────────────────────────
# The pipeline reads data/live_log.jsonl and refreshes the *_live.csv files
# the dashboard concats with the static historical CSVs. See mlops/.

class PipelineRunRequest(BaseModel):
    job: Optional[str] = Field(None, description="job name; omit to run all")


@app.get("/pipeline/status")
def pipeline_status():
    """Return registered jobs + last-run state + scheduler info."""
    try:
        from mlops import runner as _runner
        return _runner.status()
    except Exception as e:
        return {"error": str(e), "jobs": {}, "scheduler_running": False}


@app.post("/pipeline/run")
def pipeline_run(req: PipelineRunRequest):
    """Trigger one job (req.job) or all of them. Returns metrics."""
    try:
        from mlops import runner as _runner
    except Exception as e:
        raise HTTPException(503, f"MLOps not available: {e}")
    if req.job:
        return {"job": req.job, "result": _runner.run_one(req.job)}
    return {"results": _runner.run_all()}


@app.post("/pipeline/scheduler/start")
def pipeline_scheduler_start(interval_s: int = Query(0, ge=0)):
    """Start the periodic scheduler. Pass `interval_s` to override the period."""
    try:
        from mlops import runner as _runner
        _runner.start_scheduler(interval_s if interval_s > 0 else None)
        return {"started": True, "interval_s": _runner.status()["interval_s"]}
    except Exception as e:
        raise HTTPException(503, f"MLOps not available: {e}")


@app.post("/pipeline/scheduler/stop")
def pipeline_scheduler_stop():
    try:
        from mlops import runner as _runner
        _runner.stop_scheduler()
        return {"stopped": True}
    except Exception as e:
        raise HTTPException(503, f"MLOps not available: {e}")


# ── Voice transcript store ──

_voice_transcript: dict = {"text": "", "ts": 0}

@app.post("/voice/transcript")
async def post_voice_transcript(payload: dict):
    import time
    _voice_transcript["text"] = payload.get("text", "").strip()
    _voice_transcript["ts"]   = time.time()
    return {"status": "ok", "text": _voice_transcript["text"]}

@app.get("/voice/transcript")
def get_voice_transcript():
    result = dict(_voice_transcript)
    _voice_transcript["text"] = ""
    return result

_voice_ui_state: dict = {"state": "idle", "answer_preview": ""}
_voice_answer: dict = {"text": "", "ts": 0}

@app.post("/voice/state")
async def set_voice_state(payload: dict):
    _voice_ui_state["state"]          = payload.get("state", "idle")
    _voice_ui_state["answer_preview"] = payload.get("answer_preview", "")
    return _voice_ui_state

@app.get("/voice/state")
def get_voice_state():
    return dict(_voice_ui_state)

@app.post("/voice/answer")
async def post_voice_answer(payload: dict):
    import time
    _voice_answer["text"] = payload.get("text", "").strip()
    _voice_answer["ts"]   = time.time()
    return {"status": "ok"}

@app.get("/voice/answer")
def get_voice_answer():
    """Return answer and clear it so it's only spoken once."""
    result = dict(_voice_answer)
    # Clear after read — prevents avatar from re-speaking the same answer
    _voice_answer["text"] = ""
    _voice_answer["ts"]   = 0
    return result


if __name__ == "__main__":
    import uvicorn
    uvicorn.run("fastapi_backend:app", host="0.0.0.0", port=8000, reload=True)