"""
QoSBuddy M6 — FastAPI Backend (DSO3.2)
========================================
REST endpoints consumed by:
  • M1 (DSO2.2/2.3) → POST /ingest/anomaly  (push new anomaly events in real-time)
  • Streamlit dashboard → GET /kpi/summary, GET /anomalies, POST /rag/query
  • External NOC systems → GET /health, GET /docs (Swagger UI)

Design:
  • Lifespan context manager (FastAPI 0.95+) loads ChromaDB + embed model once.
  • Background tasks for KB upsert (non-blocking for M1 caller).
  • Pydantic v2 models for request/response validation.
  • CORS enabled for Streamlit (same machine, different port).
"""

import logging
import os
from contextlib import asynccontextmanager
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
# Global state (loaded once at startup)
# ─────────────────────────────────────────────

_collection   = None
_embed_model  = None
_chain        = None
_anomaly_df   = None  # in-memory cache for fast KPI queries
_causal_df    = None
_cf_df        = None

DATA_DIR = Path(os.getenv("QOSBUDDY_DATA_DIR", "./data"))


@asynccontextmanager
async def lifespan(app: FastAPI):
    """Load all heavy resources on startup; release on shutdown."""
    global _collection, _embed_model, _chain, _anomaly_df, _causal_df, _cf_df

    log.info("QoSBuddy M6 API starting up…")

    # Load data CSVs
    try:
        _anomaly_df = pd.read_csv(DATA_DIR / "anomaly_scores_ns3.xls")
        dupe_cols = [c for c in _anomaly_df.columns if c.endswith(".1")]
        _anomaly_df = _anomaly_df.drop(columns=dupe_cols)
        _anomaly_df["ae_error"] = _anomaly_df["ae_error"].fillna(0.0)

        _causal_df = pd.read_csv(DATA_DIR / "root_cause_labels.csv")
        _cf_df     = pd.read_csv(DATA_DIR / "counterfactuals.csv")
        log.info("Data CSVs loaded")
    except Exception as e:
        log.error(f"Data load failed: {e}")

    # Load RAG stack
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

    yield  # ← app runs here

    log.info("QoSBuddy M6 API shutting down")


app = FastAPI(
    title="QoSBuddy M6 API",
    description="DSO3.2 — RAG + GenAI + NOC Dashboard backend for the PingWin QoSBuddy system",
    version="1.0.0",
    lifespan=lifespan,
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],    # tighten to ["http://localhost:8501"] in production
    allow_methods=["*"],
    allow_headers=["*"],
)


# ─────────────────────────────────────────────
# Pydantic schemas
# ─────────────────────────────────────────────

class AnomalyEvent(BaseModel):
    """Schema for M1 → M6 real-time anomaly push."""
    ue_id:             int
    timestamp:         int
    severity:          str          # critical | degraded | normal
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
    # Optional M3 fields
    root_cause:        Optional[str] = None
    recommended_action: Optional[str] = None
    cf_reduction_pct:  Optional[float] = None


class RAGQueryRequest(BaseModel):
    question:          str
    mode:              str = Field("general", description="general | anomaly | causal | domain")
    ue_id:             Optional[int]   = None
    severity_filter:   Optional[str]   = None
    root_cause_filter: Optional[str]   = None


class RAGQueryResponse(BaseModel):
    answer:    str
    sources:   List[Dict[str, Any]]
    question:  str
    mode:      str


class KPISummary(BaseModel):
    total_events:      int
    anomaly_rate_pct:  float
    avg_jitter_ms:     float
    avg_packet_loss_pct: float
    avg_throughput_mbps: float
    avg_delay_ms:      float
    critical_count:    int
    degraded_count:    int
    normal_count:      int
    top_root_cause:    str


# ─────────────────────────────────────────────
# Endpoints
# ─────────────────────────────────────────────

@app.get("/health")
def health():
    """Health check — confirms API, data, and RAG are all live."""
    return {
        "status":      "ok",
        "rag_ready":   _collection is not None,
        "kb_docs":     _collection.count() if _collection else 0,
        "data_loaded": _anomaly_df is not None,
        "rows":        len(_anomaly_df) if _anomaly_df is not None else 0,
    }


@app.get("/kpi/summary", response_model=KPISummary)
def get_kpi_summary(
    severity: Optional[str] = Query(None, description="Filter by severity"),
    ue_id:    Optional[int]  = Query(None, description="Filter by UE ID"),
):
    """
    Real-time KPI summary for the Streamlit dashboard KPI cards.
    Supports metadata filtering.
    """
    if _anomaly_df is None:
        raise HTTPException(503, "Data not loaded")

    df = _anomaly_df.copy()
    if severity: df = df[df["severity"] == severity]
    if ue_id:    df = df[df["ue_id"] == ue_id]

    if len(df) == 0:
        raise HTTPException(404, "No data matching filters")

    top_rc = "unknown"
    if _causal_df is not None:
        rc = _causal_df["root_cause"].value_counts()
        top_rc = rc.index[0] if len(rc) > 0 else "unknown"

    return KPISummary(
        total_events=len(df),
        anomaly_rate_pct=round(df["if_anomaly"].mean() * 100, 2),
        avg_jitter_ms=round(df["jitter_ms"].mean(), 3),
        avg_packet_loss_pct=round(df["packet_loss_ratio"].mean() * 100, 2),
        avg_throughput_mbps=round(df["throughput_mbps"].mean(), 3),
        avg_delay_ms=round(df["delay_ms"].mean(), 2),
        critical_count=int((df["severity"] == "critical").sum()),
        degraded_count=int((df["severity"] == "degraded").sum()),
        normal_count=int((df["severity"] == "normal").sum()),
        top_root_cause=top_rc,
    )


@app.get("/anomalies")
def get_anomalies(
    severity:  Optional[str] = Query(None),
    ue_id:     Optional[int]  = Query(None),
    limit:     int = Query(100, le=5000),
    only_flagged: bool = Query(False, description="Only return IF-flagged anomalies"),
):
    """Return anomaly events as JSON — used by Streamlit triage board."""
    if _anomaly_df is None:
        raise HTTPException(503, "Data not loaded")

    df = _anomaly_df.copy()
    if severity:      df = df[df["severity"] == severity]
    if ue_id:         df = df[df["ue_id"] == ue_id]
    if only_flagged:  df = df[df["if_anomaly"] == 1]

    df = df.sort_values("if_score", ascending=False).head(limit)
    return {"count": len(df), "data": df.to_dict(orient="records")}


@app.post("/ingest/anomaly")
async def ingest_anomaly(event: AnomalyEvent, background_tasks: BackgroundTasks):
    """
    M1 → M6 real-time push endpoint.
    Called by M1 FastAPI after scoring each window.
    Upserts the new event into the ChromaDB KB in the background (non-blocking).
    """
    # Append to in-memory DataFrame immediately (for /kpi/summary freshness)
    global _anomaly_df
    new_row = pd.DataFrame([event.model_dump()])
    if _anomaly_df is not None:
        _anomaly_df = pd.concat([_anomaly_df, new_row], ignore_index=True)

    # Kick off background KB upsert
    if _collection is not None and _embed_model is not None:
        background_tasks.add_task(_upsert_event_to_kb, event)

    return {"status": "accepted", "ue_id": event.ue_id, "timestamp": event.timestamp}


def _upsert_event_to_kb(event: AnomalyEvent):
    """Background task: embed and upsert a single new anomaly event into ChromaDB."""
    from rag.build_knowledge_base import SEVERITY_EMOJI, ROOT_CAUSE_EXPLANATIONS

    sev = event.severity
    rc  = event.root_cause or "unknown"
    emoji = SEVERITY_EMOJI.get(sev, "⚪")
    rc_explain = ROOT_CAUSE_EXPLANATIONS.get(rc, rc)

    doc = f"""
[ANOMALY EVENT — LIVE] UE {event.ue_id} | Timestamp {event.timestamp} | Severity: {emoji} {sev.upper()}
Detection: IF={'YES' if event.if_anomaly else 'NO'} (score={event.if_score:.4f}) | AE={'YES' if event.ae_anomaly else 'NO'} (error={event.ae_error:.4f})
KPIs: sinr={event.sinr_dl_db:.2f}dB | throughput={event.throughput_mbps:.3f}Mbps | delay={event.delay_ms:.2f}ms | jitter={event.jitter_ms:.3f}ms | pkt_loss={event.packet_loss_ratio:.3f}
Root Cause: {rc} — {rc_explain}
Action: {event.recommended_action or 'pending M3 analysis'}
Counterfactual reduction: {event.cf_reduction_pct or 0:.1f}%
""".strip()

    embedding = _embed_model.encode([doc], normalize_embeddings=True).tolist()
    _collection.upsert(
        documents=[doc],
        metadatas=[{
            "doc_type":           "ANOMALY_EVENT",
            "ue_id":              event.ue_id,
            "timestamp":          event.timestamp,
            "severity":           sev,
            "root_cause":         rc,
            "recommended_action": event.recommended_action or "",
            "if_anomaly":         event.if_anomaly,
            "if_score":           round(event.if_score, 6),
            "ae_error":           round(event.ae_error, 6),
            "packet_loss":        round(event.packet_loss_ratio, 4),
        }],
        ids=[f"live_ue{event.ue_id}_ts{event.timestamp}"],
        embeddings=embedding,
    )
    log.info(f"Upserted live event: UE {event.ue_id} ts={event.timestamp} sev={sev}")


@app.post("/rag/query", response_model=RAGQueryResponse)
def rag_query(req: RAGQueryRequest):
    """
    RAG question-answering endpoint.
    Retrieves context from ChromaDB + calls Ollama for a structured NOC response.
    """
    if _collection is None or _embed_model is None:
        raise HTTPException(503, "RAG stack not initialised. Run build_knowledge_base.py first.")

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
    """Root cause statistics — used by Streamlit causal page."""
    if _causal_df is None:
        raise HTTPException(503, "Data not loaded")

    summary = _causal_df.groupby(["root_cause","recommended_action"]).agg(
        count=("if_anomaly","count"),
        anomaly_rate=("if_anomaly","mean"),
        avg_pkt_loss=("packet_loss_ratio","mean"),
        critical_pct=("severity", lambda x: (x=="critical").mean()),
    ).reset_index()

    return {"data": summary.round(4).to_dict(orient="records")}


@app.get("/counterfactuals")
def get_counterfactuals(ue_id: Optional[int] = Query(None)):
    """Counterfactual analysis endpoint — used by causal explorer slider."""
    if _cf_df is None:
        raise HTTPException(503, "Data not loaded")

    df = _cf_df.copy()
    if ue_id:
        df = df[df["ue_id"] == ue_id]

    return {
        "count":              len(df),
        "avg_reduction_pct":  round(df["reduction_pct"].mean(), 2),
        "avg_original_loss":  round(df["original_packet_loss"].mean(), 4),
        "avg_cf_loss":        round(df["counterfactual_if_static"].mean(), 4),
        "sample":             df.head(20).to_dict(orient="records"),
    }


if __name__ == "__main__":
    import uvicorn
    uvicorn.run("fastapi_backend:app", host="0.0.0.0", port=8000, reload=True)
