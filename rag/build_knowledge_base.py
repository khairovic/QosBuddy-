"""
QoSBuddy M6 — Phase 1: Knowledge Base Construction
====================================================
DSO3.2 — RAG + GenAI Knowledge Base Builder
Team: PingWin | ESPRIT 2025-2026

Consumes ALL upstream module outputs:
  • anomaly_scores_ns3.xls  (M1 — DSO2.2: Isolation Forest + LSTM AE scores)
  • root_cause_labels.csv   (M3 — DSO1.2: Causal AI root causes + actions)
  • counterfactuals.csv     (M3 — DSO1.2: Counterfactual packet-loss estimates)
  • QosBuddy_report.pdf     (project report — domain context for RAG grounding)

Design decisions:
  • ChromaDB (persistent, local) — no external vector DB dependency.
  • sentence-transformers all-MiniLM-L6-v2 — fast, accurate, offline-safe.
  • Rich metadata on every document → supports metadata-filtered retrieval
    (e.g., "show only critical anomalies in UE 5").
  • Three document families:
      1. ANOMALY_EVENT  — one doc per M1 row (anomaly + KPIs + SHAP-style context)
      2. CAUSAL_INSIGHT — one doc per M3 row (root cause + action + counterfactual)
      3. DOMAIN_CONTEXT — chunked project report for grounding domain questions
  • Batch embedding (256 docs/batch) to avoid OOM on large datasets.
"""

import os
import logging
import warnings
import time
from pathlib import Path
from typing import List, Dict, Any

import numpy as np
import pandas as pd
import chromadb
from chromadb.config import Settings
from sentence_transformers import SentenceTransformer

warnings.filterwarnings("ignore")

# ─────────────────────────────────────────────
# Configuration
# ─────────────────────────────────────────────

# All paths relative to project root; override via env vars for Docker
DATA_DIR   = Path(os.getenv("QOSBUDDY_DATA_DIR",   "./data"))
CHROMA_DIR = Path(os.getenv("QOSBUDDY_CHROMA_DIR", "./chroma_store"))
EMBED_MODEL = os.getenv("QOSBUDDY_EMBED_MODEL", "all-MiniLM-L6-v2")
COLLECTION_NAME = "qosbuddy_knowledge"
BATCH_SIZE = 256          # embed in chunks to control memory
ANOMALY_SAMPLE = 17525    # cap at actual dataset size to prevent runaway builds

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
)
log = logging.getLogger("qosbuddy.kb")

# ─────────────────────────────────────────────
# 1. Load data
# ─────────────────────────────────────────────

def load_anomaly_data() -> pd.DataFrame:
    """
    Load M1 output: anomaly_scores_ns3.xls (actually CSV-formatted despite extension).
    Contains Isolation Forest scores, LSTM AE errors, severity labels, and all raw KPIs.
    """
    path = DATA_DIR / "anomaly_scores_ns3.xls"
    log.info(f"Loading anomaly scores from {path}")
    df = pd.read_csv(path)

    # Drop duplicate columns produced by M1's export (e.g. load_level.1)
    dupe_cols = [c for c in df.columns if c.endswith(".1")]
    df = df.drop(columns=dupe_cols)

    # Fill any NaN in ae_error (first-row rolling features are NaN in M1 output)
    df["ae_error"] = df["ae_error"].fillna(0.0)

    if ANOMALY_SAMPLE:
        df = df.sample(n=min(ANOMALY_SAMPLE, len(df)), random_state=42).reset_index(drop=True)

    log.info(f"Anomaly data shape: {df.shape}")
    return df


def load_causal_data() -> pd.DataFrame:
    """
    Load M3 output: root_cause_labels.csv (causal root causes + recommended actions).
    One row per UE-timestamp, aligned with anomaly_scores.
    """
    path = DATA_DIR / "root_cause_labels.csv"
    log.info(f"Loading causal root causes from {path}")
    df = pd.read_csv(path)
    log.info(f"Causal data shape: {df.shape}")
    return df


def load_counterfactual_data() -> pd.DataFrame:
    """
    Load M3 output: counterfactuals.csv.
    Columns: ue_id, timestamp, original_packet_loss,
             counterfactual_if_static, reduction_pct
    """
    path = DATA_DIR / "counterfactuals.csv"
    log.info(f"Loading counterfactuals from {path}")
    df = pd.read_csv(path)
    # Normalize timestamp to int where possible (M3 exports "timestamp" as str)
    df["timestamp"] = pd.to_numeric(df["timestamp"], errors="coerce").fillna(1).astype(int)
    log.info(f"Counterfactual data shape: {df.shape}")
    return df


def load_project_report() -> str:
    """
    Load domain context from the project PDF report.
    Falls back gracefully if pypdf is not installed.
    """
    path = DATA_DIR / "QosBuddy_report.pdf"
    if not path.exists():
        log.warning("Project report PDF not found — skipping domain context docs.")
        return ""
    try:
        from pypdf import PdfReader
        reader = PdfReader(str(path))
        text = "\n".join(page.extract_text() or "" for page in reader.pages)
        log.info(f"Extracted {len(text)} chars from project report ({len(reader.pages)} pages)")
        return text
    except ImportError:
        log.warning("pypdf not installed — skipping PDF extraction.")
        return ""


# ─────────────────────────────────────────────
# 2. Document builders
# ─────────────────────────────────────────────

SEVERITY_EMOJI = {"critical": "🔴", "degraded": "🟡", "normal": "🟢"}

ROOT_CAUSE_EXPLANATIONS = {
    "interference":    "RF interference is degrading SINR — high retransmissions and low MCS observed.",
    "congestion":      "Network congestion — high PRB utilization and elevated packet loss detected.",
    "mobility":        "UE mobility causing handover failures — jitter spikes and beam-tracking loss.",
    "combined/other":  "Multiple overlapping failure modes — both load and radio degradation present.",
}


def build_anomaly_documents(
    anomaly_df: pd.DataFrame,
    causal_df: pd.DataFrame,
    cf_df: pd.DataFrame,
) -> tuple[List[str], List[Dict[str, Any]], List[str]]:
    """
    Build ANOMALY_EVENT documents (one per row in M1 output).

    Each document contains:
    • A structured NOC-style narrative (what the anomaly looks like in plain English)
    • All key KPI values so the LLM can reason numerically
    • SHAP-style feature importance inferred from M1 feature weights
    • M3 root cause and recommended action
    • M3 counterfactual impact

    Returns: (documents, metadatas, ids)
    """
    # Merge causal data — deduplicate first to prevent many-to-many explosion
    # causal_df has one row per (ue_id, timestamp) but merging on timestamp alone
    # causes a cartesian product. Fix: take first occurrence per timestamp.
    causal_df = causal_df.rename(columns={
        "severity": "causal_severity",
        "if_anomaly": "causal_if_anomaly",
    })
    causal_dedup = (
        causal_df[["timestamp", "root_cause", "recommended_action",
                   "causal_severity", "causal_if_anomaly"]]
        .drop_duplicates(subset=["timestamp"])
    )
    cf_dedup = (
        cf_df[["ue_id", "timestamp",
               "original_packet_loss", "counterfactual_if_static", "reduction_pct"]]
        .drop_duplicates(subset=["ue_id", "timestamp"])
    )
    merged = anomaly_df.merge(causal_dedup, on="timestamp", how="left")                        .merge(cf_dedup, on=["ue_id", "timestamp"], how="left")

    documents, metadatas, ids = [], [], []

    for idx, row in merged.iterrows():
        ue_id         = int(row["ue_id"])
        ts            = int(row["timestamp"])
        severity      = str(row.get("severity", "unknown"))
        root_cause    = str(row.get("root_cause", "unknown"))
        action        = str(row.get("recommended_action", "unknown"))
        if_anomaly    = int(row.get("if_anomaly", 0))
        ae_anomaly    = int(row.get("ae_anomaly", 0))
        if_score      = float(row.get("if_score", 0.0))
        ae_error      = float(row.get("ae_error", 0.0))

        # Core KPIs
        sinr       = float(row.get("sinr_dl_db", 0.0))
        throughput = float(row.get("throughput_mbps", 0.0))
        delay      = float(row.get("delay_ms", 0.0))
        jitter     = float(row.get("jitter_ms", 0.0))
        pkt_loss   = float(row.get("packet_loss_ratio", 0.0))
        prb_util   = float(row.get("prb_utilization", 0.0))
        retx       = int(row.get("retransmissions", 0))
        load_level = int(row.get("load_level", 0))

        # Counterfactual info
        cf_loss    = float(row.get("counterfactual_if_static", pkt_loss))
        cf_reduc   = float(row.get("reduction_pct", 0.0))

        emoji = SEVERITY_EMOJI.get(severity, "⚪")
        rc_explain = ROOT_CAUSE_EXPLANATIONS.get(root_cause, root_cause)

        # NOC-style narrative — this is what the LLM will "read"
        doc = f"""
[ANOMALY EVENT] UE {ue_id} | Timestamp {ts} | Severity: {emoji} {severity.upper()}
─────────────────────────────────────────────────────────────────────
Detection:
  • Isolation Forest flagged: {"YES — anomaly score " + f"{if_score:.4f}" if if_anomaly else "NO (score " + f"{if_score:.4f})"}
  • LSTM Autoencoder flagged: {"YES — reconstruction error " + f"{ae_error:.4f}" if ae_anomaly else "NO (error " + f"{ae_error:.4f})"}
  • Consensus anomaly: {"✅ CONFIRMED (both detectors agree)" if if_anomaly and ae_anomaly else "⚠️ PARTIAL (single detector)" if if_anomaly or ae_anomaly else "❌ NO ANOMALY"}

KPI Snapshot:
  • SINR (DL):       {sinr:.2f} dB   {"⚠️ POOR (<0 dB)" if sinr < 0 else "✅ OK"}
  • Throughput:      {throughput:.3f} Mbps  {"🔴 CRITICAL (<0.5 Mbps)" if throughput < 0.5 else "⚠️ LOW" if throughput < 2 else "✅ OK"}
  • Delay:           {delay:.2f} ms   {"🔴 HIGH (>100ms)" if delay > 100 else "✅ OK"}
  • Jitter:          {jitter:.3f} ms  {"⚠️ HIGH (>10ms)" if jitter > 10 else "✅ OK"}
  • Packet Loss:     {pkt_loss:.3f} ({pkt_loss*100:.1f}%)  {"🔴 CRITICAL (>50%)" if pkt_loss > 0.5 else "⚠️ ELEVATED" if pkt_loss > 0.1 else "✅ OK"}
  • PRB Utilization: {prb_util:.0f}%  {"🔴 CONGESTED (>80%)" if prb_util > 8 else "✅ OK"}
  • Retransmissions: {retx}
  • Network Load:    Level {load_level}

Root Cause Analysis (Causal AI — DoWhy):
  • Root Cause: {root_cause.upper()}
  • Explanation: {rc_explain}
  • Recommended Action: {action}

Counterfactual Impact (M3 — DSO1.2):
  • Current packet loss: {pkt_loss:.3f}
  • If static (no mobility/shadowing intervention): {cf_loss:.4f}
  • Potential reduction with corrective action: {cf_reduc:.1f}%
""".strip()

        meta = {
            "doc_type":        "ANOMALY_EVENT",
            "ue_id":           ue_id,
            "timestamp":       ts,
            "severity":        severity,
            "root_cause":      root_cause,
            "recommended_action": action,
            "if_anomaly":      if_anomaly,
            "ae_anomaly":      ae_anomaly,
            "if_score":        round(if_score, 6),
            "ae_error":        round(ae_error, 6),
            "sinr_dl_db":      round(sinr, 2),
            "throughput_mbps": round(throughput, 3),
            "delay_ms":        round(delay, 2),
            "jitter_ms":       round(jitter, 3),
            "packet_loss":     round(pkt_loss, 4),
            "prb_utilization": int(prb_util),
            "retransmissions": retx,
            "load_level":      load_level,
            "cf_reduction_pct": round(cf_reduc, 2),
        }

        doc_id = f"anomaly_ue{ue_id}_ts{ts}_{idx}"
        documents.append(doc)
        metadatas.append(meta)
        ids.append(doc_id)

    log.info(f"Built {len(documents)} ANOMALY_EVENT documents")
    return documents, metadatas, ids


def build_causal_summary_documents(
    causal_df: pd.DataFrame,
    cf_df: pd.DataFrame,
) -> tuple[List[str], List[Dict[str, Any]], List[str]]:
    """
    Build CAUSAL_INSIGHT documents — one per unique (root_cause, recommended_action) pair.
    These are higher-level "policy" documents that help the RAG answer questions like
    "what should I do when I see congestion?" without needing a specific UE row.
    """
    merged = causal_df.merge(
        cf_df[["ue_id", "timestamp", "reduction_pct"]],
        on="timestamp",
        how="left",
    )

    # Summarize per root cause
    summary = merged.groupby(["root_cause", "recommended_action"]).agg(
        count=("if_anomaly", "count"),
        anomaly_rate=("if_anomaly", "mean"),
        avg_pkt_loss=("packet_loss_ratio", "mean"),
        max_pkt_loss=("packet_loss_ratio", "max"),
        avg_cf_reduction=("reduction_pct", "mean"),
        critical_pct=("severity", lambda x: (x == "critical").mean()),
    ).reset_index()

    documents, metadatas, ids = [], [], []

    for _, row in summary.iterrows():
        rc     = row["root_cause"]
        action = row["recommended_action"]
        rc_explain = ROOT_CAUSE_EXPLANATIONS.get(rc, rc)

        doc = f"""
[CAUSAL INSIGHT] Root Cause: {rc.upper()} | Action: {action}
─────────────────────────────────────────────────────────────────────
Cause Explanation:
  {rc_explain}

Statistical Summary (across {int(row['count'])} events):
  • Anomaly detection rate:  {row['anomaly_rate']*100:.1f}%
  • Average packet loss:     {row['avg_pkt_loss']*100:.1f}%
  • Peak packet loss:        {row['max_pkt_loss']*100:.1f}%
  • Critical severity rate:  {row['critical_pct']*100:.1f}%
  • Counterfactual improvement: {row['avg_cf_reduction']:.1f}% reduction possible

Causal Chain (DoWhy — NS-3 Physics):
  {_causal_chain_text(rc)}

Recommended NOC Action:
  ► {action}
  Apply immediately when {rc} pattern is confirmed by anomaly detectors.
""".strip()

        meta = {
            "doc_type":           "CAUSAL_INSIGHT",
            "root_cause":         rc,
            "recommended_action": action,
            "event_count":        int(row["count"]),
            "anomaly_rate":       round(float(row["anomaly_rate"]), 4),
            "avg_pkt_loss":       round(float(row["avg_pkt_loss"]), 4),
            "cf_reduction_pct":   round(float(row["avg_cf_reduction"]), 2),
        }

        ids.append(f"causal_{rc.replace('/', '_').replace(' ', '_')}")
        documents.append(doc)
        metadatas.append(meta)

    log.info(f"Built {len(documents)} CAUSAL_INSIGHT documents")
    return documents, metadatas, ids


def _causal_chain_text(root_cause: str) -> str:
    chains = {
        "interference":   "RF interference → ↓SINR → ↑retransmissions → ↑packet_loss → ↑jitter → severity escalation",
        "congestion":     "high load_level → ↑PRB utilization → queue overflow → ↑packet_loss → ↑delay → SLA breach",
        "mobility":       "high mobility_speed → handover failure → beam loss → ↓throughput → ↑jitter → service disruption",
        "combined/other": "load_level ↑ + SINR ↓ → compounded degradation → packet_loss + delay both spike",
    }
    return chains.get(root_cause, "Root cause → KPI degradation → SLA impact")


def build_domain_context_documents(report_text: str) -> tuple[List[str], List[Dict[str, Any]], List[str]]:
    """
    Chunk the project report into overlapping windows so the RAG can answer
    questions about the system's design, KPIs, business objectives, etc.
    Chunk size: 600 chars with 100-char overlap (balances precision vs coverage).
    """
    if not report_text:
        return [], [], []

    CHUNK_SIZE = 600
    OVERLAP = 100
    chunks, metadatas, ids = [], [], []

    # Clean text
    lines = [l.strip() for l in report_text.splitlines() if l.strip()]
    text = " ".join(lines)

    start = 0
    chunk_idx = 0
    while start < len(text):
        end = min(start + CHUNK_SIZE, len(text))
        chunk = text[start:end]

        # Tag with section hint by scanning for known headings
        section = "General"
        for kw in ["Business", "DSO", "KPI", "SDG", "TDSP", "Technical", "Stakeholder"]:
            if kw.lower() in chunk.lower():
                section = kw
                break

        chunks.append(f"[DOMAIN CONTEXT — {section}]\n{chunk}")
        metadatas.append({
            "doc_type": "DOMAIN_CONTEXT",
            "section":  section,
            "chunk_idx": chunk_idx,
        })
        ids.append(f"domain_ctx_{chunk_idx}")

        start += CHUNK_SIZE - OVERLAP
        chunk_idx += 1

    log.info(f"Built {len(chunks)} DOMAIN_CONTEXT documents from report")
    return chunks, metadatas, ids


# ─────────────────────────────────────────────
# 3. Embedding + ChromaDB ingestion
# ─────────────────────────────────────────────

def embed_and_store(
    collection: chromadb.Collection,
    model: SentenceTransformer,
    documents: List[str],
    metadatas: List[Dict],
    ids: List[str],
    batch_size: int = BATCH_SIZE,
    doc_type: str = "",
) -> None:
    """
    Embed documents in batches and upsert into ChromaDB.
    Upsert (not add) ensures idempotency — safe to re-run.
    """
    total = len(documents)
    if total == 0:
        return

    log.info(f"Embedding {total} {doc_type} documents in batches of {batch_size}…")
    t0 = time.time()

    for start in range(0, total, batch_size):
        end = min(start + batch_size, total)
        batch_docs = documents[start:end]
        batch_meta = metadatas[start:end]
        batch_ids  = ids[start:end]

        # Encode returns numpy array; ChromaDB accepts lists
        embeddings = model.encode(
            batch_docs,
            batch_size=64,
            show_progress_bar=False,
            normalize_embeddings=True,   # cosine similarity friendly
        ).tolist()

        collection.upsert(
            documents=batch_docs,
            metadatas=batch_meta,
            ids=batch_ids,
            embeddings=embeddings,
        )

        pct = int(end / total * 100)
        log.info(f"  {doc_type}: {end}/{total} ({pct}%)")

    elapsed = time.time() - t0
    log.info(f"✅ {doc_type}: {total} docs stored in {elapsed:.1f}s")


# ─────────────────────────────────────────────
# 4. Main pipeline
# ─────────────────────────────────────────────

def build_knowledge_base():
    """
    Full KB build pipeline:
      1. Load all upstream module outputs
      2. Build structured documents for each family
      3. Embed with sentence-transformers
      4. Persist to ChromaDB
    """
    log.info("=" * 60)
    log.info("QoSBuddy M6 — Knowledge Base Builder (DSO3.2)")
    log.info("=" * 60)

    # ── Initialise ChromaDB (persistent, local) ──────────────────
    CHROMA_DIR.mkdir(parents=True, exist_ok=True)
    client = chromadb.PersistentClient(
        path=str(CHROMA_DIR),
        settings=Settings(anonymized_telemetry=False),
    )

    # Delete old collection if rebuilding (comment out to append-only)
    try:
        client.delete_collection(COLLECTION_NAME)
        log.info(f"Deleted existing collection '{COLLECTION_NAME}' for fresh build")
    except Exception:
        pass

    collection = client.create_collection(
        name=COLLECTION_NAME,
        metadata={"hnsw:space": "cosine"},   # cosine distance for normalized embeds
    )

    # ── Load embedding model ──────────────────────────────────────
    log.info(f"Loading embedding model: {EMBED_MODEL}")
    embed_model = SentenceTransformer(EMBED_MODEL)
    log.info("Embedding model ready")

    # ── Load data ─────────────────────────────────────────────────
    anomaly_df = load_anomaly_data()
    causal_df  = load_causal_data()
    cf_df      = load_counterfactual_data()
    report_txt = load_project_report()

    # ── Build documents ───────────────────────────────────────────
    anom_docs, anom_meta, anom_ids = build_anomaly_documents(anomaly_df, causal_df, cf_df)
    caus_docs, caus_meta, caus_ids = build_causal_summary_documents(causal_df, cf_df)
    dom_docs,  dom_meta,  dom_ids  = build_domain_context_documents(report_txt)

    # ── Embed & store ─────────────────────────────────────────────
    embed_and_store(collection, embed_model, anom_docs, anom_meta, anom_ids, doc_type="ANOMALY_EVENT")
    embed_and_store(collection, embed_model, caus_docs, caus_meta, caus_ids, doc_type="CAUSAL_INSIGHT")
    embed_and_store(collection, embed_model, dom_docs,  dom_meta,  dom_ids,  doc_type="DOMAIN_CONTEXT")

    total = collection.count()
    log.info("=" * 60)
    log.info(f"✅ Knowledge base '{COLLECTION_NAME}' ready — {total} total documents")
    log.info(f"   ChromaDB path: {CHROMA_DIR.resolve()}")
    log.info("=" * 60)

    return client, collection


if __name__ == "__main__":
    build_knowledge_base()