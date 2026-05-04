"""
QoSBuddy M6 — Phase 1: Knowledge Base Construction (v2)
========================================================
DSO3.2 — RAG + GenAI Knowledge Base Builder
Team: PingWin | ESPRIT 2025-2026

Consumes ALL upstream module outputs:
  • anomaly_scores_v2.csv    (M2 — DSO2.2: IF, COPOD, XGBoost, LSTM AE,
                               Anomaly Transformer, TranAD scores)
  • anomaly_scores_ns3.xls   (M1 legacy — fallback if v2 not available)
  • root_cause_labels.csv    (M3 — DSO1.2: Causal AI root causes + actions)
  • counterfactuals.csv      (M3 — DSO1.2: Counterfactual packet-loss estimates)
  • sla_breach_predictions.csv (M4 — DSO2.1: SLA breach risk predictions)
  • QosBuddy_report.pdf      (project report — domain context for RAG grounding)

v2 changes:
  • Supports new anomaly_scores_v2.csv with multi-model scores
    (IF, COPOD, OC-SVM, RF, XGBoost, GBM, LSTM-AE, AT, TranAD)
  • Updated root_cause_labels.csv with rule-based root causes:
    temporal_pattern_anomaly, model_disagreement_anomaly,
    subtle_anomaly, high_confidence_anomaly
  • Backward-compatible: falls back to anomaly_scores_ns3.xls if v2 missing
  • ANOMALY_EVENT documents now include multi-model consensus information
"""

import os
import logging
import warnings
import time
from pathlib import Path
from typing import List, Dict, Any, Tuple

import numpy as np
import pandas as pd
import chromadb
from chromadb.config import Settings
from sentence_transformers import SentenceTransformer

warnings.filterwarnings("ignore")

# ─────────────────────────────────────────────
# Configuration
# ─────────────────────────────────────────────

DATA_DIR        = Path(os.getenv("QOSBUDDY_DATA_DIR",   "./data"))
CHROMA_DIR      = Path(os.getenv("QOSBUDDY_CHROMA_DIR", "./chroma_store"))
EMBED_MODEL     = os.getenv("QOSBUDDY_EMBED_MODEL", "all-MiniLM-L6-v2")
COLLECTION_NAME = "qosbuddy_knowledge"
BATCH_SIZE      = 256
ANOMALY_SAMPLE  = None  # None = use all rows; set integer to cap

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
)
log = logging.getLogger("qosbuddy.kb")


# ─────────────────────────────────────────────
# Constants
# ─────────────────────────────────────────────

SEVERITY_EMOJI = {"critical": "🔴", "degraded": "🟡", "normal": "🟢"}

ROOT_CAUSE_EXPLANATIONS = {
    # v1 causal AI root causes (from casualAI.ipynb)
    "interference":    "RF interference is degrading SINR — high retransmissions and low MCS observed.",
    "congestion":      "Network congestion — high PRB utilization and elevated packet loss detected.",
    "mobility":        "UE mobility causing handover failures — jitter spikes and beam-tracking loss.",
    "combined/other":  "Multiple overlapping failure modes — both load and radio degradation present.",
    # v2 detection-based root causes (from modeling_DSO2_v2.ipynb)
    "temporal_pattern_anomaly":      "Temporal pattern anomaly — LSTM/Transformer detected sequential degradation invisible to classical models.",
    "model_disagreement_anomaly":    "Model disagreement — classical and deep detectors give conflicting signals, suggesting a borderline failure mode.",
    "subtle_anomaly":                "Subtle anomaly — low-confidence detection across all models; requires closer monitoring and contextual data.",
    "high_confidence_anomaly":       "High-confidence anomaly — strong consensus across IF, XGBoost, and deep models confirms genuine degradation.",
}

CAUSAL_CHAINS = {
    "interference":               "RF interference → ↓SINR → ↑retransmissions → ↑packet_loss → ↑jitter → severity escalation",
    "congestion":                 "high load_level → ↑PRB utilization → queue overflow → ↑packet_loss → ↑delay → SLA breach",
    "mobility":                   "high mobility_speed → handover failure → beam loss → ↓throughput → ↑jitter → service disruption",
    "combined/other":             "load_level ↑ + SINR ↓ → compounded degradation → packet_loss + delay both spike",
    "temporal_pattern_anomaly":   "time-series pattern shift → ↑reconstruction error → LSTM/Transformer flags → proactive alert",
    "model_disagreement_anomaly": "feature-level anomaly + normal temporal pattern → conflicting signals → requires expert review",
    "subtle_anomaly":             "marginal KPI deviation → low anomaly scores → early warning stage → monitor closely",
    "high_confidence_anomaly":    "multi-KPI breach → all models flag → confirmed degradation → immediate intervention",
}

# Score column names in anomaly_scores_v2.csv
V2_SCORE_COLUMNS = [
    "score_isolation_forest",
    "score_copod",
    "score_one-class_svm",
    "score_random_forest_supervised",
    "score_xgboost_supervised",
    "score_gradient_boosting_supervised",
    "score_lstm_autoencoder",
    "score_anomaly_transformer",
    "score_tranad",
]


# ─────────────────────────────────────────────
# 1. Data Loaders
# ─────────────────────────────────────────────

def load_anomaly_data() -> Tuple[pd.DataFrame, str]:
    """
    Load anomaly scores — prefers v2, falls back to v1.
    Returns (DataFrame, version_string).
    """
    v2_path = DATA_DIR / "anomaly_scores_v2.csv"
    v1_path = DATA_DIR / "anomaly_scores_ns3.xls"

    if v2_path.exists():
        log.info(f"Loading anomaly scores v2 from {v2_path}")
        df = pd.read_csv(v2_path)
        version = "v2"
    elif v1_path.exists():
        log.info(f"Loading anomaly scores v1 from {v1_path}")
        df = pd.read_csv(v1_path)
        dupe_cols = [c for c in df.columns if c.endswith(".1")]
        df = df.drop(columns=dupe_cols, errors="ignore")
        if "ae_error" in df.columns:
            df["ae_error"] = df["ae_error"].fillna(0.0)
        version = "v1"
    else:
        raise FileNotFoundError(
            f"No anomaly scores found. Expected {v2_path} or {v1_path}"
        )

    if ANOMALY_SAMPLE and len(df) > ANOMALY_SAMPLE:
        df = df.sample(n=ANOMALY_SAMPLE, random_state=42).reset_index(drop=True)

    log.info(f"Anomaly data ({version}): {df.shape[0]} rows × {df.shape[1]} cols")
    return df, version


def load_root_cause_labels() -> pd.DataFrame:
    """
    Load root cause labels. Handles both v1 (causal AI) and v2 (detection-based) formats.
    v1 columns: ue_id, timestamp, root_cause, recommended_action, severity, ...
    v2 columns: window_idx, true_label, root_cause, recommended_action, score_*
    """
    path = DATA_DIR / "root_cause_labels.csv"
    if not path.exists():
        log.warning("root_cause_labels.csv not found — causal documents will be empty")
        return pd.DataFrame()

    df = pd.read_csv(path)
    log.info(f"Root cause labels loaded: {df.shape}")
    return df


def load_counterfactual_data() -> pd.DataFrame:
    """Load counterfactuals.csv from causal AI phase."""
    path = DATA_DIR / "counterfactuals.csv"
    if not path.exists():
        log.warning("counterfactuals.csv not found — skipping counterfactual data")
        return pd.DataFrame()

    df = pd.read_csv(path)
    # Handle duplicate header rows if present
    if "timestamp" in df.columns:
        df = df[df["timestamp"] != "timestamp"].reset_index(drop=True)
        df["timestamp"] = pd.to_numeric(df["timestamp"], errors="coerce").fillna(1).astype(int)
    log.info(f"Counterfactual data: {df.shape}")
    return df


def load_sla_breach_data() -> pd.DataFrame:
    """Load SLA breach predictions from M4."""
    path = DATA_DIR / "sla_breach_predictions.csv"
    if not path.exists():
        log.warning("sla_breach_predictions.csv not found — skipping SLA_RISK documents")
        return pd.DataFrame()

    df = pd.read_csv(path)
    df_risk = df[df["is_high_risk_pred"] == 1].reset_index(drop=True)
    log.info(f"SLA breach data: {len(df)} total, {len(df_risk)} high-risk kept")
    return df_risk


def load_project_report() -> str:
    """Extract text from project PDF report."""
    path = DATA_DIR / "QosBuddy_report.pdf"
    if not path.exists():
        log.warning("Project report PDF not found — skipping domain context")
        return ""
    try:
        from pypdf import PdfReader
        reader = PdfReader(str(path))
        text = "\n".join(page.extract_text() or "" for page in reader.pages)
        log.info(f"Extracted {len(text)} chars from report ({len(reader.pages)} pages)")
        return text
    except ImportError:
        log.warning("pypdf not installed — skipping PDF extraction")
        return ""


# ─────────────────────────────────────────────
# 2. Document Builders
# ─────────────────────────────────────────────

def build_anomaly_documents_v2(
    anomaly_df: pd.DataFrame,
    rc_df: pd.DataFrame,
) -> Tuple[List[str], List[Dict[str, Any]], List[str]]:
    """
    Build ANOMALY_EVENT documents from v2 anomaly scores + root cause labels.
    v2 format: window_idx, true_label, score_* columns
    """
    documents, metadatas, ids = [], [], []

    # Merge root causes with anomaly scores on window_idx
    if not rc_df.empty and "window_idx" in rc_df.columns and "window_idx" in anomaly_df.columns:
        merged = anomaly_df.merge(
            rc_df[["window_idx", "root_cause", "recommended_action"]],
            on="window_idx",
            how="left",
        )
    else:
        merged = anomaly_df.copy()
        if "root_cause" not in merged.columns:
            merged["root_cause"] = "unknown"
        if "recommended_action" not in merged.columns:
            merged["recommended_action"] = "investigate further"

    for idx, row in merged.iterrows():
        window_idx = int(row.get("window_idx", idx))
        true_label = int(row.get("true_label", 0))
        root_cause = str(row.get("root_cause", "unknown"))
        action     = str(row.get("recommended_action", "investigate"))

        # Extract all model scores
        scores = {}
        for col in V2_SCORE_COLUMNS:
            if col in row.index:
                scores[col] = float(row[col]) if pd.notna(row[col]) else 0.0

        # Determine consensus
        label_text = "ANOMALY" if true_label == 1 else "NORMAL"
        rc_explain = ROOT_CAUSE_EXPLANATIONS.get(root_cause, root_cause)

        # Build score summary
        score_lines = []
        for col, val in scores.items():
            model_name = col.replace("score_", "").replace("_", " ").title()
            score_lines.append(f"  • {model_name}: {val:.4f}")
        score_block = "\n".join(score_lines) if score_lines else "  (no scores available)"

        doc = f"""
[ANOMALY EVENT — v2] Window {window_idx} | Label: {label_text}
─────────────────────────────────────────────────────────────────────
Ground Truth: {"🔴 ANOMALY CONFIRMED" if true_label == 1 else "🟢 NORMAL"}

Multi-Model Anomaly Scores:
{score_block}

Root Cause Classification: {root_cause.upper()}
  Explanation: {rc_explain}

Recommended Action: {action}
""".strip()

        meta = {
            "doc_type":           "ANOMALY_EVENT",
            "window_idx":         window_idx,
            "true_label":         true_label,
            "root_cause":         root_cause,
            "recommended_action": action,
            "severity":           "critical" if true_label == 1 else "normal",
            "data_version":       "v2",
        }

        # Add top scores to metadata for filtering
        if "score_xgboost_supervised" in scores:
            meta["xgb_score"] = round(scores["score_xgboost_supervised"], 6)
        if "score_isolation_forest" in scores:
            meta["if_score"] = round(scores["score_isolation_forest"], 6)

        ids.append(f"anomaly_v2_w{window_idx}")
        documents.append(doc)
        metadatas.append(meta)

    log.info(f"Built {len(documents)} ANOMALY_EVENT (v2) documents")
    return documents, metadatas, ids


def build_anomaly_documents_v1(
    anomaly_df: pd.DataFrame,
    causal_df: pd.DataFrame,
    cf_df: pd.DataFrame,
) -> Tuple[List[str], List[Dict[str, Any]], List[str]]:
    """
    Build ANOMALY_EVENT documents from v1 anomaly_scores_ns3.xls format.
    Preserved from original build_knowledge_base.py for backward compatibility.
    """
    # Merge causal data
    causal_cols = ["timestamp"]
    for c in ["root_cause", "recommended_action", "severity", "if_anomaly"]:
        if c in causal_df.columns:
            causal_cols.append(c)

    if not causal_df.empty and "timestamp" in causal_df.columns:
        causal_renamed = causal_df.copy()
        rename_map = {}
        for c in ["severity", "if_anomaly"]:
            if c in causal_renamed.columns and c in anomaly_df.columns:
                rename_map[c] = f"causal_{c}"
        causal_renamed = causal_renamed.rename(columns=rename_map)

        causal_dedup = causal_renamed.drop_duplicates(subset=["timestamp"])
        merged = anomaly_df.merge(causal_dedup, on="timestamp", how="left")
    else:
        merged = anomaly_df.copy()

    if not cf_df.empty and "ue_id" in cf_df.columns:
        cf_dedup = cf_df.drop_duplicates(subset=["ue_id", "timestamp"])
        merged = merged.merge(cf_dedup, on=["ue_id", "timestamp"], how="left")

    documents, metadatas, ids = [], [], []

    for idx, row in merged.iterrows():
        ue_id      = int(row.get("ue_id", 0))
        ts         = int(row.get("timestamp", 0))
        severity   = str(row.get("severity", "unknown"))
        root_cause = str(row.get("root_cause", "unknown"))
        action     = str(row.get("recommended_action", "unknown"))
        if_anomaly = int(row.get("if_anomaly", 0))
        ae_anomaly = int(row.get("ae_anomaly", 0))
        if_score   = float(row.get("if_score", 0.0))
        ae_error   = float(row.get("ae_error", 0.0))

        sinr       = float(row.get("sinr_dl_db", 0.0))
        throughput = float(row.get("throughput_mbps", 0.0))
        delay      = float(row.get("delay_ms", 0.0))
        jitter     = float(row.get("jitter_ms", 0.0))
        pkt_loss   = float(row.get("packet_loss_ratio", 0.0))
        prb_util   = float(row.get("prb_utilization", 0.0))
        retx       = int(row.get("retransmissions", 0))
        load_level = int(row.get("load_level", 0))
        cf_loss    = float(row.get("counterfactual_if_static", pkt_loss))
        cf_reduc   = float(row.get("reduction_pct", 0.0))

        emoji      = SEVERITY_EMOJI.get(severity, "⚪")
        rc_explain = ROOT_CAUSE_EXPLANATIONS.get(root_cause, root_cause)

        doc = f"""
[ANOMALY EVENT] UE {ue_id} | Timestamp {ts} | Severity: {emoji} {severity.upper()}
─────────────────────────────────────────────────────────────────────
Detection:
  • Isolation Forest: {"YES (score " + f"{if_score:.4f})" if if_anomaly else "NO (score " + f"{if_score:.4f})"}
  • LSTM Autoencoder: {"YES (error " + f"{ae_error:.4f})" if ae_anomaly else "NO (error " + f"{ae_error:.4f})"}

KPI Snapshot:
  • SINR (DL):       {sinr:.2f} dB
  • Throughput:      {throughput:.3f} Mbps
  • Delay:           {delay:.2f} ms
  • Jitter:          {jitter:.3f} ms
  • Packet Loss:     {pkt_loss:.3f} ({pkt_loss*100:.1f}%)
  • PRB Utilization: {prb_util:.0f}%
  • Retransmissions: {retx}
  • Network Load:    Level {load_level}

Root Cause (Causal AI — DoWhy): {root_cause.upper()}
  {rc_explain}
  Recommended Action: {action}

Counterfactual Impact:
  • Current packet loss: {pkt_loss:.3f}
  • If corrective action applied: {cf_loss:.4f}
  • Potential reduction: {cf_reduc:.1f}%
""".strip()

        meta = {
            "doc_type":           "ANOMALY_EVENT",
            "ue_id":              ue_id,
            "timestamp":          ts,
            "severity":           severity,
            "root_cause":         root_cause,
            "recommended_action": action,
            "if_anomaly":         if_anomaly,
            "ae_anomaly":         ae_anomaly,
            "if_score":           round(if_score, 6),
            "ae_error":           round(ae_error, 6),
            "packet_loss":        round(pkt_loss, 4),
            "load_level":         load_level,
            "cf_reduction_pct":   round(cf_reduc, 2),
            "data_version":       "v1",
        }

        ids.append(f"anomaly_ue{ue_id}_ts{ts}_{idx}")
        documents.append(doc)
        metadatas.append(meta)

    log.info(f"Built {len(documents)} ANOMALY_EVENT (v1) documents")
    return documents, metadatas, ids


def build_causal_summary_documents(
    rc_df: pd.DataFrame,
    cf_df: pd.DataFrame,
) -> Tuple[List[str], List[Dict[str, Any]], List[str]]:
    """
    Build CAUSAL_INSIGHT documents — one per unique (root_cause, recommended_action).
    Works with both v1 and v2 root cause label formats.
    """
    if rc_df.empty:
        return [], [], []

    # Build summary per root cause group
    agg_dict = {"root_cause": "count"}

    if "true_label" in rc_df.columns:
        agg_dict["true_label"] = "mean"  # anomaly rate

    summary_groups = rc_df.groupby(["root_cause", "recommended_action"]).agg(
        count=("root_cause", "count"),
    ).reset_index()

    # Add anomaly rate if available
    if "true_label" in rc_df.columns:
        ar = rc_df.groupby(["root_cause", "recommended_action"])["true_label"].mean().reset_index()
        ar.columns = ["root_cause", "recommended_action", "anomaly_rate"]
        summary_groups = summary_groups.merge(ar, on=["root_cause", "recommended_action"])
    else:
        summary_groups["anomaly_rate"] = 0.0

    # Add counterfactual reduction if available
    if not cf_df.empty and "reduction_pct" in cf_df.columns:
        avg_cf = cf_df["reduction_pct"].mean()
    else:
        avg_cf = 0.0

    documents, metadatas, ids = [], [], []

    for _, row in summary_groups.iterrows():
        rc     = row["root_cause"]
        action = row["recommended_action"]
        count  = int(row["count"])
        ar     = float(row.get("anomaly_rate", 0.0))

        rc_explain   = ROOT_CAUSE_EXPLANATIONS.get(rc, rc)
        causal_chain = CAUSAL_CHAINS.get(rc, "Root cause → KPI degradation → SLA impact")

        doc = f"""
[CAUSAL INSIGHT] Root Cause: {rc.upper()} | Action: {action}
─────────────────────────────────────────────────────────────────────
Cause Explanation:
  {rc_explain}

Statistical Summary (across {count} events):
  • Anomaly detection rate: {ar*100:.1f}%
  • Counterfactual improvement: {avg_cf:.1f}% reduction possible

Causal Chain:
  {causal_chain}

Recommended NOC Action:
  ► {action}
  Apply immediately when {rc} pattern is confirmed by anomaly detectors.
""".strip()

        meta = {
            "doc_type":           "CAUSAL_INSIGHT",
            "root_cause":         rc,
            "recommended_action": action,
            "event_count":        count,
            "anomaly_rate":       round(ar, 4),
        }

        ids.append(f"causal_{rc.replace('/', '_').replace(' ', '_')}")
        documents.append(doc)
        metadatas.append(meta)

    log.info(f"Built {len(documents)} CAUSAL_INSIGHT documents")
    return documents, metadatas, ids


def build_sla_risk_documents(
    sla_df: pd.DataFrame,
) -> Tuple[List[str], List[Dict[str, Any]], List[str]]:
    """Build SLA_RISK documents from M4 SLA breach predictions."""
    if sla_df.empty:
        return [], [], []

    LOAD_LABELS = {1: "LOW", 2: "MEDIUM", 3: "HIGH"}
    documents, metadatas, ids = [], [], []

    for idx, row in sla_df.iterrows():
        ue_id    = int(row["ue_id"])
        ts       = int(row["timestamp"])
        prob     = float(row["risk_proba"])
        load_lv  = int(row["load_level"])
        prb      = float(row.get("prb_utilization", 0))
        retx     = int(row.get("retransmissions", 0))
        risk_pct = round(prob * 100, 1)
        urgency  = "🔴 IMMINENT" if prob > 0.85 else "🟡 ELEVATED"

        if prb > 80:
            action_hint = "Reduce PRB congestion — throttle or reroute nearby UEs"
        elif retx > 5:
            action_hint = "Investigate retransmission cause — likely interference or mobility"
        else:
            action_hint = "Monitor MCS improvement — consider beam adjustment"

        doc = f"""
[SLA BREACH RISK] UE {ue_id} | Timestamp {ts}
{urgency} — Risk Probability: {risk_pct}%
────────────────────────────────────────────
Prediction Source: XGBoost Early Warning Model (DSO2.1 · M4)
Load Level: {LOAD_LABELS.get(load_lv, f"Level {load_lv}")} | PRB: {prb:.1f}% | Retx: {retx}
Suggested Action: {action_hint}
""".strip()

        metadatas.append({
            "doc_type":    "SLA_RISK",
            "ue_id":       ue_id,
            "timestamp":   ts,
            "risk_proba":  round(prob, 4),
            "severity":    "critical" if prob > 0.85 else "degraded",
        })
        # Include row index — Eya's v11 SLA dataset has multiple snapshots per
        # (ue_id, timestamp), so the (ue,ts) key alone is no longer unique.
        ids.append(f"sla_risk_ue{ue_id}_ts{ts}_r{idx}")
        documents.append(doc)

    log.info(f"Built {len(documents)} SLA_RISK documents")
    return documents, metadatas, ids


def build_domain_context_documents(
    report_text: str,
) -> Tuple[List[str], List[Dict[str, Any]], List[str]]:
    """Chunk project report into overlapping windows for RAG grounding."""
    if not report_text:
        return [], [], []

    CHUNK_SIZE = 600
    OVERLAP    = 100
    chunks, metadatas, ids = [], [], []

    lines = [l.strip() for l in report_text.splitlines() if l.strip()]
    text  = " ".join(lines)

    start     = 0
    chunk_idx = 0
    while start < len(text):
        end   = min(start + CHUNK_SIZE, len(text))
        chunk = text[start:end]

        section = "General"
        for kw in ["Business", "DSO", "KPI", "SDG", "TDSP", "Technical", "Stakeholder"]:
            if kw.lower() in chunk.lower():
                section = kw
                break

        chunks.append(f"[DOMAIN CONTEXT — {section}]\n{chunk}")
        metadatas.append({
            "doc_type":  "DOMAIN_CONTEXT",
            "section":   section,
            "chunk_idx": chunk_idx,
        })
        ids.append(f"domain_ctx_{chunk_idx}")

        start     += CHUNK_SIZE - OVERLAP
        chunk_idx += 1

    log.info(f"Built {len(chunks)} DOMAIN_CONTEXT documents")
    return chunks, metadatas, ids


# ─────────────────────────────────────────────
# 3. Embedding + ChromaDB Ingestion
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
    """Embed documents in batches and upsert into ChromaDB."""
    total = len(documents)
    if total == 0:
        return

    log.info(f"Embedding {total} {doc_type} documents in batches of {batch_size}…")
    t0 = time.time()

    for start in range(0, total, batch_size):
        end        = min(start + batch_size, total)
        batch_docs = documents[start:end]
        batch_meta = metadatas[start:end]
        batch_ids  = ids[start:end]

        embeddings = model.encode(
            batch_docs,
            batch_size=64,
            show_progress_bar=False,
            normalize_embeddings=True,
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
# 4. Main Pipeline
# ─────────────────────────────────────────────

def build_knowledge_base():
    """
    Full KB build pipeline:
      1. Load all upstream outputs (auto-detecting v1 vs v2)
      2. Build structured documents for each family
      3. Embed with sentence-transformers
      4. Persist to ChromaDB
    """
    log.info("=" * 60)
    log.info("QoSBuddy M6 — Knowledge Base Builder v2 (DSO3.2)")
    log.info("=" * 60)

    # ── Initialize ChromaDB ──
    CHROMA_DIR.mkdir(parents=True, exist_ok=True)
    client = chromadb.PersistentClient(
        path=str(CHROMA_DIR),
        settings=Settings(anonymized_telemetry=False),
    )

    try:
        client.delete_collection(COLLECTION_NAME)
        log.info(f"Deleted existing collection '{COLLECTION_NAME}' for fresh build")
    except Exception:
        pass

    collection = client.create_collection(
        name=COLLECTION_NAME,
        metadata={"hnsw:space": "cosine"},
    )

    # ── Load embedding model ──
    log.info(f"Loading embedding model: {EMBED_MODEL}")
    embed_model = SentenceTransformer(EMBED_MODEL)
    log.info("Embedding model ready")

    # ── Load data ──
    anomaly_df, data_version = load_anomaly_data()
    rc_df      = load_root_cause_labels()
    cf_df      = load_counterfactual_data()
    sla_df     = load_sla_breach_data()
    report_txt = load_project_report()

    # ── Build documents based on data version ──
    if data_version == "v2":
        anom_docs, anom_meta, anom_ids = build_anomaly_documents_v2(anomaly_df, rc_df)
    else:
        anom_docs, anom_meta, anom_ids = build_anomaly_documents_v1(anomaly_df, rc_df, cf_df)

    caus_docs, caus_meta, caus_ids = build_causal_summary_documents(rc_df, cf_df)
    sla_docs,  sla_meta,  sla_ids  = build_sla_risk_documents(sla_df)
    dom_docs,  dom_meta,  dom_ids  = build_domain_context_documents(report_txt)

    # ── Embed & store ──
    embed_and_store(collection, embed_model, anom_docs, anom_meta, anom_ids, doc_type="ANOMALY_EVENT")
    embed_and_store(collection, embed_model, caus_docs, caus_meta, caus_ids, doc_type="CAUSAL_INSIGHT")
    embed_and_store(collection, embed_model, sla_docs,  sla_meta,  sla_ids,  doc_type="SLA_RISK")
    embed_and_store(collection, embed_model, dom_docs,  dom_meta,  dom_ids,  doc_type="DOMAIN_CONTEXT")

    total = collection.count()
    log.info("=" * 60)
    log.info(f"✅ Knowledge base '{COLLECTION_NAME}' ready — {total} documents")
    log.info(f"   Data version: {data_version}")
    log.info(f"   ChromaDB path: {CHROMA_DIR.resolve()}")
    log.info("=" * 60)

    return client, collection


if __name__ == "__main__":
    build_knowledge_base()