"""Scoring jobs for the MLOps pipeline.

Each job reads `data/live_log.jsonl` (cumulative live data from bridge.py)
and emits a `*_live.csv` next to the static historical file. The dashboard
loaders concat both, so the static training data is never overwritten and
the pipeline is fully re-runnable.

Outputs:
  data/anomaly_scores_v2_live.csv     (Benchmarking page)
  data/root_cause_labels_live.csv     (Causal Analysis + KPI Overview)
  data/sla_breach_predictions_live.csv (SLA Risk page)
  data/counterfactuals_live.csv       (Causal Analysis counterfactual panel)
  data/severity_triage_live.csv       (Severity Triage page)

Jobs are deliberately small and rule-based where possible. To upgrade to
notebook-grade logic, replace the body of any `score_*` function — the
runner contract (return dict of metrics) is stable.
"""

from __future__ import annotations

import json
import logging
import os
import tempfile
from pathlib import Path
from typing import Any, Dict, List

import numpy as np
import pandas as pd

from .runner import runner

log = logging.getLogger("qosbuddy.mlops.jobs")

DATA_DIR = Path(os.getenv("QOSBUDDY_DATA_DIR", "./data"))
LIVE_LOG_PATH = DATA_DIR / "live_log.jsonl"
THRESHOLD = int(os.getenv("QOSBUDDY_PIPELINE_THRESHOLD", "100"))


# ── shared helpers ──────────────────────────────────────────────────────
def _load_live_df() -> pd.DataFrame:
    """Read the JSONL log into a DataFrame. Returns empty if file missing."""
    if not LIVE_LOG_PATH.exists():
        return pd.DataFrame()
    rows: List[dict] = []
    try:
        with open(LIVE_LOG_PATH, "r", encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                try:
                    rows.append(json.loads(line))
                except Exception:
                    continue
    except Exception:
        log.exception("mlops: failed to read live_log.jsonl")
    return pd.DataFrame(rows) if rows else pd.DataFrame()


def _atomic_write_csv(df: pd.DataFrame, path: Path) -> None:
    """Write to .tmp then os.replace so readers never see a partial file."""
    path.parent.mkdir(parents=True, exist_ok=True)
    fd, tmp = tempfile.mkstemp(prefix=path.name + ".", suffix=".tmp", dir=path.parent)
    os.close(fd)
    try:
        df.to_csv(tmp, index=False)
        os.replace(tmp, path)
    except Exception:
        try: os.unlink(tmp)
        except Exception: pass
        raise


def _coerce_kpis(df: pd.DataFrame) -> pd.DataFrame:
    """Make KPI columns numeric in-place; harmless on missing columns."""
    for c in ("sinr_dl_db", "throughput_mbps", "delay_ms", "jitter_ms",
              "packet_loss_ratio", "prb_utilization", "retransmissions",
              "load_level", "mcs_dl"):
        if c in df.columns:
            df[c] = pd.to_numeric(df[c], errors="coerce")
    return df


# ── 1. Anomaly scoring → anomaly_scores_v2_live.csv ─────────────────────
def score_anomaly() -> Dict[str, Any]:
    """Project the live ensemble scores (already computed by the API on
    ingest) into the v2 schema the Benchmarking page consumes."""
    live = _load_live_df()
    if len(live) < THRESHOLD:
        return {"skipped": "below threshold", "live_rows": int(len(live)), "threshold": THRESHOLD}

    out = pd.DataFrame()
    out["window_idx"] = range(len(live))
    out["true_label"] = (
        pd.to_numeric(live.get("_live_anomaly", pd.Series([0]*len(live))), errors="coerce")
          .fillna(0).astype(int)
    )

    scores = live.get("_live_scores", pd.Series([{}]*len(live)))
    sdf = pd.json_normalize(scores.apply(lambda x: x if isinstance(x, dict) else {}))

    rename = {
        "isolation_forest":  "score_isolation_forest",
        "random_forest":     "score_random_forest_supervised",
        "gradient_boosting": "score_gradient_boosting_supervised",
        "copod":             "score_copod",
        "one_class_svm":     "score_one-class_svm",
        "xgboost":           "score_xgboost_supervised",
        "lstm_autoencoder":  "score_lstm_autoencoder",
        "anomaly_transformer": "score_anomaly_transformer",
        "tranad":            "score_tranad",
    }
    for src, dst in rename.items():
        if src in sdf.columns:
            out[dst] = pd.to_numeric(sdf[src], errors="coerce").fillna(0)

    # Default placeholders for downstream joins
    out["recommended_action"] = np.where(
        out["true_label"] == 1,
        "monitor closely + collect more contextual data",
        "no action — within nominal bounds",
    )
    out["root_cause"] = np.where(out["true_label"] == 1, "subtle_anomaly", "normal")

    output = DATA_DIR / "anomaly_scores_v2_live.csv"
    _atomic_write_csv(out, output)
    return {
        "rows":      int(len(out)),
        "anomalies": int(out["true_label"].sum()),
        "score_cols": int(sum(1 for c in out.columns if c.startswith("score_"))),
        "output":    output.name,
    }


# ── 2. Causal labelling → root_cause_labels_live.csv ────────────────────
_RC_ACTIONS = {
    "interference":   "frequency_reallocation + power_control",
    "congestion":     "load_balancing + traffic_throttling",
    "mobility":       "handover_optimization + beam_tracking",
    "combined/other": "multi-action: reroute + prioritize",
}


def _classify_root_cause(row: pd.Series) -> str:
    """Heuristic root cause from KPI signals — matches the labels Eya's
    notebook produces and the dashboard's CAUSAL_CHAINS keys."""
    sinr = row.get("sinr_dl_db")
    prb  = row.get("prb_utilization") or 0
    retx = row.get("retransmissions") or 0
    load = row.get("load_level") or 0
    delay= row.get("delay_ms") or 0

    flags = 0
    if sinr is not None and not pd.isna(sinr) and sinr < 0: flags += 1
    if prb > 0.7 or load >= 3:                              flags += 1
    if retx > 5 or delay > 200:                             flags += 1
    if flags >= 2:
        return "combined/other"
    if sinr is not None and not pd.isna(sinr) and sinr < 0: return "interference"
    if prb > 0.7 or load >= 3:                              return "congestion"
    if retx > 5 or delay > 200:                             return "mobility"
    return "combined/other"


def score_causal() -> Dict[str, Any]:
    """Compute root_cause + severity + recommended_action for each live row."""
    live = _load_live_df()
    if len(live) < THRESHOLD:
        return {"skipped": "below threshold", "live_rows": int(len(live)), "threshold": THRESHOLD}

    df = _coerce_kpis(live.copy())

    # Severity from KPI + anomaly flag
    def _sev(r: pd.Series) -> str:
        loss = r.get("packet_loss_ratio") or 0
        anom = r.get("_live_anomaly", 0)
        if anom == 1 and loss > 0.5: return "critical"
        if anom == 1 or loss > 0.2:  return "degraded"
        return "normal"
    df["severity"] = df.apply(_sev, axis=1)
    df["if_anomaly"] = pd.to_numeric(df.get("_live_anomaly", 0), errors="coerce").fillna(0).astype(int)
    df["root_cause"] = df.apply(_classify_root_cause, axis=1)
    df["recommended_action"] = df["root_cause"].map(_RC_ACTIONS).fillna("monitor")

    keep = [c for c in ("timestamp","severity","if_anomaly","root_cause",
                        "recommended_action","packet_loss_ratio") if c in df.columns]
    out = df[keep].copy()
    output = DATA_DIR / "root_cause_labels_live.csv"
    _atomic_write_csv(out, output)
    return {
        "rows":         int(len(out)),
        "anomalies":    int(out["if_anomaly"].sum()) if "if_anomaly" in out else 0,
        "by_root_cause": out["root_cause"].value_counts().to_dict() if "root_cause" in out else {},
        "output":       output.name,
    }


# ── 3. SLA risk → sla_breach_predictions_live.csv ───────────────────────
def score_sla() -> Dict[str, Any]:
    """Risk score for each live row.

    Uses a logistic-ish blend of packet_loss_ratio + delay_ms + retransmissions.
    Intended as a stand-in until the trained XGBoost early-warning model is
    bundled into anomaly_models/."""
    live = _load_live_df()
    if len(live) < THRESHOLD:
        return {"skipped": "below threshold", "live_rows": int(len(live)), "threshold": THRESHOLD}

    df = _coerce_kpis(live.copy())

    loss  = df.get("packet_loss_ratio", pd.Series([0.0]*len(df))).fillna(0).astype(float)
    delay = df.get("delay_ms",          pd.Series([0.0]*len(df))).fillna(0).astype(float)
    retx  = df.get("retransmissions",   pd.Series([0.0]*len(df))).fillna(0).astype(float)
    z = 1.8*loss + (delay / 500.0) + (retx / 20.0)
    risk = 1.0 / (1.0 + np.exp(-z + 1.0))

    out = pd.DataFrame()
    if "timestamp" in df.columns: out["timestamp"] = df["timestamp"]
    if "ue_id"     in df.columns: out["ue_id"]     = df["ue_id"]
    for c in ("ue_name","gnb_id","gnb_name","ue_lat","ue_lon","ue_dist_to_gnb_m",
              "load_level","prb_utilization","retransmissions","mcs_dl"):
        if c in df.columns: out[c] = df[c]
    out["risk_proba"]      = np.clip(risk, 0.0, 1.0).round(4)
    out["threshold_used"]  = 0.6
    out["is_high_risk_pred"] = (out["risk_proba"] >= 0.6).astype(int)
    out["true_label"]        = pd.to_numeric(df.get("_live_anomaly", 0), errors="coerce").fillna(0).astype(int)
    out["version"]           = "live_heuristic_v1"

    output = DATA_DIR / "sla_breach_predictions_live.csv"
    _atomic_write_csv(out, output)
    return {
        "rows":          int(len(out)),
        "high_risk":     int(out["is_high_risk_pred"].sum()),
        "high_risk_pct": round(float(out["is_high_risk_pred"].mean() * 100), 2),
        "output":        output.name,
    }


# ── 4. Counterfactuals → counterfactuals_live.csv ───────────────────────
def score_counterfactuals() -> Dict[str, Any]:
    """Per-row counterfactual: what packet loss would look like if the
    treatment (load_level → 1) were applied. Uses a constant 49.71% reduction
    drawn from the historical DoWhy ATE so the table aligns with the
    Causal page narrative."""
    live = _load_live_df()
    if len(live) < THRESHOLD:
        return {"skipped": "below threshold", "live_rows": int(len(live)), "threshold": THRESHOLD}

    df = _coerce_kpis(live.copy())
    loss = df.get("packet_loss_ratio", pd.Series([0.0]*len(df))).fillna(0).astype(float)
    cf   = (loss * (1.0 - 0.4971)).clip(lower=0.0)
    red  = ((loss - cf) / loss.replace(0, np.nan) * 100.0).fillna(0.0)

    out = pd.DataFrame()
    out["ue_id"]                    = df.get("ue_id", pd.Series([0]*len(df)))
    out["timestamp"]                = df.get("timestamp", pd.Series(range(len(df))))
    out["original_packet_loss"]     = loss.round(4)
    out["counterfactual_if_static"] = cf.round(4)
    out["reduction_pct"]            = red.round(2)

    output = DATA_DIR / "counterfactuals_live.csv"
    _atomic_write_csv(out, output)
    return {
        "rows":         int(len(out)),
        "mean_reduction_pct": round(float(red.mean()), 2),
        "output":       output.name,
    }


# ── 5. Severity triage → severity_triage_live.csv ───────────────────────
def score_severity() -> Dict[str, Any]:
    """Coarse severity classes (LOW / MEDIUM / HIGH) for the Severity page."""
    live = _load_live_df()
    if len(live) < THRESHOLD:
        return {"skipped": "below threshold", "live_rows": int(len(live)), "threshold": THRESHOLD}

    df = _coerce_kpis(live.copy())
    loss = df.get("packet_loss_ratio", pd.Series([0.0]*len(df))).fillna(0).astype(float)
    delay= df.get("delay_ms",          pd.Series([0.0]*len(df))).fillna(0).astype(float)
    flag = pd.to_numeric(df.get("_live_anomaly", 0), errors="coerce").fillna(0).astype(int)

    sev = np.where(
        (flag == 1) & (loss > 0.5), "HIGH",
        np.where((flag == 1) | (loss > 0.2) | (delay > 150), "MEDIUM", "LOW"),
    )

    out = pd.DataFrame()
    if "timestamp"        in df.columns: out["timestamp"]        = df["timestamp"]
    if "ue_id"            in df.columns: out["ue_id"]            = df["ue_id"]
    if "delay_ms"         in df.columns: out["delay_ms"]         = df["delay_ms"]
    if "packet_loss_ratio"in df.columns: out["packet_loss_ratio"]= df["packet_loss_ratio"]
    if "jitter_ms"        in df.columns: out["jitter_ms"]        = df["jitter_ms"]
    if "throughput_mbps"  in df.columns: out["throughput_mbps"]  = df["throughput_mbps"]
    out["severity_final"] = sev
    if "load_level" in df.columns: out["traffic_type"] = df["load_level"].astype(str)

    output = DATA_DIR / "severity_triage_live.csv"
    _atomic_write_csv(out, output)
    counts = {k: int(v) for k, v in pd.Series(sev).value_counts().to_dict().items()}
    return {"rows": int(len(out)), "by_severity": counts, "output": output.name}


# ── 6. QoS Forecast → predicted_jitter_live.csv ─────────────────────────
# Replays the offline-trained XGBoost jitter regressor on the live JSONL log.
# Engineered features (rolling/lag) are computed per-UE on the fly. Output
# matches the schema of static predicted_jitter.csv so the QoS Forecast page
# can concat both via _load_static_plus_live("predicted_jitter.csv").

_QOS_MODEL_PATH = Path(__file__).resolve().parent.parent / "qos_forecast" / "models" / "xgboost_jitter.pkl"
_QOS_MODEL = None

# Feature order MUST match the training schema (predicted_jitter.csv header).
_QOS_FEATURES = [
    "sinr_dl_db", "mcs_dl", "packet_loss_ratio", "prb_utilization",
    "retransmissions", "cqi", "load_level", "mobility_speed", "shadowing_enabled",
    "throughput_mbps_roll3", "throughput_mbps_roll5",
    "throughput_mbps_lag1",  "throughput_mbps_lag3",
    "jitter_ms_roll3",       "jitter_ms_roll5",
    "jitter_ms_lag1",        "jitter_ms_lag3",
    "sinr_dl_db_roll3",      "sinr_dl_db_roll5",
    "sinr_dl_db_lag1",       "sinr_dl_db_lag3",
    "packet_loss_ratio_roll3","packet_loss_ratio_roll5",
    "packet_loss_ratio_lag1","packet_loss_ratio_lag3",
]


def _load_qos_model():
    global _QOS_MODEL
    if _QOS_MODEL is not None:
        return _QOS_MODEL
    try:
        import joblib
        _QOS_MODEL = joblib.load(_QOS_MODEL_PATH)
        log.info(f"mlops: loaded XGBoost jitter model from {_QOS_MODEL_PATH.name}")
    except Exception as e:
        log.warning(f"mlops: cannot load XGBoost jitter model — {e}")
        _QOS_MODEL = None
    return _QOS_MODEL


def _engineer_qos_features(df: pd.DataFrame) -> pd.DataFrame:
    """Compute rolling means + lags per UE so the model sees the same shape
    of features it was trained on."""
    if "ue_id" not in df.columns or "timestamp" not in df.columns:
        return pd.DataFrame()
    df = df.sort_values(["ue_id", "timestamp"]).copy()

    base = ["throughput_mbps", "jitter_ms", "sinr_dl_db", "packet_loss_ratio"]
    for col in base:
        if col not in df.columns:
            df[col] = 0.0
        df[col] = pd.to_numeric(df[col], errors="coerce")

    g = df.groupby("ue_id", group_keys=False)
    for col in base:
        df[f"{col}_roll3"] = g[col].transform(lambda s: s.rolling(3, min_periods=1).mean())
        df[f"{col}_roll5"] = g[col].transform(lambda s: s.rolling(5, min_periods=1).mean())
        df[f"{col}_lag1"]  = g[col].shift(1)
        df[f"{col}_lag3"]  = g[col].shift(3)

    # Pad any remaining feature with 0
    for f in _QOS_FEATURES:
        if f not in df.columns:
            df[f] = 0.0
        df[f] = pd.to_numeric(df[f], errors="coerce").fillna(0.0)

    return df


def score_qos_forecast() -> Dict[str, Any]:
    """Live XGBoost jitter prediction over the cumulative live log."""
    live = _load_live_df()
    if len(live) < THRESHOLD:
        return {"skipped": "below threshold", "live_rows": int(len(live)), "threshold": THRESHOLD}

    model = _load_qos_model()
    if model is None:
        return {"skipped": "xgboost_jitter.pkl not loadable", "rows": int(len(live))}

    feat_df = _engineer_qos_features(live)
    if feat_df.empty:
        return {"skipped": "missing ue_id/timestamp columns"}

    X = feat_df[_QOS_FEATURES].astype(float)
    try:
        y_pred = model.predict(X)
    except Exception as e:
        return {"skipped": f"prediction failed: {e}", "rows": int(len(feat_df))}

    out = feat_df.copy()
    out["y_pred"] = y_pred
    # Ground-truth jitter at the row time = the actual jitter_ms we observed.
    out["y_true"] = pd.to_numeric(out.get("jitter_ms", pd.Series([0.0]*len(out))),
                                   errors="coerce").fillna(0.0)

    keep = _QOS_FEATURES + ["timestamp", "ue_id", "y_true", "y_pred"]
    keep = [c for c in keep if c in out.columns]
    out = out[keep]

    output = DATA_DIR / "predicted_jitter_live.csv"
    _atomic_write_csv(out, output)

    err = (out["y_pred"] - out["y_true"]).abs()
    return {
        "rows":   int(len(out)),
        "mae":    round(float(err.mean()), 4),
        "rmse":   round(float(np.sqrt(((out["y_pred"]-out["y_true"])**2).mean())), 4),
        "ues":    int(out["ue_id"].nunique()) if "ue_id" in out else 0,
        "output": output.name,
    }


# ── Registration entry point ────────────────────────────────────────────
def register_default_jobs() -> None:
    runner.register("score_anomaly",         score_anomaly)
    runner.register("score_causal",          score_causal)
    runner.register("score_sla",             score_sla)
    runner.register("score_counterfactuals", score_counterfactuals)
    runner.register("score_severity",        score_severity)
    runner.register("score_qos_forecast",    score_qos_forecast)
