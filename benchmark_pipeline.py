"""
QoSBuddy — DSO1.3 Network Performance Benchmarking Pipeline
============================================================
Member 5 | Complete production-level QoS benchmarking system.

Phases:
    1  – Data Loading & Exploration
    2  – Data Preprocessing
    3  – QoS Score Engineering
    4  – Rule-Based Benchmarking
    5  – Machine Learning Classification
    6  – Category-Based Benchmarking
    7  – Visualisation
    8  – Insights Generation
    9  – Export Results
    10 – SHAP Explainability (bonus)
"""

import os
import warnings

import joblib
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import (
    ConfusionMatrixDisplay,
    accuracy_score,
    classification_report,
    f1_score,
)
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler

warnings.filterwarnings("ignore")

# ──────────────────────────────────────────────────────────────────────────────
# CONSTANTS
# ──────────────────────────────────────────────────────────────────────────────

# QoS score weights (must sum to 1)
QOS_WEIGHT_THROUGHPUT = 0.40
QOS_WEIGHT_SINR = 0.20
QOS_WEIGHT_DELAY = 0.15
QOS_WEIGHT_JITTER = 0.15
QOS_WEIGHT_PACKET_LOSS = 0.10

# Rule-based classification thresholds
THRESHOLD_THROUGHPUT_GOOD = 20.0   # Mbps
THRESHOLD_THROUGHPUT_POOR = 5.0    # Mbps
THRESHOLD_DELAY_GOOD = 30.0        # ms
THRESHOLD_DELAY_POOR = 100.0       # ms
THRESHOLD_JITTER_GOOD = 5.0        # ms
THRESHOLD_JITTER_POOR = 20.0       # ms
THRESHOLD_PACKET_LOSS_GOOD = 0.01  # ratio (1 %)
THRESHOLD_PACKET_LOSS_POOR = 0.05  # ratio (5 %)

# Feature columns used for ML
FEATURE_COLS = [
    "sinr_dl_db",
    "throughput_mbps",
    "delay_ms",
    "jitter_ms",
    "packet_loss_ratio",
    "retransmissions",
]

# Output paths
OUTPUT_DIR = "outputs"
MODEL_PATH = os.path.join(OUTPUT_DIR, "best_qos_model.pkl")
SCALER_PATH = os.path.join(OUTPUT_DIR, "qos_scaler.pkl")
REPORT_PATH = os.path.join(OUTPUT_DIR, "benchmark_report_by_category.csv")
DATASET_PATH = os.path.join(OUTPUT_DIR, "dataset_with_qos_scores.csv")


# ──────────────────────────────────────────────────────────────────────────────
# PHASE 1 — DATA LOADING & EXPLORATION
# ──────────────────────────────────────────────────────────────────────────────

def load_dataset(path: str, n_rows: int | None = None) -> pd.DataFrame:
    """Load a CSV or Parquet dataset from *path*.

    If *path* does not exist a synthetic demo dataset is generated so the
    pipeline can always be exercised end-to-end without real data.

    Parameters
    ----------
    path:
        File system path to the CSV / Parquet file.
    n_rows:
        Optional row limit (useful for large CSVs).

    Returns
    -------
    pd.DataFrame
    """
    if not os.path.exists(path):
        print(f"[INFO] '{path}' not found — generating synthetic demo dataset.")
        return _generate_synthetic_dataset()

    ext = os.path.splitext(path)[-1].lower()
    if ext == ".parquet":
        df = pd.read_parquet(path)
    else:
        df = pd.read_csv(path, nrows=n_rows)

    print(f"[INFO] Loaded {len(df):,} rows from '{path}'.")
    return df


def _generate_synthetic_dataset(n_samples: int = 5_000, seed: int = 42) -> pd.DataFrame:
    """Generate a realistic synthetic QoS dataset for demonstration."""
    rng = np.random.default_rng(seed)

    load_levels = rng.choice(["low", "medium", "high"], size=n_samples, p=[0.3, 0.4, 0.3])
    mobility_speeds = rng.choice(["stationary", "pedestrian", "vehicular"], size=n_samples, p=[0.4, 0.35, 0.25])

    # Throughput degrades at high load and high mobility
    load_factor = np.where(load_levels == "low", 1.2, np.where(load_levels == "medium", 0.9, 0.6))
    mob_factor = np.where(mobility_speeds == "stationary", 1.1, np.where(mobility_speeds == "pedestrian", 0.95, 0.75))

    throughput = np.clip(rng.normal(25, 10, n_samples) * load_factor * mob_factor, 0.5, 100)
    delay = np.clip(rng.normal(40, 20, n_samples) / (load_factor * mob_factor), 5, 300)
    jitter = np.clip(rng.normal(8, 5, n_samples) / (load_factor * mob_factor), 0.5, 80)
    packet_loss = np.clip(rng.exponential(0.02, n_samples) / (load_factor * mob_factor), 0, 0.5)
    sinr = np.clip(rng.normal(15, 8, n_samples) * mob_factor, -5, 40)
    retransmissions = np.clip(rng.poisson(3, n_samples) * (1 / (load_factor * mob_factor)), 0, 50).astype(int)

    df = pd.DataFrame(
        {
            "throughput_mbps": throughput,
            "delay_ms": delay,
            "jitter_ms": jitter,
            "packet_loss_ratio": packet_loss,
            "sinr_dl_db": sinr,
            "retransmissions": retransmissions.astype(float),
            "load_level": load_levels,
            "mobility_speed": mobility_speeds,
        }
    )
    # Inject a small fraction of missing values to make preprocessing realistic
    for col in ["throughput_mbps", "delay_ms", "jitter_ms", "packet_loss_ratio"]:
        mask = rng.random(n_samples) < 0.02
        df.loc[mask, col] = np.nan

    print(f"[INFO] Synthetic dataset generated: {len(df):,} rows.")
    return df


def explore_dataset(df: pd.DataFrame) -> None:
    """Print head, info, describe, and missing-value summary for *df*."""
    print("=" * 70)
    print("HEAD")
    print("=" * 70)
    print(df.head())

    print("\n" + "=" * 70)
    print("INFO")
    print("=" * 70)
    df.info()

    print("\n" + "=" * 70)
    print("DESCRIPTIVE STATISTICS")
    print("=" * 70)
    print(df.describe().T.to_string())

    print("\n" + "=" * 70)
    print("MISSING VALUES")
    print("=" * 70)
    missing = df.isnull().sum()
    missing_pct = (missing / len(df) * 100).round(2)
    missing_df = pd.DataFrame({"count": missing, "pct": missing_pct})
    print(missing_df[missing_df["count"] > 0].to_string() or "No missing values.")


def plot_kpi_distributions(df: pd.DataFrame) -> None:
    """Plot histograms for the five core KPI columns."""
    kpi_cols = ["throughput_mbps", "delay_ms", "jitter_ms", "packet_loss_ratio", "retransmissions"]
    existing = [c for c in kpi_cols if c in df.columns]

    fig, axes = plt.subplots(1, len(existing), figsize=(4 * len(existing), 4))
    if len(existing) == 1:
        axes = [axes]

    for ax, col in zip(axes, existing):
        ax.hist(df[col].dropna(), bins=40, color="steelblue", edgecolor="white")
        ax.set_title(col, fontsize=11)
        ax.set_xlabel("Value")
        ax.set_ylabel("Frequency")

    plt.suptitle("Phase 1 — KPI Distributions", fontsize=13, fontweight="bold")
    plt.tight_layout()
    plt.show()


def plot_correlation_matrix(df: pd.DataFrame) -> None:
    """Plot the Pearson correlation heat-map for numeric columns."""
    numeric_df = df.select_dtypes(include="number")
    corr = numeric_df.corr()

    plt.figure(figsize=(10, 8))
    mask = np.triu(np.ones_like(corr, dtype=bool))
    sns.heatmap(
        corr,
        mask=mask,
        annot=True,
        fmt=".2f",
        cmap="coolwarm",
        center=0,
        linewidths=0.5,
        square=True,
    )
    plt.title("Phase 1 — Feature Correlation Matrix", fontsize=13, fontweight="bold")
    plt.tight_layout()
    plt.show()


# ──────────────────────────────────────────────────────────────────────────────
# PHASE 2 — DATA PREPROCESSING
# ──────────────────────────────────────────────────────────────────────────────

def preprocess(df: pd.DataFrame) -> tuple[pd.DataFrame, MinMaxScaler]:
    """Handle missing values and normalise QoS feature columns.

    Returns
    -------
    df_clean : pd.DataFrame
        DataFrame with imputed values and added ``_norm`` columns.
    scaler : MinMaxScaler
        Fitted scaler (saved later for inference).
    """
    df_clean = df.copy()

    # Impute numeric NaNs with column median (robust to outliers)
    for col in FEATURE_COLS:
        if col in df_clean.columns and df_clean[col].isnull().any():
            median_val = df_clean[col].median()
            df_clean[col] = df_clean[col].fillna(median_val)
            print(f"[PREPROCESS] '{col}' — imputed {df[col].isnull().sum()} NaNs with median ({median_val:.4f})")

    # Normalise with MinMaxScaler → values in [0, 1]
    scaler = MinMaxScaler()
    existing_features = [c for c in FEATURE_COLS if c in df_clean.columns]
    scaled_values = scaler.fit_transform(df_clean[existing_features])
    norm_cols = [f"{c}_norm" for c in existing_features]
    df_clean[norm_cols] = scaled_values

    print(f"[PREPROCESS] Normalised columns: {norm_cols}")
    return df_clean, scaler


# ──────────────────────────────────────────────────────────────────────────────
# PHASE 3 — QoS SCORE ENGINEERING
# ──────────────────────────────────────────────────────────────────────────────

def compute_qos_score(df: pd.DataFrame) -> pd.DataFrame:
    """Compute a composite QoS score in [0, 1].

    Formula
    -------
    QoS_score =
        0.40 * norm_throughput
      + 0.20 * norm_sinr
      + 0.15 * (1 − norm_delay)
      + 0.15 * (1 − norm_jitter)
      + 0.10 * (1 − norm_packet_loss)

    Higher is better.
    """
    df = df.copy()

    def _norm_col(name: str) -> pd.Series:
        """Return normalised column, or zeros if unavailable."""
        col = f"{name}_norm"
        return df[col] if col in df.columns else pd.Series(0.0, index=df.index)

    qos = (
        QOS_WEIGHT_THROUGHPUT * _norm_col("throughput_mbps")
        + QOS_WEIGHT_SINR * _norm_col("sinr_dl_db")
        + QOS_WEIGHT_DELAY * (1 - _norm_col("delay_ms"))
        + QOS_WEIGHT_JITTER * (1 - _norm_col("jitter_ms"))
        + QOS_WEIGHT_PACKET_LOSS * (1 - _norm_col("packet_loss_ratio"))
    )

    # Clip to [0, 1] to guard against floating-point drift
    df["QoS_score"] = qos.clip(0, 1)
    print(f"[QoS SCORE] Mean={df['QoS_score'].mean():.4f}  Std={df['QoS_score'].std():.4f}")
    return df


# ──────────────────────────────────────────────────────────────────────────────
# PHASE 4 — RULE-BASED BENCHMARKING
# ──────────────────────────────────────────────────────────────────────────────

def classify_performance_rule_based(df: pd.DataFrame) -> pd.DataFrame:
    """Add ``performance_class_rule`` column (GOOD / MEDIUM / POOR)."""
    df = df.copy()

    throughput = df.get("throughput_mbps", pd.Series(np.nan, index=df.index))
    delay = df.get("delay_ms", pd.Series(np.nan, index=df.index))
    jitter = df.get("jitter_ms", pd.Series(np.nan, index=df.index))
    packet_loss = df.get("packet_loss_ratio", pd.Series(np.nan, index=df.index))

    is_good = (
        (throughput >= THRESHOLD_THROUGHPUT_GOOD)
        & (delay <= THRESHOLD_DELAY_GOOD)
        & (jitter <= THRESHOLD_JITTER_GOOD)
        & (packet_loss <= THRESHOLD_PACKET_LOSS_GOOD)
    )

    is_poor = (
        (throughput <= THRESHOLD_THROUGHPUT_POOR)
        | (delay >= THRESHOLD_DELAY_POOR)
        | (jitter >= THRESHOLD_JITTER_POOR)
        | (packet_loss >= THRESHOLD_PACKET_LOSS_POOR)
    )

    df["performance_class_rule"] = np.where(is_good, "GOOD", np.where(is_poor, "POOR", "MEDIUM"))

    dist = df["performance_class_rule"].value_counts()
    print(f"[RULE-BASED] Class distribution:\n{dist.to_string()}")
    return df


# ──────────────────────────────────────────────────────────────────────────────
# PHASE 5 — MACHINE LEARNING CLASSIFICATION
# ──────────────────────────────────────────────────────────────────────────────

def train_and_evaluate_models(
    df: pd.DataFrame,
    label_col: str = "performance_class_rule",
    test_size: float = 0.25,
    random_state: int = 42,
) -> tuple[object, dict]:
    """Train RandomForest and (optionally) XGBoost classifiers.

    Returns
    -------
    best_model : fitted estimator
        The model with the highest weighted F1-score on the test set.
    results : dict
        Evaluation metrics for all trained models.
    """
    existing_features = [c for c in FEATURE_COLS if c in df.columns]
    X = df[existing_features].copy()
    y = df[label_col].copy()

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=test_size, random_state=random_state, stratify=y
    )

    models = {
        "RandomForest": RandomForestClassifier(
            n_estimators=200,
            max_depth=10,
            class_weight="balanced",
            random_state=random_state,
            n_jobs=-1,
        ),
    }

    # Add XGBoost only when installed
    try:
        from xgboost import XGBClassifier  # noqa: PLC0415
        from sklearn.preprocessing import LabelEncoder  # noqa: PLC0415

        le = LabelEncoder()
        y_train_enc = le.fit_transform(y_train)
        y_test_enc = le.transform(y_test)

        xgb = XGBClassifier(
            n_estimators=200,
            max_depth=6,
            learning_rate=0.1,
            use_label_encoder=False,
            eval_metric="mlogloss",
            random_state=random_state,
            n_jobs=-1,
        )
        xgb.fit(X_train, y_train_enc)
        y_pred_xgb = xgb.predict(X_test)

        acc_xgb = accuracy_score(y_test_enc, y_pred_xgb)
        f1_xgb = f1_score(y_test_enc, y_pred_xgb, average="weighted")
        print(f"\n[XGBoost] Accuracy={acc_xgb:.4f}  F1={f1_xgb:.4f}")
        print(classification_report(y_test_enc, y_pred_xgb, target_names=le.classes_))
        _plot_confusion_matrix(xgb, X_test, y_test_enc, le.classes_, "XGBoost")

        # Store XGBoost under original label space for unified comparison
        results_xgb = {"accuracy": acc_xgb, "f1": f1_xgb, "model": xgb, "label_encoder": le}
    except ImportError:
        print("[INFO] XGBoost not installed — skipping.")
        results_xgb = None

    # Train Random Forest
    results: dict[str, dict] = {}
    rf = models["RandomForest"]
    rf.fit(X_train, y_train)
    y_pred_rf = rf.predict(X_test)

    acc_rf = accuracy_score(y_test, y_pred_rf)
    f1_rf = f1_score(y_test, y_pred_rf, average="weighted")
    print(f"\n[RandomForest] Accuracy={acc_rf:.4f}  F1={f1_rf:.4f}")
    print(classification_report(y_test, y_pred_rf))
    _plot_confusion_matrix(rf, X_test, y_test, rf.classes_, "RandomForest")

    results["RandomForest"] = {"accuracy": acc_rf, "f1": f1_rf, "model": rf}

    # Determine best model (by F1)
    best_name = "RandomForest"
    best_model = rf
    best_f1 = f1_rf

    if results_xgb and results_xgb["f1"] > best_f1:
        best_name = "XGBoost"
        best_model = results_xgb["model"]
        results["XGBoost"] = results_xgb

    print(f"\n[BEST MODEL] {best_name} selected (F1={results[best_name]['f1']:.4f})")
    return best_model, results


def _plot_confusion_matrix(model, X_test, y_test, labels, title: str) -> None:
    """Plot a confusion matrix for the given model and test data."""
    fig, ax = plt.subplots(figsize=(6, 5))
    ConfusionMatrixDisplay.from_estimator(model, X_test, y_test, display_labels=labels, ax=ax, colorbar=False)
    ax.set_title(f"Confusion Matrix — {title}", fontsize=12, fontweight="bold")
    plt.tight_layout()
    plt.show()


def save_model(model, scaler: MinMaxScaler) -> None:
    """Persist the best model and scaler to disk."""
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    joblib.dump(model, MODEL_PATH)
    joblib.dump(scaler, SCALER_PATH)
    print(f"[SAVE] Model → {MODEL_PATH}")
    print(f"[SAVE] Scaler → {SCALER_PATH}")


# ──────────────────────────────────────────────────────────────────────────────
# PHASE 6 — CATEGORY-BASED BENCHMARKING
# ──────────────────────────────────────────────────────────────────────────────

def benchmark_by_category(df: pd.DataFrame) -> pd.DataFrame:
    """Group by load_level and mobility_speed; compute average KPIs.

    Returns
    -------
    pd.DataFrame
        Aggregated benchmark report.
    """
    group_cols = [c for c in ["load_level", "mobility_speed"] if c in df.columns]
    if not group_cols:
        print("[BENCHMARK] No grouping columns found.")
        return pd.DataFrame()

    agg = (
        df.groupby(group_cols)
        .agg(
            avg_QoS_score=("QoS_score", "mean"),
            avg_throughput=("throughput_mbps", "mean"),
            avg_delay=("delay_ms", "mean"),
            avg_jitter=("jitter_ms", "mean"),
            avg_packet_loss=("packet_loss_ratio", "mean"),
            sample_count=("QoS_score", "count"),
        )
        .reset_index()
    )

    # Assign performance class to each category based on avg QoS score
    agg["performance_class"] = pd.cut(
        agg["avg_QoS_score"],
        bins=[0, 0.4, 0.65, 1.0],
        labels=["POOR", "MEDIUM", "GOOD"],
        include_lowest=True,
    )

    agg = agg.sort_values("avg_QoS_score", ascending=False)
    print("\n[CATEGORY BENCHMARK]\n", agg.to_string(index=False))
    return agg


# ──────────────────────────────────────────────────────────────────────────────
# PHASE 7 — VISUALISATION
# ──────────────────────────────────────────────────────────────────────────────

def plot_qos_score_distribution(df: pd.DataFrame) -> None:
    """Histogram + KDE of the QoS_score column."""
    fig, ax = plt.subplots(figsize=(8, 4))
    sns.histplot(df["QoS_score"], bins=50, kde=True, color="teal", ax=ax)
    ax.axvline(0.4, color="orange", linestyle="--", label="Poor threshold (0.4)")
    ax.axvline(0.65, color="green", linestyle="--", label="Good threshold (0.65)")
    ax.set_title("Phase 7 — QoS Score Distribution", fontsize=13, fontweight="bold")
    ax.set_xlabel("QoS Score")
    ax.legend()
    plt.tight_layout()
    plt.show()


def plot_performance_class_distribution(df: pd.DataFrame) -> None:
    """Bar chart of performance class counts."""
    col = "performance_class_rule" if "performance_class_rule" in df.columns else "performance_class"
    counts = df[col].value_counts()

    palette = {"GOOD": "#4CAF50", "MEDIUM": "#FF9800", "POOR": "#F44336"}
    colors = [palette.get(c, "steelblue") for c in counts.index]

    fig, ax = plt.subplots(figsize=(6, 4))
    bars = ax.bar(counts.index, counts.values, color=colors, edgecolor="white")
    for bar, val in zip(bars, counts.values):
        ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 5, str(val), ha="center", fontsize=10)
    ax.set_title("Phase 7 — Performance Class Distribution", fontsize=13, fontweight="bold")
    ax.set_ylabel("Count")
    plt.tight_layout()
    plt.show()


def plot_qos_by_load_level(df: pd.DataFrame) -> None:
    """Box plot of QoS_score grouped by load_level."""
    if "load_level" not in df.columns:
        print("[VIZ] 'load_level' column not found.")
        return

    order = ["low", "medium", "high"]
    existing_order = [o for o in order if o in df["load_level"].unique()]

    fig, ax = plt.subplots(figsize=(7, 4))
    sns.boxplot(data=df, x="load_level", y="QoS_score", order=existing_order, palette="Blues", ax=ax)
    ax.set_title("Phase 7 — QoS Score vs Load Level", fontsize=13, fontweight="bold")
    ax.set_xlabel("Load Level")
    ax.set_ylabel("QoS Score")
    plt.tight_layout()
    plt.show()


def plot_qos_by_mobility_speed(df: pd.DataFrame) -> None:
    """Box plot of QoS_score grouped by mobility_speed."""
    if "mobility_speed" not in df.columns:
        print("[VIZ] 'mobility_speed' column not found.")
        return

    order = ["stationary", "pedestrian", "vehicular"]
    existing_order = [o for o in order if o in df["mobility_speed"].unique()]

    fig, ax = plt.subplots(figsize=(7, 4))
    sns.boxplot(data=df, x="mobility_speed", y="QoS_score", order=existing_order, palette="Oranges", ax=ax)
    ax.set_title("Phase 7 — QoS Score vs Mobility Speed", fontsize=13, fontweight="bold")
    ax.set_xlabel("Mobility Speed")
    ax.set_ylabel("QoS Score")
    plt.tight_layout()
    plt.show()


# ──────────────────────────────────────────────────────────────────────────────
# PHASE 8 — INSIGHTS GENERATION
# ──────────────────────────────────────────────────────────────────────────────

def generate_insights(df: pd.DataFrame, category_report: pd.DataFrame) -> None:
    """Print key analytical insights derived from the benchmarking results."""
    print("\n" + "=" * 70)
    print("PHASE 8 — KEY INSIGHTS")
    print("=" * 70)

    # Worst conditions
    if not category_report.empty:
        worst = category_report.iloc[-1]
        print(
            f"\n[1] Worst performing condition:\n"
            f"    {dict(worst[['load_level', 'mobility_speed', 'avg_QoS_score', 'performance_class']])}"
            if "load_level" in worst and "mobility_speed" in worst
            else f"    avg_QoS_score = {worst['avg_QoS_score']:.4f}"
        )
        best = category_report.iloc[0]
        print(
            f"\n[2] Best performing condition:\n"
            f"    avg_QoS_score = {best['avg_QoS_score']:.4f}"
        )

    # Delay vs packet loss correlation
    if "delay_ms" in df.columns and "packet_loss_ratio" in df.columns:
        corr_val = df["delay_ms"].corr(df["packet_loss_ratio"])
        print(
            f"\n[3] Correlation between delay and packet loss: {corr_val:.4f}\n"
            f"    {'Strong positive' if corr_val > 0.5 else 'Moderate' if corr_val > 0.2 else 'Weak'} relationship."
        )

    # Mobility impact
    if "mobility_speed" in df.columns:
        mob_impact = df.groupby("mobility_speed")["QoS_score"].mean().sort_values()
        print(f"\n[4] Impact of mobility on QoS score (lower = worse):\n{mob_impact.to_string()}")

    # Load level impact
    if "load_level" in df.columns:
        load_impact = df.groupby("load_level")["QoS_score"].mean().sort_values()
        print(f"\n[5] Impact of load level on QoS score:\n{load_impact.to_string()}")

    print("=" * 70)


# ──────────────────────────────────────────────────────────────────────────────
# PHASE 9 — EXPORT RESULTS
# ──────────────────────────────────────────────────────────────────────────────

def export_results(df: pd.DataFrame, category_report: pd.DataFrame) -> None:
    """Write benchmark outputs to CSV files in *OUTPUT_DIR*."""
    os.makedirs(OUTPUT_DIR, exist_ok=True)

    # Full dataset with QoS scores
    df.to_csv(DATASET_PATH, index=False)
    print(f"[EXPORT] dataset_with_qos_scores → {DATASET_PATH}")

    # Category benchmark report
    if not category_report.empty:
        category_report.to_csv(REPORT_PATH, index=False)
        print(f"[EXPORT] benchmark_report_by_category → {REPORT_PATH}")


# ──────────────────────────────────────────────────────────────────────────────
# PHASE 10 — SHAP EXPLAINABILITY (BONUS)
# ──────────────────────────────────────────────────────────────────────────────

def explain_with_shap(model, df: pd.DataFrame, max_display: int = 15) -> None:
    """Generate SHAP summary and bar plots for a tree-based model.

    Falls back gracefully if the `shap` package is not installed.
    """
    try:
        import shap  # noqa: PLC0415
    except ImportError:
        print("[SHAP] 'shap' package not installed — skipping explainability.")
        return

    existing_features = [c for c in FEATURE_COLS if c in df.columns]
    X = df[existing_features].copy()

    explainer = shap.TreeExplainer(model)
    shap_values = explainer.shap_values(X)

    # Summary plot (beeswarm)
    plt.figure()
    shap.summary_plot(shap_values, X, max_display=max_display, show=False)
    plt.title("Phase 10 — SHAP Feature Importance (Beeswarm)", fontsize=12, fontweight="bold")
    plt.tight_layout()
    plt.show()

    # Bar plot
    plt.figure()
    shap.summary_plot(shap_values, X, plot_type="bar", max_display=max_display, show=False)
    plt.title("Phase 10 — SHAP Feature Importance (Bar)", fontsize=12, fontweight="bold")
    plt.tight_layout()
    plt.show()


def compare_rule_vs_ml(df: pd.DataFrame, model, feature_cols: list[str] | None = None) -> pd.DataFrame:
    """Compare rule-based labels against ML predictions.

    Returns
    -------
    pd.DataFrame
        Columns: ``performance_class_rule``, ``performance_class_ml``, ``agreement``.
    """
    cols = feature_cols or [c for c in FEATURE_COLS if c in df.columns]
    ml_preds = model.predict(df[cols])

    comparison = df[["performance_class_rule"]].copy() if "performance_class_rule" in df.columns else pd.DataFrame()
    comparison["performance_class_ml"] = ml_preds
    if "performance_class_rule" in comparison.columns:
        comparison["agreement"] = comparison["performance_class_rule"] == comparison["performance_class_ml"]
        agreement_rate = comparison["agreement"].mean()
        print(f"[COMPARE] Rule vs ML agreement rate: {agreement_rate:.2%}")
    return comparison


# ──────────────────────────────────────────────────────────────────────────────
# FULL PIPELINE RUNNER
# ──────────────────────────────────────────────────────────────────────────────

def run_full_pipeline(data_path: str = "network_data.csv", n_rows: int | None = None) -> dict:
    """Execute all 10 phases and return a results dictionary.

    Parameters
    ----------
    data_path:
        Path to the input CSV / Parquet file.
    n_rows:
        Row limit for large files (``None`` = load all).

    Returns
    -------
    dict with keys: ``df``, ``category_report``, ``best_model``, ``scaler``.
    """
    print("\n" + "█" * 70)
    print("  QoSBuddy — Network Performance Benchmarking Pipeline")
    print("█" * 70 + "\n")

    # Phase 1
    print("\n── PHASE 1: DATA LOADING & EXPLORATION ──────────────────────────────")
    df = load_dataset(data_path, n_rows=n_rows)
    explore_dataset(df)
    plot_kpi_distributions(df)
    plot_correlation_matrix(df)

    # Phase 2
    print("\n── PHASE 2: PREPROCESSING ───────────────────────────────────────────")
    df, scaler = preprocess(df)

    # Phase 3
    print("\n── PHASE 3: QoS SCORE ENGINEERING ──────────────────────────────────")
    df = compute_qos_score(df)

    # Phase 4
    print("\n── PHASE 4: RULE-BASED BENCHMARKING ────────────────────────────────")
    df = classify_performance_rule_based(df)

    # Phase 5
    print("\n── PHASE 5: MACHINE LEARNING CLASSIFICATION ─────────────────────────")
    best_model, ml_results = train_and_evaluate_models(df)
    save_model(best_model, scaler)

    # Phase 6
    print("\n── PHASE 6: CATEGORY-BASED BENCHMARKING ─────────────────────────────")
    category_report = benchmark_by_category(df)

    # Phase 7
    print("\n── PHASE 7: VISUALISATION ───────────────────────────────────────────")
    plot_qos_score_distribution(df)
    plot_performance_class_distribution(df)
    plot_qos_by_load_level(df)
    plot_qos_by_mobility_speed(df)
    plot_correlation_matrix(df)

    # Phase 8
    generate_insights(df, category_report)

    # Phase 9
    print("\n── PHASE 9: EXPORT RESULTS ──────────────────────────────────────────")
    export_results(df, category_report)

    # Phase 10
    print("\n── PHASE 10: SHAP EXPLAINABILITY (BONUS) ────────────────────────────")
    explain_with_shap(best_model, df)
    comparison = compare_rule_vs_ml(df, best_model)

    print("\n" + "█" * 70)
    print("  Pipeline complete.")
    print("█" * 70 + "\n")

    return {
        "df": df,
        "category_report": category_report,
        "best_model": best_model,
        "scaler": scaler,
        "comparison": comparison,
    }


# ──────────────────────────────────────────────────────────────────────────────
# ENTRY POINT
# ──────────────────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    run_full_pipeline()
