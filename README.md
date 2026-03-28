# QosBuddy

QoSBuddy is a data-driven project for proactive QoS monitoring in telecom networks. It uses machine learning to analyze network metrics, predict SLA risks, detect anomalies, and support intelligent, explainable optimization beyond traditional reactive monitoring.

---

## DSO1.3 — Network Performance Benchmarking (Member 5)

A complete, production-level benchmarking system for network QoS performance built on top of telecom KPI data.

### Phases

| Phase | Description |
|-------|-------------|
| 1 | Data Loading & Exploration |
| 2 | Data Preprocessing (imputation + MinMax scaling) |
| 3 | QoS Score Engineering (composite weighted score) |
| 4 | Rule-Based Benchmarking (GOOD / MEDIUM / POOR) |
| 5 | ML Classification (RandomForest + XGBoost) |
| 6 | Category-Based Benchmarking (load_level × mobility_speed) |
| 7 | Visualisation (distributions, heatmaps, box plots) |
| 8 | Insights Generation |
| 9 | Export Results (CSV reports) |
| 10 | SHAP Explainability (bonus) |

### Key Files

| File | Purpose |
|------|---------|
| `benchmark_pipeline.py` | Modular Python pipeline — all 10 phases as functions |
| `qos_benchmarking.ipynb` | Interactive notebook — step-by-step walkthrough |
| `app/main.py` | FastAPI REST endpoint |

### Quick Start

```bash
# Install dependencies
pip install -r requirements.txt

# Run the full pipeline (generates synthetic data if no CSV is present)
python benchmark_pipeline.py

# Or open the notebook
jupyter lab qos_benchmarking.ipynb
```

### QoS Score Formula

```
QoS_score =  0.40 × norm_throughput
           + 0.20 × norm_sinr
           + 0.15 × (1 − norm_delay)
           + 0.15 × (1 − norm_jitter)
           + 0.10 × (1 − norm_packet_loss)
```

Score is in **[0, 1]** — higher means better network quality.

### Outputs

After running the pipeline the `outputs/` directory contains:

- `dataset_with_qos_scores.csv` — full dataset enriched with QoS scores and performance labels
- `benchmark_report_by_category.csv` — aggregated KPIs per category (for dashboard integration)
- `best_qos_model.pkl` — serialised best ML classifier
- `qos_scaler.pkl` — fitted MinMaxScaler for inference
