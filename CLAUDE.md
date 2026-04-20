# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Overview

QoSBuddy M6 (DSO3.2) — RAG + GenAI NOC Dashboard for telecom network monitoring. Built for the PingWin ESPRIT 2025-2026 project. Uses machine learning to analyze network metrics, predict SLA risks, detect anomalies, and provide explainable optimization recommendations.

## Quick Start

### Local Development (Windows)
```bash
# Run the full stack via batch script
.\run_m6.bat
```

This script:
1. Creates/activates virtual environment at `.venv/`
2. Installs dependencies from `docker/requirements.txt`
3. Sets environment variables (`QOSBUDDY_DATA_DIR=./data`, `QOSBUDDY_CHROMA_DIR=./chroma_store`)
4. Builds the ChromaDB knowledge base via `rag/build_knowledge_base.py`
5. Starts FastAPI backend on http://localhost:8000
6. Starts Streamlit dashboard on http://localhost:8501

### Docker Deployment
```bash
cd docker
docker-compose up --build
```

Services:
- `ollama` (port 11434) — local LLM server for RAG
- `kb-builder` — one-shot knowledge base builder
- `qosbuddy-api` (port 8000) — FastAPI backend
- `qosbuddy-dashboard` (port 8501) — Streamlit dashboard

## Architecture

```
qosbuddy_m6/
├── api/                    # FastAPI REST backend
│   └── fastapi_backend.py  # Main API with all endpoints
├── dashboard/              # Streamlit NOC dashboard
│   └── streamlit_dashboard.py
├── rag/                    # RAG pipeline & knowledge base
│   ├── build_knowledge_base.py  # ChromaDB builder
│   └── rag_pipeline.py     # Query pipeline with Ollama
├── rl_agent/               # Reinforcement learning (PPO)
│   ├── env.py              # QoSNetworkEnv Gymnasium env
│   ├── policy.py           # RLPolicy wrapper for inference
│   └── models/             # Trained PPO weights
├── qos_forecast/           # M2 forecasting models
├── causal_models/          # M3 DoWhy causal inference
├── anomaly_models/         # M1 anomaly detection models
├── data/                   # NS-3 data files, predictions
└── chroma_store/           # Persistent vector database
```

## Module Responsibilities

| Module | Purpose | Key Files |
|--------|---------|-----------|
| `api/` | REST API for dashboard & external NOC systems | `fastapi_backend.py` |
| `dashboard/` | Premium NOC UI with glassmorphism design | `streamlit_dashboard.py` |
| `rag/` | ChromaDB vector search + Ollama LLM (llama3.2) | `rag_pipeline.py`, `build_knowledge_base.py` |
| `rl_agent/` | PPO policy for network optimization (DSO3.1) | `env.py`, `policy.py` |
| `anomaly_models/` | M1 ensemble: IF, XGBoost, LSTM-AE, Transformer | `.pkl`/`.json` model files |
| `qos_forecast/` | M2 jitter/throughput forecasting (Prophet, XGBoost) | `models/` |
| `causal_models/` | M3 DoWhy counterfactual estimator | `dowhy_model.pkl` |

## Key Dependencies

- **Core**: pandas, numpy, scikit-learn, xgboost
- **ML**: tensorflow, torch, stable-baselines3, dowhy, prophet
- **RAG**: langchain, sentence-transformers, chromadb, ollama
- **API**: fastapi, uvicorn, pydantic
- **Dashboard**: streamlit, plotly

## Environment Variables

| Variable | Default | Description |
|----------|---------|-------------|
| `QOSBUDDY_DATA_DIR` | `./data` | Path to data files (anomaly scores, predictions) |
| `QOSBUDDY_CHROMA_DIR` | `./chroma_store` | ChromaDB persistent storage |
| `OLLAMA_HOST` / `OLLAMA_BASE_URL` | `http://localhost:11434` | Ollama LLM endpoint |
| `QOSBUDDY_API_URL` | `http://localhost:8000` | FastAPI backend URL |

## API Endpoints (FastAPI)

- `GET /health` — Health check
- `GET /kpi/summary` — Network KPI summary
- `GET /anomalies` / `GET /anomalies/v2` — Anomaly events (v1/v2 format)
- `POST /rag/query` — RAG-powered natural language queries
- `POST /rl/predict` — RL policy action prediction
- `POST /rl/simulate` — RL simulation episodes
- `GET /models/benchmark` — Model comparison data
- `POST /ingest/anomaly` — Push new anomaly events

## Data Format

Anomaly detection uses multi-model scores (v2 format):
- `anomaly_scores_v2.csv` — Scores from IF, COPOD, OC-SVM, RF, XGBoost, GBM, LSTM-AE, AT, TranAD
- `root_cause_labels.csv` — Detection-based root causes (temporal_pattern_anomaly, model_disagreement_anomaly, etc.)
- `counterfactuals.csv` — DoWhy counterfactual estimates
- `sla_breach_predictions.csv` — SLA risk predictions

## Development Notes

- Virtual environment is at `.venv/` — activate via `.venv\Scripts\activate` (Windows)
- Knowledge base build takes 2-4 minutes on first run
- The dashboard uses a custom premium dark theme with glassmorphism styling
- RAG pipeline auto-detects query mode (anomaly/causal/sla/model_comparison) from question keywords
- RL policy loads from `rl_agent/models/ppo_qos_agent_best.zip`

## Git Branch

Current feature branch: `feature/m6-rag-dashboard`
