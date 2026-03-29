# ─────────────────────────────────────────────────────────────────────────────
# QoSBuddy M6 — Dockerfile
# DSO3.2 · PingWin · ESPRIT 2025-2026
#
# Multi-stage: builder installs deps, final image runs either
#   - Streamlit dashboard (CMD default)
#   - FastAPI backend     (override CMD)
# ─────────────────────────────────────────────────────────────────────────────

FROM python:3.11-slim AS builder

WORKDIR /app

# System deps: needed for chromadb's sqlite3 and sentence-transformers tokenizers
RUN apt-get update && apt-get install -y --no-install-recommends \
        build-essential \
        libsqlite3-dev \
        gcc \
    && rm -rf /var/lib/apt/lists/*

# Install Python deps in a virtual environment
COPY docker/requirements.txt .
RUN pip install --upgrade pip && \
    pip install --no-cache-dir -r requirements.txt

# ─────────────────────────────────────────────────────────────────────────────
FROM python:3.11-slim AS final

WORKDIR /app

# Copy site-packages from builder (avoids re-downloading in final image)
COPY --from=builder /usr/local/lib/python3.11/site-packages \
                    /usr/local/lib/python3.11/site-packages
COPY --from=builder /usr/local/bin /usr/local/bin

# Copy application code
COPY rag/       ./rag/
COPY dashboard/ ./dashboard/
COPY api/       ./api/

# ── Environment variables ────────────────────────────────────────────────────
# Mount your data directory here: docker run -v /path/to/data:/app/data ...
ENV QOSBUDDY_DATA_DIR=/app/data
ENV QOSBUDDY_CHROMA_DIR=/app/chroma_store
ENV QOSBUDDY_EMBED_MODEL=all-MiniLM-L6-v2
ENV OLLAMA_BASE_URL=http://ollama:11434

# ── Ports ─────────────────────────────────────────────────────────────────────
# Streamlit: 8501 | FastAPI: 8000
EXPOSE 8501 8000

# ── Healthcheck ───────────────────────────────────────────────────────────────
HEALTHCHECK --interval=30s --timeout=10s --start-period=60s \
  CMD python -c "import urllib.request; urllib.request.urlopen('http://localhost:8000/health')" || exit 1

# ── Default: run Streamlit dashboard ─────────────────────────────────────────
CMD ["streamlit", "run", "dashboard/streamlit_dashboard.py", \
     "--server.port=8501", "--server.address=0.0.0.0", \
     "--server.headless=true", "--browser.gatherUsageStats=false"]

# To run FastAPI instead:
# docker run ... qosbuddy-m6 uvicorn api.fastapi_backend:app --host 0.0.0.0 --port 8000
