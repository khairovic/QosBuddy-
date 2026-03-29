#!/usr/bin/env bash
# ─────────────────────────────────────────────────────────────────────────────
# QoSBuddy M6 — Local Demo Launcher
# Runs the full M6 stack (KB build → FastAPI → Streamlit) locally without Docker
# Prerequisites: Python 3.11+, Ollama installed and running
#
# Usage:
#   bash scripts/run_local.sh [--rebuild-kb]
#
# ─────────────────────────────────────────────────────────────────────────────

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
ROOT_DIR="$(dirname "$SCRIPT_DIR")"
REBUILD_KB=false

# Parse args
for arg in "$@"; do
  [[ "$arg" == "--rebuild-kb" ]] && REBUILD_KB=true
done

cd "$ROOT_DIR"

# ── Colour helpers ─────────────────────────────────────────────────────────
RED='\033[0;31m'; GREEN='\033[0;32m'; BLUE='\033[0;34m'; YELLOW='\033[1;33m'; NC='\033[0m'
info()    { echo -e "${BLUE}[INFO]${NC}  $*"; }
success() { echo -e "${GREEN}[OK]${NC}    $*"; }
warn()    { echo -e "${YELLOW}[WARN]${NC}  $*"; }
error()   { echo -e "${RED}[ERROR]${NC} $*"; exit 1; }

# ── 1. Check Python ────────────────────────────────────────────────────────
info "Checking Python version…"
python3 --version || error "Python 3 not found"

# ── 2. Virtual environment ─────────────────────────────────────────────────
if [ ! -d ".venv" ]; then
    info "Creating virtual environment…"
    python3 -m venv .venv
fi
source .venv/bin/activate
info "Installing dependencies…"
pip install -q -r docker/requirements.txt

# ── 3. Data directory ──────────────────────────────────────────────────────
export QOSBUDDY_DATA_DIR="${QOSBUDDY_DATA_DIR:-./data}"
export QOSBUDDY_CHROMA_DIR="${QOSBUDDY_CHROMA_DIR:-./chroma_store}"

if [ ! -d "$QOSBUDDY_DATA_DIR" ]; then
    warn "Data directory $QOSBUDDY_DATA_DIR not found."
    warn "Please place these files there:"
    warn "  - anomaly_scores_ns3.xls"
    warn "  - root_cause_labels.csv"
    warn "  - counterfactuals.csv"
    warn "  - QosBuddy_report.pdf (optional)"
    error "Aborting until data is placed."
fi

# ── 4. Ollama check ────────────────────────────────────────────────────────
info "Checking Ollama…"
if curl -sf http://localhost:11434/api/tags > /dev/null 2>&1; then
    success "Ollama is running"
    # Pull model if not already present
    if ! curl -sf http://localhost:11434/api/tags | grep -q "llama3.2"; then
        info "Pulling llama3.2 model (this takes a few minutes)…"
        ollama pull llama3.2
    fi
else
    warn "Ollama is not running. RAG answers will fall back to context-only mode."
    warn "Start Ollama with: ollama serve"
fi

# ── 5. Build knowledge base ────────────────────────────────────────────────
KB_FLAG="$QOSBUDDY_CHROMA_DIR/.built"
if [ "$REBUILD_KB" = true ] || [ ! -f "$KB_FLAG" ]; then
    info "Building ChromaDB knowledge base (first run or --rebuild-kb)…"
    python3 rag/build_knowledge_base.py
    touch "$KB_FLAG"
    success "Knowledge base built"
else
    success "Knowledge base exists — skipping rebuild (pass --rebuild-kb to force)"
fi

# ── 6. Start FastAPI in background ─────────────────────────────────────────
info "Starting FastAPI backend on http://localhost:8000…"
PYTHONPATH="$ROOT_DIR" uvicorn api.fastapi_backend:app \
    --host 0.0.0.0 --port 8000 --log-level warning &
API_PID=$!
sleep 3

if ! kill -0 "$API_PID" 2>/dev/null; then
    error "FastAPI failed to start. Check logs above."
fi
success "FastAPI running (PID $API_PID) — Swagger UI: http://localhost:8000/docs"

# ── 7. Start Streamlit ─────────────────────────────────────────────────────
info "Starting Streamlit dashboard on http://localhost:8501…"
echo ""
echo -e "${GREEN}══════════════════════════════════════════════════════════${NC}"
echo -e "${GREEN}  QoSBuddy NOC Dashboard ready!                          ${NC}"
echo -e "${GREEN}  Dashboard:   http://localhost:8501                      ${NC}"
echo -e "${GREEN}  API docs:    http://localhost:8000/docs                 ${NC}"
echo -e "${GREEN}  Ctrl+C to stop both services                           ${NC}"
echo -e "${GREEN}══════════════════════════════════════════════════════════${NC}"
echo ""

# Trap Ctrl+C to kill both processes
cleanup() {
    info "Stopping services…"
    kill "$API_PID" 2>/dev/null || true
    success "Done"
}
trap cleanup INT TERM

PYTHONPATH="$ROOT_DIR" streamlit run dashboard/streamlit_dashboard.py \
    --server.port 8501 \
    --server.address 0.0.0.0 \
    --browser.gatherUsageStats false

# If streamlit exits (shouldn't happen), kill API too
kill "$API_PID" 2>/dev/null || true
