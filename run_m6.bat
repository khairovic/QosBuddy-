@echo off
echo =====================================================
echo     QoSBuddy M6 (DSO3.2) - Starting...
echo =====================================================

:: Force the correct directory (no variables)
cd /d "C:\Users\GIGABYTE\Downloads\QosBuddy\qosbuddy_m6"
dir
pause
echo Current directory: %cd%
echo.

REM === 1. Virtual Environment ===
if not exist .venv (
    echo Creating virtual environment...
    python -m venv .venv
)

call .venv\Scripts\activate.bat

REM === 2. Install dependencies (quiet) ===
echo Installing dependencies...
pip install -q -r docker\requirements.txt

REM === 3. Environment variables ===
set QOSBUDDY_DATA_DIR=./data
set QOSBUDDY_CHROMA_DIR=./chroma_store

REM === 4. Build Knowledge Base ===
echo.
echo Building ChromaDB Knowledge Base...
echo (This takes 2-4 minutes the first time)
python rag\build_knowledge_base.py

echo.
echo =====================================================
echo Starting FastAPI and Streamlit...
echo Dashboard → http://localhost:8501
echo FastAPI   → http://localhost:8000
echo =====================================================
echo.

REM === 5. Start FastAPI in new window (hard-coded path) ===
start "FastAPI Backend" cmd /k "call .venv\Scripts\activate.bat && uvicorn api.fastapi_backend:app --host 0.0.0.0 --port 8000 --log-level warning"
REM Wait a bit
timeout /t 8

REM === 6. Start Streamlit ===
streamlit run dashboard\streamlit_dashboard.py --server.port 8501 --server.address 0.0.0.0 --browser.gatherUsageStats false