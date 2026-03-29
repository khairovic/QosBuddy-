@echo off
chcp 65001 >nul 2>&1
title QoSBuddy NOC Dashboard — Launcher
color 0B

echo.
echo  ╔══════════════════════════════════════════════════════════╗
echo  ║                                                          ║
echo  ║        ██████╗  ██████╗ ███████╗██████╗                  ║
echo  ║       ██╔═══██╗██╔═══██╗██╔════╝██╔══██╗                ║
echo  ║       ██║   ██║██║   ██║███████╗██████╔╝                ║
echo  ║       ██║▄▄ ██║██║   ██║╚════██║██╔══██╗                ║
echo  ║       ╚██████╔╝╚██████╔╝███████║██████╔╝                ║
echo  ║        ╚══▀▀═╝  ╚═════╝ ╚══════╝╚═════╝                ║
echo  ║                  B U D D Y                               ║
echo  ║                                                          ║
echo  ║          PingWin · ESPRIT · 2025-2026                    ║
echo  ║                                                          ║
echo  ╚══════════════════════════════════════════════════════════╝
echo.

:: ── Check Docker is running ───────────────────────────────────────────
echo  [1/4] Checking Docker...
docker info >nul 2>&1
if %errorlevel% neq 0 (
    color 0C
    echo.
    echo  [ERROR] Docker is not running!
    echo  Please start Docker Desktop first, then run this again.
    echo.
    pause
    exit /b 1
)
echo        Docker is running.

:: ── Find the docker-compose directory ─────────────────────────────────
:: Try to find the docker folder relative to where the .bat is placed
set "SCRIPT_DIR=%~dp0"

:: Option 1: .bat is inside qosbuddy_m6/docker/
if exist "%SCRIPT_DIR%docker-compose.yml" (
    set "COMPOSE_DIR=%SCRIPT_DIR%"
    goto :found
)

:: Option 2: .bat is inside qosbuddy_m6/ (project root)
if exist "%SCRIPT_DIR%docker\docker-compose.yml" (
    set "COMPOSE_DIR=%SCRIPT_DIR%docker\"
    goto :found
)

:: Option 3: .bat is on Desktop or elsewhere — use known path
if exist "%USERPROFILE%\Downloads\QosBuddy-\qosbuddy_m6\docker\docker-compose.yml" (
    set "COMPOSE_DIR=%USERPROFILE%\Downloads\QosBuddy-\qosbuddy_m6\docker\"
    goto :found
)

:: Not found — ask user
color 0E
echo.
echo  [WARN] Could not auto-detect your project folder.
echo  Please drag your "docker" folder here and press Enter:
set /p "COMPOSE_DIR=  > "
if not exist "%COMPOSE_DIR%docker-compose.yml" (
    color 0C
    echo  [ERROR] docker-compose.yml not found in: %COMPOSE_DIR%
    pause
    exit /b 1
)

:found
echo        Project: %COMPOSE_DIR%
echo.

:: ── Start the full stack ──────────────────────────────────────────────
echo  [2/4] Building and starting all services...
echo        (Ollama, KB Builder, FastAPI, Dashboard)
echo        This may take a minute on first run...
echo.

cd /d "%COMPOSE_DIR%"
docker compose up --build -d

if %errorlevel% neq 0 (
    color 0C
    echo.
    echo  [ERROR] Docker Compose failed. Check the output above.
    echo.
    pause
    exit /b 1
)

:: ── Wait for dashboard to be ready ────────────────────────────────────
echo.
echo  [3/4] Waiting for dashboard to come online...

set "READY=0"
for /L %%i in (1,1,40) do (
    if !READY! equ 0 (
        curl -sf http://localhost:8501/_stcore/health >nul 2>&1
        if !errorlevel! equ 0 (
            set "READY=1"
        ) else (
            <nul set /p "=."
            timeout /t 3 /nobreak >nul
        )
    )
)

:: Enable delayed expansion for the READY check
setlocal enabledelayedexpansion
set "READY=0"
for /L %%i in (1,1,40) do (
    if !READY! equ 0 (
        curl -sf http://localhost:8501/_stcore/health >nul 2>&1
        if !errorlevel! equ 0 (
            set "READY=1"
        ) else (
            <nul set /p "=."
            timeout /t 3 /nobreak >nul
        )
    )
)
endlocal & set "READY=%READY%"

echo.

:: ── Open browser ──────────────────────────────────────────────────────
echo  [4/4] Opening dashboard in browser...
echo.
start "" http://localhost:8501

color 0A
echo  ╔══════════════════════════════════════════════════════════╗
echo  ║                                                          ║
echo  ║   QoSBuddy is LIVE!                                      ║
echo  ║                                                          ║
echo  ║   Dashboard:  http://localhost:8501                       ║
echo  ║   API Docs:   http://localhost:8000/docs                  ║
echo  ║   Ollama:     http://localhost:11434                      ║
echo  ║                                                          ║
echo  ║   To stop: close this window or press Ctrl+C             ║
echo  ║                                                          ║
echo  ╚══════════════════════════════════════════════════════════╝
echo.

:: ── Stream logs until user stops ──────────────────────────────────────
echo  Streaming logs... (press Ctrl+C to stop)
echo.
cd /d "%COMPOSE_DIR%"
docker compose logs -f

:: ── Cleanup on exit ───────────────────────────────────────────────────
echo.
echo  Stopping QoSBuddy...
docker compose down
echo  Done. Goodbye!
pause
