@echo off
setlocal enabledelayedexpansion
title QoSBuddy NOC Dashboard
color 0B

echo.
echo  +==========================================================+
echo  ^|                                                          ^|
echo  ^|              Q o S B u d d y                             ^|
echo  ^|          PingWin . ESPRIT . 2025-2026                    ^|
echo  ^|                                                          ^|
echo  +==========================================================+
echo.

:: ---- 1. Check Docker -----------------------------------------------
echo  [1/5] Checking Docker...
docker info >nul 2>&1
if errorlevel 1 (
    color 0C
    echo.
    echo  [ERROR] Docker is not running.
    echo  Please start Docker Desktop first, then run this again.
    echo.
    pause
    exit /b 1
)
echo        OK - Docker is running.
echo.

:: ---- 2. Find docker-compose.yml ------------------------------------
echo  [2/5] Locating project...
set "COMPOSE_DIR="

if exist "%~dp0docker-compose.yml" (
    set "COMPOSE_DIR=%~dp0"
)

if "!COMPOSE_DIR!"=="" if exist "%~dp0docker\docker-compose.yml" (
    set "COMPOSE_DIR=%~dp0docker\"
)

if "!COMPOSE_DIR!"=="" if exist "%USERPROFILE%\Downloads\QosBuddy-\qosbuddy_m6\docker\docker-compose.yml" (
    set "COMPOSE_DIR=%USERPROFILE%\Downloads\QosBuddy-\qosbuddy_m6\docker\"
)

if "!COMPOSE_DIR!"=="" if exist "D:\QosBuddy\qosbuddy_m6\docker\docker-compose.yml" (
    set "COMPOSE_DIR=D:\QosBuddy\qosbuddy_m6\docker\"
)

if "!COMPOSE_DIR!"=="" (
    color 0C
    echo.
    echo  [ERROR] Could not find docker-compose.yml.
    echo  Place this .bat next to docker-compose.yml and try again.
    echo.
    pause
    exit /b 1
)

echo        OK - Found: !COMPOSE_DIR!
echo.

:: ---- 3. Start services ---------------------------------------------
echo  [3/5] Starting all services (this may take 1-2 minutes)...
echo        Building: Ollama, KB Builder, FastAPI, Dashboard...
echo.

cd /d "!COMPOSE_DIR!"
docker compose up --build -d
if errorlevel 1 (
    color 0C
    echo.
    echo  [ERROR] Docker Compose failed. Check the messages above.
    echo.
    pause
    exit /b 1
)

echo.
echo        OK - All containers started.
echo.

:: ---- 4. Wait for dashboard -----------------------------------------
echo  [4/5] Waiting for dashboard to be ready...
echo        (this can take 30-60 seconds on first run)
echo.

set /a ATTEMPTS=0

:wait_loop
set /a ATTEMPTS+=1

curl -sf http://localhost:8501/_stcore/health >nul 2>&1
if not errorlevel 1 goto dashboard_ready

curl -sf -o nul http://localhost:8501/ >nul 2>&1
if not errorlevel 1 goto dashboard_ready

if !ATTEMPTS! geq 30 goto dashboard_timeout

echo        Attempt !ATTEMPTS!/30 - waiting...
timeout /t 5 /nobreak >nul
goto wait_loop

:dashboard_timeout
color 0E
echo.
echo  [WARN] Dashboard is taking longer than expected.
echo         Opening browser anyway - it may need a few more seconds.
echo.
goto open_browser

:dashboard_ready
echo.
echo        OK - Dashboard is online.
echo.

:open_browser
echo  [5/5] Opening dashboard in your browser...
echo.

timeout /t 2 /nobreak >nul
start "" http://localhost:8501

color 0A
echo.
echo  +==========================================================+
echo  ^|                                                          ^|
echo  ^|   QoSBuddy is LIVE                                       ^|
echo  ^|                                                          ^|
echo  ^|   Dashboard:  http://localhost:8501                      ^|
echo  ^|   API Docs:   http://localhost:8000/docs                 ^|
echo  ^|   Ollama:     http://localhost:11434                     ^|
echo  ^|                                                          ^|
echo  +----------------------------------------------------------+
echo  ^|                                                          ^|
echo  ^|   Press any key to view live logs                        ^|
echo  ^|   Press Ctrl+C then Y to stop everything                 ^|
echo  ^|                                                          ^|
echo  +==========================================================+
echo.
pause >nul

:: ---- Stream logs ---------------------------------------------------
echo.
echo  Streaming logs (Ctrl+C to stop)...
echo  ----------------------------------------------------------
echo.
cd /d "!COMPOSE_DIR!"
docker compose logs -f --tail=50

:: ---- Cleanup -------------------------------------------------------
echo.
set /p "STOP=Stop all QoSBuddy services? (y/n): "
if /i "!STOP!"=="y" (
    echo  Stopping services...
    docker compose down
    echo  Done.
)
pause
