@echo off
chcp 65001 >nul 2>&1
title QoSBuddy — Stopping...
color 0E

echo.
echo  Stopping all QoSBuddy services...
echo.

set "SCRIPT_DIR=%~dp0"

if exist "%SCRIPT_DIR%docker-compose.yml" (
    cd /d "%SCRIPT_DIR%"
) else if exist "%SCRIPT_DIR%docker\docker-compose.yml" (
    cd /d "%SCRIPT_DIR%docker\"
) else if exist "%USERPROFILE%\Downloads\QosBuddy-\qosbuddy_m6\docker\docker-compose.yml" (
    cd /d "%USERPROFILE%\Downloads\QosBuddy-\qosbuddy_m6\docker\"
) else (
    echo  [ERROR] Could not find docker-compose.yml
    pause
    exit /b 1
)

docker compose down

color 0A
echo.
echo  All services stopped.
echo.
pause
