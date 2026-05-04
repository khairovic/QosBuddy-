@echo off
title QoSBuddy - Stopping...
color 0E

echo.
echo  Stopping all QoSBuddy services...
echo.

set "SCRIPT_DIR=%~dp0"

if exist "%SCRIPT_DIR%docker-compose.yml" (
    cd /d "%SCRIPT_DIR%"
) else if exist "%SCRIPT_DIR%docker\docker-compose.yml" (
    cd /d "%SCRIPT_DIR%docker\"
) else if exist "D:\QosBuddy\qosbuddy_m6\docker\docker-compose.yml" (
    cd /d "D:\QosBuddy\qosbuddy_m6\docker\"
) else (
    color 0C
    echo  [ERROR] Could not find docker-compose.yml
    echo  Make sure the project is at D:\QosBuddy\qosbuddy_m6\docker\
    pause
    exit /b 1
)

docker compose down

if errorlevel 1 (
    color 0C
    echo.
    echo  [ERROR] Docker Compose down failed. Check the messages above.
    echo.
    pause
    exit /b 1
)

color 0A
echo.
echo  All services stopped.
echo.
pause
