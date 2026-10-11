@echo off
setlocal enabledelayedexpansion

echo =====================================================================
echo       Smart Traffic Violation Detection System - Startup
echo =====================================================================
echo.

:: Check if virtual environment exists
if not exist ".venv\Scripts\python.exe" (
    echo [ERROR] Virtual environment not found at .venv\
    echo Please run setup first: py -3.13 -m venv .venv
    echo And install dependencies: .\.venv\Scripts\pip.exe install -r requirements.txt
    pause
    exit /b 1
)

echo [*] Activating Python virtual environment...
call .venv\Scripts\activate.bat

echo [*] Verifying model weights...
if not exist "models\yolov8n.pt" (
    echo [*] Downloading pretrained YOLO model...
    python scripts\download_models.py
)

echo [*] Initializing database and directories...
python -c "from backend.app.config import config; from backend.app.database.db import init_db; config.ensure_directories(); init_db(); print('Database ready.')"
if %errorlevel% neq 0 (
    echo [ERROR] Database initialization failed.
    pause
    exit /b %errorlevel%
)

echo.
echo =====================================================================
echo  TRAFFICGUARD AI SYSTEM IS NOW ONLINE
echo  Dashboard URL: http://127.0.0.1:8000
echo  API Documentation: http://127.0.0.1:8000/docs
echo =====================================================================
echo.
echo Starting FastAPI Web Server... Press Ctrl+C to terminate.
echo.

python -m uvicorn backend.app.main:app --host 127.0.0.1 --port 8000

pause
