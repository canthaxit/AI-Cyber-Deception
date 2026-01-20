@echo off
REM Quick start script for Anomaly Detection API

echo ========================================
echo Log Anomaly Detection API
echo ========================================
echo.

echo Checking Python installation...
python --version
if %ERRORLEVEL% NEQ 0 (
    echo ERROR: Python not found!
    pause
    exit /b 1
)

echo.
echo Installing/updating dependencies...
pip install -q fastapi uvicorn python-multipart pydantic

echo.
echo Starting API server...
echo API will be available at: http://localhost:8000
echo API docs will be at: http://localhost:8000/docs
echo.
echo Press Ctrl+C to stop the server
echo.

python anomaly_api.py
