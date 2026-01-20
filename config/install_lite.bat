@echo off
echo ============================================
echo Log Anomaly Detection - LITE VERSION
echo Quick Install (No TensorFlow)
echo ============================================
echo.

REM Check if Python is installed
python --version >nul 2>&1
if %errorlevel% neq 0 (
    echo ERROR: Python is not installed or not in PATH
    echo.
    echo Trying to install from Microsoft Store...
    start ms-windows-store://pdp/?ProductId=9NRWMJP3717K
    echo.
    echo Please install Python, then run this script again
    pause
    exit /b 1
)

echo Python found!
python --version
echo.

echo Installing minimal dependencies (No TensorFlow)...
echo This should take only 2-3 minutes
echo.

REM Upgrade pip
echo Upgrading pip...
python -m pip install --upgrade pip --quiet

REM Install minimal requirements
echo Installing packages...
python -m pip install -r requirements_minimal.txt --quiet

if %errorlevel% equ 0 (
    echo.
    echo ============================================
    echo SUCCESS: All dependencies installed!
    echo ============================================
    echo.
    echo Verifying installation...
    python -c "import pandas, numpy, sklearn, matplotlib, seaborn, joblib; print('All packages verified!')"
    echo.
    echo Running test with sample data...
    echo.
    python log_anomaly_detection_lite.py --data_path test_logs/ --baseline_period_days 1 --contamination 0.10
    echo.
) else (
    echo.
    echo ERROR: Installation failed
    echo Please check error messages above
    echo.
)

pause
