@echo off
echo ============================================
echo Log Anomaly Detection - Dependency Installer
echo ============================================
echo.

REM Check if Python is installed
python --version >nul 2>&1
if %errorlevel% neq 0 (
    echo ERROR: Python is not installed or not in PATH
    echo.
    echo Please install Python from: https://www.python.org/downloads/
    echo Make sure to check "Add Python to PATH" during installation
    echo.
    pause
    exit /b 1
)

echo Python found!
python --version
echo.

echo Installing dependencies...
echo This may take 10-15 minutes as TensorFlow is large (~500MB)
echo.

REM Upgrade pip first
echo Step 1/2: Upgrading pip...
python -m pip install --upgrade pip

REM Install requirements
echo.
echo Step 2/2: Installing required packages...
python -m pip install -r requirements.txt

if %errorlevel% equ 0 (
    echo.
    echo ============================================
    echo SUCCESS: All dependencies installed!
    echo ============================================
    echo.
    echo Verifying installation...
    python -c "import pandas, numpy, sklearn, tensorflow, matplotlib, seaborn, joblib; print('All packages verified!')"
    echo.
    echo You can now run the pipeline:
    echo   python intrusion_detection_pipeline.py --data_path test_logs/ --baseline_period_days 1 --contamination 0.10 --autoencoder_epochs 10
    echo.
) else (
    echo.
    echo ERROR: Installation failed
    echo Please check the error messages above
    echo.
)

pause
